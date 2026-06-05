//go:build contrast_unstable_api

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestLoadExpectedManifestRequiresDigestMatch(t *testing.T) {
	dir := t.TempDir()
	manifestPath := filepath.Join(dir, "manifest.json")
	manifest := []byte(`{
		"Policies": {
			"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": {
				"SANs": ["continuum-etcd-client", "workload-kimi-k2-6", "*"],
				"WorkloadSecretID": "apps/v1/Deployment/continuum-e37856/workload-kimi-k2-6"
			}
		}
	}`)
	if err := os.WriteFile(manifestPath, manifest, 0o600); err != nil {
		t.Fatal(err)
	}

	p := verifierPolicy{
		ManifestPath:   manifestPath,
		ManifestDigest: sha256BytesDigest(manifest),
	}
	got, err := loadExpectedManifest(t.Context(), p, nil)
	if err != nil {
		t.Fatalf("loadExpectedManifest returned error: %v", err)
	}
	if string(got) != string(manifest) {
		t.Fatalf("manifest mismatch: got %s", got)
	}

	p.ManifestDigest = "sha256:bad"
	if _, err := loadExpectedManifest(t.Context(), p, nil); err == nil {
		t.Fatal("loadExpectedManifest accepted a mismatched manifest digest")
	}
}

func TestRunRejectsTrailingJSON(t *testing.T) {
	err := runWithStdinForTest(t, `{"schema_version":"`+schemaVersion+`"}{}`)
	if err == nil || !strings.Contains(err.Error(), "must contain exactly one JSON object") {
		t.Fatalf("expected trailing JSON error, got %v", err)
	}
}

func TestRunRejectsSelectedModelPayloadMismatch(t *testing.T) {
	payload := map[string]interface{}{
		"schema_version":     schemaVersion,
		"mode":               "privatemode",
		"proxy_base_url":     "http://127.0.0.1:8080/v1",
		"policy_digest":      sha256StringDigest("policy"),
		"evidence_digest":    sha256StringDigest("evidence"),
		"verification_nonce": "nonce",
		"model_ids":          []interface{}{"privatemode/selected-model"},
		"policy": map[string]interface{}{
			"manifest_digest":        sha256StringDigest("manifest"),
			"proxy_image_digest":     sha256StringDigest("proxy"),
			"proxy_binary_digest":    sha256StringDigest("proxy"),
			"proxy_binary_path":      "/opt/privatemode/privatemode-proxy",
			"expected_workload_sans": []interface{}{"workload-kimi-k2-6"},
			"model_workload_bindings": map[string]interface{}{
				"privatemode/kimi-k2-6": map[string]interface{}{
					"workload_sans": []interface{}{"workload-kimi-k2-6"},
				},
			},
			"dump_requests":                          false,
			"shared_prompt_cache":                    false,
			"nvidia_ocsp_allow_unknown":              false,
			"nvidia_ocsp_revoked_grace_period_hours": 0,
		},
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}

	err = runWithStdinForTest(t, string(encoded))
	if err == nil {
		t.Fatal("run accepted selected model payload that disagrees with workload bindings")
	}
	if !strings.Contains(err.Error(), "model_ids must match model_workload_bindings") {
		t.Fatalf("expected selected model mismatch error, got %v", err)
	}
}

func TestRunRejectsMissingSelectedModelPayload(t *testing.T) {
	payload := map[string]interface{}{
		"schema_version":     schemaVersion,
		"mode":               "privatemode",
		"proxy_base_url":     "http://127.0.0.1:8080/v1",
		"policy_digest":      sha256StringDigest("policy"),
		"evidence_digest":    sha256StringDigest("evidence"),
		"verification_nonce": "nonce",
		"policy": map[string]interface{}{
			"manifest_digest":        sha256StringDigest("manifest"),
			"proxy_binary_digest":    sha256StringDigest("proxy"),
			"proxy_binary_path":      "/opt/privatemode/privatemode-proxy",
			"expected_workload_sans": []interface{}{"workload-kimi-k2-6"},
			"model_workload_bindings": map[string]interface{}{
				"privatemode/kimi-k2-6": map[string]interface{}{
					"workload_sans": []interface{}{"workload-kimi-k2-6"},
				},
			},
			"dump_requests":                          false,
			"shared_prompt_cache":                    false,
			"nvidia_ocsp_allow_unknown":              false,
			"nvidia_ocsp_revoked_grace_period_hours": 0,
		},
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}

	err = runWithStdinForTest(t, string(encoded))
	if err == nil {
		t.Fatal("run accepted missing selected model payload")
	}
	if !strings.Contains(err.Error(), "model_ids are required with model_workload_bindings") {
		t.Fatalf("expected selected model requirement error, got %v", err)
	}
}

func TestValidatePolicyRejectsConfidentialityDowngrades(t *testing.T) {
	p := verifierPolicy{
		ManifestDigest:                    sha256StringDigest("manifest"),
		ProxyImageDigest:                  sha256StringDigest("proxy"),
		DumpRequests:                      boolPtr(true),
		SharedPromptCache:                 boolPtr(false),
		NvidiaOCSPAllowUnknown:            boolPtr(false),
		NvidiaOCSPRevokedGracePeriodHours: intPtr(0),
	}

	if err := validatePolicy(p); err == nil {
		t.Fatal("validatePolicy accepted dump_requests=true")
	}

	p.DumpRequests = boolPtr(false)
	p.SharedPromptCache = boolPtr(true)
	if err := validatePolicy(p); err == nil {
		t.Fatal("validatePolicy accepted shared_prompt_cache=true")
	}

	p.SharedPromptCache = boolPtr(false)
	p.NvidiaOCSPAllowUnknown = boolPtr(true)
	if err := validatePolicy(p); err == nil {
		t.Fatal("validatePolicy accepted nvidia_ocsp_allow_unknown=true")
	}
}

func TestParsePolicyRejectsStringlyTypedPrivacyKnobs(t *testing.T) {
	raw := map[string]interface{}{
		"manifest_digest":                        sha256StringDigest("manifest"),
		"proxy_image_digest":                     sha256StringDigest("proxy"),
		"dump_requests":                          "false",
		"shared_prompt_cache":                    "false",
		"nvidia_ocsp_allow_unknown":              "false",
		"nvidia_ocsp_revoked_grace_period_hours": 0,
	}

	err := validatePolicy(parsePolicy(raw))
	if err == nil {
		t.Fatal("validatePolicy accepted stringly typed boolean privacy knobs")
	}
	for _, want := range []string{
		"dump_requests must be explicitly false",
		"shared_prompt_cache must be explicitly false",
		"nvidia_ocsp_allow_unknown must be explicitly false",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("expected %q in %q", want, err.Error())
		}
	}
}

func TestValidatePolicyRejectsProxyImageDigestWithoutBinaryDigest(t *testing.T) {
	raw := map[string]interface{}{
		"manifest_digest":                        sha256StringDigest("manifest"),
		"proxy_image_digest":                     sha256StringDigest("proxy-image"),
		"dump_requests":                          false,
		"shared_prompt_cache":                    false,
		"nvidia_ocsp_allow_unknown":              false,
		"nvidia_ocsp_revoked_grace_period_hours": 0,
		"expected_workload_sans":                 []interface{}{"workload-kimi-k2-6"},
		"model_workload_bindings": map[string]interface{}{
			"privatemode/kimi-k2-6": map[string]interface{}{
				"workload_sans": []interface{}{"workload-kimi-k2-6"},
			},
		},
	}

	err := validatePolicy(parsePolicy(raw))
	if err == nil {
		t.Fatal("validatePolicy accepted proxy_image_digest without verifiable proxy binary")
	}
	if !strings.Contains(err.Error(), "proxy_binary_digest is required") {
		t.Fatalf("expected proxy binary digest error, got %v", err)
	}
}

func TestParsePolicyRejectsConflictingPrivacyKnobAliases(t *testing.T) {
	raw := map[string]interface{}{
		"manifest_digest":                        sha256StringDigest("manifest"),
		"proxy_image_digest":                     sha256StringDigest("proxy"),
		"dump_requests":                          false,
		"dumpRequests":                           true,
		"shared_prompt_cache":                    false,
		"sharedPromptCache":                      true,
		"nvidia_ocsp_allow_unknown":              false,
		"nvidiaOCSPAllowUnknown":                 true,
		"nvidia_ocsp_revoked_grace_period_hours": 0,
		"nvidiaOCSPRevokedGracePeriod":           48,
	}

	err := validatePolicy(parsePolicy(raw))
	if err == nil {
		t.Fatal("validatePolicy accepted conflicting privacy knob aliases")
	}
	for _, want := range []string{
		"dump_requests must be explicitly false",
		"shared_prompt_cache must be explicitly false",
		"nvidia_ocsp_allow_unknown must be explicitly false",
		"nvidia_ocsp_revoked_grace_period_hours must be 0",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("expected %q in %q", want, err.Error())
		}
	}
}

func TestParsePolicyRejectsConflictingArtifactDigestAliases(t *testing.T) {
	raw := map[string]interface{}{
		"manifest_digest":                        sha256StringDigest("manifest"),
		"manifestDigest":                         sha256StringDigest("other-manifest"),
		"proxy_image_digest":                     sha256StringDigest("proxy"),
		"proxyImageDigest":                       sha256StringDigest("other-proxy"),
		"dump_requests":                          false,
		"shared_prompt_cache":                    false,
		"nvidia_ocsp_allow_unknown":              false,
		"nvidia_ocsp_revoked_grace_period_hours": 0,
	}

	err := validatePolicy(parsePolicy(raw))
	if err == nil {
		t.Fatal("validatePolicy accepted conflicting artifact digest aliases")
	}
	for _, want := range []string{
		"manifest_digest aliases must match",
		"proxy_image_digest aliases must match",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("expected %q in %q", want, err.Error())
		}
	}
}

func TestParsePolicyRejectsNonStringRemoteURLs(t *testing.T) {
	raw := map[string]interface{}{
		"manifest_digest":                        sha256StringDigest("manifest"),
		"proxy_image_digest":                     sha256StringDigest("proxy"),
		"dump_requests":                          false,
		"shared_prompt_cache":                    false,
		"nvidia_ocsp_allow_unknown":              false,
		"nvidia_ocsp_revoked_grace_period_hours": 0,
		"api_base_url":                           map[string]interface{}{"url": "https://api.privatemode.ai"},
		"cdn_base_url":                           123,
		"manifest_url":                           []interface{}{"https://cdn.confidential.cloud/manifest.json"},
	}

	err := validatePolicy(parsePolicy(raw))
	if err == nil {
		t.Fatal("validatePolicy accepted non-string remote URL policy fields")
	}
	for _, want := range []string{
		"api_base_url must be a string",
		"cdn_base_url must be a string",
		"manifest_url must be a string",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("expected %q in %q", want, err.Error())
		}
	}
}

func TestParsePolicyRejectsNonStringInputSourceFields(t *testing.T) {
	raw := map[string]interface{}{
		"manifest_digest":                        sha256StringDigest("manifest"),
		"proxy_image_digest":                     sha256StringDigest("proxy"),
		"dump_requests":                          false,
		"shared_prompt_cache":                    false,
		"nvidia_ocsp_allow_unknown":              false,
		"nvidia_ocsp_revoked_grace_period_hours": 0,
		"api_key_env":                            map[string]interface{}{"env": "PRIVATEMODE_API_KEY"},
		"api_key_file":                           123,
		"manifest_path":                          []interface{}{"/var/lib/privatemode/manifest.json"},
		"manifest_b64":                           map[string]interface{}{"value": "e30="},
		"manifest_log_dir":                       []interface{}{"/var/lib/privatemode/log"},
		"proxy_binary_path":                      map[string]interface{}{"path": "/opt/privatemode/proxy"},
	}

	err := validatePolicy(parsePolicy(raw))
	if err == nil {
		t.Fatal("validatePolicy accepted non-string verifier input source fields")
	}
	for _, want := range []string{
		"api_key_env must be a string",
		"api_key_file must be a string",
		"manifest_path must be a string",
		"manifest_b64 must be a string",
		"manifest_log_dir must be a string",
		"proxy_binary_path must be a string",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("expected %q in %q", want, err.Error())
		}
	}
}

func TestParsePolicyRejectsBlankInputSourceFields(t *testing.T) {
	raw := map[string]interface{}{
		"manifest_digest":                        sha256StringDigest("manifest"),
		"proxy_image_digest":                     sha256StringDigest("proxy"),
		"dump_requests":                          false,
		"shared_prompt_cache":                    false,
		"nvidia_ocsp_allow_unknown":              false,
		"nvidia_ocsp_revoked_grace_period_hours": 0,
		"api_base_url":                           " ",
		"cdn_base_url":                           "",
		"manifest_url":                           " ",
		"api_key_env":                            " ",
		"api_key_file":                           "",
		"manifest_path":                          " ",
		"manifest_b64":                           "",
		"manifest_log_dir":                       " ",
		"proxy_binary_path":                      "",
	}

	err := validatePolicy(parsePolicy(raw))
	if err == nil {
		t.Fatal("validatePolicy accepted blank verifier input source fields")
	}
	for _, want := range []string{
		"api_base_url must be a non-empty string",
		"cdn_base_url must be a non-empty string",
		"manifest_url must be a non-empty string",
		"api_key_env must be a non-empty string",
		"api_key_file must be a non-empty string",
		"manifest_path must be a non-empty string",
		"manifest_b64 must be a non-empty string",
		"manifest_log_dir must be a non-empty string",
		"proxy_binary_path must be a non-empty string",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("expected %q in %q", want, err.Error())
		}
	}
}

func TestParsePolicyRejectsFractionalOCSPGracePeriod(t *testing.T) {
	raw := map[string]interface{}{
		"manifest_digest":                        sha256StringDigest("manifest"),
		"proxy_image_digest":                     sha256StringDigest("proxy"),
		"dump_requests":                          false,
		"shared_prompt_cache":                    false,
		"nvidia_ocsp_allow_unknown":              false,
		"nvidia_ocsp_revoked_grace_period_hours": 0.5,
	}

	err := validatePolicy(parsePolicy(raw))
	if err == nil || !strings.Contains(err.Error(), "nvidia_ocsp_revoked_grace_period_hours must be 0") {
		t.Fatalf("expected OCSP grace-period type error, got %v", err)
	}
}

func TestValidatePolicyRejectsMalformedDigestPins(t *testing.T) {
	p := verifierPolicy{
		ManifestDigest:                    "sha256:manifest",
		ProxyImageDigest:                  sha256StringDigest("proxy"),
		DumpRequests:                      boolPtr(false),
		SharedPromptCache:                 boolPtr(false),
		NvidiaOCSPAllowUnknown:            boolPtr(false),
		NvidiaOCSPRevokedGracePeriodHours: intPtr(0),
		ExpectedWorkloadSANs:              []string{"workload-kimi-k2-6"},
	}

	err := validatePolicy(p)
	if err == nil || !strings.Contains(err.Error(), "manifest_digest must be a sha256 digest") {
		t.Fatalf("expected manifest digest policy error, got %v", err)
	}

	p.ManifestDigest = sha256StringDigest("manifest")
	p.ProxyImageDigest = "sha256:proxy"
	err = validatePolicy(p)
	if err == nil || !strings.Contains(err.Error(), "proxy_image_digest must be a sha256 digest") {
		t.Fatalf("expected proxy image digest policy error, got %v", err)
	}
}

func TestValidatePolicyRejectsCredentialBearingURLs(t *testing.T) {
	valid := verifierPolicy{
		ManifestDigest:                    sha256StringDigest("manifest"),
		ProxyImageDigest:                  sha256StringDigest("proxy"),
		DumpRequests:                      boolPtr(false),
		SharedPromptCache:                 boolPtr(false),
		NvidiaOCSPAllowUnknown:            boolPtr(false),
		NvidiaOCSPRevokedGracePeriodHours: intPtr(0),
		ExpectedWorkloadSANs:              []string{"workload-kimi-k2-6"},
		ModelWorkloadBindings: map[string]modelWorkloadBinding{
			"privatemode/kimi-k2-6": {
				WorkloadSANs: []string{"workload-kimi-k2-6"},
			},
		},
	}
	cases := []struct {
		name    string
		mutate  func(*verifierPolicy)
		wantErr string
	}{
		{
			name:    "api base",
			mutate:  func(p *verifierPolicy) { p.APIBaseURL = "https://user:pass@api.privatemode.ai" },
			wantErr: "api_base_url must not contain userinfo",
		},
		{
			name:    "cdn base",
			mutate:  func(p *verifierPolicy) { p.CDNBaseURL = "https://user:pass@cdn.confidential.cloud/privatemode/v2" },
			wantErr: "cdn_base_url must not contain userinfo",
		},
		{
			name:    "manifest",
			mutate:  func(p *verifierPolicy) { p.ManifestURL = "https://user:pass@example.test/manifest.json" },
			wantErr: "manifest_url must not contain userinfo",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := valid
			tc.mutate(&p)
			err := validatePolicy(p)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected %q, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestValidatePolicyRejectsNonHTTPSRemoteURLs(t *testing.T) {
	valid := verifierPolicy{
		ManifestDigest:                    sha256StringDigest("manifest"),
		ProxyImageDigest:                  sha256StringDigest("proxy"),
		DumpRequests:                      boolPtr(false),
		SharedPromptCache:                 boolPtr(false),
		NvidiaOCSPAllowUnknown:            boolPtr(false),
		NvidiaOCSPRevokedGracePeriodHours: intPtr(0),
		ExpectedWorkloadSANs:              []string{"workload-kimi-k2-6"},
		ModelWorkloadBindings: map[string]modelWorkloadBinding{
			"privatemode/kimi-k2-6": {
				WorkloadSANs: []string{"workload-kimi-k2-6"},
			},
		},
	}
	cases := []struct {
		name    string
		mutate  func(*verifierPolicy)
		wantErr string
	}{
		{
			name:    "api base",
			mutate:  func(p *verifierPolicy) { p.APIBaseURL = "http://api.privatemode.ai" },
			wantErr: "api_base_url must use https",
		},
		{
			name:    "cdn base",
			mutate:  func(p *verifierPolicy) { p.CDNBaseURL = "http://cdn.confidential.cloud/privatemode/v2" },
			wantErr: "cdn_base_url must use https",
		},
		{
			name:    "manifest",
			mutate:  func(p *verifierPolicy) { p.ManifestURL = "http://example.test/manifest.json" },
			wantErr: "manifest_url must use https",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := valid
			tc.mutate(&p)
			err := validatePolicy(p)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected %q, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestValidatePolicyRejectsRelativeRemoteURLs(t *testing.T) {
	valid := verifierPolicy{
		ManifestDigest:                    sha256StringDigest("manifest"),
		ProxyImageDigest:                  sha256StringDigest("proxy"),
		DumpRequests:                      boolPtr(false),
		SharedPromptCache:                 boolPtr(false),
		NvidiaOCSPAllowUnknown:            boolPtr(false),
		NvidiaOCSPRevokedGracePeriodHours: intPtr(0),
	}
	cases := []struct {
		name    string
		mutate  func(*verifierPolicy)
		wantErr string
	}{
		{
			name:    "api base",
			mutate:  func(p *verifierPolicy) { p.APIBaseURL = "api.privatemode.ai" },
			wantErr: "api_base_url must be an absolute URL",
		},
		{
			name:    "cdn base",
			mutate:  func(p *verifierPolicy) { p.CDNBaseURL = "cdn.confidential.cloud/privatemode/v2" },
			wantErr: "cdn_base_url must be an absolute URL",
		},
		{
			name:    "manifest",
			mutate:  func(p *verifierPolicy) { p.ManifestURL = "example.test/manifest.json" },
			wantErr: "manifest_url must be an absolute URL",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := valid
			tc.mutate(&p)
			err := validatePolicy(p)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected %q, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestValidatePayloadBindingsRejectsPlaceholderDigests(t *testing.T) {
	err := validatePayloadBindings(verifierPayload{
		PolicyDigest:      "sha256:policy",
		EvidenceDigest:    sha256StringDigest("evidence"),
		VerificationNonce: "nonce",
	})

	if err == nil {
		t.Fatal("validatePayloadBindings accepted placeholder policy digest")
	}
	if !strings.Contains(err.Error(), "policy_digest must be a full sha256 digest") {
		t.Fatalf("error = %q, want policy_digest", err)
	}
}

func TestValidatePayloadBindingsRejectsRepeatedHexPlaceholderDigests(t *testing.T) {
	err := validatePayloadBindings(verifierPayload{
		PolicyDigest:      "sha256:" + strings.Repeat("a", 64),
		EvidenceDigest:    sha256StringDigest("evidence"),
		VerificationNonce: "nonce",
		ProxyBaseURL:      "http://127.0.0.1:8080",
	})

	if err == nil {
		t.Fatal("validatePayloadBindings accepted repeated-hex placeholder digest")
	}
	if !strings.Contains(err.Error(), "policy_digest must be a full sha256 digest") {
		t.Fatalf("error = %q, want policy_digest", err)
	}
}

func TestValidatePolicyRejectsRepeatedHexPlaceholderArtifactDigests(t *testing.T) {
	policy := verifierPolicy{
		ManifestDigest:                    "sha256:" + strings.Repeat("b", 64),
		ProxyBinaryDigest:                 sha256StringDigest("proxy"),
		ProxyBinaryPath:                   "/opt/privatemode/privatemode-proxy",
		DumpRequests:                      boolPtr(false),
		SharedPromptCache:                 boolPtr(false),
		NvidiaOCSPAllowUnknown:            boolPtr(false),
		NvidiaOCSPRevokedGracePeriodHours: intPtr(0),
		ExpectedWorkloadSANs:              []string{"workload-kimi-k2-6"},
		ModelWorkloadBindings: map[string]modelWorkloadBinding{
			"privatemode/kimi-k2-6": {
				WorkloadSANs: []string{"workload-kimi-k2-6"},
			},
		},
	}

	err := validatePolicy(policy)

	if err == nil {
		t.Fatal("validatePolicy accepted repeated-hex placeholder manifest digest")
	}
	if !strings.Contains(err.Error(), "manifest_digest must be a sha256 digest") {
		t.Fatalf("error = %q, want manifest_digest", err)
	}
}

func TestValidatePayloadBindingsRejectsMissingNonce(t *testing.T) {
	err := validatePayloadBindings(verifierPayload{
		PolicyDigest:   sha256StringDigest("policy"),
		EvidenceDigest: sha256StringDigest("evidence"),
	})

	if err == nil {
		t.Fatal("validatePayloadBindings accepted missing nonce")
	}
	if !strings.Contains(err.Error(), "verification_nonce is required") {
		t.Fatalf("error = %q, want verification_nonce", err)
	}
}

func TestValidatePayloadBindingsRejectsMalformedVerifierDigest(t *testing.T) {
	err := validatePayloadBindings(verifierPayload{
		PolicyDigest:      sha256StringDigest("policy"),
		EvidenceDigest:    sha256StringDigest("evidence"),
		VerificationNonce: "nonce",
		VerifierDigest:    "sha256:verifier",
	})

	if err == nil {
		t.Fatal("validatePayloadBindings accepted malformed verifier digest")
	}
	if !strings.Contains(err.Error(), "verifier_command_digest must be a full sha256 digest") {
		t.Fatalf("error = %q, want verifier_command_digest", err)
	}
}

func TestValidatePayloadBindingsRequiresProxyBaseURL(t *testing.T) {
	err := validatePayloadBindings(verifierPayload{
		PolicyDigest:      sha256StringDigest("policy"),
		EvidenceDigest:    sha256StringDigest("evidence"),
		VerificationNonce: "nonce",
	})

	if err == nil {
		t.Fatal("validatePayloadBindings accepted missing proxy_base_url")
	}
	if !strings.Contains(err.Error(), "proxy_base_url is required") {
		t.Fatalf("error = %q, want proxy_base_url is required", err)
	}
}

func TestValidatePayloadBindingsRejectsUnsafeProxyBaseURL(t *testing.T) {
	valid := verifierPayload{
		PolicyDigest:      sha256StringDigest("policy"),
		EvidenceDigest:    sha256StringDigest("evidence"),
		VerificationNonce: "nonce",
	}
	cases := []struct {
		name    string
		value   string
		wantErr string
	}{
		{
			name:    "credential bearing",
			value:   "http://operator:secret@127.0.0.1:8080/v1",
			wantErr: "proxy_base_url must not contain userinfo",
		},
		{
			name:    "remote host",
			value:   "http://privatemode-proxy:8080/v1",
			wantErr: "proxy_base_url must use a loopback host",
		},
		{
			name:    "relative loopback",
			value:   "127.0.0.1:8080/v1",
			wantErr: "proxy_base_url must be an absolute URL",
		},
		{
			name:    "unsupported scheme",
			value:   "ws://127.0.0.1:8080/v1",
			wantErr: "proxy_base_url must use http or https",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := valid
			p.ProxyBaseURL = tc.value
			err := validatePayloadBindings(p)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected %q, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestValidatePayloadBindingsAcceptsLoopbackProxyBaseURL(t *testing.T) {
	err := validatePayloadBindings(verifierPayload{
		PolicyDigest:      sha256StringDigest("policy"),
		EvidenceDigest:    sha256StringDigest("evidence"),
		VerificationNonce: "nonce",
		ProxyBaseURL:      "http://127.0.0.1:8080/v1",
	})

	if err != nil {
		t.Fatalf("validatePayloadBindings rejected loopback proxy URL: %v", err)
	}
}

func TestBuildClaimsBindsVerifiedEvidenceAndArtifact(t *testing.T) {
	dir := t.TempDir()
	proxyPath := filepath.Join(dir, "privatemode-proxy")
	proxyBytes := []byte("proxy artifact")
	if err := os.WriteFile(proxyPath, proxyBytes, 0o700); err != nil {
		t.Fatal(err)
	}

	manifest := []byte(`{
		"Policies": {
			"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": {
				"SANs": ["continuum-etcd-client", "workload-kimi-k2-6", "*"],
				"WorkloadSecretID": "apps/v1/Deployment/continuum-e37856/workload-kimi-k2-6"
			}
		}
	}`)
	evidence := verifiedPrivatemodeEvidence{
		Manifest:                   manifest,
		CoordinatorAttestationDoc:  []byte("attestation-doc"),
		MeshCAPEM:                  []byte("mesh-ca"),
		SecretServiceCertificate:   []byte("secret-service-cert"),
		InferenceSecretID:          "secret-id",
		InferenceSecret:            [32]byte{1, 2, 3},
		PromptEncryptionCiphertext: `"secret-id:nonce:iv:ciphertext"`,
	}
	p := verifierPolicy{
		ManifestDigest:                    sha256BytesDigest(manifest),
		ProxyBinaryPath:                   proxyPath,
		ProxyBinaryDigest:                 sha256BytesDigest(proxyBytes),
		DumpRequests:                      boolPtr(false),
		SharedPromptCache:                 boolPtr(false),
		NvidiaOCSPAllowUnknown:            boolPtr(false),
		NvidiaOCSPRevokedGracePeriodHours: intPtr(0),
		ExpectedWorkloadSANs:              []string{"workload-kimi-k2-6"},
	}

	claims, err := buildClaims(p, evidence, "http://127.0.0.1:8080/v1")
	if err != nil {
		t.Fatalf("buildClaims returned error: %v", err)
	}
	if claims["manifest_digest"] != p.ManifestDigest {
		t.Fatalf("manifest digest claim mismatch: %v", claims["manifest_digest"])
	}
	if claims["proxy_binary_digest"] != p.ProxyBinaryDigest {
		t.Fatalf("proxy binary digest claim mismatch: %v", claims["proxy_binary_digest"])
	}
	if claims["transport"] != "privatemode-proxy" {
		t.Fatalf("transport claim mismatch: %v", claims["transport"])
	}
	if claims["proxy_base_url"] != "http://127.0.0.1:8080/v1" {
		t.Fatalf("proxy_base_url claim mismatch: %v", claims["proxy_base_url"])
	}
	if claims["coordinator_measurement"] != sha256BytesDigest(evidence.CoordinatorAttestationDoc) {
		t.Fatalf("coordinator claim mismatch: %v", claims["coordinator_measurement"])
	}
	if claims["secret_service_measurement"] != sha256BytesDigest(evidence.SecretServiceCertificate) {
		t.Fatalf("secret-service claim mismatch: %v", claims["secret_service_measurement"])
	}
	if claims["ai_worker_measurement"] != claims["attested_workload_policy_digest"] {
		t.Fatalf("AI-worker claim mismatch: %v", claims["ai_worker_measurement"])
	}
	if claims["attested_workload_identity_digest"] == "" {
		t.Fatal("attested workload identity digest claim is empty")
	}
	if claims["attested_workload_policy_digest"] == "" {
		t.Fatal("attested workload policy digest claim is empty")
	}
	if claims["expected_workload_identity_digest"] != expectedWorkloadIdentityDigest(p) {
		t.Fatalf("expected workload digest mismatch: %v", claims["expected_workload_identity_digest"])
	}
	if claims["gpu_attestation_policy"] != "nvidia-ocsp-good-only" {
		t.Fatalf("GPU policy claim mismatch: %v", claims["gpu_attestation_policy"])
	}
	if claims["key_release_binding"] == "" {
		t.Fatal("key release binding claim is empty")
	}
	steps, ok := claims["verification_steps"].(map[string]bool)
	if !ok {
		t.Fatalf("verification_steps has unexpected type %T", claims["verification_steps"])
	}
	for _, step := range privatemodeVerificationStepNames {
		if !steps[step] {
			t.Fatalf("verification step %q was not marked true", step)
		}
	}
}

func TestBuildClaimsRequiresManifestLogInclusion(t *testing.T) {
	dir := t.TempDir()
	proxyPath := filepath.Join(dir, "privatemode-proxy")
	proxyBytes := []byte("proxy artifact")
	if err := os.WriteFile(proxyPath, proxyBytes, 0o700); err != nil {
		t.Fatal(err)
	}

	logDir := filepath.Join(dir, "manifests")
	if err := os.Mkdir(logDir, 0o700); err != nil {
		t.Fatal(err)
	}
	manifest := []byte(`{
		"Policies": {
			"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": {
				"SANs": ["continuum-etcd-client", "workload-kimi-k2-6", "*"],
				"WorkloadSecretID": "apps/v1/Deployment/continuum-e37856/workload-kimi-k2-6"
			}
		}
	}`)
	evidence := verifiedPrivatemodeEvidence{
		Manifest:                   manifest,
		CoordinatorAttestationDoc:  []byte("attestation-doc"),
		MeshCAPEM:                  []byte("mesh-ca"),
		SecretServiceCertificate:   []byte("secret-service-cert"),
		InferenceSecretID:          "secret-id",
		InferenceSecret:            [32]byte{1, 2, 3},
		PromptEncryptionCiphertext: `"secret-id:nonce:iv:ciphertext"`,
		OCSPPolicyHeader:           "ocsp-policy",
		OCSPPolicyMAC:              "ocsp-mac",
	}
	p := verifierPolicy{
		ManifestLogDir:                    logDir,
		ProxyBinaryPath:                   proxyPath,
		ProxyBinaryDigest:                 sha256BytesDigest(proxyBytes),
		DumpRequests:                      boolPtr(false),
		SharedPromptCache:                 boolPtr(false),
		NvidiaOCSPAllowUnknown:            boolPtr(false),
		NvidiaOCSPRevokedGracePeriodHours: intPtr(0),
		ExpectedWorkloadSANs:              []string{"workload-kimi-k2-6"},
	}

	_, err := buildClaims(p, evidence, "http://127.0.0.1:8080/v1")
	if err == nil {
		t.Fatal("buildClaims accepted manifest_log_dir without active manifest inclusion")
	}

	manifestName := "manifest-2026-05-28.json"
	if err := os.WriteFile(filepath.Join(logDir, manifestName), manifest, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(logDir, "log.txt"), []byte("2026-05-28T00:00:00Z "+manifestName+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	claims, err := buildClaims(p, evidence, "http://127.0.0.1:8080/v1")
	if err != nil {
		t.Fatalf("buildClaims returned error after manifest log inclusion: %v", err)
	}
	if claims["manifest_log_manifest_digest"] != sha256BytesDigest(manifest) {
		t.Fatalf("manifest log inclusion digest mismatch: %v", claims["manifest_log_manifest_digest"])
	}
	if claims["manifest_log_entry"] == "" {
		t.Fatal("manifest_log_entry claim is empty")
	}
}

func TestBuildClaimsRequiresExpectedWorkloadsInAttestedManifest(t *testing.T) {
	manifest := []byte(`{
		"Policies": {
			"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": {
				"SANs": ["continuum-etcd-client", "workload-gpt-oss-120b", "*"],
				"WorkloadSecretID": "apps/v1/Deployment/continuum-e37856/workload-gpt-oss-120b"
			}
		}
	}`)
	evidence := verifiedPrivatemodeEvidence{
		Manifest:                   manifest,
		CoordinatorAttestationDoc:  []byte("attestation-doc"),
		MeshCAPEM:                  []byte("mesh-ca"),
		SecretServiceCertificate:   []byte("secret-service-cert"),
		InferenceSecretID:          "secret-id",
		InferenceSecret:            [32]byte{1, 2, 3},
		PromptEncryptionCiphertext: `"secret-id:nonce:iv:ciphertext"`,
		OCSPPolicyHeader:           "ocsp-policy",
		OCSPPolicyMAC:              "ocsp-mac",
	}
	p := parsePolicy(map[string]interface{}{
		"manifest_digest":                        sha256BytesDigest(manifest),
		"proxy_image_digest":                     sha256StringDigest("proxy"),
		"expected_workload_sans":                 []interface{}{"workload-kimi-k2-6"},
		"dump_requests":                          false,
		"shared_prompt_cache":                    false,
		"nvidia_ocsp_allow_unknown":              false,
		"nvidia_ocsp_revoked_grace_period_hours": 0,
	})

	_, err := buildClaims(p, evidence, "http://127.0.0.1:8080/v1")
	if err == nil {
		t.Fatal("buildClaims accepted a manifest missing the expected Privatemode workload")
	}
	if !strings.Contains(err.Error(), "expected workload") {
		t.Fatalf("expected workload binding error, got %v", err)
	}
}

func TestValidatePolicyRequiresModelWorkloadBindings(t *testing.T) {
	p := parsePolicy(map[string]interface{}{
		"manifest_digest":                        sha256StringDigest("manifest"),
		"proxy_image_digest":                     sha256StringDigest("proxy"),
		"expected_workload_sans":                 []interface{}{"workload-kimi-k2-6"},
		"dump_requests":                          false,
		"shared_prompt_cache":                    false,
		"nvidia_ocsp_allow_unknown":              false,
		"nvidia_ocsp_revoked_grace_period_hours": 0,
	})

	err := validatePolicy(p)
	if err == nil {
		t.Fatal("validatePolicy accepted a Privatemode policy without model_workload_bindings")
	}
	if !strings.Contains(err.Error(), "model_workload_bindings is required") {
		t.Fatalf("expected model_workload_bindings error, got %v", err)
	}
}

func TestValidatePolicyRejectsUnprefixedModelWorkloadBindings(t *testing.T) {
	p := parsePolicy(map[string]interface{}{
		"manifest_digest":        sha256StringDigest("manifest"),
		"proxy_image_digest":     sha256StringDigest("proxy"),
		"expected_workload_sans": []interface{}{"workload-kimi-k2-6"},
		"model_workload_bindings": map[string]interface{}{
			"kimi-k2-6": map[string]interface{}{
				"workload_sans": []interface{}{"workload-kimi-k2-6"},
			},
		},
		"dump_requests":                          false,
		"shared_prompt_cache":                    false,
		"nvidia_ocsp_allow_unknown":              false,
		"nvidia_ocsp_revoked_grace_period_hours": 0,
	})

	err := validatePolicy(p)
	if err == nil {
		t.Fatal("validatePolicy accepted unprefixed Privatemode model binding")
	}
	if !strings.Contains(err.Error(), "model_workload_bindings keys must start with privatemode/") {
		t.Fatalf("expected Privatemode model prefix error, got %v", err)
	}
}

func TestValidatePolicyRejectsCaseVariantModelWorkloadBindings(t *testing.T) {
	p := parsePolicy(map[string]interface{}{
		"manifest_digest":        sha256StringDigest("manifest"),
		"proxy_binary_digest":    sha256StringDigest("proxy-binary"),
		"proxy_binary_path":      "/opt/privatemode/privatemode-proxy",
		"expected_workload_sans": []interface{}{"workload-kimi-k2-6"},
		"model_workload_bindings": map[string]interface{}{
			"privatemode/Kimi-K2-6": map[string]interface{}{
				"workload_sans": []interface{}{"workload-kimi-k2-6"},
			},
			"privatemode/kimi-k2-6": map[string]interface{}{
				"workload_sans": []interface{}{"workload-kimi-k2-6"},
			},
		},
		"dump_requests":                          false,
		"shared_prompt_cache":                    false,
		"nvidia_ocsp_allow_unknown":              false,
		"nvidia_ocsp_revoked_grace_period_hours": 0,
	})

	err := validatePolicy(p)
	if err == nil {
		t.Fatal("validatePolicy accepted case-variant Privatemode model bindings")
	}
	if !strings.Contains(err.Error(), "model_workload_bindings contains duplicate model binding privatemode/kimi-k2-6") {
		t.Fatalf("expected duplicate model binding error, got %v", err)
	}
}

func TestValidatePolicyRejectsScalarExpectedWorkloadSelectors(t *testing.T) {
	p := parsePolicy(map[string]interface{}{
		"manifest_digest":        sha256StringDigest("manifest"),
		"proxy_image_digest":     sha256StringDigest("proxy"),
		"expected_workload_sans": "workload-kimi-k2-6",
		"model_workload_bindings": map[string]interface{}{
			"privatemode/kimi-k2-6": map[string]interface{}{
				"workload_sans": []interface{}{"workload-kimi-k2-6"},
			},
		},
		"dump_requests":                          false,
		"shared_prompt_cache":                    false,
		"nvidia_ocsp_allow_unknown":              false,
		"nvidia_ocsp_revoked_grace_period_hours": 0,
	})

	err := validatePolicy(p)
	if err == nil {
		t.Fatal("validatePolicy accepted scalar expected_workload_sans")
	}
	if !strings.Contains(err.Error(), "expected_workload_sans must be a list of non-empty strings") {
		t.Fatalf("expected expected_workload_sans list error, got %v", err)
	}
}

func TestValidatePolicyRejectsDuplicateExpectedWorkloadSelectors(t *testing.T) {
	p := parsePolicy(map[string]interface{}{
		"manifest_digest":    sha256StringDigest("manifest"),
		"proxy_image_digest": sha256StringDigest("proxy"),
		"expected_workload_sans": []interface{}{
			"workload-kimi-k2-6",
			" workload-kimi-k2-6 ",
		},
		"model_workload_bindings": map[string]interface{}{
			"privatemode/kimi-k2-6": map[string]interface{}{
				"workload_sans": []interface{}{"workload-kimi-k2-6"},
			},
		},
		"dump_requests":                          false,
		"shared_prompt_cache":                    false,
		"nvidia_ocsp_allow_unknown":              false,
		"nvidia_ocsp_revoked_grace_period_hours": 0,
	})

	err := validatePolicy(p)
	if err == nil {
		t.Fatal("validatePolicy accepted duplicate expected_workload_sans")
	}
	if !strings.Contains(err.Error(), "expected_workload_sans must not contain duplicates") {
		t.Fatalf("expected duplicate expected_workload_sans error, got %v", err)
	}
}

func TestBuildClaimsIncludesModelWorkloadBindingDigest(t *testing.T) {
	manifest := []byte(`{
		"Policies": {
			"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": {
				"SANs": ["continuum-etcd-client", "workload-kimi-k2-6", "*"],
				"WorkloadSecretID": "apps/v1/Deployment/continuum-e37856/workload-kimi-k2-6"
			}
		}
	}`)
	evidence := verifiedPrivatemodeEvidence{
		Manifest:                   manifest,
		CoordinatorAttestationDoc:  []byte("attestation-doc"),
		MeshCAPEM:                  []byte("mesh-ca"),
		SecretServiceCertificate:   []byte("secret-service-cert"),
		InferenceSecretID:          "secret-id",
		InferenceSecret:            [32]byte{1, 2, 3},
		PromptEncryptionCiphertext: `"secret-id:nonce:iv:ciphertext"`,
		OCSPPolicyHeader:           "ocsp-policy",
		OCSPPolicyMAC:              "ocsp-mac",
	}
	p := parsePolicy(map[string]interface{}{
		"manifest_digest":        sha256BytesDigest(manifest),
		"proxy_image_digest":     sha256StringDigest("proxy"),
		"expected_workload_sans": []interface{}{"workload-kimi-k2-6"},
		"model_workload_bindings": map[string]interface{}{
			"privatemode/kimi-k2-6": map[string]interface{}{
				"workload_sans": []interface{}{"workload-kimi-k2-6"},
			},
		},
		"dump_requests":                          false,
		"shared_prompt_cache":                    false,
		"nvidia_ocsp_allow_unknown":              false,
		"nvidia_ocsp_revoked_grace_period_hours": 0,
	})

	claims, err := buildClaims(p, evidence, "http://127.0.0.1:8080/v1")
	if err != nil {
		t.Fatalf("buildClaims returned error: %v", err)
	}
	if claims["model_workload_binding_digest"] != modelWorkloadBindingDigest(p) {
		t.Fatalf("model workload binding digest mismatch: %v", claims["model_workload_binding_digest"])
	}
	selectedModels, ok := claims["selected_model_ids"].([]string)
	if !ok || len(selectedModels) != 1 || selectedModels[0] != "privatemode/kimi-k2-6" {
		t.Fatalf("selected model ids = %#v, want privatemode/kimi-k2-6", claims["selected_model_ids"])
	}
}

func TestVerifierResultBindsPayloadDigests(t *testing.T) {
	payload := verifierPayload{
		SchemaVersion:     schemaVersion,
		Mode:              "privatemode",
		PolicyDigest:      sha256StringDigest("policy"),
		EvidenceDigest:    sha256StringDigest("evidence"),
		VerificationNonce: "nonce",
	}
	result := verifierResultForPayload(payload, map[string]interface{}{
		"manifest_digest": sha256StringDigest("manifest"),
	}, time.Unix(1_700_000_000, 0))

	if !result.Verified {
		t.Fatal("result was not verified")
	}
	if result.PolicyDigest != payload.PolicyDigest {
		t.Fatalf("policy digest mismatch: %q", result.PolicyDigest)
	}
	if result.EvidenceDigest != payload.EvidenceDigest {
		t.Fatalf("evidence digest mismatch: %q", result.EvidenceDigest)
	}
	if result.VerificationNonce != payload.VerificationNonce {
		t.Fatalf("verification nonce mismatch: %q", result.VerificationNonce)
	}
	if result.Claims["payload_policy_digest"] != payload.PolicyDigest {
		t.Fatalf("payload policy claim mismatch: %v", result.Claims["payload_policy_digest"])
	}
}

func boolPtr(v bool) *bool {
	return &v
}

func intPtr(v int) *int {
	return &v
}

func runWithStdinForTest(t *testing.T, input string) error {
	t.Helper()
	file, err := os.CreateTemp(t.TempDir(), "verifier-input-*.json")
	if err != nil {
		t.Fatalf("create temp stdin: %v", err)
	}
	if _, err := file.WriteString(input); err != nil {
		t.Fatalf("write temp stdin: %v", err)
	}
	if _, err := file.Seek(0, 0); err != nil {
		t.Fatalf("rewind temp stdin: %v", err)
	}
	oldStdin := os.Stdin
	os.Stdin = file
	t.Cleanup(func() {
		os.Stdin = oldStdin
		_ = file.Close()
	})
	return run()
}
