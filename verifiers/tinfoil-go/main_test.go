package main

import (
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/tinfoilsh/tinfoil-go/verifier/attestation"
	"github.com/tinfoilsh/tinfoil-go/verifier/client"
)

func testEHBPKeyConfig(publicKey byte) string {
	keyConfig := append(
		[]byte{0x00, 0x00, 0x20},
		bytesOf(publicKey, 32)...,
	)
	keyConfig = append(keyConfig, 0x00, 0x04, 0x00, 0x01, 0x00, 0x02)
	return base64.StdEncoding.EncodeToString(keyConfig)
}

func bytesOf(value byte, count int) []byte {
	out := make([]byte, count)
	for i := range out {
		out[i] = value
	}
	return out
}

func testTinfoilAttestationBody(report []byte) string {
	var compressed bytes.Buffer
	writer := gzip.NewWriter(&compressed)
	_, _ = writer.Write(report)
	_ = writer.Close()
	return base64.StdEncoding.EncodeToString(compressed.Bytes())
}

func writeTestProxyBinary(t *testing.T, contents []byte) string {
	t.Helper()
	path := t.TempDir() + "/ppq-private-proxy"
	if err := os.WriteFile(path, contents, 0o755); err != nil {
		t.Fatalf("write proxy binary: %v", err)
	}
	return path
}

func testPPQPrivateProxyPolicy(t *testing.T) map[string]interface{} {
	t.Helper()
	contents := []byte("test ppq private proxy")
	return map[string]interface{}{
		"proxy_binary_path":   writeTestProxyBinary(t, contents),
		"proxy_binary_digest": sha256Bytes(contents),
	}
}

func withTestPPQPrivateProxyPolicy(
	t *testing.T,
	policy map[string]interface{},
) map[string]interface{} {
	t.Helper()
	for key, value := range testPPQPrivateProxyPolicy(t) {
		policy[key] = value
	}
	return policy
}

func TestEHBPPublicKeyHex(t *testing.T) {
	got, err := ehbpPublicKeyHex(testEHBPKeyConfig(0x33))
	if err != nil {
		t.Fatalf("ehbpPublicKeyHex returned error: %v", err)
	}
	want := strings.Repeat("33", 32)
	if got != want {
		t.Fatalf("public key = %q, want %q", got, want)
	}
}

func TestEHBPPublicKeyHexRejectsUnsupportedSuite(t *testing.T) {
	keyConfig := append([]byte{0x00, 0x00, 0x20}, bytesOf(0x33, 32)...)
	keyConfig = append(keyConfig, 0x00, 0x04, 0x00, 0x01, 0x00, 0x01)

	_, err := ehbpPublicKeyHex(base64.StdEncoding.EncodeToString(keyConfig))
	if err == nil {
		t.Fatal("ehbpPublicKeyHex returned nil error for unsupported suite")
	}
	if !strings.Contains(err.Error(), "unsupported suite") {
		t.Fatalf("error = %q, want unsupported suite", err)
	}
}

func TestVerifierNameForModeIsProviderSpecific(t *testing.T) {
	cases := []struct {
		mode string
		want string
	}{
		{"tinfoil", "routstr-tinfoil-go-verifier"},
		{"ppq-private-tee", "routstr-ppq-private-go-verifier"},
	}

	for _, tc := range cases {
		if got := verifierNameForMode(tc.mode); got != tc.want {
			t.Fatalf("verifierNameForMode(%q) = %q, want %q", tc.mode, got, tc.want)
		}
	}
}

func TestVerifyPPQPrivateRetriesAttestationBundleHPKEMismatch(t *testing.T) {
	releaseDigest := strings.Repeat("a1", 32)
	matchingKey := strings.Repeat("33", 32)
	mismatchedKey := strings.Repeat("44", 32)
	measurement := strings.Repeat("b1", 48)
	callCount := 0
	oldFetch := fetchAndVerifyFromURLJSON
	fetchAndVerifyFromURLJSON = func(
		attestationBundleURL string,
		repo string,
		sigstoreTrustedRootJSON []byte,
	) (string, error) {
		callCount++
		hpkeKey := mismatchedKey
		if callCount == 2 {
			hpkeKey = matchingKey
		}
		groundTruth := client.GroundTruth{
			TLSPublicKey:  strings.Repeat("55", 32),
			HPKEPublicKey: hpkeKey,
			Digest:        releaseDigest,
			CodeMeasurement: &attestation.Measurement{
				Type:      attestation.SevGuestV2,
				Registers: []string{measurement},
			},
			EnclaveMeasurement: &attestation.Measurement{
				Type:      attestation.SevGuestV2,
				Registers: []string{measurement},
			},
		}
		encoded, err := json.Marshal(groundTruth)
		if err != nil {
			t.Fatalf("marshal ground truth: %v", err)
		}
		return string(encoded), nil
	}
	t.Cleanup(func() { fetchAndVerifyFromURLJSON = oldFetch })

	claims, err := verifyPPQPrivate(payload{
		Mode:                 "ppq-private-tee",
		BaseURL:              "https://api.ppq.ai/private/v1",
		ModelIDs:             []string{"private/gpt-oss-120b"},
		AttestationBundleURL: "https://api.ppq.ai/private",
		HPKEKeyConfigB64:     testEHBPKeyConfig(0x33),
		Policy: withTestPPQPrivateProxyPolicy(t, map[string]interface{}{
			"repo":                    "tinfoilsh/confidential-model-router",
			"expected_release_digest": releaseDigest,
		}),
	})

	if err != nil {
		t.Fatalf("verifyPPQPrivate returned error: %v", err)
	}
	if callCount != 2 {
		t.Fatalf("fetch attempts = %d, want 2", callCount)
	}
	if claims["attested_hpke_public_key_hex"] != matchingKey {
		t.Fatalf("attested HPKE key = %v, want %s", claims["attested_hpke_public_key_hex"], matchingKey)
	}
	if claims["client_encryption_boundary"] != "routstr-tee-ehbp-proxy" {
		t.Fatalf(
			"client_encryption_boundary = %v, want routstr-tee-ehbp-proxy",
			claims["client_encryption_boundary"],
		)
	}
}

func TestVerifyPPQPrivateAcceptsEHBPWithoutProxyBinaryPolicy(t *testing.T) {
	releaseDigest := strings.Repeat("a1", 32)
	matchingKey := strings.Repeat("33", 32)
	measurement := strings.Repeat("b1", 48)
	oldFetch := fetchAndVerifyFromURLJSON
	fetchAndVerifyFromURLJSON = func(
		attestationBundleURL string,
		repo string,
		sigstoreTrustedRootJSON []byte,
	) (string, error) {
		groundTruth := client.GroundTruth{
			TLSPublicKey:  strings.Repeat("55", 32),
			HPKEPublicKey: matchingKey,
			Digest:        releaseDigest,
			CodeMeasurement: &attestation.Measurement{
				Type:      attestation.SevGuestV2,
				Registers: []string{measurement},
			},
			EnclaveMeasurement: &attestation.Measurement{
				Type:      attestation.SevGuestV2,
				Registers: []string{measurement},
			},
		}
		encoded, err := json.Marshal(groundTruth)
		if err != nil {
			t.Fatalf("marshal ground truth: %v", err)
		}
		return string(encoded), nil
	}
	t.Cleanup(func() { fetchAndVerifyFromURLJSON = oldFetch })

	claims, err := verifyPPQPrivate(payload{
		Mode:                 "ppq-private-tee",
		BaseURL:              "https://api.ppq.ai/private/v1",
		ModelIDs:             []string{"private/gpt-oss-120b"},
		AttestationBundleURL: "https://api.ppq.ai/private",
		HPKEKeyConfigB64:     testEHBPKeyConfig(0x33),
		Policy: map[string]interface{}{
			"repo":                    "tinfoilsh/confidential-model-router",
			"expected_release_digest": releaseDigest,
		},
	})

	if err != nil {
		t.Fatalf("verifyPPQPrivate returned error: %v", err)
	}
	if _, ok := claims["proxy_binary_digest"]; ok {
		t.Fatalf("claims unexpectedly include proxy_binary_digest: %#v", claims)
	}
}

func TestVerifyPPQPrivateIgnoresLegacyProxyBinaryDigestMismatch(t *testing.T) {
	releaseDigest := strings.Repeat("a1", 32)
	matchingKey := strings.Repeat("33", 32)
	measurement := strings.Repeat("b1", 48)
	proxyPath := writeTestProxyBinary(t, []byte("actual proxy binary"))
	oldFetch := fetchAndVerifyFromURLJSON
	fetchAndVerifyFromURLJSON = func(
		attestationBundleURL string,
		repo string,
		sigstoreTrustedRootJSON []byte,
	) (string, error) {
		groundTruth := client.GroundTruth{
			TLSPublicKey:  strings.Repeat("55", 32),
			HPKEPublicKey: matchingKey,
			Digest:        releaseDigest,
			CodeMeasurement: &attestation.Measurement{
				Type:      attestation.SevGuestV2,
				Registers: []string{measurement},
			},
			EnclaveMeasurement: &attestation.Measurement{
				Type:      attestation.SevGuestV2,
				Registers: []string{measurement},
			},
		}
		encoded, err := json.Marshal(groundTruth)
		if err != nil {
			t.Fatalf("marshal ground truth: %v", err)
		}
		return string(encoded), nil
	}
	t.Cleanup(func() { fetchAndVerifyFromURLJSON = oldFetch })

	claims, err := verifyPPQPrivate(payload{
		Mode:                 "ppq-private-tee",
		BaseURL:              "https://api.ppq.ai/private/v1",
		ModelIDs:             []string{"private/gpt-oss-120b"},
		AttestationBundleURL: "https://api.ppq.ai/private",
		HPKEKeyConfigB64:     testEHBPKeyConfig(0x33),
		Policy: map[string]interface{}{
			"repo":                    "tinfoilsh/confidential-model-router",
			"expected_release_digest": releaseDigest,
			"proxy_binary_path":       proxyPath,
			"proxy_binary_digest":     sha256Bytes([]byte("different proxy binary")),
		},
	})

	if err != nil {
		t.Fatalf("verifyPPQPrivate returned error: %v", err)
	}
	if _, ok := claims["proxy_binary_digest"]; ok {
		t.Fatalf("claims unexpectedly include proxy_binary_digest: %#v", claims)
	}
}

func TestVerifyPPQPrivateRejectsPersistentAttestationBundleHPKEMismatch(t *testing.T) {
	releaseDigest := strings.Repeat("a1", 32)
	mismatchedKey := strings.Repeat("44", 32)
	measurement := strings.Repeat("b1", 48)
	callCount := 0
	oldFetch := fetchAndVerifyFromURLJSON
	fetchAndVerifyFromURLJSON = func(
		attestationBundleURL string,
		repo string,
		sigstoreTrustedRootJSON []byte,
	) (string, error) {
		callCount++
		groundTruth := client.GroundTruth{
			TLSPublicKey:  strings.Repeat("55", 32),
			HPKEPublicKey: mismatchedKey,
			Digest:        releaseDigest,
			CodeMeasurement: &attestation.Measurement{
				Type:      attestation.SevGuestV2,
				Registers: []string{measurement},
			},
			EnclaveMeasurement: &attestation.Measurement{
				Type:      attestation.SevGuestV2,
				Registers: []string{measurement},
			},
		}
		encoded, err := json.Marshal(groundTruth)
		if err != nil {
			t.Fatalf("marshal ground truth: %v", err)
		}
		return string(encoded), nil
	}
	t.Cleanup(func() { fetchAndVerifyFromURLJSON = oldFetch })

	_, err := verifyPPQPrivate(payload{
		Mode:                 "ppq-private-tee",
		BaseURL:              "https://api.ppq.ai/private/v1",
		ModelIDs:             []string{"private/gpt-oss-120b"},
		AttestationBundleURL: "https://api.ppq.ai/private",
		HPKEKeyConfigB64:     testEHBPKeyConfig(0x33),
		Policy: withTestPPQPrivateProxyPolicy(t, map[string]interface{}{
			"repo":                    "tinfoilsh/confidential-model-router",
			"expected_release_digest": releaseDigest,
		}),
	})

	if err == nil {
		t.Fatal("verifyPPQPrivate accepted a persistent HPKE mismatch")
	}
	if !strings.Contains(err.Error(), "HPKE public key mismatch") {
		t.Fatalf("error = %q, want HPKE mismatch", err)
	}
	if callCount != ppqPrivateAttestationBundleAttempts {
		t.Fatalf("fetch attempts = %d, want %d", callCount, ppqPrivateAttestationBundleAttempts)
	}
}

func TestVerifyPPQPrivateEmitsTrimmedSelectedModelIDs(t *testing.T) {
	releaseDigest := strings.Repeat("a1", 32)
	matchingKey := strings.Repeat("33", 32)
	measurement := strings.Repeat("b1", 48)
	oldFetch := fetchAndVerifyFromURLJSON
	fetchAndVerifyFromURLJSON = func(
		attestationBundleURL string,
		repo string,
		sigstoreTrustedRootJSON []byte,
	) (string, error) {
		groundTruth := client.GroundTruth{
			TLSPublicKey:  strings.Repeat("55", 32),
			HPKEPublicKey: matchingKey,
			Digest:        releaseDigest,
			CodeMeasurement: &attestation.Measurement{
				Type:      attestation.SevGuestV2,
				Registers: []string{measurement},
			},
			EnclaveMeasurement: &attestation.Measurement{
				Type:      attestation.SevGuestV2,
				Registers: []string{measurement},
			},
		}
		encoded, err := json.Marshal(groundTruth)
		if err != nil {
			t.Fatalf("marshal ground truth: %v", err)
		}
		return string(encoded), nil
	}
	t.Cleanup(func() { fetchAndVerifyFromURLJSON = oldFetch })

	claims, err := verifyPPQPrivate(payload{
		Mode:                 "ppq-private-tee",
		BaseURL:              "https://api.ppq.ai/private/v1",
		ModelIDs:             []string{" private/gpt-oss-120b "},
		AttestationBundleURL: "https://api.ppq.ai/private",
		HPKEKeyConfigB64:     testEHBPKeyConfig(0x33),
		Policy: withTestPPQPrivateProxyPolicy(t, map[string]interface{}{
			"repo":                    "tinfoilsh/confidential-model-router",
			"expected_release_digest": releaseDigest,
		}),
	})

	if err != nil {
		t.Fatalf("verifyPPQPrivate returned error: %v", err)
	}
	selectedModels, ok := claims["selected_model_ids"].([]string)
	if !ok {
		t.Fatalf("selected_model_ids = %#v, want []string", claims["selected_model_ids"])
	}
	if len(selectedModels) != 1 || selectedModels[0] != "private/gpt-oss-120b" {
		t.Fatalf("selected_model_ids = %#v, want trimmed private/gpt-oss-120b", selectedModels)
	}
}

func TestVerifyPPQPrivateRequiresBackendModelAttestationPayloads(t *testing.T) {
	releaseDigest := strings.Repeat("a1", 32)
	matchingKey := strings.Repeat("33", 32)
	measurement := strings.Repeat("b1", 48)
	oldFetch := fetchAndVerifyFromURLJSON
	fetchAndVerifyFromURLJSON = func(
		attestationBundleURL string,
		repo string,
		sigstoreTrustedRootJSON []byte,
	) (string, error) {
		groundTruth := client.GroundTruth{
			TLSPublicKey:  strings.Repeat("55", 32),
			HPKEPublicKey: matchingKey,
			Digest:        releaseDigest,
			CodeMeasurement: &attestation.Measurement{
				Type:      attestation.SevGuestV2,
				Registers: []string{measurement},
			},
			EnclaveMeasurement: &attestation.Measurement{
				Type:      attestation.SevGuestV2,
				Registers: []string{measurement},
			},
		}
		encoded, err := json.Marshal(groundTruth)
		if err != nil {
			t.Fatalf("marshal ground truth: %v", err)
		}
		return string(encoded), nil
	}
	t.Cleanup(func() { fetchAndVerifyFromURLJSON = oldFetch })

	_, err := verifyPPQPrivate(payload{
		Mode:                 "ppq-private-tee",
		BaseURL:              "https://api.ppq.ai/private/v1",
		ModelIDs:             []string{"private/gpt-oss-120b"},
		AttestationBundleURL: "https://api.ppq.ai/private",
		HPKEKeyConfigB64:     testEHBPKeyConfig(0x33),
		Policy: withTestPPQPrivateProxyPolicy(t, map[string]interface{}{
			"repo":                       "tinfoilsh/confidential-model-router",
			"expected_release_digest":    releaseDigest,
			"require_model_attestations": true,
			"model_attestation_targets": map[string]interface{}{
				"private/gpt-oss-120b": map[string]interface{}{
					"host":                    "gpt-oss-120b-1.inf10.tinfoil.sh",
					"repo":                    "tinfoilsh/confidential-gpt-oss-120b",
					"expected_release_digest": "sha256:" + strings.Repeat("b7", 32),
				},
			},
		}),
	})

	if err == nil {
		t.Fatal("verifyPPQPrivate accepted policy requiring backend model attestations without payloads")
	}
	if !strings.Contains(err.Error(), "model_attestations must cover every configured model target") {
		t.Fatalf("error = %q, want missing backend model attestation payloads", err)
	}
}

func TestGroundTruthClaimsNormalizesSEVSNPMeasurementFingerprints(t *testing.T) {
	nativeSEVSNPMeasurement := strings.Repeat("b1", 48)
	otherNativeMeasurement := strings.Repeat("c2", 48)
	codeMeasurement := &attestation.Measurement{
		Type: attestation.SnpTdxMultiPlatformV1,
		Registers: []string{
			nativeSEVSNPMeasurement,
			otherNativeMeasurement,
			strings.Repeat("d3", 48),
		},
	}
	enclaveMeasurement := &attestation.Measurement{
		Type:      attestation.SevGuestV2,
		Registers: []string{nativeSEVSNPMeasurement},
	}

	claims, err := groundTruthClaims(
		"tinfoilsh/confidential-model-router",
		strings.Repeat("a", 64),
		codeMeasurement,
		enclaveMeasurement,
		nil,
		strings.Repeat("1", 64),
		strings.Repeat("2", 64),
	)
	if err != nil {
		t.Fatalf("groundTruthClaims returned error: %v", err)
	}

	for _, claimName := range []string{
		"code_measurement_fingerprint",
		"enclave_measurement_fingerprint",
	} {
		claim, ok := claims[claimName].(string)
		if !ok {
			t.Fatalf("%s claim type = %T, want string", claimName, claims[claimName])
		}
		if len(claim) != 64 {
			t.Fatalf("%s claim length = %d, want 64", claimName, len(claim))
		}
		if _, err := normalizeReleaseDigest(claim); err != nil {
			t.Fatalf("%s claim is not normalized sha256 hex: %v", claimName, err)
		}
		if claim == nativeSEVSNPMeasurement {
			t.Fatalf("%s claim leaked native SEV-SNP measurement", claimName)
		}
	}
}

func TestVerifyTLSBindingRejectsSkipPolicy(t *testing.T) {
	err := verifyTLSBinding(
		payload{
			Policy: map[string]interface{}{
				"skip_tls_key_binding": true,
			},
		},
		nil,
	)
	if err == nil {
		t.Fatal("verifyTLSBinding accepted skip_tls_key_binding")
	}
	if !strings.Contains(err.Error(), "skip_tls_key_binding") {
		t.Fatalf("error = %q, want skip_tls_key_binding", err)
	}
}

func TestTinfoilTLSBindingRequiredDefaultsToRequired(t *testing.T) {
	if !tinfoilTLSBindingRequired(map[string]interface{}{}) {
		t.Fatal("empty policy disabled TLS binding")
	}
	if !tinfoilTLSBindingRequired(map[string]interface{}{
		"transport_security": "tls",
	}) {
		t.Fatal("tls policy disabled TLS binding")
	}
}

func TestTinfoilTLSBindingRequiredAllowsExplicitEHBP(t *testing.T) {
	if tinfoilTLSBindingRequired(map[string]interface{}{
		"transport_security": "ehbp",
	}) {
		t.Fatal("transport_security=ehbp required TLS binding")
	}
	if tinfoilTLSBindingRequired(map[string]interface{}{
		"tinfoil_transport_security": "ehbp",
	}) {
		t.Fatal("tinfoil_transport_security=ehbp required TLS binding")
	}
}

func TestTinfoilTLSBindingRequiredRejectsSkipPolicy(t *testing.T) {
	_, err := requireTinfoilTLSBindingPolicy(map[string]interface{}{
		"skip_tls_key_binding": true,
	})
	if err == nil {
		t.Fatal("requireTinfoilTLSBindingPolicy accepted skip_tls_key_binding")
	}
	if !strings.Contains(err.Error(), "skip_tls_key_binding") {
		t.Fatalf("error = %q, want skip_tls_key_binding", err)
	}
}

func TestRunRejectsTrailingJSON(t *testing.T) {
	err := runWithStdinForTest(t, `{"schema_version":"`+schemaVersion+`"}{}`)
	if err == nil || !strings.Contains(err.Error(), "must contain exactly one JSON object") {
		t.Fatalf("expected trailing JSON error, got %v", err)
	}
}

func TestVerifierHostRejectsCredentialBearingHost(t *testing.T) {
	_, err := verifierHost(payload{
		Policy: map[string]interface{}{
			"expected_enclave_host": "token@inference.tinfoil.sh",
		},
	})

	if err == nil {
		t.Fatal("verifierHost accepted credential-bearing host")
	}
	if !strings.Contains(err.Error(), "expected_enclave_host must not contain userinfo") {
		t.Fatalf("error = %q, want expected_enclave_host", err)
	}
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

func TestValidatePayloadBindingsRejectsPlaceholderDigests(t *testing.T) {
	err := validatePayloadBindings(payload{
		PolicyDigest:      "sha256:policy",
		EvidenceDigest:    "sha256:" + strings.Repeat("a", 64),
		VerificationNonce: "nonce",
	})

	if err == nil {
		t.Fatal("validatePayloadBindings accepted placeholder policy digest")
	}
	if !strings.Contains(err.Error(), "policy_digest must be a full sha256 digest") {
		t.Fatalf("error = %q, want policy_digest", err)
	}
}

func TestValidatePayloadBindingsRejectsMissingNonce(t *testing.T) {
	err := validatePayloadBindings(payload{
		PolicyDigest:   "sha256:" + strings.Repeat("a", 64),
		EvidenceDigest: "sha256:" + strings.Repeat("b", 64),
	})

	if err == nil {
		t.Fatal("validatePayloadBindings accepted missing nonce")
	}
	if !strings.Contains(err.Error(), "verification_nonce is required") {
		t.Fatalf("error = %q, want verification_nonce", err)
	}
}

func TestValidatePayloadBindingsRejectsMalformedVerifierDigest(t *testing.T) {
	err := validatePayloadBindings(payload{
		PolicyDigest:          "sha256:" + strings.Repeat("a", 64),
		EvidenceDigest:        "sha256:" + strings.Repeat("b", 64),
		VerificationNonce:     "nonce",
		VerifierCommandDigest: "sha256:verifier",
	})

	if err == nil {
		t.Fatal("validatePayloadBindings accepted malformed verifier digest")
	}
	if !strings.Contains(err.Error(), "verifier_command_digest must be a full sha256 digest") {
		t.Fatalf("error = %q, want verifier_command_digest", err)
	}
}

func TestValidatePayloadBindingsRejectsMalformedTinfoilAttestationDigest(t *testing.T) {
	err := validatePayloadBindings(payload{
		Mode:              "tinfoil",
		BaseURL:           "https://inference.tinfoil.sh/v1",
		PolicyDigest:      "sha256:" + strings.Repeat("a", 64),
		EvidenceDigest:    "sha256:" + strings.Repeat("b", 64),
		VerificationNonce: "nonce",
		Attestation: map[string]interface{}{
			"format":        "https://tinfoil.sh/predicate/sev-snp-guest/v2",
			"body":          testTinfoilAttestationBody([]byte("attestation-report")),
			"report_digest": "sha256:report",
		},
	})

	if err == nil {
		t.Fatal("validatePayloadBindings accepted placeholder attestation report digest")
	}
	if !strings.Contains(err.Error(), "attestation report_digest must be a full sha256 digest") {
		t.Fatalf("error = %q, want attestation report_digest", err)
	}
}

func TestValidatePayloadBindingsRejectsTinfoilMissingRouteHost(t *testing.T) {
	report := []byte("attestation-report")
	err := validatePayloadBindings(payload{
		Mode:              "tinfoil",
		PolicyDigest:      "sha256:" + strings.Repeat("a", 64),
		EvidenceDigest:    "sha256:" + strings.Repeat("b", 64),
		VerificationNonce: "nonce",
		Attestation: map[string]interface{}{
			"format":        "https://tinfoil.sh/predicate/sev-snp-guest/v2",
			"body":          testTinfoilAttestationBody(report),
			"report_digest": sha256Bytes(report),
		},
	})

	if err == nil {
		t.Fatal("validatePayloadBindings accepted Tinfoil payload without route host")
	}
	if !strings.Contains(err.Error(), "base_url, origin, or enclave_host is required for tinfoil") {
		t.Fatalf("error = %q, want Tinfoil route host requirement", err)
	}
}

func TestValidatePayloadBindingsRejectsPPQPrivateMissingBaseURL(t *testing.T) {
	err := validatePayloadBindings(payload{
		Mode:                 "ppq-private-tee",
		ModelIDs:             []string{"private/gpt-oss-120b"},
		PolicyDigest:         "sha256:" + strings.Repeat("a1", 32),
		EvidenceDigest:       "sha256:" + strings.Repeat("b2", 32),
		VerificationNonce:    "nonce",
		AttestationBundleURL: "https://api.ppq.ai/private/attestation.json",
	})

	if err == nil {
		t.Fatal("validatePayloadBindings accepted PPQ private payload without base_url")
	}
	if !strings.Contains(err.Error(), "base_url is required for ppq-private-tee") {
		t.Fatalf("error = %q, want base_url requirement", err)
	}
}

func TestValidatePayloadBindingsRejectsPPQPrivateNonPrivateBaseURL(t *testing.T) {
	err := validatePayloadBindings(payload{
		Mode:                 "ppq-private-tee",
		BaseURL:              "https://api.ppq.ai/v1",
		ModelIDs:             []string{"private/gpt-oss-120b"},
		PolicyDigest:         "sha256:" + strings.Repeat("a1", 32),
		EvidenceDigest:       "sha256:" + strings.Repeat("b2", 32),
		VerificationNonce:    "nonce",
		AttestationBundleURL: "https://api.ppq.ai/private/attestation.json",
	})

	if err == nil {
		t.Fatal("validatePayloadBindings accepted PPQ private payload with normal API base_url")
	}
	if !strings.Contains(err.Error(), "base_url must include a private path segment for ppq-private-tee") {
		t.Fatalf("error = %q, want private path segment requirement", err)
	}
}

func TestValidatePayloadBindingsRejectsPPQPrivateMissingBackendModelAttestations(t *testing.T) {
	err := validatePayloadBindings(payload{
		Mode:                 "ppq-private-tee",
		BaseURL:              "https://api.ppq.ai/private/v1",
		ModelIDs:             []string{"private/gpt-oss-120b"},
		PolicyDigest:         "sha256:" + strings.Repeat("a1", 32),
		EvidenceDigest:       "sha256:" + strings.Repeat("b2", 32),
		VerificationNonce:    "nonce",
		AttestationBundleURL: "https://api.ppq.ai/private/attestation.json",
		Policy: map[string]interface{}{
			"repo":                    "tinfoilsh/confidential-model-router",
			"expected_release_digest": "sha256:" + strings.Repeat("c3", 32),
		},
	})

	if err == nil {
		t.Fatal("validatePayloadBindings accepted PPQ private payload without backend model attestations")
	}
	if !strings.Contains(err.Error(), "model_attestations require require_model_attestations=true") {
		t.Fatalf("error = %q, want backend model attestation requirement", err)
	}
}

func TestValidatePayloadBindingsRejectsDuplicatePPQPrivateModelIDs(t *testing.T) {
	err := validatePayloadBindings(payload{
		Mode:                 "ppq-private-tee",
		BaseURL:              "https://api.ppq.ai/private/v1",
		ModelIDs:             []string{"private/gpt-oss-120b", " private/gpt-oss-120b "},
		PolicyDigest:         "sha256:" + strings.Repeat("a", 64),
		EvidenceDigest:       "sha256:" + strings.Repeat("b", 64),
		VerificationNonce:    "nonce",
		AttestationBundleURL: "https://api.ppq.ai/private/attestation.json",
	})

	if err == nil {
		t.Fatal("validatePayloadBindings accepted duplicate PPQ private model_ids")
	}
	if !strings.Contains(err.Error(), "model_ids must not contain duplicates") {
		t.Fatalf("error = %q, want duplicate model_ids rejection", err)
	}
}

func TestValidatePayloadBindingsRejectsCaseVariantPPQPrivateModelIDs(t *testing.T) {
	err := validatePayloadBindings(payload{
		Mode:                 "ppq-private-tee",
		BaseURL:              "https://api.ppq.ai/private/v1",
		ModelIDs:             []string{"private/GPT-OSS-120B", "private/gpt-oss-120b"},
		PolicyDigest:         "sha256:" + strings.Repeat("a1", 32),
		EvidenceDigest:       "sha256:" + strings.Repeat("b2", 32),
		VerificationNonce:    "nonce",
		AttestationBundleURL: "https://api.ppq.ai/private/attestation.json",
	})

	if err == nil {
		t.Fatal("validatePayloadBindings accepted case-variant PPQ private model_ids")
	}
	if !strings.Contains(err.Error(), "model_ids must not contain duplicates") {
		t.Fatalf("error = %q, want duplicate model_ids rejection", err)
	}
}

func TestValidatePayloadBindingsRejectsCredentialBearingURLs(t *testing.T) {
	valid := payload{
		PolicyDigest:      "sha256:" + strings.Repeat("a", 64),
		EvidenceDigest:    "sha256:" + strings.Repeat("b", 64),
		VerificationNonce: "nonce",
	}
	cases := []struct {
		name    string
		mutate  func(*payload)
		wantErr string
	}{
		{
			name:    "base url",
			mutate:  func(p *payload) { p.BaseURL = "https://user:pass@inference.tinfoil.sh/v1" },
			wantErr: "base_url must not contain userinfo",
		},
		{
			name:    "origin",
			mutate:  func(p *payload) { p.Origin = "https://token@inference.tinfoil.sh" },
			wantErr: "origin must not contain userinfo",
		},
		{
			name: "policy enclave host",
			mutate: func(p *payload) {
				p.Policy = map[string]interface{}{"expected_enclave_host": "token@inference.tinfoil.sh"}
			},
			wantErr: "expected_enclave_host must not contain userinfo",
		},
		{
			name:    "payload attestation bundle",
			mutate:  func(p *payload) { p.AttestationBundleURL = "https://token@api.ppq.ai/private" },
			wantErr: "attestation_bundle_url must not contain userinfo",
		},
		{
			name: "policy attestation bundle",
			mutate: func(p *payload) {
				p.Policy = map[string]interface{}{
					"attestation_bundle_url": "https://token@api.ppq.ai/private",
				}
			},
			wantErr: "attestation_bundle_url must not contain userinfo",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := valid
			tc.mutate(&p)
			err := validatePayloadBindings(p)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected %q, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestValidatePayloadBindingsRejectsNonHTTPSRemoteURLs(t *testing.T) {
	valid := payload{
		PolicyDigest:      "sha256:" + strings.Repeat("a", 64),
		EvidenceDigest:    "sha256:" + strings.Repeat("b", 64),
		VerificationNonce: "nonce",
	}
	cases := []struct {
		name    string
		mutate  func(*payload)
		wantErr string
	}{
		{
			name:    "base url",
			mutate:  func(p *payload) { p.BaseURL = "http://inference.tinfoil.sh/v1" },
			wantErr: "base_url must use https",
		},
		{
			name:    "origin",
			mutate:  func(p *payload) { p.Origin = "http://inference.tinfoil.sh" },
			wantErr: "origin must use https",
		},
		{
			name:    "payload attestation bundle",
			mutate:  func(p *payload) { p.AttestationBundleURL = "http://api.ppq.ai/private" },
			wantErr: "attestation_bundle_url must use https",
		},
		{
			name: "policy attestation bundle",
			mutate: func(p *payload) {
				p.Policy = map[string]interface{}{
					"attestation_bundle_url": "http://api.ppq.ai/private",
				}
			},
			wantErr: "attestation_bundle_url must use https",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := valid
			tc.mutate(&p)
			err := validatePayloadBindings(p)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected %q, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestValidatePayloadBindingsRejectsRelativeRemoteURLs(t *testing.T) {
	valid := payload{
		PolicyDigest:      "sha256:" + strings.Repeat("a", 64),
		EvidenceDigest:    "sha256:" + strings.Repeat("b", 64),
		VerificationNonce: "nonce",
	}
	cases := []struct {
		name    string
		mutate  func(*payload)
		wantErr string
	}{
		{
			name:    "base url",
			mutate:  func(p *payload) { p.BaseURL = "inference.tinfoil.sh/v1" },
			wantErr: "base_url must be an absolute URL",
		},
		{
			name:    "origin",
			mutate:  func(p *payload) { p.Origin = "inference.tinfoil.sh" },
			wantErr: "origin must be an absolute URL",
		},
		{
			name:    "payload attestation bundle",
			mutate:  func(p *payload) { p.AttestationBundleURL = "api.ppq.ai/private" },
			wantErr: "attestation_bundle_url must be an absolute URL",
		},
		{
			name: "policy attestation bundle",
			mutate: func(p *payload) {
				p.Policy = map[string]interface{}{
					"attestation_bundle_url": "api.ppq.ai/private",
				}
			},
			wantErr: "attestation_bundle_url must be an absolute URL",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := valid
			tc.mutate(&p)
			err := validatePayloadBindings(p)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected %q, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestValidatePayloadBindingsRejectsRemoteHostMismatch(t *testing.T) {
	valid := payload{
		PolicyDigest:      "sha256:" + strings.Repeat("a", 64),
		EvidenceDigest:    "sha256:" + strings.Repeat("b", 64),
		VerificationNonce: "nonce",
		BaseURL:           "https://inference.tinfoil.sh/v1",
		Origin:            "https://inference.tinfoil.sh",
	}
	cases := []struct {
		name    string
		mutate  func(*payload)
		wantErr string
	}{
		{
			name:    "origin",
			mutate:  func(p *payload) { p.Origin = "https://other.tinfoil.sh" },
			wantErr: "origin host must match base_url host",
		},
		{
			name:    "payload attestation bundle",
			mutate:  func(p *payload) { p.AttestationBundleURL = "https://other.tinfoil.sh/bundle.json" },
			wantErr: "attestation_bundle_url host must match base_url host",
		},
		{
			name: "policy attestation bundle",
			mutate: func(p *payload) {
				p.Policy = map[string]interface{}{
					"attestation_bundle_url": "https://other.tinfoil.sh/bundle.json",
				}
			},
			wantErr: "attestation_bundle_url host must match base_url host",
		},
		{
			name: "expected enclave host",
			mutate: func(p *payload) {
				p.Policy = map[string]interface{}{
					"expected_enclave_host": "other.tinfoil.sh",
				}
			},
			wantErr: "expected_enclave_host must match base_url host",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := valid
			tc.mutate(&p)
			err := validatePayloadBindings(p)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected %q, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestValidatePayloadBindingsRejectsRemoteOriginMismatch(t *testing.T) {
	valid := payload{
		Mode:                 "ppq-private-tee",
		ModelIDs:             []string{"private/gpt-oss-120b"},
		PolicyDigest:         "sha256:" + strings.Repeat("a", 64),
		EvidenceDigest:       "sha256:" + strings.Repeat("b", 64),
		VerificationNonce:    "nonce",
		BaseURL:              "https://api.ppq.ai:8443/private/v1",
		Origin:               "https://api.ppq.ai:8443",
		AttestationBundleURL: "https://api.ppq.ai/private",
	}

	err := validatePayloadBindings(valid)
	if err == nil || !strings.Contains(
		err.Error(),
		"attestation_bundle_url origin must match base_url origin",
	) {
		t.Fatalf("expected origin mismatch error, got %v", err)
	}
}

func TestValidatePayloadBindingsRejectsRemoteInvalidPort(t *testing.T) {
	valid := payload{
		Mode:              "tinfoil",
		PolicyDigest:      "sha256:" + strings.Repeat("a", 64),
		EvidenceDigest:    "sha256:" + strings.Repeat("b", 64),
		VerificationNonce: "nonce",
		BaseURL:           "https://inference.tinfoil.sh:bad/v1",
		Attestation: map[string]interface{}{
			"format":        "https://tinfoil.sh/predicate/sev-snp-guest/v2",
			"body":          testTinfoilAttestationBody([]byte("report")),
			"report_digest": sha256Bytes([]byte("report")),
		},
	}

	err := validatePayloadBindings(valid)
	if err == nil || !strings.Contains(
		err.Error(),
		"base_url must include a valid port",
	) {
		t.Fatalf("expected invalid port error, got %v", err)
	}
}

func TestValidatePayloadBindingsRejectsPPQPrivateNonPrivateBundleURL(t *testing.T) {
	valid := payload{
		Mode:                 "ppq-private-tee",
		ModelIDs:             []string{"private/gpt-oss-120b"},
		PolicyDigest:         "sha256:" + strings.Repeat("a", 64),
		EvidenceDigest:       "sha256:" + strings.Repeat("b", 64),
		VerificationNonce:    "nonce",
		BaseURL:              "https://api.ppq.ai/private/v1",
		AttestationBundleURL: "https://api.ppq.ai/attestation",
	}

	err := validatePayloadBindings(valid)
	if err == nil || !strings.Contains(
		err.Error(),
		"attestation_bundle_url must include a private path segment",
	) {
		t.Fatalf("expected private attestation bundle path error, got %v", err)
	}
}

func TestValidatePayloadBindingsRejectsPPQPrivateConflictingBundleURLs(t *testing.T) {
	valid := payload{
		Mode:                 "ppq-private-tee",
		ModelIDs:             []string{"private/gpt-oss-120b"},
		PolicyDigest:         "sha256:" + strings.Repeat("a", 64),
		EvidenceDigest:       "sha256:" + strings.Repeat("b", 64),
		VerificationNonce:    "nonce",
		BaseURL:              "https://api.ppq.ai/private/v1",
		AttestationBundleURL: "https://api.ppq.ai/private/attestation-a.json",
		Policy: map[string]interface{}{
			"attestation_bundle_url": "https://api.ppq.ai/private/attestation-b.json",
		},
	}

	err := validatePayloadBindings(valid)
	if err == nil {
		t.Fatal("validatePayloadBindings accepted conflicting PPQ private bundle URLs")
	}
	if !strings.Contains(err.Error(), "attestation_bundle_url must match policy") {
		t.Fatalf("error = %q, want bundle URL mismatch", err)
	}
}

func TestValidatePayloadBindingsRejectsNonStringPolicyIdentity(t *testing.T) {
	valid := payload{
		PolicyDigest:      "sha256:" + strings.Repeat("a", 64),
		EvidenceDigest:    "sha256:" + strings.Repeat("b", 64),
		VerificationNonce: "nonce",
	}
	cases := []struct {
		key     string
		wantErr string
	}{
		{key: "repo", wantErr: "repo must be a string"},
		{key: "expected_repo", wantErr: "expected_repo must be a string"},
		{key: "expected_enclave_host", wantErr: "expected_enclave_host must be a string"},
		{key: "attestation_bundle_url", wantErr: "attestation_bundle_url must be a string"},
	}

	for _, tc := range cases {
		t.Run(tc.key, func(t *testing.T) {
			p := valid
			p.Policy = map[string]interface{}{
				tc.key: json.Number("123"),
			}
			err := validatePayloadBindings(p)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected %q, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestValidatePayloadBindingsRejectsBlankPolicyIdentity(t *testing.T) {
	valid := payload{
		PolicyDigest:      "sha256:" + strings.Repeat("a", 64),
		EvidenceDigest:    "sha256:" + strings.Repeat("b", 64),
		VerificationNonce: "nonce",
	}
	cases := []string{
		"repo",
		"expected_repo",
		"expected_enclave_host",
		"attestation_bundle_url",
	}

	for _, key := range cases {
		t.Run(key, func(t *testing.T) {
			p := valid
			p.Policy = map[string]interface{}{
				key: " ",
			}
			err := validatePayloadBindings(p)
			wantErr := key + " must be a non-empty string"
			if err == nil || !strings.Contains(err.Error(), wantErr) {
				t.Fatalf("expected %q, got %v", wantErr, err)
			}
		})
	}
}

func TestValidatePayloadBindingsRejectsConflictingPolicyIdentityAliases(t *testing.T) {
	valid := payload{
		PolicyDigest:      "sha256:" + strings.Repeat("a", 64),
		EvidenceDigest:    "sha256:" + strings.Repeat("b", 64),
		VerificationNonce: "nonce",
		BaseURL:           "https://inference.tinfoil.sh/v1",
		Policy: map[string]interface{}{
			"repo":                  "tinfoilsh/confidential-model-router",
			"expected_repo":         "attacker/conflicting-router",
			"enclave_host":          "inference.tinfoil.sh",
			"expected_enclave_host": "other.tinfoil.sh",
		},
	}

	err := validatePayloadBindings(valid)
	if err == nil {
		t.Fatal("validatePayloadBindings accepted conflicting policy aliases")
	}
	if !strings.Contains(err.Error(), "repo aliases must match") {
		t.Fatalf("error = %q, want repo aliases", err)
	}
	if !strings.Contains(err.Error(), "enclave host aliases must match") {
		t.Fatalf("error = %q, want enclave host aliases", err)
	}
}

func TestReleaseDigestForPolicyUsesPinnedExpectedDigest(t *testing.T) {
	pinned := strings.Repeat("a1", 32)

	got, pinnedByPolicy, err := releaseDigestForPolicy(map[string]interface{}{
		"expected_release_digest": "sha256:" + pinned,
	})
	if err != nil {
		t.Fatalf("releaseDigestForPolicy returned error: %v", err)
	}
	if !pinnedByPolicy {
		t.Fatal("releaseDigestForPolicy did not report a pinned digest")
	}
	if got != pinned {
		t.Fatalf("release digest = %q, want %q", got, pinned)
	}
}

func TestReleaseDigestForPolicyUsesSingleAllowedDigest(t *testing.T) {
	pinned := strings.Repeat("b2", 32)

	got, pinnedByPolicy, err := releaseDigestForPolicy(map[string]interface{}{
		"allowed_release_digests": []interface{}{pinned},
	})
	if err != nil {
		t.Fatalf("releaseDigestForPolicy returned error: %v", err)
	}
	if !pinnedByPolicy {
		t.Fatal("releaseDigestForPolicy did not report a pinned digest")
	}
	if got != pinned {
		t.Fatalf("release digest = %q, want %q", got, pinned)
	}
}

func TestReleaseDigestsForPolicyUsesAllAllowedDigests(t *testing.T) {
	first := strings.Repeat("c3", 32)
	second := strings.Repeat("d4", 32)

	got, pinnedByPolicy, err := releaseDigestsForPolicy(map[string]interface{}{
		"allowed_release_digests": []interface{}{"sha256:" + first, second},
	})
	if err != nil {
		t.Fatalf("releaseDigestsForPolicy returned error: %v", err)
	}
	if !pinnedByPolicy {
		t.Fatal("releaseDigestsForPolicy did not report pinned digests")
	}
	want := []string{first, second}
	if len(got) != len(want) {
		t.Fatalf("release digests = %#v, want %#v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("release digests = %#v, want %#v", got, want)
		}
	}
}

func TestReleaseDigestsForPolicyRejectsNonStringAllowedDigest(t *testing.T) {
	_, _, err := releaseDigestsForPolicy(map[string]interface{}{
		"allowed_release_digests": []interface{}{json.Number(strings.Repeat("1", 64))},
	})
	if err == nil || !strings.Contains(err.Error(), "policy list entries must be non-empty strings") {
		t.Fatalf("expected non-string release digest policy error, got %v", err)
	}
}

func TestReleaseDigestsForPolicyRejectsMalformedAllowedDigestAlias(t *testing.T) {
	pinned := strings.Repeat("a1", 32)

	_, _, err := releaseDigestsForPolicy(map[string]interface{}{
		"expected_release_digest": "sha256:" + pinned,
		"allowed_release_digests": []interface{}{json.Number(strings.Repeat("1", 64))},
	})
	if err == nil || !strings.Contains(err.Error(), "policy list entries must be non-empty strings") {
		t.Fatalf("expected malformed release digest alias error, got %v", err)
	}
}

func TestReleaseDigestsForPolicyRejectsBlankAllowedDigestAlias(t *testing.T) {
	pinned := strings.Repeat("a1", 32)

	_, _, err := releaseDigestsForPolicy(map[string]interface{}{
		"expected_release_digest": "sha256:" + pinned,
		"allowed_release_digest":  "",
	})
	if err == nil || !strings.Contains(err.Error(), "policy values must be non-empty strings") {
		t.Fatalf("expected blank release digest alias error, got %v", err)
	}
}

func TestReleaseDigestsForPolicyRejectsRepeatedHexPlaceholder(t *testing.T) {
	_, _, err := releaseDigestsForPolicy(map[string]interface{}{
		"expected_release_digest": "sha256:" + strings.Repeat("a", 64),
	})
	if err == nil || !strings.Contains(err.Error(), "placeholder") {
		t.Fatalf("expected placeholder release digest policy error, got %v", err)
	}
}

func TestReleaseDigestsForPolicyRejectsExpectedDigestOutsideAllowedSet(t *testing.T) {
	expected := strings.Repeat("a1", 32)
	allowed := strings.Repeat("b2", 32)

	_, _, err := releaseDigestsForPolicy(map[string]interface{}{
		"expected_release_digest": "sha256:" + expected,
		"allowed_release_digests": []interface{}{allowed},
	})
	if err == nil || !strings.Contains(err.Error(), "release_digest expected value must be allowed") {
		t.Fatalf("expected conflicting release digest policy error, got %v", err)
	}
}

func TestCodeMeasurementFingerprintsForPolicyRejectsMalformedAllowedAlias(t *testing.T) {
	pinned := strings.Repeat("f1", 32)

	_, err := codeMeasurementFingerprintsForPolicy(map[string]interface{}{
		"expected_code_measurement_fingerprint": "sha256:" + pinned,
		"allowed_code_measurement_fingerprints": []interface{}{
			json.Number(strings.Repeat("1", 64)),
		},
	})
	if err == nil || !strings.Contains(err.Error(), "policy list entries must be non-empty strings") {
		t.Fatalf("expected malformed code measurement alias error, got %v", err)
	}
}

func TestCodeMeasurementFingerprintsForPolicyRejectsBlankAllowedAlias(t *testing.T) {
	pinned := strings.Repeat("f1", 32)

	_, err := codeMeasurementFingerprintsForPolicy(map[string]interface{}{
		"expected_code_measurement_fingerprint": "sha256:" + pinned,
		"allowed_code_measurement_fingerprint":  "",
	})
	if err == nil || !strings.Contains(err.Error(), "policy values must be non-empty strings") {
		t.Fatalf("expected blank code measurement alias error, got %v", err)
	}
}

func TestCodeMeasurementFingerprintsForPolicyRejectsRepeatedHexPlaceholder(t *testing.T) {
	_, err := codeMeasurementFingerprintsForPolicy(map[string]interface{}{
		"expected_code_measurement_fingerprint": "sha256:" + strings.Repeat("b", 64),
	})
	if err == nil || !strings.Contains(err.Error(), "placeholder") {
		t.Fatalf("expected placeholder code measurement policy error, got %v", err)
	}
}

func TestCodeMeasurementFingerprintsForPolicyRejectsExpectedDigestOutsideAllowedSet(
	t *testing.T,
) {
	expected := strings.Repeat("f1", 32)
	allowed := strings.Repeat("e2", 32)

	_, err := codeMeasurementFingerprintsForPolicy(map[string]interface{}{
		"expected_code_measurement_fingerprint": "sha256:" + expected,
		"allowed_code_measurement_fingerprints": []interface{}{allowed},
	})
	if err == nil || !strings.Contains(err.Error(), "code_measurement_fingerprint expected value must be allowed") {
		t.Fatalf("expected conflicting code measurement policy error, got %v", err)
	}
}

func TestEnclaveMeasurementFingerprintsForPolicyRejectsMalformedAllowedAlias(
	t *testing.T,
) {
	pinned := strings.Repeat("e1", 32)

	_, err := enclaveMeasurementFingerprintsForPolicy(map[string]interface{}{
		"expected_enclave_measurement_fingerprint": "sha256:" + pinned,
		"allowed_enclave_measurement_fingerprints": []interface{}{
			json.Number(strings.Repeat("1", 64)),
		},
	})
	if err == nil || !strings.Contains(err.Error(), "policy list entries must be non-empty strings") {
		t.Fatalf("expected malformed enclave measurement alias error, got %v", err)
	}
}

func TestEnclaveMeasurementFingerprintsForPolicyRejectsExpectedDigestOutsideAllowedSet(
	t *testing.T,
) {
	expected := strings.Repeat("e1", 32)
	allowed := strings.Repeat("f2", 32)

	_, err := enclaveMeasurementFingerprintsForPolicy(map[string]interface{}{
		"expected_enclave_measurement_fingerprint": "sha256:" + expected,
		"allowed_enclave_measurement_fingerprints": []interface{}{allowed},
	})
	if err == nil || !strings.Contains(err.Error(), "enclave_measurement_fingerprint expected value must be allowed") {
		t.Fatalf("expected conflicting enclave measurement policy error, got %v", err)
	}
}

func TestRequireArtifactIdentityPolicyRejectsMissingPin(t *testing.T) {
	err := requireArtifactIdentityPolicy(map[string]interface{}{})

	if err == nil {
		t.Fatal("requireArtifactIdentityPolicy accepted missing artifact identity pin")
	}
	if !strings.Contains(err.Error(), "expected_release_digest") {
		t.Fatalf("error = %q, want expected_release_digest", err)
	}
}

func TestValidateTinfoilModelAttestationsRequiresConfiguredPayload(t *testing.T) {
	release := strings.Repeat("b7", 32)

	violations := validateTinfoilModelAttestationPayloads(payload{
		Mode: "tinfoil",
		Policy: map[string]interface{}{
			"require_model_attestations": true,
			"model_attestation_targets": map[string]interface{}{
				"tinfoil/kimi-k2-6": map[string]interface{}{
					"host":                    "kimi-k2-6.inf13.tinfoil.sh",
					"repo":                    "tinfoilsh/confidential-kimi-k2-6-b200",
					"expected_release_digest": "sha256:" + release,
				},
			},
		},
	})

	if !containsString(violations, "model_attestations is missing tinfoil/kimi-k2-6") {
		t.Fatalf("violations = %#v, want missing configured model", violations)
	}
}

func TestValidateTinfoilModelAttestationsRejectsSelectedModelTargetMismatch(t *testing.T) {
	release := strings.Repeat("b7", 32)

	violations := validateTinfoilModelAttestationPayloads(payload{
		Mode:     "tinfoil",
		ModelIDs: []string{"tinfoil/selected-model"},
		Policy: map[string]interface{}{
			"require_model_attestations": true,
			"model_attestation_targets": map[string]interface{}{
				"tinfoil/kimi-k2-6": map[string]interface{}{
					"host":                    "kimi-k2-6.inf13.tinfoil.sh",
					"repo":                    "tinfoilsh/confidential-kimi-k2-6-b200",
					"expected_release_digest": "sha256:" + release,
				},
			},
		},
	})

	if !containsString(violations, "model_ids must match model_attestation_targets") {
		t.Fatalf("violations = %#v, want selected model mismatch", violations)
	}
}

func TestValidateTinfoilModelAttestationsRejectsCaseVariantDuplicatePayload(t *testing.T) {
	release := strings.Repeat("b7", 32)

	violations := validateTinfoilModelAttestationPayloads(payload{
		Mode:     "tinfoil",
		ModelIDs: []string{"tinfoil/kimi-k2-6"},
		Policy: map[string]interface{}{
			"require_model_attestations": true,
			"model_attestation_targets": map[string]interface{}{
				"tinfoil/kimi-k2-6": map[string]interface{}{
					"host":                    "kimi-k2-6.inf13.tinfoil.sh",
					"repo":                    "tinfoilsh/confidential-kimi-k2-6-b200",
					"expected_release_digest": "sha256:" + release,
				},
			},
		},
		ModelAttestations: []modelAttestation{
			{ModelID: "tinfoil/Kimi-K2-6"},
			{ModelID: "tinfoil/kimi-k2-6"},
		},
	})

	if !containsString(violations, "duplicate model_attestations entry for tinfoil/kimi-k2-6") {
		t.Fatalf("violations = %#v, want duplicate model_attestations entry", violations)
	}
}

func TestTinfoilModelAttestationTargetPoliciesRejectsMalformedTarget(t *testing.T) {
	_, err := tinfoilModelAttestationTargetPolicies(map[string]interface{}{
		"model_attestation_targets": map[string]interface{}{
			"tinfoil/kimi-k2-6": map[string]interface{}{
				"host": "kimi-k2-6.inf13.tinfoil.sh",
				"repo": "tinfoilsh/confidential-kimi-k2-6-b200",
			},
		},
	})

	if err == nil {
		t.Fatal("accepted model target without release or measurement pin")
	}
	if !strings.Contains(err.Error(), "expected_release_digest") {
		t.Fatalf("error = %q, want expected_release_digest", err)
	}
}

func TestTinfoilModelAttestationTargetPoliciesRejectsDuplicateTrimmedTarget(t *testing.T) {
	release := strings.Repeat("b7", 32)

	_, err := tinfoilModelAttestationTargetPolicies(map[string]interface{}{
		"model_attestation_targets": map[string]interface{}{
			"tinfoil/kimi-k2-6": map[string]interface{}{
				"host":                    "kimi-k2-6.inf13.tinfoil.sh",
				"repo":                    "tinfoilsh/confidential-kimi-k2-6-b200",
				"expected_release_digest": "sha256:" + release,
			},
			" tinfoil/kimi-k2-6 ": map[string]interface{}{
				"host":                    "other.inf13.tinfoil.sh",
				"repo":                    "attacker/conflicting-kimi",
				"expected_release_digest": "sha256:" + release,
			},
		},
	})

	if err == nil {
		t.Fatal("accepted duplicate model target keys after trimming")
	}
	if !strings.Contains(err.Error(), "contains duplicate model target tinfoil/kimi-k2-6") {
		t.Fatalf("error = %q, want duplicate model target", err)
	}
}

func TestTinfoilModelAttestationTargetPoliciesRejectsDuplicateCaseVariantTarget(t *testing.T) {
	release := strings.Repeat("b7", 32)

	_, err := tinfoilModelAttestationTargetPolicies(map[string]interface{}{
		"model_attestation_targets": map[string]interface{}{
			"tinfoil/Kimi-K2-6": map[string]interface{}{
				"host":                    "kimi-k2-6.inf13.tinfoil.sh",
				"repo":                    "tinfoilsh/confidential-kimi-k2-6-b200",
				"expected_release_digest": "sha256:" + release,
			},
			"tinfoil/kimi-k2-6": map[string]interface{}{
				"host":                    "other.inf13.tinfoil.sh",
				"repo":                    "attacker/conflicting-kimi",
				"expected_release_digest": "sha256:" + release,
			},
		},
	})

	if err == nil {
		t.Fatal("accepted case-variant duplicate model target keys")
	}
	if !strings.Contains(err.Error(), "contains duplicate model target tinfoil/kimi-k2-6") {
		t.Fatalf("error = %q, want duplicate model target", err)
	}
}

func TestTinfoilModelAttestationTargetPoliciesRejectsConflictingTargetAliases(t *testing.T) {
	release := strings.Repeat("b7", 32)

	_, err := tinfoilModelAttestationTargetPolicies(map[string]interface{}{
		"model_attestation_targets": map[string]interface{}{
			"tinfoil/kimi-k2-6": map[string]interface{}{
				"host":                    "kimi-k2-6.inf13.tinfoil.sh",
				"repo":                    "tinfoilsh/confidential-kimi-k2-6-b200",
				"expected_release_digest": "sha256:" + release,
			},
		},
		"modelAttestationTargets": map[string]interface{}{
			"tinfoil/kimi-k2-6": map[string]interface{}{
				"host":                    "other.inf13.tinfoil.sh",
				"repo":                    "tinfoilsh/confidential-other",
				"expected_release_digest": "sha256:" + release,
			},
		},
	})

	if err == nil {
		t.Fatal("accepted conflicting model target aliases")
	}
	if !strings.Contains(err.Error(), "model_attestation_targets aliases must match") {
		t.Fatalf("error = %q, want target alias mismatch", err)
	}
}

func TestTinfoilModelAttestationTargetPoliciesRejectsConflictingIdentityAliases(t *testing.T) {
	release := strings.Repeat("b7", 32)

	_, err := tinfoilModelAttestationTargetPolicies(map[string]interface{}{
		"model_attestation_targets": map[string]interface{}{
			"tinfoil/kimi-k2-6": map[string]interface{}{
				"host":                    "kimi-k2-6.inf13.tinfoil.sh",
				"expected_enclave_host":   "other.inf13.tinfoil.sh",
				"repo":                    "tinfoilsh/confidential-kimi-k2-6-b200",
				"expected_repo":           "attacker/conflicting-kimi",
				"expected_release_digest": "sha256:" + release,
			},
		},
	})

	if err == nil {
		t.Fatal("accepted model target with conflicting identity aliases")
	}
	if !strings.Contains(err.Error(), "enclave host aliases must match") {
		t.Fatalf("error = %q, want enclave host aliases", err)
	}
	if !strings.Contains(err.Error(), "repo aliases must match") {
		t.Fatalf("error = %q, want repo aliases", err)
	}
}

func TestTinfoilModelAttestationTargetPoliciesAcceptsReleasePinnedTarget(t *testing.T) {
	release := strings.Repeat("b7", 32)

	targets, err := tinfoilModelAttestationTargetPolicies(map[string]interface{}{
		"model_attestation_targets": map[string]interface{}{
			"tinfoil/kimi-k2-6": map[string]interface{}{
				"host":                    "kimi-k2-6.inf13.tinfoil.sh",
				"repo":                    "tinfoilsh/confidential-kimi-k2-6-b200",
				"expected_release_digest": "sha256:" + release,
			},
		},
	})

	if err != nil {
		t.Fatalf("rejected release-pinned model target: %v", err)
	}
	if targets["tinfoil/kimi-k2-6"]["repo"] != "tinfoilsh/confidential-kimi-k2-6-b200" {
		t.Fatalf("targets = %#v", targets)
	}
}

func TestEnforceArtifactIdentityClaimsRejectsEnclaveMeasurementMismatch(t *testing.T) {
	release := strings.Repeat("e5", 32)
	expected := strings.Repeat("a9", 32)
	actual := strings.Repeat("b8", 32)

	err := enforceArtifactIdentityClaims(
		map[string]interface{}{
			"expected_release_digest":                  release,
			"expected_enclave_measurement_fingerprint": expected,
		},
		map[string]interface{}{
			"release_digest":                  release,
			"enclave_measurement_fingerprint": actual,
		},
	)

	if err == nil {
		t.Fatal("enforceArtifactIdentityClaims accepted mismatched enclave fingerprint")
	}
	if !strings.Contains(err.Error(), "enclave_measurement_fingerprint") {
		t.Fatalf("error = %q, want enclave_measurement_fingerprint", err)
	}
}

func TestRequireArtifactIdentityPolicyAcceptsCodeMeasurementPin(t *testing.T) {
	pinned := strings.Repeat("a9", 32)

	err := requireArtifactIdentityPolicy(map[string]interface{}{
		"expected_code_measurement_fingerprint": "sha256:" + pinned,
	})

	if err != nil {
		t.Fatalf("requireArtifactIdentityPolicy rejected code measurement pin: %v", err)
	}
}

func TestRequireArtifactIdentityPolicyRejectsMalformedCodeMeasurementPin(
	t *testing.T,
) {
	err := requireArtifactIdentityPolicy(map[string]interface{}{
		"expected_code_measurement_fingerprint": "fixture-code-fp",
	})

	if err == nil {
		t.Fatal("requireArtifactIdentityPolicy accepted malformed code measurement pin")
	}
	if !strings.Contains(err.Error(), "measurement fingerprint") {
		t.Fatalf("error = %q, want measurement fingerprint", err)
	}
}

func TestRequireArtifactIdentityPolicyRejectsPrefixedNativeSEVSNPMeasurementPin(
	t *testing.T,
) {
	err := requireArtifactIdentityPolicy(map[string]interface{}{
		"expected_code_measurement_fingerprint": "sha256:" + strings.Repeat("b1", 48),
	})

	if err == nil {
		t.Fatal("requireArtifactIdentityPolicy accepted prefixed native SEV-SNP measurement pin")
	}
	if !strings.Contains(err.Error(), "measurement fingerprint") {
		t.Fatalf("error = %q, want measurement fingerprint", err)
	}
}

func TestVerifyCodeMeasurementRejectsCodeMeasurementOnlyPolicyBeforeLatestLookup(
	t *testing.T,
) {
	pinned := strings.Repeat("a9", 32)

	_, _, _, err := verifyCodeMeasurement(
		"tinfoilsh/confidential-model-router",
		&attestation.Verification{
			Measurement: &attestation.Measurement{},
		},
		map[string]interface{}{
			"expected_code_measurement_fingerprint": "sha256:" + pinned,
		},
	)

	if err == nil {
		t.Fatal("verifyCodeMeasurement accepted code measurement without release digest")
	}
	if !strings.Contains(err.Error(), "expected_release_digest") {
		t.Fatalf("error = %q, want expected_release_digest", err)
	}
}

func TestEnforceArtifactIdentityClaimsRejectsCodeMeasurementMismatch(t *testing.T) {
	expected := strings.Repeat("a9", 32)
	actual := strings.Repeat("b8", 32)

	err := enforceArtifactIdentityClaims(
		map[string]interface{}{
			"expected_code_measurement_fingerprint": expected,
		},
		map[string]interface{}{
			"code_measurement_fingerprint": actual,
		},
	)

	if err == nil {
		t.Fatal("enforceArtifactIdentityClaims accepted mismatched code fingerprint")
	}
	if !strings.Contains(err.Error(), "code_measurement_fingerprint") {
		t.Fatalf("error = %q, want code_measurement_fingerprint", err)
	}
}

func TestEnforceArtifactIdentityClaimsAcceptsPrefixedCodeMeasurementPin(
	t *testing.T,
) {
	pinned := strings.Repeat("a9", 32)

	err := enforceArtifactIdentityClaims(
		map[string]interface{}{
			"expected_code_measurement_fingerprint": "sha256:" + pinned,
		},
		map[string]interface{}{
			"code_measurement_fingerprint": pinned,
		},
	)

	if err != nil {
		t.Fatalf("enforceArtifactIdentityClaims rejected code fingerprint: %v", err)
	}
}

func TestEnforceArtifactIdentityClaimsAcceptsAllowedReleaseDigest(t *testing.T) {
	pinned := strings.Repeat("e5", 32)

	err := enforceArtifactIdentityClaims(
		map[string]interface{}{
			"allowed_release_digests": []interface{}{"sha256:" + pinned},
		},
		map[string]interface{}{
			"release_digest": pinned,
		},
	)

	if err != nil {
		t.Fatalf("enforceArtifactIdentityClaims rejected release digest: %v", err)
	}
}
