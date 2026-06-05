//go:build contrast_unstable_api

package main

import (
	"bytes"
	"context"
	"crypto/ecdsa"
	"crypto/hpke"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/edgelesssys/continuum/internal/oss/crypto"
	"github.com/edgelesssys/continuum/internal/oss/httpapi"
	"github.com/edgelesssys/continuum/internal/oss/ocspheader"
	"github.com/edgelesssys/continuum/internal/oss/secretexchange"
	contrastsdk "github.com/edgelesssys/contrast/sdk"
)

const (
	schemaVersion   = "routstr-confidential-verifier-v1"
	defaultAPIBase  = "https://api.privatemode.ai"
	defaultCDNBase  = "https://cdn.confidential.cloud/privatemode/v2"
	defaultAPIKeyEn = "PRIVATEMODE_API_KEY"
)

var privatemodeVerificationStepNames = []string{
	"contrast_manifest",
	"coordinator_attestation",
	"mesh_ca_binding",
	"secret_service_tls",
	"ai_worker_attestation",
	"gpu_attestation",
	"key_release_binding",
	"prompt_encryption",
	"nvidia_ocsp_revocation",
}

type verifierPayload struct {
	SchemaVersion     string                 `json:"schema_version"`
	ProviderType      string                 `json:"provider_type"`
	Mode              string                 `json:"mode"`
	ProxyBaseURL      string                 `json:"proxy_base_url"`
	ModelIDs          []string               `json:"model_ids"`
	PolicyDigest      string                 `json:"policy_digest"`
	Policy            map[string]interface{} `json:"policy"`
	EvidenceDigest    string                 `json:"evidence_digest"`
	VerificationNonce string                 `json:"verification_nonce"`
	VerifierDigest    string                 `json:"verifier_command_digest"`
	VerifierArtifact  string                 `json:"verifier_artifact_path"`
}

type verifierPolicy struct {
	APIBaseURL                        string
	CDNBaseURL                        string
	StringFieldViolations             []string
	StringListFieldViolations         []string
	ModelWorkloadBindingViolations    []string
	APIKeyEnv                         string
	APIKeyFile                        string
	InlineAPIKey                      string
	ManifestPath                      string
	ManifestB64                       string
	ManifestURL                       string
	ManifestDigest                    string
	ManifestLogDir                    string
	ProxyImageDigest                  string
	ProxyBinaryPath                   string
	ProxyBinaryDigest                 string
	DumpRequests                      *bool
	SharedPromptCache                 *bool
	NvidiaOCSPAllowUnknown            *bool
	NvidiaOCSPRevokedGracePeriodHours *int
	ExpectedWorkloadSANs              []string
	ExpectedWorkloadIDs               []string
	ModelWorkloadBindings             map[string]modelWorkloadBinding
	TimeoutSeconds                    float64
}

type verifiedPrivatemodeEvidence struct {
	Manifest                   []byte
	CoordinatorAttestationDoc  []byte
	MeshCAPEM                  []byte
	SecretServiceCertificate   []byte
	InferenceSecretID          string
	InferenceSecret            [32]byte
	PromptEncryptionCiphertext string
	OCSPPolicyHeader           string
	OCSPPolicyMAC              string
}

type verifierResult struct {
	Verified          bool                   `json:"verified"`
	Verifier          string                 `json:"verifier"`
	PolicyDigest      string                 `json:"policy_digest"`
	EvidenceDigest    string                 `json:"evidence_digest"`
	VerificationNonce string                 `json:"verification_nonce"`
	VerifiedAt        int64                  `json:"verified_at"`
	Claims            map[string]interface{} `json:"claims"`
}

type privatemodeManifest struct {
	Policies map[string]privatemodeManifestPolicy `json:"Policies"`
}

type privatemodeManifestPolicy struct {
	Role             string   `json:"Role,omitempty"`
	SANs             []string `json:"SANs,omitempty"`
	WorkloadSecretID string   `json:"WorkloadSecretID,omitempty"`
}

type privatemodeWorkloadIdentity struct {
	PolicyDigest     string   `json:"policy_digest"`
	Role             string   `json:"role,omitempty"`
	SANs             []string `json:"sans,omitempty"`
	WorkloadID       string   `json:"workload_id,omitempty"`
	WorkloadSANs     []string `json:"workload_sans,omitempty"`
	WorkloadSecretID string   `json:"workload_secret_id,omitempty"`
}

type modelWorkloadBinding struct {
	WorkloadIDs  []string
	WorkloadSANs []string
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	var payload verifierPayload
	decoder := json.NewDecoder(os.Stdin)
	decoder.UseNumber()
	if err := decodeSingleJSON(decoder, &payload, "verifier payload"); err != nil {
		return err
	}
	if payload.SchemaVersion != schemaVersion {
		return fmt.Errorf("unexpected schema_version %q", payload.SchemaVersion)
	}
	if payload.Mode != "privatemode" {
		return fmt.Errorf("unsupported verifier mode %q", payload.Mode)
	}
	if err := validatePayloadBindings(payload); err != nil {
		return err
	}

	policy := parsePolicy(payload.Policy)
	if err := validatePolicy(policy); err != nil {
		return err
	}
	if err := validatePayloadModelIDs(payload.ModelIDs, policy); err != nil {
		return err
	}
	apiKey, err := apiKeyFromPolicy(policy)
	if err != nil {
		return err
	}

	timeout := policy.timeout()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	httpClient := &http.Client{Timeout: timeout}
	manifest, err := loadExpectedManifest(ctx, policy, httpClient)
	if err != nil {
		return err
	}
	evidence, err := verifyPrivatemodeDeployment(ctx, policy, apiKey, manifest, httpClient)
	if err != nil {
		return err
	}
	claims, err := buildClaims(policy, evidence, payload.ProxyBaseURL)
	if err != nil {
		return err
	}

	result := verifierResultForPayload(payload, claims, time.Now().UTC())
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(result)
}

func decodeSingleJSON(decoder *json.Decoder, target interface{}, label string) error {
	if err := decoder.Decode(target); err != nil {
		return fmt.Errorf("decoding %s: %w", label, err)
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("%s must contain exactly one JSON object", label)
		}
		return fmt.Errorf("decoding trailing %s: %w", label, err)
	}
	return nil
}

func parsePolicy(raw map[string]interface{}) verifierPolicy {
	modelWorkloadBindings, modelWorkloadBindingViolations := modelWorkloadBindingsValue(raw)
	return verifierPolicy{
		APIBaseURL:                        stringValue(raw, "api_base_url", "apiBaseURL"),
		CDNBaseURL:                        stringValue(raw, "cdn_base_url", "cdnBaseURL"),
		StringFieldViolations:             policyStringViolations(raw),
		StringListFieldViolations:         policyStringListViolations(raw),
		ModelWorkloadBindingViolations:    modelWorkloadBindingViolations,
		APIKeyEnv:                         stringValue(raw, "api_key_env", "apiKeyEnv"),
		APIKeyFile:                        stringValue(raw, "api_key_file", "apiKeyFile"),
		InlineAPIKey:                      stringValue(raw, "api_key", "apiKey"),
		ManifestPath:                      stringValue(raw, "manifest_path", "manifestPath"),
		ManifestB64:                       stringValue(raw, "manifest_b64", "manifestB64"),
		ManifestURL:                       stringValue(raw, "manifest_url", "manifestURL"),
		ManifestDigest:                    stringValue(raw, "manifest_digest", "manifestDigest"),
		ManifestLogDir:                    stringValue(raw, "manifest_log_dir", "manifestLogDir"),
		ProxyImageDigest:                  stringValue(raw, "proxy_image_digest", "proxyImageDigest"),
		ProxyBinaryPath:                   stringValue(raw, "proxy_binary_path", "proxyBinaryPath"),
		ProxyBinaryDigest:                 stringValue(raw, "proxy_binary_digest", "proxyBinaryDigest"),
		DumpRequests:                      boolValue(raw, "dump_requests", "dumpRequests"),
		SharedPromptCache:                 boolValue(raw, "shared_prompt_cache", "sharedPromptCache"),
		NvidiaOCSPAllowUnknown:            boolValue(raw, "nvidia_ocsp_allow_unknown", "nvidiaOCSPAllowUnknown"),
		NvidiaOCSPRevokedGracePeriodHours: intValue(raw, "nvidia_ocsp_revoked_grace_period_hours", "nvidiaOCSPRevokedGracePeriod"),
		ExpectedWorkloadSANs:              stringListValue(raw, "expected_workload_sans", "expectedWorkloadSANs"),
		ExpectedWorkloadIDs:               stringListValue(raw, "expected_workload_ids", "expectedWorkloadIDs"),
		ModelWorkloadBindings:             modelWorkloadBindings,
		TimeoutSeconds:                    floatValue(raw, "timeout_seconds", "timeoutSeconds"),
	}
}

func validatePolicy(p verifierPolicy) error {
	var violations []string
	violations = append(violations, p.StringFieldViolations...)
	violations = append(violations, p.StringListFieldViolations...)
	violations = append(violations, p.ModelWorkloadBindingViolations...)
	if strings.TrimSpace(p.InlineAPIKey) != "" {
		violations = append(violations, "api_key must not be embedded in verifier policy")
	}
	if len(p.ExpectedWorkloadSANs) == 0 && len(p.ExpectedWorkloadIDs) == 0 {
		violations = append(violations, "expected_workload_sans or expected_workload_ids is required")
	}
	violations = append(violations, modelWorkloadBindingPolicyViolations(p)...)
	if p.DumpRequests == nil || *p.DumpRequests {
		violations = append(violations, "dump_requests must be explicitly false")
	}
	if p.SharedPromptCache == nil || *p.SharedPromptCache {
		violations = append(violations, "shared_prompt_cache must be explicitly false")
	}
	if p.NvidiaOCSPAllowUnknown == nil || *p.NvidiaOCSPAllowUnknown {
		violations = append(violations, "nvidia_ocsp_allow_unknown must be explicitly false")
	}
	if p.NvidiaOCSPRevokedGracePeriodHours == nil || *p.NvidiaOCSPRevokedGracePeriodHours != 0 {
		violations = append(violations, "nvidia_ocsp_revoked_grace_period_hours must be 0")
	}
	if p.ManifestDigest == "" && p.ManifestLogDir == "" {
		violations = append(violations, "manifest_digest or manifest_log_dir is required")
	}
	for _, check := range []struct {
		label string
		value string
	}{
		{"manifest_digest", p.ManifestDigest},
		{"proxy_image_digest", p.ProxyImageDigest},
		{"proxy_binary_digest", p.ProxyBinaryDigest},
	} {
		if check.value != "" && !isPrefixedSHA256Digest(check.value) {
			violations = append(violations, check.label+" must be a sha256 digest")
		}
	}
	if p.ProxyBinaryDigest == "" {
		violations = append(
			violations,
			"proxy_binary_digest is required for full Privatemode proxy artifact verification",
		)
	}
	if p.ProxyBinaryDigest != "" && p.ProxyBinaryPath == "" {
		violations = append(violations, "proxy_binary_path is required when proxy_binary_digest is set")
	}
	for _, check := range []struct {
		label string
		value string
	}{
		{"api_base_url", p.APIBaseURL},
		{"cdn_base_url", p.CDNBaseURL},
		{"manifest_url", p.ManifestURL},
	} {
		if violation := remoteURLViolation(check.value, check.label); violation != "" {
			violations = append(violations, violation)
		}
	}
	if len(violations) > 0 {
		return errors.New(strings.Join(violations, "; "))
	}
	return nil
}

func validatePayloadBindings(payload verifierPayload) error {
	var violations []string
	for _, check := range []struct {
		label string
		value string
	}{
		{"policy_digest", payload.PolicyDigest},
		{"evidence_digest", payload.EvidenceDigest},
	} {
		if !isPrefixedSHA256Digest(check.value) {
			violations = append(
				violations,
				check.label+" must be a full sha256 digest",
			)
		}
	}
	if strings.TrimSpace(payload.VerificationNonce) == "" {
		violations = append(violations, "verification_nonce is required")
	}
	if payload.VerifierDigest != "" && !isPrefixedSHA256Digest(payload.VerifierDigest) {
		violations = append(
			violations,
			"verifier_command_digest must be a full sha256 digest",
		)
	}
	if proxyViolation := proxyBaseURLViolation(payload.ProxyBaseURL); proxyViolation != "" {
		violations = append(violations, proxyViolation)
	}
	if len(violations) > 0 {
		return errors.New(strings.Join(violations, "; "))
	}
	return nil
}

func validatePayloadModelIDs(modelIDs []string, policy verifierPolicy) error {
	normalized, violations := normalizedPayloadModelIDs(modelIDs)
	if len(violations) > 0 {
		return errors.New(strings.Join(violations, "; "))
	}
	boundModelIDs := modelWorkloadBindingModelIDs(policy)
	if len(boundModelIDs) > 0 && len(normalized) == 0 {
		return errors.New("model_ids are required with model_workload_bindings")
	}
	if len(normalized) != len(boundModelIDs) {
		return errors.New("model_ids must match model_workload_bindings")
	}
	for index := range normalized {
		if normalized[index] != boundModelIDs[index] {
			return errors.New("model_ids must match model_workload_bindings")
		}
	}
	return nil
}

func normalizedPayloadModelIDs(modelIDs []string) ([]string, []string) {
	normalized := make([]string, 0, len(modelIDs))
	seen := map[string]bool{}
	var violations []string
	for _, modelID := range modelIDs {
		modelID = strings.TrimSpace(modelID)
		if modelID == "" {
			violations = append(violations, "model_ids must contain only non-empty strings")
			continue
		}
		if !strings.HasPrefix(modelID, "privatemode/") {
			violations = append(violations, "model_ids must start with privatemode/")
			continue
		}
		if seen[modelID] {
			violations = append(violations, "model_ids must not contain duplicates")
			continue
		}
		seen[modelID] = true
		normalized = append(normalized, modelID)
	}
	sort.Strings(normalized)
	return normalized, violations
}

func proxyBaseURLViolation(value string) string {
	rawURL := strings.TrimSpace(value)
	if rawURL == "" {
		return "proxy_base_url is required"
	}
	parsed, err := url.Parse(rawURL)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return "proxy_base_url must be an absolute URL"
	}
	if parsed.User != nil || strings.Contains(parsed.Host, "@") {
		return "proxy_base_url must not contain userinfo"
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "proxy_base_url must use http or https"
	}
	if parsed.Hostname() == "" {
		return "proxy_base_url must include a host"
	}
	if !isLoopbackHostname(parsed.Hostname()) {
		return "proxy_base_url must use a loopback host"
	}
	return ""
}

func isLoopbackHostname(hostname string) bool {
	normalized := strings.TrimSuffix(strings.ToLower(strings.TrimSpace(hostname)), ".")
	if normalized == "localhost" {
		return true
	}
	ip := net.ParseIP(normalized)
	return ip != nil && ip.IsLoopback()
}

func isPrefixedSHA256Digest(value string) bool {
	digest := strings.TrimSpace(strings.ToLower(value))
	if !strings.HasPrefix(digest, "sha256:") {
		return false
	}
	hexDigest := strings.TrimPrefix(digest, "sha256:")
	if len(hexDigest) != 64 {
		return false
	}
	if isRepeatedHexPlaceholder(hexDigest) {
		return false
	}
	for _, char := range hexDigest {
		if !strings.ContainsRune("0123456789abcdef", char) {
			return false
		}
	}
	return true
}

func isRepeatedHexPlaceholder(value string) bool {
	digest := strings.TrimSpace(strings.ToLower(value))
	if strings.HasPrefix(digest, "sha256:") {
		digest = strings.TrimPrefix(digest, "sha256:")
	}
	if len(digest) != 64 {
		return false
	}
	for _, char := range digest {
		if !strings.ContainsRune("0123456789abcdef", char) {
			return false
		}
		if char != rune(digest[0]) {
			return false
		}
	}
	return true
}

func remoteURLViolation(value string, label string) string {
	rawURL := strings.TrimSpace(value)
	if rawURL == "" {
		return ""
	}
	parsed, err := url.Parse(rawURL)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return label + " must be an absolute URL"
	}
	if parsed.User != nil || strings.Contains(parsed.Host, "@") {
		return label + " must not contain userinfo"
	}
	if parsed.Scheme != "https" {
		return label + " must use https"
	}
	return ""
}

func loadExpectedManifest(ctx context.Context, p verifierPolicy, client *http.Client) ([]byte, error) {
	var manifest []byte
	var err error
	switch {
	case p.ManifestPath != "":
		manifest, err = os.ReadFile(p.ManifestPath)
		if err != nil {
			return nil, fmt.Errorf("reading manifest_path: %w", err)
		}
	case p.ManifestB64 != "":
		manifest, err = base64.StdEncoding.DecodeString(p.ManifestB64)
		if err != nil {
			return nil, fmt.Errorf("decoding manifest_b64: %w", err)
		}
	case p.ManifestURL != "":
		manifest, err = fetchURL(ctx, client, p.ManifestURL)
		if err != nil {
			return nil, fmt.Errorf("fetching manifest_url: %w", err)
		}
	default:
		cdnBase := p.CDNBaseURL
		if cdnBase == "" {
			cdnBase = defaultCDNBase
		}
		manifestURL := strings.TrimRight(cdnBase, "/") + "/manifest.json?t=" + fmt.Sprint(time.Now().UnixMilli())
		manifest, err = fetchURL(ctx, client, manifestURL)
		if err != nil {
			return nil, fmt.Errorf("fetching Privatemode manifest: %w", err)
		}
	}

	if len(manifest) == 0 {
		return nil, errors.New("manifest is empty")
	}
	if p.ManifestDigest != "" && sha256BytesDigest(manifest) != p.ManifestDigest {
		return nil, fmt.Errorf(
			"manifest digest mismatch: expected %s, got %s",
			p.ManifestDigest,
			sha256BytesDigest(manifest),
		)
	}
	return manifest, nil
}

func verifyPrivatemodeDeployment(
	ctx context.Context,
	p verifierPolicy,
	apiKey string,
	manifest []byte,
	client *http.Client,
) (verifiedPrivatemodeEvidence, error) {
	if client == nil {
		client = http.DefaultClient
	}
	apiHost, err := apiHost(p)
	if err != nil {
		return verifiedPrivatemodeEvidence{}, err
	}

	nonce := make([]byte, 32)
	if _, err := rand.Read(nonce); err != nil {
		return verifiedPrivatemodeEvidence{}, fmt.Errorf("generating attestation nonce: %w", err)
	}
	reqBody, err := json.Marshal(httpapi.AttestReq{Nonce: nonce})
	if err != nil {
		return verifiedPrivatemodeEvidence{}, fmt.Errorf("marshaling attest request: %w", err)
	}
	attestRespBody, err := httpapi.Do(
		ctx,
		client,
		http.MethodPost,
		"https://"+apiHost+"/privatemode/v1/attest",
		reqBody,
		apiKey,
	)
	if err != nil {
		return verifiedPrivatemodeEvidence{}, fmt.Errorf("doing Coordinator attest request: %w", err)
	}
	var attestResp httpapi.AttestResp
	if err := json.Unmarshal(attestRespBody, &attestResp); err != nil {
		return verifiedPrivatemodeEvidence{}, fmt.Errorf("unmarshaling attest response: %w", err)
	}
	if len(attestResp.AttestationDoc) == 0 {
		return verifiedPrivatemodeEvidence{}, errors.New("Coordinator attestation document is empty")
	}

	contrastClient := contrastsdk.New()
	state, err := contrastClient.ValidateAttestation(ctx, nonce, attestResp.AttestationDoc)
	if err != nil {
		return verifiedPrivatemodeEvidence{}, fmt.Errorf("validating Coordinator attestation: %w", err)
	}
	if len(state.Manifests) != 1 {
		return verifiedPrivatemodeEvidence{}, fmt.Errorf("expected exactly one active manifest, got %d", len(state.Manifests))
	}
	if !bytes.Equal(state.Manifests[0], manifest) {
		return verifiedPrivatemodeEvidence{}, errors.New("active manifest does not match expected manifest")
	}

	meshCACert, err := parseMeshCACertificate(state.MeshCA)
	if err != nil {
		return verifiedPrivatemodeEvidence{}, err
	}
	secretID, secret, secretServiceCert, err := exchangeSecret(ctx, client, apiHost, meshCACert, apiKey)
	if err != nil {
		return verifiedPrivatemodeEvidence{}, err
	}

	cipher, err := crypto.NewRequestCipher(secret[:], secretID)
	if err != nil {
		return verifiedPrivatemodeEvidence{}, fmt.Errorf("creating prompt cipher: %w", err)
	}
	ciphertext, err := cipher.Encrypt("routstr-privatemode-verifier")
	if err != nil {
		return verifiedPrivatemodeEvidence{}, fmt.Errorf("encrypting prompt canary: %w", err)
	}

	ocspHeader := ocspheader.NewHeader(
		[]ocspheader.AllowStatus{ocspheader.AllowStatusGood},
		time.Now().UTC(),
	)
	ocspPolicy, err := ocspHeader.Marshal()
	if err != nil {
		return verifiedPrivatemodeEvidence{}, fmt.Errorf("marshaling NVIDIA OCSP policy: %w", err)
	}
	ocspMAC, err := ocspHeader.MarshalMACHeader(secret)
	if err != nil {
		return verifiedPrivatemodeEvidence{}, fmt.Errorf("signing NVIDIA OCSP policy: %w", err)
	}

	return verifiedPrivatemodeEvidence{
		Manifest:                   manifest,
		CoordinatorAttestationDoc:  attestResp.AttestationDoc,
		MeshCAPEM:                  state.MeshCA,
		SecretServiceCertificate:   secretServiceCert,
		InferenceSecretID:          secretID,
		InferenceSecret:            secret,
		PromptEncryptionCiphertext: ciphertext,
		OCSPPolicyHeader:           ocspPolicy,
		OCSPPolicyMAC:              ocspMAC,
	}, nil
}

func exchangeSecret(
	ctx context.Context,
	client *http.Client,
	apiHost string,
	meshCA *x509.Certificate,
	apiKey string,
) (string, [32]byte, []byte, error) {
	var secret [32]byte
	priv, err := hpke.MLKEM768X25519().GenerateKey()
	if err != nil {
		return "", secret, nil, fmt.Errorf("generating secret-exchange key: %w", err)
	}
	secretReq := httpapi.SecretReq{PublicKey: priv.PublicKey().Bytes()}
	reqBody, err := json.Marshal(secretReq)
	if err != nil {
		return "", secret, nil, fmt.Errorf("marshaling secret request: %w", err)
	}
	body, err := httpapi.Do(
		ctx,
		client,
		http.MethodPost,
		"https://"+apiHost+"/privatemode/v1/secret",
		reqBody,
		apiKey,
	)
	if err != nil {
		return "", secret, nil, fmt.Errorf("doing secret request: %w", err)
	}
	var resp httpapi.SecretResp
	if err := json.Unmarshal(body, &resp); err != nil {
		return "", secret, nil, fmt.Errorf("unmarshaling secret response: %w", err)
	}
	secretServicePublicKey, err := verifiedSecretServicePublicKey(resp.MeshCert, meshCA)
	if err != nil {
		return "", secret, nil, err
	}
	if !ecdsa.VerifyASN1(
		secretServicePublicKey,
		secretexchange.Hash(secretReq.PublicKey, resp.EncapsulatedKey),
		resp.Signature,
	) {
		return "", secret, nil, errors.New("secret service signature verification failed")
	}

	recipient, err := hpke.NewRecipient(resp.EncapsulatedKey, priv, hpke.HKDFSHA256(), hpke.ExportOnly(), nil)
	if err != nil {
		return "", secret, nil, fmt.Errorf("creating HPKE recipient: %w", err)
	}
	secretBytes, err := recipient.Export("", 32)
	if err != nil {
		return "", secret, nil, fmt.Errorf("exporting inference secret: %w", err)
	}
	if len(secretBytes) != len(secret) {
		return "", secret, nil, fmt.Errorf("unexpected inference secret length: %d", len(secretBytes))
	}
	copy(secret[:], secretBytes)
	return secretexchange.ID(secretReq.PublicKey), secret, resp.MeshCert, nil
}

func buildClaims(p verifierPolicy, evidence verifiedPrivatemodeEvidence, proxyBaseURL string) (map[string]interface{}, error) {
	manifestDigest := sha256BytesDigest(evidence.Manifest)
	if p.ManifestDigest != "" && manifestDigest != p.ManifestDigest {
		return nil, fmt.Errorf("manifest digest mismatch: expected %s, got %s", p.ManifestDigest, manifestDigest)
	}

	workloads, err := attestedWorkloadIdentities(evidence.Manifest)
	if err != nil {
		return nil, err
	}
	if err := requireExpectedWorkloads(p, workloads); err != nil {
		return nil, err
	}
	workloadIdentityDigest := sha256JSONDigest(workloads)
	workloadPolicyDigest := sha256JSONDigest(attestedWorkloadPolicyEvidence(workloads))
	expectedWorkloadDigest := expectedWorkloadIdentityDigest(p)
	bindingDigest := modelWorkloadBindingDigest(p)

	claims := map[string]interface{}{
		"manifest_digest":                     manifestDigest,
		"transport":                           "privatemode-proxy",
		"trust_tier":                          "app-e2ee",
		"proxy_base_url":                      strings.TrimSpace(proxyBaseURL),
		"coordinator_measurement":             sha256BytesDigest(evidence.CoordinatorAttestationDoc),
		"coordinator_attestation_doc_digest":  sha256BytesDigest(evidence.CoordinatorAttestationDoc),
		"mesh_ca_digest":                      sha256BytesDigest(evidence.MeshCAPEM),
		"secret_service_measurement":          sha256BytesDigest(evidence.SecretServiceCertificate),
		"secret_service_certificate_digest":   sha256BytesDigest(evidence.SecretServiceCertificate),
		"ai_worker_measurement":               workloadPolicyDigest,
		"ai_worker_manifest_digest":           manifestDigest,
		"attested_workload_identity_digest":   workloadIdentityDigest,
		"attested_workload_policy_digest":     workloadPolicyDigest,
		"expected_workload_identity_digest":   expectedWorkloadDigest,
		"model_workload_binding_digest":       bindingDigest,
		"selected_model_ids":                  modelWorkloadBindingModelIDs(p),
		"gpu_attestation_policy":              "nvidia-ocsp-good-only",
		"nvidia_ocsp_policy_header_digest":    sha256StringDigest(evidence.OCSPPolicyHeader),
		"nvidia_ocsp_policy_mac_digest":       sha256StringDigest(evidence.OCSPPolicyMAC),
		"prompt_encryption_ciphertext_digest": sha256StringDigest(evidence.PromptEncryptionCiphertext),
		"inference_secret_id_digest":          sha256StringDigest(evidence.InferenceSecretID),
		"verification_steps":                  privatemodeVerificationSteps(),
	}

	if p.ManifestLogDir != "" {
		digest, err := sha256DirDigest(p.ManifestLogDir)
		if err != nil {
			return nil, fmt.Errorf("hashing manifest_log_dir: %w", err)
		}
		logEntry, err := manifestLogEntryForManifest(p.ManifestLogDir, evidence.Manifest)
		if err != nil {
			return nil, err
		}
		claims["manifest_log_digest"] = digest
		claims["manifest_log_manifest_digest"] = manifestDigest
		claims["manifest_log_entry"] = logEntry
	}
	if p.ProxyImageDigest != "" {
		claims["proxy_image_digest"] = p.ProxyImageDigest
	}
	if p.ProxyBinaryDigest != "" {
		actualDigest, err := sha256FileDigest(p.ProxyBinaryPath)
		if err != nil {
			return nil, fmt.Errorf("hashing proxy_binary_path: %w", err)
		}
		if actualDigest != p.ProxyBinaryDigest {
			return nil, fmt.Errorf("proxy binary digest mismatch: expected %s, got %s", p.ProxyBinaryDigest, actualDigest)
		}
		claims["proxy_binary_digest"] = actualDigest
	}

	claims["key_release_binding"] = sha256JSONDigest(map[string]interface{}{
		"attested_workload_policy_digest":   claims["attested_workload_policy_digest"],
		"expected_workload_identity_digest": claims["expected_workload_identity_digest"],
		"model_workload_binding_digest":     claims["model_workload_binding_digest"],
		"manifest_digest":                   claims["manifest_digest"],
		"mesh_ca_digest":                    claims["mesh_ca_digest"],
		"secret_service_certificate_digest": claims["secret_service_certificate_digest"],
		"inference_secret_id_digest":        claims["inference_secret_id_digest"],
		"nvidia_ocsp_policy_mac_digest":     claims["nvidia_ocsp_policy_mac_digest"],
	})
	return claims, nil
}

func attestedWorkloadIdentities(manifest []byte) ([]privatemodeWorkloadIdentity, error) {
	decoder := json.NewDecoder(bytes.NewReader(manifest))
	decoder.UseNumber()
	var parsed privatemodeManifest
	if err := decodeSingleJSON(decoder, &parsed, "Privatemode manifest"); err != nil {
		return nil, err
	}
	if parsed.Policies == nil {
		return nil, errors.New("Privatemode manifest Policies object is required")
	}

	workloads := make([]privatemodeWorkloadIdentity, 0, len(parsed.Policies))
	for policyDigest, policy := range parsed.Policies {
		sans := canonicalStringList(policy.SANs)
		workloadSANs := workloadSANs(sans)
		workloadSecretID := strings.TrimSpace(policy.WorkloadSecretID)
		workloadID := workloadIDFromSecretID(workloadSecretID)
		if len(workloadSANs) == 0 && !strings.HasPrefix(workloadID, "workload-") {
			continue
		}
		workloads = append(workloads, privatemodeWorkloadIdentity{
			PolicyDigest:     strings.TrimSpace(policyDigest),
			Role:             strings.TrimSpace(policy.Role),
			SANs:             sans,
			WorkloadID:       workloadID,
			WorkloadSANs:     workloadSANs,
			WorkloadSecretID: workloadSecretID,
		})
	}
	sort.Slice(workloads, func(i, j int) bool {
		left := workloads[i]
		right := workloads[j]
		return strings.Join([]string{
			left.WorkloadID,
			left.WorkloadSecretID,
			left.PolicyDigest,
		}, "\x00") < strings.Join([]string{
			right.WorkloadID,
			right.WorkloadSecretID,
			right.PolicyDigest,
		}, "\x00")
	})
	return workloads, nil
}

func requireExpectedWorkloads(p verifierPolicy, workloads []privatemodeWorkloadIdentity) error {
	if len(p.ExpectedWorkloadSANs) == 0 && len(p.ExpectedWorkloadIDs) == 0 {
		return errors.New("expected_workload_sans or expected_workload_ids is required")
	}

	allSANs := map[string]bool{}
	allIDs := map[string]bool{}
	for _, workload := range workloads {
		for _, san := range workload.WorkloadSANs {
			allSANs[san] = true
		}
		if workload.WorkloadID != "" {
			allIDs[workload.WorkloadID] = true
		}
		if workload.WorkloadSecretID != "" {
			allIDs[workload.WorkloadSecretID] = true
		}
	}
	for _, expectedSAN := range p.ExpectedWorkloadSANs {
		if !allSANs[expectedSAN] {
			return fmt.Errorf("expected workload SAN %q not present in attested manifest", expectedSAN)
		}
	}
	for _, expectedID := range p.ExpectedWorkloadIDs {
		if !allIDs[expectedID] {
			return fmt.Errorf("expected workload ID %q not present in attested manifest", expectedID)
		}
	}
	return nil
}

func attestedWorkloadPolicyEvidence(workloads []privatemodeWorkloadIdentity) []map[string]interface{} {
	evidence := make([]map[string]interface{}, 0, len(workloads))
	for _, workload := range workloads {
		evidence = append(evidence, map[string]interface{}{
			"policy_digest":      workload.PolicyDigest,
			"role":               workload.Role,
			"workload_id":        workload.WorkloadID,
			"workload_sans":      workload.WorkloadSANs,
			"workload_secret_id": workload.WorkloadSecretID,
		})
	}
	return evidence
}

func expectedWorkloadIdentityDigest(p verifierPolicy) string {
	return sha256JSONDigest(map[string]interface{}{
		"ids":  canonicalStringList(p.ExpectedWorkloadIDs),
		"sans": canonicalStringList(p.ExpectedWorkloadSANs),
	})
}

func modelWorkloadBindingDigest(p verifierPolicy) string {
	return sha256JSONDigest(canonicalModelWorkloadBindings(p.ModelWorkloadBindings))
}

func modelWorkloadBindingModelIDs(p verifierPolicy) []string {
	modelIDs := make([]string, 0, len(p.ModelWorkloadBindings))
	for modelID := range p.ModelWorkloadBindings {
		modelIDs = append(modelIDs, modelID)
	}
	sort.Strings(modelIDs)
	return modelIDs
}

func canonicalModelWorkloadBindings(bindings map[string]modelWorkloadBinding) map[string]map[string][]string {
	canonical := make(map[string]map[string][]string, len(bindings))
	for modelID, binding := range bindings {
		canonical[modelID] = map[string][]string{
			"workload_ids":  canonicalStringList(binding.WorkloadIDs),
			"workload_sans": canonicalStringList(binding.WorkloadSANs),
		}
	}
	return canonical
}

func modelWorkloadBindingPolicyViolations(p verifierPolicy) []string {
	if len(p.ModelWorkloadBindings) == 0 {
		return []string{"model_workload_bindings is required"}
	}

	var violations []string
	expectedSANs := stringSet(p.ExpectedWorkloadSANs)
	expectedIDs := stringSet(p.ExpectedWorkloadIDs)
	boundSANs := map[string]bool{}
	boundIDs := map[string]bool{}

	modelIDs := make([]string, 0, len(p.ModelWorkloadBindings))
	for modelID := range p.ModelWorkloadBindings {
		modelIDs = append(modelIDs, modelID)
	}
	sort.Strings(modelIDs)

	for _, modelID := range modelIDs {
		if !strings.HasPrefix(modelID, "privatemode/") {
			violations = append(violations, "model_workload_bindings keys must start with privatemode/")
		}
		binding := p.ModelWorkloadBindings[modelID]
		if len(binding.WorkloadSANs) == 0 && len(binding.WorkloadIDs) == 0 {
			violations = append(violations, fmt.Sprintf("model_workload_bindings for %s requires workload_sans or workload_ids", modelID))
			continue
		}
		for _, san := range binding.WorkloadSANs {
			if !expectedSANs[san] {
				violations = append(violations, fmt.Sprintf("model_workload_bindings for %s references workload_sans not listed in expected_workload_sans", modelID))
				break
			}
			boundSANs[san] = true
		}
		for _, workloadID := range binding.WorkloadIDs {
			if !expectedIDs[workloadID] {
				violations = append(violations, fmt.Sprintf("model_workload_bindings for %s references workload_ids not listed in expected_workload_ids", modelID))
				break
			}
			boundIDs[workloadID] = true
		}
	}

	for _, expectedSAN := range p.ExpectedWorkloadSANs {
		if !boundSANs[expectedSAN] {
			violations = append(violations, "expected_workload_sans contains unbound workloads")
			break
		}
	}
	for _, expectedID := range p.ExpectedWorkloadIDs {
		if !boundIDs[expectedID] {
			violations = append(violations, "expected_workload_ids contains unbound workloads")
			break
		}
	}
	return violations
}

func stringSet(values []string) map[string]bool {
	set := make(map[string]bool, len(values))
	for _, value := range values {
		set[value] = true
	}
	return set
}

func workloadSANs(sans []string) []string {
	var values []string
	for _, san := range sans {
		if strings.HasPrefix(san, "workload-") {
			values = append(values, san)
		}
	}
	return canonicalStringList(values)
}

func workloadIDFromSecretID(value string) string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return ""
	}
	parts := strings.Split(trimmed, "/")
	return strings.TrimSpace(parts[len(parts)-1])
}

func manifestLogEntryForManifest(root string, manifest []byte) (string, error) {
	logPath := filepath.Join(root, "log.txt")
	logBytes, err := os.ReadFile(logPath)
	if err != nil {
		return "", fmt.Errorf("reading manifest log: %w", err)
	}
	lines := strings.Split(string(logBytes), "\n")
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		fields := strings.Fields(trimmed)
		if len(fields) < 2 {
			continue
		}
		manifestPath, err := safeManifestLogPath(root, fields[len(fields)-1])
		if err != nil {
			return "", err
		}
		loggedManifest, err := os.ReadFile(manifestPath)
		if err != nil {
			return "", fmt.Errorf("reading manifest log entry %q: %w", fields[len(fields)-1], err)
		}
		if bytes.Equal(loggedManifest, manifest) {
			return trimmed, nil
		}
	}
	return "", errors.New("manifest log does not include active manifest")
}

func safeManifestLogPath(root string, name string) (string, error) {
	if strings.TrimSpace(name) == "" {
		return "", errors.New("manifest log entry filename is empty")
	}
	if filepath.IsAbs(name) {
		return "", fmt.Errorf("manifest log entry must be relative: %s", name)
	}
	clean := filepath.Clean(name)
	if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("manifest log entry escapes manifest log dir: %s", name)
	}
	path := filepath.Join(root, clean)
	rel, err := filepath.Rel(root, path)
	if err != nil {
		return "", err
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		return "", fmt.Errorf("manifest log entry escapes manifest log dir: %s", name)
	}
	return path, nil
}

func verifierResultForPayload(payload verifierPayload, claims map[string]interface{}, verifiedAt time.Time) verifierResult {
	if claims == nil {
		claims = map[string]interface{}{}
	}
	claims["payload_policy_digest"] = payload.PolicyDigest
	claims["payload_evidence_digest"] = payload.EvidenceDigest
	claims["payload_verification_nonce"] = payload.VerificationNonce
	return verifierResult{
		Verified:          true,
		Verifier:          "routstr-privatemode-go-verifier",
		PolicyDigest:      payload.PolicyDigest,
		EvidenceDigest:    payload.EvidenceDigest,
		VerificationNonce: payload.VerificationNonce,
		VerifiedAt:        verifiedAt.UTC().Unix(),
		Claims:            claims,
	}
}

func apiKeyFromPolicy(p verifierPolicy) (string, error) {
	if p.APIKeyFile != "" {
		apiKeyBytes, err := os.ReadFile(p.APIKeyFile)
		if err != nil {
			return "", fmt.Errorf("reading api_key_file: %w", err)
		}
		apiKey := strings.TrimSpace(string(apiKeyBytes))
		if apiKey == "" {
			return "", errors.New("api_key_file is empty")
		}
		return apiKey, nil
	}
	envName := p.APIKeyEnv
	if envName == "" {
		envName = defaultAPIKeyEn
	}
	apiKey := strings.TrimSpace(os.Getenv(envName))
	if apiKey == "" {
		return "", fmt.Errorf("%s is not set", envName)
	}
	return apiKey, nil
}

func apiHost(p verifierPolicy) (string, error) {
	apiBase := p.APIBaseURL
	if apiBase == "" {
		apiBase = defaultAPIBase
	}
	if violation := remoteURLViolation(apiBase, "api_base_url"); violation != "" {
		return "", errors.New(violation)
	}
	parsed, err := url.Parse(apiBase)
	if err != nil {
		return "", fmt.Errorf("parsing api_base_url: %w", err)
	}
	if parsed.Host == "" {
		return "", fmt.Errorf("api_base_url has no host: %s", apiBase)
	}
	return parsed.Host, nil
}

func parseMeshCACertificate(meshCAPEM []byte) (*x509.Certificate, error) {
	block, _ := pem.Decode(meshCAPEM)
	if block == nil {
		return nil, errors.New("decoding mesh CA certificate failed")
	}
	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("parsing mesh CA certificate: %w", err)
	}
	return cert, nil
}

func verifiedSecretServicePublicKey(certRaw []byte, meshCA *x509.Certificate) (*ecdsa.PublicKey, error) {
	cert, err := x509.ParseCertificate(certRaw)
	if err != nil {
		return nil, fmt.Errorf("parsing secret-service certificate: %w", err)
	}
	roots := x509.NewCertPool()
	roots.AddCert(meshCA)
	if _, err := cert.Verify(x509.VerifyOptions{Roots: roots}); err != nil {
		return nil, fmt.Errorf("verifying secret-service certificate: %w", err)
	}
	pub, ok := cert.PublicKey.(*ecdsa.PublicKey)
	if !ok {
		return nil, errors.New("secret-service public key is not ECDSA")
	}
	return pub, nil
}

func (p verifierPolicy) timeout() time.Duration {
	if p.TimeoutSeconds <= 0 {
		return 30 * time.Second
	}
	seconds := p.TimeoutSeconds
	if seconds < 1 {
		seconds = 1
	}
	if seconds > 60 {
		seconds = 60
	}
	return time.Duration(seconds * float64(time.Second))
}

func privatemodeVerificationSteps() map[string]bool {
	steps := make(map[string]bool, len(privatemodeVerificationStepNames))
	for _, step := range privatemodeVerificationStepNames {
		steps[step] = true
	}
	return steps
}

func sha256BytesDigest(value []byte) string {
	digest := sha256.Sum256(value)
	return "sha256:" + hex.EncodeToString(digest[:])
}

func sha256StringDigest(value string) string {
	return sha256BytesDigest([]byte(value))
}

func sha256FileDigest(path string) (string, error) {
	value, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return sha256BytesDigest(value), nil
}

func sha256JSONDigest(value interface{}) string {
	encoded, _ := json.Marshal(value)
	return sha256BytesDigest(encoded)
}

func sha256DirDigest(root string) (string, error) {
	var files []string
	if err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("manifest log contains symlink: %s", path)
		}
		files = append(files, path)
		return nil
	}); err != nil {
		return "", err
	}
	sort.Strings(files)

	h := sha256.New()
	for _, path := range files {
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return "", err
		}
		content, err := os.ReadFile(path)
		if err != nil {
			return "", err
		}
		h.Write([]byte(filepath.ToSlash(rel)))
		h.Write([]byte{0})
		h.Write([]byte(sha256BytesDigest(content)))
		h.Write([]byte{'\n'})
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil)), nil
}

func fetchURL(ctx context.Context, client *http.Client, url string) ([]byte, error) {
	if client == nil {
		client = http.DefaultClient
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status %s: %s", resp.Status, body)
	}
	return body, nil
}

func stringValue(raw map[string]interface{}, keys ...string) string {
	var selected string
	found := false
	for _, key := range keys {
		if value, ok := raw[key]; ok {
			if stringValue, ok := value.(string); ok {
				trimmed := strings.TrimSpace(stringValue)
				if !found {
					selected = trimmed
					found = true
					continue
				}
				if selected != trimmed {
					return ""
				}
			}
		}
	}
	return selected
}

func policyStringViolations(raw map[string]interface{}) []string {
	violations := stringFieldViolations(
		raw,
		"api_base_url",
		"apiBaseURL",
		"cdn_base_url",
		"cdnBaseURL",
		"manifest_url",
		"manifestURL",
		"api_key_env",
		"apiKeyEnv",
		"api_key_file",
		"apiKeyFile",
		"manifest_path",
		"manifestPath",
		"manifest_b64",
		"manifestB64",
		"manifest_log_dir",
		"manifestLogDir",
		"proxy_binary_path",
		"proxyBinaryPath",
		"manifest_digest",
		"manifestDigest",
		"proxy_image_digest",
		"proxyImageDigest",
		"proxy_binary_digest",
		"proxyBinaryDigest",
	)
	for _, group := range []struct {
		label string
		keys  []string
	}{
		{"api_base_url", []string{"api_base_url", "apiBaseURL"}},
		{"cdn_base_url", []string{"cdn_base_url", "cdnBaseURL"}},
		{"manifest_url", []string{"manifest_url", "manifestURL"}},
		{"api_key_env", []string{"api_key_env", "apiKeyEnv"}},
		{"api_key_file", []string{"api_key_file", "apiKeyFile"}},
		{"manifest_path", []string{"manifest_path", "manifestPath"}},
		{"manifest_b64", []string{"manifest_b64", "manifestB64"}},
		{"manifest_log_dir", []string{"manifest_log_dir", "manifestLogDir"}},
		{"proxy_binary_path", []string{"proxy_binary_path", "proxyBinaryPath"}},
		{"manifest_digest", []string{"manifest_digest", "manifestDigest"}},
		{"proxy_image_digest", []string{"proxy_image_digest", "proxyImageDigest"}},
		{"proxy_binary_digest", []string{"proxy_binary_digest", "proxyBinaryDigest"}},
	} {
		violations = append(violations, stringAliasViolations(raw, group.label, group.keys...)...)
	}
	return violations
}

func policyStringListViolations(raw map[string]interface{}) []string {
	var violations []string
	for _, group := range []struct {
		label string
		keys  []string
	}{
		{"expected_workload_sans", []string{"expected_workload_sans", "expectedWorkloadSANs"}},
		{"expected_workload_ids", []string{"expected_workload_ids", "expectedWorkloadIDs"}},
	} {
		for _, key := range group.keys {
			if value, ok := raw[key]; ok {
				violations = append(violations, stringListFieldViolations(key, value)...)
			}
		}
		violations = append(violations, stringListAliasViolations(raw, group.label, group.keys...)...)
	}
	return violations
}

func modelWorkloadBindingsValue(raw map[string]interface{}) (map[string]modelWorkloadBinding, []string) {
	var present []map[string]modelWorkloadBinding
	var violations []string
	for _, key := range []string{"model_workload_bindings", "modelWorkloadBindings"} {
		value, ok := raw[key]
		if !ok {
			continue
		}
		bindings, bindingViolations := parseModelWorkloadBindings(key, value)
		violations = append(violations, bindingViolations...)
		if len(bindings) > 0 {
			present = append(present, bindings)
		}
	}
	if len(present) == 0 {
		return nil, violations
	}
	if len(present) > 1 && !sameModelWorkloadBindings(present[0], present[1]) {
		violations = append(violations, "model_workload_bindings aliases must match")
	}
	return present[0], violations
}

func parseModelWorkloadBindings(key string, value interface{}) (map[string]modelWorkloadBinding, []string) {
	rawBindings, ok := value.(map[string]interface{})
	if !ok || len(rawBindings) == 0 {
		return nil, []string{key + " must be a non-empty object"}
	}

	bindings := make(map[string]modelWorkloadBinding, len(rawBindings))
	seenModelIDs := map[string]string{}
	var violations []string
	for rawModelID, rawBinding := range rawBindings {
		modelID := strings.TrimSpace(rawModelID)
		if modelID == "" {
			violations = append(violations, key+" keys must be non-empty model strings")
			continue
		}
		normalizedModelID := strings.ToLower(modelID)
		if previous, exists := seenModelIDs[normalizedModelID]; exists {
			duplicateModelID := modelID
			if strings.ToLower(previous) == normalizedModelID && strings.Compare(modelID, previous) < 0 {
				duplicateModelID = previous
			}
			violations = append(violations, fmt.Sprintf("%s contains duplicate model binding %s", key, duplicateModelID))
			continue
		}
		seenModelIDs[normalizedModelID] = modelID
		bindingObject, ok := rawBinding.(map[string]interface{})
		if !ok {
			violations = append(violations, fmt.Sprintf("%s for %s must be an object", key, modelID))
			continue
		}
		bindingViolations := modelWorkloadBindingFieldViolations(bindingObject)
		if len(bindingViolations) > 0 {
			for _, violation := range bindingViolations {
				violations = append(violations, fmt.Sprintf("%s for %s: %s", key, modelID, violation))
			}
			continue
		}

		workloadSANs := stringListValue(bindingObject, "workload_sans", "workloadSANs")
		workloadIDs := stringListValue(bindingObject, "workload_ids", "workloadIDs")
		if len(workloadSANs) == 0 && len(workloadIDs) == 0 {
			violations = append(violations, fmt.Sprintf("%s for %s requires workload_sans or workload_ids", key, modelID))
			continue
		}
		if _, exists := bindings[modelID]; exists {
			violations = append(violations, fmt.Sprintf("%s contains duplicate model binding %s", key, modelID))
			continue
		}
		bindings[modelID] = modelWorkloadBinding{
			WorkloadIDs:  canonicalStringList(workloadIDs),
			WorkloadSANs: canonicalStringList(workloadSANs),
		}
	}
	return bindings, violations
}

func modelWorkloadBindingFieldViolations(raw map[string]interface{}) []string {
	var violations []string
	for _, group := range []struct {
		label string
		keys  []string
	}{
		{"workload_sans", []string{"workload_sans", "workloadSANs"}},
		{"workload_ids", []string{"workload_ids", "workloadIDs"}},
	} {
		for _, key := range group.keys {
			if value, ok := raw[key]; ok {
				violations = append(violations, stringListFieldViolations(key, value)...)
			}
		}
		violations = append(violations, stringListAliasViolations(raw, group.label, group.keys...)...)
	}
	return violations
}

func sameModelWorkloadBindings(left map[string]modelWorkloadBinding, right map[string]modelWorkloadBinding) bool {
	return modelWorkloadBindingDigest(verifierPolicy{ModelWorkloadBindings: left}) ==
		modelWorkloadBindingDigest(verifierPolicy{ModelWorkloadBindings: right})
}

func stringListValue(raw map[string]interface{}, keys ...string) []string {
	var selected []string
	found := false
	for _, key := range keys {
		value, ok := raw[key]
		if !ok {
			continue
		}
		values, ok := stringListFromValue(value)
		if !ok || len(values) == 0 {
			continue
		}
		if !found {
			selected = values
			found = true
			continue
		}
		if !sameStringList(selected, values) {
			return nil
		}
	}
	return selected
}

func stringListFromValue(value interface{}) ([]string, bool) {
	switch typed := value.(type) {
	case []interface{}:
		values := make([]string, 0, len(typed))
		for _, item := range typed {
			itemString, ok := item.(string)
			if !ok {
				return nil, false
			}
			trimmed := strings.TrimSpace(itemString)
			if trimmed == "" {
				return nil, false
			}
			values = append(values, trimmed)
		}
		if len(values) == 0 {
			return nil, false
		}
		return canonicalStringList(values), true
	case []string:
		values := canonicalStringList(typed)
		if len(values) == 0 {
			return nil, false
		}
		return values, true
	default:
		return nil, false
	}
}

func stringListFieldViolations(key string, value interface{}) []string {
	switch typed := value.(type) {
	case []interface{}:
		if len(typed) == 0 {
			return []string{key + " must be a list of non-empty strings"}
		}
		seen := map[string]bool{}
		for _, item := range typed {
			itemString, ok := item.(string)
			trimmed := strings.TrimSpace(itemString)
			if !ok || trimmed == "" {
				return []string{key + " must be a list of non-empty strings"}
			}
			if seen[trimmed] {
				return []string{key + " must not contain duplicates"}
			}
			seen[trimmed] = true
		}
		return nil
	case []string:
		if len(typed) == 0 {
			return []string{key + " must be a list of non-empty strings"}
		}
		seen := map[string]bool{}
		for _, item := range typed {
			trimmed := strings.TrimSpace(item)
			if trimmed == "" {
				return []string{key + " must be a list of non-empty strings"}
			}
			if seen[trimmed] {
				return []string{key + " must not contain duplicates"}
			}
			seen[trimmed] = true
		}
		return nil
	default:
		return []string{key + " must be a list of non-empty strings"}
	}
}

func stringListAliasViolations(raw map[string]interface{}, label string, keys ...string) []string {
	var selected []string
	found := false
	for _, key := range keys {
		value, ok := raw[key]
		if !ok {
			continue
		}
		values, ok := stringListFromValue(value)
		if !ok || len(values) == 0 {
			continue
		}
		if !found {
			selected = values
			found = true
			continue
		}
		if !sameStringList(selected, values) {
			return []string{label + " aliases must match"}
		}
	}
	return nil
}

func canonicalStringList(values []string) []string {
	seen := map[string]bool{}
	canonical := make([]string, 0, len(values))
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" || seen[trimmed] {
			continue
		}
		seen[trimmed] = true
		canonical = append(canonical, trimmed)
	}
	sort.Strings(canonical)
	return canonical
}

func sameStringList(left []string, right []string) bool {
	left = canonicalStringList(left)
	right = canonicalStringList(right)
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func stringAliasViolations(raw map[string]interface{}, label string, keys ...string) []string {
	var selected string
	found := false
	for _, key := range keys {
		value, ok := raw[key]
		if !ok {
			continue
		}
		stringValue, ok := value.(string)
		if !ok {
			continue
		}
		trimmed := strings.TrimSpace(stringValue)
		if trimmed == "" {
			continue
		}
		if !found {
			selected = trimmed
			found = true
			continue
		}
		if selected != trimmed {
			return []string{label + " aliases must match"}
		}
	}
	return nil
}

func stringFieldViolations(raw map[string]interface{}, keys ...string) []string {
	var violations []string
	for _, key := range keys {
		value, ok := raw[key]
		if !ok {
			continue
		}
		if stringValue, ok := value.(string); ok {
			if strings.TrimSpace(stringValue) == "" {
				violations = append(violations, key+" must be a non-empty string")
			}
			continue
		}
		violations = append(violations, key+" must be a string")
	}
	return violations
}

func boolValue(raw map[string]interface{}, keys ...string) *bool {
	var selected *bool
	for _, key := range keys {
		value, ok := raw[key]
		if !ok {
			continue
		}
		typed, ok := value.(bool)
		if !ok {
			return nil
		}
		if selected == nil {
			selectedValue := typed
			selected = &selectedValue
			continue
		}
		if *selected != typed {
			return nil
		}
	}
	return selected
}

func intValue(raw map[string]interface{}, keys ...string) *int {
	var selected *int
	for _, key := range keys {
		value, ok := raw[key]
		if !ok {
			continue
		}
		var parsed int
		switch typed := value.(type) {
		case json.Number:
			integer, err := typed.Int64()
			if err != nil {
				return nil
			}
			parsed = int(integer)
		case float64:
			if typed != float64(int(typed)) {
				return nil
			}
			parsed = int(typed)
		case int:
			parsed = typed
		default:
			return nil
		}
		if selected == nil {
			selectedValue := parsed
			selected = &selectedValue
			continue
		}
		if *selected != parsed {
			return nil
		}
	}
	return selected
}

func floatValue(raw map[string]interface{}, keys ...string) float64 {
	for _, key := range keys {
		value, ok := raw[key]
		if !ok {
			continue
		}
		switch typed := value.(type) {
		case json.Number:
			v, _ := typed.Float64()
			return v
		case float64:
			return typed
		case int:
			return float64(typed)
		}
	}
	return 0
}
