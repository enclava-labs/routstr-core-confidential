package main

import (
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/tinfoilsh/tinfoil-go/verifier/attestation"
)

func validPayload(t *testing.T) payload {
	t.Helper()
	hpkePublicKey := strings.Repeat("11", 32)
	hpkePublicKeyBytes, err := hex.DecodeString(hpkePublicKey)
	if err != nil {
		t.Fatalf("decode HPKE public key fixture: %v", err)
	}
	hpkeKeyConfig := ehbpKeyConfigForTest(hpkePublicKeyBytes)
	p := payload{
		SchemaVersion:             schemaVersion,
		Service:                   "routstr",
		RoutingPolicyDigest:       sha256String("routing-policy"),
		AttestationEvidenceDigest: sha256String("quote"),
		AttestationDocumentFormat: "tdx_quote",
		PublicKeyDigest:           sha256String("public-key"),
		HPKEKeyConfigB64:          base64.StdEncoding.EncodeToString(hpkeKeyConfig),
		HPKEKeyConfigDigest:       sha256Bytes(hpkeKeyConfig),
		HPKEPublicKeyHex:          hpkePublicKey,
		HPKEPublicKeyDigest:       sha256Bytes(hpkePublicKeyBytes),
		VerificationNonce:         "nonce",
		Policy: map[string]interface{}{
			"expected_routstr_code_measurement": testMeasurementFingerprintDigest(t),
		},
	}
	p.TEEReportDataDigest = expectedReportDataDigest(p)
	p.TEEReportDataHex = expectedReportDataHex(p)
	return p
}

func ehbpKeyConfigForTest(publicKey []byte) []byte {
	config := []byte{0x00, 0x00, 0x20}
	config = append(config, publicKey...)
	config = append(config, 0x00, 0x04, 0x00, 0x01, 0x00, 0x02)
	return config
}

func testMeasurement() *attestation.Measurement {
	return &attestation.Measurement{
		Type: attestation.TdxGuestV2,
		Registers: []string{
			strings.Repeat("01", 48),
			strings.Repeat("02", 48),
			strings.Repeat("03", 48),
			strings.Repeat("04", 48),
			strings.Repeat("00", 48),
		},
	}
}

func testMeasurementFingerprint(t *testing.T) string {
	t.Helper()
	fingerprint, err := measurementFingerprint(testMeasurement())
	if err != nil {
		t.Fatalf("measurement fingerprint: %v", err)
	}
	return fingerprint
}

func testMeasurementFingerprintDigest(t *testing.T) string {
	t.Helper()
	fingerprint := strings.TrimPrefix(testMeasurementFingerprint(t), "sha256:")
	return "sha256:" + fingerprint
}

func verificationForReportData(reportDataHex string) *attestation.Verification {
	return &attestation.Verification{
		Measurement:    testMeasurement(),
		TLSPublicKeyFP: reportDataHex[:64],
		HPKEPublicKey:  reportDataHex[64:],
	}
}

func writeLocalArtifact(t *testing.T, contents []byte) string {
	t.Helper()
	path := t.TempDir() + "/local-artifact"
	if err := os.WriteFile(path, contents, 0o755); err != nil {
		t.Fatalf("write local artifact: %v", err)
	}
	return path
}

func sha256FileForTest(t *testing.T, path string) string {
	t.Helper()
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read local artifact: %v", err)
	}
	return sha256Bytes(contents)
}

func TestBuildClaimsBindsReportDataAndMeasurements(t *testing.T) {
	p := validPayload(t)

	claims, err := buildClaims(p, verificationForReportData(p.TEEReportDataHex))
	if err != nil {
		t.Fatalf("buildClaims returned error: %v", err)
	}

	if got := claims["tee_report_data_hex"]; got != p.TEEReportDataHex {
		t.Fatalf("tee_report_data_hex = %v, want %s", got, p.TEEReportDataHex)
	}
	if got := claims["tee_report_data_digest"]; got != p.TEEReportDataDigest {
		t.Fatalf("tee_report_data_digest = %v, want %s", got, p.TEEReportDataDigest)
	}
	if got := claims["tee_report_nonce"]; got != p.VerificationNonce {
		t.Fatalf("tee_report_nonce = %v, want %s", got, p.VerificationNonce)
	}
	if got := claims["tee_report_nonce_digest"]; got != sha256String(p.VerificationNonce) {
		t.Fatalf("tee_report_nonce_digest = %v, want %s", got, sha256String(p.VerificationNonce))
	}
	if got := claims["routstr_config_measurement"]; got != p.RoutingPolicyDigest {
		t.Fatalf("routstr_config_measurement = %v, want %s", got, p.RoutingPolicyDigest)
	}
	if got := claims["routstr_code_measurement"]; got != testMeasurementFingerprintDigest(t) {
		t.Fatalf("routstr_code_measurement = %v, want %s", got, testMeasurementFingerprintDigest(t))
	}
	steps, ok := claims["verification_steps"].(map[string]bool)
	if !ok || !steps["tee_attestation_report"] || !steps["runtime_policy_binding"] {
		t.Fatalf("verification_steps missing required steps: %#v", claims["verification_steps"])
	}
}

func TestBuildClaimsIncludesRequiredPrivatemodeLocalArtifact(t *testing.T) {
	p := validPayload(t)
	proxyPath := writeLocalArtifact(t, []byte("privatemode-proxy-binary"))
	proxyDigest := sha256FileForTest(t, proxyPath)
	p.RoutingPolicy = map[string]interface{}{
		"mode":     "required",
		"required": true,
		"providers": []interface{}{
			map[string]interface{}{
				"provider_type": "privatemode",
				"confidentiality": map[string]interface{}{
					"verified": true,
					"mode":     "privatemode",
				},
				"confidentiality_policy": map[string]interface{}{
					"proxy_binary_digest": proxyDigest,
					"proxy_binary_path":   proxyPath,
				},
			},
		},
	}

	claims, err := buildClaims(p, verificationForReportData(p.TEEReportDataHex))
	if err != nil {
		t.Fatalf("buildClaims returned error: %v", err)
	}

	localArtifacts, ok := claims["attested_local_artifacts"].(map[string]interface{})
	if !ok {
		t.Fatalf("attested_local_artifacts missing or malformed: %#v", claims["attested_local_artifacts"])
	}
	if got := localArtifacts["privatemode_proxy_binary"]; got != proxyDigest {
		t.Fatalf("privatemode_proxy_binary = %v, want %s", got, proxyDigest)
	}
}

func TestBuildClaimsRejectsUnverifiedLocalArtifactDigest(t *testing.T) {
	p := validPayload(t)
	p.RoutingPolicy = map[string]interface{}{
		"mode":     "required",
		"required": true,
		"providers": []interface{}{
			map[string]interface{}{
				"provider_type": "ppq-private",
				"confidentiality": map[string]interface{}{
					"verified": true,
					"mode":     "ppq-private-tee",
				},
				"confidentiality_policy": map[string]interface{}{
					"proxy_binary_digest": sha256String("declared-ppq-proxy-binary"),
				},
			},
		},
	}

	_, err := buildClaims(p, verificationForReportData(p.TEEReportDataHex))
	if err == nil || !strings.Contains(err.Error(), "proxy_binary_path is required") {
		t.Fatalf("expected missing proxy_binary_path rejection, got %v", err)
	}
}

func TestBuildClaimsVerifiesPrivateLocalArtifactInput(t *testing.T) {
	p := validPayload(t)
	proxyPath := writeLocalArtifact(t, []byte("ppq proxy binary from private input"))
	proxyDigest := sha256FileForTest(t, proxyPath)
	p.LocalArtifacts = map[string]localArtifact{
		"ppq_proxy_binary": {
			Digest: proxyDigest,
			Path:   proxyPath,
		},
	}
	p.RoutingPolicy = map[string]interface{}{
		"mode":     "required",
		"required": true,
		"providers": []interface{}{
			map[string]interface{}{
				"provider_type": "ppq-private",
				"confidentiality": map[string]interface{}{
					"verified": true,
					"mode":     "ppq-private-tee",
				},
				"confidentiality_policy": map[string]interface{}{
					"proxy_binary_digest": proxyDigest,
				},
			},
		},
	}

	claims, err := buildClaims(p, verificationForReportData(p.TEEReportDataHex))
	if err != nil {
		t.Fatalf("buildClaims returned error: %v", err)
	}
	localArtifacts, ok := claims["attested_local_artifacts"].(map[string]interface{})
	if !ok {
		t.Fatalf("attested_local_artifacts missing or malformed: %#v", claims["attested_local_artifacts"])
	}
	if got := localArtifacts["ppq_proxy_binary"]; got != proxyDigest {
		t.Fatalf("ppq_proxy_binary = %v, want %s", got, proxyDigest)
	}
}

func TestBuildClaimsRejectsMismatchedLocalArtifactDigest(t *testing.T) {
	p := validPayload(t)
	proxyPath := writeLocalArtifact(t, []byte("actual ppq proxy binary"))
	p.RoutingPolicy = map[string]interface{}{
		"mode":     "required",
		"required": true,
		"providers": []interface{}{
			map[string]interface{}{
				"provider_type": "ppq-private",
				"confidentiality": map[string]interface{}{
					"verified": true,
					"mode":     "ppq-private-tee",
				},
				"confidentiality_policy": map[string]interface{}{
					"proxy_binary_digest": sha256String("different ppq proxy binary"),
					"proxy_binary_path":   proxyPath,
				},
			},
		},
	}

	_, err := buildClaims(p, verificationForReportData(p.TEEReportDataHex))
	if err == nil || !strings.Contains(err.Error(), "proxy_binary_digest does not match proxy_binary_path") {
		t.Fatalf("expected proxy digest mismatch rejection, got %v", err)
	}
}

func TestBuildClaimsRejectsConflictingLocalArtifactDigests(t *testing.T) {
	p := validPayload(t)
	firstPath := writeLocalArtifact(t, []byte("first-privatemode-proxy-binary"))
	secondPath := writeLocalArtifact(t, []byte("second-privatemode-proxy-binary"))
	firstDigest := sha256FileForTest(t, firstPath)
	secondDigest := sha256FileForTest(t, secondPath)
	p.RoutingPolicy = map[string]interface{}{
		"mode":     "required",
		"required": true,
		"providers": []interface{}{
			map[string]interface{}{
				"provider_type": "privatemode",
				"confidentiality": map[string]interface{}{
					"verified": true,
					"mode":     "privatemode",
				},
				"confidentiality_policy": map[string]interface{}{
					"proxy_binary_digest": firstDigest,
					"proxy_binary_path":   firstPath,
				},
			},
			map[string]interface{}{
				"provider_type": "privatemode",
				"confidentiality": map[string]interface{}{
					"verified": true,
					"mode":     "privatemode",
				},
				"confidentiality_policy": map[string]interface{}{
					"proxy_binary_digest": secondDigest,
					"proxy_binary_path":   secondPath,
				},
			},
		},
	}

	_, err := buildClaims(p, verificationForReportData(p.TEEReportDataHex))
	if err == nil || !strings.Contains(err.Error(), "attested_local_artifacts.privatemode_proxy_binary has conflicting routing policy digests") {
		t.Fatalf("expected conflicting local artifact digest error, got %v", err)
	}
}

func TestRunRejectsTrailingJSON(t *testing.T) {
	err := runWithStdinForTest(t, `{"schema_version":"`+schemaVersion+`"}{}`)
	if err == nil || !strings.Contains(err.Error(), "must contain exactly one JSON object") {
		t.Fatalf("expected trailing JSON error, got %v", err)
	}
}

func TestValidatePayloadBindingsRejectsPlaceholderDigests(t *testing.T) {
	p := validPayload(t)
	p.RoutingPolicyDigest = "sha256:policy"

	err := validatePayloadBindings(p)
	if err == nil {
		t.Fatal("validatePayloadBindings accepted placeholder routing policy digest")
	}
	if !strings.Contains(err.Error(), "routing_policy_digest must be a full sha256 digest") {
		t.Fatalf("error = %q, want routing_policy_digest", err)
	}
}

func TestValidatePayloadBindingsRejectsRepeatedHexPlaceholderDigests(t *testing.T) {
	p := validPayload(t)
	p.RoutingPolicyDigest = "sha256:" + strings.Repeat("a", 64)

	err := validatePayloadBindings(p)
	if err == nil {
		t.Fatal("validatePayloadBindings accepted repeated-hex placeholder digest")
	}
	if !strings.Contains(err.Error(), "routing_policy_digest must be a full sha256 digest") {
		t.Fatalf("error = %q, want routing_policy_digest", err)
	}
}

func TestValidatePayloadBindingsRejectsMissingNonce(t *testing.T) {
	p := validPayload(t)
	p.VerificationNonce = ""
	p.TEEReportDataDigest = expectedReportDataDigest(p)
	p.TEEReportDataHex = expectedReportDataHex(p)

	err := validatePayloadBindings(p)
	if err == nil {
		t.Fatal("validatePayloadBindings accepted missing nonce")
	}
	if !strings.Contains(err.Error(), "verification_nonce is required") {
		t.Fatalf("error = %q, want verification_nonce", err)
	}
}

func TestValidatePayloadBindingsRejectsMalformedVerifierDigest(t *testing.T) {
	p := validPayload(t)
	p.VerifierCommandDigest = "sha256:verifier"

	err := validatePayloadBindings(p)
	if err == nil {
		t.Fatal("validatePayloadBindings accepted malformed verifier digest")
	}
	if !strings.Contains(err.Error(), "verifier_command_digest must be a full sha256 digest") {
		t.Fatalf("error = %q, want verifier_command_digest", err)
	}
}

func TestValidatePayloadBindingsRejectsHPKEKeyConfigDigestMismatch(t *testing.T) {
	p := validPayload(t)
	p.HPKEKeyConfigDigest = sha256String("different-key-config")
	p.TEEReportDataDigest = expectedReportDataDigest(p)
	p.TEEReportDataHex = expectedReportDataHex(p)

	err := validatePayloadBindings(p)
	if err == nil {
		t.Fatal("validatePayloadBindings accepted HPKE key config digest mismatch")
	}
	if !strings.Contains(err.Error(), "hpke_key_config_digest does not match hpke_key_config_b64") {
		t.Fatalf("error = %q, want HPKE key config digest mismatch", err)
	}
}

func TestValidatePayloadBindingsRejectsHPKEPublicKeyDigestMismatch(t *testing.T) {
	p := validPayload(t)
	p.HPKEPublicKeyDigest = sha256String("different-public-key")
	p.TEEReportDataDigest = expectedReportDataDigest(p)
	p.TEEReportDataHex = expectedReportDataHex(p)

	err := validatePayloadBindings(p)
	if err == nil {
		t.Fatal("validatePayloadBindings accepted HPKE public key digest mismatch")
	}
	if !strings.Contains(err.Error(), "hpke_public_key_digest does not match hpke_public_key_hex") {
		t.Fatalf("error = %q, want HPKE public key digest mismatch", err)
	}
}

func TestValidatePayloadBindingsRejectsHPKEKeyConfigPublicKeyMismatch(t *testing.T) {
	p := validPayload(t)
	otherPublicKey := strings.Repeat("22", 32)
	otherPublicKeyBytes, err := hex.DecodeString(otherPublicKey)
	if err != nil {
		t.Fatalf("decode other public key: %v", err)
	}
	otherKeyConfig := ehbpKeyConfigForTest(otherPublicKeyBytes)
	p.HPKEKeyConfigB64 = base64.StdEncoding.EncodeToString(otherKeyConfig)
	p.HPKEKeyConfigDigest = sha256Bytes(otherKeyConfig)
	p.TEEReportDataDigest = expectedReportDataDigest(p)
	p.TEEReportDataHex = expectedReportDataHex(p)

	err = validatePayloadBindings(p)
	if err == nil {
		t.Fatal("validatePayloadBindings accepted HPKE key config public-key mismatch")
	}
	if !strings.Contains(err.Error(), "hpke_key_config_b64 public key does not match hpke_public_key_hex") {
		t.Fatalf("error = %q, want HPKE key config public-key mismatch", err)
	}
}

func TestReportDataBindingMatchesPythonGateVector(t *testing.T) {
	p := validPayload(t)

	if got, want := expectedReportDataDigest(p), "sha256:ae5ad6f499cc61730579aca80d2e36ee3fee1dc033234824434a0ceb364d6a5b"; got != want {
		t.Fatalf("expectedReportDataDigest = %s, want %s", got, want)
	}
	if got, want := expectedReportDataHex(p), "626d3a4ee9d0d5ae3259c57257b60ff04ff438df5b72724793e955988900eaf39601e87f5f9f2a3c5a1adf6a306a4bb09528ac3e6cf8634935d4765f3e41a9b9"; got != want {
		t.Fatalf("expectedReportDataHex = %s, want %s", got, want)
	}
}

func TestBuildClaimsRejectsReportDataMismatch(t *testing.T) {
	p := validPayload(t)

	_, err := buildClaims(p, verificationForReportData(strings.Repeat("11", 64)))
	if err == nil || !strings.Contains(err.Error(), "report data mismatch") {
		t.Fatalf("expected report data mismatch, got %v", err)
	}
}

func TestBuildClaimsRejectsCodeMeasurementOutsidePolicy(t *testing.T) {
	p := validPayload(t)
	p.Policy = map[string]interface{}{
		"allowed_routstr_code_measurements": []interface{}{sha256String("other-code-measurement")},
	}

	_, err := buildClaims(p, verificationForReportData(p.TEEReportDataHex))
	if err == nil || !strings.Contains(err.Error(), "routstr_code_measurement claim does not match policy") {
		t.Fatalf("expected measurement policy error, got %v", err)
	}
}

func TestBuildClaimsRejectsMalformedCodeMeasurementPolicy(t *testing.T) {
	p := validPayload(t)
	p.Policy = map[string]interface{}{
		"expected_routstr_code_measurement": "sha256:code",
	}

	_, err := buildClaims(p, verificationForReportData(p.TEEReportDataHex))
	if err == nil || !strings.Contains(err.Error(), "routstr_code_measurement policy values must be sha256 digests") {
		t.Fatalf("expected malformed measurement policy error, got %v", err)
	}
}

func TestBuildClaimsRejectsRepeatedHexPlaceholderCodeMeasurementPolicy(t *testing.T) {
	p := validPayload(t)
	p.Policy = map[string]interface{}{
		"expected_routstr_code_measurement": "sha256:" + strings.Repeat("b", 64),
	}

	_, err := buildClaims(p, verificationForReportData(p.TEEReportDataHex))
	if err == nil || !strings.Contains(err.Error(), "routstr_code_measurement policy values must be sha256 digests") {
		t.Fatalf("expected placeholder measurement policy error, got %v", err)
	}
}

func TestBuildClaimsRejectsBlankCodeMeasurementPolicyAlias(t *testing.T) {
	p := validPayload(t)
	p.Policy = map[string]interface{}{
		"expected_routstr_code_measurement": "",
		"allowed_routstr_code_measurements": []interface{}{
			testMeasurementFingerprintDigest(t),
		},
	}

	_, err := buildClaims(p, verificationForReportData(p.TEEReportDataHex))
	if err == nil || !strings.Contains(err.Error(), "routstr_code_measurement policy values must be sha256 digests") {
		t.Fatalf("expected blank measurement policy error, got %v", err)
	}
}

func TestBuildClaimsRejectsNonStringCodeMeasurementPolicy(t *testing.T) {
	p := validPayload(t)
	p.Policy = map[string]interface{}{
		"allowed_routstr_code_measurements": []interface{}{
			json.Number(strings.Repeat("1", 64)),
		},
	}

	_, err := buildClaims(p, verificationForReportData(p.TEEReportDataHex))
	if err == nil || !strings.Contains(err.Error(), "routstr_code_measurement policy values must be strings") {
		t.Fatalf("expected non-string measurement policy error, got %v", err)
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
