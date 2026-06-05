package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"os"
	"strings"
	"testing"
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
		AttestationDocumentFormat: "tdx_quote",
		Policy: map[string]interface{}{
			"tdx_quote_command": []interface{}{
				os.Args[0],
				"-test.run=TestExternalQuoteHelper",
				"--",
			},
			"tdx_quote_command_digest": sha256File(t, os.Args[0]),
		},
		PublicKeyDigest:          sha256String("public-key"),
		HPKEKeyConfigDigest:      sha256BytesForTest(hpkeKeyConfig),
		HPKEPublicKeyDigest:      sha256BytesForTest(hpkePublicKeyBytes),
		VerificationNonce:        "nonce",
		HPKEKeyConfigB64:         base64.StdEncoding.EncodeToString(hpkeKeyConfig),
		HPKEPublicKeyHex:         hpkePublicKey,
		AttestationCommandDigest: sha256String("attestation-command"),
		VerifierCommandDigest:    sha256String("verifier"),
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

func sha256File(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read helper: %v", err)
	}
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func sha256String(value string) string {
	sum := sha256.Sum256([]byte(value))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func sha256BytesForTest(value []byte) string {
	sum := sha256.Sum256(value)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func runPayload(t *testing.T, p payload) attestationResult {
	t.Helper()
	t.Setenv("GO_WANT_ROUTSTR_TDX_QUOTE_HELPER", "1")
	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	var output bytes.Buffer
	if err := runWithIO(bytes.NewReader(input), &output); err != nil {
		t.Fatalf("runWithIO returned error: %v", err)
	}
	var result attestationResult
	if err := json.Unmarshal(output.Bytes(), &result); err != nil {
		t.Fatalf("decode output: %v", err)
	}
	return result
}

func TestExternalTDXQuoteCommandBindsReportDataAndDigest(t *testing.T) {
	p := validPayload(t)

	result := runPayload(t, p)

	if result.SchemaVersion != "routstr-tee-attestation-document-v1" {
		t.Fatalf("schema_version = %s", result.SchemaVersion)
	}
	if result.AttestationDocumentFormat != p.AttestationDocumentFormat {
		t.Fatalf(
			"attestation_document_format = %s, want %s",
			result.AttestationDocumentFormat,
			p.AttestationDocumentFormat,
		)
	}
	if result.TEEReportDataHex != p.TEEReportDataHex {
		t.Fatalf("tee_report_data_hex = %s, want %s", result.TEEReportDataHex, p.TEEReportDataHex)
	}
	if result.TEEReportDataDigest != p.TEEReportDataDigest {
		t.Fatalf("tee_report_data_digest = %s, want %s", result.TEEReportDataDigest, p.TEEReportDataDigest)
	}

	rawQuote, err := base64.StdEncoding.DecodeString(result.AttestationDocumentB64)
	if err != nil {
		t.Fatalf("decode attestation document: %v", err)
	}
	var quote map[string]interface{}
	if err := json.Unmarshal(rawQuote, &quote); err != nil {
		t.Fatalf("decode helper quote: %v", err)
	}
	if quote["report_data_hex"] != p.TEEReportDataHex {
		t.Fatalf("helper quote report_data_hex = %v", quote["report_data_hex"])
	}
	if quote["verification_nonce"] != p.VerificationNonce {
		t.Fatalf("helper quote verification_nonce = %v", quote["verification_nonce"])
	}
}

func TestRunWithIORejectsTrailingJSON(t *testing.T) {
	err := runWithIO(bytes.NewReader([]byte(`{"schema_version":"`+schemaVersion+`"}{}`)), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "must contain exactly one JSON object") {
		t.Fatalf("expected trailing JSON error, got %v", err)
	}
}

func TestPayloadBindingsRejectPlaceholderDigests(t *testing.T) {
	p := validPayload(t)
	p.RoutingPolicyDigest = "sha256:policy"

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "routing_policy_digest must be a full sha256 digest") {
		t.Fatalf("expected routing policy digest error, got %v", err)
	}
}

func TestPayloadBindingsRejectRepeatedHexPlaceholderDigests(t *testing.T) {
	p := validPayload(t)
	p.RoutingPolicyDigest = "sha256:" + strings.Repeat("a", 64)

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "routing_policy_digest must be a full sha256 digest") {
		t.Fatalf("expected routing policy digest error, got %v", err)
	}
}

func TestPayloadBindingsRejectMissingNonce(t *testing.T) {
	p := validPayload(t)
	p.VerificationNonce = ""

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "verification_nonce is required") {
		t.Fatalf("expected nonce error, got %v", err)
	}
}

func TestPayloadBindingsRejectMalformedVerifierDigest(t *testing.T) {
	p := validPayload(t)
	p.VerifierCommandDigest = "sha256:verifier"

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "verifier_command_digest must be a full sha256 digest") {
		t.Fatalf("expected verifier digest error, got %v", err)
	}
}

func TestPayloadBindingsRejectHPKEKeyConfigDigestMismatch(t *testing.T) {
	p := validPayload(t)
	p.HPKEKeyConfigDigest = sha256String("different-hpke-key-config")

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "hpke_key_config_digest does not match hpke_key_config_b64") {
		t.Fatalf("expected HPKE key config digest mismatch, got %v", err)
	}
}

func TestPayloadBindingsRejectHPKEPublicKeyDigestMismatch(t *testing.T) {
	p := validPayload(t)
	p.HPKEPublicKeyDigest = sha256String("different-hpke-public-key")

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "hpke_public_key_digest does not match hpke_public_key_hex") {
		t.Fatalf("expected HPKE public key digest mismatch, got %v", err)
	}
}

func TestPayloadBindingsRejectHPKEKeyConfigPublicKeyMismatch(t *testing.T) {
	p := validPayload(t)
	otherPublicKey := bytes.Repeat([]byte{0x22}, 32)
	p.HPKEKeyConfigB64 = base64.StdEncoding.EncodeToString(
		ehbpKeyConfigForTest(otherPublicKey),
	)
	p.HPKEKeyConfigDigest = sha256BytesForTest(ehbpKeyConfigForTest(otherPublicKey))
	p.TEEReportDataDigest = expectedReportDataDigest(p)
	p.TEEReportDataHex = expectedReportDataHex(p)

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "hpke_key_config_b64 public key does not match hpke_public_key_hex") {
		t.Fatalf("expected HPKE key config public-key mismatch, got %v", err)
	}
}

func TestPayloadBindingsRejectReportDataDigestMismatch(t *testing.T) {
	p := validPayload(t)
	p.TEEReportDataDigest = sha256String("different-report-data-binding")

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "tee_report_data_digest does not match policy and keys") {
		t.Fatalf("expected report data digest mismatch, got %v", err)
	}
}

func TestExternalTDXQuoteCommandRequiresDigestPin(t *testing.T) {
	p := validPayload(t)
	delete(p.Policy, "tdx_quote_command_digest")

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "tdx_quote_command_digest is required") {
		t.Fatalf("expected digest pin error, got %v", err)
	}
}

func TestExternalTDXQuoteCommandRejectsBlankPolicyDigestBeforeEnvFallback(t *testing.T) {
	p := validPayload(t)
	p.Policy["tdx_quote_command_digest"] = ""
	t.Setenv("ROUTSTR_TDX_QUOTE_COMMAND_DIGEST", sha256File(t, os.Args[0]))
	t.Setenv("GO_WANT_ROUTSTR_TDX_QUOTE_HELPER", "1")

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "tdx_quote_command_digest must be a full sha256 digest") {
		t.Fatalf("expected blank digest policy error, got %v", err)
	}
}

func TestExternalTDXQuoteCommandRejectsMalformedDigestBeforeHashing(t *testing.T) {
	p := validPayload(t)
	p.Policy["tdx_quote_command_digest"] = "sha256:tdx"
	p.Policy["tdx_quote_artifact_path"] = "/definitely/missing/tdx-quote-wrapper"

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "tdx_quote_command_digest must be a full sha256 digest") {
		t.Fatalf("expected malformed digest error, got %v", err)
	}
	if strings.Contains(err.Error(), "hashing TDX quote command artifact") {
		t.Fatalf("digest validation happened after artifact hashing: %v", err)
	}
}

func TestExternalTDXQuoteCommandRejectsRepeatedHexPlaceholderDigest(t *testing.T) {
	p := validPayload(t)
	p.Policy["tdx_quote_command_digest"] = "sha256:" + strings.Repeat("b", 64)

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "tdx_quote_command_digest must be a full sha256 digest") {
		t.Fatalf("expected placeholder digest policy error, got %v", err)
	}
}

func TestExternalTDXQuoteCommandRejectsNonStringPolicyDigestBeforeEnvFallback(t *testing.T) {
	p := validPayload(t)
	p.Policy["tdx_quote_command_digest"] = json.Number("123")
	t.Setenv("ROUTSTR_TDX_QUOTE_COMMAND_DIGEST", sha256File(t, os.Args[0]))

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "tdx_quote_command_digest must be a full sha256 digest") {
		t.Fatalf("expected non-string digest policy error, got %v", err)
	}
}

func TestExternalTDXQuoteCommandRejectsBlankCommandBeforeEnvFallback(t *testing.T) {
	p := validPayload(t)
	p.Policy["tdx_quote_command"] = ""
	t.Setenv("ROUTSTR_TDX_QUOTE_COMMAND", os.Args[0]+" -test.run=TestExternalQuoteHelper --")
	t.Setenv("GO_WANT_ROUTSTR_TDX_QUOTE_HELPER", "1")

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "tdx_quote_command is empty") {
		t.Fatalf("expected blank command policy error, got %v", err)
	}
}

func TestExternalTDXQuoteCommandRejectsNonStringCommandEntryBeforeHashing(t *testing.T) {
	p := validPayload(t)
	p.Policy["tdx_quote_command"] = []interface{}{123}

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "tdx_quote_command entries must be non-empty strings") {
		t.Fatalf("expected command entry type error, got %v", err)
	}
	if strings.Contains(err.Error(), "hashing TDX quote command artifact") {
		t.Fatalf("command validation happened after artifact hashing: %v", err)
	}
}

func TestExternalTDXQuoteCommandRejectsUnboundArtifactPathBeforeHashing(t *testing.T) {
	p := validPayload(t)
	p.Policy["tdx_quote_artifact_path"] = "/definitely/missing/tdx-quote-wrapper"

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "tdx_quote_artifact_path must match tdx_quote_command executable or argument") {
		t.Fatalf("expected unbound artifact path error, got %v", err)
	}
	if strings.Contains(err.Error(), "hashing TDX quote command artifact") {
		t.Fatalf("artifact binding validation happened after artifact hashing: %v", err)
	}
}

func TestExternalTDXQuoteCommandRejectsNonStringArtifactPathBeforeDefault(t *testing.T) {
	p := validPayload(t)
	p.Policy["tdx_quote_artifact_path"] = json.Number("123")

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "tdx_quote_artifact_path must be a non-empty string") {
		t.Fatalf("expected non-string artifact path error, got %v", err)
	}
}

func TestExternalTDXQuoteCommandRejectsBlankArtifactPathBeforeEnvFallback(t *testing.T) {
	p := validPayload(t)
	p.Policy["tdx_quote_artifact_path"] = ""
	t.Setenv("ROUTSTR_TDX_QUOTE_ARTIFACT_PATH", os.Args[0])
	t.Setenv("GO_WANT_ROUTSTR_TDX_QUOTE_HELPER", "1")

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "tdx_quote_artifact_path must be a non-empty string") {
		t.Fatalf("expected blank artifact path error, got %v", err)
	}
}

func TestSEVSNPPolicyRejectsNonStringDevicePathBeforeDefault(t *testing.T) {
	p := validPayload(t)
	p.AttestationDocumentFormat = "sev_snp_report"
	p.Policy = map[string]interface{}{
		"sev_guest_device_path": json.Number("123"),
	}

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "sev_guest_device_path must be a non-empty string") {
		t.Fatalf("expected non-string device path error, got %v", err)
	}
	if strings.Contains(err.Error(), "opening SEV-SNP guest device") {
		t.Fatalf("device path validation happened after device open: %v", err)
	}
}

func TestSEVSNPPolicyRejectsBlankDevicePathBeforeDefault(t *testing.T) {
	p := validPayload(t)
	p.AttestationDocumentFormat = "sev_snp_report"
	p.Policy = map[string]interface{}{
		"sev_guest_device_path": "",
	}

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "sev_guest_device_path must be a non-empty string") {
		t.Fatalf("expected blank device path error, got %v", err)
	}
	if strings.Contains(err.Error(), "opening SEV-SNP guest device") {
		t.Fatalf("device path validation happened after device open: %v", err)
	}
}

func TestSEVSNPPolicyRejectsMalformedVMPLBeforeDeviceOpen(t *testing.T) {
	for name, value := range map[string]interface{}{
		"string":       "1",
		"fractional":   json.Number("0.5"),
		"null":         nil,
		"out-of-range": json.Number("4"),
	} {
		t.Run(name, func(t *testing.T) {
			p := validPayload(t)
			p.AttestationDocumentFormat = "sev_snp_report"
			p.Policy = map[string]interface{}{
				"sev_guest_device_path": "/definitely/missing/sev-guest",
				"sev_snp_vmpl":          value,
			}

			input, err := json.Marshal(p)
			if err != nil {
				t.Fatalf("marshal payload: %v", err)
			}
			err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
			if err == nil || !strings.Contains(err.Error(), "sev_snp_vmpl must be an integer from 0 to 3") {
				t.Fatalf("expected vmpl policy error, got %v", err)
			}
			if strings.Contains(err.Error(), "opening SEV-SNP guest device") {
				t.Fatalf("vmpl validation happened after device open: %v", err)
			}
		})
	}
}

func TestSEVSNPPolicyRejectsMalformedVMPLAliasWhenPrimaryIsValid(t *testing.T) {
	for name, value := range map[string]interface{}{
		"string": "1",
		"null":   nil,
	} {
		t.Run(name, func(t *testing.T) {
			p := validPayload(t)
			p.AttestationDocumentFormat = "sev_snp_report"
			p.Policy = map[string]interface{}{
				"sev_guest_device_path": "/definitely/missing/sev-guest",
				"sev_snp_vmpl":          json.Number("0"),
				"vmpl":                  value,
			}

			input, err := json.Marshal(p)
			if err != nil {
				t.Fatalf("marshal payload: %v", err)
			}
			err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
			if err == nil || !strings.Contains(err.Error(), "vmpl must be an integer from 0 to 3") {
				t.Fatalf("expected vmpl alias policy error, got %v", err)
			}
			if strings.Contains(err.Error(), "opening SEV-SNP guest device") {
				t.Fatalf("vmpl alias validation happened after device open: %v", err)
			}
		})
	}
}

func TestSEVSNPPolicyRejectsConflictingVMPLAliasesBeforeDeviceOpen(t *testing.T) {
	p := validPayload(t)
	p.AttestationDocumentFormat = "sev_snp_report"
	p.Policy = map[string]interface{}{
		"sev_guest_device_path": "/definitely/missing/sev-guest",
		"sev_snp_vmpl":          json.Number("0"),
		"vmpl":                  json.Number("1"),
	}

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "sev_snp_vmpl and vmpl aliases must match") {
		t.Fatalf("expected vmpl alias mismatch error, got %v", err)
	}
	if strings.Contains(err.Error(), "opening SEV-SNP guest device") {
		t.Fatalf("vmpl alias validation happened after device open: %v", err)
	}
}

func TestExternalTDXQuoteCommandRejectsReportDataMismatch(t *testing.T) {
	p := validPayload(t)
	p.Policy["tdx_quote_mismatch_for_test"] = true
	t.Setenv("GO_WANT_ROUTSTR_TDX_QUOTE_HELPER", "1")

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "different report data value") {
		t.Fatalf("expected report data mismatch, got %v", err)
	}
}

func TestReportDataHexMustBeExactly64Bytes(t *testing.T) {
	p := validPayload(t)
	p.TEEReportDataHex = "abcd"

	input, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	err = runWithIO(bytes.NewReader(input), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "tee_report_data_hex must encode exactly 64 bytes") {
		t.Fatalf("expected report data length error, got %v", err)
	}
}

func TestExternalQuoteHelper(t *testing.T) {
	if os.Getenv("GO_WANT_ROUTSTR_TDX_QUOTE_HELPER") != "1" {
		return
	}

	var p map[string]interface{}
	if err := json.NewDecoder(os.Stdin).Decode(&p); err != nil {
		t.Fatalf("decode helper payload: %v", err)
	}
	reportDataHex := p["tee_report_data_hex"].(string)
	if policy, ok := p["policy"].(map[string]interface{}); ok {
		if policy["tdx_quote_mismatch_for_test"] == true {
			reportDataHex = strings.Repeat("00", 64)
		}
	}
	quote := map[string]interface{}{
		"schema_version":      "fixture-tdx-quote-v4",
		"report_data_hex":     reportDataHex,
		"verification_nonce":  p["verification_nonce"],
		"routing_policy_hash": p["routing_policy_digest"],
	}
	quoteBytes, err := json.Marshal(quote)
	if err != nil {
		t.Fatalf("marshal quote: %v", err)
	}
	result := map[string]interface{}{
		"schema_version":               "routstr-tee-attestation-document-v1",
		"attestation_document_format":  p["attestation_document_format"],
		"attestation_document_b64":     base64.StdEncoding.EncodeToString(quoteBytes),
		"tee_report_data_hex":          reportDataHex,
		"tee_report_data_digest":       p["tee_report_data_digest"],
		"external_quote_command_nonce": p["verification_nonce"],
	}
	if err := json.NewEncoder(os.Stdout).Encode(result); err != nil {
		t.Fatalf("encode helper result: %v", err)
	}
	os.Exit(0)
}
