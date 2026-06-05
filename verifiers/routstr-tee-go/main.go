package main

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/tinfoilsh/tinfoil-go/verifier/attestation"
)

const schemaVersion = "routstr-tee-verifier-v1"

var hpkePublicKeySizes = map[uint16]int{
	0x0010: 65,
	0x0011: 97,
	0x0012: 133,
	0x0020: 32,
	0x0021: 56,
}

var routstrTEEVerificationSteps = map[string]bool{
	"tee_attestation_report": true,
	"tee_certificate_chain":  true,
	"measurement_match":      true,
	"runtime_policy_binding": true,
	"hpke_key_binding":       true,
	"public_key_binding":     true,
	"freshness":              true,
}

type payload struct {
	SchemaVersion             string                   `json:"schema_version"`
	Service                   string                   `json:"service"`
	Version                   string                   `json:"version"`
	RoutingPolicy             map[string]interface{}   `json:"routing_policy"`
	LocalArtifacts            map[string]localArtifact `json:"local_artifacts"`
	RoutingPolicyDigest       string                   `json:"routing_policy_digest"`
	Policy                    map[string]interface{}   `json:"policy"`
	AttestationDocumentFormat string                   `json:"attestation_document_format"`
	AttestationDocumentB64    string                   `json:"attestation_document_b64"`
	AttestationEvidenceDigest string                   `json:"attestation_evidence_digest"`
	PublicKeyDigest           string                   `json:"public_key_digest"`
	HPKEKeyConfigB64          string                   `json:"hpke_key_config_b64"`
	HPKEKeyConfigDigest       string                   `json:"hpke_key_config_digest"`
	HPKEPublicKeyHex          string                   `json:"hpke_public_key_hex"`
	HPKEPublicKeyDigest       string                   `json:"hpke_public_key_digest"`
	VerificationNonce         string                   `json:"verification_nonce"`
	TEEReportDataHex          string                   `json:"tee_report_data_hex"`
	TEEReportDataDigest       string                   `json:"tee_report_data_digest"`
	VerifierCommandDigest     string                   `json:"verifier_command_digest"`
	VerifierArtifactPath      string                   `json:"verifier_artifact_path"`
}

type verifierResult struct {
	Verified                  bool                   `json:"verified"`
	Verifier                  string                 `json:"verifier"`
	RoutingPolicyDigest       string                 `json:"routing_policy_digest"`
	AttestationEvidenceDigest string                 `json:"attestation_evidence_digest"`
	HPKEKeyConfigDigest       string                 `json:"hpke_key_config_digest"`
	VerificationNonce         string                 `json:"verification_nonce"`
	VerifiedAt                int64                  `json:"verified_at"`
	Claims                    map[string]interface{} `json:"claims"`
}

type localArtifact struct {
	Digest string `json:"digest"`
	Path   string `json:"path"`
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	var p payload
	decoder := json.NewDecoder(os.Stdin)
	decoder.UseNumber()
	if err := decodeSingleJSON(decoder, &p, "Routstr TEE verifier payload"); err != nil {
		return err
	}
	if p.SchemaVersion != schemaVersion {
		return fmt.Errorf("unexpected schema_version %q", p.SchemaVersion)
	}
	if err := validatePayloadBindings(p); err != nil {
		return err
	}

	verification, evidenceDigest, err := verifyAttestationEvidence(p)
	if err != nil {
		return err
	}
	if evidenceDigest != p.AttestationEvidenceDigest {
		return fmt.Errorf(
			"attestation_evidence_digest mismatch: payload has %s, evidence has %s",
			p.AttestationEvidenceDigest,
			evidenceDigest,
		)
	}

	claims, err := buildClaims(p, verification)
	if err != nil {
		return err
	}

	result := verifierResult{
		Verified:                  true,
		Verifier:                  "routstr-tee-go-verifier",
		RoutingPolicyDigest:       p.RoutingPolicyDigest,
		AttestationEvidenceDigest: p.AttestationEvidenceDigest,
		HPKEKeyConfigDigest:       p.HPKEKeyConfigDigest,
		VerificationNonce:         p.VerificationNonce,
		VerifiedAt:                time.Now().UTC().Unix(),
		Claims:                    claims,
	}

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

func validatePayloadBindings(p payload) error {
	var violations []string
	for _, check := range []struct {
		label string
		value string
	}{
		{"routing_policy_digest", p.RoutingPolicyDigest},
		{"attestation_evidence_digest", p.AttestationEvidenceDigest},
		{"public_key_digest", p.PublicKeyDigest},
		{"hpke_key_config_digest", p.HPKEKeyConfigDigest},
		{"hpke_public_key_digest", p.HPKEPublicKeyDigest},
		{"tee_report_data_digest", p.TEEReportDataDigest},
	} {
		if !isFullSHA256Digest(check.value) {
			violations = append(
				violations,
				check.label+" must be a full sha256 digest",
			)
		}
	}
	if strings.TrimSpace(p.VerificationNonce) == "" {
		violations = append(violations, "verification_nonce is required")
	}
	if p.VerifierCommandDigest != "" && !isFullSHA256Digest(p.VerifierCommandDigest) {
		violations = append(
			violations,
			"verifier_command_digest must be a full sha256 digest",
		)
	}
	if err := validateHPKEPayloadBindings(p); err != nil {
		violations = append(violations, err.Error())
	}
	if len(violations) > 0 {
		return errors.New(strings.Join(violations, "; "))
	}
	return nil
}

func validateHPKEPayloadBindings(p payload) error {
	keyConfig, err := base64.StdEncoding.DecodeString(strings.TrimSpace(p.HPKEKeyConfigB64))
	if err != nil || len(keyConfig) == 0 {
		return errors.New("hpke_key_config_b64 must be non-empty base64")
	}
	if sha256Bytes(keyConfig) != p.HPKEKeyConfigDigest {
		return errors.New("hpke_key_config_digest does not match hpke_key_config_b64")
	}
	publicKey, err := hex.DecodeString(strings.TrimSpace(p.HPKEPublicKeyHex))
	if err != nil || len(publicKey) != 32 {
		return errors.New("hpke_public_key_hex must encode exactly 32 bytes")
	}
	if sha256Bytes(publicKey) != p.HPKEPublicKeyDigest {
		return errors.New("hpke_public_key_digest does not match hpke_public_key_hex")
	}
	configPublicKey, err := hpkeKeyConfigPublicKey(keyConfig)
	if err != nil {
		return err
	}
	if !bytes.Equal(configPublicKey, publicKey) {
		return errors.New("hpke_key_config_b64 public key does not match hpke_public_key_hex")
	}
	return nil
}

func hpkeKeyConfigPublicKey(keyConfig []byte) ([]byte, error) {
	if len(keyConfig) < 1+2+2 {
		return nil, errors.New("invalid hpke_key_config_b64: too short")
	}
	offset := 1
	kemID, offset, err := readU16(keyConfig, offset)
	if err != nil {
		return nil, err
	}
	publicKeySize, ok := hpkePublicKeySizes[kemID]
	if !ok {
		return nil, fmt.Errorf("invalid hpke_key_config_b64: unsupported KEM 0x%04x", kemID)
	}
	if offset+publicKeySize > len(keyConfig) {
		return nil, errors.New("invalid hpke_key_config_b64: truncated public key")
	}
	publicKey := keyConfig[offset : offset+publicKeySize]
	offset += publicKeySize
	suitesLen, offset, err := readU16(keyConfig, offset)
	if err != nil {
		return nil, err
	}
	if suitesLen == 0 {
		return nil, errors.New("invalid hpke_key_config_b64: no cipher suites")
	}
	if suitesLen%4 != 0 {
		return nil, errors.New("invalid hpke_key_config_b64: malformed cipher suite list")
	}
	if offset+int(suitesLen) > len(keyConfig) {
		return nil, errors.New("invalid hpke_key_config_b64: truncated cipher suite list")
	}
	offset += int(suitesLen)
	if offset != len(keyConfig) {
		return nil, errors.New("invalid hpke_key_config_b64: trailing bytes")
	}
	return publicKey, nil
}

func readU16(data []byte, offset int) (uint16, int, error) {
	if offset+2 > len(data) {
		return 0, offset, errors.New("invalid hpke_key_config_b64: truncated uint16")
	}
	value := uint16(data[offset])<<8 | uint16(data[offset+1])
	return value, offset + 2, nil
}

func isFullSHA256Digest(value string) bool {
	digest := strings.TrimSpace(strings.ToLower(value))
	if !strings.HasPrefix(digest, "sha256:") {
		return false
	}
	hexDigest := strings.TrimPrefix(digest, "sha256:")
	if len(hexDigest) != 64 {
		return false
	}
	if _, err := hex.DecodeString(hexDigest); err != nil {
		return false
	}
	return !isRepeatedHexPlaceholder(hexDigest)
}

func isRepeatedHexPlaceholder(value string) bool {
	digest := strings.TrimSpace(strings.ToLower(value))
	if strings.HasPrefix(digest, "sha256:") {
		digest = strings.TrimPrefix(digest, "sha256:")
	}
	if len(digest) != 64 {
		return false
	}
	if _, err := hex.DecodeString(digest); err != nil {
		return false
	}
	for _, char := range digest {
		if char != rune(digest[0]) {
			return false
		}
	}
	return true
}

func verifyAttestationEvidence(p payload) (*attestation.Verification, string, error) {
	if strings.TrimSpace(p.AttestationDocumentB64) == "" {
		return nil, "", errors.New("attestation_document_b64 is required")
	}
	rawEvidence, err := base64.StdEncoding.DecodeString(p.AttestationDocumentB64)
	if err != nil {
		return nil, "", fmt.Errorf("decoding attestation_document_b64: %w", err)
	}
	format, err := predicateTypeForFormat(p.AttestationDocumentFormat)
	if err != nil {
		return nil, "", err
	}

	body, err := gzipBase64(rawEvidence)
	if err != nil {
		return nil, "", err
	}
	document := &attestation.Document{
		Format: format,
		Body:   body,
	}
	verification, err := document.Verify()
	if err != nil {
		return nil, "", fmt.Errorf("verifying Routstr TEE attestation report: %w", err)
	}
	return verification, sha256Bytes(rawEvidence), nil
}

func buildClaims(p payload, verification *attestation.Verification) (map[string]interface{}, error) {
	if verification == nil || verification.Measurement == nil {
		return nil, errors.New("TEE measurement is required")
	}

	expectedDigest := expectedReportDataDigest(p)
	if expectedDigest != p.TEEReportDataDigest {
		return nil, fmt.Errorf(
			"tee_report_data_digest mismatch: payload has %s, expected %s",
			p.TEEReportDataDigest,
			expectedDigest,
		)
	}
	expectedReportData := expectedReportDataHex(p)
	if expectedReportData != p.TEEReportDataHex {
		return nil, fmt.Errorf(
			"tee_report_data_hex mismatch: payload has %s, expected %s",
			p.TEEReportDataHex,
			expectedReportData,
		)
	}
	actualReportData := strings.ToLower(
		verification.TLSPublicKeyFP + verification.HPKEPublicKey,
	)
	if actualReportData != p.TEEReportDataHex {
		return nil, fmt.Errorf("report data mismatch: quote has %s, expected %s", actualReportData, p.TEEReportDataHex)
	}

	rawCodeMeasurement, err := measurementFingerprint(verification.Measurement)
	if err != nil {
		return nil, err
	}
	codeMeasurement, err := normalizeSHA256Digest(rawCodeMeasurement)
	if err != nil {
		return nil, fmt.Errorf("routstr_code_measurement claim must be a sha256 digest: %w", err)
	}
	allowedMeasurement, err := allowedPolicyDigestValue(
		p.Policy,
		codeMeasurement,
		"expected_routstr_code_measurement",
		"allowed_routstr_code_measurements",
	)
	if err != nil {
		return nil, err
	}
	if !allowedMeasurement {
		return nil, errors.New("routstr_code_measurement claim does not match policy")
	}
	localArtifacts, err := requiredLocalProxyBinaryArtifacts(p.RoutingPolicy, p.LocalArtifacts)
	if err != nil {
		return nil, err
	}

	claims := map[string]interface{}{
		"attestation_document_format":   p.AttestationDocumentFormat,
		"attestation_evidence_digest":   p.AttestationEvidenceDigest,
		"hpke_key_config_digest":        p.HPKEKeyConfigDigest,
		"hpke_public_key_digest":        p.HPKEPublicKeyDigest,
		"public_key_digest":             p.PublicKeyDigest,
		"payload_routing_policy_digest": p.RoutingPolicyDigest,
		"payload_verification_nonce":    p.VerificationNonce,
		"routstr_code_measurement":      codeMeasurement,
		"routstr_config_measurement":    p.RoutingPolicyDigest,
		"tee_attestation_report_digest": p.AttestationEvidenceDigest,
		"tee_certificate_chain_digest":  sha256String("verified-chain:" + p.AttestationDocumentFormat + ":" + p.AttestationEvidenceDigest),
		"tee_report_data_hex":           p.TEEReportDataHex,
		"tee_report_data_digest":        p.TEEReportDataDigest,
		"verification_steps":            routstrTEEVerificationSteps,
	}
	if len(localArtifacts) > 0 {
		claims["attested_local_artifacts"] = localArtifacts
	}
	if p.VerificationNonce != "" {
		claims["tee_report_nonce"] = p.VerificationNonce
		claims["tee_report_nonce_digest"] = sha256String(p.VerificationNonce)
	}
	if verification.Measurement != nil {
		claims["tee_measurement"] = map[string]interface{}{
			"type":      string(verification.Measurement.Type),
			"registers": verification.Measurement.Registers,
		}
	}
	return claims, nil
}

func requiredLocalProxyBinaryArtifacts(
	routingPolicy map[string]interface{},
	localArtifactInputs map[string]localArtifact,
) (map[string]interface{}, error) {
	if routingPolicy == nil {
		return nil, nil
	}
	mode, _ := routingPolicy["mode"].(string)
	required, _ := routingPolicy["required"].(bool)
	if mode != "required" || !required {
		return nil, nil
	}
	providers, ok := routingPolicy["providers"].([]interface{})
	if !ok {
		return nil, nil
	}

	artifacts := map[string]interface{}{}
	for _, rawProvider := range providers {
		provider, ok := rawProvider.(map[string]interface{})
		if !ok {
			continue
		}
		providerType, _ := provider["provider_type"].(string)
		confidentiality, _ := provider["confidentiality"].(map[string]interface{})
		confidentialityMode, _ := confidentiality["mode"].(string)
		verified, _ := confidentiality["verified"].(bool)
		if !verified {
			continue
		}

		artifactClaim := ""
		switch {
		case providerType == "privatemode" || confidentialityMode == "privatemode":
			artifactClaim = "privatemode_proxy_binary"
		case providerType == "ppq-private" || confidentialityMode == "ppq-private-tee":
			artifactClaim = "ppq_proxy_binary"
		default:
			continue
		}

		policy, _ := provider["confidentiality_policy"].(map[string]interface{})
		digest := policyDigestString(policy, "proxy_binary_digest", "proxyBinaryDigest")
		if digest == "" {
			continue
		}
		path := policyString(policy, "proxy_binary_path", "proxyBinaryPath")
		if path == "" {
			localArtifactInput, ok := localArtifactInputs[artifactClaim]
			if ok {
				if strings.TrimSpace(localArtifactInput.Digest) != digest {
					return nil, fmt.Errorf("local_artifacts.%s digest does not match routing policy", artifactClaim)
				}
				path = localArtifactInput.Path
			}
		}
		if path == "" {
			return nil, errors.New("proxy_binary_path is required when proxy_binary_digest is set")
		}
		actualDigest, err := sha256File(path)
		if err != nil {
			return nil, fmt.Errorf("hashing proxy_binary_path: %w", err)
		}
		if actualDigest != digest {
			return nil, errors.New("proxy_binary_digest does not match proxy_binary_path")
		}
		if existing, ok := artifacts[artifactClaim]; ok && existing != digest {
			return nil, fmt.Errorf("attested_local_artifacts.%s has conflicting routing policy digests", artifactClaim)
		}
		artifacts[artifactClaim] = digest
	}
	if len(artifacts) == 0 {
		return nil, nil
	}
	return artifacts, nil
}

func policyDigestString(policy map[string]interface{}, keys ...string) string {
	for _, key := range keys {
		value, ok := policy[key].(string)
		if !ok {
			continue
		}
		value = strings.TrimSpace(value)
		if isFullSHA256Digest(value) {
			return value
		}
	}
	return ""
}

func policyString(policy map[string]interface{}, keys ...string) string {
	for _, key := range keys {
		value, ok := policy[key].(string)
		if !ok {
			continue
		}
		value = strings.TrimSpace(value)
		if value != "" {
			return value
		}
	}
	return ""
}

func sha256File(path string) (string, error) {
	data, err := os.ReadFile(strings.TrimSpace(path))
	if err != nil {
		return "", err
	}
	return sha256Bytes(data), nil
}

func predicateTypeForFormat(format string) (attestation.PredicateType, error) {
	switch strings.TrimSpace(format) {
	case "tdx_quote", "tdx_guest_v2", string(attestation.TdxGuestV2):
		return attestation.TdxGuestV2, nil
	case "sev_snp_quote", "sev_snp_report", "sev_guest_v2", string(attestation.SevGuestV2):
		return attestation.SevGuestV2, nil
	default:
		return "", fmt.Errorf("unsupported attestation_document_format %q", format)
	}
}

func expectedReportDataDigest(p payload) string {
	return sha256JSONDigest(reportDataBinding(p))
}

func expectedReportDataHex(p payload) string {
	encoded := canonicalJSON(reportDataBinding(p))
	sum := sha512.Sum512(encoded)
	return hex.EncodeToString(sum[:])
}

func reportDataBinding(p payload) map[string]string {
	return map[string]string{
		"schema_version":         "routstr-tee-report-data-v1",
		"service":                "routstr",
		"routing_policy_digest":  p.RoutingPolicyDigest,
		"hpke_key_config_digest": p.HPKEKeyConfigDigest,
		"hpke_public_key_digest": p.HPKEPublicKeyDigest,
		"public_key_digest":      p.PublicKeyDigest,
		"verification_nonce":     p.VerificationNonce,
	}
}

func measurementFingerprint(measurement *attestation.Measurement) (string, error) {
	if measurement == nil {
		return "", errors.New("measurement is required")
	}
	return attestation.Fingerprint(measurement, nil, measurement.Type)
}

func allowedPolicyDigestValue(policy map[string]interface{}, actual string, keys ...string) (bool, error) {
	allowed, err := policyStringValues(policy, keys...)
	if err != nil {
		return false, err
	}
	if len(allowed) == 0 {
		return false, nil
	}
	actualDigest, err := normalizeSHA256Digest(actual)
	if err != nil {
		return false, fmt.Errorf("routstr_code_measurement claim must be a sha256 digest: %w", err)
	}
	for _, value := range allowed {
		allowedDigest, err := normalizeSHA256Digest(value)
		if err != nil {
			return false, errors.New("routstr_code_measurement policy values must be sha256 digests")
		}
		if allowedDigest == actualDigest {
			return true, nil
		}
	}
	return false, nil
}

func normalizeSHA256Digest(value string) (string, error) {
	digest := strings.TrimSpace(strings.ToLower(value))
	if strings.HasPrefix(digest, "sha256:") {
		digest = strings.TrimPrefix(digest, "sha256:")
	}
	if len(digest) != 64 {
		return "", fmt.Errorf("wrong length %d", len(digest))
	}
	for _, char := range digest {
		if !strings.ContainsRune("0123456789abcdef", char) {
			return "", fmt.Errorf("invalid hex character %q", char)
		}
	}
	if isRepeatedHexPlaceholder(digest) {
		return "", errors.New("placeholder digest")
	}
	return "sha256:" + digest, nil
}

func policyStringValues(policy map[string]interface{}, keys ...string) ([]string, error) {
	values := []string{}
	for _, key := range keys {
		value, ok := policy[key]
		if !ok {
			continue
		}
		switch typed := value.(type) {
		case string:
			trimmed := strings.TrimSpace(typed)
			if trimmed == "" {
				return nil, errors.New("routstr_code_measurement policy values must be sha256 digests")
			}
			values = append(values, trimmed)
		case []interface{}:
			if len(typed) == 0 {
				return nil, errors.New("routstr_code_measurement policy values must be sha256 digests")
			}
			for _, item := range typed {
				text, ok := item.(string)
				trimmed := strings.TrimSpace(text)
				if !ok {
					return nil, errors.New("routstr_code_measurement policy values must be strings")
				}
				if trimmed == "" {
					return nil, errors.New("routstr_code_measurement policy values must be sha256 digests")
				}
				values = append(values, trimmed)
			}
		case []string:
			if len(typed) == 0 {
				return nil, errors.New("routstr_code_measurement policy values must be sha256 digests")
			}
			for _, item := range typed {
				text := strings.TrimSpace(item)
				if text == "" {
					return nil, errors.New("routstr_code_measurement policy values must be sha256 digests")
				}
				values = append(values, text)
			}
		case nil:
			return nil, errors.New("routstr_code_measurement policy values must be sha256 digests")
		default:
			return nil, errors.New("routstr_code_measurement policy values must be strings")
		}
	}
	return values, nil
}

func gzipBase64(value []byte) (string, error) {
	var compressed bytes.Buffer
	writer := gzip.NewWriter(&compressed)
	if _, err := writer.Write(value); err != nil {
		return "", fmt.Errorf("compressing attestation evidence: %w", err)
	}
	if err := writer.Close(); err != nil {
		return "", fmt.Errorf("closing gzip writer: %w", err)
	}
	return base64.StdEncoding.EncodeToString(compressed.Bytes()), nil
}

func canonicalJSON(value interface{}) []byte {
	encoded, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	return encoded
}

func sha256JSONDigest(value interface{}) string {
	return sha256Bytes(canonicalJSON(value))
}

func sha256String(value string) string {
	return sha256Bytes([]byte(value))
}

func sha256Bytes(value []byte) string {
	sum := sha256.Sum256(value)
	return "sha256:" + hex.EncodeToString(sum[:])
}
