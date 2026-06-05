package main

import (
	"bytes"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"

	"github.com/google/go-sev-guest/client"
)

const schemaVersion = "routstr-tee-attestation-request-v1"

var hpkePublicKeySizes = map[uint16]int{
	0x0010: 65,
	0x0011: 97,
	0x0012: 133,
	0x0020: 32,
	0x0021: 56,
}

type payload struct {
	SchemaVersion             string                 `json:"schema_version"`
	Service                   string                 `json:"service"`
	Version                   string                 `json:"version"`
	RoutingPolicyDigest       string                 `json:"routing_policy_digest"`
	Policy                    map[string]interface{} `json:"policy"`
	AttestationDocumentFormat string                 `json:"attestation_document_format"`
	PublicKeyDigest           string                 `json:"public_key_digest"`
	HPKEKeyConfigB64          string                 `json:"hpke_key_config_b64"`
	HPKEKeyConfigDigest       string                 `json:"hpke_key_config_digest"`
	HPKEPublicKeyHex          string                 `json:"hpke_public_key_hex"`
	HPKEPublicKeyDigest       string                 `json:"hpke_public_key_digest"`
	VerificationNonce         string                 `json:"verification_nonce"`
	TEEReportDataHex          string                 `json:"tee_report_data_hex"`
	TEEReportDataDigest       string                 `json:"tee_report_data_digest"`
	AttestationCommandDigest  string                 `json:"attestation_command_digest"`
	AttestationArtifactPath   string                 `json:"attestation_artifact_path"`
	VerifierCommandDigest     string                 `json:"verifier_command_digest"`
	VerifierArtifactPath      string                 `json:"verifier_artifact_path"`
}

type attestationResult struct {
	SchemaVersion             string `json:"schema_version"`
	AttestationDocumentFormat string `json:"attestation_document_format"`
	AttestationDocumentB64    string `json:"attestation_document_b64"`
	TEEReportDataHex          string `json:"tee_report_data_hex"`
	TEEReportDataDigest       string `json:"tee_report_data_digest"`
}

type externalQuoteResult struct {
	SchemaVersion             string `json:"schema_version"`
	AttestationDocumentFormat string `json:"attestation_document_format"`
	AttestationDocumentB64    string `json:"attestation_document_b64"`
	TEEReportDataHex          string `json:"tee_report_data_hex"`
	TEEReportDataDigest       string `json:"tee_report_data_digest"`
}

func main() {
	if err := runWithIO(os.Stdin, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func runWithIO(input io.Reader, output io.Writer) error {
	var p payload
	decoder := json.NewDecoder(input)
	decoder.UseNumber()
	if err := decodeSingleJSON(decoder, &p, "Routstr TEE attestation payload"); err != nil {
		return err
	}
	if p.SchemaVersion != schemaVersion {
		return fmt.Errorf("unexpected schema_version %q", p.SchemaVersion)
	}
	if err := validatePayloadBindings(p); err != nil {
		return err
	}

	reportData, err := reportDataBytes(p.TEEReportDataHex)
	if err != nil {
		return err
	}
	rawEvidence, err := generateAttestationDocument(p, reportData)
	if err != nil {
		return err
	}
	if len(rawEvidence) == 0 {
		return errors.New("attestation evidence is empty")
	}

	result := attestationResult{
		SchemaVersion:             "routstr-tee-attestation-document-v1",
		AttestationDocumentFormat: p.AttestationDocumentFormat,
		AttestationDocumentB64:    base64.StdEncoding.EncodeToString(rawEvidence),
		TEEReportDataHex:          p.TEEReportDataHex,
		TEEReportDataDigest:       p.TEEReportDataDigest,
	}
	encoder := json.NewEncoder(output)
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
		{"public_key_digest", p.PublicKeyDigest},
		{"hpke_key_config_digest", p.HPKEKeyConfigDigest},
		{"hpke_public_key_digest", p.HPKEPublicKeyDigest},
		{"tee_report_data_digest", p.TEEReportDataDigest},
		{"attestation_command_digest", p.AttestationCommandDigest},
	} {
		if !isFullSHA256Digest(check.value) {
			violations = append(
				violations,
				check.label+" must be a full sha256 digest",
			)
		}
	}
	if p.VerifierCommandDigest != "" && !isFullSHA256Digest(p.VerifierCommandDigest) {
		violations = append(
			violations,
			"verifier_command_digest must be a full sha256 digest",
		)
	}
	if strings.TrimSpace(p.VerificationNonce) == "" {
		violations = append(violations, "verification_nonce is required")
	}
	if err := validateHPKEPayloadBindings(p); err != nil {
		violations = append(violations, err.Error())
	}
	if _, err := reportDataBytes(p.TEEReportDataHex); err != nil {
		violations = append(violations, err.Error())
	} else {
		if expected := expectedReportDataDigest(p); expected != p.TEEReportDataDigest {
			violations = append(
				violations,
				"tee_report_data_digest does not match policy and keys",
			)
		}
		if expected := expectedReportDataHex(p); expected != strings.ToLower(strings.TrimSpace(p.TEEReportDataHex)) {
			violations = append(
				violations,
				"tee_report_data_hex does not match policy and keys",
			)
		}
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

func sha256JSONDigest(value interface{}) string {
	return sha256Bytes(canonicalJSON(value))
}

func canonicalJSON(value interface{}) []byte {
	data, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	return data
}

func sha256Bytes(value []byte) string {
	sum := sha256.Sum256(value)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func generateAttestationDocument(p payload, reportData [64]byte) ([]byte, error) {
	switch attestationKind(p.AttestationDocumentFormat) {
	case "sev-snp":
		return generateSEVSNPReport(p, reportData)
	case "tdx":
		return generateTDXQuote(p)
	default:
		return nil, fmt.Errorf(
			"unsupported attestation_document_format %q",
			p.AttestationDocumentFormat,
		)
	}
}

func attestationKind(format string) string {
	switch strings.TrimSpace(format) {
	case "sev_snp_quote", "sev_snp_report", "sev_guest_v2",
		"https://tinfoil.sh/predicate/sev-snp-guest/v2":
		return "sev-snp"
	case "tdx_quote", "tdx_guest_v2", "https://tinfoil.sh/predicate/tdx-guest/v2":
		return "tdx"
	default:
		return ""
	}
}

func reportDataBytes(reportDataHex string) ([64]byte, error) {
	var out [64]byte
	decoded, err := hex.DecodeString(strings.TrimSpace(reportDataHex))
	if err != nil {
		return out, fmt.Errorf("tee_report_data_hex is not valid hex: %w", err)
	}
	if len(decoded) != len(out) {
		return out, fmt.Errorf(
			"tee_report_data_hex must encode exactly 64 bytes, got %d",
			len(decoded),
		)
	}
	copy(out[:], decoded)
	return out, nil
}

func generateSEVSNPReport(p payload, reportData [64]byte) ([]byte, error) {
	devicePath, err := policyNonEmptyString(p.Policy, "sev_guest_device_path")
	if err != nil {
		return nil, err
	}
	if devicePath == "" {
		devicePath = strings.TrimSpace(os.Getenv("ROUTSTR_SEV_GUEST_DEVICE"))
	}
	if devicePath == "" {
		devicePath = "/dev/sev-guest"
	}
	vmpl, err := policyIntInRange(p.Policy, 0, 0, 3, "sev_snp_vmpl", "vmpl")
	if err != nil {
		return nil, err
	}

	device := &client.LinuxDevice{}
	if err := device.Open(devicePath); err != nil {
		return nil, fmt.Errorf("opening SEV-SNP guest device %s: %w", devicePath, err)
	}
	defer device.Close()

	report, err := client.GetRawReportAtVmpl(device, reportData, vmpl)
	if err != nil {
		return nil, fmt.Errorf("getting SEV-SNP report: %w", err)
	}
	return report, nil
}

func generateTDXQuote(p payload) ([]byte, error) {
	command, err := policyCommand(p.Policy, "tdx_quote_command")
	if err != nil {
		return nil, err
	}
	if len(command) == 0 {
		command, err = envCommand("ROUTSTR_TDX_QUOTE_COMMAND")
		if err != nil {
			return nil, err
		}
	}
	if len(command) == 0 {
		return nil, errors.New(
			"tdx_quote_command is required: Linux TDX_CMD_GET_REPORT0 returns " +
				"TDREPORT0, but Routstr's verifier expects a remotely verifiable " +
				"TDX QuoteV4",
		)
	}

	expectedDigest, err := policyFullSHA256Digest(p.Policy, "tdx_quote_command_digest")
	if err != nil {
		return nil, err
	}
	if expectedDigest == "" {
		expectedDigest = strings.TrimSpace(os.Getenv("ROUTSTR_TDX_QUOTE_COMMAND_DIGEST"))
	}
	if expectedDigest == "" {
		return nil, errors.New("tdx_quote_command_digest is required")
	}
	if !isFullSHA256Digest(expectedDigest) {
		return nil, errors.New("tdx_quote_command_digest must be a full sha256 digest")
	}
	artifactPath, err := policyNonEmptyString(p.Policy, "tdx_quote_artifact_path")
	if err != nil {
		return nil, err
	}
	if artifactPath == "" {
		artifactPath = strings.TrimSpace(os.Getenv("ROUTSTR_TDX_QUOTE_ARTIFACT_PATH"))
	}
	if artifactPath == "" {
		artifactPath = command[0]
	}
	if !commandContainsPath(command, artifactPath) {
		return nil, errors.New("tdx_quote_artifact_path must match tdx_quote_command executable or argument")
	}
	actualDigest, err := sha256FileDigest(artifactPath)
	if err != nil {
		return nil, fmt.Errorf("hashing TDX quote command artifact: %w", err)
	}
	if actualDigest != expectedDigest {
		return nil, fmt.Errorf(
			"tdx_quote_command_digest mismatch: expected %s, got %s",
			expectedDigest,
			actualDigest,
		)
	}

	requestBody, err := json.Marshal(map[string]interface{}{
		"schema_version":                     "routstr-tdx-quote-request-v1",
		"service":                            p.Service,
		"version":                            p.Version,
		"routing_policy_digest":              p.RoutingPolicyDigest,
		"policy":                             p.Policy,
		"attestation_document_format":        p.AttestationDocumentFormat,
		"public_key_digest":                  p.PublicKeyDigest,
		"hpke_key_config_digest":             p.HPKEKeyConfigDigest,
		"hpke_public_key_digest":             p.HPKEPublicKeyDigest,
		"verification_nonce":                 p.VerificationNonce,
		"tee_report_data_hex":                p.TEEReportDataHex,
		"tee_report_data_digest":             p.TEEReportDataDigest,
		"tdx_quote_command_digest":           actualDigest,
		"tdx_quote_command_artifact":         artifactPath,
		"routstr_attestation_command_digest": p.AttestationCommandDigest,
		"routstr_attestation_artifact":       p.AttestationArtifactPath,
		"routstr_verifier_digest":            p.VerifierCommandDigest,
		"routstr_verifier_artifact":          p.VerifierArtifactPath,
		"routstr_attestation_contract":       schemaVersion,
	})
	if err != nil {
		return nil, fmt.Errorf("encoding TDX quote request: %w", err)
	}

	cmd := exec.Command(command[0], command[1:]...)
	cmd.Stdin = bytes.NewReader(requestBody)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf(
			"TDX quote command failed: %w%s",
			err,
			stderrFailureDetail(stderr.Bytes()),
		)
	}

	var result externalQuoteResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		return nil, fmt.Errorf("TDX quote command did not return JSON: %w", err)
	}
	if result.SchemaVersion != "" &&
		result.SchemaVersion != "routstr-tee-attestation-document-v1" {
		return nil, fmt.Errorf("TDX quote command returned unexpected schema")
	}
	if result.AttestationDocumentFormat != "" &&
		result.AttestationDocumentFormat != p.AttestationDocumentFormat {
		return nil, fmt.Errorf("TDX quote command returned a different document format")
	}
	if result.TEEReportDataHex != "" && result.TEEReportDataHex != p.TEEReportDataHex {
		return nil, fmt.Errorf("TDX quote command returned a different report data value")
	}
	if result.TEEReportDataDigest != "" &&
		result.TEEReportDataDigest != p.TEEReportDataDigest {
		return nil, fmt.Errorf("TDX quote command returned a different report data digest")
	}
	if strings.TrimSpace(result.AttestationDocumentB64) == "" {
		return nil, errors.New("TDX quote command did not return attestation_document_b64")
	}
	rawEvidence, err := base64.StdEncoding.DecodeString(result.AttestationDocumentB64)
	if err != nil {
		return nil, fmt.Errorf("TDX quote command returned invalid base64: %w", err)
	}
	if len(rawEvidence) == 0 {
		return nil, errors.New("TDX quote command returned empty evidence")
	}
	return rawEvidence, nil
}

func policyFullSHA256Digest(policy map[string]interface{}, key string) (string, error) {
	value, ok := policy[key]
	if !ok {
		return "", nil
	}
	raw, ok := value.(string)
	if !ok || strings.TrimSpace(raw) == "" || !isFullSHA256Digest(raw) {
		return "", fmt.Errorf("%s must be a full sha256 digest", key)
	}
	return strings.TrimSpace(raw), nil
}

func policyNonEmptyString(policy map[string]interface{}, key string) (string, error) {
	value, ok := policy[key]
	if !ok {
		return "", nil
	}
	raw, ok := value.(string)
	if !ok || strings.TrimSpace(raw) == "" {
		return "", fmt.Errorf("%s must be a non-empty string", key)
	}
	return strings.TrimSpace(raw), nil
}

func policyIntInRange(
	policy map[string]interface{},
	defaultValue int,
	minValue int,
	maxValue int,
	keys ...string,
) (int, error) {
	selected := defaultValue
	found := false
	selectedKey := ""
	for _, key := range keys {
		value, ok := policy[key]
		if !ok {
			continue
		}
		var parsed int
		switch v := value.(type) {
		case int:
			parsed = v
		case int64:
			parsed = int(v)
		case json.Number:
			parsedValue, err := strconv.Atoi(v.String())
			if err != nil {
				return 0, fmt.Errorf(
					"%s must be an integer from %d to %d",
					key,
					minValue,
					maxValue,
				)
			}
			parsed = parsedValue
		default:
			return 0, fmt.Errorf(
				"%s must be an integer from %d to %d",
				key,
				minValue,
				maxValue,
			)
		}
		if parsed < minValue || parsed > maxValue {
			return 0, fmt.Errorf(
				"%s must be an integer from %d to %d",
				key,
				minValue,
				maxValue,
			)
		}
		if found && parsed != selected {
			return 0, fmt.Errorf("%s and %s aliases must match", selectedKey, key)
		}
		selected = parsed
		selectedKey = key
		found = true
	}
	return selected, nil
}

func policyCommand(policy map[string]interface{}, keys ...string) ([]string, error) {
	for _, key := range keys {
		value, ok := policy[key]
		if !ok {
			continue
		}
		command, err := commandFromValue(value)
		if err != nil {
			return nil, err
		}
		if len(command) > 0 {
			return command, nil
		}
	}
	return nil, nil
}

func commandFromValue(value interface{}) ([]string, error) {
	switch v := value.(type) {
	case string:
		if strings.TrimSpace(v) == "" {
			return nil, errors.New("tdx_quote_command is empty")
		}
		return envCommandString(v)
	case []string:
		if len(v) == 0 {
			return nil, errors.New("tdx_quote_command is empty")
		}
		command := make([]string, 0, len(v))
		for _, item := range v {
			if strings.TrimSpace(item) == "" {
				return nil, errors.New("tdx_quote_command entries must be non-empty strings")
			}
			command = append(command, item)
		}
		return command, nil
	case []interface{}:
		if len(v) == 0 {
			return nil, errors.New("tdx_quote_command is empty")
		}
		command := make([]string, 0, len(v))
		for _, item := range v {
			s, ok := item.(string)
			if !ok || strings.TrimSpace(s) == "" {
				return nil, errors.New("tdx_quote_command entries must be non-empty strings")
			}
			command = append(command, s)
		}
		return command, nil
	case nil:
		return nil, errors.New("tdx_quote_command is empty")
	default:
		return nil, errors.New("tdx_quote_command must be a string or string array")
	}
}

func envCommand(key string) ([]string, error) {
	return envCommandString(os.Getenv(key))
}

func envCommandString(command string) ([]string, error) {
	command = strings.TrimSpace(command)
	if command == "" {
		return nil, nil
	}
	if strings.HasPrefix(command, "[") {
		var argv []interface{}
		if err := json.Unmarshal([]byte(command), &argv); err != nil {
			return nil, fmt.Errorf("tdx_quote_command JSON must be a string array: %w", err)
		} else {
			return commandFromValue(argv)
		}
	}
	return strings.Fields(command), nil
}

func commandContainsPath(command []string, path string) bool {
	path = strings.TrimSpace(path)
	if path == "" {
		return false
	}
	for _, item := range command {
		if strings.TrimSpace(item) == path {
			return true
		}
	}
	return false
}

func sha256FileDigest(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

func stderrFailureDetail(stderr []byte) string {
	if len(stderr) == 0 {
		return ""
	}
	sum := sha256.Sum256(stderr)
	return fmt.Sprintf(
		": stderr redacted (%d bytes, sha256:%s)",
		len(stderr),
		hex.EncodeToString(sum[:]),
	)
}
