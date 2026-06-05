package main

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/tinfoilsh/tinfoil-go/verifier/attestation"
	"github.com/tinfoilsh/tinfoil-go/verifier/client"
	"github.com/tinfoilsh/tinfoil-go/verifier/github"
	"github.com/tinfoilsh/tinfoil-go/verifier/sigstore"
)

const schemaVersion = "routstr-confidential-verifier-v1"
const ppqPrivateAttestationBundleAttempts = 8

var fetchAndVerifyFromURLJSON = client.FetchAndVerifyFromURLJSON

var ehbpVerificationSteps = map[string]bool{
	"hardware_attestation_report":    true,
	"hardware_certificate_chain":     true,
	"code_transparency":              true,
	"measurement_match":              true,
	"attested_transport_key_binding": true,
	"freshness":                      true,
}

type payload struct {
	SchemaVersion         string                 `json:"schema_version"`
	Mode                  string                 `json:"mode"`
	BaseURL               string                 `json:"base_url"`
	Origin                string                 `json:"origin"`
	ModelIDs              []string               `json:"model_ids"`
	PolicyDigest          string                 `json:"policy_digest"`
	Policy                map[string]interface{} `json:"policy"`
	Attestation           map[string]interface{} `json:"attestation"`
	AttestationBundleURL  string                 `json:"attestation_bundle_url"`
	HPKEKeyConfigB64      string                 `json:"hpke_key_config_b64"`
	ModelAttestations     []modelAttestation     `json:"model_attestations"`
	EvidenceDigest        string                 `json:"evidence_digest"`
	VerificationNonce     string                 `json:"verification_nonce"`
	VerifierCommandDigest string                 `json:"verifier_command_digest"`
	VerifierArtifactPath  string                 `json:"verifier_artifact_path"`
}

type modelAttestation struct {
	ModelID          string                 `json:"model_id"`
	Policy           map[string]interface{} `json:"policy"`
	AttestationURL   string                 `json:"attestation_url"`
	Attestation      map[string]interface{} `json:"attestation"`
	HPKEKeysURL      string                 `json:"hpke_keys_url"`
	HPKEKeyConfigB64 string                 `json:"hpke_key_config_b64"`
	HPKEPublicKeyHex string                 `json:"hpke_public_key_hex"`
	EvidenceDigest   string                 `json:"evidence_digest"`
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
	if err := decodeSingleJSON(decoder, &p, "verifier payload"); err != nil {
		return err
	}
	if p.SchemaVersion != schemaVersion {
		return fmt.Errorf("unexpected schema_version %q", p.SchemaVersion)
	}
	if err := validatePayloadBindings(p); err != nil {
		return err
	}

	var claims map[string]interface{}
	var err error
	switch p.Mode {
	case "tinfoil":
		claims, err = verifyTinfoil(p)
	case "ppq-private-tee":
		claims, err = verifyPPQPrivate(p)
	default:
		err = fmt.Errorf("unsupported verifier mode %q", p.Mode)
	}
	if err != nil {
		return err
	}

	claims["payload_policy_digest"] = p.PolicyDigest
	claims["payload_evidence_digest"] = p.EvidenceDigest
	claims["payload_verification_nonce"] = p.VerificationNonce
	claims["verification_steps"] = ehbpVerificationSteps

	result := verifierResult{
		Verified:          true,
		Verifier:          verifierNameForMode(p.Mode),
		PolicyDigest:      p.PolicyDigest,
		EvidenceDigest:    p.EvidenceDigest,
		VerificationNonce: p.VerificationNonce,
		VerifiedAt:        time.Now().UTC().Unix(),
		Claims:            claims,
	}

	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(result)
}

func verifierNameForMode(mode string) string {
	if mode == "ppq-private-tee" {
		return "routstr-ppq-private-go-verifier"
	}
	return "routstr-tinfoil-go-verifier"
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
		{"policy_digest", p.PolicyDigest},
		{"evidence_digest", p.EvidenceDigest},
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
	if p.Mode == "tinfoil" {
		violations = append(violations, validateTinfoilAttestationPayload(p.Attestation)...)
		violations = append(violations, validateTinfoilModelAttestationPayloads(p)...)
	}
	violations = append(
		violations,
		policyStringFieldViolations(
			p.Policy,
			"repo",
			"expected_repo",
			"enclave_host",
			"expected_enclave_host",
			"attestation_bundle_url",
			"proxy_binary_digest",
			"proxyBinaryDigest",
			"proxy_binary_path",
			"proxyBinaryPath",
		)...,
	)
	violations = append(
		violations,
		policyAliasViolations(
			p.Policy,
			"repo",
			"repo",
			"expected_repo",
		)...,
	)
	violations = append(
		violations,
		policyAliasViolations(
			p.Policy,
			"enclave host",
			"enclave_host",
			"expected_enclave_host",
		)...,
	)
	routeHost := ""
	routeOrigin := ""
	routeLabel := ""
	for _, check := range []struct {
		label string
		value string
	}{
		{"base_url", p.BaseURL},
		{"origin", p.Origin},
	} {
		identity, violation := remoteURLIdentity(check.value, check.label)
		if violation != "" {
			violations = append(violations, violation)
			continue
		}
		if identity.host == "" {
			continue
		}
		if routeHost == "" {
			routeHost = identity.host
			routeOrigin = identity.origin
			routeLabel = check.label
			continue
		}
		if identity.host != routeHost {
			violations = append(
				violations,
				check.label+" host must match "+routeLabel+" host",
			)
			continue
		}
		if identity.origin != "" && routeOrigin != "" && identity.origin != routeOrigin {
			violations = append(
				violations,
				check.label+" origin must match "+routeLabel+" origin",
			)
		}
	}
	for _, check := range []struct {
		label string
		value string
	}{
		{"attestation_bundle_url", p.AttestationBundleURL},
		{"attestation_bundle_url", policyString(p.Policy, "attestation_bundle_url")},
	} {
		identity, violation := remoteURLIdentity(check.value, check.label)
		if violation != "" {
			violations = append(violations, violation)
			continue
		}
		if identity.host != "" && routeHost != "" && identity.host != routeHost {
			violations = append(
				violations,
				check.label+" host must match "+routeLabel+" host",
			)
			continue
		}
		if identity.origin != "" && routeOrigin != "" && identity.origin != routeOrigin {
			violations = append(
				violations,
				check.label+" origin must match "+routeLabel+" origin",
			)
		}
	}
	tinfoilHasRouteHost := routeHost != ""
	for _, check := range []struct {
		label string
		value string
	}{
		{"enclave_host", policyString(p.Policy, "enclave_host")},
		{"expected_enclave_host", policyString(p.Policy, "expected_enclave_host")},
	} {
		host, violation := hostIdentityHost(check.value, check.label)
		if violation != "" {
			violations = append(
				violations,
				violation,
			)
			continue
		}
		if host != "" {
			tinfoilHasRouteHost = true
		}
		if host != "" && routeHost != "" && host != routeHost {
			violations = append(
				violations,
				check.label+" must match "+routeLabel+" host",
			)
		}
	}
	if p.Mode == "tinfoil" && !tinfoilHasRouteHost {
		violations = append(
			violations,
			"base_url, origin, or enclave_host is required for tinfoil",
		)
	}
	if p.Mode == "ppq-private-tee" {
		_, modelIDViolations := normalizedPPQPrivateModelIDs(p.ModelIDs)
		violations = append(violations, modelIDViolations...)
		violations = append(
			violations,
			validateTinfoilModelAttestationPayloads(p)...,
		)
		if violation := ppqPrivateBaseURLViolation(p.BaseURL); violation != "" {
			violations = append(violations, violation)
		}
		for _, check := range []struct {
			label string
			value string
		}{
			{"attestation_bundle_url", p.AttestationBundleURL},
			{"attestation_bundle_url", policyString(p.Policy, "attestation_bundle_url")},
		} {
			if violation := ppqPrivateRemoteURLPathViolation(
				check.value,
				check.label,
			); violation != "" {
				violations = append(violations, violation)
			}
		}
		policyBundleURL := policyString(p.Policy, "attestation_bundle_url")
		if strings.TrimSpace(p.AttestationBundleURL) != "" &&
			policyBundleURL != "" &&
			strings.TrimSpace(p.AttestationBundleURL) != policyBundleURL {
			violations = append(
				violations,
				"attestation_bundle_url must match policy",
			)
		}
	}
	if len(violations) > 0 {
		return errors.New(strings.Join(violations, "; "))
	}
	return nil
}

func normalizedPPQPrivateModelIDs(modelIDs []string) ([]string, []string) {
	var violations []string
	if len(modelIDs) == 0 {
		return nil, []string{"model_ids are required for ppq-private-tee"}
	}
	normalizedModelIDs := make([]string, 0, len(modelIDs))
	seenModelIDs := map[string]bool{}
	for _, modelID := range modelIDs {
		modelID = strings.TrimSpace(modelID)
		if modelID == "" {
			violations = append(violations, "model_ids must contain only non-empty strings")
			continue
		}
		normalizedModelID := strings.ToLower(modelID)
		if seenModelIDs[normalizedModelID] {
			violations = append(violations, "model_ids must not contain duplicates")
			continue
		}
		seenModelIDs[normalizedModelID] = true
		if !strings.HasPrefix(modelID, "private/") {
			violations = append(
				violations,
				"model_ids must start with private/ for ppq-private-tee",
			)
			continue
		}
		normalizedModelIDs = append(normalizedModelIDs, modelID)
	}
	return normalizedModelIDs, violations
}

func remoteURLViolation(value string, label string) string {
	_, violation := remoteURLHost(value, label)
	return violation
}

type remoteURLInfo struct {
	host   string
	origin string
}

func remoteURLIdentity(value string, label string) (remoteURLInfo, string) {
	rawURL := strings.TrimSpace(value)
	if rawURL == "" {
		return remoteURLInfo{}, ""
	}
	parsed, err := url.Parse(rawURL)
	if err != nil {
		if strings.Contains(err.Error(), "invalid port") {
			return remoteURLInfo{}, label + " must include a valid port"
		}
		return remoteURLInfo{}, label + " must be an absolute URL"
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return remoteURLInfo{}, label + " must be an absolute URL"
	}
	if parsed.User != nil || strings.Contains(parsed.Host, "@") {
		return remoteURLInfo{}, label + " must not contain userinfo"
	}
	if !strings.EqualFold(parsed.Scheme, "https") {
		return remoteURLInfo{}, label + " must use https"
	}
	hostname := normalizeHostname(parsed.Hostname())
	if hostname == "" {
		return remoteURLInfo{}, label + " must include a host"
	}
	scheme := strings.ToLower(parsed.Scheme)
	port := parsed.Port()
	origin := scheme + "://" + hostname
	if port != "" && !(scheme == "https" && port == "443") {
		origin += ":" + port
	}
	return remoteURLInfo{host: hostname, origin: origin}, ""
}

func remoteURLHost(value string, label string) (string, string) {
	identity, violation := remoteURLIdentity(value, label)
	return identity.host, violation
}

func hostIdentityViolation(value string, label string) string {
	_, violation := hostIdentityHost(value, label)
	return violation
}

func ppqPrivateBaseURLViolation(value string) string {
	rawURL := strings.TrimSpace(value)
	if rawURL == "" {
		return "base_url is required for ppq-private-tee"
	}
	parsed, err := url.Parse(rawURL)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return ""
	}
	for _, segment := range strings.Split(parsed.Path, "/") {
		if strings.EqualFold(segment, "private") {
			return ""
		}
	}
	return "base_url must include a private path segment for ppq-private-tee"
}

func ppqPrivateRemoteURLPathViolation(value string, label string) string {
	rawURL := strings.TrimSpace(value)
	if rawURL == "" {
		return ""
	}
	if _, violation := remoteURLHost(rawURL, label); violation != "" {
		return ""
	}
	parsed, err := url.Parse(rawURL)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return ""
	}
	for _, segment := range strings.Split(parsed.Path, "/") {
		if strings.EqualFold(segment, "private") {
			return ""
		}
	}
	return label + " must include a private path segment"
}

func validateTinfoilAttestationPayload(attestation map[string]interface{}) []string {
	var violations []string

	formatValue, formatOK := attestation["format"].(string)
	if !formatOK || strings.TrimSpace(formatValue) == "" {
		violations = append(violations, "attestation format is required")
	}
	bodyValue, bodyOK := attestation["body"].(string)
	if !bodyOK || strings.TrimSpace(bodyValue) == "" {
		violations = append(violations, "attestation body is required")
	}
	reportDigestValue, reportDigestOK := attestation["report_digest"].(string)
	reportDigest := strings.TrimSpace(reportDigestValue)
	if !reportDigestOK || !isFullSHA256Digest(reportDigest) {
		violations = append(
			violations,
			"attestation report_digest must be a full sha256 digest",
		)
	}
	if encodingValue, ok := attestation["body_encoding"]; ok {
		encoding, ok := encodingValue.(string)
		if !ok || strings.TrimSpace(encoding) != "base64+gzip" {
			violations = append(
				violations,
				"attestation body_encoding must be base64+gzip",
			)
		}
	}
	if gzipDigestValue, ok := attestation["body_gzip_digest"]; ok {
		gzipDigest, ok := gzipDigestValue.(string)
		if !ok || !isFullSHA256Digest(gzipDigest) {
			violations = append(
				violations,
				"attestation body_gzip_digest must be a full sha256 digest",
			)
		}
	}
	if !bodyOK || strings.TrimSpace(bodyValue) == "" {
		return violations
	}

	bodyBytes, err := base64.StdEncoding.DecodeString(strings.TrimSpace(bodyValue))
	if err != nil {
		return append(violations, "attestation body must be valid base64")
	}
	if gzipDigestValue, ok := attestation["body_gzip_digest"].(string); ok &&
		isFullSHA256Digest(gzipDigestValue) &&
		sha256Bytes(bodyBytes) != strings.TrimSpace(gzipDigestValue) {
		violations = append(
			violations,
			"attestation body_gzip_digest does not match body",
		)
	}

	reader, err := gzip.NewReader(bytes.NewReader(bodyBytes))
	if err != nil {
		return append(violations, "attestation body must be valid gzip")
	}
	reportBytes, readErr := io.ReadAll(reader)
	closeErr := reader.Close()
	if readErr != nil {
		return append(violations, "attestation body gzip payload is unreadable")
	}
	if closeErr != nil {
		return append(violations, "attestation body gzip payload is invalid")
	}
	if reportDigestOK && isFullSHA256Digest(reportDigest) &&
		sha256Bytes(reportBytes) != reportDigest {
		violations = append(
			violations,
			"attestation report_digest does not match body",
		)
	}
	if reportB64Value, ok := attestation["report_b64"]; ok {
		reportB64, ok := reportB64Value.(string)
		if !ok || strings.TrimSpace(reportB64) == "" {
			return append(violations, "attestation report_b64 must be valid base64")
		}
		reportFromB64, err := base64.StdEncoding.DecodeString(strings.TrimSpace(reportB64))
		if err != nil {
			return append(violations, "attestation report_b64 must be valid base64")
		}
		if !bytes.Equal(reportBytes, reportFromB64) {
			violations = append(
				violations,
				"attestation report_b64 does not match body",
			)
		}
	}
	return violations
}

func validateTinfoilModelAttestationPayloads(p payload) []string {
	var violations []string
	required := boolPolicyValue(p.Policy, "require_model_attestations", "requireModelAttestations")
	if p.Mode == "ppq-private-tee" && !required {
		return []string{"model_attestations require require_model_attestations=true"}
	}
	if len(p.ModelAttestations) == 0 && !required {
		return violations
	}
	if !required {
		return []string{"model_attestations require require_model_attestations=true"}
	}
	targets, err := tinfoilModelAttestationTargetPolicies(p.Policy)
	if err != nil {
		return []string{err.Error()}
	}
	violations = append(violations, validateTinfoilPayloadModelIDs(p.ModelIDs, targets)...)
	seen := map[string]bool{}
	for _, modelAttestation := range p.ModelAttestations {
		modelID := strings.TrimSpace(modelAttestation.ModelID)
		if modelID == "" {
			violations = append(violations, "model_attestations model_id is required")
			continue
		}
		normalizedModelID := strings.ToLower(modelID)
		if seen[normalizedModelID] {
			violations = append(violations, "duplicate model_attestations entry for "+modelID)
			continue
		}
		seen[normalizedModelID] = true
		if _, ok := targets[modelID]; !ok {
			violations = append(
				violations,
				"model_attestations contains unconfigured model "+modelID,
			)
		}
		for _, violation := range validateTinfoilAttestationPayload(modelAttestation.Attestation) {
			violations = append(
				violations,
				"model_attestations for "+modelID+": "+violation,
			)
		}
		if !isFullSHA256Digest(modelAttestation.EvidenceDigest) {
			violations = append(
				violations,
				"model_attestations for "+modelID+" evidence_digest must be a full sha256 digest",
			)
		}
		if strings.TrimSpace(modelAttestation.HPKEKeyConfigB64) == "" {
			violations = append(
				violations,
				"model_attestations for "+modelID+" hpke_key_config_b64 is required",
			)
		}
		for _, check := range []struct {
			label string
			value string
		}{
			{"model_attestations attestation_url", modelAttestation.AttestationURL},
			{"model_attestations hpke_keys_url", modelAttestation.HPKEKeysURL},
		} {
			if violation := remoteURLViolation(check.value, check.label); violation != "" {
				violations = append(violations, violation)
			}
		}
	}
	for modelID := range targets {
		if !seen[modelID] {
			violations = append(
				violations,
				"model_attestations is missing "+modelID,
			)
		}
	}
	return violations
}

func validateTinfoilPayloadModelIDs(
	modelIDs []string,
	targets map[string]map[string]interface{},
) []string {
	normalized, violations := normalizedTinfoilPayloadModelIDs(modelIDs)
	if len(violations) > 0 {
		return violations
	}
	if len(normalized) == 0 {
		return []string{"model_ids are required with model_attestation_targets"}
	}
	targetModelIDs := make([]string, 0, len(targets))
	for modelID := range targets {
		targetModelIDs = append(targetModelIDs, modelID)
	}
	sort.Strings(targetModelIDs)
	if len(normalized) != len(targetModelIDs) {
		return []string{"model_ids must match model_attestation_targets"}
	}
	for index := range normalized {
		if normalized[index] != targetModelIDs[index] {
			return []string{"model_ids must match model_attestation_targets"}
		}
	}
	return nil
}

func normalizedTinfoilPayloadModelIDs(modelIDs []string) ([]string, []string) {
	normalized := make([]string, 0, len(modelIDs))
	seen := map[string]bool{}
	var violations []string
	for _, modelID := range modelIDs {
		modelID = strings.TrimSpace(modelID)
		if modelID == "" {
			violations = append(violations, "model_ids must contain only non-empty strings")
			continue
		}
		normalizedModelID := strings.ToLower(modelID)
		if seen[normalizedModelID] {
			violations = append(violations, "model_ids must not contain duplicates")
			continue
		}
		seen[normalizedModelID] = true
		normalized = append(normalized, modelID)
	}
	sort.Strings(normalized)
	return normalized, violations
}

func hostIdentityHost(value string, label string) (string, string) {
	rawURL := strings.TrimSpace(value)
	if rawURL == "" {
		return "", ""
	}
	if urlContainsUserinfo(rawURL) {
		return "", label + " must not contain userinfo"
	}
	if strings.Contains(rawURL, "://") {
		parsed, err := url.Parse(rawURL)
		if err != nil || parsed.Host == "" {
			return "", label + " must be a host or absolute URL"
		}
		if !strings.EqualFold(parsed.Scheme, "https") {
			return "", label + " must use https"
		}
		return normalizeHostname(parsed.Hostname()), ""
	}
	if strings.ContainsAny(rawURL, "/?#") {
		return "", label + " must be a host or absolute URL"
	}
	host := rawURL
	if splitHost, _, err := net.SplitHostPort(rawURL); err == nil {
		host = splitHost
	} else if strings.Contains(rawURL, ":") && net.ParseIP(rawURL) == nil {
		return "", label + " must be a host or absolute URL"
	}
	host = normalizeHostname(host)
	if host == "" {
		return "", label + " must include a host"
	}
	return host, ""
}

func normalizeHostname(hostname string) string {
	return strings.TrimSuffix(strings.ToLower(strings.TrimSpace(hostname)), ".")
}

func urlContainsUserinfo(value string) bool {
	rawURL := strings.TrimSpace(value)
	if rawURL == "" {
		return false
	}
	if !strings.Contains(rawURL, "://") {
		rawURL = "https://" + rawURL
	}
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return false
	}
	return parsed.User != nil || strings.Contains(parsed.Host, "@")
}

func policyStringFieldViolations(policy map[string]interface{}, keys ...string) []string {
	var violations []string
	for _, key := range keys {
		value, ok := policy[key]
		if !ok {
			continue
		}
		stringValue, ok := value.(string)
		if !ok {
			violations = append(violations, key+" must be a string")
			continue
		}
		if strings.TrimSpace(stringValue) == "" {
			violations = append(violations, key+" must be a non-empty string")
		}
	}
	return violations
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
	if len(digest) != 64 && len(digest) != 96 {
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

func sha256Bytes(value []byte) string {
	sum := sha256.Sum256(value)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func verifyTinfoil(p payload) (map[string]interface{}, error) {
	repo := policyString(p.Policy, "repo", "expected_repo")
	if repo == "" {
		return nil, errors.New("repo or expected_repo is required")
	}
	if err := requireReleaseDigestPolicy(p.Policy); err != nil {
		return nil, err
	}

	format := stringMapValue(p.Attestation, "format")
	body := stringMapValue(p.Attestation, "body")
	reportDigest := stringMapValue(p.Attestation, "report_digest")
	if format == "" || body == "" || reportDigest == "" {
		return nil, errors.New("attestation format, body, and report_digest are required")
	}

	doc := &attestation.Document{
		Format: attestation.PredicateType(format),
		Body:   body,
	}
	enclaveVerification, err := doc.Verify()
	if err != nil {
		return nil, fmt.Errorf("verifying enclave attestation report: %w", err)
	}

	codeMeasurement, digest, hardwareMeasurement, err := verifyCodeMeasurement(
		repo,
		enclaveVerification,
		p.Policy,
	)
	if err != nil {
		return nil, err
	}

	tlsBindingRequired, err := requireTinfoilTLSBindingPolicy(p.Policy)
	if err != nil {
		return nil, err
	}
	if tlsBindingRequired {
		if err := verifyTLSBinding(p, enclaveVerification); err != nil {
			return nil, err
		}
	}

	if err := verifyHPKEBinding(p.HPKEKeyConfigB64, enclaveVerification.HPKEPublicKey); err != nil {
		return nil, err
	}

	claims, err := groundTruthClaims(
		repo,
		digest,
		codeMeasurement,
		enclaveVerification.Measurement,
		hardwareMeasurement,
		enclaveVerification.TLSPublicKeyFP,
		enclaveVerification.HPKEPublicKey,
	)
	if err != nil {
		return nil, err
	}
	claims["attestation_format"] = format
	claims["attestation_report_digest"] = reportDigest
	claims["attested_hpke_public_key_hex"] = enclaveVerification.HPKEPublicKey
	claims["tls_public_key_fingerprint_sha256"] = enclaveVerification.TLSPublicKeyFP
	if err := enforceArtifactIdentityClaims(p.Policy, claims); err != nil {
		return nil, err
	}
	modelClaims, err := verifyTinfoilModelAttestations(p)
	if err != nil {
		return nil, err
	}
	if len(modelClaims) > 0 {
		claims["model_attestations"] = modelClaims
	}
	return claims, nil
}

func verifyTinfoilModelAttestations(p payload) (map[string]interface{}, error) {
	if !boolPolicyValue(p.Policy, "require_model_attestations", "requireModelAttestations") {
		if len(p.ModelAttestations) > 0 {
			return nil, errors.New("model_attestations are not allowed by policy")
		}
		return nil, nil
	}
	targets, err := tinfoilModelAttestationTargetPolicies(p.Policy)
	if err != nil {
		return nil, err
	}
	if len(targets) == 0 {
		return nil, errors.New("model_attestation_targets is required")
	}
	if len(p.ModelAttestations) != len(targets) {
		return nil, errors.New("model_attestations must cover every configured model target")
	}

	claimsByModel := map[string]interface{}{}
	seen := map[string]bool{}
	for _, modelAttestation := range p.ModelAttestations {
		modelID := strings.TrimSpace(modelAttestation.ModelID)
		if modelID == "" {
			return nil, errors.New("model_attestations model_id is required")
		}
		normalizedModelID := strings.ToLower(modelID)
		if seen[normalizedModelID] {
			return nil, fmt.Errorf("duplicate model_attestations entry for %s", modelID)
		}
		seen[normalizedModelID] = true
		targetPolicy, ok := targets[modelID]
		if !ok {
			return nil, fmt.Errorf("model_attestations contains unconfigured model %s", modelID)
		}
		modelClaims, err := verifyTinfoilModelAttestation(modelAttestation, targetPolicy, p.Policy)
		if err != nil {
			return nil, fmt.Errorf("model_attestations for %s: %w", modelID, err)
		}
		claimsByModel[modelID] = modelClaims
	}
	for modelID := range targets {
		if !seen[modelID] {
			return nil, fmt.Errorf("model_attestations is missing %s", modelID)
		}
	}
	return claimsByModel, nil
}

func verifyTinfoilModelAttestation(
	modelAttestation modelAttestation,
	targetPolicy map[string]interface{},
	parentPolicy map[string]interface{},
) (map[string]interface{}, error) {
	repo := policyString(targetPolicy, "repo", "expected_repo")
	if repo == "" {
		return nil, errors.New("repo or expected_repo is required")
	}
	if err := requireReleaseDigestPolicy(targetPolicy); err != nil {
		return nil, err
	}

	format := stringMapValue(modelAttestation.Attestation, "format")
	body := stringMapValue(modelAttestation.Attestation, "body")
	reportDigest := stringMapValue(modelAttestation.Attestation, "report_digest")
	if format == "" || body == "" || reportDigest == "" {
		return nil, errors.New("attestation format, body, and report_digest are required")
	}

	doc := &attestation.Document{
		Format: attestation.PredicateType(format),
		Body:   body,
	}
	enclaveVerification, err := doc.Verify()
	if err != nil {
		return nil, fmt.Errorf("verifying enclave attestation report: %w", err)
	}
	codeMeasurement, digest, hardwareMeasurement, err := verifyCodeMeasurement(
		repo,
		enclaveVerification,
		targetPolicy,
	)
	if err != nil {
		return nil, err
	}
	tlsBindingRequired, err := requireTinfoilTLSBindingForTarget(targetPolicy, parentPolicy)
	if err != nil {
		return nil, err
	}
	if tlsBindingRequired {
		if err := verifyTLSBindingForModelTarget(modelAttestation, targetPolicy, enclaveVerification); err != nil {
			return nil, err
		}
	}
	if err := verifyHPKEBinding(modelAttestation.HPKEKeyConfigB64, enclaveVerification.HPKEPublicKey); err != nil {
		return nil, err
	}
	claims, err := groundTruthClaims(
		repo,
		digest,
		codeMeasurement,
		enclaveVerification.Measurement,
		hardwareMeasurement,
		enclaveVerification.TLSPublicKeyFP,
		enclaveVerification.HPKEPublicKey,
	)
	if err != nil {
		return nil, err
	}
	claims["attestation_format"] = format
	claims["attestation_report_digest"] = reportDigest
	claims["attested_hpke_public_key_hex"] = enclaveVerification.HPKEPublicKey
	claims["tls_public_key_fingerprint_sha256"] = enclaveVerification.TLSPublicKeyFP
	claims["verification_steps"] = ehbpVerificationSteps
	if err := enforceArtifactIdentityClaims(targetPolicy, claims); err != nil {
		return nil, err
	}
	return claims, nil
}

func verifyPPQPrivate(p payload) (map[string]interface{}, error) {
	repo := policyString(p.Policy, "repo", "expected_repo")
	if repo == "" {
		return nil, errors.New("repo or expected_repo is required")
	}
	if err := requireArtifactIdentityPolicy(p.Policy); err != nil {
		return nil, err
	}
	bundleURL := firstNonEmpty(
		p.AttestationBundleURL,
		policyString(p.Policy, "attestation_bundle_url"),
	)
	if bundleURL == "" {
		return nil, errors.New("attestation_bundle_url is required")
	}

	groundTruth, err := fetchPPQPrivateGroundTruthWithHPKEBinding(p, bundleURL, repo)
	if err != nil {
		return nil, err
	}

	claims, err := groundTruthClaims(
		repo,
		groundTruth.Digest,
		groundTruth.CodeMeasurement,
		groundTruth.EnclaveMeasurement,
		groundTruth.HardwareMeasurement,
		groundTruth.TLSPublicKey,
		groundTruth.HPKEPublicKey,
	)
	if err != nil {
		return nil, err
	}
	claims["attestation_bundle_url"] = bundleURL
	claims["attested_hpke_public_key_hex"] = groundTruth.HPKEPublicKey
	claims["client_encryption_boundary"] = "routstr-tee-ehbp-proxy"
	claims["tls_public_key_fingerprint_sha256"] = groundTruth.TLSPublicKey
	selectedModelIDs, selectedModelViolations := normalizedPPQPrivateModelIDs(p.ModelIDs)
	if len(selectedModelViolations) > 0 {
		return nil, errors.New(strings.Join(selectedModelViolations, "; "))
	}
	claims["selected_model_ids"] = selectedModelIDs
	if err := enforceArtifactIdentityClaims(p.Policy, claims); err != nil {
		return nil, err
	}
	backendModelClaims, err := verifyTinfoilModelAttestations(p)
	if err != nil {
		return nil, err
	}
	if len(backendModelClaims) > 0 {
		claims["backend_model_attestations"] = backendModelClaims
	}
	return claims, nil
}

func fetchPPQPrivateGroundTruthWithHPKEBinding(
	p payload,
	bundleURL string,
	repo string,
) (*client.GroundTruth, error) {
	hpkePublicKey, err := ehbpPublicKeyHex(p.HPKEKeyConfigB64)
	if err != nil {
		return nil, err
	}

	var lastMismatch error
	for attempt := 0; attempt < ppqPrivateAttestationBundleAttempts; attempt++ {
		groundTruthJSON, err := fetchAndVerifyFromURLJSON(bundleURL, repo, nil)
		if err != nil {
			return nil, fmt.Errorf("verifying attestation bundle: %w", err)
		}

		var groundTruth client.GroundTruth
		if err := json.Unmarshal([]byte(groundTruthJSON), &groundTruth); err != nil {
			return nil, fmt.Errorf("decoding ground truth: %w", err)
		}
		if hpkePublicKey == strings.ToLower(groundTruth.HPKEPublicKey) {
			return &groundTruth, nil
		}
		lastMismatch = fmt.Errorf(
			"HPKE public key mismatch: key config has %s, attestation has %s",
			hpkePublicKey,
			groundTruth.HPKEPublicKey,
		)
	}
	if lastMismatch != nil {
		return nil, lastMismatch
	}
	return nil, errors.New("verifying attestation bundle did not return ground truth")
}

func verifyCodeMeasurement(
	repo string,
	enclaveVerification *attestation.Verification,
	policy map[string]interface{},
) (*attestation.Measurement, string, *attestation.HardwareMeasurement, error) {
	if enclaveVerification == nil || enclaveVerification.Measurement == nil {
		return nil, "", nil, errors.New("enclave measurement is required")
	}
	if _, err := enclaveMeasurementFingerprintsForPolicy(policy); err != nil {
		return nil, "", nil, err
	}

	digests, pinnedByPolicy, err := releaseDigestsForPolicy(policy)
	if err != nil {
		return nil, "", nil, err
	}
	if !pinnedByPolicy {
		return nil, "", nil, releaseDigestRequiredError()
	}
	sigstoreClient, err := sigstore.NewClient()
	if err != nil {
		return nil, "", nil, fmt.Errorf("creating Sigstore verifier: %w", err)
	}
	var verificationErrors []string
	for _, digest := range digests {
		codeMeasurement, hardwareMeasurement, err := verifyCodeMeasurementDigest(
			repo,
			digest,
			enclaveVerification,
			sigstoreClient,
		)
		if err == nil {
			return codeMeasurement, digest, hardwareMeasurement, nil
		}
		verificationErrors = append(
			verificationErrors,
			fmt.Sprintf("%s: %v", digest, err),
		)
	}

	return nil, "", nil, fmt.Errorf(
		"no release digest matched enclave measurement: %s",
		strings.Join(verificationErrors, "; "),
	)
}

func verifyCodeMeasurementDigest(
	repo string,
	digest string,
	enclaveVerification *attestation.Verification,
	sigstoreClient *sigstore.Client,
) (*attestation.Measurement, *attestation.HardwareMeasurement, error) {
	bundle, err := github.FetchAttestationBundle(repo, digest)
	if err != nil {
		return nil, nil, fmt.Errorf("fetching Sigstore attestation bundle: %w", err)
	}
	codeMeasurement, err := sigstoreClient.VerifyAttestation(bundle, repo, digest)
	if err != nil {
		return nil, nil, fmt.Errorf("verifying Sigstore attestation bundle: %w", err)
	}

	var hardwareMeasurement *attestation.HardwareMeasurement
	if enclaveVerification.Measurement.Type == attestation.TdxGuestV2 {
		hardwareMeasurements, err := sigstoreClient.LatestHardwareMeasurements()
		if err != nil {
			return nil, nil, fmt.Errorf("fetching hardware measurements: %w", err)
		}
		hardwareMeasurement, err = attestation.VerifyHardware(
			hardwareMeasurements,
			enclaveVerification.Measurement,
		)
		if err != nil {
			return nil, nil, fmt.Errorf("verifying hardware measurements: %w", err)
		}
	}

	if err := codeMeasurement.Equals(enclaveVerification.Measurement); err != nil {
		return nil, nil, fmt.Errorf("measurement mismatch: %w", err)
	}
	return codeMeasurement, hardwareMeasurement, nil
}

func releaseDigestForPolicy(policy map[string]interface{}) (string, bool, error) {
	digests, pinnedByPolicy, err := releaseDigestsForPolicy(policy)
	if err != nil || len(digests) == 0 {
		return "", pinnedByPolicy, err
	}
	return digests[0], pinnedByPolicy, nil
}

func releaseDigestsForPolicy(policy map[string]interface{}) ([]string, bool, error) {
	if err := validatePolicyStringValues(
		policy,
		"expected_release_digest",
		"release_digest",
		"allowed_release_digests",
		"allowed_release_digest",
	); err != nil {
		return nil, false, err
	}

	expectedDigests, err := normalizedPolicyValues(
		policy,
		normalizeReleaseDigest,
		"expected_release_digest",
		"release_digest",
	)
	if err != nil {
		return nil, false, err
	}
	allowedDigests, err := normalizedPolicyValues(
		policy,
		normalizeReleaseDigest,
		"allowed_release_digests",
		"allowed_release_digest",
	)
	if err != nil {
		return nil, false, err
	}
	if err := requireExpectedPolicyValuesAllowed(
		"release_digest",
		expectedDigests,
		allowedDigests,
	); err != nil {
		return nil, false, err
	}

	rawDigest := policyString(policy, "expected_release_digest", "release_digest")
	if rawDigest != "" {
		digest, err := normalizeReleaseDigest(rawDigest)
		if err != nil {
			return nil, false, err
		}
		return []string{digest}, true, nil
	}

	rawAllowedDigests, err := policyStringSlice(
		policy,
		"allowed_release_digests",
		"allowed_release_digest",
	)
	if err != nil {
		return nil, false, err
	}
	if len(rawAllowedDigests) == 0 {
		return nil, false, nil
	}

	var digests []string
	for _, rawAllowedDigest := range rawAllowedDigests {
		digest, err := normalizeReleaseDigest(rawAllowedDigest)
		if err != nil {
			return nil, false, err
		}
		if !containsString(digests, digest) {
			digests = append(digests, digest)
		}
	}
	return digests, true, nil
}

func requireArtifactIdentityPolicy(policy map[string]interface{}) error {
	_, releasePinned, err := releaseDigestsForPolicy(policy)
	if err != nil {
		return err
	}
	codeFingerprints, err := codeMeasurementFingerprintsForPolicy(policy)
	if err != nil {
		return err
	}
	if releasePinned || len(codeFingerprints) > 0 {
		return nil
	}
	return errors.New(
		"expected_release_digest, allowed_release_digests, " +
			"expected_code_measurement_fingerprint, or " +
			"allowed_code_measurement_fingerprints is required",
	)
}

func requireReleaseDigestPolicy(policy map[string]interface{}) error {
	_, releasePinned, err := releaseDigestsForPolicy(policy)
	if err != nil {
		return err
	}
	if releasePinned {
		return nil
	}
	return releaseDigestRequiredError()
}

func releaseDigestRequiredError() error {
	return errors.New(
		"expected_release_digest or allowed_release_digests is required " +
			"for Tinfoil code transparency verification",
	)
}

func enforceArtifactIdentityClaims(
	policy map[string]interface{},
	claims map[string]interface{},
) error {
	if err := requireArtifactIdentityPolicy(policy); err != nil {
		return err
	}

	releaseDigests, releasePinned, err := releaseDigestsForPolicy(policy)
	if err != nil {
		return err
	}
	if releasePinned {
		actualDigest, err := normalizeReleaseDigest(claimString(claims, "release_digest"))
		if err != nil || !containsString(releaseDigests, actualDigest) {
			return errors.New("release_digest claim does not match policy")
		}
	}

	codeFingerprints, err := codeMeasurementFingerprintsForPolicy(policy)
	if err != nil {
		return err
	}
	if len(codeFingerprints) > 0 {
		actualFingerprint, err := normalizeMeasurementFingerprint(
			claimString(claims, "code_measurement_fingerprint"),
		)
		if err != nil || !containsString(codeFingerprints, actualFingerprint) {
			return errors.New(
				"code_measurement_fingerprint claim does not match policy",
			)
		}
	}

	enclaveFingerprints, err := enclaveMeasurementFingerprintsForPolicy(policy)
	if err != nil {
		return err
	}
	if len(enclaveFingerprints) > 0 {
		actualFingerprint, err := normalizeMeasurementFingerprint(
			claimString(claims, "enclave_measurement_fingerprint"),
		)
		if err != nil || !containsString(enclaveFingerprints, actualFingerprint) {
			return errors.New(
				"enclave_measurement_fingerprint claim does not match policy",
			)
		}
	}
	return nil
}

func codeMeasurementFingerprintsForPolicy(
	policy map[string]interface{},
) ([]string, error) {
	return measurementFingerprintsForPolicy(
		policy,
		"code_measurement_fingerprint",
		[]string{
			"expected_code_measurement_fingerprint",
			"code_measurement_fingerprint",
		},
		[]string{
			"allowed_code_measurement_fingerprints",
			"allowed_code_measurement_fingerprint",
			"allowed_code_measurements",
		},
	)
}

func enclaveMeasurementFingerprintsForPolicy(
	policy map[string]interface{},
) ([]string, error) {
	return measurementFingerprintsForPolicy(
		policy,
		"enclave_measurement_fingerprint",
		[]string{
			"expected_enclave_measurement_fingerprint",
			"enclave_measurement_fingerprint",
		},
		[]string{
			"allowed_enclave_measurement_fingerprints",
			"allowed_enclave_measurement_fingerprint",
			"allowed_enclave_measurements",
		},
	)
}

func measurementFingerprintsForPolicy(
	policy map[string]interface{},
	label string,
	exactKeys []string,
	allowedKeys []string,
) ([]string, error) {
	keys := append(append([]string{}, exactKeys...), allowedKeys...)
	if err := validatePolicyStringValues(
		policy,
		keys...,
	); err != nil {
		return nil, err
	}

	expectedFingerprints, err := normalizedPolicyValues(
		policy,
		normalizeMeasurementFingerprint,
		exactKeys...,
	)
	if err != nil {
		return nil, err
	}
	allowedFingerprints, err := normalizedPolicyValues(
		policy,
		normalizeMeasurementFingerprint,
		allowedKeys...,
	)
	if err != nil {
		return nil, err
	}
	if err := requireExpectedPolicyValuesAllowed(
		label,
		expectedFingerprints,
		allowedFingerprints,
	); err != nil {
		return nil, err
	}

	rawFingerprint := policyString(policy, exactKeys...)
	if rawFingerprint != "" {
		fingerprint, err := normalizeMeasurementFingerprint(rawFingerprint)
		if err != nil {
			return nil, err
		}
		return []string{fingerprint}, nil
	}

	rawAllowedFingerprints, err := policyStringSlice(policy, allowedKeys...)
	if err != nil {
		return nil, err
	}
	var fingerprints []string
	for _, rawAllowedFingerprint := range rawAllowedFingerprints {
		fingerprint, err := normalizeMeasurementFingerprint(rawAllowedFingerprint)
		if err != nil {
			return nil, err
		}
		if fingerprint != "" && !containsString(fingerprints, fingerprint) {
			fingerprints = append(fingerprints, fingerprint)
		}
	}
	return fingerprints, nil
}

type policyValueNormalizer func(string) (string, error)

func normalizedPolicyValues(
	policy map[string]interface{},
	normalize policyValueNormalizer,
	keys ...string,
) ([]string, error) {
	rawValues, err := policyStringValuesForPresentKeys(policy, keys...)
	if err != nil {
		return nil, err
	}
	var values []string
	for _, rawValue := range rawValues {
		value, err := normalize(rawValue)
		if err != nil {
			return nil, err
		}
		if !containsString(values, value) {
			values = append(values, value)
		}
	}
	return values, nil
}

func requireExpectedPolicyValuesAllowed(
	label string,
	expectedValues []string,
	allowedValues []string,
) error {
	if len(expectedValues) <= 1 && (len(expectedValues) == 0 || len(allowedValues) == 0) {
		return nil
	}
	if len(expectedValues) > 1 {
		return fmt.Errorf("%s expected aliases must match", label)
	}
	if !containsString(allowedValues, expectedValues[0]) {
		return fmt.Errorf("%s expected value must be allowed", label)
	}
	return nil
}

func containsString(values []string, value string) bool {
	for _, candidate := range values {
		if candidate == value {
			return true
		}
	}
	return false
}

func containsStringFold(values []string, value string) bool {
	value = strings.TrimSpace(value)
	if value == "" {
		return false
	}
	for _, candidate := range values {
		if strings.EqualFold(strings.TrimSpace(candidate), value) {
			return true
		}
	}
	return false
}

func claimString(claims map[string]interface{}, key string) string {
	if claims == nil {
		return ""
	}
	value, ok := claims[key]
	if !ok || value == nil {
		return ""
	}
	if stringValue, ok := value.(string); ok {
		return strings.TrimSpace(stringValue)
	}
	return ""
}

func normalizeReleaseDigest(rawDigest string) (string, error) {
	digest := strings.ToLower(strings.TrimSpace(rawDigest))
	digest = strings.TrimPrefix(digest, "sha256:")
	if len(digest) != 64 {
		return "", fmt.Errorf("release digest must be a sha256 hex digest: %q", rawDigest)
	}
	if _, err := hex.DecodeString(digest); err != nil {
		return "", fmt.Errorf("release digest must be valid hex: %w", err)
	}
	if isRepeatedHexPlaceholder(digest) {
		return "", fmt.Errorf("release digest must not be a placeholder digest: %q", rawDigest)
	}
	return digest, nil
}

func normalizeMeasurementFingerprint(rawFingerprint string) (string, error) {
	fingerprint := strings.ToLower(strings.TrimSpace(rawFingerprint))
	hasSHA256Prefix := strings.HasPrefix(fingerprint, "sha256:")
	if hasSHA256Prefix {
		fingerprint = strings.TrimPrefix(fingerprint, "sha256:")
	}
	if len(fingerprint) == 64 {
		if _, err := hex.DecodeString(fingerprint); err != nil {
			return "", fmt.Errorf("measurement fingerprint must be valid hex: %w", err)
		}
		if isRepeatedHexPlaceholder(fingerprint) {
			return "", fmt.Errorf(
				"measurement fingerprint must not be a placeholder digest: %q",
				rawFingerprint,
			)
		}
		return fingerprint, nil
	}
	if hasSHA256Prefix {
		return "", fmt.Errorf(
			"measurement fingerprint with sha256 prefix must be a sha256 hex digest: %q",
			rawFingerprint,
		)
	}
	if len(fingerprint) == 96 {
		decoded, err := hex.DecodeString(fingerprint)
		if err != nil {
			return "", fmt.Errorf("measurement fingerprint must be valid hex: %w", err)
		}
		if isRepeatedHexPlaceholder(fingerprint) {
			return "", fmt.Errorf(
				"measurement fingerprint must not be a placeholder digest: %q",
				rawFingerprint,
			)
		}
		sum := sha256.Sum256(decoded)
		return hex.EncodeToString(sum[:]), nil
	}
	return "", fmt.Errorf(
		"measurement fingerprint must be a sha256 hex digest or native SEV-SNP measurement: %q",
		rawFingerprint,
	)
}

func verifyTLSBinding(p payload, enclaveVerification *attestation.Verification) error {
	if boolPolicyValue(p.Policy, "skip_tls_key_binding") {
		return errors.New("skip_tls_key_binding is not allowed")
	}
	host, err := verifierHost(p)
	if err != nil {
		return err
	}
	tlsKey, err := attestation.TLSPublicKey(host, false)
	if err != nil {
		return fmt.Errorf("fetching TLS public key: %w", err)
	}
	if tlsKey != enclaveVerification.TLSPublicKeyFP {
		return fmt.Errorf(
			"TLS public key mismatch: connection has %s, attestation has %s",
			tlsKey,
			enclaveVerification.TLSPublicKeyFP,
		)
	}
	return nil
}

func tinfoilTLSBindingRequired(policy map[string]interface{}) bool {
	required, err := requireTinfoilTLSBindingPolicy(policy)
	if err != nil {
		return true
	}
	return required
}

func requireTinfoilTLSBindingPolicy(policy map[string]interface{}) (bool, error) {
	if boolPolicyValue(policy, "skip_tls_key_binding") {
		return true, errors.New("skip_tls_key_binding is not allowed")
	}
	for _, key := range []string{"transport_security", "tinfoil_transport_security"} {
		value, ok := policy[key]
		if !ok {
			continue
		}
		stringValue, ok := value.(string)
		if !ok {
			continue
		}
		if strings.EqualFold(strings.TrimSpace(stringValue), "ehbp") {
			return false, nil
		}
	}
	return true, nil
}

func requireTinfoilTLSBindingForTarget(
	targetPolicy map[string]interface{},
	parentPolicy map[string]interface{},
) (bool, error) {
	targetRequired, err := requireTinfoilTLSBindingPolicy(targetPolicy)
	if err != nil {
		return true, err
	}
	if !targetRequired {
		return false, nil
	}
	parentRequired, err := requireTinfoilTLSBindingPolicy(parentPolicy)
	if err != nil {
		return true, err
	}
	return parentRequired, nil
}

func verifyTLSBindingForModelTarget(
	modelAttestation modelAttestation,
	targetPolicy map[string]interface{},
	enclaveVerification *attestation.Verification,
) error {
	if boolPolicyValue(targetPolicy, "skip_tls_key_binding") {
		return errors.New("skip_tls_key_binding is not allowed")
	}
	host := policyString(targetPolicy, "host", "expected_enclave_host")
	if host == "" {
		identity, violation := remoteURLIdentity(
			modelAttestation.AttestationURL,
			"model_attestation attestation_url",
		)
		if violation != "" {
			return errors.New(violation)
		}
		host = identity.host
	}
	normalizedHost, violation := hostIdentityHost(host, "model_attestation host")
	if violation != "" {
		return errors.New(violation)
	}
	if normalizedHost == "" {
		return errors.New("model_attestation host is required")
	}
	tlsKey, err := attestation.TLSPublicKey(normalizedHost, false)
	if err != nil {
		return fmt.Errorf("fetching TLS public key: %w", err)
	}
	if tlsKey != enclaveVerification.TLSPublicKeyFP {
		return fmt.Errorf(
			"TLS public key mismatch: connection has %s, attestation has %s",
			tlsKey,
			enclaveVerification.TLSPublicKeyFP,
		)
	}
	return nil
}

func verifyHPKEBinding(keyConfigB64 string, expectedHPKEPublicKey string) error {
	hpkePublicKey, err := ehbpPublicKeyHex(keyConfigB64)
	if err != nil {
		return err
	}
	if hpkePublicKey != strings.ToLower(expectedHPKEPublicKey) {
		return fmt.Errorf(
			"HPKE public key mismatch: key config has %s, attestation has %s",
			hpkePublicKey,
			expectedHPKEPublicKey,
		)
	}
	return nil
}

func groundTruthClaims(
	repo string,
	digest string,
	codeMeasurement *attestation.Measurement,
	enclaveMeasurement *attestation.Measurement,
	hardwareMeasurement *attestation.HardwareMeasurement,
	tlsPublicKey string,
	hpkePublicKey string,
) (map[string]interface{}, error) {
	codeMeasurementFingerprint, err := measurementFingerprint(
		codeMeasurement,
		hardwareMeasurement,
		enclaveMeasurement,
	)
	if err != nil {
		return nil, fmt.Errorf("code_measurement_fingerprint: %w", err)
	}
	enclaveMeasurementFingerprint, err := measurementFingerprint(
		enclaveMeasurement,
		hardwareMeasurement,
		enclaveMeasurement,
	)
	if err != nil {
		return nil, fmt.Errorf("enclave_measurement_fingerprint: %w", err)
	}
	claims := map[string]interface{}{
		"repo":                            repo,
		"release_digest":                  digest,
		"code_measurement":                measurementClaims(codeMeasurement),
		"enclave_measurement":             measurementClaims(enclaveMeasurement),
		"code_measurement_fingerprint":    codeMeasurementFingerprint,
		"enclave_measurement_fingerprint": enclaveMeasurementFingerprint,
		"tls_public_key":                  tlsPublicKey,
		"hpke_public_key":                 hpkePublicKey,
	}
	if hardwareMeasurement != nil {
		claims["hardware_measurement"] = map[string]string{
			"id":    hardwareMeasurement.ID,
			"mrtd":  hardwareMeasurement.MRTD,
			"rtmr0": hardwareMeasurement.RTMR0,
		}
	}
	return claims, nil
}

func measurementClaims(m *attestation.Measurement) map[string]interface{} {
	if m == nil {
		return nil
	}
	return map[string]interface{}{
		"type":      string(m.Type),
		"registers": m.Registers,
	}
}

func measurementFingerprint(
	measurement *attestation.Measurement,
	hardwareMeasurement *attestation.HardwareMeasurement,
	targetMeasurement *attestation.Measurement,
) (string, error) {
	if measurement == nil || targetMeasurement == nil {
		return "", errors.New("measurement is required")
	}
	fingerprint, err := attestation.Fingerprint(
		measurement,
		hardwareMeasurement,
		targetMeasurement.Type,
	)
	if err != nil {
		return "", err
	}
	return normalizeMeasurementFingerprint(fingerprint)
}

func ehbpPublicKeyHex(keyConfigB64 string) (string, error) {
	data, err := base64.StdEncoding.DecodeString(keyConfigB64)
	if err != nil {
		return "", fmt.Errorf("decoding EHBP key config: %w", err)
	}
	if len(data) < 5 {
		return "", errors.New("invalid EHBP key config: too short")
	}
	if data[0] != 0 {
		return "", fmt.Errorf("invalid EHBP key config: unsupported key id %d", data[0])
	}
	kemID := binary.BigEndian.Uint16(data[1:3])
	publicKeyLen, ok := publicKeyLength(kemID)
	if !ok {
		return "", fmt.Errorf("invalid EHBP key config: unsupported KEM 0x%04x", kemID)
	}
	publicKeyStart := 3
	publicKeyEnd := publicKeyStart + publicKeyLen
	if len(data) < publicKeyEnd+2 {
		return "", errors.New("invalid EHBP key config: truncated public key")
	}
	suitesLen := int(binary.BigEndian.Uint16(data[publicKeyEnd : publicKeyEnd+2]))
	if suitesLen < 4 || suitesLen%4 != 0 {
		return "", errors.New("invalid EHBP key config: malformed cipher suite list")
	}
	suitesStart := publicKeyEnd + 2
	if len(data) != suitesStart+suitesLen {
		return "", errors.New("invalid EHBP key config: truncated or trailing data")
	}
	kdfID := binary.BigEndian.Uint16(data[suitesStart : suitesStart+2])
	aeadID := binary.BigEndian.Uint16(data[suitesStart+2 : suitesStart+4])
	if kemID != 0x0020 || kdfID != 0x0001 || aeadID != 0x0002 {
		return "", fmt.Errorf(
			"invalid EHBP key config: unsupported suite KEM=0x%04x KDF=0x%04x AEAD=0x%04x",
			kemID,
			kdfID,
			aeadID,
		)
	}
	return fmt.Sprintf("%x", data[publicKeyStart:publicKeyEnd]), nil
}

func publicKeyLength(kemID uint16) (int, bool) {
	switch kemID {
	case 0x0010:
		return 65, true
	case 0x0011:
		return 97, true
	case 0x0012:
		return 133, true
	case 0x0020:
		return 32, true
	case 0x0021:
		return 56, true
	default:
		return 0, false
	}
}

func verifierHost(p payload) (string, error) {
	var raw string
	for _, check := range []struct {
		label string
		value string
	}{
		{"enclave_host", policyString(p.Policy, "enclave_host")},
		{"expected_enclave_host", policyString(p.Policy, "expected_enclave_host")},
		{"origin", p.Origin},
		{"base_url", p.BaseURL},
	} {
		if strings.TrimSpace(check.value) == "" {
			continue
		}
		var violation string
		var host string
		if check.label == "base_url" || check.label == "origin" {
			host, violation = remoteURLHost(check.value, check.label)
		} else {
			host, violation = hostIdentityHost(check.value, check.label)
		}
		if violation != "" {
			return "", errors.New(violation)
		}
		raw = host
		break
	}
	if raw == "" {
		return "", errors.New("base_url, origin, or enclave_host is required")
	}
	return raw, nil
}

func stringMapValue(m map[string]interface{}, key string) string {
	if m == nil {
		return ""
	}
	value, ok := m[key]
	if !ok {
		return ""
	}
	if stringValue, ok := value.(string); ok {
		return strings.TrimSpace(stringValue)
	}
	return ""
}

func policyString(policy map[string]interface{}, keys ...string) string {
	for _, key := range keys {
		if value := stringMapValue(policy, key); value != "" {
			return value
		}
	}
	return ""
}

func policyStringSlice(policy map[string]interface{}, keys ...string) ([]string, error) {
	for _, key := range keys {
		value, ok := policy[key]
		if !ok {
			continue
		}
		switch typed := value.(type) {
		case string:
			trimmed := strings.TrimSpace(typed)
			if trimmed == "" {
				return nil, errors.New("policy values must be non-empty strings")
			}
			return []string{trimmed}, nil
		case []string:
			var values []string
			for _, item := range typed {
				trimmed := strings.TrimSpace(item)
				if trimmed == "" {
					return nil, errors.New("policy list entries must be non-empty strings")
				}
				values = append(values, trimmed)
			}
			if len(values) > 0 {
				return values, nil
			}
		case []interface{}:
			var values []string
			for _, item := range typed {
				raw, ok := item.(string)
				trimmed := strings.TrimSpace(raw)
				if !ok || trimmed == "" {
					return nil, errors.New("policy list entries must be non-empty strings")
				}
				values = append(values, trimmed)
			}
			if len(values) > 0 {
				return values, nil
			}
		case nil:
			return nil, errors.New("policy value must be a string or string array")
		default:
			return nil, errors.New("policy list value must be a string or string array")
		}
	}
	return nil, nil
}

func tinfoilModelAttestationTargetPolicies(
	policy map[string]interface{},
) (map[string]map[string]interface{}, error) {
	var parsedValues []map[string]map[string]interface{}
	for _, key := range []string{
		"model_attestation_targets",
		"modelAttestationTargets",
		"model_enclave_bindings",
		"modelEnclaveBindings",
	} {
		rawTargets, ok := policy[key]
		if !ok {
			continue
		}
		targets, ok := rawTargets.(map[string]interface{})
		if !ok || len(targets) == 0 {
			return nil, errors.New("model_attestation_targets must be a non-empty object")
		}
		parsed := map[string]map[string]interface{}{}
		seenModelIDs := map[string]bool{}
		for rawModelID, rawTarget := range targets {
			modelID := strings.TrimSpace(rawModelID)
			if modelID == "" {
				return nil, errors.New("model_attestation_targets keys must be non-empty model strings")
			}
			normalizedModelID := strings.ToLower(modelID)
			if seenModelIDs[normalizedModelID] {
				return nil, fmt.Errorf(
					"model_attestation_targets contains duplicate model target %s",
					normalizedModelID,
				)
			}
			seenModelIDs[normalizedModelID] = true
			targetPolicy, ok := rawTarget.(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf("model_attestation_targets for %s must be an object", modelID)
			}
			if err := requireTinfoilModelTargetPolicy(targetPolicy); err != nil {
				return nil, fmt.Errorf("model_attestation_targets for %s: %w", modelID, err)
			}
			parsed[modelID] = targetPolicy
		}
		parsedValues = append(parsedValues, parsed)
	}
	if len(parsedValues) == 0 {
		return nil, nil
	}
	first := parsedValues[0]
	for _, parsed := range parsedValues[1:] {
		if !sameTinfoilModelAttestationTargetPolicies(first, parsed) {
			return nil, errors.New("model_attestation_targets aliases must match")
		}
	}
	return first, nil
}

func sameTinfoilModelAttestationTargetPolicies(
	left map[string]map[string]interface{},
	right map[string]map[string]interface{},
) bool {
	if len(left) != len(right) {
		return false
	}
	leftJSON, err := json.Marshal(left)
	if err != nil {
		return false
	}
	rightJSON, err := json.Marshal(right)
	if err != nil {
		return false
	}
	return bytes.Equal(leftJSON, rightJSON)
}

func requireTinfoilModelTargetPolicy(policy map[string]interface{}) error {
	var violations []string
	violations = append(
		violations,
		policyStringFieldViolations(
			policy,
			"host",
			"expected_enclave_host",
			"attestation_url",
			"hpke_keys_url",
			"repo",
			"expected_repo",
		)...,
	)
	violations = append(
		violations,
		policyAliasViolations(
			policy,
			"enclave host",
			"host",
			"expected_enclave_host",
		)...,
	)
	violations = append(
		violations,
		policyAliasViolations(
			policy,
			"repo",
			"repo",
			"expected_repo",
		)...,
	)
	for _, check := range []struct {
		label string
		value string
	}{
		{"attestation_url", policyString(policy, "attestation_url")},
		{"hpke_keys_url", policyString(policy, "hpke_keys_url")},
	} {
		if violation := remoteURLViolation(check.value, check.label); violation != "" {
			violations = append(violations, violation)
		}
	}
	host := policyString(policy, "host", "expected_enclave_host")
	if host != "" {
		if violation := hostIdentityViolation(host, "host"); violation != "" {
			violations = append(violations, violation)
		}
	}
	if host == "" && policyString(policy, "attestation_url") == "" {
		violations = append(violations, "host or attestation_url is required")
	}
	if policyString(policy, "repo", "expected_repo") == "" {
		violations = append(violations, "repo or expected_repo is required")
	}
	if err := requireArtifactIdentityPolicy(policy); err != nil {
		violations = append(violations, err.Error())
	}
	if len(violations) > 0 {
		return errors.New(strings.Join(violations, "; "))
	}
	return nil
}

func validatePolicyStringValues(policy map[string]interface{}, keys ...string) error {
	_, err := policyStringValuesForPresentKeys(policy, keys...)
	return err
}

func policyAliasViolations(
	policy map[string]interface{},
	label string,
	keys ...string,
) []string {
	values, err := policyStringValuesForPresentKeys(policy, keys...)
	if err != nil || len(values) == 0 {
		return nil
	}
	seen := map[string]bool{}
	for _, value := range values {
		normalized := strings.TrimSpace(value)
		if normalized != "" {
			seen[normalized] = true
		}
	}
	if len(seen) > 1 {
		return []string{label + " aliases must match"}
	}
	return nil
}

func policyStringValuesForPresentKeys(policy map[string]interface{}, keys ...string) ([]string, error) {
	var values []string
	for _, key := range keys {
		value, ok := policy[key]
		if !ok {
			continue
		}
		switch typed := value.(type) {
		case string:
			trimmed := strings.TrimSpace(typed)
			if trimmed == "" {
				return nil, errors.New("policy values must be non-empty strings")
			}
			values = append(values, trimmed)
		case []string:
			for _, item := range typed {
				trimmed := strings.TrimSpace(item)
				if trimmed == "" {
					return nil, errors.New("policy list entries must be non-empty strings")
				}
				values = append(values, trimmed)
			}
		case []interface{}:
			for _, item := range typed {
				raw, ok := item.(string)
				trimmed := strings.TrimSpace(raw)
				if !ok || trimmed == "" {
					return nil, errors.New("policy list entries must be non-empty strings")
				}
				values = append(values, trimmed)
			}
		case nil:
			return nil, errors.New("policy value must be a string or string array")
		default:
			return nil, errors.New("policy list value must be a string or string array")
		}
	}
	return values, nil
}

func boolPolicyValue(policy map[string]interface{}, keys ...string) bool {
	if policy == nil {
		return false
	}
	for _, key := range keys {
		value, ok := policy[key]
		if !ok {
			continue
		}
		boolValue, ok := value.(bool)
		if ok && boolValue {
			return true
		}
	}
	return false
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}
