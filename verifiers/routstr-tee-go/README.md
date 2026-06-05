# Routstr TEE Verifier

This verifier is an optional isolated command for Routstr's
`routstr-tee-verifier-v1` contract. Build it as a standalone binary, pin the
binary digest in `ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST`, and configure it as
`ROUTSTR_TEE_VERIFIER_COMMAND` when `CONFIDENTIAL_ROUTING_MODE=required`.

It verifies TDX/SEV-SNP attestation evidence using
`github.com/tinfoilsh/tinfoil-go/verifier`, checks that the quote report data
matches Routstr's `tee_report_data_hex`, enforces the configured Routstr code
measurement policy, and emits the claims required by the Python routing gate.

## Build

```sh
cd verifiers/routstr-tee-go
go build -trimpath -ldflags="-s -w" -o routstr-tee-go-verifier .
sha256sum routstr-tee-go-verifier
```

Use the resulting `sha256:<hex>` value as `ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST`.

## Runtime Shape

Routstr sends this command a `routstr-tee-verifier-v1` JSON payload on stdin.
The deployment must provide either a static attestation document path for test
fixtures/offline validation, or a digest-pinned attestation command for live
nonce-bound quote generation:

- `ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT` as `tdx_quote` or `sev_snp_quote`
- `ROUTSTR_TEE_ATTESTATION_DOCUMENT_PATH`, or
  `ROUTSTR_TEE_ATTESTATION_COMMAND` plus
  `ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST`
- `ROUTSTR_TEE_PUBLIC_KEY` or `ROUTSTR_TEE_PUBLIC_KEY_PATH`
- `ROUTSTR_TEE_HPKE_KEY_CONFIG_B64` or `ROUTSTR_TEE_HPKE_KEY_CONFIG_PATH`
- `ROUTSTR_TEE_VERIFIER_POLICY_JSON`

The policy must pin the expected Routstr code measurement as a raw 64-hex
SHA-256 fingerprint or a prefixed `sha256:<64 hex>` value:

```json
{
  "expected_routstr_code_measurement": "sha256:<measurement fingerprint>"
}
```

or an allow-list:

```json
{
  "allowed_routstr_code_measurements": [
    "sha256:<measurement fingerprint>"
  ]
}
```

The verifier normalizes raw fingerprints to prefixed SHA-256 digests in its
output, and malformed placeholder pins fail before a verified result is emitted.

`tee_report_data_hex` is the exact 64-byte value the deployment should place
into the TDX/SEV report data field when generating the quote. The verifier
recomputes that value from the routing policy digest, public key digest, HPKE
key config digest, HPKE public key digest, and verifier nonce, then checks that
the hardware report contains the same value. Verified output includes both
`tee_report_nonce` and `tee_report_nonce_digest` so remote auditors can
recompute the published report-data binding from public proof claims.

For live deployments, prefer `ROUTSTR_TEE_ATTESTATION_COMMAND`. The
`verifiers/routstr-tee-attest-go` command implements this contract for SEV-SNP
directly and can bridge TDX to a digest-pinned QuoteV4 generator. Routstr sends
it a `routstr-tee-attestation-request-v1` JSON payload containing
`tee_report_data_hex` and `tee_report_data_digest`; the command must generate
fresh evidence inside the TEE and return:

```json
{
  "schema_version": "routstr-tee-attestation-document-v1",
  "attestation_document_format": "tdx_quote",
  "attestation_document_b64": "<raw quote bytes as base64>",
  "tee_report_data_hex": "<same value from request>",
  "tee_report_data_digest": "<same value from request>"
}
```

The command should be a small platform-specific wrapper around the TDX or
SEV-SNP quote generator. Pin the wrapper or binary digest with
`ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST`; set
`ROUTSTR_TEE_ATTESTATION_ARTIFACT_PATH` when the command invokes an interpreter
and the script, not the interpreter, is the artifact to hash.

This command does not fetch provider attestations. Use `verifiers/tinfoil-go`
for Tinfoil/PPQ private providers and `verifiers/privatemode-go` for
Privatemode.
