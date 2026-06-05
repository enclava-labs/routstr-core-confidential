# Routstr TEE Attestation Command

This command implements Routstr's `routstr-tee-attestation-request-v1`
contract. It is intended for `ROUTSTR_TEE_ATTESTATION_COMMAND` and should be
digest-pinned with `ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST`.

## Build

```sh
cd verifiers/routstr-tee-attest-go
go build -trimpath -ldflags="-s -w" -o routstr-tee-attest-go .
sha256sum routstr-tee-attest-go
```

Use the resulting `sha256:<hex>` value as
`ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST`.

## SEV-SNP

For `sev_snp_quote`, `sev_snp_report`, or `sev_guest_v2`, this command opens the
Linux SEV guest device and requests a fresh report with Routstr's
`tee_report_data_hex` in REPORT_DATA.

Defaults:

- device path: `/dev/sev-guest`
- VMPL: `0`

Optional policy fields:

```json
{
  "sev_guest_device_path": "/dev/sev-guest",
  "sev_snp_vmpl": 0
}
```

The device path can also be set with `ROUTSTR_SEV_GUEST_DEVICE`.

## TDX

For `tdx_quote` or `tdx_guest_v2`, this command requires a pinned external TDX
QuoteV4 generator. Linux `TDX_CMD_GET_REPORT0` returns a TDREPORT0; the
Routstr verifier expects remotely verifiable QuoteV4 evidence, so this command
does not treat TDREPORT0 as sufficient.

Policy fields:

```json
{
  "tdx_quote_command": ["/usr/local/bin/tdx-quote-wrapper"],
  "tdx_quote_command_digest": "sha256:<hex>",
  "tdx_quote_artifact_path": "/usr/local/bin/tdx-quote-wrapper"
}
```

`tdx_quote_command_digest` must be a full `sha256:<64 hex>` digest and is
validated before the command hashes or executes the quote generator artifact.
If `tdx_quote_artifact_path` is set, it must be the executable or an explicit
argument in `tdx_quote_command`; otherwise the digest pin could cover a
different file than the command that actually runs.

The external command receives `routstr-tdx-quote-request-v1` JSON on stdin with
`tee_report_data_hex`, `tee_report_data_digest`, routing-policy digest, public
key digests, and verifier artifact digests. It must return:

```json
{
  "schema_version": "routstr-tee-attestation-document-v1",
  "attestation_document_format": "tdx_quote",
  "attestation_document_b64": "<raw QuoteV4 bytes as base64>",
  "tee_report_data_hex": "<same value from request>",
  "tee_report_data_digest": "<same value from request>"
}
```

The returned quote is then verified by `verifiers/routstr-tee-go`.
