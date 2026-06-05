# Routstr Tinfoil/PPQ Verifier

This verifier is an optional isolated command for Routstr's
`routstr-confidential-verifier-v1` contract. Build it as a standalone binary,
pin the binary digest in `provider_settings.confidentiality.policy`, and use it
as the `verifier_command` for:

- `mode: "tinfoil"`
- `mode: "ppq-private-tee"`

It uses `github.com/tinfoilsh/tinfoil-go/verifier` in a separate Go module so
the Python service does not inherit the Tinfoil SDK dependency graph.
The command validates payload-bound policy/evidence digests, nonce presence,
optional verifier artifact digest pins, and credential-bearing URL inputs before
it performs TLS checks or attestation-bundle fetches.
The same binary is used for both modes, but successful results emit
mode-specific verifier identities: `routstr-tinfoil-go-verifier` for Tinfoil
and `routstr-ppq-private-go-verifier` for PPQ private TEE. This keeps public
attestation/status output aligned with the integration being verified while
reusing the Tinfoil-shaped EHBP and backend-attestation implementation.

## Build

```sh
cd verifiers/tinfoil-go
go build -trimpath -ldflags="-s -w" -o routstr-tinfoil-go-verifier .
sha256sum routstr-tinfoil-go-verifier
```

Use the resulting `sha256:<hex>` value as `verifier_command_digest`.

## Tinfoil Policy Shape

```json
{
  "repo": "tinfoilsh/confidential-model-router",
  "expected_release_digest": "sha256:<tinfoil release digest>",
  "verifier_command": ["/opt/routstr/bin/routstr-tinfoil-go-verifier"],
  "verifier_command_digest": "sha256:<binary digest>"
}
```

The verifier consumes the attestation document and EHBP key config fetched by
Routstr, verifies the hardware attestation report and Sigstore code
measurements, checks measurement equality, verifies the live TLS public key
binding, and checks that the attested HPKE key matches the EHBP key config.
When `expected_release_digest` is set, the verifier fetches and verifies the
Sigstore bundle for that exact release digest instead of using the latest
release. When `allowed_release_digests` is set, every listed digest is treated
as an exact candidate and verification succeeds only when one candidate's
Sigstore measurements match the live enclave evidence. Tinfoil policies must
pin one of those release-digest fields; code-measurement fingerprints can be
used as an additional check, but not as the only code-transparency input.
Release digests may be configured as raw 64-hex SHA-256 values or
`sha256:<64 hex>`. Measurement fingerprints may use those SHA-256 forms, or
the native unprefixed 96-hex SEV-SNP measurement emitted by Tinfoil evidence;
the verifier normalizes native SEV-SNP measurements to SHA-256 fingerprints
before comparing claims.

When `require_model_attestations` is enabled, payload `model_ids` must exactly
match the `model_attestation_targets` keys. Duplicate target keys after
trimming and conflicting target aliases are rejected before model proof can be
emitted.

## PPQ Private Policy Shape

```json
{
  "repo": "tinfoilsh/confidential-model-router",
  "attestation_bundle_url": "https://api.ppq.ai/private",
  "expected_release_digest": "sha256:<PPQ private deployment digest>",
  "require_model_attestations": true,
  "model_attestation_targets": {
    "private/gpt-oss-120b": {
      "host": "gpt-oss-120b-1.inf10.tinfoil.sh",
      "repo": "tinfoilsh/confidential-gpt-oss-120b",
      "expected_release_digest": "sha256:<backend model release digest>"
    }
  },
  "verifier_command": ["/opt/routstr/bin/routstr-tinfoil-go-verifier"],
  "verifier_command_digest": "sha256:<binary digest>"
}
```

The verifier fetches and verifies the Tinfoil-style attestation bundle from
`attestation_bundle_url`, then checks the attested HPKE key against the EHBP key
config fetched by Routstr. The current live PPQ private endpoint has verified
with `repo: "tinfoilsh/confidential-model-router"` and deployment digest
`sha256:ebb921d8572bc8f6b4e04812a7e65bffcf6ac1a2fabe8a55115a13d61e901b85`;
reconfirm the intended production provenance identity before rollout. PPQ
private policies must also pin either
`expected_release_digest` / `allowed_release_digests` or
`expected_code_measurement_fingerprint` /
`allowed_code_measurement_fingerprints`. Measurement pins may be raw 64-hex
SHA-256 values, `sha256:<64 hex>`, or native unprefixed 96-hex SEV-SNP
measurements; native SEV-SNP measurements are normalized to SHA-256
fingerprints before comparison. If PPQ serves a verified bundle whose attested
HPKE key does not match the fetched EHBP key config, the verifier retries
boundedly and still fails closed unless it finds a coherent attestation/key pair.
Routstr implements PPQ private EHBP forwarding directly, so PPQ private policies
do not require a local proxy binary artifact. Use
`scripts/generate_ppq_private_policy.py` to derive the policy from current
`confidential-inference` attestation results.
For selected-model routing, PPQ private policies must also enable
`require_model_attestations` and provide exact `model_attestation_targets` for
every selected `private/*` model. Routstr fetches each backend Tinfoil model
attestation and sends those payloads as `model_attestations`; the verifier
validates them and emits route/public proof under `backend_model_attestations`.
Remote evidence URLs such as `base_url`, `origin`, and `attestation_bundle_url`
must be absolute HTTPS URLs and must not contain URL userinfo; API credentials
belong in normal provider auth, not in verifier evidence URLs. Evidence URLs
and explicit enclave-host pins must resolve to the same host as the configured
provider `base_url`; the verifier must not attest one host while Routstr routes
to another. For `ppq-private-tee`, both `base_url` and
`attestation_bundle_url` must include an explicit `private` path segment before
the verifier fetches evidence.

This command does not verify Privatemode. Privatemode needs a separate verifier
that understands Contrast manifests, mesh CA handoff, secret-service and
AI-worker attestations, NVIDIA GPU attestation/OCSP policy, and key release.
