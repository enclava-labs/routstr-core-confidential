# Routstr Privatemode Verifier

This is an isolated verifier command for Routstr's `privatemode` provider mode.
It reads the `routstr-confidential-verifier-v1` JSON payload on stdin and writes
the verifier result JSON on stdout.

Build with the Contrast SDK build tag:

```bash
go test -tags contrast_unstable_api ./...
go build -tags contrast_unstable_api -trimpath -ldflags='-s -w' -o routstr-privatemode-go-verifier .
sha256sum routstr-privatemode-go-verifier
```

The command uses the official Privatemode public module and Contrast SDK flow to:

- load a pinned manifest from `manifest_path`, `manifest_b64`, `manifest_url`, or
  the official CDN
- verify the Coordinator attestation against that exact manifest
- parse and bind the attested mesh CA
- complete the secret-service key exchange and verify the mesh certificate and
  signature
- parse the attested manifest's workload policies and require configured
  `expected_workload_sans` or `expected_workload_ids` to be present
- require `model_workload_bindings` to map selected Routstr model IDs to those
  workload SANs or IDs, and emit `model_workload_binding_digest`
- prove local prompt encryption with the released inference secret
- bind the strict NVIDIA OCSP policy header MAC to the same secret
- derive `key_release_binding` from the manifest, mesh CA, secret-service
  certificate, inference-secret, NVIDIA OCSP MAC, expected workload, and
  model-workload binding digests that Routstr rechecks
- emit Routstr's required payload binding values and full verification-step map

Routstr computes the provider status `runtime_evidence_digest` from the returned
runtime claims so evidence freshness is not confused with the static policy
digest.

The verifier also enforces its own policy boundary: manifest/proxy artifact pins
must be full `sha256:<64 hex>` digests, confidentiality downgrade settings must
be disabled, `expected_workload_sans` or `expected_workload_ids` must bind the
attested Privatemode workload identities, `model_workload_bindings` must bind
selected Routstr model IDs to those exact workload SANs or IDs, and
`api_base_url`, `cdn_base_url`, and `manifest_url` must be absolute HTTPS URLs
without URL userinfo. If `proxy_base_url` is present in the verifier payload,
it must be an absolute loopback URL without URL userinfo; the local proxy is the
plaintext-to-encrypted Privatemode boundary.
If Routstr supplies top-level payload `model_ids`, they must exactly match the
`model_workload_bindings` keys.

Recommended provider policy shape:

```json
{
  "manifest_path": "/etc/routstr/privatemode/manifest.json",
  "manifest_digest": "sha256:<manifest-json-digest>",
  "manifest_log_dir": "/var/lib/privatemode-proxy/manifests",
  "expected_workload_sans": ["workload-kimi-k2-6"],
  "model_workload_bindings": {
    "privatemode/kimi-k2-6": {
      "workload_sans": ["workload-kimi-k2-6"]
    }
  },
  "proxy_binary_path": "/usr/local/bin/privatemode-proxy",
  "proxy_binary_digest": "sha256:<proxy-binary-digest>",
  "dump_requests": false,
  "shared_prompt_cache": false,
  "nvidia_ocsp_allow_unknown": false,
  "nvidia_ocsp_revoked_grace_period_hours": 0,
  "api_key_env": "PRIVATEMODE_API_KEY",
  "api_base_url": "https://api.privatemode.ai",
  "verifier_command": ["/usr/local/bin/routstr-privatemode-go-verifier"],
  "verifier_command_digest": "sha256:<this-verifier-binary-digest>"
}
```

Do not put the Privatemode API key in provider settings. Use `api_key_env` or
`api_key_file` so the key stays out of Routstr's policy digest and status output.
