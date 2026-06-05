# Confidential Routing Examples

These files are deployment templates for verified-only confidential routing.
They are intentionally not production-ready as-is: every `sha256:aaaa...`,
`sha256:bbbb...`, `sha256:cccc...`, `sha256:dddd...`, and `sha256:eeee...`
value must be replaced with a live measurement, release, artifact, verifier, or
key digest from the target deployment.

Use the templates as the starting policy shape:

- `tinfoil-provider-settings.example.json` selects one Tinfoil model, explicitly
  uses Tinfoil EHBP transport, and requires router proof plus a per-model
  attestation target.
- `ppq-private-provider-settings.example.json` selects one PPQ `private/*`
  model and requires the private attestation bundle, EHBP key binding, and the
  mapped downstream model-enclave attestation target.
- `privatemode-provider-settings.example.json` selects one Privatemode model and
  binds it to an expected Contrast workload identity through
  `model_workload_bindings`.
- `routstr-tee-environment.example` lists the local Routstr TEE inputs needed before
  strict required-mode routing can pass.

The `CONFIDENTIAL_ROUTING_MODE` and `ROUTSTR_TEE_*` deployment controls are
environment-authoritative. Persisted settings and admin settings updates cannot
weaken or authorize these controls after startup; change them in the deployment
environment and restart/reload Routstr instead.

Validate edited provider policies before deployment:

```sh
make confidential-preflight \
  POLICY_JSON="examples/confidential-routing/tinfoil-provider-settings.example.json examples/confidential-routing/ppq-private-provider-settings.example.json examples/confidential-routing/privatemode-provider-settings.example.json"
```

That command is intentionally a shape check. It can succeed for these templates
while strict deployment still fails, because placeholder digests, missing local
Routstr TEE quote inputs, stale external attestation results, or missing
provider/model evidence must not authorize routing. As of the 2026-06-04
strict preflight recheck, the current sibling `confidential-inference` evidence
produces no `routable_with_full_attestation` models for Tinfoil, PPQ private,
or Privatemode: Tinfoil lacks verified router TLS binding and has stale,
missing, or release-digest-failed selected model rows; PPQ private has
model-aware targets but no verified selected-model result rows; Privatemode
remains reference-only until the local proxy verifies Contrast attestation and
the enforced manifest. Treat a model as deployable only after the strict
production preflight below reports `deployment_ready=true` and at least one
selected model appears in `routable_with_full_attestation`.

For Tinfoil models, generate the route policy from fresh
`confidential-inference` target and result evidence instead of copying router
and model release digests by hand:

```sh
python scripts/generate_tinfoil_policy.py \
  --provider-catalog-json ~/s/p/enclava/confidential-inference/src/data/providers.json \
  --attestation-results-json ~/s/p/enclava/confidential-inference/src/data/attestation-results.json \
  --attestation-targets-json ~/s/p/enclava/confidential-inference/src/data/attestation-targets.json \
  --model-id tinfoil/kimi-k2-6 \
  --transport-security ehbp \
  --output ./tinfoil-provider-settings.json
```

The generator fails closed unless the selected Tinfoil router and model rows are
verified, exact release digests are present, model transparency checks are true,
the selected model is present in the sibling provider catalog, the selected model
maps to the published attestation target, and the local Tinfoil verifier digest
is pinned. Omit `--transport-security ehbp` to require TLS binding rows in the
external results before generating a policy.
By default, the attestation results document must have a `last_run` timestamp no
older than 86400 seconds. Use `--max-attestation-result-age-seconds` to tighten
that production window.

For PPQ private models, generate the route policy from fresh
`confidential-inference` evidence instead of copying release digests by hand:

```sh
python scripts/generate_ppq_private_policy.py \
  --provider-catalog-json ~/s/p/enclava/confidential-inference/src/data/providers.json \
  --attestation-results-json ~/s/p/enclava/confidential-inference/src/data/attestation-results.json \
  --attestation-targets-json ~/s/p/enclava/confidential-inference/src/data/attestation-targets.json \
  --model-id private/gpt-oss-120b \
  --output ./ppq-private-provider-settings.json
```

The generator fails closed unless the selected PPQ model rows are verified, the
router and backend release digests are present, the backend TLS/Sigstore/image
checks are true, the sibling `--provider-catalog-json` lists every selected
`private/*` model, the sibling `--attestation-targets-json` maps every selected
model to provider-level target evidence, and the local PPQ/Tinfoil verifier
digest is pinned. Result check IDs must also map to the selected model and any
published backend host/repo target for that model. Provider catalog matching is
intentionally scoped to PPQ private models; descriptive aliases such as
`qwen3-vl-30b-a3b` can match `private/qwen3-vl-30b`, but those aliases are not
applied to unrelated providers. Routstr implements PPQ private EHBP forwarding
directly, so the generated PPQ policy does not require or emit
`proxy_binary_path` or `proxy_binary_digest`.
The PPQ generator currently pins the shared Tinfoil-shaped verifier artifact
path, `build/confidential-verifiers/routstr-tinfoil-go-verifier`, because PPQ's
private flow uses the same EHBP verifier module for the provider and backend
model attestations. Successful PPQ verifier output still reports the public
verifier identity `routstr-ppq-private-go-verifier`, so public status and audit
reports distinguish PPQ private proof from Tinfoil router proof.
By default, the attestation results document must have a `last_run` timestamp no
older than 86400 seconds. Use `--max-attestation-result-age-seconds` to tighten
that production window.

For Privatemode models, generate the route policy from pinned Privatemode
manifest, proxy, verifier, component, key-release, and workload evidence instead
of copying the static template:

```sh
python scripts/generate_privatemode_policy.py \
  --provider-catalog-json ~/s/p/enclava/confidential-inference/src/data/providers.json \
  --model-id privatemode/gpt-oss-120b \
  --manifest-digest sha256:<privatemode-manifest-json-digest> \
  --proxy-binary-path /opt/privatemode/privatemode-proxy \
  --expected-coordinator-measurement sha256:<coordinator-measurement> \
  --expected-secret-service-measurement sha256:<secret-service-measurement> \
  --expected-ai-worker-measurement sha256:<ai-worker-measurement> \
  --expected-gpu-attestation-policy nvidia-ocsp-good-only \
  --expected-key-release-binding sha256:<key-release-binding> \
  --workload-san gpt-oss-120b.default.svc.cluster.local \
  --output ./privatemode-provider-settings.json
```

The generator fails closed unless selected models use exact `privatemode/*`
IDs, the selected models are present in the sibling provider catalog, the
Privatemode proxy URL is local loopback, the manifest/proxy/verifier digests are
full non-placeholder `sha256:` pins, downgrade settings are fixed off, and
`model_workload_bindings` exactly bind every selected model to the expected
workload SANs or IDs. When selecting more than one Privatemode model, pass
explicit `--model-workload-san MODEL=SAN` or
`--model-workload-id MODEL=ID` bindings so workloads cannot be assigned by
accident.

That command validates policy shape and verifier artifact buildability only. It
does not prove the selected providers are currently acceptable for production.

For production, also supply the external guardrails and strict environment gate:

```sh
make confidential-preflight \
  POLICY_JSON="./tinfoil-provider-settings.json ./ppq-private-provider-settings.json ./privatemode-provider-settings.json" \
  ATTESTATION_TARGETS_JSON=~/s/p/enclava/confidential-inference/src/data/attestation-targets.json \
  ATTESTATION_RESULTS_JSON=~/s/p/enclava/confidential-inference/src/data/attestation-results.json \
  PROVIDER_CATALOG_JSON=~/s/p/enclava/confidential-inference/src/data/providers.json \
  MAX_ATTESTATION_RESULT_AGE_SECONDS=86400 \
  STRICT_ATTESTATION_TARGETS=1 \
  STRICT_ATTESTATION_RESULTS=1 \
  STRICT_PROVIDER_CATALOG=1 \
  VERIFY_ARTIFACTS=1 \
  STRICT_ENV=1 \
  STRICT_DEPLOYMENT_READY=1
```

Strict guardrails are expected to fail when the latest external evidence is not
good enough. For Tinfoil's default full-channel policy, a router TLS-binding
failure must reject the policy. If the policy explicitly sets
`transport_security` to `ehbp`, preflight and live-check allow that TLS-only
directory failure but still require Routstr's live verifier to prove the
attested HPKE/EHBP key binding, and selected model transparency checks still
fail closed. PPQ private and Privatemode entries in the external directory are
guardrails only; they still require Routstr's live provider verifier claims.

Then prove the live public surfaces and real requests for every selected route
from the target TEE deployment. `EXPECT_PROVIDER` and `EXPECT_INFERENCE` accept
space-separated entries; the Makefile expands each entry into its own live-check
flag:

```sh
ROUTSTR_API_KEY=replace-with-live-routstr-key \
make confidential-live-check \
  ROUTSTR_URL=https://your-routstr.example \
  EXPECT_PROVIDER="tinfoil:tinfoil/kimi-k2-6 ppq-private:private/gpt-oss-120b privatemode:privatemode/gpt-oss-120b" \
  EXPECT_INFERENCE="tinfoil:tinfoil/kimi-k2-6:chat-completions ppq-private:private/gpt-oss-120b:chat-completions privatemode:privatemode/gpt-oss-120b:messages" \
  PROVIDER_CATALOG_JSON=~/s/p/enclava/confidential-inference/src/data/providers.json \
  ATTESTATION_TARGETS_JSON=~/s/p/enclava/confidential-inference/src/data/attestation-targets.json \
  ATTESTATION_RESULTS_JSON=~/s/p/enclava/confidential-inference/src/data/attestation-results.json \
  MAX_ATTESTATION_RESULT_AGE_SECONDS=86400 \
  STRICT_EXTERNAL_GUARDRAILS=1 \
  RUN_INFERENCE=1 \
  STRICT_INFERENCE_EXERCISED=1 \
  STRICT_CONFIDENTIAL_ROUTES_READY=1 \
  BEARER_TOKEN_ENV=ROUTSTR_API_KEY \
  LIVE_CHECK_JSON=1
```
