"""Tests for the model prioritization algorithm."""

import hashlib
import json
import os
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

# Set required env vars before importing
os.environ["UPSTREAM_BASE_URL"] = "http://test"
os.environ["UPSTREAM_API_KEY"] = "test"

from routstr.algorithm import (  # noqa: E402
    _known_provider_selectors_match,
    calculate_model_cost_score,
    create_model_mappings,
    get_provider_penalty,
    has_current_confidential_verification,
    is_confidential_provider_for_model,
    provider_endpoint_requirement_for_path,
    provider_supports_required_endpoint,
    public_confidentiality_metadata,
    public_supported_endpoints_for_provider,
)
from routstr.payment.models import Architecture, Model, Pricing  # noqa: E402


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


VALID_POLICY_DIGEST = _digest("policy")
VALID_EVIDENCE_DIGEST = _digest("evidence")
VALID_ATTESTATION_REPORT_DIGEST = _digest("attestation-report")
VALID_TLS_DIGEST = _digest("tls")
VALID_RELEASE_DIGEST = _digest("release")
PLACEHOLDER_DIGEST = "sha256:" + ("6" * 64)
VALID_PPQ_ATTESTATION_BUNDLE_URL = "https://api.ppq.ai/private"
VALID_PPQ_ATTESTATION_BUNDLE_URL_DIGEST = (
    "sha256:"
    + hashlib.sha256(
        json.dumps(
            VALID_PPQ_ATTESTATION_BUNDLE_URL,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
)


def _ehbp_verification_steps() -> dict[str, bool]:
    return {
        "hardware_attestation_report": True,
        "hardware_certificate_chain": True,
        "code_transparency": True,
        "measurement_match": True,
        "attested_transport_key_binding": True,
        "freshness": True,
    }


def _privatemode_verification_steps() -> dict[str, bool]:
    return {
        "contrast_manifest": True,
        "coordinator_attestation": True,
        "mesh_ca_binding": True,
        "secret_service_tls": True,
        "ai_worker_attestation": True,
        "gpu_attestation": True,
        "key_release_binding": True,
        "prompt_encryption": True,
        "nvidia_ocsp_revocation": True,
    }


def _sha256_json_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _privatemode_key_release_binding_digest(claims: dict[str, object]) -> str:
    return _sha256_json_digest(
        {
            "attested_workload_policy_digest": claims.get(
                "attested_workload_policy_digest"
            ),
            "expected_workload_identity_digest": claims.get(
                "expected_workload_identity_digest"
            ),
            "model_workload_binding_digest": claims.get(
                "model_workload_binding_digest"
            ),
            "manifest_digest": claims.get("manifest_digest"),
            "mesh_ca_digest": claims.get("mesh_ca_digest"),
            "secret_service_certificate_digest": claims.get(
                "secret_service_certificate_digest"
            ),
            "inference_secret_id_digest": claims.get("inference_secret_id_digest"),
            "nvidia_ocsp_policy_mac_digest": claims.get(
                "nvidia_ocsp_policy_mac_digest"
            ),
        }
    )


def provider_proof_claims_for_mode(mode: str) -> dict[str, object]:
    if mode == "privatemode":
        claims: dict[str, object] = {
            "transport": "privatemode-proxy",
            "trust_tier": "app-e2ee",
            "proxy_base_url": "http://127.0.0.1:8080/v1",
            "payload_policy_digest": VALID_POLICY_DIGEST,
            "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
            "manifest_digest": _digest("privatemode-manifest"),
            "proxy_image_digest": _digest("privatemode-proxy-image"),
            "proxy_binary_digest": _digest("privatemode-proxy-binary"),
            "coordinator_measurement": _digest("privatemode-coordinator"),
            "secret_service_measurement": _digest("privatemode-secret-service"),
            "ai_worker_measurement": _digest("privatemode-ai-worker"),
            "gpu_attestation_policy": "nvidia-ocsp-good-only",
            "attested_workload_identity_digest": _digest("attested-workload-identity"),
            "attested_workload_policy_digest": _digest("privatemode-ai-worker"),
            "expected_workload_identity_digest": _digest("expected-workload-identity"),
            "model_workload_binding_digest": _digest("model-workload-binding"),
            "coordinator_attestation_doc_digest": _digest("coordinator-attestation-doc"),
            "mesh_ca_digest": _digest("mesh-ca"),
            "secret_service_certificate_digest": _digest("secret-service-certificate"),
            "ai_worker_manifest_digest": _digest("ai-worker-manifest"),
            "nvidia_ocsp_policy_header_digest": _digest("nvidia-ocsp-policy-header"),
            "nvidia_ocsp_policy_mac_digest": _digest("nvidia-ocsp-policy-mac"),
            "prompt_encryption_ciphertext_digest": _digest("prompt-encryption"),
            "inference_secret_id_digest": _digest("inference-secret"),
            "verification_steps": _privatemode_verification_steps(),
        }
        claims["key_release_binding"] = _privatemode_key_release_binding_digest(claims)
        return claims

    claims: dict[str, object] = {
        "transport": "ehbp",
        "repo": "tinfoilsh/confidential-model-router",
        "payload_policy_digest": VALID_POLICY_DIGEST,
        "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
        "attested_hpke_public_key_hex": "b" * 64,
        "enclave_measurement_fingerprint": _digest("provider-enclave-measurement"),
        "code_measurement_fingerprint": _digest("provider-code-measurement"),
        "tls_public_key_fingerprint_sha256": VALID_TLS_DIGEST,
        "verification_steps": _ehbp_verification_steps(),
    }
    if mode == "tinfoil":
        claims.update(
            {
                "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
                "attestation_report_digest": VALID_ATTESTATION_REPORT_DIGEST,
                "release_digest": VALID_RELEASE_DIGEST,
            }
        )
    if mode == "ppq-private-tee":
        claims.update(
            {
                "attestation_bundle_url": VALID_PPQ_ATTESTATION_BUNDLE_URL,
                "attestation_bundle_url_digest": VALID_PPQ_ATTESTATION_BUNDLE_URL_DIGEST,
                "client_encryption_boundary": "routstr-tee-ehbp-proxy",
                "release_digest": VALID_RELEASE_DIGEST,
                "repo": "ppq-ai/private-tee",
            }
        )
    return claims


def provider_claims_digest(claims: dict[str, object]) -> str:
    payload = json.dumps(
        claims,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def provider_claims_digest_for_mode(mode: str) -> str:
    return provider_claims_digest(provider_proof_claims_for_mode(mode))


def tinfoil_verifier_raw_digest_claims() -> dict[str, object]:
    claims = provider_proof_claims_for_mode("tinfoil")
    claims["release_digest"] = VALID_RELEASE_DIGEST.removeprefix("sha256:")
    claims["tls_public_key_fingerprint_sha256"] = VALID_TLS_DIGEST.removeprefix(
        "sha256:"
    )
    return claims


def tinfoil_model_attestation_claims(
    model_id: str,
    *,
    release_digest: str = VALID_RELEASE_DIGEST,
    tls_digest: str = VALID_TLS_DIGEST,
) -> dict[str, object]:
    model_slug = model_id.rsplit("/", 1)[-1]
    return {
        "repo": f"tinfoilsh/confidential-{model_slug}",
        "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
        "attestation_report_digest": VALID_ATTESTATION_REPORT_DIGEST,
        "attested_hpke_public_key_hex": "b" * 64,
        "enclave_measurement_fingerprint": _digest("model-enclave-measurement"),
        "code_measurement_fingerprint": _digest("model-code-measurement"),
        "tls_public_key_fingerprint_sha256": tls_digest,
        "release_digest": release_digest,
        "verification_steps": _ehbp_verification_steps(),
    }


def tinfoil_claims_for_model_ids(
    model_ids: list[str],
    *,
    raw_router_digests: bool = False,
) -> dict[str, object]:
    claims = (
        tinfoil_verifier_raw_digest_claims()
        if raw_router_digests
        else provider_proof_claims_for_mode("tinfoil")
    )
    release_digest = (
        VALID_RELEASE_DIGEST.removeprefix("sha256:")
        if raw_router_digests
        else VALID_RELEASE_DIGEST
    )
    tls_digest = (
        VALID_TLS_DIGEST.removeprefix("sha256:")
        if raw_router_digests
        else VALID_TLS_DIGEST
    )
    claims["model_attestations"] = {
        model_id: tinfoil_model_attestation_claims(
            model_id,
            release_digest=release_digest,
            tls_digest=tls_digest,
        )
        for model_id in model_ids
    }
    return claims


def ppq_private_claims_for_model_ids(model_ids: list[str]) -> dict[str, object]:
    claims = provider_proof_claims_for_mode("ppq-private-tee")
    claims["selected_model_ids"] = list(model_ids)
    claims["backend_model_attestations"] = {
        model_id: tinfoil_model_attestation_claims(model_id)
        for model_id in model_ids
    }
    return claims


def configure_tinfoil_model_attestation_policy(
    provider: Mock,
    model_ids: list[str],
) -> None:
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="tinfoil",
        model_ids=list(model_ids),
        model_id_prefixes=[],
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": VALID_RELEASE_DIGEST,
            "require_model_attestations": True,
            "model_attestation_targets": {
                model_id: {
                    "host": f"{model_id.rsplit('/', 1)[-1]}.tinfoil.example",
                    "repo": f"tinfoilsh/confidential-{model_id.rsplit('/', 1)[-1]}",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                }
                for model_id in model_ids
            },
        },
    )


def configure_ppq_private_policy(
    provider: Mock,
    *,
    attestation_bundle_url: str = VALID_PPQ_ATTESTATION_BUNDLE_URL,
    model_ids: list[str] | None = None,
) -> None:
    if model_ids is None:
        status = provider.confidentiality_status.return_value
        model_ids = list(getattr(status, "model_ids", []) or [])
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="ppq-private-tee",
        model_ids=list(model_ids),
        model_id_prefixes=[],
        policy={
            "repo": "ppq-ai/private-tee",
            "attestation_bundle_url": attestation_bundle_url,
            "attestation_bundle_url_digest": VALID_PPQ_ATTESTATION_BUNDLE_URL_DIGEST,
            "expected_release_digest": VALID_RELEASE_DIGEST,
            "expected_code_measurement_fingerprint": _digest(
                "provider-code-measurement"
            ),
            "proxy_binary_digest": _digest("privatemode-proxy-binary"),
            "require_model_attestations": True,
            "model_attestation_targets": {
                model_id: {
                    "host": f"{model_id.rsplit('/', 1)[-1]}.tinfoil.example",
                    "repo": f"tinfoilsh/confidential-{model_id.rsplit('/', 1)[-1]}",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                }
                for model_id in model_ids
            },
        },
    )


def configure_privatemode_policy(
    provider: Mock,
    *,
    model_ids: list[str] | None = None,
    workload_sans: list[str] | None = None,
) -> None:
    if model_ids is None:
        status = provider.confidentiality_status.return_value
        model_ids = list(getattr(status, "model_ids", []) or [])
    if workload_sans is None:
        workload_sans = ["gpt-oss-120b.default.svc.cluster.local"]
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="privatemode",
        model_ids=list(model_ids),
        model_id_prefixes=[],
        policy={
            "manifest_digest": _digest("privatemode-manifest"),
            "proxy_image_digest": _digest("privatemode-proxy-image"),
            "proxy_binary_digest": _digest("privatemode-proxy-binary"),
            "expected_coordinator_measurement": _digest("privatemode-coordinator"),
            "expected_secret_service_measurement": _digest(
                "privatemode-secret-service"
            ),
            "expected_ai_worker_measurement": _digest("privatemode-ai-worker"),
            "expected_trust_tier": "app-e2ee",
            "expected_gpu_attestation_policy": "nvidia-ocsp-good-only",
            "expected_key_release_binding": provider.confidentiality_status.return_value.verified_claims[
                "key_release_binding"
            ],
            "expected_workload_sans": workload_sans,
            "model_workload_bindings": {
                model_id: {"workload_sans": workload_sans} for model_id in model_ids
            },
        },
    )


def create_test_model(
    model_id: str,
    prompt_price: float = 0.001,
    completion_price: float = 0.002,
    request_price: float = 0.0,
) -> Model:
    """Helper to create a test model with given pricing."""
    model = Model(
        id=model_id,
        name=f"Test {model_id}",
        created=1234567890,
        description="Test model",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(
            prompt=prompt_price,
            completion=completion_price,
            request=request_price,
            image=0.0,
            web_search=0.0,
            internal_reasoning=0.0,
        ),
    )
    model.supported_endpoints = ["/v1/chat/completions"]
    return model


def create_test_provider(
    name: str,
    base_url: str = "http://test.com",
    *,
    db_id: int | None = None,
    models: list[Model] | None = None,
    upstream_name: str | None = None,
    confidentiality_status: SimpleNamespace | None = None,
) -> Mock:
    """Helper to create a test provider mock."""
    provider = Mock()
    provider.provider_type = name
    provider.base_url = base_url
    provider.db_id = db_id
    provider.provider_fee = 1.01
    provider.upstream_name = upstream_name or name
    provider.supports_anthropic_messages = False
    provider.supports_messages_count_tokens_api = name not in {
        "ppq-private",
        "tinfoil",
        "privatemode",
    }
    provider.supports_audio_api = name != "ppq-private"
    provider.supports_audio_transcriptions_api = name != "ppq-private"
    provider.supports_audio_translations_api = name not in {
        "ppq-private",
        "privatemode",
    }
    provider.supports_audio_speech_api = name not in {
        "ppq-private",
        "tinfoil",
        "privatemode",
    }
    provider.supports_completions_api = name not in {"ppq-private", "tinfoil"}
    provider.supports_embeddings_api = True
    provider.supports_images_api = name not in {"ppq-private", "tinfoil", "privatemode"}
    provider.supports_moderations_api = name not in {
        "ppq-private",
        "tinfoil",
        "privatemode",
    }
    provider.supports_responses_api = True
    provider.requires_verified_confidential_transport = name == "privatemode"
    provider.requires_verified_ehbp_transport = name in {"tinfoil", "ppq-private"}
    provider.get_cached_models.return_value = models or []
    provider.confidentiality_status.return_value = (
        confidentiality_status
        or SimpleNamespace(
            enabled=False,
            verified=False,
            mode="none",
            expires_at=None,
            model_ids=[],
            model_id_prefixes=[],
        )
    )
    policy_digest = getattr(provider.confidentiality_status.return_value, "policy_digest", None)
    if isinstance(policy_digest, str):
        provider.confidentiality_policy.return_value = SimpleNamespace(digest=policy_digest)
    else:
        provider.confidentiality_policy.return_value = None
    return provider


def create_confidential_status(**overrides: object) -> SimpleNamespace:
    mode = str(overrides.get("mode") or "test-verifier")
    values: dict[str, object] = {
        "enabled": True,
        "verified": True,
        "mode": mode,
        "verified_at": 1_700_000_000,
        "expires_at": 4_102_444_800,
        "model_ids": [],
        "model_id_prefixes": [],
        "verifier": "unit-test-verifier",
        "policy_digest": VALID_POLICY_DIGEST,
        "evidence_digest": VALID_EVIDENCE_DIGEST,
        "verified_claims": provider_proof_claims_for_mode(mode),
    }
    values.update(overrides)
    verified_claims = values.get("verified_claims")
    if (
        values.get("mode") in {"ppq-private-tee", "privatemode"}
        and isinstance(verified_claims, dict)
        and "selected_model_ids" not in verified_claims
        and isinstance(values.get("model_ids"), list)
    ):
        verified_claims["selected_model_ids"] = list(values["model_ids"])
    if (
        values.get("mode") == "ppq-private-tee"
        and isinstance(verified_claims, dict)
        and "backend_model_attestations" not in verified_claims
        and isinstance(values.get("model_ids"), list)
    ):
        verified_claims["backend_model_attestations"] = {
            model_id: tinfoil_model_attestation_claims(model_id)
            for model_id in values["model_ids"]
            if isinstance(model_id, str)
        }
    return SimpleNamespace(**values)


def create_db_attestation_row(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "provider_id": 1,
        "model_id": "tinfoil/gpt-secure",
        "mode": "tinfoil",
        "verified": True,
        "verified_at": 1_700_000_000,
        "expires_at": 4_102_444_800,
        "policy_digest": VALID_POLICY_DIGEST,
        "evidence_digest": VALID_EVIDENCE_DIGEST,
        "verifier": "unit-test-verifier",
        "claims_json": "{}",
        "failure_reason": None,
        "updated_at": 1_700_000_000,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_calculate_model_cost_score_basic() -> None:
    """Test basic cost calculation."""
    model = create_test_model("test-model", prompt_price=0.001, completion_price=0.002)
    cost = calculate_model_cost_score(model)

    # Expected: (1000 tokens * 0.001) + (500 tokens * 0.002) = 0.001 + 0.001 = 0.002
    assert cost == 0.002


def test_calculate_model_cost_score_with_request_fee() -> None:
    """Test cost calculation with request fee."""
    model = create_test_model(
        "test-model",
        prompt_price=0.001,
        completion_price=0.002,
        request_price=0.0005,
    )
    cost = calculate_model_cost_score(model)

    # Expected: 0.001 + 0.001 + 0.0005 = 0.0025
    assert cost == 0.0025


def test_calculate_model_cost_score_expensive_model() -> None:
    """Test cost calculation for expensive model."""
    model = create_test_model(
        "expensive-model", prompt_price=0.03, completion_price=0.06
    )
    cost = calculate_model_cost_score(model)

    # Expected: (1000 * 0.03) + (500 * 0.06) = 0.03 + 0.03 = 0.06
    assert cost == 0.06


def test_get_provider_penalty_regular_provider() -> None:
    """Test penalty for regular provider."""
    provider = create_test_provider("regular-provider", "http://provider.com")
    penalty = get_provider_penalty(provider)
    assert penalty == 1.0


def test_get_provider_penalty_openrouter() -> None:
    """Test penalty for OpenRouter."""
    provider = create_test_provider("openrouter", "https://openrouter.ai/api/v1")
    penalty = get_provider_penalty(provider)
    assert penalty == 1.001


def test_create_model_mappings_includes_db_override_for_missing_cached_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model overrides should still map when provider discovery misses the model."""
    provider = create_test_provider(
        "azure",
        "https://example.openai.azure.com/openai/v1",
        db_id=7,
        models=[],
    )
    override_model = create_test_model("azure/gpt-4o")
    override_model.canonical_slug = "azure-deployment"

    def fake_row_to_model(*args, **kwargs) -> Model:  # type: ignore[no-untyped-def]
        return override_model

    monkeypatch.setattr("routstr.payment.models._row_to_model", fake_row_to_model)

    override_row = SimpleNamespace(
        id="azure/gpt-4o", upstream_provider_id=7, enabled=True
    )

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={"azure/gpt-4o": (override_row, 1.01)},
        disabled_model_ids=set(),
    )

    assert "azure/gpt-4o" in model_instances
    assert provider_map["azure/gpt-4o"] == [provider]
    assert "gpt-4o" in unique_models


def test_create_model_mappings_dedupes_with_provider_identity_not_provider_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different provider instances of same type should both survive dedupe."""
    provider_a_model = create_test_model(
        "azure/gpt-4o", prompt_price=0.01, completion_price=0.01
    )
    provider_a = create_test_provider(
        "azure",
        "https://a.openai.azure.com/openai/v1",
        db_id=1,
        models=[provider_a_model],
        upstream_name="azure-a",
    )
    provider_b = create_test_provider(
        "azure",
        "https://b.openai.azure.com/openai/v1",
        db_id=2,
        models=[],
        upstream_name="azure-b",
    )

    override_model = create_test_model(
        "azure/gpt-4o", prompt_price=0.001, completion_price=0.001
    )
    override_model.canonical_slug = "azure-b-deployment"

    def fake_row_to_model(*args, **kwargs) -> Model:  # type: ignore[no-untyped-def]
        return override_model

    monkeypatch.setattr("routstr.payment.models._row_to_model", fake_row_to_model)

    override_row = SimpleNamespace(
        id="azure/gpt-4o", upstream_provider_id=2, enabled=True
    )

    _, provider_map, _ = create_model_mappings(
        upstreams=[provider_a, provider_b],
        overrides_by_id={"azure/gpt-4o": (override_row, 1.01)},
        disabled_model_ids=set(),
    )

    providers_for_alias = provider_map["azure/gpt-4o"]
    assert provider_a in providers_for_alias
    assert provider_b in providers_for_alias
    assert len(providers_for_alias) == 2


def test_create_model_mappings_filters_unverified_providers_when_confidential_required() -> (
    None
):
    """Confidential routing must never publish plaintext fallback providers."""
    plain_model = create_test_model(
        "gpt-secure", prompt_price=0.001, completion_price=0.001
    )
    confidential_model = create_test_model(
        "gpt-secure", prompt_price=0.01, completion_price=0.01
    )
    plain_provider = create_test_provider(
        "plain",
        "https://plain.example/v1",
        db_id=1,
        models=[plain_model],
    )
    verified_claims = tinfoil_claims_for_model_ids(["gpt-secure"])
    confidential_provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=2,
        models=[confidential_model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            verified_claims=verified_claims,
        ),
    )
    configure_tinfoil_model_attestation_policy(confidential_provider, ["gpt-secure"])

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[plain_provider, confidential_provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert "gpt-secure" in model_instances
    assert provider_map["gpt-secure"] == [confidential_provider]
    assert unique_models["gpt-secure"].upstream_provider_id == "tinfoil"
    confidentiality = unique_models["gpt-secure"].confidentiality
    assert confidentiality is not None
    assert confidentiality["_verified_claims"] == verified_claims
    assert confidentiality["_confidentiality_policy"] == (
        confidential_provider.confidentiality_policy.return_value.policy
    )
    public_confidentiality = {
        key: value
        for key, value in confidentiality.items()
        if not key.startswith("_")
    }
    assert public_confidentiality == {
        "enabled": True,
        "verified": True,
        "attestation_status": "verified",
        "mode": "tinfoil",
        "provider_type": "tinfoil",
        "verifier": "unit-test-verifier",
        "policy_digest": VALID_POLICY_DIGEST,
        "evidence_digest": VALID_EVIDENCE_DIGEST,
        "verified_claims_digest": provider_claims_digest(verified_claims),
        "verified_at": 1_700_000_000,
        "expires_at": 4_102_444_800,
        "model_ids": ["gpt-secure"],
        "model_id_prefixes": [],
        "supported_endpoints": ["/v1/chat/completions"],
        "metadata_leakage": ["model", "usage"],
        "proof_claims": {
            "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
            "attestation_report_digest": VALID_ATTESTATION_REPORT_DIGEST,
            "attested_hpke_public_key_hex": "b" * 64,
            "code_measurement_fingerprint": _digest("provider-code-measurement"),
            "enclave_measurement_fingerprint": _digest(
                "provider-enclave-measurement"
            ),
            "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
            "payload_policy_digest": VALID_POLICY_DIGEST,
            "release_digest": VALID_RELEASE_DIGEST,
            "repo": "tinfoilsh/confidential-model-router",
            "tls_public_key_fingerprint_sha256": VALID_TLS_DIGEST,
            "transport": "ehbp",
            "model_attestations": {
                "gpt-secure": tinfoil_model_attestation_claims("gpt-secure")
            },
            "verification_steps": _ehbp_verification_steps(),
        },
    }


def test_confidential_provider_requires_loaded_db_attestation_row() -> None:
    model = create_test_model("tinfoil/gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=tinfoil_claims_for_model_ids(["tinfoil/gpt-secure"]),
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["tinfoil/gpt-secure"])
    provider._db_confidential_attestations = {}

    assert has_current_confidential_verification(provider) is True
    assert is_confidential_provider_for_model(provider, model, "gpt-secure") is False

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert "gpt-secure" not in provider_map
    assert "gpt-secure" not in unique_models


def test_confidential_provider_rejects_expired_db_attestation_row() -> None:
    model = create_test_model("tinfoil/gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=tinfoil_claims_for_model_ids(["tinfoil/gpt-secure"]),
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["tinfoil/gpt-secure"])
    provider._db_confidential_attestations = {
        ("tinfoil/gpt-secure", "tinfoil"): create_db_attestation_row(
            expires_at=1_700_000_001
        )
    }

    assert has_current_confidential_verification(provider) is True
    assert is_confidential_provider_for_model(provider, model, "gpt-secure") is False


def test_confidential_provider_rejects_db_attestation_policy_digest_mismatch() -> None:
    model = create_test_model("tinfoil/gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=tinfoil_claims_for_model_ids(["tinfoil/gpt-secure"]),
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["tinfoil/gpt-secure"])
    provider._db_confidential_attestations = {
        ("tinfoil/gpt-secure", "tinfoil"): create_db_attestation_row(
            policy_digest=_digest("old-policy")
        )
    }

    assert has_current_confidential_verification(provider) is True
    assert is_confidential_provider_for_model(provider, model, "gpt-secure") is False


def test_confidential_provider_accepts_fresh_matching_db_attestation_row() -> None:
    model = create_test_model("tinfoil/gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=tinfoil_claims_for_model_ids(["tinfoil/gpt-secure"]),
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["tinfoil/gpt-secure"])
    provider._db_confidential_attestations = {
        ("tinfoil/gpt-secure", "tinfoil"): create_db_attestation_row()
    }

    assert (
        is_confidential_provider_for_model(provider, model, "tinfoil/gpt-secure")
        is True
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert provider_map["tinfoil/gpt-secure"] == [provider]
    assert "tinfoil/gpt-secure" in unique_models


def test_confidential_routing_rejects_known_provider_mode_mismatch() -> None:
    """A verified Tinfoil provider must not route with PPQ private evidence."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["gpt-secure"],
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_accepts_tinfoil_raw_verifier_digest_claims() -> None:
    """Tinfoil verifier emits raw release and TLS digests; routing must accept them."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            verified_claims=tinfoil_claims_for_model_ids(
                ["gpt-secure"],
                raw_router_digests=True,
            ),
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["gpt-secure"])

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is True
    assert provider_map["gpt-secure"] == [provider]
    assert unique_models["gpt-secure"].confidentiality is not None
    assert unique_models["gpt-secure"].confidentiality["verified"] is True


def test_confidential_routing_accepts_tinfoil_runtime_evidence_digest() -> None:
    """Live Tinfoil evidence stores the verified runtime claims digest on status."""
    model_id = "tinfoil/gpt-secure"
    claims = tinfoil_claims_for_model_ids([model_id])
    runtime_evidence_digest = _sha256_json_digest(
        {
            key: value
            for key, value in claims.items()
            if key
            not in {
                "payload_policy_digest",
                "payload_evidence_digest",
                "payload_verification_nonce",
                "runtime_evidence_digest",
            }
        }
    )
    assert runtime_evidence_digest != VALID_EVIDENCE_DIGEST
    claims["runtime_evidence_digest"] = runtime_evidence_digest
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        upstream_name="tinfoil",
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=[model_id],
            evidence_digest=runtime_evidence_digest,
            verified_claims=claims,
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, [model_id])
    provider._db_confidential_attestations = {
        (model_id, "tinfoil"): create_db_attestation_row(
            model_id=model_id,
            evidence_digest=runtime_evidence_digest,
        )
    }

    assert has_current_confidential_verification(provider) is True

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert provider_map[model_id] == [provider]
    assert unique_models[model_id].confidentiality is not None
    assert unique_models[model_id].confidentiality["verified"] is True


def test_tinfoil_prefixed_policy_covers_bare_catalog_model() -> None:
    """Tinfoil's catalog returns bare IDs while verifier policy uses tinfoil/* IDs."""
    verified_model_id = "tinfoil/kimi-k2-6"
    catalog_model = create_test_model("kimi-k2-6")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        upstream_name="tinfoil",
        models=[catalog_model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=[verified_model_id],
            verified_claims=tinfoil_claims_for_model_ids([verified_model_id]),
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, [verified_model_id])
    provider._db_confidential_attestations = {
        (verified_model_id, "tinfoil"): create_db_attestation_row(
            model_id=verified_model_id,
        )
    }

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=False,
    )

    assert provider_map["kimi-k2-6"] == [provider]
    assert unique_models["kimi-k2-6"].confidentiality is not None
    assert unique_models["kimi-k2-6"].confidentiality["verified"] is True


def test_confidential_routing_rejects_known_provider_without_local_policy() -> None:
    """Verified-looking status is not enough without a local pinned verifier policy."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            verified_claims=tinfoil_verifier_raw_digest_claims(),
        ),
    )
    provider.confidentiality_policy.return_value = None

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_stale_policy_digest() -> None:
    """Runtime status must match the provider's current verifier policy digest."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            policy_digest=VALID_POLICY_DIGEST,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest="sha256:" + ("9" * 64)
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_accepts_current_policy_digest() -> None:
    """A verified route remains eligible when status and current policy match."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            policy_digest=VALID_POLICY_DIGEST,
            verified_claims=tinfoil_claims_for_model_ids(["gpt-secure"]),
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["gpt-secure"])

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is True
    assert provider_map["gpt-secure"] == [provider]
    assert unique_models["gpt-secure"].confidentiality is not None


def test_confidential_routing_rejects_policy_provider_type_mismatch() -> None:
    """Route selection must bind local verifier policy identity to provider type."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            policy_digest=VALID_POLICY_DIGEST,
            verified_claims=tinfoil_claims_for_model_ids(["gpt-secure"]),
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["gpt-secure"])
    provider.confidentiality_policy.return_value.provider_type = "ppq-private"

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_policy_mode_mismatch() -> None:
    """Route selection must bind local verifier policy mode to provider type."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            policy_digest=VALID_POLICY_DIGEST,
            verified_claims=tinfoil_claims_for_model_ids(["gpt-secure"]),
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["gpt-secure"])
    provider.confidentiality_policy.return_value.mode = "privatemode"

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_payload_policy_digest_mismatch() -> None:
    """Verifier payload binding must match the provider status policy digest."""
    claims = tinfoil_claims_for_model_ids(["gpt-secure"])
    claims["payload_policy_digest"] = "sha256:" + ("9" * 64)
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            policy_digest=VALID_POLICY_DIGEST,
            verified_claims=claims,
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["gpt-secure"])

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_payload_evidence_digest_mismatch() -> None:
    """Verifier payload binding must match the provider status evidence digest."""
    claims = tinfoil_claims_for_model_ids(["gpt-secure"])
    claims["payload_evidence_digest"] = "sha256:" + ("9" * 64)
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_claims=claims,
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["gpt-secure"])

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_verification_requires_selected_model_scope() -> None:
    """Verified provider evidence without selectors must not be route-ready."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=[],
            model_id_prefixes=[],
            policy_digest=VALID_POLICY_DIGEST,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_ppq_private_without_tls_binding_claim() -> None:
    """PPQ private EHBP proof must bind the routed TLS endpoint."""
    model = create_test_model("private/gpt-oss-120b")
    claims = provider_proof_claims_for_mode("ppq-private-tee")
    claims.pop("tls_public_key_fingerprint_sha256", None)
    claims.pop("tls_public_key", None)
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=claims,
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_privatemode_proxy_base_url_claim_mismatch() -> (
    None
):
    """Privatemode proof must bind to the exact local proxy URL being routed."""
    model = create_test_model("privatemode/gpt-secure")
    claims = provider_proof_claims_for_mode("privatemode")
    claims["proxy_base_url"] = "http://127.0.0.1:9090/v1"
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/gpt-secure"],
            verified_claims=claims,
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_privatemode_remote_proxy_base_url() -> None:
    """Privatemode routing must require the local verified proxy boundary."""
    model = create_test_model("privatemode/gpt-secure")
    claims = provider_proof_claims_for_mode("privatemode")
    claims["proxy_base_url"] = "https://api.privatemode.ai/v1"
    provider = create_test_provider(
        "privatemode",
        "https://api.privatemode.ai/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/gpt-secure"],
            verified_claims=claims,
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_privatemode_proxy_base_url_invalid_port() -> None:
    """Privatemode routing must reject malformed loopback proxy origins."""
    model = create_test_model("privatemode/gpt-secure")
    claims = provider_proof_claims_for_mode("privatemode")
    claims["proxy_base_url"] = "http://127.0.0.1:bad/v1"
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:bad/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/gpt-secure"],
            verified_claims=claims,
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_known_provider_without_required_proof_claims() -> (
    None
):
    """A known confidential provider needs provider-specific proof, not any claim."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            verified_claims={"tee": "unit-test"},
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_unknown_provider_verified_status() -> None:
    """A custom provider needs its own verifier contract before required routing."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "custom",
        "https://custom.example/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="custom-confidential",
            model_ids=["gpt-secure"],
            verified_claims=provider_proof_claims_for_mode("tinfoil"),
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_non_json_verified_claims() -> None:
    """Verified claims are digest-bound proof material and must be JSON-shaped."""

    class NonSerializableClaim:
        pass

    claims = provider_proof_claims_for_mode("tinfoil")
    claims["unexpected_object"] = NonSerializableClaim()
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            verified_claims=claims,
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_non_canonical_json_verified_claims() -> None:
    """NaN is accepted by Python JSON by default but is not canonical proof JSON."""
    claims = provider_proof_claims_for_mode("tinfoil")
    claims["non_canonical"] = float("nan")
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            verified_claims=claims,
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_non_string_verifier_from_status_like_object() -> (
    None
):
    """Verifier identity is a proof selector and must not be string-coerced."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            verifier=123,
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_truthy_string_evidence_checker_result() -> None:
    """Status-like evidence helpers must return strict True, not a truthy value."""

    class TruthyEvidenceStatus(SimpleNamespace):
        def has_required_verification_evidence(self) -> str:
            return "false"

    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=TruthyEvidenceStatus(
            enabled=True,
            verified=True,
            mode="tinfoil",
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            model_ids=["gpt-secure"],
            model_id_prefixes=[],
            verified_claims=provider_proof_claims_for_mode("tinfoil"),
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_non_string_status_mode_identity() -> None:
    """Provider mode is a proof selector and must not be string-coerced."""

    class StringyMode:
        def __str__(self) -> str:
            return "tinfoil"

    model = create_test_model("gpt-secure")
    status = create_confidential_status(
        mode="tinfoil",
        model_ids=["gpt-secure"],
    )
    status.mode = StringyMode()
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=status,
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_non_string_provider_type_identity() -> None:
    """Provider type chooses proof rules and must not be string-coerced."""

    class StringyProviderType:
        def __str__(self) -> str:
            return "tinfoil"

    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
        ),
    )
    provider.provider_type = StringyProviderType()

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_public_confidentiality_metadata_rejects_non_string_identity_fields() -> None:
    """Public model metadata must not stringify proof selector identities."""

    class StringyIdentity:
        def __str__(self) -> str:
            return "tinfoil"

    model = create_test_model("gpt-secure")

    status_with_bad_mode = create_confidential_status(
        mode="tinfoil",
        model_ids=["gpt-secure"],
    )
    status_with_bad_mode.mode = StringyIdentity()
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        models=[model],
        confidentiality_status=status_with_bad_mode,
    )

    assert public_confidentiality_metadata(provider, model, "gpt-secure") is None

    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
        ),
    )
    provider.provider_type = StringyIdentity()

    assert public_confidentiality_metadata(provider, model, "gpt-secure") is None


def test_confidential_routing_skips_non_string_provider_model_id() -> None:
    """Provider catalog model IDs are route selectors and must be real strings."""

    class StringyModelId:
        def __str__(self) -> str:
            return "gpt-secure"

    model = create_test_model("placeholder")
    model.id = StringyModelId()  # type: ignore[assignment]
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_non_string_upstream_namespace_alias() -> None:
    """Provider namespaces synthesize routable aliases and must be real strings."""

    class StringyNamespace:
        def __str__(self) -> str:
            return "tinfoil"

    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
        ),
    )
    provider.upstream_name = StringyNamespace()

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_string_expires_at_from_status_like_object() -> (
    None
):
    """Routing freshness must stay strict even if status validation is bypassed."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            expires_at="4102444800",
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_string_verified_at_from_status_like_object() -> (
    None
):
    """Verifier freshness timestamps are proof fields, not display strings."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            verified_at="1700000000",
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_missing_verified_at_from_status_like_object() -> (
    None
):
    """Verifier freshness metadata must prove when the evidence was emitted."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            verified_at=None,
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_future_verified_at_from_status_like_object() -> (
    None
):
    """The final routing predicate must not trust proof timestamps from the future."""
    model = create_test_model("gpt-secure")
    now = int(time.time())
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            verified_at=now + 60,
            expires_at=now + 300,
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_endpoint_capabilities_reject_truthy_string_flags() -> None:
    """Endpoint capability flags gate routing and must not use truthiness."""
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        confidentiality_status=create_confidential_status(mode="tinfoil"),
    )
    provider.supports_responses_api = "false"

    assert (
        provider_supports_required_endpoint(
            provider,
            "supports_responses_api",
        )
        is False
    )
    assert "/v1/responses" not in public_supported_endpoints_for_provider(provider)


def test_confidential_model_metadata_rejects_non_string_selector_values() -> None:
    """Malformed selector diagnostics must not publish routable metadata."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=[
                "gpt-secure",
                {"api_key": "SECRET_SELECTOR"},
                123,
            ],
            model_id_prefixes=[
                "tinfoil/",
                {"raw_prompt": "SECRET_PREFIX"},
            ],
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert provider_map == {}
    assert unique_models == {}
    assert "SECRET_SELECTOR" not in str(provider_map)
    assert "SECRET_PREFIX" not in str(unique_models)


def test_confidential_routing_rejects_malformed_tinfoil_tls_proof_alias() -> None:
    """Routing proof aliases must all be valid when copied status bypasses parsing."""
    model = create_test_model("gpt-secure")
    claims = provider_proof_claims_for_mode("tinfoil")
    claims["tls_public_key_fingerprint_sha256"] = ""
    claims["tls_public_key"] = VALID_TLS_DIGEST
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            verified_claims=claims,
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_conflicting_tinfoil_tls_proof_aliases() -> None:
    """TLS proof aliases describe the same attested key and must agree."""
    model = create_test_model("gpt-secure")
    claims = provider_proof_claims_for_mode("tinfoil")
    claims["tls_public_key"] = "sha256:" + ("e" * 64)
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            verified_claims=claims,
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


@pytest.mark.parametrize(
    ("malformed_claim", "fallback_claim"),
    (
        ("manifest_digest", "manifest_log_manifest_digest"),
        ("proxy_image_digest", "proxy_binary_digest"),
    ),
)
def test_confidential_routing_rejects_malformed_privatemode_proof_alias(
    malformed_claim: str,
    fallback_claim: str,
) -> None:
    """Routing must not let one valid Privatemode alias mask a bad present alias."""
    model = create_test_model("privatemode/secure-model")
    claims = provider_proof_claims_for_mode("privatemode")
    claims[fallback_claim] = claims[malformed_claim]
    claims[malformed_claim] = ""
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model"],
            verified_claims=claims,
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_privatemode_image_only_proxy_proof() -> None:
    """Copied Privatemode proof must include the locally verified proxy binary."""
    model = create_test_model("privatemode/secure-model")
    claims = provider_proof_claims_for_mode("privatemode")
    claims.pop("proxy_binary_digest", None)
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model"],
            verified_claims=claims,
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_routing_rejects_conflicting_privatemode_manifest_aliases() -> (
    None
):
    """Privatemode manifest proof aliases must describe the same manifest."""
    model = create_test_model("privatemode/secure-model")
    claims = provider_proof_claims_for_mode("privatemode")
    claims["manifest_log_manifest_digest"] = "sha256:" + ("f" * 64)
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model"],
            verified_claims=claims,
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def _privatemode_selected_model_policy() -> dict[str, object]:
    return {
        "manifest_digest": _digest("privatemode-manifest"),
        "expected_trust_tier": "app-e2ee",
        "expected_workload_sans": ["secure-model.default.svc.cluster.local"],
        "proxy_binary_digest": _digest("privatemode-proxy-binary"),
        "model_workload_bindings": {
            "privatemode/secure-model": {
                "workload_sans": ["secure-model.default.svc.cluster.local"],
            }
        },
    }


def _privatemode_claims_matching_selected_model_policy() -> dict[str, object]:
    claims = provider_proof_claims_for_mode("privatemode")
    policy = _privatemode_selected_model_policy()
    claims["expected_workload_identity_digest"] = _sha256_json_digest(
        {
            "ids": [],
            "sans": ["secure-model.default.svc.cluster.local"],
        }
    )
    claims["model_workload_binding_digest"] = _sha256_json_digest(
        {
            "privatemode/secure-model": {
                "workload_ids": [],
                "workload_sans": ["secure-model.default.svc.cluster.local"],
            }
        }
    )
    claims["selected_model_ids"] = ["privatemode/secure-model"]
    claims["key_release_binding"] = _privatemode_key_release_binding_digest(claims)
    assert policy
    return claims


def test_privatemode_selected_model_status_requires_policy_binding_claims() -> None:
    """Route-time proof checks must bind Privatemode proof to selected workloads."""
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model"],
            verified_claims=provider_proof_claims_for_mode("privatemode"),
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="privatemode",
        model_ids=["privatemode/secure-model"],
        model_id_prefixes=[],
        policy=_privatemode_selected_model_policy(),
    )

    assert has_current_confidential_verification(provider) is False


def test_privatemode_selected_model_status_accepts_policy_binding_claims() -> None:
    """Matching Privatemode workload-binding proof remains routable."""
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model"],
            verified_claims=_privatemode_claims_matching_selected_model_policy(),
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="privatemode",
        model_ids=["privatemode/secure-model"],
        model_id_prefixes=[],
        policy=_privatemode_selected_model_policy(),
    )

    assert has_current_confidential_verification(provider) is True


def test_privatemode_selected_model_status_requires_app_e2ee_trust_tier() -> None:
    """Route-time Privatemode proof must match the app-E2EE preflight tier."""
    claims = _privatemode_claims_matching_selected_model_policy()
    claims["trust_tier"] = "reference-only"
    claims["key_release_binding"] = _privatemode_key_release_binding_digest(claims)
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model"],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy=_privatemode_selected_model_policy(),
    )

    assert has_current_confidential_verification(provider) is False


def test_privatemode_selected_model_status_requires_app_e2ee_policy_pin() -> None:
    """Route-time policy must pin the same app-E2EE tier that proof claims assert."""
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model"],
            verified_claims=_privatemode_claims_matching_selected_model_policy(),
        ),
    )
    policy = _privatemode_selected_model_policy()
    policy.pop("expected_trust_tier", None)
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="privatemode",
        model_ids=["privatemode/secure-model"],
        model_id_prefixes=[],
        policy=policy,
    )

    assert has_current_confidential_verification(provider) is False


def test_privatemode_route_time_requires_manifest_identity_policy_pin() -> None:
    """Route-time Privatemode proof must bind to a manifest digest or log."""
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model"],
            verified_claims=_privatemode_claims_matching_selected_model_policy(),
        ),
    )
    policy = _privatemode_selected_model_policy()
    policy.pop("manifest_digest", None)
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="privatemode",
        model_ids=["privatemode/secure-model"],
        model_id_prefixes=[],
        policy=policy,
    )

    assert has_current_confidential_verification(provider) is False


def test_privatemode_route_time_accepts_manifest_log_policy_binding() -> None:
    """Manifest-log-backed Privatemode policy remains routable with log proof."""
    claims = _privatemode_claims_matching_selected_model_policy()
    claims["manifest_log_digest"] = _digest("privatemode-manifest-log")
    claims["manifest_log_manifest_digest"] = claims["manifest_digest"]
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model"],
            verified_claims=claims,
        ),
    )
    policy = _privatemode_selected_model_policy()
    policy.pop("manifest_digest", None)
    policy["manifest_log_dir"] = "/state/privatemode-manifest-log"
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="privatemode",
        model_ids=["privatemode/secure-model"],
        model_id_prefixes=[],
        policy=policy,
    )

    assert has_current_confidential_verification(provider) is True


def test_privatemode_selected_model_status_requires_selected_model_claims() -> None:
    """Copied Privatemode proof must explicitly cover the selected model IDs."""
    claims = _privatemode_claims_matching_selected_model_policy()
    claims["selected_model_ids"] = ["privatemode/other-model"]
    claims["key_release_binding"] = _privatemode_key_release_binding_digest(claims)
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model"],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy=_privatemode_selected_model_policy(),
    )

    assert has_current_confidential_verification(provider) is False


def test_privatemode_route_time_requires_local_proxy_binary_policy_pin() -> None:
    """Copied Privatemode proof is not enough without a local proxy binary pin."""
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model"],
            verified_claims=_privatemode_claims_matching_selected_model_policy(),
        ),
    )
    policy = _privatemode_selected_model_policy()
    policy.pop("proxy_binary_digest", None)
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy=policy,
    )

    assert has_current_confidential_verification(provider) is False


def test_privatemode_route_time_policy_rejects_scalar_workload_selectors() -> None:
    """Copied Privatemode proof cannot make malformed workload policy routable."""
    scalar_policy = {
        "expected_workload_sans": "secure-model.default.svc.cluster.local",
        "model_workload_bindings": {
            "privatemode/secure-model": {
                "workload_sans": "secure-model.default.svc.cluster.local",
            }
        },
    }
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model"],
            verified_claims=_privatemode_claims_matching_selected_model_policy(),
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy=scalar_policy,
    )

    assert has_current_confidential_verification(provider) is False


def test_privatemode_route_time_policy_rejects_duplicate_trimmed_binding_keys() -> None:
    """Copied Privatemode proof must not hide overwritten selected-model bindings."""
    duplicate_policy = {
        "expected_workload_sans": [
            "secure-model.default.svc.cluster.local",
            "other.default.svc.cluster.local",
        ],
        "model_workload_bindings": {
            "privatemode/secure-model": {
                "workload_sans": ["secure-model.default.svc.cluster.local"],
            },
            " privatemode/secure-model ": {
                "workload_sans": ["other.default.svc.cluster.local"],
            },
        },
    }
    claims = provider_proof_claims_for_mode("privatemode")
    claims["expected_workload_identity_digest"] = _sha256_json_digest(
        {
            "ids": [],
            "sans": [
                "other.default.svc.cluster.local",
                "secure-model.default.svc.cluster.local",
            ],
        }
    )
    claims["model_workload_binding_digest"] = _sha256_json_digest(
        {
            "privatemode/secure-model": {
                "workload_ids": [],
                "workload_sans": ["other.default.svc.cluster.local"],
            }
        }
    )
    claims["selected_model_ids"] = ["privatemode/secure-model"]
    claims["key_release_binding"] = _privatemode_key_release_binding_digest(claims)
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model"],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy=duplicate_policy,
    )

    assert has_current_confidential_verification(provider) is False


def test_privatemode_route_time_policy_rejects_conflicting_workload_aliases() -> None:
    """Route-time Privatemode policy aliases must not broaden workload identity."""
    conflicting_policy = {
        "expected_workload_sans": ["secure-model.default.svc.cluster.local"],
        "expectedWorkloadSANs": ["other.default.svc.cluster.local"],
        "model_workload_bindings": {
            "privatemode/secure-model": {
                "workload_sans": ["other.default.svc.cluster.local"],
            },
        },
    }
    claims = provider_proof_claims_for_mode("privatemode")
    claims["expected_workload_identity_digest"] = _sha256_json_digest(
        {
            "ids": [],
            "sans": [
                "other.default.svc.cluster.local",
                "secure-model.default.svc.cluster.local",
            ],
        }
    )
    claims["model_workload_binding_digest"] = _sha256_json_digest(
        {
            "privatemode/secure-model": {
                "workload_ids": [],
                "workload_sans": ["other.default.svc.cluster.local"],
            }
        }
    )
    claims["selected_model_ids"] = ["privatemode/secure-model"]
    claims["key_release_binding"] = _privatemode_key_release_binding_digest(claims)
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model"],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy=conflicting_policy,
    )

    assert has_current_confidential_verification(provider) is False


def test_privatemode_route_time_policy_rejects_conflicting_binding_aliases() -> None:
    """Route-time Privatemode binding aliases must describe one selected model map."""
    conflicting_policy = {
        "expected_workload_sans": ["secure-model.default.svc.cluster.local"],
        "model_workload_bindings": {
            "privatemode/secure-model": {
                "workload_sans": ["secure-model.default.svc.cluster.local"],
            },
        },
        "modelWorkloadBindings": {
            "privatemode/secure-model": {
                "workload_sans": ["other.default.svc.cluster.local"],
            },
        },
    }
    claims = _privatemode_claims_matching_selected_model_policy()
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model"],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy=conflicting_policy,
    )

    assert has_current_confidential_verification(provider) is False


def test_privatemode_route_time_policy_rejects_unbound_expected_workload() -> None:
    """Route-time Privatemode policy must bind every expected workload to a model."""
    selected_model = "privatemode/secure-model"
    policy = _privatemode_selected_model_policy()
    policy["expected_workload_sans"] = [
        "secure-model.default.svc.cluster.local",
        "other.default.svc.cluster.local",
    ]
    policy["model_workload_bindings"] = {
        selected_model: {
            "workload_sans": ["secure-model.default.svc.cluster.local"],
        }
    }
    claims = _privatemode_claims_matching_selected_model_policy()
    claims["expected_workload_identity_digest"] = _sha256_json_digest(
        {
            "ids": [],
            "sans": [
                "other.default.svc.cluster.local",
                "secure-model.default.svc.cluster.local",
            ],
        }
    )
    claims["model_workload_binding_digest"] = _sha256_json_digest(
        {
            selected_model: {
                "workload_ids": [],
                "workload_sans": ["secure-model.default.svc.cluster.local"],
            }
        }
    )
    claims["key_release_binding"] = _privatemode_key_release_binding_digest(claims)
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=[selected_model],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy=policy,
    )

    assert has_current_confidential_verification(provider) is False


def test_confidential_model_metadata_includes_public_provider_proof_claims() -> None:
    """Model discovery should carry enough public proof to choose attested routes."""
    model = create_test_model("gpt-secure")
    verified_claims = tinfoil_claims_for_model_ids(["gpt-secure"])
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            verified_claims=verified_claims,
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["gpt-secure"])

    _, _, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    confidentiality = unique_models["gpt-secure"].confidentiality
    assert confidentiality is not None
    assert confidentiality["verified_claims_digest"] == provider_claims_digest(
        verified_claims
    )
    assert confidentiality["proof_claims"] == {
        "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
        "attestation_report_digest": VALID_ATTESTATION_REPORT_DIGEST,
        "attested_hpke_public_key_hex": "b" * 64,
        "code_measurement_fingerprint": _digest("provider-code-measurement"),
        "enclave_measurement_fingerprint": _digest(
            "provider-enclave-measurement"
        ),
        "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
        "payload_policy_digest": VALID_POLICY_DIGEST,
        "release_digest": VALID_RELEASE_DIGEST,
        "repo": "tinfoilsh/confidential-model-router",
        "tls_public_key_fingerprint_sha256": VALID_TLS_DIGEST,
        "transport": "ehbp",
        "model_attestations": {
            "gpt-secure": tinfoil_model_attestation_claims("gpt-secure")
        },
        "verification_steps": {
            "attested_transport_key_binding": True,
            "code_transparency": True,
            "freshness": True,
            "hardware_attestation_report": True,
            "hardware_certificate_chain": True,
            "measurement_match": True,
        },
    }
    serialized = str(confidentiality)
    assert "SECRET_KEY_CONFIG" not in serialized
    assert "SECRET_PROVIDER_KEY" not in serialized


def test_confidential_model_metadata_requires_public_proof_claims_to_cover_public_selectors() -> (
    None
):
    """Published proof claims must bind the exact public model selector casing."""
    model = create_test_model("gpt-secure")
    verified_claims = tinfoil_claims_for_model_ids(["gpt-secure"])
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["GPT-Secure"],
            verified_claims=verified_claims,
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["gpt-secure"])

    assert has_current_confidential_verification(provider) is True

    confidentiality = public_confidentiality_metadata(provider, model, "gpt-secure")

    assert confidentiality is None


def test_required_confidential_routing_excludes_public_metadata_mismatch() -> None:
    """Required-confidential maps must not publish routes without public proof metadata."""
    model = create_test_model("gpt-secure")
    verified_claims = tinfoil_claims_for_model_ids(["gpt-secure"])
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["GPT-Secure"],
            verified_claims=verified_claims,
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["gpt-secure"])

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is True
    assert public_confidentiality_metadata(provider, model, "gpt-secure") is None
    assert model_instances == {}
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_model_metadata_downgrades_malformed_public_claims() -> None:
    """Generated model metadata must not hide malformed proof-bearing claims."""
    model = create_test_model("gpt-secure")
    verified_claims = provider_proof_claims_for_mode("tinfoil")
    verified_claims["manifest_digest"] = "not-a-digest"
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            verified_claims=verified_claims,
        ),
    )

    _, _, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert unique_models == {}
    confidentiality = public_confidentiality_metadata(provider, model, "gpt-secure")
    assert confidentiality is not None
    assert confidentiality["verified"] is False
    assert confidentiality["attestation_status"] == "unavailable"
    assert "proof_claims" not in confidentiality


def test_confidential_model_metadata_omits_unverified_proof_digests() -> None:
    """Unverified public metadata must not expose digest fields as proof evidence."""
    model = create_test_model("gpt-secure")
    verified_claims = provider_proof_claims_for_mode("tinfoil")
    verified_claims["manifest_digest"] = "not-a-digest"
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            verified_claims=verified_claims,
        ),
    )

    confidentiality = public_confidentiality_metadata(provider, model, "gpt-secure")

    assert confidentiality is not None
    assert confidentiality["verified"] is False
    assert confidentiality["attestation_status"] == "unavailable"
    assert "policy_digest" not in confidentiality
    assert "evidence_digest" not in confidentiality
    assert "verified_claims_digest" not in confidentiality


def test_required_confidential_routing_requires_explicit_model_selectors() -> None:
    """Verified providers must not implicitly cover every model in required mode."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(mode="tinfoil"),
    )

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert "gpt-secure" not in model_instances
    assert "gpt-secure" not in provider_map
    assert unique_models == {}


def test_required_confidential_routing_excludes_unroutable_model_instances() -> None:
    """Required mode should not publish aliases with no verified provider."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(mode="tinfoil"),
    )

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert "gpt-secure" not in model_instances
    assert "gpt-secure" not in provider_map
    assert unique_models == {}


def test_required_confidential_alias_collision_uses_verified_model_metadata() -> None:
    """The selected model object must match the verified provider, not a cheaper plain alias."""
    plain_model = create_test_model(
        "gpt-oss-120b", prompt_price=0.001, completion_price=0.001
    )
    plain_model.forwarded_model_id = "plain/gpt-oss-120b"
    confidential_model = create_test_model(
        "gpt-oss-120b", prompt_price=0.02, completion_price=0.02
    )
    confidential_model.forwarded_model_id = "tinfoil/gpt-oss-120b"
    plain_provider = create_test_provider(
        "plain",
        "https://plain.example/v1",
        db_id=1,
        models=[plain_model],
    )
    confidential_provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=2,
        models=[confidential_model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-oss-120b", "tinfoil/gpt-oss-120b"],
            verified_claims=tinfoil_claims_for_model_ids(
                ["gpt-oss-120b", "tinfoil/gpt-oss-120b"]
            ),
        ),
    )
    configure_tinfoil_model_attestation_policy(
        confidential_provider,
        ["gpt-oss-120b", "tinfoil/gpt-oss-120b"],
    )

    model_instances, provider_map, _ = create_model_mappings(
        upstreams=[plain_provider, confidential_provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert provider_map["gpt-oss-120b"] == [confidential_provider]
    assert model_instances["gpt-oss-120b"].forwarded_model_id == (
        "tinfoil/gpt-oss-120b"
    )
    assert model_instances["gpt-oss-120b"].pricing.prompt == pytest.approx(0.02)


def test_required_confidential_alias_accepts_exact_forwarded_backend_attestation() -> (
    None
):
    """A local alias may route when it forwards exactly to the attested backend."""
    model = create_test_model(
        "gpt-secure",
        prompt_price=0.001,
        completion_price=0.001,
    )
    model.forwarded_model_id = "tinfoil/gpt-secure"
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=tinfoil_claims_for_model_ids(["tinfoil/gpt-secure"]),
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["tinfoil/gpt-secure"])

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert provider_map["gpt-secure"] == [provider]
    assert model_instances["gpt-secure"].forwarded_model_id == "tinfoil/gpt-secure"
    assert unique_models["tinfoil/gpt-secure"].id == "gpt-secure"
    assert unique_models["tinfoil/gpt-secure"].confidentiality is not None
    assert unique_models["tinfoil/gpt-secure"].confidentiality["verified"] is True


def test_required_confidential_fallbacks_must_share_selected_forwarding_shape() -> None:
    """Fallback providers must not receive another provider's model object."""
    primary_model = create_test_model(
        "secure-alias",
        prompt_price=0.001,
        completion_price=0.001,
    )
    primary_model.forwarded_model_id = "tinfoil/secure-alias"
    fallback_model = create_test_model(
        "different-secure-backend",
        prompt_price=0.002,
        completion_price=0.002,
    )
    fallback_model.alias_ids = ["secure-alias"]
    fallback_model.forwarded_model_id = "tinfoil/different-secure-backend"
    primary_provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[primary_model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["secure-alias"],
            verified_claims=tinfoil_claims_for_model_ids(["secure-alias"]),
        ),
    )
    configure_tinfoil_model_attestation_policy(primary_provider, ["secure-alias"])
    fallback_provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=2,
        models=[fallback_model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["secure-alias"],
            verified_claims=tinfoil_claims_for_model_ids(["secure-alias"]),
        ),
    )
    configure_tinfoil_model_attestation_policy(fallback_provider, ["secure-alias"])

    model_instances, provider_map, _ = create_model_mappings(
        upstreams=[primary_provider, fallback_provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert model_instances["secure-alias"].forwarded_model_id == "tinfoil/secure-alias"
    assert provider_map["secure-alias"] == [primary_provider]


def test_confidential_alias_must_cover_different_forwarded_backend() -> None:
    """An attested alias must not route to a different unattested forwarded model."""
    model = create_test_model(
        "secure-alias",
        prompt_price=0.001,
        completion_price=0.001,
    )
    model.forwarded_model_id = "tinfoil/different-secure-backend"
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["secure-alias"],
            verified_claims=tinfoil_claims_for_model_ids(["secure-alias"]),
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["secure-alias"])

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert "secure-alias" not in model_instances
    assert "secure-alias" not in provider_map
    assert unique_models == {}


def test_create_model_mappings_marks_unverified_confidential_capability_unavailable() -> (
    None
):
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=SimpleNamespace(
            enabled=True,
            verified=False,
            mode="tinfoil",
            expires_at=None,
            model_ids=["gpt-secure"],
            model_id_prefixes=[],
            verifier=None,
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_claims={},
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=False,
    )

    assert provider_map["gpt-secure"] == [provider]
    assert unique_models["gpt-secure"].confidentiality is not None
    assert unique_models["gpt-secure"].confidentiality["verified"] is False
    assert (
        unique_models["gpt-secure"].confidentiality["attestation_status"]
        == "unavailable"
    )
    assert unique_models["gpt-secure"].confidentiality["verifier"] is None


def test_create_model_mappings_rejects_verified_flag_without_evidence() -> None:
    """A bare verified=True flag is not enough to publish confidential routes."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=SimpleNamespace(
            enabled=True,
            verified=True,
            mode="tinfoil",
            expires_at=None,
            model_ids=[],
            model_id_prefixes=[],
        ),
    )

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert "gpt-secure" not in model_instances
    assert "gpt-secure" not in provider_map
    assert unique_models == {}


def test_create_model_mappings_rejects_truthy_string_confidential_verified() -> None:
    """Malformed proof booleans must not become verified-only routes."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            verified="false",
            model_ids=["gpt-secure"],
        ),
    )

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert "gpt-secure" not in model_instances
    assert "gpt-secure" not in provider_map
    assert unique_models == {}


def test_create_model_mappings_rejects_truthy_string_confidential_enabled() -> None:
    """A malformed enabled flag must not publish verified-only routes."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            enabled="false",
            model_ids=["gpt-secure"],
        ),
    )

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert "gpt-secure" not in model_instances
    assert "gpt-secure" not in provider_map
    assert unique_models == {}


def test_create_model_mappings_rejects_placeholder_evidence_digests() -> None:
    """Digest labels are not proof material for verified-only routing."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            policy_digest="sha256:policy",
            evidence_digest="sha256:evidence",
        ),
    )

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert "gpt-secure" not in model_instances
    assert "gpt-secure" not in provider_map
    assert unique_models == {}


def test_create_model_mappings_rejects_repeated_hex_placeholder_evidence_digests() -> None:
    """Template digest values are not proof material for verified-only routing."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            policy_digest="sha256:" + ("a" * 64),
            evidence_digest="sha256:" + ("b" * 64),
        ),
    )

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert "gpt-secure" not in model_instances
    assert "gpt-secure" not in provider_map
    assert unique_models == {}


def test_create_model_mappings_excludes_confidential_status_without_expiry() -> None:
    """Verified confidentiality evidence must have bounded freshness."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            expires_at=None,
        ),
    )

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert "gpt-secure" not in model_instances
    assert "gpt-secure" not in provider_map
    assert unique_models == {}


def test_create_model_mappings_excludes_stale_confidential_attestation() -> None:
    """Expired attestation evidence must remove a provider from confidential routing."""
    model = create_test_model("gpt-secure")
    stale_provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            expires_at=1,
        ),
    )

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[stale_provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert "gpt-secure" not in model_instances
    assert "gpt-secure" not in provider_map
    assert unique_models == {}


def test_confidential_model_ids_do_not_implicitly_select_prefixed_aliases() -> None:
    """Exact confidential model selectors should not broaden to provider-prefixed aliases."""
    model = create_test_model("tinfoil/gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            verified_claims=tinfoil_claims_for_model_ids(["gpt-secure"]),
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["gpt-secure"])

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert provider_map["gpt-secure"] == [provider]
    assert "tinfoil/gpt-secure" not in provider_map
    assert "gpt-secure" in unique_models


def test_provider_prefixed_confidential_model_ids_do_not_publish_base_alias() -> None:
    """Exact provider-prefixed selectors should not publish unselected base aliases."""
    model = create_test_model("tinfoil/gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=tinfoil_claims_for_model_ids(["tinfoil/gpt-secure"]),
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["tinfoil/gpt-secure"])

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert provider_map["tinfoil/gpt-secure"] == [provider]
    assert "gpt-secure" not in provider_map
    assert "tinfoil/gpt-secure" in unique_models
    assert "gpt-secure" not in unique_models


def test_ppq_private_model_ids_do_not_publish_unselected_base_alias() -> None:
    """PPQ private proof for private/* must not publish a generic base alias."""
    model = create_test_model("private/gpt-oss-120b")
    model.canonical_slug = "gpt-oss-120b"
    model.alias_ids = ["gpt-oss-120b"]
    model.forwarded_model_id = "private/gpt-oss-120b"
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=provider_proof_claims_for_mode("ppq-private-tee"),
        ),
    )
    provider.supports_chat_completions_api = True
    configure_ppq_private_policy(provider)

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert "private/gpt-oss-120b" in provider_map
    assert "gpt-oss-120b" not in provider_map
    assert "private/gpt-oss-120b" in unique_models
    assert "gpt-oss-120b" not in unique_models


def test_ppq_private_status_accepts_raw_attestation_bundle_url_policy() -> None:
    """Runtime PPQ policy can bind the private bundle by URL before sanitization."""
    model_id = "private/gpt-oss-120b"
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=[model_id],
            verified_claims=ppq_private_claims_for_model_ids([model_id]),
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="ppq-private-tee",
        model_ids=[model_id],
        model_id_prefixes=[],
        policy={
            "repo": "ppq-ai/private-tee",
            "attestation_bundle_url": VALID_PPQ_ATTESTATION_BUNDLE_URL,
            "expected_code_measurement_fingerprint": _digest(
                "provider-code-measurement"
            ),
            "proxy_binary_digest": _digest("privatemode-proxy-binary"),
            "require_model_attestations": True,
            "model_attestation_targets": {
                model_id: {
                    "host": "gpt-oss-120b.tinfoil.example",
                    "repo": "tinfoilsh/confidential-gpt-oss-120b",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                }
            },
        },
    )

    assert has_current_confidential_verification(provider) is True


def test_tinfoil_prefix_status_is_not_current_confidential_verification() -> None:
    """Route-time proof checks must not accept stale broad Tinfoil selectors."""
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_id_prefixes=["tinfoil/"],
        ),
    )

    assert has_current_confidential_verification(provider) is False


def test_tinfoil_selected_model_status_requires_model_attestation_claims() -> None:
    """Route-time proof checks must not trust router-only proof for selected models."""
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=provider_proof_claims_for_mode("tinfoil"),
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="tinfoil",
        model_ids=["tinfoil/gpt-secure"],
        model_id_prefixes=[],
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": VALID_RELEASE_DIGEST,
            "require_model_attestations": True,
            "model_attestation_targets": {
                "tinfoil/gpt-secure": {
                    "host": "gpt-secure.tinfoil.example",
                    "repo": "tinfoilsh/confidential-gpt-secure",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                }
            },
        },
    )

    assert has_current_confidential_verification(provider) is False


def test_tinfoil_selected_model_status_requires_model_attestation_policy() -> None:
    """Route-time proof checks must not rely on preflight to require model proofs."""
    claims = provider_proof_claims_for_mode("tinfoil")
    claims["model_attestations"] = {
        "tinfoil/gpt-secure": {
            "repo": "tinfoilsh/confidential-gpt-secure",
            "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
            "attestation_report_digest": VALID_ATTESTATION_REPORT_DIGEST,
            "attested_hpke_public_key_hex": "b" * 64,
            "enclave_measurement_fingerprint": _digest(
                "model-enclave-measurement"
            ),
            "code_measurement_fingerprint": _digest("model-code-measurement"),
            "tls_public_key_fingerprint_sha256": VALID_TLS_DIGEST,
            "release_digest": VALID_RELEASE_DIGEST,
            "verification_steps": _ehbp_verification_steps(),
        }
    }
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="tinfoil",
        model_ids=["tinfoil/gpt-secure"],
        model_id_prefixes=[],
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": VALID_RELEASE_DIGEST,
        },
    )

    assert has_current_confidential_verification(provider) is False


def test_tinfoil_selected_model_status_rechecks_model_attestation_policy_pins() -> None:
    """Copied model proof must not bypass local release/code-measurement pins."""
    claims = provider_proof_claims_for_mode("tinfoil")
    claims["model_attestations"] = {
        "tinfoil/gpt-secure": {
            "repo": "tinfoilsh/confidential-gpt-secure",
            "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
            "attestation_report_digest": VALID_ATTESTATION_REPORT_DIGEST,
            "attested_hpke_public_key_hex": "b" * 64,
            "enclave_measurement_fingerprint": _digest(
                "model-enclave-measurement"
            ),
            "code_measurement_fingerprint": _digest("model-code-measurement"),
            "tls_public_key_fingerprint_sha256": VALID_TLS_DIGEST,
            "release_digest": "sha256:" + ("f" * 64),
            "verification_steps": _ehbp_verification_steps(),
        }
    }
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="tinfoil",
        model_ids=["tinfoil/gpt-secure"],
        model_id_prefixes=[],
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": VALID_RELEASE_DIGEST,
            "require_model_attestations": True,
            "model_attestation_targets": {
                "tinfoil/gpt-secure": {
                    "host": "gpt-secure.tinfoil.example",
                    "repo": "tinfoilsh/confidential-gpt-secure",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                    "expected_code_measurement_fingerprint": "sha256:" + ("3" * 64),
                }
            },
        },
    )

    assert has_current_confidential_verification(provider) is False


def test_tinfoil_selected_model_status_rejects_conflicting_release_policy() -> None:
    """Copied model proof must not bypass exact/allowed release pin consistency."""
    claims = tinfoil_claims_for_model_ids(["tinfoil/gpt-secure"])
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="tinfoil",
        model_ids=["tinfoil/gpt-secure"],
        model_id_prefixes=[],
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": VALID_RELEASE_DIGEST,
            "require_model_attestations": True,
            "model_attestation_targets": {
                "tinfoil/gpt-secure": {
                    "host": "gpt-secure.tinfoil.example",
                    "repo": "tinfoilsh/confidential-gpt-secure",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                    "allowed_release_digests": ["sha256:" + ("f" * 64)],
                },
            },
        },
    )

    assert has_current_confidential_verification(provider) is False


def test_tinfoil_selected_model_status_rechecks_router_policy_pins() -> None:
    """Copied router proof must not bypass local Tinfoil router release pins."""
    claims = tinfoil_claims_for_model_ids(["tinfoil/gpt-secure"])
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="tinfoil",
        model_ids=["tinfoil/gpt-secure"],
        model_id_prefixes=[],
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": "sha256:" + ("f" * 64),
            "require_model_attestations": True,
            "model_attestation_targets": {
                "tinfoil/gpt-secure": {
                    "host": "gpt-secure.tinfoil.example",
                    "repo": "tinfoilsh/confidential-gpt-secure",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                },
            },
        },
    )

    assert has_current_confidential_verification(provider) is False


def test_tinfoil_status_rejects_placeholder_attestation_report_claim() -> None:
    """Route-time Tinfoil proof must require concrete attestation evidence."""
    claims = tinfoil_claims_for_model_ids(["tinfoil/gpt-secure"])
    claims["attestation_report_digest"] = PLACEHOLDER_DIGEST
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=claims,
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["tinfoil/gpt-secure"])

    assert has_current_confidential_verification(provider) is False


def test_tinfoil_selected_model_status_accepts_matching_model_attestation_claims() -> None:
    """Selected Tinfoil routes stay routable when router and model proofs match."""
    claims = provider_proof_claims_for_mode("tinfoil")
    claims["model_attestations"] = {
        "tinfoil/gpt-secure": {
            "repo": "tinfoilsh/confidential-gpt-secure",
            "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
            "attestation_report_digest": VALID_ATTESTATION_REPORT_DIGEST,
            "attested_hpke_public_key_hex": "b" * 64,
            "enclave_measurement_fingerprint": _digest(
                "model-enclave-measurement"
            ),
            "code_measurement_fingerprint": _digest("model-code-measurement"),
            "tls_public_key_fingerprint_sha256": VALID_TLS_DIGEST,
            "release_digest": VALID_RELEASE_DIGEST,
            "verification_steps": _ehbp_verification_steps(),
        }
    }
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="tinfoil",
        model_ids=["tinfoil/gpt-secure"],
        model_id_prefixes=[],
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": VALID_RELEASE_DIGEST,
            "require_model_attestations": True,
            "model_attestation_targets": {
                "tinfoil/gpt-secure": {
                    "host": "gpt-secure.tinfoil.example",
                    "repo": "tinfoilsh/confidential-gpt-secure",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                }
            },
        },
    )

    assert has_current_confidential_verification(provider) is True


def test_tinfoil_ehbp_status_accepts_matching_claims_without_tls_binding() -> None:
    """Explicit EHBP Tinfoil policy can route without a TLS fingerprint claim."""
    model_id = "tinfoil/gpt-secure"
    claims = tinfoil_claims_for_model_ids([model_id])
    claims.pop("tls_public_key_fingerprint_sha256")
    model_claims = claims["model_attestations"][model_id]
    assert isinstance(model_claims, dict)
    model_claims.pop("tls_public_key_fingerprint_sha256")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[create_test_model(model_id)],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=[model_id],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="tinfoil",
        model_ids=[model_id],
        model_id_prefixes=[],
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "transport_security": "ehbp",
            "expected_release_digest": VALID_RELEASE_DIGEST,
            "require_model_attestations": True,
            "model_attestation_targets": {
                model_id: {
                    "host": "gpt-secure.tinfoil.example",
                    "repo": "tinfoilsh/confidential-gpt-secure",
                    "transport_security": "ehbp",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                }
            },
        },
    )

    assert has_current_confidential_verification(provider) is True

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert model_id in model_instances
    assert provider_map[model_id] == [provider]
    assert unique_models[model_id].upstream_provider_id == "tinfoil"


def test_tinfoil_ehbp_status_rejects_malformed_present_tls_binding() -> None:
    """Explicit EHBP policy must not ignore malformed TLS binding evidence."""
    model_id = "tinfoil/gpt-secure"
    claims = tinfoil_claims_for_model_ids([model_id])
    claims["tls_public_key_fingerprint_sha256"] = "not-a-sha256"
    model_claims = claims["model_attestations"][model_id]
    assert isinstance(model_claims, dict)
    model_claims["tls_public_key_fingerprint_sha256"] = "not-a-sha256"
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=[model_id],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "transport_security": "ehbp",
            "expected_release_digest": VALID_RELEASE_DIGEST,
            "require_model_attestations": True,
            "model_attestation_targets": {
                model_id: {
                    "host": "gpt-secure.tinfoil.example",
                    "repo": "tinfoilsh/confidential-gpt-secure",
                    "transport_security": "ehbp",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                }
            },
        },
    )

    assert has_current_confidential_verification(provider) is False


def test_tinfoil_status_rejects_non_default_provider_base_url() -> None:
    """Route-time Tinfoil proof must stay bound to the attested public router."""
    claims = tinfoil_claims_for_model_ids(["tinfoil/gpt-secure"])
    provider = create_test_provider(
        "tinfoil",
        "https://attacker.example/v1",
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=claims,
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["tinfoil/gpt-secure"])

    assert has_current_confidential_verification(provider) is False


def test_tinfoil_status_rejects_duplicate_selected_model_policy_entries() -> None:
    """Copied Tinfoil proof must not hide duplicate selected model policy entries."""
    claims = tinfoil_claims_for_model_ids(["tinfoil/gpt-secure"])
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure", "tinfoil/gpt-secure"],
            verified_claims=claims,
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["tinfoil/gpt-secure"])

    assert has_current_confidential_verification(provider) is False


def test_tinfoil_route_time_policy_rejects_conflicting_target_aliases() -> None:
    """Route-time Tinfoil target aliases must describe one selected model map."""
    claims = tinfoil_claims_for_model_ids(["tinfoil/gpt-secure"])
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": VALID_RELEASE_DIGEST,
            "require_model_attestations": True,
            "model_attestation_targets": {
                "tinfoil/gpt-secure": {
                    "host": "gpt-secure.tinfoil.example",
                    "repo": "tinfoilsh/confidential-gpt-secure",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                },
            },
            "modelAttestationTargets": {
                "tinfoil/gpt-secure": {
                    "host": "other-model.tinfoil.example",
                    "repo": "tinfoilsh/confidential-other-model",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                },
            },
        },
    )

    assert has_current_confidential_verification(provider) is False


def test_tinfoil_route_time_policy_rejects_case_variant_target_keys() -> None:
    """Route-time Tinfoil target maps must not hide duplicate model keys by case."""
    claims = tinfoil_claims_for_model_ids(["tinfoil/gpt-secure"])
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": VALID_RELEASE_DIGEST,
            "require_model_attestations": True,
            "model_attestation_targets": {
                "tinfoil/gpt-secure": {
                    "host": "gpt-secure.tinfoil.example",
                    "repo": "tinfoilsh/confidential-gpt-secure",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                },
                "TINFOIL/gpt-secure": {
                    "host": "other-model.tinfoil.example",
                    "repo": "tinfoilsh/confidential-other-model",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                },
            },
        },
    )

    assert has_current_confidential_verification(provider) is False


def test_tinfoil_route_time_policy_rejects_conflicting_target_repo_aliases() -> None:
    """Route-time Tinfoil target identity aliases must match preflight rules."""
    claims = tinfoil_claims_for_model_ids(["tinfoil/gpt-secure"])
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": VALID_RELEASE_DIGEST,
            "require_model_attestations": True,
            "model_attestation_targets": {
                "tinfoil/gpt-secure": {
                    "host": "gpt-secure.tinfoil.example",
                    "repo": "tinfoilsh/confidential-gpt-secure",
                    "expected_repo": "attacker/conflicting-gpt-secure",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                },
            },
        },
    )

    assert has_current_confidential_verification(provider) is False


def test_tinfoil_status_rejects_non_list_selected_model_policy_entries() -> None:
    """Route-time selected model policy must be JSON-list shaped."""
    claims = tinfoil_claims_for_model_ids(["tinfoil/gpt-secure"])
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=("tinfoil/gpt-secure",),
            verified_claims=claims,
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["tinfoil/gpt-secure"])

    assert has_current_confidential_verification(provider) is False


def test_ppq_private_prefix_status_is_not_current_confidential_verification() -> None:
    """Route-time proof checks must not accept stale broad PPQ selectors."""
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_id_prefixes=["private/"],
        ),
    )

    assert has_current_confidential_verification(provider) is False


def test_ppq_private_policy_prefix_is_not_current_confidential_verification() -> None:
    """Route-time proof checks must not accept broad PPQ policy selectors."""
    claims = ppq_private_claims_for_model_ids(["private/gpt-oss-120b"])
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=claims,
        ),
    )
    configure_ppq_private_policy(provider)
    provider.confidentiality_policy.return_value.model_ids = []
    provider.confidentiality_policy.return_value.model_id_prefixes = ["private/"]

    assert has_current_confidential_verification(provider) is False


def test_ppq_private_policy_non_string_model_id_fails_closed() -> None:
    """Malformed selected-model policy selectors must not crash routing checks."""
    claims = ppq_private_claims_for_model_ids(["private/gpt-oss-120b"])
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=claims,
        ),
    )
    configure_ppq_private_policy(provider)
    provider.confidentiality_policy.return_value.model_ids = [
        "private/gpt-oss-120b",
        42,
    ]

    assert has_current_confidential_verification(provider) is False


def test_ppq_private_status_rechecks_present_policy_release_pin() -> None:
    """Copied PPQ private proof must not bypass local release identity pins."""
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=provider_proof_claims_for_mode("ppq-private-tee"),
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy={
            "repo": "ppq-ai/private-tee",
            "expected_release_digest": "sha256:" + ("f" * 64),
        },
    )

    assert has_current_confidential_verification(provider) is False


def test_ppq_private_status_requires_local_artifact_identity_pin() -> None:
    """Copied PPQ private proof must not bypass local artifact identity pins."""
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=provider_proof_claims_for_mode("ppq-private-tee"),
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy={
            "repo": "ppq-ai/private-tee",
        },
    )

    assert has_current_confidential_verification(provider) is False


def test_ppq_private_status_accepts_ehbp_without_local_proxy_binary_pin() -> None:
    """PPQ private routing is bound to EHBP proof, not a local proxy artifact."""
    claims = ppq_private_claims_for_model_ids(["private/gpt-oss-120b"])
    claims.pop("proxy_binary_digest", None)
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="ppq-private-tee",
        model_ids=["private/gpt-oss-120b"],
        model_id_prefixes=[],
        policy={
            "repo": "ppq-ai/private-tee",
            "attestation_bundle_url": VALID_PPQ_ATTESTATION_BUNDLE_URL,
            "attestation_bundle_url_digest": VALID_PPQ_ATTESTATION_BUNDLE_URL_DIGEST,
            "expected_code_measurement_fingerprint": _digest(
                "provider-code-measurement"
            ),
            "require_model_attestations": True,
            "model_attestation_targets": {
                "private/gpt-oss-120b": {
                    "host": "gpt-oss-120b.tinfoil.example",
                    "repo": "tinfoilsh/confidential-gpt-oss-120b",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                }
            },
        },
    )

    assert has_current_confidential_verification(provider) is True


def test_ppq_private_status_ignores_placeholder_proxy_binary_claim() -> None:
    """Legacy PPQ proxy artifact claims are not part of the routing predicate."""
    claims = ppq_private_claims_for_model_ids(["private/gpt-oss-120b"])
    claims["proxy_binary_digest"] = PLACEHOLDER_DIGEST
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=claims,
        ),
    )
    configure_ppq_private_policy(provider)

    assert has_current_confidential_verification(provider) is True


def test_ppq_private_status_ignores_placeholder_proxy_binary_policy_pin() -> None:
    """Legacy PPQ proxy artifact policy pins are not part of the predicate."""
    claims = ppq_private_claims_for_model_ids(["private/gpt-oss-120b"])
    claims["proxy_binary_digest"] = PLACEHOLDER_DIGEST
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=claims,
        ),
    )
    configure_ppq_private_policy(provider)
    policy = dict(provider.confidentiality_policy.return_value.policy)
    policy["proxy_binary_digest"] = PLACEHOLDER_DIGEST
    original_policy = provider.confidentiality_policy.return_value
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode=original_policy.mode,
        model_ids=list(original_policy.model_ids),
        model_id_prefixes=list(original_policy.model_id_prefixes),
        policy=policy,
    )

    assert has_current_confidential_verification(provider) is True


def test_ppq_private_status_ignores_conflicting_proxy_binary_policy_aliases() -> None:
    """Legacy PPQ proxy aliases do not affect EHBP-backed routing."""
    claims = provider_proof_claims_for_mode("ppq-private-tee")
    claims["proxy_binary_digest"] = "sha256:" + ("6" * 64)
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="ppq-private-tee",
        model_ids=["private/gpt-oss-120b"],
        model_id_prefixes=[],
        policy={
            "repo": "ppq-ai/private-tee",
            "attestation_bundle_url": VALID_PPQ_ATTESTATION_BUNDLE_URL,
            "expected_code_measurement_fingerprint": _digest(
                "provider-code-measurement"
            ),
            "proxy_binary_digest": "sha256:" + ("6" * 64),
            "proxyBinaryDigest": "sha256:" + ("7" * 64),
            "require_model_attestations": True,
            "model_attestation_targets": {
                "private/gpt-oss-120b": {
                    "host": "gpt-oss-120b.tinfoil.example",
                    "repo": "tinfoilsh/confidential-gpt-oss-120b",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                }
            },
        },
    )

    assert has_current_confidential_verification(provider) is True


def test_ppq_private_status_requires_local_policy_body() -> None:
    """Copied PPQ private proof must not route with only a matching policy digest."""
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=provider_proof_claims_for_mode("ppq-private-tee"),
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
    )

    assert has_current_confidential_verification(provider) is False


def test_ppq_private_status_requires_local_attestation_bundle_policy() -> None:
    """Copied PPQ private proof must be bound to the local private bundle policy."""
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=provider_proof_claims_for_mode("ppq-private-tee"),
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy={
            "repo": "ppq-ai/private-tee",
            "expected_code_measurement_fingerprint": "sha256:" + ("2" * 64),
        },
    )

    assert has_current_confidential_verification(provider) is False


def test_ppq_private_status_rechecks_local_attestation_bundle_policy() -> None:
    """Copied PPQ private proof must not bypass the configured bundle URL."""
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=provider_proof_claims_for_mode("ppq-private-tee"),
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy={
            "repo": "ppq-ai/private-tee",
            "expected_code_measurement_fingerprint": "sha256:" + ("2" * 64),
            "attestation_bundle_url": "https://api.ppq.ai/private/other-bundle",
        },
    )

    assert has_current_confidential_verification(provider) is False


def test_ppq_private_status_rejects_malformed_attestation_bundle_digest_policy() -> None:
    """Copied PPQ proof must not bypass a malformed present bundle digest pin."""
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=provider_proof_claims_for_mode("ppq-private-tee"),
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy={
            "repo": "ppq-ai/private-tee",
            "expected_code_measurement_fingerprint": "sha256:" + ("2" * 64),
            "attestation_bundle_url": VALID_PPQ_ATTESTATION_BUNDLE_URL,
            "attestation_bundle_url_digest": "not-a-sha256-digest",
            "proxy_binary_digest": "sha256:" + ("6" * 64),
        },
    )

    assert has_current_confidential_verification(provider) is False


def test_ppq_private_status_rejects_conflicting_repo_aliases() -> None:
    """Copied PPQ private proof must not bypass local repo alias consistency."""
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=provider_proof_claims_for_mode("ppq-private-tee"),
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy={
            "repo": "ppq-ai/private-tee",
            "expected_repo": "attacker/conflicting-private-tee",
        },
    )

    assert has_current_confidential_verification(provider) is False


def test_ppq_private_status_requires_local_repo_policy_pin() -> None:
    """Copied PPQ private proof must be bound to local provenance identity."""
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=provider_proof_claims_for_mode("ppq-private-tee"),
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy={
            "attestation_bundle_url": VALID_PPQ_ATTESTATION_BUNDLE_URL,
            "attestation_bundle_url_digest": VALID_PPQ_ATTESTATION_BUNDLE_URL_DIGEST,
            "expected_code_measurement_fingerprint": _digest(
                "provider-code-measurement"
            ),
            "proxy_binary_digest": _digest("privatemode-proxy-binary"),
        },
    )

    assert has_current_confidential_verification(provider) is False


def test_ppq_private_status_requires_backend_model_attestation_proof() -> None:
    """PPQ private proof must bind the mapped downstream model enclave too."""
    claims = provider_proof_claims_for_mode("ppq-private-tee")
    claims["backend_model_attestations"] = None
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy={
            "repo": "ppq-ai/private-tee",
            "attestation_bundle_url": VALID_PPQ_ATTESTATION_BUNDLE_URL,
            "attestation_bundle_url_digest": VALID_PPQ_ATTESTATION_BUNDLE_URL_DIGEST,
            "expected_code_measurement_fingerprint": _digest(
                "provider-code-measurement"
            ),
            "proxy_binary_digest": _digest("privatemode-proxy-binary"),
            "model_attestation_targets": {
                "private/gpt-oss-120b": {
                    "host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                    "repo": "tinfoilsh/confidential-gpt-oss-120b",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                }
            },
        },
    )

    assert has_current_confidential_verification(provider) is False


def test_ppq_private_status_accepts_router_code_measurement_policy_pin() -> None:
    """Route-time PPQ private policy can pin router code measurement."""
    claims = ppq_private_claims_for_model_ids(["private/gpt-oss-120b"])
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="ppq-private-tee",
        model_ids=["private/gpt-oss-120b"],
        model_id_prefixes=[],
        policy={
            "repo": "ppq-ai/private-tee",
            "attestation_bundle_url": VALID_PPQ_ATTESTATION_BUNDLE_URL,
            "attestation_bundle_url_digest": VALID_PPQ_ATTESTATION_BUNDLE_URL_DIGEST,
            "expected_code_measurement_fingerprint": _digest(
                "provider-code-measurement"
            ),
            "proxy_binary_digest": _digest("privatemode-proxy-binary"),
            "require_model_attestations": True,
            "model_attestation_targets": {
                "private/gpt-oss-120b": {
                    "host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                    "repo": "tinfoilsh/confidential-gpt-oss-120b",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                }
            },
        },
    )

    assert has_current_confidential_verification(provider) is True


def test_ppq_private_status_requires_router_release_or_code_policy_pin() -> None:
    """Route-time PPQ private policy still needs a router identity pin."""
    claims = ppq_private_claims_for_model_ids(["private/gpt-oss-120b"])
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=claims,
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy={
            "repo": "ppq-ai/private-tee",
            "attestation_bundle_url": VALID_PPQ_ATTESTATION_BUNDLE_URL,
            "proxy_binary_digest": _digest("privatemode-proxy-binary"),
            "require_model_attestations": True,
            "model_attestation_targets": {
                "private/gpt-oss-120b": {
                    "host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                    "repo": "tinfoilsh/confidential-gpt-oss-120b",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                }
            },
        },
    )

    assert has_current_confidential_verification(provider) is False


def test_ppq_private_status_rejects_selected_model_claim_mismatch() -> None:
    """Copied PPQ private proof must bind to the exact selected private model set."""
    claims = provider_proof_claims_for_mode("ppq-private-tee")
    claims["selected_model_ids"] = ["private/kimi-k2-6"]
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=claims,
        ),
    )

    assert has_current_confidential_verification(provider) is False


def test_ppq_private_status_rejects_duplicate_selected_model_claims() -> None:
    """Copied PPQ private proof must not hide duplicate selected model claims."""
    claims = provider_proof_claims_for_mode("ppq-private-tee")
    claims["selected_model_ids"] = [
        "private/gpt-oss-120b",
        "private/gpt-oss-120b",
    ]
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=claims,
        ),
    )

    assert has_current_confidential_verification(provider) is False


def test_ppq_private_status_rejects_scalar_selected_model_claims() -> None:
    """Route-time PPQ private proof must require JSON-list selected model claims."""
    claims = provider_proof_claims_for_mode("ppq-private-tee")
    claims["selected_model_ids"] = "private/gpt-oss-120b"
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/gpt-oss-120b"],
            verified_claims=claims,
        ),
    )

    assert has_current_confidential_verification(provider) is False


def test_privatemode_status_rejects_duplicate_selected_model_policy_entries() -> None:
    """Copied Privatemode proof must not hide duplicate selected model policy entries."""
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model", "privatemode/secure-model"],
            verified_claims=_privatemode_claims_matching_selected_model_policy(),
        ),
    )
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="privatemode",
        model_ids=["privatemode/secure-model"],
        model_id_prefixes=[],
        policy=_privatemode_selected_model_policy(),
    )

    assert has_current_confidential_verification(provider) is False


def test_privatemode_status_rejects_case_variant_workload_binding_keys() -> None:
    """Copied Privatemode proof must not hide duplicate workload bindings by case."""
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model"],
            verified_claims=_privatemode_claims_matching_selected_model_policy(),
        ),
    )
    policy = _privatemode_selected_model_policy()
    policy["model_workload_bindings"] = {
        "privatemode/secure-model": {
            "workload_sans": ["secure-model.default.svc.cluster.local"],
        },
        "PRIVATEMODE/secure-model": {
            "workload_sans": ["other.default.svc.cluster.local"],
        },
    }
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="privatemode",
        model_ids=["privatemode/secure-model"],
        model_id_prefixes=[],
        policy=policy,
    )

    assert has_current_confidential_verification(provider) is False


def test_privatemode_status_rechecks_present_component_policy_pin() -> None:
    """Copied Privatemode proof must not bypass local component identity pins."""
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model"],
            verified_claims=_privatemode_claims_matching_selected_model_policy(),
        ),
    )
    policy = _privatemode_selected_model_policy()
    policy["expected_coordinator_measurement"] = "sha256:" + ("f" * 64)
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy=policy,
    )

    assert has_current_confidential_verification(provider) is False


def test_privatemode_status_rejects_component_expected_pin_not_allowed() -> None:
    """Copied Privatemode proof must not bypass conflicting expected/allowed pins."""
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model"],
            verified_claims=_privatemode_claims_matching_selected_model_policy(),
        ),
    )
    policy = _privatemode_selected_model_policy()
    policy["expected_coordinator_measurement"] = "sha256:" + ("f" * 64)
    policy["allowed_coordinator_measurements"] = ["sha256:" + ("3" * 64)]
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy=policy,
    )

    assert has_current_confidential_verification(provider) is False


def test_privatemode_status_rejects_gpu_policy_expected_pin_not_allowed() -> None:
    """Copied Privatemode proof must not bypass conflicting GPU policy pins."""
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model"],
            verified_claims=_privatemode_claims_matching_selected_model_policy(),
        ),
    )
    policy = _privatemode_selected_model_policy()
    policy["expected_gpu_attestation_policy"] = "strict-gpu-policy"
    policy["allowed_gpu_attestation_policies"] = ["nvidia-ocsp-good-only"]
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy=policy,
    )

    assert has_current_confidential_verification(provider) is False


def test_privatemode_status_rejects_conflicting_component_policy_aliases() -> None:
    """Copied Privatemode proof must not bypass ambiguous local component pins."""
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=["privatemode/secure-model"],
            verified_claims=_privatemode_claims_matching_selected_model_policy(),
        ),
    )
    policy = _privatemode_selected_model_policy()
    policy["proxyBinaryDigest"] = "sha256:" + ("7" * 64)
    provider.confidentiality_policy.return_value = SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        policy=policy,
    )

    assert has_current_confidential_verification(provider) is False


def test_create_model_mappings_honors_exact_confidential_model_ids() -> None:
    """Provider-level attestation may cover only selected exact model IDs."""
    private_model = create_test_model("private/llama-3.1")
    public_model = create_test_model("public/llama-3.1")
    ppq_provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        db_id=1,
        models=[private_model, public_model],
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/llama-3.1"],
        ),
    )
    configure_ppq_private_policy(ppq_provider)

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[ppq_provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert provider_map["private/llama-3.1"] == [ppq_provider]
    assert "llama-3.1" not in provider_map
    assert "public/llama-3.1" not in provider_map
    assert "private/llama-3.1" in unique_models
    assert "llama-3.1" not in unique_models
    assert "public/llama-3.1" not in unique_models


def test_create_model_mappings_rejects_ppq_private_without_bundle_binding() -> None:
    """PPQ private routing needs proof of the private attestation bundle path."""
    private_model = create_test_model("private/llama-3.1")
    claims = provider_proof_claims_for_mode("ppq-private-tee")
    claims.pop("attestation_bundle_url", None)
    claims.pop("attestation_bundle_url_digest", None)
    ppq_provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        db_id=1,
        models=[private_model],
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/llama-3.1"],
            verified_claims=claims,
        ),
    )
    configure_ppq_private_policy(ppq_provider)

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[ppq_provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert "private/llama-3.1" not in model_instances
    assert "private/llama-3.1" not in provider_map
    assert unique_models == {}


def test_create_model_mappings_rejects_ppq_private_bundle_host_mismatch() -> None:
    """PPQ private proof must bind the private bundle URL to the routed host."""
    private_model = create_test_model("private/llama-3.1")
    claims = provider_proof_claims_for_mode("ppq-private-tee")
    claims["attestation_bundle_url"] = "https://attest.ppq.ai/private"
    claims["attestation_bundle_url_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                claims["attestation_bundle_url"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    ppq_provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        db_id=1,
        models=[private_model],
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/llama-3.1"],
            verified_claims=claims,
        ),
    )
    configure_ppq_private_policy(
        ppq_provider,
        attestation_bundle_url="https://attest.ppq.ai/private",
    )

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[ppq_provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert "private/llama-3.1" not in model_instances
    assert "private/llama-3.1" not in provider_map
    assert unique_models == {}


def test_create_model_mappings_rejects_ppq_private_bundle_origin_mismatch() -> None:
    """PPQ private proof must bind the private bundle URL to the routed origin."""
    private_model = create_test_model("private/llama-3.1")
    claims = provider_proof_claims_for_mode("ppq-private-tee")
    claims["attestation_bundle_url"] = "https://api.ppq.ai/private"
    claims["attestation_bundle_url_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                claims["attestation_bundle_url"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    ppq_provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai:8443/private/v1",
        db_id=1,
        models=[private_model],
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/llama-3.1"],
            verified_claims=claims,
        ),
    )
    configure_ppq_private_policy(ppq_provider)

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[ppq_provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert "private/llama-3.1" not in model_instances
    assert "private/llama-3.1" not in provider_map
    assert unique_models == {}


def test_confidential_selector_values_must_be_strings_to_route() -> None:
    """Malformed selector metadata must not be coerced into routable strings."""
    numeric_model = create_test_model("123")
    prefixed_model = create_test_model("private/llama-3.1")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[numeric_model, prefixed_model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=[123],
            model_id_prefixes=[{"prefix": "private/"}],
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert provider_map == {}
    assert unique_models == {}


def test_confidential_selector_malformed_alias_does_not_hide_behind_valid_selector() -> (
    None
):
    """Every present selector entry is route policy, even when another one matches."""
    model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=[
                "gpt-secure",
                {"unexpected": "object"},
            ],
        ),
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert provider_map == {}
    assert unique_models == {}


def test_confidential_selector_case_variant_alias_does_not_hide_behind_valid_selector() -> (
    None
):
    """Case-variant selector aliases must not become separate proof coverage."""
    model = create_test_model("private/llama-3.1")
    provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=[
                "Private/llama-3.1",
                "private/llama-3.1",
            ],
        ),
    )
    configure_ppq_private_policy(provider)

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_known_provider_selector_match_rejects_case_variant_duplicate_model_ids() -> (
    None
):
    provider = SimpleNamespace(provider_type="tinfoil")

    assert (
        _known_provider_selectors_match(
            provider,
            ["TINFOIL/gpt-secure", "tinfoil/gpt-secure"],
            [],
        )
        is False
    )


def test_create_model_mappings_exposes_confidential_endpoint_capabilities() -> None:
    """Model metadata should match the endpoints confidential routing will allow."""
    private_model = create_test_model("private/llama-3.1")
    ppq_provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        db_id=1,
        models=[private_model],
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/llama-3.1"],
        ),
    )
    configure_ppq_private_policy(ppq_provider)
    ppq_provider.supports_embeddings_api = False
    ppq_provider.supports_responses_api = False

    _, _, unique_models = create_model_mappings(
        upstreams=[ppq_provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    confidentiality = unique_models["private/llama-3.1"].confidentiality
    assert confidentiality is not None
    assert confidentiality["supported_endpoints"] == ["/v1/chat/completions"]


def test_required_confidential_routing_publishes_only_privatemode_selected_alias() -> (
    None
):
    """Privatemode raw proxy IDs should not bypass exact selected model routing."""
    selected_model = "privatemode/gpt-oss-120b"
    workload_sans = ["gpt-oss-120b.default.svc.cluster.local"]
    claims = provider_proof_claims_for_mode("privatemode")
    claims["selected_model_ids"] = [selected_model]
    claims["expected_workload_identity_digest"] = _sha256_json_digest(
        {"ids": [], "sans": workload_sans}
    )
    claims["model_workload_binding_digest"] = _sha256_json_digest(
        {
            selected_model: {
                "workload_ids": [],
                "workload_sans": workload_sans,
            }
        }
    )
    claims["key_release_binding"] = _privatemode_key_release_binding_digest(claims)
    model = create_test_model("gpt-oss-120b")
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=[selected_model],
            verified_claims=claims,
        ),
    )
    configure_privatemode_policy(
        provider,
        model_ids=[selected_model],
        workload_sans=workload_sans,
    )

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert "gpt-oss-120b" not in model_instances
    assert selected_model in model_instances
    assert selected_model in provider_map
    assert selected_model in unique_models
    assert unique_models[selected_model].confidentiality is not None
    assert unique_models[selected_model].confidentiality["verified"] is True


def test_required_confidential_routing_rejects_unprefixed_privatemode_selector() -> (
    None
):
    """Privatemode verifier proof must bind Routstr's public privatemode/* alias."""
    unprefixed_model = "gpt-oss-120b"
    workload_sans = ["gpt-oss-120b.default.svc.cluster.local"]
    claims = provider_proof_claims_for_mode("privatemode")
    claims["selected_model_ids"] = [unprefixed_model]
    claims["expected_workload_identity_digest"] = _sha256_json_digest(
        {"ids": [], "sans": workload_sans}
    )
    claims["model_workload_binding_digest"] = _sha256_json_digest(
        {
            unprefixed_model: {
                "workload_ids": [],
                "workload_sans": workload_sans,
            }
        }
    )
    claims["key_release_binding"] = _privatemode_key_release_binding_digest(claims)
    model = create_test_model(unprefixed_model)
    provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="privatemode",
            model_ids=[unprefixed_model],
            verified_claims=claims,
        ),
    )
    configure_privatemode_policy(
        provider,
        model_ids=[unprefixed_model],
        workload_sans=workload_sans,
    )

    _, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert has_current_confidential_verification(provider) is False
    assert provider_map == {}
    assert unique_models == {}


def test_confidential_model_metadata_filters_model_endpoint_capabilities() -> None:
    """A verified provider endpoint is not enough when the model cannot serve it."""
    model = create_test_model("gpt-secure")
    model.supported_endpoints = ["/v1/chat/completions", "/v1/responses"]
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
        ),
    )

    confidentiality = public_confidentiality_metadata(provider, model, "gpt-secure")

    assert confidentiality is not None
    assert confidentiality["supported_endpoints"] == [
        "/v1/chat/completions",
        "/v1/responses",
    ]


def test_required_confidential_routing_excludes_models_with_no_supported_endpoint() -> (
    None
):
    """Verified catalog presence is not enough if Routstr cannot serve any endpoint."""
    model = create_test_model("qwen3-tts")
    model.architecture = Architecture(
        modality="text->audio",
        input_modalities=["text"],
        output_modalities=["audio"],
        tokenizer="unknown",
        instruct_type=None,
    )
    model.supported_endpoints = ["/v1/audio/speech"]
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["qwen3-tts"],
        ),
    )

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert model_instances == {}
    assert provider_map == {}
    assert unique_models == {}


def test_required_confidential_routing_db_override_cannot_broaden_catalog_endpoint_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admin model overrides must not replace confidential catalog capabilities."""
    catalog_model = create_test_model("qwen3-tts")
    catalog_model.architecture = Architecture(
        modality="text->audio",
        input_modalities=["text"],
        output_modalities=["audio"],
        tokenizer="unknown",
        instruct_type=None,
    )
    catalog_model.supported_endpoints = ["/v1/audio/speech"]
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=7,
        models=[catalog_model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["qwen3-tts"],
        ),
    )

    override_model = create_test_model("qwen3-tts")
    override_model.supported_endpoints = ["/v1/chat/completions"]

    def fake_row_to_model(*args, **kwargs) -> Model:  # type: ignore[no-untyped-def]
        return override_model

    monkeypatch.setattr("routstr.payment.models._row_to_model", fake_row_to_model)

    override_row = SimpleNamespace(id="qwen3-tts", upstream_provider_id=7, enabled=True)

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={"qwen3-tts": (override_row, 1.01)},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert model_instances == {}
    assert provider_map == {}
    assert unique_models == {}


def test_required_confidential_routing_skips_db_override_without_provider_catalog_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static DB rows cannot invent a verified confidential provider model."""
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=7,
        models=[],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
        ),
    )
    override_model = create_test_model("gpt-secure")
    override_model.supported_endpoints = ["/v1/chat/completions"]

    def fake_row_to_model(*args, **kwargs) -> Model:  # type: ignore[no-untyped-def]
        return override_model

    monkeypatch.setattr("routstr.payment.models._row_to_model", fake_row_to_model)

    override_row = SimpleNamespace(
        id="gpt-secure", upstream_provider_id=7, enabled=True
    )

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={"gpt-secure": (override_row, 1.01)},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert model_instances == {}
    assert provider_map == {}
    assert unique_models == {}


def test_required_confidential_routing_skips_invalid_db_override_json() -> None:
    """Malformed DB override JSON should not crash required confidential map builds."""
    catalog_model = create_test_model("gpt-secure")
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=7,
        models=[catalog_model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
        ),
    )
    invalid_override_row = SimpleNamespace(
        id="gpt-secure",
        upstream_provider_id=7,
        enabled=True,
        name="GPT Secure",
        created=1,
        description="test",
        context_length=4096,
        architecture=(
            '{"modality":"text->text","input_modalities":["text"],'
            '"output_modalities":["text"],"tokenizer":"gpt",'
            '"instruct_type":null}'
        ),
        pricing=(
            '{"prompt":NaN,"completion":0.0,"request":0.0,'
            '"image":0.0,"web_search":0.0,"internal_reasoning":0.0}'
        ),
        per_request_limits=None,
        top_provider=None,
        canonical_slug=None,
        alias_ids=None,
        forwarded_model_id=None,
    )

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={"gpt-secure": (invalid_override_row, 1.01)},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert model_instances == {}
    assert provider_map == {}
    assert unique_models == {}


def test_required_confidential_routing_excludes_tool_catalog_entries() -> None:
    """Catalog tool services must not become prompt-bearing model routes."""
    model = create_test_model("websearch")
    model.architecture = Architecture(
        modality="tool",
        input_modalities=["tool"],
        output_modalities=["tool"],
        tokenizer="unknown",
        instruct_type=None,
    )
    model.supported_endpoints = ["/v1/chat/completions", "/v1/responses"]
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["websearch"],
        ),
    )

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert model_instances == {}
    assert provider_map == {}
    assert unique_models == {}


def test_known_confidential_provider_capabilities_do_not_inherit_broad_defaults() -> (
    None
):
    """Provider type contracts should cap unsafe endpoint flags for known modes."""
    private_model = create_test_model("private/llama-3.1")
    private_model.supported_endpoints = ["/v1/chat/completions"]
    ppq_provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
        db_id=1,
        models=[private_model],
        confidentiality_status=create_confidential_status(
            mode="ppq-private-tee",
            model_ids=["private/llama-3.1"],
        ),
    )
    configure_ppq_private_policy(ppq_provider)

    assert ppq_provider.supports_embeddings_api is True
    assert ppq_provider.supports_responses_api is True
    assert (
        provider_supports_required_endpoint(
            ppq_provider,
            "supports_embeddings_api",
        )
        is False
    )
    assert (
        provider_supports_required_endpoint(
            ppq_provider,
            "supports_responses_api",
        )
        is False
    )
    assert public_supported_endpoints_for_provider(ppq_provider) == [
        "/v1/chat/completions"
    ]

    _, _, unique_models = create_model_mappings(
        upstreams=[ppq_provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    confidentiality = unique_models["private/llama-3.1"].confidentiality
    assert confidentiality is not None
    assert confidentiality["supported_endpoints"] == ["/v1/chat/completions"]


def test_known_confidential_provider_requires_explicit_model_endpoint_metadata() -> (
    None
):
    """Known confidential routes must not infer endpoints from broad model shape."""
    model = create_test_model("tinfoil/gpt-secure")
    model.supported_endpoints = None
    provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        db_id=1,
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=tinfoil_claims_for_model_ids(["tinfoil/gpt-secure"]),
        ),
    )
    configure_tinfoil_model_attestation_policy(provider, ["tinfoil/gpt-secure"])

    assert public_supported_endpoints_for_provider(provider, model=model) == []


def test_required_confidential_routing_skips_malformed_provider_identity() -> None:
    """Bad provider identity data must fail closed instead of crashing map build."""
    model = create_test_model("gpt-secure")
    verified_provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            verified_claims=tinfoil_claims_for_model_ids(["gpt-secure"]),
        ),
    )
    configure_tinfoil_model_attestation_policy(verified_provider, ["gpt-secure"])
    malformed_provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
        models=[model],
        confidentiality_status=create_confidential_status(
            mode="tinfoil",
            model_ids=["gpt-secure"],
            verified_claims=tinfoil_claims_for_model_ids(["gpt-secure"]),
        ),
    )
    configure_tinfoil_model_attestation_policy(malformed_provider, ["gpt-secure"])
    malformed_provider.base_url = {"api_key": "SECRET_BASE_URL"}

    model_instances, provider_map, unique_models = create_model_mappings(
        upstreams=[malformed_provider, verified_provider],
        overrides_by_id={},
        disabled_model_ids=set(),
        require_confidential=True,
    )

    assert model_instances["gpt-secure"].id == "gpt-secure"
    assert provider_map["gpt-secure"] == [verified_provider]
    assert "gpt-secure" in unique_models


def test_messages_endpoint_requirement_preserves_plain_provider_translation() -> None:
    requirement = provider_endpoint_requirement_for_path("v1/messages")
    assert requirement == ("supports_anthropic_messages", "Messages API")
    count_tokens_requirement = provider_endpoint_requirement_for_path(
        "v1/messages/count_tokens"
    )
    assert count_tokens_requirement == (
        "supports_messages_count_tokens_api",
        "Messages Count Tokens API",
    )

    normal_provider = create_test_provider("openai", "https://api.openai.com/v1")
    ehbp_provider = create_test_provider("tinfoil", "https://inference.tinfoil.sh/v1")
    native_provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
    )
    native_provider.supports_anthropic_messages = True

    support_attr, _ = requirement
    assert provider_supports_required_endpoint(normal_provider, support_attr) is True
    assert provider_supports_required_endpoint(ehbp_provider, support_attr) is False
    assert provider_supports_required_endpoint(native_provider, support_attr) is True

    count_tokens_attr, _ = count_tokens_requirement
    assert (
        provider_supports_required_endpoint(normal_provider, count_tokens_attr) is True
    )
    assert (
        provider_supports_required_endpoint(ehbp_provider, count_tokens_attr) is False
    )
    assert (
        provider_supports_required_endpoint(native_provider, count_tokens_attr) is False
    )


def test_messages_count_tokens_requires_exact_model_endpoint_declaration() -> None:
    model = create_test_model("claude-secure")
    model.supported_endpoints = ["/v1/messages"]
    provider = create_test_provider("openai", "https://api.openai.com/v1")
    provider.supports_messages_count_tokens_api = True

    assert (
        provider_supports_required_endpoint(
            provider,
            "supports_messages_count_tokens_api",
            model=model,
        )
        is False
    )

    model.supported_endpoints = ["/v1/messages/count_tokens"]

    assert (
        provider_supports_required_endpoint(
            provider,
            "supports_messages_count_tokens_api",
            model=model,
        )
        is True
    )


def test_legacy_completions_endpoint_requirement_is_provider_capability() -> None:
    requirement = provider_endpoint_requirement_for_path("v1/completions")
    assert requirement == ("supports_completions_api", "Completions API")

    normal_provider = create_test_provider("openai", "https://api.openai.com/v1")
    ppq_private_provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
    )
    privatemode_provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
    )

    support_attr, _ = requirement
    assert provider_supports_required_endpoint(normal_provider, support_attr) is True
    assert (
        provider_supports_required_endpoint(ppq_private_provider, support_attr) is False
    )
    assert (
        provider_supports_required_endpoint(privatemode_provider, support_attr) is True
    )


def test_audio_endpoint_requirement_is_provider_capability() -> None:
    requirement = provider_endpoint_requirement_for_path("v1/audio/transcriptions")
    assert requirement == (
        "supports_audio_transcriptions_api",
        "Audio Transcriptions API",
    )
    speech_requirement = provider_endpoint_requirement_for_path("v1/audio/speech")
    assert speech_requirement == ("supports_audio_speech_api", "Audio Speech API")
    translations_requirement = provider_endpoint_requirement_for_path(
        "v1/audio/translations"
    )
    assert translations_requirement == (
        "supports_audio_translations_api",
        "Audio Translations API",
    )
    assert provider_endpoint_requirement_for_path("v1/audio") is None
    assert provider_endpoint_requirement_for_path("v1/audio/unknown") is None

    normal_provider = create_test_provider("openai", "https://api.openai.com/v1")
    ppq_private_provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
    )
    tinfoil_provider = create_test_provider(
        "tinfoil",
        "https://inference.tinfoil.sh/v1",
    )
    privatemode_provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
    )

    support_attr, _ = requirement
    assert provider_supports_required_endpoint(normal_provider, support_attr) is True
    assert (
        provider_supports_required_endpoint(ppq_private_provider, support_attr) is False
    )
    assert provider_supports_required_endpoint(tinfoil_provider, support_attr) is True
    assert (
        provider_supports_required_endpoint(privatemode_provider, support_attr) is True
    )

    translations_support_attr, _ = translations_requirement
    assert (
        provider_supports_required_endpoint(normal_provider, translations_support_attr)
        is True
    )
    assert (
        provider_supports_required_endpoint(
            ppq_private_provider,
            translations_support_attr,
        )
        is False
    )
    assert (
        provider_supports_required_endpoint(tinfoil_provider, translations_support_attr)
        is True
    )
    assert (
        provider_supports_required_endpoint(
            privatemode_provider,
            translations_support_attr,
        )
        is False
    )

    speech_support_attr, _ = speech_requirement
    assert (
        provider_supports_required_endpoint(normal_provider, speech_support_attr)
        is True
    )
    assert (
        provider_supports_required_endpoint(ppq_private_provider, speech_support_attr)
        is False
    )
    assert (
        provider_supports_required_endpoint(tinfoil_provider, speech_support_attr)
        is False
    )
    assert (
        provider_supports_required_endpoint(privatemode_provider, speech_support_attr)
        is False
    )


def test_public_supported_endpoints_use_exact_audio_paths_when_partial() -> None:
    normal_provider = create_test_provider("openai", "https://api.openai.com/v1")
    privatemode_provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
    )

    assert "/v1/audio" in public_supported_endpoints_for_provider(normal_provider)
    privatemode_endpoints = public_supported_endpoints_for_provider(
        privatemode_provider
    )
    assert "/v1/audio/transcriptions" in privatemode_endpoints
    assert "/v1/audio" not in privatemode_endpoints
    assert "/v1/audio/speech" not in privatemode_endpoints


def test_images_and_moderations_endpoint_requirements_are_provider_capabilities() -> (
    None
):
    normal_provider = create_test_provider("openai", "https://api.openai.com/v1")
    ppq_private_provider = create_test_provider(
        "ppq-private",
        "https://api.ppq.ai/private/v1",
    )
    privatemode_provider = create_test_provider(
        "privatemode",
        "http://127.0.0.1:8080/v1",
    )

    images_requirement = provider_endpoint_requirement_for_path("v1/images/generations")
    assert images_requirement == ("supports_images_api", "Images API")
    images_support_attr, _ = images_requirement
    assert (
        provider_supports_required_endpoint(normal_provider, images_support_attr)
        is True
    )
    assert (
        provider_supports_required_endpoint(ppq_private_provider, images_support_attr)
        is False
    )
    assert (
        provider_supports_required_endpoint(privatemode_provider, images_support_attr)
        is False
    )

    moderations_requirement = provider_endpoint_requirement_for_path("v1/moderations")
    assert moderations_requirement == ("supports_moderations_api", "Moderations API")
    moderations_support_attr, _ = moderations_requirement
    assert (
        provider_supports_required_endpoint(normal_provider, moderations_support_attr)
        is True
    )
    assert (
        provider_supports_required_endpoint(
            ppq_private_provider,
            moderations_support_attr,
        )
        is False
    )
    assert (
        provider_supports_required_endpoint(
            privatemode_provider,
            moderations_support_attr,
        )
        is False
    )
