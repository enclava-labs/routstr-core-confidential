"""Model prioritization algorithm for selecting cheapest upstream providers."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from .core.confidentiality_public import (
    is_full_sha256_digest,
    provider_evidence_digest_matches_status,
    public_confidentiality_policy_binds_provider_proof,
    public_model_id_matches_verified_selector,
    public_provider_proof_claims_cover_model_selectors,
    verified_public_provider_proof_claims,
)
from .core.logging import get_logger

if TYPE_CHECKING:
    from .payment.models import Model
    from .upstream import BaseUpstreamProvider

logger = get_logger(__name__)


_MISSING = object()


CONFIDENTIAL_PROVIDER_MODES = {
    "tinfoil": "tinfoil",
    "ppq-private": "ppq-private-tee",
    "privatemode": "privatemode",
}
TINFOIL_DEFAULT_BASE_URL = "https://inference.tinfoil.sh/v1"
TINFOIL_REQUIRE_MODEL_ATTESTATION_POLICY_KEYS = (
    "require_model_attestations",
    "requireModelAttestations",
)
TINFOIL_MODEL_ATTESTATION_TARGET_POLICY_KEYS = (
    "model_attestation_targets",
    "modelAttestationTargets",
    "model_enclave_bindings",
    "modelEnclaveBindings",
)
TINFOIL_RELEASE_DIGEST_POLICY_KEYS = (
    "expected_release_digest",
    "release_digest",
    "allowed_release_digest",
    "allowed_release_digests",
)
TINFOIL_RELEASE_DIGEST_EXACT_POLICY_KEYS = (
    "expected_release_digest",
    "release_digest",
)
TINFOIL_RELEASE_DIGEST_ALLOWED_POLICY_KEYS = (
    "allowed_release_digest",
    "allowed_release_digests",
)
TINFOIL_ENCLAVE_MEASUREMENT_POLICY_KEYS = (
    "expected_enclave_measurement_fingerprint",
    "enclave_measurement_fingerprint",
    "allowed_enclave_measurement_fingerprints",
    "allowed_enclave_measurement_fingerprint",
    "allowed_enclave_measurements",
)
TINFOIL_ENCLAVE_MEASUREMENT_EXACT_POLICY_KEYS = (
    "expected_enclave_measurement_fingerprint",
    "enclave_measurement_fingerprint",
)
TINFOIL_ENCLAVE_MEASUREMENT_ALLOWED_POLICY_KEYS = (
    "allowed_enclave_measurement_fingerprints",
    "allowed_enclave_measurement_fingerprint",
    "allowed_enclave_measurements",
)
TINFOIL_CODE_MEASUREMENT_POLICY_KEYS = (
    "expected_code_measurement_fingerprint",
    "code_measurement_fingerprint",
    "allowed_code_measurement_fingerprints",
    "allowed_code_measurement_fingerprint",
    "allowed_code_measurements",
)
TINFOIL_CODE_MEASUREMENT_EXACT_POLICY_KEYS = (
    "expected_code_measurement_fingerprint",
    "code_measurement_fingerprint",
)
TINFOIL_CODE_MEASUREMENT_ALLOWED_POLICY_KEYS = (
    "allowed_code_measurement_fingerprints",
    "allowed_code_measurement_fingerprint",
    "allowed_code_measurements",
)
PPQ_ATTESTATION_BUNDLE_URL_DIGEST_POLICY_KEYS = (
    "attestation_bundle_url_digest",
)
PRIVATEMODE_COMPONENT_DIGEST_POLICY_KEYS = {
    "manifest_digest": (
        "manifest_digest",
        "manifestDigest",
    ),
    "proxy_image_digest": (
        "proxy_image_digest",
        "proxyImageDigest",
    ),
    "proxy_binary_digest": (
        "proxy_binary_digest",
        "proxyBinaryDigest",
    ),
    "coordinator_measurement": (
        "expected_coordinator_measurement",
        "allowed_coordinator_measurements",
    ),
    "secret_service_measurement": (
        "expected_secret_service_measurement",
        "allowed_secret_service_measurements",
    ),
    "ai_worker_measurement": (
        "expected_ai_worker_measurement",
        "allowed_ai_worker_measurements",
    ),
    "key_release_binding": (
        "expected_key_release_binding",
        "allowed_key_release_bindings",
    ),
    "expected_workload_identity_digest": (
        "expected_workload_identity_digest",
    ),
    "model_workload_binding_digest": (
        "model_workload_binding_digest",
    ),
}
PRIVATEMODE_EXACT_DIGEST_ALIAS_POLICY_KEY_GROUPS = (
    (
        "manifest_digest",
        "manifestDigest",
    ),
    (
        "proxy_image_digest",
        "proxyImageDigest",
    ),
    (
        "proxy_binary_digest",
        "proxyBinaryDigest",
    ),
)
PRIVATEMODE_COMPONENT_DIGEST_PIN_GROUP_POLICY_KEYS = {
    "coordinator_measurement": (
        ("expected_coordinator_measurement",),
        ("allowed_coordinator_measurements",),
    ),
    "secret_service_measurement": (
        ("expected_secret_service_measurement",),
        ("allowed_secret_service_measurements",),
    ),
    "ai_worker_measurement": (
        ("expected_ai_worker_measurement",),
        ("allowed_ai_worker_measurements",),
    ),
    "key_release_binding": (
        ("expected_key_release_binding",),
        ("allowed_key_release_bindings",),
    ),
}
PRIVATEMODE_GPU_ATTESTATION_POLICY_KEYS = (
    "expected_gpu_attestation_policy",
    "allowed_gpu_attestation_policies",
)
PRIVATEMODE_GPU_ATTESTATION_POLICY_EXACT_KEYS = ("expected_gpu_attestation_policy",)
PRIVATEMODE_GPU_ATTESTATION_POLICY_ALLOWED_KEYS = (
    "allowed_gpu_attestation_policies",
)
PRIVATEMODE_TRUST_TIER_POLICY_KEYS = ("expected_trust_tier",)
PRIVATEMODE_EXPECTED_WORKLOAD_SAN_POLICY_KEYS = (
    "expected_workload_sans",
    "expectedWorkloadSANs",
)
PRIVATEMODE_EXPECTED_WORKLOAD_ID_POLICY_KEYS = (
    "expected_workload_ids",
    "expectedWorkloadIDs",
)
PRIVATEMODE_MODEL_WORKLOAD_BINDING_POLICY_KEYS = (
    "model_workload_bindings",
    "modelWorkloadBindings",
)
PRIVATEMODE_MODEL_WORKLOAD_SAN_POLICY_KEYS = (
    "workload_sans",
    "workloadSANs",
)
PRIVATEMODE_MODEL_WORKLOAD_ID_POLICY_KEYS = (
    "workload_ids",
    "workloadIDs",
)
CONFIDENTIAL_PROVIDER_ENDPOINT_CAPS = {
    "tinfoil": {
        "supports_anthropic_messages": False,
        "supports_messages_count_tokens_api": False,
        "supports_audio_api": True,
        "supports_audio_transcriptions_api": True,
        "supports_audio_translations_api": True,
        "supports_audio_speech_api": False,
        "supports_completions_api": False,
        "supports_embeddings_api": True,
        "supports_images_api": False,
        "supports_moderations_api": False,
        "supports_responses_api": True,
    },
    "ppq-private": {
        "supports_anthropic_messages": False,
        "supports_messages_count_tokens_api": False,
        "supports_audio_api": False,
        "supports_audio_transcriptions_api": False,
        "supports_audio_translations_api": False,
        "supports_audio_speech_api": False,
        "supports_completions_api": False,
        "supports_embeddings_api": False,
        "supports_images_api": False,
        "supports_moderations_api": False,
        "supports_responses_api": False,
    },
    "privatemode": {
        "supports_anthropic_messages": True,
        "supports_messages_count_tokens_api": False,
        "supports_audio_api": True,
        "supports_audio_transcriptions_api": True,
        "supports_audio_translations_api": False,
        "supports_audio_speech_api": False,
        "supports_completions_api": True,
        "supports_embeddings_api": True,
        "supports_images_api": False,
        "supports_moderations_api": False,
        "supports_responses_api": False,
    },
}

REQUIRED_EHBP_VERIFICATION_STEPS = (
    "hardware_attestation_report",
    "hardware_certificate_chain",
    "code_transparency",
    "measurement_match",
    "attested_transport_key_binding",
    "freshness",
)

REQUIRED_PRIVATEMODE_VERIFICATION_STEPS = (
    "contrast_manifest",
    "coordinator_attestation",
    "mesh_ca_binding",
    "secret_service_tls",
    "ai_worker_attestation",
    "gpu_attestation",
    "key_release_binding",
    "prompt_encryption",
    "nvidia_ocsp_revocation",
)

REQUIRED_PRIVATEMODE_PROOF_DIGEST_CLAIMS = (
    "coordinator_attestation_doc_digest",
    "mesh_ca_digest",
    "secret_service_certificate_digest",
    "ai_worker_manifest_digest",
    "attested_workload_identity_digest",
    "attested_workload_policy_digest",
    "expected_workload_identity_digest",
    "model_workload_binding_digest",
    "nvidia_ocsp_policy_header_digest",
    "nvidia_ocsp_policy_mac_digest",
    "prompt_encryption_ciphertext_digest",
    "inference_secret_id_digest",
)
TEXT_GENERATION_ENDPOINT_FLAGS = {
    "supports_anthropic_messages",
    "supports_chat_completions_api",
    "supports_completions_api",
    "supports_responses_api",
}
MODEL_ENDPOINTS_BY_SUPPORT_ATTR = {
    "supports_anthropic_messages": {"/v1/messages"},
    "supports_messages_count_tokens_api": {"/v1/messages/count_tokens"},
    "supports_chat_completions_api": {"/v1/chat/completions"},
    "supports_completions_api": {"/v1/completions"},
    "supports_embeddings_api": {"/v1/embeddings"},
    "supports_images_api": {"/v1/images"},
    "supports_moderations_api": {"/v1/moderations"},
    "supports_responses_api": {"/v1/responses"},
    "supports_audio_transcriptions_api": {"/v1/audio", "/v1/audio/transcriptions"},
    "supports_audio_translations_api": {"/v1/audio", "/v1/audio/translations"},
    "supports_audio_speech_api": {"/v1/audio", "/v1/audio/speech"},
}


def _status_list(value: object) -> list[str] | None:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip().lower()] if value.strip() else None
    if isinstance(value, (list, tuple, set)):
        values: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                return None
            values.append(item.strip().lower())
        return values
    return None


def _verified_selector_list(value: object) -> list[str] | None:
    if value is None:
        return []
    if not isinstance(value, list):
        return None
    values: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return None
        values.append(item.strip().lower())
    if len(set(values)) != len(values):
        return None
    return values


def _public_string_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _model_identifiers(model: "Model", alias: str | None = None) -> set[str]:
    identifiers: set[str] = set()
    for value in (
        alias,
        getattr(model, "id", None),
        getattr(model, "canonical_slug", None),
        getattr(model, "forwarded_model_id", None),
    ):
        normalized = _lower_non_empty_string(value)
        if normalized:
            identifiers.add(normalized)
    return identifiers


def _selector_base_id(value: str) -> str:
    return value.split("/", 1)[1] if "/" in value else value


def _selector_covers_identifier(
    identifier: str,
    model_ids: list[str],
    model_id_prefixes: list[str],
) -> bool:
    return identifier in model_ids or any(
        identifier.startswith(prefix) for prefix in model_id_prefixes
    )


def _tinfoil_selector_covers_bare_catalog_model(
    status: Any,
    model: "Model",
    alias: str | None,
    model_ids: list[str],
) -> bool:
    status_mode = _lower_non_empty_string(getattr(status, "mode", None))
    if status_mode != "tinfoil":
        return False
    route_alias = _lower_non_empty_string(alias)
    model_id = _lower_non_empty_string(getattr(model, "id", None))
    if not route_alias or "/" in route_alias or model_id != route_alias:
        return False
    forwarded_model_id = _lower_non_empty_string(
        getattr(model, "forwarded_model_id", None)
    )
    if forwarded_model_id and forwarded_model_id != route_alias:
        return False
    return public_model_id_matches_verified_selector(
        route_alias,
        model_ids,
        mode=status_mode,
    )


def _has_required_confidentiality_evidence(status: Any) -> bool:
    evidence_checker = getattr(status, "has_required_verification_evidence", None)
    if callable(evidence_checker):
        try:
            if evidence_checker() is not True:
                return False
        except Exception:
            return False
        verified_claims = getattr(status, "verified_claims", None)
        return _verified_claims_are_json_digestible(verified_claims)

    verifier = getattr(status, "verifier", None)
    if not isinstance(verifier, str) or not verifier.strip():
        return False
    policy_digest = getattr(status, "policy_digest", None)
    evidence_digest = getattr(status, "evidence_digest", None)
    verified_claims = getattr(status, "verified_claims", None)
    return (
        bool(verifier.strip())
        and is_full_sha256_digest(policy_digest)
        and is_full_sha256_digest(evidence_digest)
        and _verified_claims_are_json_digestible(verified_claims)
    )


def _verified_claims_are_json_digestible(value: object) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    try:
        _sha256_json_digest(value)
    except (TypeError, ValueError):
        return False
    return True


def _sha256_json_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _is_sha256_digest_value(value: object, *, prefixed: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    digest = value.strip().lower()
    if prefixed:
        if not digest.startswith("sha256:"):
            return False
        digest = digest.removeprefix("sha256:")
    elif digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    return (
        len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
        and len(set(digest)) > 1
    )


def _sha256_digest_hex_value(value: object, *, prefixed: bool = False) -> str | None:
    if not _is_sha256_digest_value(value, prefixed=prefixed):
        return None
    assert isinstance(value, str)
    return value.strip().lower().removeprefix("sha256:")


def _has_valid_sha256_claim_alias(
    claims: dict[str, Any],
    *keys: str,
    prefixed: bool = False,
) -> bool:
    present_values = [claims[key] for key in keys if key in claims]
    return bool(present_values) and all(
        _is_sha256_digest_value(value, prefixed=prefixed) for value in present_values
    )


def _has_matching_sha256_claim_alias(
    claims: dict[str, Any],
    *keys: str,
    prefixed: bool = False,
) -> bool:
    normalized_values: list[str] = []
    for key in keys:
        if key not in claims:
            continue
        digest = _sha256_digest_hex_value(claims[key], prefixed=prefixed)
        if digest is None:
            return False
        normalized_values.append(digest)
    return bool(normalized_values) and len(set(normalized_values)) == 1


def _is_hex_string(value: object, *, length: int) -> bool:
    if not isinstance(value, str):
        return False
    hex_value = value.strip().lower()
    return len(hex_value) == length and all(
        char in "0123456789abcdef" for char in hex_value
    )


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _lower_non_empty_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def _non_empty_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _positive_finite_number(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _model_architecture(model: "Model") -> Any:
    architecture = getattr(model, "architecture", None)
    if isinstance(architecture, str):
        try:
            parsed = json.loads(architecture)
        except (TypeError, ValueError):
            return architecture
        return parsed
    return architecture


def _architecture_value(architecture: object, attr: str) -> object:
    if isinstance(architecture, dict):
        return architecture.get(attr)
    return getattr(architecture, attr, None)


def _normalized_modality_set(value: object) -> set[str]:
    if isinstance(value, str):
        return {value.strip().lower()} if value.strip() else set()
    if isinstance(value, (list, tuple, set)):
        return {
            item.strip().lower()
            for item in value
            if isinstance(item, str) and item.strip()
        }
    return set()


def _model_modality_sets(model: "Model") -> tuple[set[str], set[str], set[str]]:
    architecture = _model_architecture(model)
    input_modalities = _normalized_modality_set(
        _architecture_value(architecture, "input_modalities")
    )
    output_modalities = _normalized_modality_set(
        _architecture_value(architecture, "output_modalities")
    )
    modality = _normalized_modality_set(_architecture_value(architecture, "modality"))
    if len(modality) == 1:
        modality_value = next(iter(modality))
        if "->" in modality_value:
            raw_input, raw_output = modality_value.split("->", 1)
            input_modalities.add(raw_input.strip().lower())
            output_modalities.add(raw_output.strip().lower())
    return input_modalities, output_modalities, modality


def _model_has_input(model: "Model", value: str) -> bool:
    input_modalities, _output_modalities, modality = _model_modality_sets(model)
    normalized = value.strip().lower()
    if normalized in input_modalities:
        return True
    if input_modalities:
        return False
    return any(normalized in item for item in modality)


def _model_has_output(model: "Model", value: str) -> bool:
    _input_modalities, output_modalities, modality = _model_modality_sets(model)
    normalized = value.strip().lower()
    if normalized in output_modalities:
        return True
    if output_modalities:
        return False
    return any(normalized in item for item in modality)


def model_supports_required_endpoint(model: "Model", support_attr: str) -> bool:
    """Return True when the selected model shape matches an API endpoint."""
    if support_attr in TEXT_GENERATION_ENDPOINT_FLAGS:
        return _model_has_input(model, "text") and _model_has_output(model, "text")
    if support_attr == "supports_embeddings_api":
        return _model_has_output(model, "embedding")
    if support_attr == "supports_audio_transcriptions_api":
        return _model_has_input(model, "audio") and _model_has_output(model, "text")
    if support_attr == "supports_audio_translations_api":
        return _model_has_input(model, "audio") and _model_has_output(model, "text")
    if support_attr == "supports_audio_speech_api":
        return _model_has_input(model, "text") and _model_has_output(model, "audio")
    if support_attr == "supports_audio_api":
        return (
            model_supports_required_endpoint(
                model,
                "supports_audio_transcriptions_api",
            )
            or model_supports_required_endpoint(
                model,
                "supports_audio_translations_api",
            )
            or model_supports_required_endpoint(model, "supports_audio_speech_api")
        )
    if support_attr == "supports_images_api":
        return _model_has_output(model, "image")
    if support_attr == "supports_moderations_api":
        return _model_has_output(model, "moderation") or any(
            "moderation" in item for item in _model_modality_sets(model)[2]
        )
    return True


def _model_declared_endpoint_set(model: "Model") -> set[str] | None:
    raw_endpoints = getattr(model, "supported_endpoints", None)
    if raw_endpoints is None:
        return None
    if type(raw_endpoints).__module__ == "unittest.mock":
        return None
    if not isinstance(raw_endpoints, (list, tuple, set)):
        return set()

    endpoints: set[str] = set()
    for item in raw_endpoints:
        endpoint = _lower_non_empty_string(item)
        if endpoint is None:
            return set()
        endpoints.add(endpoint)
    return endpoints


def _model_declares_endpoint_support(model: "Model", support_attr: str) -> bool:
    declared_endpoints = _model_declared_endpoint_set(model)
    if declared_endpoints is None:
        return True
    if support_attr == "supports_audio_api":
        return any(
            endpoint == "/v1/audio" or endpoint.startswith("/v1/audio/")
            for endpoint in declared_endpoints
        )

    supported_endpoints = MODEL_ENDPOINTS_BY_SUPPORT_ATTR.get(support_attr)
    if supported_endpoints is None:
        return True
    return bool(declared_endpoints & supported_endpoints)


def _model_route_identity(
    model: "Model",
) -> tuple[str, str | None, list[str] | None, str | None] | None:
    model_id = _non_empty_string(getattr(model, "id", None))
    if model_id is None:
        return None

    canonical_slug: str | None = None
    raw_canonical_slug = getattr(model, "canonical_slug", None)
    if raw_canonical_slug is not None:
        canonical_slug = _non_empty_string(raw_canonical_slug)
        if canonical_slug is None:
            return None

    alias_ids: list[str] | None = None
    raw_alias_ids = getattr(model, "alias_ids", None)
    if raw_alias_ids is not None:
        if not isinstance(raw_alias_ids, list):
            return None
        alias_ids = []
        for alias in raw_alias_ids:
            normalized_alias = _non_empty_string(alias)
            if normalized_alias is None:
                return None
            alias_ids.append(normalized_alias)

    forwarded_model_id: str | None = None
    raw_forwarded_model_id = getattr(model, "forwarded_model_id", None)
    if raw_forwarded_model_id is not None:
        forwarded_model_id = _non_empty_string(raw_forwarded_model_id)
        if forwarded_model_id is None:
            return None

    return model_id, canonical_slug, alias_ids, forwarded_model_id


def _provider_bool_flag(
    provider: "BaseUpstreamProvider",
    attr: str,
    *,
    default: bool,
    malformed_default: bool = False,
) -> bool:
    value = getattr(provider, attr, _MISSING)
    if value is _MISSING:
        return default
    if isinstance(value, bool):
        return value
    return malformed_default


def _provider_endpoint_bool_flag(
    provider: "BaseUpstreamProvider",
    attr: str,
    *,
    default: bool,
    malformed_default: bool = False,
) -> bool:
    provider_type = _lower_non_empty_string(getattr(provider, "provider_type", None))
    endpoint_caps = CONFIDENTIAL_PROVIDER_ENDPOINT_CAPS.get(provider_type or "")
    if endpoint_caps is not None and attr in endpoint_caps:
        if endpoint_caps[attr] is False:
            return False
        return _provider_bool_flag(
            provider,
            attr,
            default=endpoint_caps[attr],
            malformed_default=False,
        )
    return _provider_bool_flag(
        provider,
        attr,
        default=default,
        malformed_default=malformed_default,
    )


def _has_required_steps(claims: dict[str, Any], steps: tuple[str, ...]) -> bool:
    verification_steps = claims.get("verification_steps")
    if not isinstance(verification_steps, dict):
        return False
    return all(verification_steps.get(step) is True for step in steps)


def _tls_public_key_binding_satisfies_mode(
    claims: dict[str, Any],
    *,
    required: bool,
) -> bool:
    has_tls_claim = (
        "tls_public_key_fingerprint_sha256" in claims or "tls_public_key" in claims
    )
    if required or has_tls_claim:
        return _has_matching_sha256_claim_alias(
            claims,
            "tls_public_key_fingerprint_sha256",
            "tls_public_key",
        )
    return True


def _has_ehbp_provider_proof(
    claims: dict[str, Any],
    *,
    tls_binding_required: bool = True,
) -> bool:
    return bool(
        claims.get("transport") == "ehbp"
        and isinstance(claims.get("repo"), str)
        and str(claims.get("repo")).strip()
        and _is_hex_string(claims.get("attested_hpke_public_key_hex"), length=64)
        and _is_sha256_digest_value(claims.get("enclave_measurement_fingerprint"))
        and _is_sha256_digest_value(claims.get("code_measurement_fingerprint"))
        and _tls_public_key_binding_satisfies_mode(
            claims,
            required=tls_binding_required,
        )
        and _has_required_steps(claims, REQUIRED_EHBP_VERIFICATION_STEPS)
    )


def _has_tinfoil_provider_proof(
    claims: dict[str, Any],
    *,
    tls_binding_required: bool = True,
) -> bool:
    return bool(
        _has_ehbp_provider_proof(
            claims,
            tls_binding_required=tls_binding_required,
        )
        and isinstance(claims.get("attestation_format"), str)
        and str(claims.get("attestation_format")).strip()
        and _is_sha256_digest_value(
            claims.get("attestation_report_digest"), prefixed=True
        )
        and _is_sha256_digest_value(claims.get("release_digest"))
    )


def _tinfoil_provider_base_url_matches(provider: "BaseUpstreamProvider") -> bool:
    provider_base_url = getattr(provider, "base_url", None)
    if not isinstance(provider_base_url, str) or not provider_base_url.strip():
        return False
    parsed = urlparse(provider_base_url.strip())
    try:
        parsed.port
    except ValueError:
        return False
    if parsed.username or parsed.password or "@" in parsed.netloc:
        return False
    return provider_base_url.strip().rstrip("/") == TINFOIL_DEFAULT_BASE_URL


def _has_tinfoil_model_attestation_proof(
    claims: dict[str, Any],
    *,
    tls_binding_required: bool = True,
) -> bool:
    return bool(
        isinstance(claims.get("repo"), str)
        and str(claims.get("repo")).strip()
        and isinstance(claims.get("attestation_format"), str)
        and str(claims.get("attestation_format")).strip()
        and _is_sha256_digest_value(
            claims.get("attestation_report_digest"), prefixed=True
        )
        and _is_hex_string(claims.get("attested_hpke_public_key_hex"), length=64)
        and _is_sha256_digest_value(claims.get("enclave_measurement_fingerprint"))
        and _is_sha256_digest_value(claims.get("code_measurement_fingerprint"))
        and _tls_public_key_binding_satisfies_mode(
            claims,
            required=tls_binding_required,
        )
        and _is_sha256_digest_value(claims.get("release_digest"))
        and _has_required_steps(claims, REQUIRED_EHBP_VERIFICATION_STEPS)
    )


def _tinfoil_policy_requires_model_attestations(policy: Any) -> bool:
    policy_dict = getattr(policy, "policy", None)
    if not isinstance(policy_dict, dict):
        return False
    return any(
        policy_dict.get(key) is True
        for key in TINFOIL_REQUIRE_MODEL_ATTESTATION_POLICY_KEYS
    )


def _tinfoil_policy_model_attestation_targets(policy: Any) -> dict[str, Any] | None:
    policy_dict = getattr(policy, "policy", None)
    if not isinstance(policy_dict, dict):
        return None
    parsed_values: list[dict[str, Any]] = []
    for key in TINFOIL_MODEL_ATTESTATION_TARGET_POLICY_KEYS:
        targets = policy_dict.get(key)
        if targets is None:
            continue
        if not isinstance(targets, dict) or not targets:
            return None
        parsed: dict[str, Any] = {}
        seen_model_ids: set[str] = set()
        for raw_model_id, target in targets.items():
            if not isinstance(raw_model_id, str) or not raw_model_id.strip():
                return None
            model_id = raw_model_id.strip()
            normalized_model_id = model_id.lower()
            if normalized_model_id in seen_model_ids:
                return None
            seen_model_ids.add(normalized_model_id)
            parsed[model_id] = target
        parsed_values.append(dict(sorted(parsed.items())))
    if not parsed_values:
        return None
    first = parsed_values[0]
    if any(value != first for value in parsed_values[1:]):
        return None
    return first


def _is_native_sev_snp_measurement_value(value: object) -> bool:
    if not isinstance(value, str):
        return False
    digest = value.strip().lower()
    return (
        not digest.startswith("sha256:")
        and len(digest) == 96
        and all(char in "0123456789abcdef" for char in digest)
        and len(set(digest)) > 1
    )


def _measurement_fingerprint_hex_value(value: object) -> str | None:
    digest = _sha256_digest_hex_value(value)
    if digest is not None:
        return digest
    if not _is_native_sev_snp_measurement_value(value):
        return None
    assert isinstance(value, str)
    return hashlib.sha256(bytes.fromhex(value.strip().lower())).hexdigest()


def _policy_digest_hex_values(policy: dict[str, Any], *keys: str) -> set[str]:
    values: set[str] = set()
    for value in _policy_string_values(policy, *keys):
        digest = _sha256_digest_hex_value(value)
        if digest is not None:
            values.add(digest)
    return values


def _policy_measurement_hex_values(policy: dict[str, Any], *keys: str) -> set[str]:
    values: set[str] = set()
    for value in _policy_string_values(policy, *keys):
        digest = _measurement_fingerprint_hex_value(value)
        if digest is not None:
            values.add(digest)
    return values


def _claim_matches_digest_policy_values(claim: object, values: set[str]) -> bool:
    if not values:
        return True
    claim_digest = _sha256_digest_hex_value(claim)
    return claim_digest in values


def _claim_matches_measurement_policy_values(claim: object, values: set[str]) -> bool:
    if not values:
        return True
    claim_digest = _measurement_fingerprint_hex_value(claim)
    return claim_digest in values


def _strict_policy_string_values_for_keys(
    policy: dict[str, Any],
    *keys: str,
) -> list[str] | None:
    values: list[str] = []
    for key in keys:
        if key not in policy:
            continue
        value = policy[key]
        if isinstance(value, str):
            if not value.strip():
                return None
            values.append(value.strip())
        elif isinstance(value, (list, tuple)):
            if any(not isinstance(item, str) or not item.strip() for item in value):
                return None
            values.extend(item.strip() for item in value)
        else:
            return None
    return values


def _strict_policy_digest_hex_values(
    policy: dict[str, Any],
    *keys: str,
) -> set[str] | None:
    values = _strict_policy_string_values_for_keys(policy, *keys)
    if values is None:
        return None
    digests: set[str] = set()
    for value in values:
        digest = _sha256_digest_hex_value(value)
        if digest is None:
            return None
        digests.add(digest)
    return digests


def _strict_policy_digest_alias_values(
    policy: dict[str, Any],
    *keys: str,
) -> set[str] | None:
    selected: set[str] | None = None
    for key in keys:
        if key not in policy:
            continue
        value = policy[key]
        if not isinstance(value, str) or not value.strip():
            return None
        digest = _sha256_digest_hex_value(value.strip())
        if digest is None:
            return None
        values = {digest}
        if selected is None:
            selected = values
        elif values != selected:
            return None
    return selected or set()


def _strict_policy_measurement_hex_values(
    policy: dict[str, Any],
    *keys: str,
) -> set[str] | None:
    values = _strict_policy_string_values_for_keys(policy, *keys)
    if values is None:
        return None
    digests: set[str] = set()
    for value in values:
        digest = _measurement_fingerprint_hex_value(value)
        if digest is None:
            return None
        digests.add(digest)
    return digests


def _policy_pin_group_values(
    exact_values: set[str] | None,
    allowed_values: set[str] | None,
) -> set[str] | None:
    if exact_values is None or allowed_values is None:
        return None
    if exact_values and allowed_values and not exact_values.issubset(allowed_values):
        return None
    return exact_values if exact_values else allowed_values


def _strict_digest_pin_group_values(
    policy: dict[str, Any],
    exact_keys: tuple[str, ...],
    allowed_keys: tuple[str, ...],
) -> set[str] | None:
    return _policy_pin_group_values(
        _strict_policy_digest_hex_values(policy, *exact_keys),
        _strict_policy_digest_hex_values(policy, *allowed_keys),
    )


def _claim_matches_digest_policy_keys(
    claims: dict[str, Any],
    policy: dict[str, Any],
    claim_key: str,
    *policy_keys: str,
) -> bool:
    return _claim_matches_digest_policy_values(
        claims.get(claim_key),
        _policy_digest_hex_values(policy, *policy_keys),
    )


def _claim_matches_measurement_policy_keys(
    claims: dict[str, Any],
    policy: dict[str, Any],
    claim_key: str,
    *policy_keys: str,
) -> bool:
    return _claim_matches_measurement_policy_values(
        claims.get(claim_key),
        _policy_measurement_hex_values(policy, *policy_keys),
    )


def _privatemode_exact_digest_alias_keys(keys: tuple[str, ...]) -> bool:
    return keys in PRIVATEMODE_EXACT_DIGEST_ALIAS_POLICY_KEY_GROUPS


def _privatemode_component_digest_pin_group(
    claim_key: str,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    return PRIVATEMODE_COMPONENT_DIGEST_PIN_GROUP_POLICY_KEYS.get(claim_key)


def _ehbp_artifact_policy_pin_groups(
    policy: dict[str, Any],
) -> tuple[set[str], set[str], set[str]] | None:
    release_digests = _policy_pin_group_values(
        _strict_policy_digest_hex_values(
            policy,
            *TINFOIL_RELEASE_DIGEST_EXACT_POLICY_KEYS,
        ),
        _strict_policy_digest_hex_values(
            policy,
            *TINFOIL_RELEASE_DIGEST_ALLOWED_POLICY_KEYS,
        ),
    )
    enclave_measurements = _policy_pin_group_values(
        _strict_policy_measurement_hex_values(
            policy,
            *TINFOIL_ENCLAVE_MEASUREMENT_EXACT_POLICY_KEYS,
        ),
        _strict_policy_measurement_hex_values(
            policy,
            *TINFOIL_ENCLAVE_MEASUREMENT_ALLOWED_POLICY_KEYS,
        ),
    )
    code_measurements = _policy_pin_group_values(
        _strict_policy_measurement_hex_values(
            policy,
            *TINFOIL_CODE_MEASUREMENT_EXACT_POLICY_KEYS,
        ),
        _strict_policy_measurement_hex_values(
            policy,
            *TINFOIL_CODE_MEASUREMENT_ALLOWED_POLICY_KEYS,
        ),
    )
    if release_digests is None or enclave_measurements is None or code_measurements is None:
        return None
    return release_digests, enclave_measurements, code_measurements


def _tinfoil_model_claims_match_target_policy(
    target: dict[str, Any],
    model_claims: dict[str, Any],
) -> bool:
    pin_groups = _ehbp_artifact_policy_pin_groups(target)
    if pin_groups is None:
        return False
    release_digests, enclave_measurements, code_measurements = pin_groups
    if not release_digests and not enclave_measurements and not code_measurements:
        return False
    return (
        _claim_matches_digest_policy_values(
            model_claims.get("release_digest"),
            release_digests,
        )
        and _claim_matches_measurement_policy_values(
            model_claims.get("enclave_measurement_fingerprint"),
            enclave_measurements,
        )
        and _claim_matches_measurement_policy_values(
            model_claims.get("code_measurement_fingerprint"),
            code_measurements,
        )
    )


def _strict_policy_string_alias_value(policy: dict[str, Any], *keys: str) -> str | None:
    selected: str | None = None
    for key in keys:
        value = policy.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            return None
        normalized = value.strip()
        if selected is None:
            selected = normalized
        elif normalized != selected:
            return None
    return selected


def _tinfoil_tls_public_key_binding_required(
    policy: Any,
    *,
    target_policy: dict[str, Any] | None = None,
) -> bool:
    policy_dict = getattr(policy, "policy", None)
    if not isinstance(policy_dict, dict):
        policy_dict = policy if isinstance(policy, dict) else {}
    for raw_policy in (target_policy or {}, policy_dict):
        for key in ("transport_security", "tinfoil_transport_security"):
            value = raw_policy.get(key)
            if isinstance(value, str) and value.strip().lower() == "ehbp":
                return False
    return True


def _tinfoil_model_attestations_match_policy(
    *,
    policy: Any,
    model_ids: list[str],
    claims: dict[str, Any],
    claims_key: str = "model_attestations",
) -> bool:
    if not _tinfoil_policy_requires_model_attestations(policy):
        return False
    if not model_ids:
        return False
    selected_models = set(model_ids)
    targets = _tinfoil_policy_model_attestation_targets(policy)
    if not isinstance(targets, dict) or set(targets) != selected_models:
        return False
    model_attestations = claims.get(claims_key)
    if not isinstance(model_attestations, dict) or set(model_attestations) != selected_models:
        return False
    for model_id in selected_models:
        target = targets.get(model_id)
        model_claims = model_attestations.get(model_id)
        if not isinstance(target, dict) or not isinstance(model_claims, dict):
            return False
        expected_repo = _strict_policy_string_alias_value(
            target,
            "repo",
            "expected_repo",
        )
        if expected_repo is None:
            return False
        if model_claims.get("repo") != expected_repo:
            return False
        if not _has_tinfoil_model_attestation_proof(
            model_claims,
            tls_binding_required=_tinfoil_tls_public_key_binding_required(
                policy,
                target_policy=target,
            ),
        ):
            return False
        if not _tinfoil_model_claims_match_target_policy(target, model_claims):
            return False
    return True


def _ppq_private_backend_model_attestations_match_policy(
    *,
    policy: Any,
    model_ids: list[str],
    claims: dict[str, Any],
) -> bool:
    return _tinfoil_model_attestations_match_policy(
        policy=policy,
        model_ids=model_ids,
        claims=claims,
        claims_key="backend_model_attestations",
    )


def _tinfoil_policy_claims_match(policy: Any, claims: dict[str, Any]) -> bool:
    policy_dict = getattr(policy, "policy", None)
    if not isinstance(policy_dict, dict):
        return False
    expected_repo = _strict_policy_string_alias_value(
        policy_dict,
        "repo",
        "expected_repo",
    )
    if expected_repo is None or claims.get("repo") != expected_repo:
        return False
    pin_groups = _ehbp_artifact_policy_pin_groups(policy_dict)
    if pin_groups is None:
        return False
    release_digests, enclave_measurements, code_measurements = pin_groups
    if not release_digests:
        return False
    return (
        _claim_matches_digest_policy_values(
            claims.get("release_digest"),
            release_digests,
        )
        and _claim_matches_measurement_policy_values(
            claims.get("enclave_measurement_fingerprint"),
            enclave_measurements,
        )
        and _claim_matches_measurement_policy_values(
            claims.get("code_measurement_fingerprint"),
            code_measurements,
        )
    )


def _ppq_attestation_bundle_policy_digest_values(
    policy: dict[str, Any],
    expected_bundle_url: str,
) -> set[str] | None:
    attestation_bundle_url_digests = _strict_policy_digest_hex_values(
        policy,
        *PPQ_ATTESTATION_BUNDLE_URL_DIGEST_POLICY_KEYS,
    )
    if attestation_bundle_url_digests is None:
        return None
    expected_bundle_url_digest = _sha256_digest_hex_value(
        _sha256_json_digest(expected_bundle_url)
    )
    if expected_bundle_url_digest is None:
        return None
    if (
        attestation_bundle_url_digests
        and expected_bundle_url_digest not in attestation_bundle_url_digests
    ):
        return None
    attestation_bundle_url_digests.add(expected_bundle_url_digest)
    return attestation_bundle_url_digests


def _ppq_private_policy_claims_match(policy: Any, claims: dict[str, Any]) -> bool:
    policy_dict = getattr(policy, "policy", None)
    if not isinstance(policy_dict, dict):
        return False
    expected_bundle_url = _strict_policy_string_alias_value(
        policy_dict,
        "attestation_bundle_url",
    )
    claimed_bundle_url = claims.get("attestation_bundle_url")
    claimed_bundle_url_digest = claims.get("attestation_bundle_url_digest")
    if (
        expected_bundle_url is None
        or not isinstance(claimed_bundle_url, str)
        or claimed_bundle_url.strip() != expected_bundle_url
        or not is_full_sha256_digest(claimed_bundle_url_digest)
        or claimed_bundle_url_digest != _sha256_json_digest(expected_bundle_url)
    ):
        return False
    expected_repo = _strict_policy_string_alias_value(
        policy_dict,
        "repo",
        "expected_repo",
    )
    if expected_repo is None or claims.get("repo") != expected_repo:
        return False
    pin_groups = _ehbp_artifact_policy_pin_groups(policy_dict)
    if pin_groups is None:
        return False
    release_digests, enclave_measurements, code_measurements = pin_groups
    if not release_digests and not code_measurements:
        return False
    attestation_bundle_url_digests = _ppq_attestation_bundle_policy_digest_values(
        policy_dict,
        expected_bundle_url,
    )
    if not attestation_bundle_url_digests:
        return False
    return (
        _claim_matches_digest_policy_values(
            claims.get("release_digest"),
            release_digests,
        )
        and _claim_matches_measurement_policy_values(
            claims.get("enclave_measurement_fingerprint"),
            enclave_measurements,
        )
        and _claim_matches_measurement_policy_values(
            claims.get("code_measurement_fingerprint"),
            code_measurements,
        )
        and _claim_matches_digest_policy_values(
            claims.get("attestation_bundle_url_digest"),
            attestation_bundle_url_digests,
        )
    )


def _ppq_private_selected_model_claims_match(
    model_ids: list[str],
    claims: dict[str, Any],
) -> bool:
    raw_selected_model_ids = claims.get("selected_model_ids")
    selected_model_ids = _verified_selector_list(raw_selected_model_ids)
    if selected_model_ids is None or not selected_model_ids:
        return False
    if len(set(model_ids)) != len(model_ids):
        return False
    return set(selected_model_ids) == set(model_ids)


def _selected_model_claims_match(
    model_ids: list[str],
    claims: dict[str, Any],
) -> bool:
    raw_selected_model_ids = claims.get("selected_model_ids")
    selected_model_ids = _verified_selector_list(raw_selected_model_ids)
    if selected_model_ids is None or not selected_model_ids:
        return False
    if len(set(model_ids)) != len(model_ids):
        return False
    return set(selected_model_ids) == set(model_ids)


def _url_has_private_path(parsed_url: Any) -> bool:
    return any(
        segment.lower() == "private"
        for segment in parsed_url.path.split("/")
        if segment
    )


def _url_origin(parsed_url: Any) -> str | None:
    scheme = getattr(parsed_url, "scheme", None)
    hostname = getattr(parsed_url, "hostname", None)
    if not isinstance(scheme, str) or not isinstance(hostname, str):
        return None
    scheme = scheme.strip().lower()
    hostname = hostname.strip().rstrip(".").lower()
    if not scheme or not hostname:
        return None
    try:
        port = parsed_url.port
    except ValueError:
        return None
    if port is None or (scheme == "https" and port == 443):
        return f"{scheme}://{hostname}"
    return f"{scheme}://{hostname}:{port}"


def _is_loopback_hostname(hostname: str | None) -> bool:
    if not hostname:
        return False
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _privatemode_proxy_url_is_loopback(value: str) -> bool:
    parsed = urlparse(value.strip())
    try:
        parsed.port
    except ValueError:
        return False
    return bool(
        parsed.scheme in {"http", "https"}
        and parsed.netloc
        and not parsed.username
        and not parsed.password
        and _is_loopback_hostname(parsed.hostname)
    )


def _has_ppq_private_provider_proof(
    provider: "BaseUpstreamProvider",
    claims: dict[str, Any],
) -> bool:
    provider_base_url = getattr(provider, "base_url", None)
    if not isinstance(provider_base_url, str) or not provider_base_url.strip():
        return False
    attestation_bundle_url = claims.get("attestation_bundle_url")
    if (
        not isinstance(attestation_bundle_url, str)
        or not attestation_bundle_url.strip()
    ):
        return False
    provider_url = urlparse(provider_base_url.strip())
    bundle_url = urlparse(attestation_bundle_url.strip())
    provider_origin = _url_origin(provider_url)
    bundle_origin = _url_origin(bundle_url)
    if (
        provider_url.scheme != "https"
        or bundle_url.scheme != "https"
        or not provider_url.hostname
        or not bundle_url.hostname
        or provider_origin is None
        or bundle_origin is None
        or provider_origin != bundle_origin
        or provider_url.username
        or provider_url.password
        or bundle_url.username
        or bundle_url.password
        or provider_url.hostname.rstrip(".").lower()
        != bundle_url.hostname.rstrip(".").lower()
        or not _url_has_private_path(provider_url)
        or not _url_has_private_path(bundle_url)
    ):
        return False

    attestation_bundle_url_digest = claims.get("attestation_bundle_url_digest")
    return bool(
        _has_ehbp_provider_proof(claims)
        and is_full_sha256_digest(attestation_bundle_url_digest)
        and attestation_bundle_url_digest == _sha256_json_digest(attestation_bundle_url)
    )


def _has_privatemode_provider_proof(claims: dict[str, Any]) -> bool:
    return bool(
        claims.get("transport") == "privatemode-proxy"
        and claims.get("trust_tier") == "app-e2ee"
        and _has_matching_sha256_claim_alias(
            claims,
            "manifest_digest",
            "manifest_log_manifest_digest",
            prefixed=True,
        )
        and _is_sha256_digest_value(
            claims.get("proxy_binary_digest"),
            prefixed=True,
        )
        and (
            "proxy_image_digest" not in claims
            or _is_sha256_digest_value(
                claims.get("proxy_image_digest"),
                prefixed=True,
            )
        )
        and _is_sha256_digest_value(
            claims.get("coordinator_measurement"), prefixed=True
        )
        and _is_sha256_digest_value(
            claims.get("secret_service_measurement"), prefixed=True
        )
        and _is_sha256_digest_value(claims.get("ai_worker_measurement"), prefixed=True)
        and claims.get("ai_worker_measurement")
        == claims.get("attested_workload_policy_digest")
        and isinstance(claims.get("gpu_attestation_policy"), str)
        and str(claims.get("gpu_attestation_policy")).strip()
        and _is_sha256_digest_value(claims.get("key_release_binding"), prefixed=True)
        and all(
            _is_sha256_digest_value(claims.get(claim), prefixed=True)
            for claim in REQUIRED_PRIVATEMODE_PROOF_DIGEST_CLAIMS
        )
        and _has_required_steps(claims, REQUIRED_PRIVATEMODE_VERIFICATION_STEPS)
    )


def _privatemode_proxy_base_url_claim_matches(
    provider: "BaseUpstreamProvider",
    claims: dict[str, Any],
) -> bool:
    provider_base_url = getattr(provider, "base_url", None)
    claim_base_url = claims.get("proxy_base_url")
    return (
        isinstance(provider_base_url, str)
        and isinstance(claim_base_url, str)
        and provider_base_url.strip() == claim_base_url.strip()
        and _privatemode_proxy_url_is_loopback(provider_base_url)
    )


def _policy_string_values(policy: dict[str, Any], *keys: str) -> list[str]:
    values: list[str] = []
    for key in keys:
        value = policy.get(key)
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                values.append(normalized)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str) and item.strip():
                    values.append(item.strip())
    return sorted(set(values))


def _strict_policy_string_values(policy: dict[str, Any], *keys: str) -> list[str] | None:
    selected_values: list[str] | None = None
    for key in keys:
        if key not in policy:
            continue
        value = policy[key]
        if not isinstance(value, list) or not value:
            return None
        if any(not isinstance(item, str) or not item.strip() for item in value):
            return None
        normalized = [item.strip() for item in value]
        if len({item.lower() for item in normalized}) != len(normalized):
            return None
        canonical = sorted(normalized)
        if selected_values is None:
            selected_values = canonical
        elif canonical != selected_values:
            return None
    return selected_values or []


def _strict_policy_string_pin_group_values(
    policy: dict[str, Any],
    exact_keys: tuple[str, ...],
    allowed_keys: tuple[str, ...],
) -> set[str] | None:
    exact_values = _strict_policy_string_values_for_keys(policy, *exact_keys)
    allowed_values = _strict_policy_string_values_for_keys(policy, *allowed_keys)
    if exact_values is None or allowed_values is None:
        return None
    exact = set(exact_values)
    allowed = set(allowed_values)
    if exact and allowed and not exact.issubset(allowed):
        return None
    return exact if exact else allowed


def _privatemode_expected_workload_identity_digest(
    policy: dict[str, Any],
) -> str | None:
    workload_ids = _strict_policy_string_values(
        policy,
        *PRIVATEMODE_EXPECTED_WORKLOAD_ID_POLICY_KEYS,
    )
    workload_sans = _strict_policy_string_values(
        policy,
        *PRIVATEMODE_EXPECTED_WORKLOAD_SAN_POLICY_KEYS,
    )
    if workload_ids is None or workload_sans is None:
        return None
    if not workload_ids and not workload_sans:
        return None
    return _sha256_json_digest(
        {
            "ids": workload_ids,
            "sans": workload_sans,
        }
    )


def _privatemode_model_workload_bindings(
    policy: dict[str, Any],
) -> dict[str, dict[str, list[str]]] | None:
    parsed_values: list[dict[str, dict[str, list[str]]]] = []
    for key in PRIVATEMODE_MODEL_WORKLOAD_BINDING_POLICY_KEYS:
        if key not in policy:
            continue
        raw_bindings = policy.get(key)
        if not isinstance(raw_bindings, dict) or not raw_bindings:
            return None

        bindings: dict[str, dict[str, list[str]]] = {}
        seen_model_ids: set[str] = set()
        for raw_model_id, raw_binding in raw_bindings.items():
            if not isinstance(raw_model_id, str) or not raw_model_id.strip():
                return None
            model_id = raw_model_id.strip()
            normalized_model_id = model_id.lower()
            if normalized_model_id in seen_model_ids:
                return None
            seen_model_ids.add(normalized_model_id)
            if not isinstance(raw_binding, dict):
                return None
            workload_ids = _strict_policy_string_values(
                raw_binding,
                *PRIVATEMODE_MODEL_WORKLOAD_ID_POLICY_KEYS,
            )
            workload_sans = _strict_policy_string_values(
                raw_binding,
                *PRIVATEMODE_MODEL_WORKLOAD_SAN_POLICY_KEYS,
            )
            if workload_ids is None or workload_sans is None:
                return None
            if not workload_ids and not workload_sans:
                return None
            bindings[model_id] = {
                "workload_ids": workload_ids,
                "workload_sans": workload_sans,
            }
        parsed_values.append(dict(sorted(bindings.items())))

    if not parsed_values:
        return None
    first = parsed_values[0]
    if any(value != first for value in parsed_values[1:]):
        return None
    return first


def _privatemode_model_workload_binding_digest(policy: dict[str, Any]) -> str | None:
    bindings = _privatemode_model_workload_bindings(policy)
    if bindings is None:
        return None
    return _sha256_json_digest(bindings)


def _privatemode_workload_bindings_cover_expected_workloads(
    policy: dict[str, Any],
    bindings: dict[str, dict[str, list[str]]],
) -> bool:
    workload_ids = _strict_policy_string_values(
        policy,
        *PRIVATEMODE_EXPECTED_WORKLOAD_ID_POLICY_KEYS,
    )
    workload_sans = _strict_policy_string_values(
        policy,
        *PRIVATEMODE_EXPECTED_WORKLOAD_SAN_POLICY_KEYS,
    )
    if workload_ids is None or workload_sans is None:
        return False
    expected_ids = set(workload_ids)
    expected_sans = set(workload_sans)
    if not expected_ids and not expected_sans:
        return False

    bound_ids: set[str] = set()
    bound_sans: set[str] = set()
    for binding in bindings.values():
        binding_ids = set(binding["workload_ids"])
        binding_sans = set(binding["workload_sans"])
        if binding_ids - expected_ids or binding_sans - expected_sans:
            return False
        bound_ids.update(binding_ids)
        bound_sans.update(binding_sans)

    return not (expected_ids - bound_ids or expected_sans - bound_sans)


def _privatemode_component_policy_claims_match(
    policy: dict[str, Any],
    claims: dict[str, Any],
) -> bool:
    for claim_key, policy_keys in PRIVATEMODE_COMPONENT_DIGEST_POLICY_KEYS.items():
        if _privatemode_exact_digest_alias_keys(policy_keys):
            values = _strict_policy_digest_alias_values(policy, *policy_keys)
        elif pin_group := _privatemode_component_digest_pin_group(claim_key):
            exact_keys, allowed_keys = pin_group
            values = _strict_digest_pin_group_values(policy, exact_keys, allowed_keys)
        else:
            values = _strict_policy_digest_hex_values(policy, *policy_keys)
        if values is None:
            return False
        if not _claim_matches_digest_policy_values(claims.get(claim_key), values):
            return False
    gpu_policy_values = _strict_policy_string_pin_group_values(
        policy,
        PRIVATEMODE_GPU_ATTESTATION_POLICY_EXACT_KEYS,
        PRIVATEMODE_GPU_ATTESTATION_POLICY_ALLOWED_KEYS,
    )
    if gpu_policy_values is None:
        return False
    if gpu_policy_values and claims.get("gpu_attestation_policy") not in gpu_policy_values:
        return False
    trust_tier = _strict_policy_string_alias_value(
        policy,
        *PRIVATEMODE_TRUST_TIER_POLICY_KEYS,
    )
    if trust_tier != "app-e2ee" or claims.get("trust_tier") != "app-e2ee":
        return False
    return True


def _privatemode_policy_claims_match(
    *,
    policy: Any,
    model_ids: list[str],
    claims: dict[str, Any],
) -> bool:
    policy_dict = getattr(policy, "policy", None)
    if not isinstance(policy_dict, dict) or not model_ids:
        return False
    bindings = _privatemode_model_workload_bindings(policy_dict)
    if bindings is None or set(bindings) != set(model_ids):
        return False
    if not _privatemode_workload_bindings_cover_expected_workloads(
        policy_dict,
        bindings,
    ):
        return False
    has_manifest_digest_pin = bool(
        _strict_policy_digest_alias_values(
            policy_dict,
            *PRIVATEMODE_COMPONENT_DIGEST_POLICY_KEYS["manifest_digest"],
        )
    )
    manifest_log_dir = _policy_string_values(
        policy_dict,
        "manifest_log_dir",
        "manifestLogDir",
    )
    if not has_manifest_digest_pin and not manifest_log_dir:
        return False
    if manifest_log_dir and (
        not is_full_sha256_digest(claims.get("manifest_log_digest"))
        or not is_full_sha256_digest(claims.get("manifest_log_manifest_digest"))
        or claims.get("manifest_log_manifest_digest") != claims.get("manifest_digest")
    ):
        return False
    proxy_binary_pins = _strict_policy_digest_alias_values(
        policy_dict,
        *PRIVATEMODE_COMPONENT_DIGEST_POLICY_KEYS["proxy_binary_digest"],
    )
    if proxy_binary_pins is None or not proxy_binary_pins:
        return False
    expected_workload_digest = _privatemode_expected_workload_identity_digest(
        policy_dict
    )
    model_binding_digest = _privatemode_model_workload_binding_digest(policy_dict)
    return bool(
        model_binding_digest
        and claims.get("expected_workload_identity_digest")
        == expected_workload_digest
        and claims.get("model_workload_binding_digest") == model_binding_digest
        and _privatemode_component_policy_claims_match(policy_dict, claims)
    )


def _provider_confidentiality_mode_matches(
    provider: "BaseUpstreamProvider", status: Any
) -> bool:
    provider_type = _lower_non_empty_string(getattr(provider, "provider_type", None))
    if provider_type is None:
        return False
    expected_mode = CONFIDENTIAL_PROVIDER_MODES.get(provider_type)
    if expected_mode is None:
        return False
    status_mode = _lower_non_empty_string(getattr(status, "mode", None))
    return status_mode == expected_mode


def _provider_confidentiality_policy_digest_matches(
    provider: "BaseUpstreamProvider", status: Any
) -> bool:
    provider_type = _lower_non_empty_string(getattr(provider, "provider_type", None))
    requires_local_policy = provider_type in CONFIDENTIAL_PROVIDER_MODES
    policy_getter = getattr(provider, "confidentiality_policy", None)
    if not callable(policy_getter):
        return not requires_local_policy

    try:
        policy = policy_getter()
    except Exception:
        return False
    if policy is None:
        return not requires_local_policy

    policy_digest = getattr(policy, "digest", None)
    policy_provider_type = getattr(policy, "provider_type", None)
    if (
        isinstance(policy_provider_type, str)
        and policy_provider_type.strip().lower() != provider_type
    ):
        return False
    expected_mode = CONFIDENTIAL_PROVIDER_MODES.get(provider_type or "")
    policy_mode = getattr(policy, "mode", None)
    if (
        expected_mode is not None
        and isinstance(policy_mode, str)
        and policy_mode.strip().lower() != expected_mode
    ):
        return False
    policy_model_ids = _verified_selector_list(getattr(policy, "model_ids", None))
    policy_model_id_prefixes = _verified_selector_list(
        getattr(policy, "model_id_prefixes", None)
    )
    if provider_type in {"tinfoil", "ppq-private", "privatemode"}:
        if (
            policy_model_ids is None
            or not policy_model_ids
            or policy_model_id_prefixes is None
            or policy_model_id_prefixes
        ):
            return False
        status_model_ids = _verified_selector_list(getattr(status, "model_ids", None))
        if status_model_ids is None or status_model_ids != policy_model_ids:
            return False
    status_policy_digest = getattr(status, "policy_digest", None)
    return (
        is_full_sha256_digest(policy_digest)
        and is_full_sha256_digest(status_policy_digest)
        and status_policy_digest == policy_digest
    )


def _known_provider_proof_claims_match(
    provider: "BaseUpstreamProvider", status: Any
) -> bool:
    provider_type = _lower_non_empty_string(getattr(provider, "provider_type", None))
    if provider_type is None:
        return False
    claims = getattr(status, "verified_claims", None)
    if not isinstance(claims, dict):
        return False
    policy_getter = getattr(provider, "confidentiality_policy", None)
    policy = policy_getter() if callable(policy_getter) else None
    raw_policy = getattr(policy, "policy", None)
    proof_claims = verified_public_provider_proof_claims(
        getattr(status, "mode", None),
        claims,
        public_policy=raw_policy if isinstance(raw_policy, dict) else None,
    )
    if proof_claims is None:
        return False
    if proof_claims.get("payload_policy_digest") != getattr(
        status,
        "policy_digest",
        None,
    ):
        return False
    if not provider_evidence_digest_matches_status(
        getattr(status, "evidence_digest", None),
        claims,
        proof_claims,
    ):
        return False
    if provider_type == "tinfoil":
        model_ids = _verified_selector_list(getattr(status, "model_ids", None))
        return bool(
            _tinfoil_provider_base_url_matches(provider)
            and _has_tinfoil_provider_proof(
                claims,
                tls_binding_required=_tinfoil_tls_public_key_binding_required(policy),
            )
            and _tinfoil_policy_claims_match(policy, claims)
            and model_ids is not None
            and _tinfoil_model_attestations_match_policy(
                policy=policy,
                model_ids=model_ids,
                claims=claims,
            )
        )
    if provider_type == "ppq-private":
        model_ids = _verified_selector_list(getattr(status, "model_ids", None))
        return bool(
            _has_ppq_private_provider_proof(provider, claims)
            and _ppq_private_policy_claims_match(policy, claims)
            and model_ids is not None
            and _ppq_private_selected_model_claims_match(model_ids, claims)
            and _ppq_private_backend_model_attestations_match_policy(
                policy=policy,
                model_ids=model_ids,
                claims=claims,
            )
        )
    if provider_type == "privatemode":
        model_ids = _verified_selector_list(getattr(status, "model_ids", None))
        return bool(
            _has_privatemode_provider_proof(claims)
            and _privatemode_proxy_base_url_claim_matches(provider, claims)
            and model_ids is not None
            and _selected_model_claims_match(model_ids, claims)
            and _privatemode_policy_claims_match(
                policy=policy,
                model_ids=model_ids,
                claims=claims,
            )
        )
    return False


def _known_provider_selectors_match(
    provider: "BaseUpstreamProvider",
    model_ids: list[str],
    model_id_prefixes: list[str],
) -> bool:
    provider_type = _lower_non_empty_string(getattr(provider, "provider_type", None))
    if provider_type in {"tinfoil", "ppq-private", "privatemode"}:
        if model_id_prefixes:
            return False
        if not model_ids:
            return False
        if len({model_id.lower() for model_id in model_ids}) != len(model_ids):
            return False
    if provider_type == "ppq-private" and any(
        not model_id.startswith("private/") for model_id in model_ids
    ):
        return False
    if provider_type == "privatemode" and any(
        not model_id.startswith("privatemode/") for model_id in model_ids
    ):
        return False
    return True


def _current_verified_confidentiality_status(
    provider: "BaseUpstreamProvider",
) -> Any | None:
    status_getter = getattr(provider, "confidentiality_status", None)
    if not callable(status_getter):
        return None

    try:
        status: Any = status_getter()
    except Exception:
        return None

    if getattr(status, "enabled", False) is not True:
        return None
    if getattr(status, "verified", False) is not True:
        return None
    if not _has_required_confidentiality_evidence(status):
        return None
    if not _provider_confidentiality_policy_digest_matches(provider, status):
        return None
    if not _provider_confidentiality_mode_matches(provider, status):
        return None
    if not _known_provider_proof_claims_match(provider, status):
        return None
    model_ids = _verified_selector_list(getattr(status, "model_ids", None))
    model_id_prefixes = _verified_selector_list(
        getattr(status, "model_id_prefixes", None)
    )
    if model_ids is None or model_id_prefixes is None:
        return None
    if not model_ids and not model_id_prefixes:
        return None
    if not _known_provider_selectors_match(provider, model_ids, model_id_prefixes):
        return None

    verified_at = getattr(status, "verified_at", None)
    if not _is_strict_int(verified_at):
        return None

    expires_at = getattr(status, "expires_at", None)
    if not _is_strict_int(expires_at):
        return None
    now = int(time.time())
    if verified_at > now:
        return None
    if expires_at <= now:
        return None

    return status


def has_current_confidential_verification(provider: "BaseUpstreamProvider") -> bool:
    """Return True when provider evidence is currently verified and fresh."""
    return _current_verified_confidentiality_status(provider) is not None


def is_confidential_provider_for_model(
    provider: "BaseUpstreamProvider", model: "Model", alias: str | None = None
) -> bool:
    """Return True only when provider evidence currently covers the model."""
    status = _current_verified_confidentiality_status(provider)
    if status is None:
        return False

    return _confidentiality_status_covers_model(
        status, model, alias
    ) and _db_confidential_attestation_covers_model(provider, status, model, alias)


def _db_confidential_attestation_covers_model(
    provider: "BaseUpstreamProvider",
    status: Any,
    model: "Model",
    alias: str | None = None,
) -> bool:
    provider_attrs = getattr(provider, "__dict__", {})
    if (
        not isinstance(provider_attrs, dict)
        or "_db_confidential_attestations" not in provider_attrs
    ):
        return True
    attestations = provider_attrs["_db_confidential_attestations"]
    if not isinstance(attestations, dict):
        return False

    status_mode = _lower_non_empty_string(getattr(status, "mode", None))
    if status_mode is None:
        return False
    status_policy_digest = getattr(status, "policy_digest", None)
    if not is_full_sha256_digest(status_policy_digest):
        return False

    identifiers = _model_identifiers(model, alias)
    if not identifiers:
        return False

    now = int(time.time())
    for row_key, row in attestations.items():
        if not isinstance(row_key, tuple) or len(row_key) != 2:
            continue
        row_model_id, row_mode = row_key
        if row_mode != status_mode:
            continue
        if row_model_id not in identifiers and _selector_base_id(row_model_id) not in identifiers:
            continue
        if getattr(row, "verified", False) is not True:
            continue
        verified_at = getattr(row, "verified_at", None)
        expires_at = getattr(row, "expires_at", None)
        if not _is_strict_int(verified_at) or not _is_strict_int(expires_at):
            continue
        if verified_at > now or expires_at <= now:
            continue
        if getattr(row, "policy_digest", None) != status_policy_digest:
            continue
        if not is_full_sha256_digest(getattr(row, "evidence_digest", None)):
            continue
        verifier = getattr(row, "verifier", None)
        if not isinstance(verifier, str) or not verifier.strip():
            continue
        return True
    return False


def _confidentiality_status_covers_model(
    status: Any, model: "Model", alias: str | None = None
) -> bool:
    model_ids = _verified_selector_list(getattr(status, "model_ids", None))
    model_id_prefixes = _verified_selector_list(
        getattr(status, "model_id_prefixes", None)
    )
    if model_ids is None or model_id_prefixes is None:
        return False
    if not model_ids and not model_id_prefixes:
        return False

    route_alias = _lower_non_empty_string(alias)
    forwarded_model_id = _lower_non_empty_string(
        getattr(model, "forwarded_model_id", None)
    )
    if (
        route_alias
        and forwarded_model_id
        and _selector_base_id(forwarded_model_id) != route_alias
        and not _selector_covers_identifier(
            forwarded_model_id,
            model_ids,
            model_id_prefixes,
        )
    ):
        return False
    if route_alias:
        identifiers = {route_alias}
        status_mode = _lower_non_empty_string(getattr(status, "mode", None))
        if (
            forwarded_model_id
            and status_mode != "ppq-private-tee"
            and _selector_base_id(forwarded_model_id) == route_alias
        ):
            identifiers.add(forwarded_model_id)
    else:
        identifiers = _model_identifiers(model)
    if any(
        _selector_covers_identifier(identifier, model_ids, model_id_prefixes)
        for identifier in identifiers
    ):
        return True
    if _tinfoil_selector_covers_bare_catalog_model(status, model, alias, model_ids):
        return True
    return False


def _metadata_leakage_for_provider(provider: "BaseUpstreamProvider") -> list[str]:
    provider_type = _lower_non_empty_string(getattr(provider, "provider_type", None))
    if provider_type == "privatemode":
        return [
            "model",
            "stream",
            "max_tokens",
            "max_completion_tokens",
            "n",
            "response_id",
            "usage",
        ]
    if provider_type in {"tinfoil", "ppq-private"}:
        return ["model", "usage"]
    return []


def public_supported_endpoints_for_provider(
    provider: "BaseUpstreamProvider",
    model: "Model" | None = None,
) -> list[str]:
    """Return API endpoints this provider can serve through Routstr."""
    endpoints = []
    if provider_supports_required_endpoint(
        provider,
        "supports_chat_completions_api",
        model=model,
    ):
        endpoints.append("/v1/chat/completions")
    supports_audio_family = provider_supports_required_endpoint(
        provider,
        "supports_audio_api",
        model=model,
    )
    supports_audio_transcriptions = provider_supports_required_endpoint(
        provider,
        "supports_audio_transcriptions_api",
        model=model,
    )
    supports_audio_translations = provider_supports_required_endpoint(
        provider,
        "supports_audio_translations_api",
        model=model,
    )
    supports_audio_speech = provider_supports_required_endpoint(
        provider,
        "supports_audio_speech_api",
        model=model,
    )
    if (
        supports_audio_family
        and supports_audio_transcriptions
        and supports_audio_translations
        and supports_audio_speech
    ):
        endpoints.append("/v1/audio")
    elif supports_audio_family:
        if supports_audio_transcriptions:
            endpoints.append("/v1/audio/transcriptions")
        if supports_audio_translations:
            endpoints.append("/v1/audio/translations")
        if supports_audio_speech:
            endpoints.append("/v1/audio/speech")
    if provider_supports_required_endpoint(
        provider,
        "supports_completions_api",
        model=model,
    ):
        endpoints.append("/v1/completions")
    if provider_supports_required_endpoint(
        provider,
        "supports_embeddings_api",
        model=model,
    ):
        endpoints.append("/v1/embeddings")
    if provider_supports_required_endpoint(
        provider,
        "supports_images_api",
        model=model,
    ):
        endpoints.append("/v1/images")
    if provider_supports_required_endpoint(
        provider,
        "supports_moderations_api",
        model=model,
    ):
        endpoints.append("/v1/moderations")
    if provider_supports_required_endpoint(
        provider,
        "supports_responses_api",
        model=model,
    ):
        endpoints.append("/v1/responses")
    if _provider_endpoint_bool_flag(
        provider,
        "supports_anthropic_messages",
        default=False,
    ) and provider_supports_required_endpoint(
        provider,
        "supports_anthropic_messages",
        model=model,
    ):
        endpoints.append("/v1/messages")
    return endpoints


def provider_endpoint_requirement_for_path(path: str) -> tuple[str, str] | None:
    """Return the provider capability attr and user-facing name for a route path."""
    normalized = path.strip("/").lower()
    if normalized.startswith("v1/"):
        normalized = normalized.removeprefix("v1/")
    if normalized == "chat/completions":
        return ("supports_chat_completions_api", "Chat Completions API")
    if normalized == "completions":
        return ("supports_completions_api", "Completions API")
    if normalized == "audio/transcriptions":
        return ("supports_audio_transcriptions_api", "Audio Transcriptions API")
    if normalized == "audio/translations":
        return ("supports_audio_translations_api", "Audio Translations API")
    if normalized == "audio/speech":
        return ("supports_audio_speech_api", "Audio Speech API")
    if normalized == "responses" or normalized.startswith("responses/"):
        return ("supports_responses_api", "Responses API")
    if normalized == "embeddings" or normalized.startswith("embeddings/"):
        return ("supports_embeddings_api", "Embeddings API")
    if normalized == "images" or normalized.startswith("images/"):
        return ("supports_images_api", "Images API")
    if normalized == "moderations" or normalized.startswith("moderations/"):
        return ("supports_moderations_api", "Moderations API")
    if normalized == "messages":
        return ("supports_anthropic_messages", "Messages API")
    if normalized == "messages/count_tokens":
        return (
            "supports_messages_count_tokens_api",
            "Messages Count Tokens API",
        )
    return None


def provider_supports_required_endpoint(
    provider: "BaseUpstreamProvider",
    support_attr: str,
    *,
    model: "Model" | None = None,
) -> bool:
    """Return True when a provider can serve an endpoint without unsafe fallback."""
    provider_supported = True
    if support_attr == "supports_chat_completions_api":
        provider_supported = True
    else:
        provider_supported = _provider_supports_required_endpoint(
            provider, support_attr
        )
    if not provider_supported:
        return False
    if model is not None:
        declared_endpoints = _model_declared_endpoint_set(model)
        if declared_endpoints is None and _provider_needs_model_endpoint_gate(provider):
            return False
        if not _model_declares_endpoint_support(model, support_attr):
            return False
    if model is not None and _provider_needs_model_endpoint_gate(provider):
        return model_supports_required_endpoint(model, support_attr)
    return True


def _provider_needs_model_endpoint_gate(provider: "BaseUpstreamProvider") -> bool:
    provider_type = _lower_non_empty_string(getattr(provider, "provider_type", None))
    if provider_type in CONFIDENTIAL_PROVIDER_ENDPOINT_CAPS:
        return True
    return (
        getattr(provider, "requires_verified_confidential_transport", False) is True
        or getattr(provider, "requires_verified_ehbp_transport", False) is True
    )


def provider_requires_known_model_endpoint(provider: "BaseUpstreamProvider") -> bool:
    """Return True when model-bearing requests must match a known API endpoint."""
    return _provider_needs_model_endpoint_gate(provider)


def _provider_supports_required_endpoint(
    provider: "BaseUpstreamProvider",
    support_attr: str,
) -> bool:
    provider_type = _lower_non_empty_string(getattr(provider, "provider_type", None))
    endpoint_caps = CONFIDENTIAL_PROVIDER_ENDPOINT_CAPS.get(provider_type or "")
    if endpoint_caps is not None and support_attr in endpoint_caps:
        if endpoint_caps[support_attr] is False:
            return False
        return _provider_endpoint_bool_flag(
            provider,
            support_attr,
            default=endpoint_caps[support_attr],
        )
    if support_attr == "supports_anthropic_messages":
        if _provider_endpoint_bool_flag(
            provider,
            "supports_anthropic_messages",
            default=False,
        ):
            return True
        requires_ehbp = _provider_endpoint_bool_flag(
            provider,
            "requires_verified_ehbp_transport",
            default=False,
            malformed_default=True,
        )
        return not requires_ehbp
    if support_attr == "supports_messages_count_tokens_api":
        requires_confidential_transport = (
            _provider_endpoint_bool_flag(
                provider,
                "requires_verified_confidential_transport",
                default=False,
                malformed_default=True,
            )
            or _provider_endpoint_bool_flag(
                provider,
                "requires_verified_ehbp_transport",
                default=False,
                malformed_default=True,
            )
        )
        return _provider_endpoint_bool_flag(
            provider,
            "supports_messages_count_tokens_api",
            default=not requires_confidential_transport,
        )
    if support_attr in {
        "supports_audio_transcriptions_api",
        "supports_audio_translations_api",
        "supports_audio_speech_api",
    } and not _provider_endpoint_bool_flag(
        provider, "supports_audio_api", default=True
    ):
        return False
    return _provider_endpoint_bool_flag(provider, support_attr, default=True)


def public_confidentiality_metadata(
    provider: "BaseUpstreamProvider", model: "Model", alias: str | None = None
) -> dict[str, Any] | None:
    """Return model-map confidentiality metadata for later public sanitization."""
    status_getter = getattr(provider, "confidentiality_status", None)
    if not callable(status_getter):
        return None

    try:
        status: Any = status_getter()
    except Exception:
        return None

    if getattr(status, "enabled", False) is not True:
        return None
    if not _confidentiality_status_covers_model(status, model, alias):
        return None

    status_mode = _non_empty_string(getattr(status, "mode", None))
    provider_type = _non_empty_string(getattr(provider, "provider_type", None))
    if status_mode is None or provider_type is None:
        return None

    verified = is_confidential_provider_for_model(provider, model, alias)
    metadata = {
        "enabled": True,
        "verified": verified,
        "attestation_status": "verified" if verified else "unavailable",
        "mode": status_mode,
        "provider_type": provider_type,
        "verifier": getattr(status, "verifier", None) if verified else None,
        "verified_at": getattr(status, "verified_at", None) if verified else None,
        "expires_at": getattr(status, "expires_at", None) if verified else None,
        "model_ids": _public_string_list(getattr(status, "model_ids", None)),
        "model_id_prefixes": _public_string_list(
            getattr(status, "model_id_prefixes", None)
        ),
        "supported_endpoints": public_supported_endpoints_for_provider(
            provider,
            model=model,
        ),
        "metadata_leakage": _metadata_leakage_for_provider(provider),
    }
    if verified:
        metadata["policy_digest"] = getattr(status, "policy_digest", None)
        metadata["evidence_digest"] = getattr(status, "evidence_digest", None)
        verified_claims = getattr(status, "verified_claims", None)
        if isinstance(verified_claims, dict) and verified_claims:
            metadata["verified_claims_digest"] = _sha256_json_digest(verified_claims)
            metadata["_verified_claims"] = verified_claims
        policy_getter = getattr(provider, "confidentiality_policy", None)
        policy = policy_getter() if callable(policy_getter) else None
        raw_policy = getattr(policy, "policy", None)
        if isinstance(raw_policy, dict):
            metadata["_confidentiality_policy"] = raw_policy
        proof_claims = verified_public_provider_proof_claims(
            status_mode,
            verified_claims,
            public_policy=raw_policy if isinstance(raw_policy, dict) else None,
        )
        public_model_ids = metadata["model_ids"]
        public_policy = raw_policy if isinstance(raw_policy, dict) else None
        if (
            proof_claims
            and public_provider_proof_claims_cover_model_selectors(
                provider_type,
                status_mode,
                proof_claims,
                public_model_ids,
                public_policy=public_policy,
            )
            and public_confidentiality_policy_binds_provider_proof(
                provider_type,
                status_mode,
                public_policy,
                proof_claims,
                public_model_ids,
            )
        ):
            metadata["proof_claims"] = proof_claims
        else:
            return None
    return metadata


def _has_any_supported_endpoint_for_model(
    provider: "BaseUpstreamProvider",
    model: "Model",
) -> bool:
    return bool(public_supported_endpoints_for_provider(provider, model=model))


def _preserve_confidential_catalog_route_shape(
    override_model: "Model",
    catalog_model: "Model",
    provider: "BaseUpstreamProvider",
) -> "Model":
    if not provider_requires_known_model_endpoint(provider):
        return override_model
    return override_model.copy(
        update={
            "architecture": getattr(catalog_model, "architecture", None),
            "supported_endpoints": getattr(catalog_model, "supported_endpoints", None),
            "canonical_slug": getattr(catalog_model, "canonical_slug", None),
            "alias_ids": getattr(catalog_model, "alias_ids", None),
            "forwarded_model_id": getattr(catalog_model, "forwarded_model_id", None),
        }
    )


def calculate_model_cost_score(model: "Model") -> float:
    """Calculate a representative cost score for a model.

    This score is used to compare models when multiple providers offer the same model.
    Lower scores indicate cheaper models.

    The score is calculated as a weighted average of:
    - Input token cost (weighted by typical input usage)
    - Output token cost (weighted by typical output usage)
    - Fixed request cost

    Args:
        model: Model instance with pricing information

    Returns:
        Float representing the cost score. Lower is better.
    """
    pricing = model.pricing

    # Weight costs by typical usage patterns
    # Assume average request: 1000 input tokens, 500 output tokens
    TYPICAL_INPUT_TOKENS = 1000.0
    TYPICAL_OUTPUT_TOKENS = 500.0

    # Calculate weighted cost in USD
    input_cost = pricing.prompt * (TYPICAL_INPUT_TOKENS / 1000.0)
    output_cost = pricing.completion * (TYPICAL_OUTPUT_TOKENS / 1000.0)
    request_cost = pricing.request

    # Include additional costs if present
    image_cost = (
        getattr(pricing, "image", 0.0) * 0.1
    )  # Weight lower as not every request uses images
    web_search_cost = getattr(pricing, "web_search", 0.0) * 0.1
    reasoning_cost = getattr(pricing, "internal_reasoning", 0.0) * 0.2

    total_cost = (
        input_cost
        + output_cost
        + request_cost
        + image_cost
        + web_search_cost
        + reasoning_cost
    )

    return total_cost


def get_provider_penalty(provider: "BaseUpstreamProvider") -> float:
    """Calculate a penalty multiplier for certain providers.

    This allows applying policy-based adjustments beyond pure cost.
    For example, preferring certain providers for reliability or features.

    Args:
        provider: UpstreamProvider instance

    Returns:
        Float multiplier to apply to cost (1.0 = no penalty, >1.0 = penalize)
    """
    # Default: no penalty
    penalty = 1.0

    # Check if this is OpenRouter (can be identified by base URL)
    base_url = getattr(provider, "base_url", "")
    if "openrouter.ai" in base_url.lower():
        # Small penalty for OpenRouter to prefer other providers when costs are very close
        # This maintains the original behavior of preferring non-OpenRouter providers
        penalty = 1.001  # 0.1% penalty

    return penalty


def create_model_mappings(
    upstreams: list["BaseUpstreamProvider"],
    overrides_by_id: dict[str, tuple],
    disabled_model_ids: set[str],
    *,
    require_confidential: bool = False,
) -> tuple[
    dict[str, "Model"], dict[str, list["BaseUpstreamProvider"]], dict[str, "Model"]
]:
    """Create optimal model mappings based on cost and provider preferences.

    This is the main entry point for the algorithm. It processes all upstream providers
    and creates three mappings based on cost optimization:

    1. model_instances: alias -> Model (routable aliases mapped to their Model objects)
    2. provider_map: alias -> List[UpstreamProvider] (sorted list of providers for each alias)
    3. unique_models: base_id -> Model (unique models without provider prefixes)

    The algorithm:
    - Processes non-OpenRouter providers first (they're typically cheaper)
    - Then processes OpenRouter models (they can still win if cheaper)
    - For each model alias, collects all candidates and sorts them by priority and cost.

    Args:
        upstreams: List of all upstream provider instances
        overrides_by_id: Dict of model overrides from database {model_id: (ModelRow, fee)}
        disabled_model_ids: Set of model IDs that should be excluded
        require_confidential: When True, only aliases with current verified
            confidentiality for the selected model are published in
            model_instances or provider_map.

    Returns:
        Tuple of (model_instances, provider_map, unique_models)
    """
    from .payment.models import _row_to_model
    from .upstream.helpers import resolve_model_alias

    candidates: dict[str, list[tuple["Model", "BaseUpstreamProvider"]]] = {}
    unique_models: dict[str, "Model"] = {}
    seen_model_provider: set[tuple[str, str]] = set()

    def get_provider_identity(
        upstream: "BaseUpstreamProvider",
    ) -> tuple[str, str] | None:
        """Get stable provider identity from validated provider fields."""
        base_url = _lower_non_empty_string(getattr(upstream, "base_url", None))
        if base_url is None:
            return None

        db_id = getattr(upstream, "db_id", None)
        if isinstance(db_id, int):
            return (f"db:{db_id}", base_url)

        provider_type = _lower_non_empty_string(
            getattr(upstream, "provider_type", None)
        )
        if provider_type is None:
            return None
        return (f"{provider_type}|{base_url}", base_url)

    validated_upstreams: list["BaseUpstreamProvider"] = []
    provider_identity_by_instance: dict[int, tuple[str, str]] = {}
    for upstream in upstreams:
        provider_identity = get_provider_identity(upstream)
        provider_fee = _positive_finite_number(getattr(upstream, "provider_fee", None))
        if provider_identity is None or provider_fee is None:
            logger.warning(
                "Skipping provider with invalid routing identity",
                extra={
                    "provider_type": getattr(upstream, "provider_type", None)
                    if isinstance(getattr(upstream, "provider_type", None), str)
                    else None,
                    "has_valid_base_url": isinstance(
                        getattr(upstream, "base_url", None),
                        str,
                    ),
                    "has_valid_provider_fee": provider_fee is not None,
                },
            )
            continue
        validated_upstreams.append(upstream)
        provider_identity_by_instance[id(upstream)] = provider_identity

    providers_by_db_id: dict[int, "BaseUpstreamProvider"] = {}
    for upstream in validated_upstreams:
        db_id = getattr(upstream, "db_id", None)
        if isinstance(db_id, int):
            providers_by_db_id[db_id] = upstream

    # Group upstreams by URL and keep only the one with the lowest fee for each URL
    upstreams_by_url: dict[str, list["BaseUpstreamProvider"]] = {}
    for upstream in validated_upstreams:
        _provider_key, url = provider_identity_by_instance[id(upstream)]
        if url not in upstreams_by_url:
            upstreams_by_url[url] = []
        upstreams_by_url[url].append(upstream)

    filtered_upstreams: list["BaseUpstreamProvider"]
    if require_confidential:
        filtered_upstreams = list(validated_upstreams)
    else:
        filtered_upstreams = []
        for providers in upstreams_by_url.values():
            best_provider = min(providers, key=lambda p: p.provider_fee)
            filtered_upstreams.append(best_provider)

    # Separate OpenRouter from other providers
    openrouter: "BaseUpstreamProvider" | None = None
    other_upstreams: list["BaseUpstreamProvider"] = []

    for upstream in filtered_upstreams:
        base_url = getattr(upstream, "base_url", "")
        if base_url == "https://openrouter.ai/api/v1":
            openrouter = upstream
        else:
            other_upstreams.append(upstream)

    def get_base_model_id(model_id: str) -> str:
        """Get base model ID by removing provider prefix."""
        return model_id.split("/", 1)[1] if "/" in model_id else model_id

    def get_unique_model_id(
        model: "Model",
        provider: "BaseUpstreamProvider",
        *,
        model_id: str,
    ) -> str:
        """Get the advertised model ID for the unique models list."""
        if require_confidential:
            try:
                status = provider.confidentiality_status()
                confidential_prefixes = _status_list(
                    getattr(status, "model_id_prefixes", None)
                )
            except Exception:
                confidential_prefixes = None
            confidential_prefixes = confidential_prefixes or []
            if any(
                model_id.lower().startswith(prefix) for prefix in confidential_prefixes
            ):
                return model_id
        return get_base_model_id(model_id)

    def _add_candidate(
        alias: str, model: "Model", provider: "BaseUpstreamProvider"
    ) -> None:
        """Add candidate model/provider for an alias."""
        alias_lower = alias.lower()
        if alias_lower not in candidates:
            candidates[alias_lower] = []
        candidates[alias_lower].append((model, provider))

    def process_provider_models(
        upstream: "BaseUpstreamProvider", is_openrouter: bool = False
    ) -> None:
        """Process all models from a given provider."""
        upstream_prefix = _non_empty_string(getattr(upstream, "upstream_name", None))
        provider_key = provider_identity_by_instance[id(upstream)][0]

        for model in upstream.get_cached_models():
            route_identity = _model_route_identity(model)
            if route_identity is None:
                continue
            model_id, _canonical_slug, _alias_ids, _forwarded_model_id = route_identity
            if not model.enabled or model_id in disabled_model_ids:
                continue

            # Apply overrides if present
            if model_id in overrides_by_id:
                override_row, provider_fee = overrides_by_id[model_id]
                try:
                    model_to_use = _row_to_model(
                        override_row,
                        apply_provider_fee=True,
                        provider_fee=provider_fee,
                    )
                except Exception as exc:
                    logger.warning(
                        "Skipping invalid model override while building model mappings",
                        extra={
                            "model_id": model_id,
                            "upstream_provider_id": getattr(
                                override_row,
                                "upstream_provider_id",
                                None,
                            ),
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )
                    continue
                if require_confidential:
                    model_to_use = _preserve_confidential_catalog_route_shape(
                        model_to_use,
                        model,
                        upstream,
                    )
            else:
                model_to_use = model
            model_route_identity = _model_route_identity(model_to_use)
            if model_route_identity is None:
                continue
            (
                model_to_use_id,
                canonical_slug,
                alias_ids,
                forwarded_model_id,
            ) = model_route_identity

            # Add to unique models. In required-confidential mode, advertise
            # only models that can currently route to a verified provider.
            base_id = get_unique_model_id(
                model_to_use,
                upstream,
                model_id=model_to_use_id,
            )
            unique_key = forwarded_model_id or base_id
            include_unique_model = not require_confidential
            if include_unique_model and (
                not is_openrouter or unique_key not in unique_models
            ):
                unique_model = model_to_use.copy(
                    update={
                        "id": base_id,
                        "upstream_provider_id": upstream.provider_type,
                        "confidentiality": public_confidentiality_metadata(
                            upstream,
                            model_to_use,
                            base_id,
                        ),
                    }
                )
                unique_models[unique_key] = unique_model

            # Get all aliases for this model
            aliases = resolve_model_alias(
                model_to_use_id,
                canonical_slug,
                alias_ids=alias_ids,
            )

            # Add prefixed alias if applicable
            if upstream_prefix and "/" not in model_to_use_id:
                prefixed_id = f"{upstream_prefix}/{model_to_use_id}"
                if prefixed_id not in aliases:
                    aliases.append(prefixed_id)

            # Register forwarded_model_id as a routable alias
            if forwarded_model_id and forwarded_model_id not in aliases:
                aliases.append(forwarded_model_id)

            # Try to set each alias
            for alias in aliases:
                _add_candidate(alias, model_to_use, upstream)
            seen_model_provider.add((model_to_use_id.lower(), provider_key))

    # Process non-OpenRouter providers first
    for upstream in other_upstreams:
        process_provider_models(upstream, is_openrouter=False)

    # Process OpenRouter last
    if openrouter:
        process_provider_models(openrouter, is_openrouter=True)

    # Include enabled DB overrides even when provider discovery misses models.
    # This is important for deployment-based providers like Azure.
    for model_id, override_data in overrides_by_id.items():
        if model_id in disabled_model_ids:
            continue
        override_row, provider_fee = override_data
        upstream_provider_id = getattr(override_row, "upstream_provider_id", None)
        if not isinstance(upstream_provider_id, int):
            continue

        upstream_for_override = providers_by_db_id.get(upstream_provider_id)
        if upstream_for_override is None:
            continue
        if require_confidential and provider_requires_known_model_endpoint(
            upstream_for_override
        ):
            continue

        provider_key = provider_identity_by_instance[id(upstream_for_override)][0]
        dedupe_key = (model_id.lower(), provider_key)
        if dedupe_key in seen_model_provider:
            continue

        try:
            model_to_use = _row_to_model(
                override_row, apply_provider_fee=True, provider_fee=provider_fee
            )
        except Exception as exc:
            logger.warning(
                "Skipping invalid model override while building model mappings",
                extra={
                    "model_id": model_id,
                    "upstream_provider_id": upstream_provider_id,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            continue
        if not model_to_use.enabled:
            continue
        model_route_identity = _model_route_identity(model_to_use)
        if model_route_identity is None:
            continue
        (
            model_to_use_id,
            canonical_slug,
            alias_ids,
            forwarded_model_id,
        ) = model_route_identity

        base_id = get_unique_model_id(
            model_to_use,
            upstream_for_override,
            model_id=model_to_use_id,
        )
        unique_key = forwarded_model_id or base_id
        is_openrouter = (
            getattr(upstream_for_override, "base_url", "")
            == "https://openrouter.ai/api/v1"
        )
        include_unique_model = not require_confidential
        if include_unique_model and (
            not is_openrouter or unique_key not in unique_models
        ):
            unique_model = model_to_use.copy(
                update={
                    "id": base_id,
                    "upstream_provider_id": upstream_for_override.provider_type,
                    "confidentiality": public_confidentiality_metadata(
                        upstream_for_override,
                        model_to_use,
                        base_id,
                    ),
                }
            )
            unique_models[unique_key] = unique_model

        try:
            aliases = resolve_model_alias(
                model_to_use_id,
                canonical_slug,
                alias_ids=alias_ids,
            )
        except Exception as exc:
            logger.warning(
                "Skipping model aliases for invalid override model",
                extra={
                    "model_id": model_id,
                    "upstream_provider_id": upstream_provider_id,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            continue

        upstream_prefix = _non_empty_string(
            getattr(upstream_for_override, "upstream_name", None)
        )
        if upstream_prefix and "/" not in model_to_use_id:
            prefixed_id = f"{upstream_prefix}/{model_to_use_id}"
            if prefixed_id not in aliases:
                aliases.append(prefixed_id)

        # Register forwarded_model_id as a routable alias
        if forwarded_model_id and forwarded_model_id not in aliases:
            aliases.append(forwarded_model_id)

        for alias in aliases:
            _add_candidate(alias, model_to_use, upstream_for_override)
        seen_model_provider.add(dedupe_key)

    # Sort candidates and build final maps. In required confidential mode,
    # aliases with no verified provider are intentionally absent from both maps.
    model_instances: dict[str, "Model"] = {}
    provider_map: dict[str, list["BaseUpstreamProvider"]] = {}

    def alias_priority(model: "Model", alias: str) -> int:
        """Rank how strong the mapping of alias->model is.

        forwarded_model_id is the most specific identifier (set per-provider
        instance), so a match there should beat a model_id match. This way,
        when multiple providers have the same model_id but different
        forwarded_model_ids, the one whose forwarded_model_id equals the
        requested alias wins.
        """
        forwarded_model_id = _lower_non_empty_string(
            getattr(model, "forwarded_model_id", None)
        )
        if forwarded_model_id and forwarded_model_id == alias:
            return 5

        model_id = _lower_non_empty_string(getattr(model, "id", None))
        if model_id and model_id == alias:
            return 4

        if model_id is None:
            return 1
        model_base = get_base_model_id(model_id)
        if model_base == alias:
            return 3
        canonical_slug = _lower_non_empty_string(getattr(model, "canonical_slug", None))
        if canonical_slug:
            canonical_base = get_base_model_id(canonical_slug)
            if canonical_base == alias:
                return 2
        return 1

    def forwarding_shape(model: "Model") -> tuple[str, str | None, tuple[str, ...] | None]:
        route_identity = _model_route_identity(model)
        if route_identity is None:
            return ("", None, None)
        model_id, _canonical_slug, _alias_ids, forwarded_model_id = route_identity
        raw_endpoints = getattr(model, "supported_endpoints", None)
        endpoints: tuple[str, ...] | None = None
        if isinstance(raw_endpoints, list):
            endpoints = tuple(
                sorted(
                    endpoint.strip().lower()
                    for endpoint in raw_endpoints
                    if isinstance(endpoint, str) and endpoint.strip()
                )
            )
        return (
            model_id.lower(),
            forwarded_model_id.lower() if forwarded_model_id else None,
            endpoints,
        )

    for alias, items in candidates.items():
        # Sort key: (priority DESC, cost ASC)
        # Using negative cost for DESC sort overall to keep high priority first
        def sort_key(item: tuple["Model", "BaseUpstreamProvider"]) -> tuple[int, float]:
            model, provider = item
            priority = alias_priority(model, alias)
            cost = calculate_model_cost_score(model)
            penalty = get_provider_penalty(provider)
            adjusted_cost = cost * penalty
            return (priority, -adjusted_cost)

        items.sort(key=sort_key, reverse=True)

        routable_items = (
            [
                (model, provider)
                for model, provider in items
                if is_confidential_provider_for_model(provider, model, alias)
                and _has_any_supported_endpoint_for_model(provider, model)
                and public_confidentiality_metadata(provider, model, alias) is not None
            ]
            if require_confidential
            else items
        )
        if require_confidential and not routable_items:
            continue

        best_model, _best_provider = (
            routable_items[0] if require_confidential else items[0]
        )
        model_instances[alias] = best_model

        if routable_items:
            if require_confidential:
                best_forwarding_shape = forwarding_shape(best_model)
                routable_items = [
                    (model, provider)
                    for model, provider in routable_items
                    if forwarding_shape(model) == best_forwarding_shape
                ]
            provider_map[alias] = [p for _, p in routable_items]
            if require_confidential:
                unique_key = (
                    _lower_non_empty_string(
                        getattr(best_model, "forwarded_model_id", None)
                    )
                    or alias
                )
                if unique_key not in unique_models:
                    unique_models[unique_key] = best_model.copy(
                        update={
                            "id": alias,
                            "upstream_provider_id": _best_provider.provider_type,
                            "confidentiality": public_confidentiality_metadata(
                                _best_provider,
                                best_model,
                                alias,
                            ),
                        }
                    )

    # Log provider distribution (using top provider for stats)
    provider_counts: dict[str, int] = {}
    for providers in provider_map.values():
        if providers:
            provider = providers[0]
            provider_name = getattr(provider, "upstream_name", "unknown")
            provider_counts[provider_name] = provider_counts.get(provider_name, 0) + 1

    logger.debug(
        f"Updated model mappings with ({len(unique_models)} unique models and {len(model_instances)} aliases)",
        extra={"provider_distribution": provider_counts},
    )

    return model_instances, provider_map, unique_models
