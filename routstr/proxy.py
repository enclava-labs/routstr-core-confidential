import hashlib
import ipaddress
import json
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from sqlmodel import select

from .algorithm import (
    create_model_mappings,
    has_current_confidential_verification,
    is_confidential_provider_for_model,
    provider_endpoint_requirement_for_path,
    provider_requires_known_model_endpoint,
    provider_supports_required_endpoint,
    public_confidentiality_metadata,
    public_supported_endpoints_for_provider,
)
from .auth import pay_for_request, revert_pay_for_request, validate_bearer_key
from .core import get_logger
from .core.confidentiality_public import (
    is_full_sha256_digest,
    provider_evidence_digest_matches_status,
    public_confidentiality_policy_binds_provider_proof,
    public_provider_proof_claims_cover_model_selectors,
    public_provider_type_satisfies_mode,
    verified_public_provider_proof_claims,
)
from .core.db import (
    ApiKey,
    AsyncSession,
    ModelRow,
    ProviderModelAttestationRow,
    UpstreamProviderRow,
    canonical_json,
    create_session,
    get_session,
)
from .core.exceptions import UpstreamError
from .core.logging import (
    credential_fingerprint,
    redact_sensitive_text,
    redact_url_userinfo,
)
from .core.not_found import build_not_found_response
from .core.policy_secrets import inline_policy_secret_violations
from .core.settings import settings
from .payment.helpers import (
    calculate_discounted_max_cost,
    check_token_balance,
    create_error_response,
    get_max_cost_for_model,
)
from .payment.models import Model
from .upstream import BaseUpstreamProvider
from .upstream.helpers import init_upstreams
from .upstream.strict_json import loads_strict_json

logger = get_logger(__name__)


PUBLIC_CONFIDENTIALITY_POLICY_KEYS = {
    "allowed_code_measurement_fingerprint",
    "allowed_code_measurement_fingerprints",
    "allowed_code_measurements",
    "allowed_enclave_measurement_fingerprint",
    "allowed_enclave_measurement_fingerprints",
    "allowed_enclave_measurements",
    "allowed_release_digest",
    "allowed_release_digests",
    "allowed_coordinator_measurements",
    "allowed_gpu_attestation_policies",
    "allowed_key_release_bindings",
    "allowed_secret_service_measurements",
    "allowed_ai_worker_measurements",
    "code_measurement_fingerprint",
    "enclave_measurement_fingerprint",
    "expected_ai_worker_measurement",
    "expected_code_measurement_fingerprint",
    "expected_coordinator_measurement",
    "expected_enclave_measurement_fingerprint",
    "expected_gpu_attestation_policy",
    "expected_key_release_binding",
    "expected_release_digest",
    "expected_secret_service_measurement",
    "expected_trust_tier",
    "expectedWorkloadIDs",
    "expectedWorkloadSANs",
    "expected_workload_ids",
    "expected_workload_sans",
    "expected_workload_identity_digest",
    "manifestDigest",
    "manifest_digest",
    "manifestLogDir",
    "manifest_log_dir",
    "modelAttestationTargets",
    "modelEnclaveBindings",
    "modelWorkloadBindings",
    "model_workload_bindings",
    "model_workload_binding_digest",
    "model_attestation_targets",
    "model_enclave_bindings",
    "proxyBinaryDigest",
    "proxyImageDigest",
    "proxy_binary_digest",
    "proxy_image_digest",
    "release_digest",
    "repo",
    "expected_repo",
    "requireModelAttestations",
    "require_model_attestations",
    "tinfoil_transport_security",
    "transport_security",
}
PUBLIC_TINFOIL_TARGET_POLICY_KEYS = {
    "allowed_code_measurement_fingerprint",
    "allowed_code_measurement_fingerprints",
    "allowed_code_measurements",
    "allowed_enclave_measurement_fingerprint",
    "allowed_enclave_measurement_fingerprints",
    "allowed_enclave_measurements",
    "allowed_release_digest",
    "allowed_release_digests",
    "code_measurement_fingerprint",
    "enclave_measurement_fingerprint",
    "expected_code_measurement_fingerprint",
    "expected_enclave_host",
    "expected_enclave_measurement_fingerprint",
    "expected_release_digest",
    "expected_repo",
    "host",
    "release_digest",
    "repo",
    "tinfoil_transport_security",
    "transport_security",
}
PUBLIC_PRIVATEMODE_WORKLOAD_BINDING_POLICY_KEYS = {
    "workloadIDs",
    "workloadSANs",
    "workload_ids",
    "workload_sans",
}
PUBLIC_BOOLEAN_CONFIDENTIALITY_POLICY_KEYS = {
    "requireModelAttestations",
    "require_model_attestations",
}
PUBLIC_DIGEST_CONFIDENTIALITY_POLICY_KEYS = {
    "attestation_bundle_url_digest",
    "code_measurement_fingerprint",
    "enclave_measurement_fingerprint",
    "expected_ai_worker_measurement",
    "expected_code_measurement_fingerprint",
    "expected_coordinator_measurement",
    "expected_enclave_measurement_fingerprint",
    "expected_key_release_binding",
    "expected_release_digest",
    "expected_secret_service_measurement",
    "expected_workload_identity_digest",
    "manifestDigest",
    "manifest_digest",
    "model_workload_binding_digest",
    "proxyBinaryDigest",
    "proxyImageDigest",
    "proxy_binary_digest",
    "proxy_image_digest",
    "release_digest",
}
PUBLIC_DIGEST_LIST_CONFIDENTIALITY_POLICY_KEYS = {
    "allowed_ai_worker_measurements",
    "allowed_code_measurement_fingerprint",
    "allowed_code_measurement_fingerprints",
    "allowed_code_measurements",
    "allowed_coordinator_measurements",
    "allowed_enclave_measurement_fingerprint",
    "allowed_enclave_measurement_fingerprints",
    "allowed_enclave_measurements",
    "allowed_key_release_bindings",
    "allowed_release_digest",
    "allowed_release_digests",
    "allowed_secret_service_measurements",
}
PUBLIC_DIGEST_TINFOIL_TARGET_POLICY_KEYS = {
    "code_measurement_fingerprint",
    "enclave_measurement_fingerprint",
    "expected_code_measurement_fingerprint",
    "expected_enclave_measurement_fingerprint",
    "expected_release_digest",
    "release_digest",
}
PUBLIC_DIGEST_LIST_TINFOIL_TARGET_POLICY_KEYS = {
    "allowed_code_measurement_fingerprint",
    "allowed_code_measurement_fingerprints",
    "allowed_code_measurements",
    "allowed_enclave_measurement_fingerprint",
    "allowed_enclave_measurement_fingerprints",
    "allowed_enclave_measurements",
    "allowed_release_digest",
    "allowed_release_digests",
}
TINFOIL_DEFAULT_BASE_URL = "https://inference.tinfoil.sh/v1"
proxy_router = APIRouter()

_upstreams: list[BaseUpstreamProvider] = []
_model_instances: dict[str, Model] = {}  # All aliases -> Model
_provider_map: dict[
    str, list[BaseUpstreamProvider]
] = {}  # All aliases -> List[Provider]
_unique_models: dict[str, Model] = {}  # Unique model.id -> Model (no duplicates)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_json_digest(value: object) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _safe_error_message(exc: BaseException | object) -> str:
    return str(redact_sensitive_text(str(exc)))


def _public_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _public_string_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]


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


def _url_has_private_path(value: str) -> bool:
    parsed = urlsplit(value)
    return any(
        segment.lower() == "private"
        for segment in parsed.path.split("/")
        if segment
    )


def _public_provider_base_url(
    value: object,
    *,
    provider_type: str,
    confidentiality_mode: object,
) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    base_url = value.strip()
    try:
        parsed = urlsplit(base_url)
    except ValueError:
        return None
    if not parsed.scheme or not parsed.netloc or not parsed.hostname:
        return None
    try:
        parsed.port
    except ValueError:
        return None
    if "@" in parsed.netloc:
        base_url = urlunsplit(
            (
                parsed.scheme,
                parsed.netloc.rsplit("@", 1)[-1],
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )
        parsed = urlsplit(base_url)

    mode = (
        confidentiality_mode.strip().lower()
        if isinstance(confidentiality_mode, str)
        else ""
    )
    normalized_provider = provider_type.strip().lower()
    if normalized_provider == "tinfoil" or mode == "tinfoil":
        return base_url if base_url.rstrip("/") == TINFOIL_DEFAULT_BASE_URL else None
    if normalized_provider == "privatemode" or mode == "privatemode":
        if parsed.scheme not in {"http", "https"} or not _is_loopback_hostname(
            parsed.hostname
        ):
            return None
        return base_url
    if normalized_provider == "ppq-private" or mode == "ppq-private-tee":
        if parsed.scheme != "https" or not _url_has_private_path(base_url):
            return None
        return base_url
    if parsed.scheme != "https":
        return None
    return base_url


def _routstr_tee_attested_tls_boundary(
    value: object,
    *,
    require_public_key_digest: bool = True,
) -> bool:
    if not isinstance(value, dict):
        return False
    public_key_digest = value.get("attested_tls_public_key_digest")
    if require_public_key_digest:
        public_key_digest_valid = is_full_sha256_digest(public_key_digest)
    else:
        public_key_digest_valid = public_key_digest is None or is_full_sha256_digest(
            public_key_digest
        )
    return (
        value.get("mode") == "attested-tls-termination"
        and value.get("tls_terminates_in_attested_tee") is True
        and value.get("inbound_ehbp_ohttp_request_decryption") is False
        and public_key_digest_valid
    )


def _fresh_routstr_tee_local_verification(
    value: dict[str, Any], attestation_evidence_digest: object
) -> dict[str, Any] | None:
    local_verification = value.get("local_verification")
    if not isinstance(local_verification, dict):
        return None
    if local_verification.get("verified") is not True:
        return None
    verified_at = local_verification.get("verified_at")
    expires_at = local_verification.get("expires_at")
    if type(verified_at) is not int or type(expires_at) is not int:
        return None
    now = int(time.time())
    if verified_at > now or expires_at <= now:
        return None
    if local_verification.get("evidence_digest") != attestation_evidence_digest:
        return None
    if not is_full_sha256_digest(local_verification.get("verified_claims_digest")):
        return None
    return local_verification


def _routstr_tee_public_status_has_hpke_proof(
    value: dict[str, Any],
    *,
    client_confidentiality: dict[str, Any],
) -> bool:
    attestation_evidence_digest = value.get("attestation_evidence_digest")
    hpke_key_config_digest = value.get("hpke_key_config_digest")
    hpke_public_key_digest = value.get("hpke_public_key_digest")
    if not (
        is_full_sha256_digest(attestation_evidence_digest)
        and is_full_sha256_digest(hpke_key_config_digest)
        and is_full_sha256_digest(hpke_public_key_digest)
    ):
        return False

    local_verification = _fresh_routstr_tee_local_verification(
        value, attestation_evidence_digest
    )
    if local_verification is None:
        return False

    proof_claims = local_verification.get("proof_claims")
    if not isinstance(proof_claims, dict):
        return False
    if proof_claims.get("hpke_key_config_digest") != hpke_key_config_digest:
        return False
    if proof_claims.get("hpke_public_key_digest") != hpke_public_key_digest:
        return False
    if (
        proof_claims.get("public_key_digest")
        != client_confidentiality.get("attested_tls_public_key_digest")
    ):
        return False
    return True


def _routstr_tee_public_status_has_cap_proof(value: dict[str, Any]) -> bool:
    attestation_evidence_digest = value.get("attestation_evidence_digest")
    if not is_full_sha256_digest(attestation_evidence_digest):
        return False

    local_verification = _fresh_routstr_tee_local_verification(
        value, attestation_evidence_digest
    )
    if local_verification is None:
        return False
    if local_verification.get("verifier") != "cap-attestation-proxy":
        return False

    proof_claims = local_verification.get("proof_claims")
    if not isinstance(proof_claims, dict):
        return False
    if proof_claims.get("attestation_document_format") != "cap-attestation-proxy-status":
        return False
    if proof_claims.get("cap_claims_verified") is not True:
        return False
    if proof_claims.get("cap_state") != "unlocked":
        return False
    if proof_claims.get("client_confidentiality_boundary") != "attested-tls-termination":
        return False
    if proof_claims.get("tls_terminates_in_attested_tee") is not True:
        return False
    for claim in (
        "cap_attestation_url",
        "cap_claims_instance_id",
        "cap_status_url",
        "cap_tee_domain",
        "cap_tenant_id",
    ):
        if _public_string(proof_claims.get(claim)) is None:
            return False
    for url_claim in ("cap_attestation_url", "cap_status_url"):
        claim_url = _public_string(proof_claims.get(url_claim))
        if claim_url is None or urlsplit(claim_url).scheme != "https":
            return False
    if proof_claims.get("cap_status_digest") != attestation_evidence_digest:
        return False
    if not is_full_sha256_digest(proof_claims.get("routstr_config_measurement")):
        return False
    verification_steps = proof_claims.get("verification_steps")
    if not isinstance(verification_steps, dict):
        return False
    for step in (
        "cap_status_verified",
        "cap_claims_verified",
        "cap_state_unlocked",
        "tls_terminates_in_attested_tee",
        "freshness",
    ):
        if verification_steps.get(step) is not True:
            return False
    return True


def _safe_public_routstr_tee_status(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    status = dict(value)
    if status.get("ready") is True:
        client_confidentiality = status.get("client_confidentiality")
        hpke_ready = _routstr_tee_attested_tls_boundary(
            client_confidentiality
        ) and (
            isinstance(client_confidentiality, dict)
            and _routstr_tee_public_status_has_hpke_proof(
                status,
                client_confidentiality=client_confidentiality,
            )
        )
        cap_ready = _routstr_tee_attested_tls_boundary(
            client_confidentiality,
            require_public_key_digest=False,
        ) and _routstr_tee_public_status_has_cap_proof(status)
        if not (hpke_ready or cap_ready):
            status["ready"] = False
    return status


def _invalid_request_field(field: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "error": {
                "type": "invalid_request_error",
                "code": f"invalid_{field}",
            }
        },
    )


def _validated_positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _invalid_request_field(field)
    return value


def _validated_single_completion_count(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value != 1:
        raise _invalid_request_field("n")
    return value


def _validated_positive_int_string(value: str, field: str) -> int:
    stripped = value.strip()
    if not stripped.isdigit():
        raise _invalid_request_field(field)
    return _validated_positive_int(int(stripped), field)


def _extract_redirect_error_message(
    response: Response | StreamingResponse,
    upstream: BaseUpstreamProvider,
) -> str:
    error_message = ""
    try:
        body_bytes = getattr(response, "body", b"")
        if isinstance(body_bytes, str):
            body_bytes = body_bytes.encode()
        if not isinstance(body_bytes, bytes):
            return error_message
        data = json.loads(body_bytes)
        if isinstance(data, dict):
            error_data = data.get("error")
            raw_message: object = None
            if isinstance(error_data, dict):
                raw_message = error_data.get("message")
            elif isinstance(error_data, str):
                raw_message = error_data
            if isinstance(raw_message, (str, int, float)):
                error_message = str(raw_message)
    except Exception:
        return error_message

    confidential = False
    confidential_logging_enabled = getattr(
        upstream, "confidential_logging_enabled", None
    )
    if callable(confidential_logging_enabled):
        confidential = bool(confidential_logging_enabled())
    if confidential:
        return str(redact_sensitive_text(error_message))
    return error_message


async def initialize_upstreams() -> None:
    """Initialize upstream providers from database during application startup."""
    global _upstreams
    _upstreams = await init_upstreams()
    logger.info(f"Initialized {len(_upstreams)} upstream providers")
    await refresh_model_maps()


async def reinitialize_upstreams() -> None:
    """Re-initialize upstream providers from database (called after admin changes)."""
    global _upstreams
    _upstreams = await init_upstreams()
    logger.info(
        "Re-initialized upstream providers from admin action",
        extra={"provider_count": len(_upstreams)},
    )
    await refresh_model_maps()


def get_upstreams() -> list[BaseUpstreamProvider]:
    """Get the initialized upstream providers.

    Returns:
        List of upstream provider instances
    """
    return _upstreams


def get_model_instance(model_id: str) -> Model | None:
    """Get Model instance by ID from global cache."""
    if not model_id:
        return None

    model_id_lower = model_id.lower()
    # Try exact match first
    if model := _model_instances.get(model_id_lower):
        return model

    # Try stripping common version suffixes (e.g., -20251222)
    # This handles cases where upstream returns a specific version
    # but we only track the base model name.
    import re

    base_model_id = re.sub(r"-\d{8}$", "", model_id_lower)
    if base_model_id != model_id_lower:
        if model := _model_instances.get(base_model_id):
            return model

    return None


def get_provider_for_model(model_id: str) -> list[BaseUpstreamProvider] | None:
    """Get UpstreamProvider list for model ID from global cache."""
    return _provider_map.get(model_id.lower())


def get_unique_models() -> list[Model]:
    """Get list of unique models (no duplicates from aliases)."""
    return list(_unique_models.values())


def is_routable_confidential_model(model_id: str, model: Model) -> bool:
    upstreams = get_provider_for_model(model_id)
    if not upstreams:
        return False
    return any(
        is_confidential_provider_for_model(upstream, model, model_id)
        and _has_any_supported_endpoint_for_public_status(upstream, model)
        and public_confidentiality_metadata(upstream, model, model_id) is not None
        for upstream in upstreams
    )


def _has_routable_confidential_model() -> bool:
    for alias, model in _model_instances.items():
        if is_routable_confidential_model(alias, model):
            return True
    return False


def _empty_routable_with_full_attestation() -> dict[str, list[str]]:
    return {"tinfoil": [], "ppq-private": [], "privatemode": []}


def _routable_with_full_attestation(
    *,
    routstr_tee_ready: bool,
) -> dict[str, list[str]]:
    routable = _empty_routable_with_full_attestation()
    if not confidential_routing_required() or not routstr_tee_ready:
        return routable
    seen: dict[str, set[str]] = {provider: set() for provider in routable}
    for alias, model in sorted(_model_instances.items()):
        upstreams = get_provider_for_model(alias) or []
        for upstream in upstreams:
            provider_type = _public_string(getattr(upstream, "provider_type", None))
            if provider_type not in routable:
                continue
            if (
                is_confidential_provider_for_model(upstream, model, alias)
                and _has_any_supported_endpoint_for_public_status(upstream, model)
                and public_confidentiality_metadata(upstream, model, alias) is not None
            ):
                seen[provider_type].add(alias)
    return {provider: sorted(models) for provider, models in seen.items()}


def _has_any_supported_endpoint_for_public_status(
    upstream: BaseUpstreamProvider,
    model: Model,
) -> bool:
    return bool(public_supported_endpoints_for_provider(upstream, model=model))


def confidential_routing_required() -> bool:
    """Return True when plaintext/non-attested upstream routing is disallowed."""
    mode = str(getattr(settings, "confidential_routing_mode", "disabled") or "")
    return mode.strip().lower() in {"required", "require", "strict", "enforced"}


def routstr_tee_attestation_required() -> bool:
    """Return True when Routstr itself must publish local TEE evidence to route."""
    return confidential_routing_required() and bool(
        getattr(settings, "routstr_tee_attestation_required", False)
    )


def get_routstr_tee_unavailable_message() -> str | None:
    if not routstr_tee_attestation_required():
        return None
    from .core.attestation import get_routstr_tee_readiness

    readiness = get_routstr_tee_readiness()
    if readiness.get("ready", False) is True:
        return None
    detail = str(
        redact_sensitive_text(str(readiness.get("failure_reason") or ""))
    ).strip()
    suffix = f": {detail}" if detail else ""
    return (
        "Routstr TEE attestation is required for confidential routing, "
        "but configured TEE evidence or HPKE key config is unavailable"
        f"{suffix}"
    )


async def refresh_confidentiality_statuses(*, persist: bool = True) -> None:
    """Refresh attestation state for all runtime providers before map publish."""
    import asyncio

    async def refresh_one(upstream: BaseUpstreamProvider) -> None:
        refresh = getattr(upstream, "refresh_confidentiality_status", None)
        if not callable(refresh):
            return
        try:
            await refresh()
        except Exception as exc:
            failure_reason = str(redact_sensitive_text(str(exc)))
            current = upstream.confidentiality_status()
            failed_status = current.copy(
                update={
                    "verified": False,
                    "verified_at": None,
                    "expires_at": None,
                    "failure_reason": failure_reason,
                    "verifier": None,
                    "evidence_digest": None,
                    "verified_claims": {},
                }
            )
            upstream.set_confidentiality_status(failed_status)
            logger.warning(
                "Provider confidentiality refresh failed",
                extra={
                    "provider": upstream.provider_type,
                    "base_url": redact_url_userinfo(upstream.base_url),
                    "error": failure_reason,
                    "error_type": type(exc).__name__,
                },
            )

    await asyncio.gather(*(refresh_one(upstream) for upstream in _upstreams))
    if persist:
        async with create_session() as session:
            await _persist_provider_model_attestations(session)


async def _persist_provider_model_attestations(session: AsyncSession) -> None:
    for upstream in _upstreams:
        provider_id = getattr(upstream, "db_id", None)
        if not isinstance(provider_id, int):
            continue
        try:
            status = upstream.confidentiality_status()
        except Exception:
            continue
        mode = _public_string(getattr(status, "mode", None))
        if not mode:
            continue
        model_ids = _public_string_list(getattr(status, "model_ids", None))
        if not model_ids:
            continue
        claims = getattr(status, "verified_claims", None)
        claims_json = None
        if isinstance(claims, dict) and claims:
            try:
                claims_json = canonical_json(claims)
            except (TypeError, ValueError):
                claims_json = None

        now = int(time.time())
        for model_id in model_ids:
            row = await session.get(
                ProviderModelAttestationRow,
                (provider_id, model_id, mode),
            )
            if row is None:
                row = ProviderModelAttestationRow(
                    provider_id=provider_id,
                    model_id=model_id,
                    mode=mode,
                )
                session.add(row)
            row.verified = getattr(status, "verified", False) is True
            row.verified_at = getattr(status, "verified_at", None)
            row.expires_at = getattr(status, "expires_at", None)
            row.policy_digest = getattr(status, "policy_digest", None)
            row.evidence_digest = getattr(status, "evidence_digest", None)
            row.verifier = getattr(status, "verifier", None)
            row.claims_json = claims_json
            row.failure_reason = getattr(status, "failure_reason", None)
            row.updated_at = now
    await session.commit()


async def _load_provider_model_attestations(
    session: AsyncSession,
    provider_ids: set[int],
) -> dict[int, dict[tuple[str, str], ProviderModelAttestationRow]]:
    if not provider_ids:
        return {}
    result = await session.exec(
        select(ProviderModelAttestationRow).where(
            ProviderModelAttestationRow.provider_id.in_(provider_ids)
        )
    )
    rows = result.all()

    attestations: dict[int, dict[tuple[str, str], ProviderModelAttestationRow]] = {}
    for row in rows:
        model_id = _public_string(row.model_id)
        mode = _public_string(row.mode)
        if not model_id or not mode:
            continue
        provider_rows = attestations.setdefault(row.provider_id, {})
        provider_rows[(model_id.lower(), mode.lower())] = row
    return attestations


def _public_confidentiality_status(
    status: Any,
    *,
    provider_type: str | None = None,
    currently_verified: bool | None = None,
    public_policy: dict[str, Any] | None = None,
    include_verified_claims: bool = False,
) -> dict[str, Any]:
    """Return public provider attestation status without raw verifier data."""
    # Compatibility no-op: public diagnostics expose digests and audited proof_claims,
    # never raw verifier output.
    status_dict = status.dict()

    def public_string_list(value: object) -> list[str]:
        if not isinstance(value, (list, tuple, set)):
            return []
        return [
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        ]

    def strict_public_string_list(value: object) -> list[str] | None:
        if value is None:
            return []
        if not isinstance(value, list):
            return None
        values: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                return None
            values.append(item.strip())
        if len({value.lower() for value in values}) != len(values):
            return None
        return values

    def downgrade_verified_status() -> None:
        status_dict["verified"] = False
        status_dict["verifier"] = None
        status_dict.pop("policy_digest", None)
        status_dict["evidence_digest"] = None
        status_dict["verified_at"] = None
        status_dict["expires_at"] = None
        status_dict.pop("verified_claims", None)
        status_dict.pop("verified_claims_digest", None)
        status_dict.pop("proof_claims", None)

    for list_key in ("model_ids", "model_id_prefixes"):
        strict_values = strict_public_string_list(status_dict.get(list_key))
        if status_dict.get("verified") is True and strict_values is None:
            downgrade_verified_status()
        status_dict[list_key] = public_string_list(status_dict.get(list_key))

    force_unverified = (
        status_dict.get("verified") is True and currently_verified is not True
    )
    if force_unverified:
        downgrade_verified_status()
    if status_dict.get("verified") is not True:
        downgrade_verified_status()
    if status_dict.get("verified") is True and not public_provider_type_satisfies_mode(
        provider_type,
        status_dict.get("mode"),
    ):
        downgrade_verified_status()

    failure_reason = status_dict.pop("failure_reason", None)
    if isinstance(failure_reason, str) and failure_reason.strip():
        status_dict["failure_reason_digest"] = _sha256_json_digest(failure_reason)
    verified_claims = status_dict.pop("verified_claims", {})
    if status_dict.get("verified") is True:
        if not isinstance(verified_claims, dict) or not verified_claims:
            downgrade_verified_status()
            return status_dict
        try:
            verified_claims_digest = _sha256_json_digest(verified_claims)
        except (TypeError, ValueError):
            downgrade_verified_status()
        else:
            proof_claims = verified_public_provider_proof_claims(
                status_dict.get("mode"),
                verified_claims,
                public_policy=public_policy,
            )
            if proof_claims is None:
                downgrade_verified_status()
                return status_dict
            if proof_claims.get("payload_policy_digest") != status_dict.get(
                "policy_digest"
            ):
                downgrade_verified_status()
                return status_dict
            if not provider_evidence_digest_matches_status(
                status_dict.get("evidence_digest"),
                verified_claims,
                proof_claims,
            ):
                downgrade_verified_status()
                return status_dict
            if not public_provider_proof_claims_cover_model_selectors(
                provider_type,
                status_dict.get("mode"),
                proof_claims,
                status_dict.get("model_ids"),
                public_policy=public_policy,
            ):
                downgrade_verified_status()
                return status_dict
            if not public_confidentiality_policy_binds_provider_proof(
                provider_type,
                status_dict.get("mode"),
                public_policy,
                proof_claims,
                status_dict.get("model_ids"),
            ):
                downgrade_verified_status()
                return status_dict
            status_dict["verified_claims_digest"] = verified_claims_digest
            status_dict["proof_claims"] = proof_claims
            if include_verified_claims:
                status_dict["_verified_claims_for_validation"] = verified_claims
    return status_dict


def _public_confidentiality_policy(policy: object) -> dict[str, Any] | None:
    """Return non-secret policy pins suitable for public attestation statements."""
    raw_policy = getattr(policy, "policy", None)
    if not isinstance(raw_policy, dict) or not raw_policy:
        return None
    if inline_policy_secret_violations(raw_policy):
        return None

    public: dict[str, Any] = {}
    attestation_bundle_url = raw_policy.get("attestation_bundle_url")
    if isinstance(attestation_bundle_url, str) and attestation_bundle_url.strip():
        public["attestation_bundle_url_digest"] = _sha256_json_digest(
            attestation_bundle_url.strip()
        )

    invalid_policy_value = object()

    def public_policy_value(value: object, *, allow_bool: bool) -> object:
        if isinstance(value, str):
            if not value.strip():
                return invalid_policy_value
            return value.strip()
        if allow_bool and isinstance(value, bool):
            return value
        if isinstance(value, list) and all(
            isinstance(item, str) and item.strip() for item in value
        ):
            return [item.strip() for item in value]
        return invalid_policy_value

    def public_digest_policy_value(value: object) -> object:
        if isinstance(value, str) and is_full_sha256_digest(value):
            return value
        return invalid_policy_value

    def public_digest_list_policy_value(value: object) -> object:
        if isinstance(value, str) and is_full_sha256_digest(value):
            return value
        if isinstance(value, list) and all(
            isinstance(item, str) and is_full_sha256_digest(item) for item in value
        ):
            return list(value)
        return invalid_policy_value

    for key in sorted(PUBLIC_CONFIDENTIALITY_POLICY_KEYS):
        value = raw_policy.get(key)
        if value is None:
            continue
        if key in {
            "model_attestation_targets",
            "modelAttestationTargets",
            "model_enclave_bindings",
            "modelEnclaveBindings",
        }:
            if not isinstance(value, dict):
                return None
            public_targets: dict[str, dict[str, Any]] = {}
            seen_model_ids: set[str] = set()
            for raw_model_id, raw_target in value.items():
                if not isinstance(raw_model_id, str) or not raw_model_id.strip():
                    return None
                model_id = raw_model_id.strip()
                normalized_model_id = model_id.lower()
                if normalized_model_id in seen_model_ids:
                    return None
                seen_model_ids.add(normalized_model_id)
                if not isinstance(raw_target, dict):
                    return None
                public_target = {}
                for target_key in sorted(PUBLIC_TINFOIL_TARGET_POLICY_KEYS):
                    if target_key not in raw_target:
                        continue
                    if target_key in PUBLIC_DIGEST_TINFOIL_TARGET_POLICY_KEYS:
                        public_value = public_digest_policy_value(
                            raw_target[target_key]
                        )
                    elif target_key in PUBLIC_DIGEST_LIST_TINFOIL_TARGET_POLICY_KEYS:
                        public_value = public_digest_list_policy_value(
                            raw_target[target_key]
                        )
                    else:
                        public_value = public_policy_value(
                            raw_target[target_key],
                            allow_bool=False,
                        )
                    if public_value is invalid_policy_value:
                        return None
                    public_target[target_key] = public_value
                if not public_target:
                    return None
                public_targets[model_id] = public_target
            public[key] = dict(sorted(public_targets.items()))
        elif key in {"model_workload_bindings", "modelWorkloadBindings"}:
            if not isinstance(value, dict):
                return None
            public_bindings: dict[str, dict[str, Any]] = {}
            seen_model_ids: set[str] = set()
            for raw_model_id, raw_binding in value.items():
                if not isinstance(raw_model_id, str) or not raw_model_id.strip():
                    return None
                model_id = raw_model_id.strip()
                normalized_model_id = model_id.lower()
                if normalized_model_id in seen_model_ids:
                    return None
                seen_model_ids.add(normalized_model_id)
                if not isinstance(raw_binding, dict):
                    return None
                public_binding = {}
                for binding_key in sorted(
                    PUBLIC_PRIVATEMODE_WORKLOAD_BINDING_POLICY_KEYS
                ):
                    if binding_key not in raw_binding:
                        continue
                    public_value = public_policy_value(
                        raw_binding[binding_key],
                        allow_bool=False,
                    )
                    if public_value is invalid_policy_value:
                        return None
                    public_binding[binding_key] = public_value
                if not public_binding:
                    return None
                public_bindings[model_id] = public_binding
            public[key] = dict(sorted(public_bindings.items()))
        else:
            if key in PUBLIC_DIGEST_CONFIDENTIALITY_POLICY_KEYS:
                public_value = public_digest_policy_value(value)
            elif key in PUBLIC_DIGEST_LIST_CONFIDENTIALITY_POLICY_KEYS:
                public_value = public_digest_list_policy_value(value)
            else:
                public_value = public_policy_value(
                    value,
                    allow_bool=key in PUBLIC_BOOLEAN_CONFIDENTIALITY_POLICY_KEYS,
                )
            if public_value is invalid_policy_value:
                return None
            public[key] = public_value
    return public or None


def get_confidentiality_status(
    *,
    include_provider_urls: bool = False,
    include_provider_policy: bool = True,
    include_verified_claims: bool = False,
    include_routstr_tee: bool = True,
    routstr_tee_ready_override: bool | None = None,
) -> dict[str, Any]:
    """Return a safe public snapshot of provider confidentiality state."""
    mode = _public_string(getattr(settings, "confidential_routing_mode", None))
    providers = []
    for upstream in _upstreams:
        provider_type = _public_string(getattr(upstream, "provider_type", None))
        if provider_type is None:
            continue
        upstream_name = _public_string(getattr(upstream, "upstream_name", None))
        status = upstream.confidentiality_status()
        public_policy = None
        if include_provider_urls or include_provider_policy:
            policy_getter = getattr(upstream, "confidentiality_policy", None)
            policy = policy_getter() if callable(policy_getter) else None
            public_policy = _public_confidentiality_policy(policy)
        public_confidentiality = _public_confidentiality_status(
            status,
            provider_type=provider_type,
            currently_verified=has_current_confidential_verification(upstream),
            public_policy=public_policy,
            include_verified_claims=include_verified_claims,
        )
        provider_status: dict[str, Any] = {
            "provider_type": provider_type,
            "supported_endpoints": public_supported_endpoints_for_provider(upstream),
            "confidentiality": public_confidentiality,
        }
        if upstream_name is not None:
            provider_status["upstream_name"] = upstream_name
        if include_provider_urls:
            base_url = _public_provider_base_url(
                getattr(upstream, "base_url", None),
                provider_type=provider_type,
                confidentiality_mode=getattr(status, "mode", None),
            )
            if base_url is not None:
                provider_status["base_url"] = _public_string(
                    redact_url_userinfo(base_url)
                )
                provider_status["db_id"] = getattr(upstream, "db_id", None)
        if public_policy:
            provider_status["confidentiality_policy"] = public_policy
        providers.append(provider_status)
    result: dict[str, Any] = {
        "mode": mode or "disabled",
        "required": confidential_routing_required(),
        "providers": providers,
    }
    routstr_tee = None
    if include_routstr_tee:
        from .core.attestation import get_public_routstr_tee_status

        routstr_tee = _safe_public_routstr_tee_status(get_public_routstr_tee_status())
        if routstr_tee is not None:
            result["routstr_tee"] = routstr_tee
    if routstr_tee_ready_override is None:
        routstr_tee_ready = (
            isinstance(routstr_tee, dict) and routstr_tee.get("ready") is True
        )
    else:
        routstr_tee_ready = routstr_tee_ready_override is True
    routable_with_full_attestation = _routable_with_full_attestation(
        routstr_tee_ready=routstr_tee_ready,
    )
    result["routable_with_full_attestation"] = routable_with_full_attestation
    result["end_to_end_ready"] = (
        result["required"] is True
        and any(routable_with_full_attestation.values())
        and routstr_tee_ready
    )
    return result


async def refresh_model_maps() -> None:
    """Refresh global model and provider maps using the cost-based algorithm."""
    from sqlalchemy.orm import selectinload

    global _model_instances, _provider_map, _unique_models
    await refresh_confidentiality_statuses(persist=False)

    attestation_rows: dict[int, dict[tuple[str, str], ProviderModelAttestationRow]]
    async with create_session() as session:
        # Fetch all providers with their models in a single logical operation
        query = select(UpstreamProviderRow).options(
            selectinload(UpstreamProviderRow.models)  # type: ignore
        )
        result = await session.exec(query)
        provider_rows = result.all()
        await _persist_provider_model_attestations(session)
        provider_ids = {
            db_id
            for upstream in _upstreams
            if isinstance((db_id := getattr(upstream, "db_id", None)), int)
        }
        attestation_rows = await _load_provider_model_attestations(
            session,
            provider_ids,
        )

    overrides_by_id: dict[str, tuple[ModelRow, float]] = {}
    disabled_model_ids: set[str] = set()

    for provider in provider_rows:
        if not provider.enabled:
            continue
        for model in provider.models:
            if model.enabled:
                overrides_by_id[model.id] = (model, provider.provider_fee)
            else:
                disabled_model_ids.add(model.id)

    for upstream in _upstreams:
        db_id = getattr(upstream, "db_id", None)
        if isinstance(db_id, int):
            setattr(
                upstream,
                "_db_confidential_attestations",
                attestation_rows.get(db_id, {}),
            )

    _model_instances, _provider_map, _unique_models = create_model_mappings(
        upstreams=_upstreams,
        overrides_by_id=overrides_by_id,
        disabled_model_ids=disabled_model_ids,
        require_confidential=confidential_routing_required(),
    )


async def refresh_model_maps_periodically() -> None:
    """Background task to refresh model maps every minute."""
    import asyncio

    while True:
        try:
            await asyncio.sleep(60)
            await refresh_model_maps()
        except asyncio.CancelledError:
            break
        except Exception as e:
            error = _safe_error_message(e)
            logger.error(
                "Error refreshing model maps",
                extra={"error": error, "error_type": type(e).__name__},
            )


_API_PATH_PREFIXES = (
    "v1/",
    "responses",
    "chat/",
    "completions",
    "models",
    "embeddings",
    "audio/",
    "images/",
    "moderations",
    "providers",
    "tee/",
)


@proxy_router.api_route("/{path:path}", methods=["GET", "POST"], response_model=None)
async def proxy(
    request: Request, path: str, session: AsyncSession = Depends(get_session)
) -> Response | StreamingResponse:
    # GET requests must hit a known API prefix; otherwise return a 404 (HTML
    # for browsers, JSON for API clients). POST requests are always forwarded
    # so that OpenAI-style endpoints work with or without the `v1/` prefix
    # (e.g. `/chat/completions` as well as `/v1/chat/completions`).
    if request.method == "GET" and not path.startswith(_API_PATH_PREFIXES):
        return build_not_found_response(request, path)

    headers = dict(request.headers)

    is_responses_api = path.startswith("v1/responses") or path.startswith("responses")
    request_body = await request.body()
    request_body_dict = await parse_request_body_metadata(request, request_body, path)

    # /tee/* GET requests (e.g. attestation) don't map to models — just
    # forward to verified upstreams without model/cost/auth lookups.
    if request.method == "GET" and path.startswith("tee/"):
        all_upstreams = _upstreams
        if confidential_routing_required():
            all_upstreams = [
                upstream
                for upstream in all_upstreams
                if has_current_confidential_verification(upstream)
            ]
            if not all_upstreams:
                return create_error_response(
                    "confidential_route_unavailable",
                    "No verified confidential provider found for TEE endpoint",
                    503,
                    request=request,
                )
            if message := get_routstr_tee_unavailable_message():
                return create_error_response(
                    "routstr_tee_attestation_unavailable",
                    message,
                    503,
                    request=request,
                )
        last_error_response = None
        for i, upstream in enumerate(all_upstreams):
            try:
                headers = upstream.prepare_headers(dict(request.headers))
                response = await upstream.forward_get_request(request, path, headers)
                if response.status_code in [502, 429] and i < len(all_upstreams) - 1:
                    logger.warning(
                        "Upstream %s returned %s for tee GET %s, trying next",
                        upstream.provider_type,
                        response.status_code,
                        path,
                    )
                    continue
                return response
            except UpstreamError as e:
                error = _safe_error_message(e)
                logger.warning(
                    "Upstream %s failed for tee GET %s: %s",
                    upstream.provider_type,
                    path,
                    error,
                )
                if i == len(all_upstreams) - 1:
                    last_error_response = create_error_response(
                        "upstream_error", error, 502, request=request
                    )
                continue
        return last_error_response or create_error_response(
            "upstream_error", "All upstreams failed", 502, request=request
        )

    if is_responses_api:
        model_id = extract_model_from_responses_request(request_body_dict)
    else:
        model_id = request_body_dict.get("model", "unknown")

    model_obj = get_model_instance(model_id)

    if not model_obj:
        return create_error_response(
            "invalid_model", f"Model '{model_id}' not found", 400, request=request
        )

    upstreams = get_provider_for_model(model_id)
    if not upstreams:
        if confidential_routing_required():
            return create_error_response(
                "confidential_route_unavailable",
                f"No verified confidential provider found for model '{model_id}'",
                503,
                request=request,
            )
        return create_error_response(
            "invalid_model",
            f"No provider found for model '{model_id}'",
            400,
            request=request,
        )

    endpoint_requirement = provider_endpoint_requirement_for_path(path)
    if endpoint_requirement is not None:
        support_attr, endpoint_name = endpoint_requirement
        supported_upstreams = [
            upstream
            for upstream in upstreams
            if provider_supports_required_endpoint(
                upstream,
                support_attr,
                model=model_obj,
            )
        ]
        if not supported_upstreams:
            provider_names = ", ".join(
                sorted({upstream.provider_type for upstream in upstreams})
            )
            provider_suffix = (
                f" by provider(s): {provider_names}" if provider_names else ""
            )
            return create_error_response(
                "unsupported_endpoint",
                (
                    f"Model '{model_id}' is not supported on the {endpoint_name}"
                    f"{provider_suffix}"
                ),
                400,
                request=request,
            )
        upstreams = supported_upstreams
    else:
        endpoint_gated_upstreams = [
            upstream
            for upstream in upstreams
            if provider_requires_known_model_endpoint(upstream)
        ]
        if confidential_routing_required() or (
            endpoint_gated_upstreams
            and len(endpoint_gated_upstreams) == len(upstreams)
        ):
            return create_error_response(
                "unsupported_endpoint",
                f"Model '{model_id}' is not supported on an unrecognized API path",
                400,
                request=request,
            )
        if endpoint_gated_upstreams:
            upstreams = [
                upstream
                for upstream in upstreams
                if not provider_requires_known_model_endpoint(upstream)
            ]

    if confidential_routing_required():
        upstreams = [
            upstream
            for upstream in upstreams
            if is_confidential_provider_for_model(upstream, model_obj, model_id)
            and public_confidentiality_metadata(upstream, model_obj, model_id)
            is not None
        ]
        if not upstreams:
            return create_error_response(
                "confidential_route_unavailable",
                f"No verified confidential provider found for model '{model_id}'",
                503,
                request=request,
            )

    if message := get_routstr_tee_unavailable_message():
        return create_error_response(
            "routstr_tee_attestation_unavailable",
            message,
            503,
            request=request,
        )

    # todo figure out cost calculation since fallback provider is usually not the same price
    # Use first provider for initial checks/cost calculation
    # primary_upstream = upstreams[0]

    _max_cost_for_model = await get_max_cost_for_model(
        model=model_id, session=session, model_obj=model_obj
    )
    max_cost_for_model = await calculate_discounted_max_cost(
        _max_cost_for_model, request_body_dict, model_obj=model_obj
    )
    # Ensure max_cost_for_model is at least the minimum allowed request cost
    max_cost_for_model = max(max_cost_for_model, settings.min_request_msat)

    check_token_balance(headers, request_body_dict, max_cost_for_model)

    if x_cashu := headers.get("x-cashu", None):
        last_error = None
        for i, upstream in enumerate(upstreams):
            try:
                if is_responses_api:
                    return await upstream.handle_x_cashu_responses(
                        request,
                        x_cashu,
                        path,
                        max_cost_for_model,
                        model_obj,
                        route_alias=model_id,
                    )
                else:
                    return await upstream.handle_x_cashu(
                        request,
                        x_cashu,
                        path,
                        max_cost_for_model,
                        model_obj,
                        route_alias=model_id,
                    )
            except UpstreamError as e:
                error = _safe_error_message(e)
                logger.warning(
                    "Upstream %s failed (x-cashu) for model=%s: %s",
                    upstream.provider_type,
                    model_id,
                    error,
                    extra={
                        "provider": upstream.provider_type,
                        "model": model_id,
                        "status_code": e.status_code,
                        "error": error,
                    },
                )
                if i == len(upstreams) - 1:
                    last_error = e
                continue

        return create_error_response(
            "upstream_error",
            _safe_error_message(last_error) if last_error else "All upstreams failed",
            502,
            request=request,
        )

    elif auth := headers.get("authorization", None):
        key = await get_bearer_token_key(
            headers, path, session, auth, max_cost_for_model, model_id
        )

    else:
        if request.method not in ["GET"]:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {"type": "invalid_request_error", "code": "unauthorized"}
                },
            )

        logger.debug("Processing unauthenticated GET request", extra={"path": path})

        last_error_response = None
        for i, upstream in enumerate(upstreams):
            try:
                headers = upstream.prepare_headers(dict(request.headers))
                response = await upstream.forward_get_request(request, path, headers)

                if response.status_code in [502, 429] and i < len(upstreams) - 1:
                    error_message = _extract_redirect_error_message(response, upstream)

                    await upstream.on_upstream_error_redirect(
                        response.status_code, error_message
                    )

                    logger.warning(
                        f"Upstream {upstream.provider_type} returned {response.status_code} (GET), trying next provider",
                        extra={
                            "status_code": response.status_code,
                            "upstream": upstream.provider_type,
                        },
                    )
                    continue
                return response
            except UpstreamError as e:
                error = _safe_error_message(e)
                logger.warning(
                    "Upstream %s failed (GET): %s",
                    upstream.provider_type,
                    error,
                    extra={"provider": upstream.provider_type, "error": error},
                )
                if i == len(upstreams) - 1:
                    last_error_response = create_error_response(
                        "upstream_error", error, 502, request=request
                    )
                continue
        return last_error_response or create_error_response(
            "upstream_error", "All upstreams failed", 502, request=request
        )

    if request_body_dict:
        await pay_for_request(key, max_cost_for_model, session)

    for i, upstream in enumerate(upstreams):
        headers = upstream.prepare_headers(dict(request.headers))

        try:
            try:
                if is_responses_api:
                    response = await upstream.forward_responses_request(
                        request,
                        path,
                        headers,
                        request_body,
                        key,
                        max_cost_for_model,
                        session,
                        model_obj,
                        route_alias=model_id,
                    )
                else:
                    response = await upstream.forward_request(
                        request,
                        path,
                        headers,
                        request_body,
                        key,
                        max_cost_for_model,
                        session,
                        model_obj,
                        route_alias=model_id,
                    )
            except UpstreamError:
                # Let the outer UpstreamError handler manage retry/revert
                raise
            except Exception as e:
                # Unexpected error (not an upstream failure) — revert and propagate
                error = _safe_error_message(e)
                logger.error(
                    "Unexpected error in upstream request, reverting payment",
                    extra={
                        "error": error,
                        "error_type": type(e).__name__,
                        "path": path,
                        "key_hash": key.hashed_key[:8] + "...",
                        "max_cost_for_model": max_cost_for_model,
                    },
                )
                await revert_pay_for_request(key, session, max_cost_for_model)
                raise

            if response.status_code != 200:
                # Check if we should retry (502 Upstream Error or 429 Rate Limit)
                should_retry = response.status_code in [502, 429, 400, 401, 403, 404]
                if should_retry and i < len(upstreams) - 1:
                    error_message = _extract_redirect_error_message(response, upstream)

                    await upstream.on_upstream_error_redirect(
                        response.status_code, error_message
                    )

                    logger.warning(
                        "Upstream %s returned %s for model=%s, trying next provider",
                        upstream.provider_type,
                        response.status_code,
                        model_id,
                        extra={
                            "status_code": response.status_code,
                            "provider": upstream.provider_type,
                            "model": model_id,
                        },
                    )
                    continue

                # 4xx error (user error), or other non-retryable error, or last provider failed
                await revert_pay_for_request(key, session, max_cost_for_model)
                logger.warning(
                    "Upstream request failed, revert payment "
                    "(provider=%s model=%s status=%s path=%s)",
                    upstream.provider_type,
                    model_id,
                    response.status_code,
                    path,
                    extra={
                        "status_code": response.status_code,
                        "path": path,
                        "provider": upstream.provider_type,
                        "model": model_id,
                        "key_hash": key.hashed_key[:8] + "...",
                        "key_balance": key.balance,
                        "max_cost_for_model": max_cost_for_model,
                    },
                )
                return response

            return response

        except UpstreamError as e:
            error = _safe_error_message(e)
            logger.warning(
                "Upstream %s failed for model=%s: %s",
                upstream.provider_type,
                model_id,
                error,
                extra={
                    "provider": upstream.provider_type,
                    "model": model_id,
                    "status_code": e.status_code,
                    "retry": i < len(upstreams) - 1,
                    "error": error,
                },
            )

            # If this was the last provider
            if i == len(upstreams) - 1:
                await revert_pay_for_request(key, session, max_cost_for_model)
                return create_error_response(
                    "upstream_error", error, 502, request=request
                )

            # Otherwise loop continues to next provider
            continue

    # Should not be reached given logic above
    return create_error_response(
        "upstream_error", "All upstreams failed", 502, request=request
    )


async def get_bearer_token_key(
    headers: dict,
    path: str,
    session: AsyncSession,
    auth: str,
    min_cost: int = 0,
    model_id: str = "unknown",
) -> ApiKey:
    """Handle bearer token authentication proxy requests."""
    parts = auth.split()
    bearer_key = parts[1] if len(parts) > 1 and parts[0].lower() == "bearer" else ""
    refund_address = headers.get("Refund-LNURL", None)
    key_expiry_time = headers.get("Key-Expiry-Time", None)

    logger.debug(
        "Processing bearer token",
        extra={
            "path": path,
            "has_refund_address": bool(refund_address),
            "has_expiry_time": bool(key_expiry_time),
            "bearer_key_fingerprint": credential_fingerprint(bearer_key),
            "min_cost": min_cost,
        },
    )

    # Validate key_expiry_time header
    if key_expiry_time:
        try:
            key_expiry_time = int(key_expiry_time)  # type: ignore
            logger.debug(
                "Key expiry time validated",
                extra={"expiry_time": key_expiry_time, "path": path},
            )
        except ValueError:
            logger.error(
                "Invalid Key-Expiry-Time header",
                extra={"key_expiry_time": key_expiry_time, "path": path},
            )
            raise HTTPException(
                status_code=400,
                detail="Invalid Key-Expiry-Time: must be a valid Unix timestamp",
            )
        if not refund_address:
            logger.error(
                "Missing Refund-LNURL header with Key-Expiry-Time",
                extra={"path": path, "expiry_time": key_expiry_time},
            )
            raise HTTPException(
                status_code=400,
                detail="Error: Refund-LNURL header required when using Key-Expiry-Time",
            )
    else:
        key_expiry_time = None

    try:
        key = await validate_bearer_key(
            bearer_key,
            session,
            refund_address,
            key_expiry_time,  # type: ignore
            min_cost=min_cost,
        )
        logger.info(
            "Bearer token validated successfully",
            extra={
                "path": path,
                "key_hash": key.hashed_key[:8] + "...",
                "key_balance": key.balance,
            },
        )
        return key
    except Exception as e:
        error = _safe_error_message(e)
        key_fingerprint = credential_fingerprint(bearer_key)
        logger.error(
            f"Bearer token validation failed: {type(e).__name__}: {error} path={path} model={model_id!r} min_cost={min_cost} key_fingerprint={key_fingerprint!r}",
            extra={
                "error": error,
                "error_type": type(e).__name__,
                "path": path,
                "model_id": model_id,
                "min_cost_msat": min_cost,
                "bearer_key_fingerprint": key_fingerprint,
            },
        )
        raise


def extract_model_from_responses_request(request_body_dict: dict[str, Any]) -> str:
    if isinstance(model := request_body_dict.get("model"), str) and model:
        return model

    if input_data := request_body_dict.get("input"):
        if (
            isinstance(input_data, dict)
            and isinstance(model := input_data.get("model"), str)
            and model
        ):
            return model

    if request_body_dict.get("messages"):
        return "unknown"

    logger.warning(
        "No model found in Responses API request",
        extra={"body_keys": list(request_body_dict.keys())},
    )
    return "unknown"


def _is_multipart_form_data(content_type: str | None) -> bool:
    if not content_type:
        return False
    return content_type.split(";", 1)[0].strip().lower() == "multipart/form-data"


async def parse_request_body_metadata(
    request: Request, request_body: bytes, path: str
) -> dict[str, Any]:
    """Parse request metadata needed before routing without storing file bodies."""
    content_type = request.headers.get("content-type")
    if request_body and _is_multipart_form_data(content_type):
        try:
            form = await request.form()
        except Exception as e:
            redact_body = confidential_routing_required()
            error = _safe_error_message(e)
            logger.error(
                "Invalid multipart form request",
                extra={
                    "error": error,
                    "path": path,
                    "confidential": redact_body,
                },
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "type": "invalid_request_error",
                        "code": "invalid_multipart",
                    }
                },
            ) from e

        request_metadata: dict[str, Any] = {}
        for key in ("model", "max_tokens", "stream", "n"):
            value = form.get(key)
            if isinstance(value, str):
                if key == "max_tokens":
                    request_metadata[key] = _validated_positive_int_string(value, key)
                elif key == "n":
                    request_metadata[key] = _validated_single_completion_count(
                        _validated_positive_int_string(value, key)
                    )
                else:
                    request_metadata[key] = value

        logger.debug(
            "Multipart request metadata parsed",
            extra={
                "path": path,
                "body_keys": list(request_metadata.keys()),
                "model": request_metadata.get("model", "not_specified"),
            },
        )
        return request_metadata

    return parse_request_body_json(request_body, path)


def parse_request_body_json(request_body: bytes, path: str) -> dict[str, Any]:
    request_body_dict = {}
    if request_body:
        try:
            request_body_dict = loads_strict_json(request_body, "request body JSON")
            if not isinstance(request_body_dict, dict):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "type": "invalid_request_error",
                            "code": "invalid_json",
                        }
                    },
                )

            if "model" in request_body_dict and not isinstance(
                request_body_dict["model"], str
            ):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "type": "invalid_request_error",
                            "code": "invalid_model",
                        }
                    },
                )
            input_data = request_body_dict.get("input")
            if (
                isinstance(input_data, dict)
                and "model" in input_data
                and not isinstance(input_data["model"], str)
            ):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "type": "invalid_request_error",
                            "code": "invalid_model",
                        }
                    },
                )

            if (
                (path.startswith("v1/responses") or path.startswith("responses"))
                and isinstance(input_data, dict)
                and isinstance(request_body_dict.get("model"), str)
                and isinstance(input_data.get("model"), str)
                and request_body_dict["model"].strip()
                and input_data["model"].strip()
                and request_body_dict["model"].strip() != input_data["model"].strip()
            ):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "type": "invalid_request_error",
                            "code": "invalid_model",
                        }
                    },
                )

            if "max_tokens" in request_body_dict:
                request_body_dict["max_tokens"] = _validated_positive_int(
                    request_body_dict["max_tokens"],
                    "max_tokens",
                )

            if "n" in request_body_dict:
                request_body_dict["n"] = _validated_single_completion_count(
                    request_body_dict["n"]
                )

            logger.debug(
                "Request body parsed",
                extra={
                    "path": path,
                    "body_keys": list(request_body_dict.keys()),
                    "model": request_body_dict.get("model", "not_specified"),
                },
            )
        except (json.JSONDecodeError, ValueError) as e:
            redact_body = confidential_routing_required()
            error = _safe_error_message(e)
            logger.error(
                "Invalid JSON in request body",
                extra={
                    "error": error,
                    "path": path,
                    "body_preview": None
                    if redact_body
                    else (
                        request_body[:200].decode(errors="ignore")
                        if request_body
                        else "empty"
                    ),
                    "confidential": redact_body,
                },
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {"type": "invalid_request_error", "code": "invalid_json"}
                },
            )

    return request_body_dict
