import asyncio
import hashlib
import json
import math
import random
import time
from typing import Any, Callable

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel as V2BaseModel
from pydantic.v1 import BaseModel, validator
from sqlmodel.ext.asyncio.session import AsyncSession

from ..core.confidentiality_public import (
    is_full_sha256_digest,
    provider_evidence_digest_matches_status,
    public_confidentiality_policy_binds_provider_proof,
    public_model_id_matches_verified_selector,
    public_provider_proof_claims_cover_model_selectors,
    public_provider_type_satisfies_mode,
    public_verified_model_selectors_satisfy_provider,
    verified_public_provider_proof_claims,
)
from ..core.db import ModelRow, UpstreamProviderRow, get_session
from ..core.logging import get_logger
from ..core.policy_secrets import inline_policy_secret_violations
from ..core.settings import settings
from .price import sats_usd_price

logger = get_logger(__name__)

models_router = APIRouter()


PUBLIC_CONFIDENTIALITY_KEYS = {
    "attestation_status",
    "enabled",
    "evidence_digest",
    "expires_at",
    "metadata_leakage",
    "mode",
    "model_id_prefixes",
    "model_ids",
    "policy_digest",
    "provider_type",
    "supported_endpoints",
    "verified",
    "verified_at",
    "verified_claims_digest",
    "verifier",
}

PUBLIC_CONFIDENTIALITY_LIST_KEYS = {
    "metadata_leakage",
    "model_id_prefixes",
    "model_ids",
    "supported_endpoints",
}
REMOTE_MODEL_PUBLIC_PROOF_FIELDS = {
    "attestation_evidence_digest",
    "attestation_provider",
    "attestation_status",
    "confidential",
    "confidentiality",
    "provider_attestation_status",
    "routstr_tee",
}
PUBLIC_MODEL_CONFIDENTIALITY_POLICY_KEYS = {
    "allowed_ai_worker_measurements",
    "allowed_code_measurement_fingerprint",
    "allowed_code_measurement_fingerprints",
    "allowed_code_measurements",
    "allowed_coordinator_measurements",
    "allowed_enclave_measurement_fingerprint",
    "allowed_enclave_measurement_fingerprints",
    "allowed_enclave_measurements",
    "allowed_gpu_attestation_policies",
    "allowed_key_release_bindings",
    "allowed_release_digest",
    "allowed_release_digests",
    "allowed_secret_service_measurements",
    "attestation_bundle_url_digest",
    "code_measurement_fingerprint",
    "enclave_measurement_fingerprint",
    "expectedWorkloadIDs",
    "expectedWorkloadSANs",
    "expected_ai_worker_measurement",
    "expected_code_measurement_fingerprint",
    "expected_coordinator_measurement",
    "expected_enclave_measurement_fingerprint",
    "expected_gpu_attestation_policy",
    "expected_key_release_binding",
    "expected_release_digest",
    "expected_repo",
    "expected_secret_service_measurement",
    "expected_trust_tier",
    "expected_workload_ids",
    "expected_workload_identity_digest",
    "expected_workload_sans",
    "manifestDigest",
    "manifest_digest",
    "manifestLogDir",
    "manifest_log_dir",
    "modelAttestationTargets",
    "modelEnclaveBindings",
    "modelWorkloadBindings",
    "model_attestation_targets",
    "model_enclave_bindings",
    "model_workload_binding_digest",
    "model_workload_bindings",
    "proxyBinaryDigest",
    "proxyImageDigest",
    "proxy_binary_digest",
    "proxy_image_digest",
    "release_digest",
    "repo",
    "requireModelAttestations",
    "require_model_attestations",
    "tinfoil_transport_security",
    "transport_security",
}


def _public_model_confidentiality_policy(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not value:
        return None
    if inline_policy_secret_violations(value):
        return None

    public: dict[str, Any] = {}
    attestation_bundle_url = value.get("attestation_bundle_url")
    if isinstance(attestation_bundle_url, str) and attestation_bundle_url.strip():
        public["attestation_bundle_url_digest"] = _sha256_json_digest(
            attestation_bundle_url.strip()
        )

    for key in PUBLIC_MODEL_CONFIDENTIALITY_POLICY_KEYS:
        if key in value:
            public[key] = value[key]
    try:
        json.dumps(public, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return None
    return public or None


def _json_constant_rejecter(label: str) -> Callable[[str], None]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} must not contain {value}")

    return reject_constant


def _loads_strict_json(data: str, label: str) -> Any:
    return json.loads(data, parse_constant=_json_constant_rejecter(label))


class Architecture(BaseModel):
    modality: str
    input_modalities: list[str]
    output_modalities: list[str]
    tokenizer: str
    instruct_type: str | None


class Pricing(BaseModel):
    prompt: float
    completion: float
    request: float = 0.0
    image: float = 0.0
    web_search: float = 0.0
    internal_reasoning: float = 0.0
    input_cache_read: float = 0.0
    input_cache_write: float = 0.0
    max_prompt_cost: float = 0.0  # in sats not msats
    max_completion_cost: float = 0.0  # in sats not msats
    max_cost: float = 0.0  # in sats not msats

    @validator("*")
    def pricing_values_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("pricing values must be finite")
        if value < 0:
            raise ValueError("pricing values must be non-negative")
        return value


class TopProvider(BaseModel):
    context_length: int | None = None
    max_completion_tokens: int | None = None
    is_moderated: bool | None = None


class Model(BaseModel):
    id: str
    name: str
    created: int
    description: str
    context_length: int
    architecture: Architecture
    pricing: Pricing
    sats_pricing: Pricing | None = None
    per_request_limits: dict | None = None
    top_provider: TopProvider | None = None
    enabled: bool = True
    upstream_provider_id: int | str | None = None
    canonical_slug: str | None = None
    alias_ids: list[str] | None = None
    forwarded_model_id: str | None = None
    supported_endpoints: list[str] | None = None
    confidentiality: dict[str, Any] | None = None
    last_seen_at: int | None = None
    availability_status: str | None = None
    capabilities_json: str | None = None

    def __hash__(self) -> int:
        return hash(self.id)


def remote_model_without_public_proof(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    model = dict(value)
    for key in REMOTE_MODEL_PUBLIC_PROOF_FIELDS:
        model.pop(key, None)
    return model


def _sha256_json_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _public_string_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _strict_public_string_list(value: object) -> list[str] | None:
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


def _public_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _public_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _public_model_confidentiality(
    value: object,
    *,
    model_id: str | None = None,
) -> dict[str, Any] | None:
    """Return confidentiality metadata safe for /v1/models."""
    if not isinstance(value, dict) or not value:
        return None

    public: dict[str, Any] = {}
    for key in ("enabled", "verified"):
        if key in value:
            public[key] = value[key] is True

    if "attestation_status" in value:
        public_status = _public_string(value["attestation_status"])
        expected_status = (
            "verified" if public.get("verified") is True else "unavailable"
        )
        if public_status != expected_status:
            return None
        public["attestation_status"] = public_status

    for key in (
        "mode",
        "provider_type",
        "verifier",
    ):
        if key not in value:
            continue
        public_value: str | int | list[str] | None = _public_string(value[key])
        if public_value is None:
            return None
        public[key] = public_value

    for key in (
        "policy_digest",
        "evidence_digest",
        "verified_claims_digest",
    ):
        if key not in value or value[key] is None:
            continue
        if public.get("verified") is not True:
            return None
        if not is_full_sha256_digest(value[key]):
            return None
        public[key] = value[key]

    for key in ("verified_at", "expires_at"):
        if key not in value or value[key] is None:
            continue
        public_value = _public_int(value[key])
        if public_value is None:
            return None
        public[key] = public_value

    for key in PUBLIC_CONFIDENTIALITY_LIST_KEYS:
        if key in value:
            public_value = (
                _strict_public_string_list(value[key])
                if public.get("verified") is True
                else _public_string_list(value[key])
            )
            if public_value is None:
                return None
            public[key] = public_value

    verified_claims = value.get("_verified_claims")
    if verified_claims is None:
        verified_claims = value.get("verified_claims")
    public_policy = _public_model_confidentiality_policy(
        value.get("_confidentiality_policy", value.get("confidentiality_policy"))
    )
    if public.get("verified") is True:
        if not public_provider_type_satisfies_mode(
            public.get("provider_type"),
            public.get("mode"),
        ):
            return None
        if not public_verified_model_selectors_satisfy_provider(
            public.get("provider_type"),
            public.get("model_ids"),
            public.get("model_id_prefixes"),
        ):
            return None
        if isinstance(model_id, str) and model_id.strip():
            public_model_ids = public.get("model_ids")
            if not isinstance(public_model_ids, list):
                return None
            if not public_model_id_matches_verified_selector(
                model_id,
                public_model_ids,
                provider_type=public.get("provider_type"),
                mode=public.get("mode"),
            ):
                return None
        now = int(time.time())
        if (
            not isinstance(public.get("verified_at"), int)
            or public["verified_at"] > now
        ):
            return None
        if not isinstance(public.get("expires_at"), int) or public["expires_at"] <= now:
            return None
        if not isinstance(verified_claims, dict) or not verified_claims:
            return None
        try:
            public["verified_claims_digest"] = _sha256_json_digest(verified_claims)
        except (TypeError, ValueError):
            return None
        proof_claim_source = value.get("proof_claims")
        reject_secret_source = isinstance(proof_claim_source, dict)
        if not reject_secret_source:
            proof_claim_source = verified_claims
        proof_claims = verified_public_provider_proof_claims(
            public.get("mode"),
            proof_claim_source,
            public_policy=public_policy,
            reject_secret_source=reject_secret_source,
        )
        if proof_claims:
            if proof_claims.get("payload_policy_digest") != public.get("policy_digest"):
                return None
            if not provider_evidence_digest_matches_status(
                public.get("evidence_digest"),
                verified_claims,
                proof_claims,
            ):
                return None
            if not public_provider_proof_claims_cover_model_selectors(
                public.get("provider_type"),
                public.get("mode"),
                proof_claims,
                public.get("model_ids"),
                public_policy=public_policy,
            ):
                return None
            if not public_confidentiality_policy_binds_provider_proof(
                public.get("provider_type"),
                public.get("mode"),
                public_policy,
                proof_claims,
                public.get("model_ids"),
            ):
                return None
            public["proof_claims"] = proof_claims
        else:
            return None

    if public.get("verified") is True and not (
        public.get("mode")
        and public.get("provider_type")
        and public.get("verifier")
        and public.get("policy_digest")
        and public.get("evidence_digest")
        and public.get("verified_claims_digest")
        and isinstance(public.get("verified_at"), int)
        and isinstance(public.get("expires_at"), int)
    ):
        return None
    return public or None


def _has_valid_pricing(model: dict) -> bool:
    """Check if model has valid pricing (not free, no negative values)."""
    pricing = model.get("pricing", {})
    if not pricing:
        return False

    try:
        prompt = float(pricing.get("prompt", 0))
        completion = float(pricing.get("completion", 0))
    except (ValueError, TypeError):
        return False

    if prompt < 0 or completion < 0:
        return False

    if prompt == 0 and completion == 0:
        return False

    return True


async def async_fetch_openrouter_models(source_filter: str | None = None) -> list[dict]:
    """Asynchronously fetch model information from OpenRouter API."""
    base_url = "https://openrouter.ai/api/v1"

    try:
        async with httpx.AsyncClient() as client:
            models_response, embeddings_response = await asyncio.gather(
                client.get(f"{base_url}/models", timeout=30),
                client.get(f"{base_url}/embeddings/models", timeout=30),
                return_exceptions=True,
            )

            def process_models_response(
                response: httpx.Response | BaseException,
            ) -> list[dict]:
                if not isinstance(response, BaseException):
                    response.raise_for_status()
                    data = response.json()
                    return [
                        model
                        for model in data.get("data", [])
                        if ":free" not in model.get("id", "").lower()
                    ]
                return []

            models_data: list[dict] = []
            models_data.extend(process_models_response(models_response))
            models_data.extend(process_models_response(embeddings_response))

            # Apply source filter and exclusions
            filtered_models = []
            for model in models_data:
                model_id = model.get("id", "")

                if source_filter:
                    source_prefix = f"{source_filter}/"
                    if not model_id.startswith(source_prefix):
                        continue

                    model = dict(model)
                    model["id"] = model_id[len(source_prefix) :]
                    model_id = model["id"]

                if "(free)" in model.get("name", ""):
                    continue

                if not _has_valid_pricing(model):
                    continue

                safe_model = remote_model_without_public_proof(model)
                if safe_model is not None:
                    filtered_models.append(safe_model)

            return filtered_models
    except Exception as e:
        logger.error(f"Error (async) fetching models from OpenRouter API: {e}")
        return []


def _row_to_model(
    row: ModelRow, apply_provider_fee: bool = False, provider_fee: float = 1.01
) -> Model:
    architecture = _loads_strict_json(row.architecture, "model architecture JSON")
    pricing = _loads_strict_json(row.pricing, "model pricing JSON")
    per_request_limits = (
        _loads_strict_json(row.per_request_limits, "model per_request_limits JSON")
        if row.per_request_limits
        else None
    )
    top_provider_dict = (
        _loads_strict_json(row.top_provider, "model top_provider JSON")
        if row.top_provider
        else None
    )
    capabilities = (
        _loads_strict_json(row.capabilities_json, "model capabilities_json")
        if row.capabilities_json
        else {}
    )
    supported_endpoints = None
    if isinstance(capabilities, dict) and isinstance(
        capabilities.get("supported_endpoints"), list
    ):
        supported_endpoints = [
            item
            for item in capabilities["supported_endpoints"]
            if isinstance(item, str)
        ]

    if apply_provider_fee and isinstance(pricing, dict):
        pricing = {k: float(v) * provider_fee for k, v in pricing.items()}

    if isinstance(pricing, dict) and float(pricing.get("request", 0.0)) <= 0.0:
        pricing["request"] = max(pricing.get("request", 0.0), 0.0)

    parsed_pricing = Pricing.parse_obj(pricing)
    model = Model(
        id=row.id,
        name=row.name,
        created=row.created,
        description=row.description,
        context_length=row.context_length,
        architecture=Architecture.parse_obj(architecture),
        pricing=parsed_pricing,
        sats_pricing=None,
        per_request_limits=per_request_limits,
        top_provider=TopProvider.parse_obj(top_provider_dict)
        if top_provider_dict
        else None,
        enabled=row.enabled,
        upstream_provider_id=row.upstream_provider_id,
        canonical_slug=getattr(row, "canonical_slug", None),
        alias_ids=_loads_strict_json(row.alias_ids, "model alias_ids JSON")
        if row.alias_ids
        else None,
        forwarded_model_id=getattr(row, "forwarded_model_id", None) or row.id,
        supported_endpoints=supported_endpoints,
        last_seen_at=getattr(row, "last_seen_at", None),
        availability_status=getattr(row, "availability_status", None),
        capabilities_json=getattr(row, "capabilities_json", None),
    )

    if apply_provider_fee:
        (
            parsed_pricing.max_prompt_cost,
            parsed_pricing.max_completion_cost,
            parsed_pricing.max_cost,
        ) = _calculate_usd_max_costs(model)

    try:
        sats_to_usd = sats_usd_price()
        model = _update_model_sats_pricing(model, sats_to_usd)
    except Exception as e:
        logger.warning(f"Could not calculate sats pricing: {e}")

    return model


async def list_models(
    session: AsyncSession,
    upstream_id: int,
    include_disabled: bool = False,
    apply_fees: bool = True,
) -> list[Model]:
    from sqlmodel import select

    from ..core.db import UpstreamProviderRow

    query = select(ModelRow)
    if upstream_id is not None:
        query = query.where(ModelRow.upstream_provider_id == upstream_id)
    if not include_disabled:
        query = query.where(ModelRow.enabled)

    rows = (await session.exec(query)).all()  # type: ignore
    provider_result = await session.exec(select(UpstreamProviderRow))
    providers_by_id = {p.id: p for p in provider_result.all()}
    return [
        _row_to_model(
            r,
            apply_provider_fee=apply_fees,
            provider_fee=providers_by_id[r.upstream_provider_id].provider_fee
            if r.upstream_provider_id in providers_by_id
            else 1.01,
        )
        for r in rows
        if include_disabled
        or (
            r.upstream_provider_id in providers_by_id
            and providers_by_id[r.upstream_provider_id].enabled
        )
    ]


def _calculate_usd_max_costs(model: Model) -> tuple[float, float, float]:
    """Calculate max costs in USD based on model context/token limits.

    Args:
        model: Model object

    Returns:
        Tuple of (max_prompt_cost, max_completion_cost, max_cost) in USD
    """
    min_req_msat = max(1, int(getattr(settings, "min_request_msat", 1)))
    min_req_usd = float(min_req_msat) / 1_000_000.0

    prompt_price = model.pricing.prompt
    completion_price = model.pricing.completion

    if model.top_provider and (
        model.top_provider.context_length or model.top_provider.max_completion_tokens
    ):
        if (cl := model.top_provider.context_length) and (
            mct := model.top_provider.max_completion_tokens
        ):
            if cl <= mct:
                return (
                    cl * prompt_price,
                    cl * completion_price,
                    cl * max(completion_price, prompt_price),
                )
            return (
                cl * prompt_price,
                mct * completion_price,
                (cl - mct) * prompt_price + mct * completion_price,
            )
        elif cl := model.top_provider.context_length:
            return (
                cl * prompt_price,
                cl * completion_price,
                cl * max(completion_price, prompt_price),
            )
        elif mct := model.top_provider.max_completion_tokens:
            return (
                mct * prompt_price,
                mct * completion_price,
                mct * completion_price,
            )
    elif model.context_length:
        return (
            model.context_length * prompt_price,
            model.context_length * completion_price,
            model.context_length * max(completion_price, prompt_price),
        )

    p = prompt_price * 1_000_000
    c = completion_price * 32_000
    r = model.pricing.request * 100_000
    i = model.pricing.image * 100
    w = model.pricing.web_search * 1000
    ir = model.pricing.internal_reasoning * 100
    return (p, c, max(p + c + r + i + w + ir, min_req_usd))


def _update_model_sats_pricing(model: Model, sats_to_usd: float) -> Model:
    """Update a model's sats_pricing based on USD pricing and exchange rate.

    Args:
        model: Model object to update
        sats_to_usd: Current sats to USD exchange rate

    Returns:
        Updated Model object with new sats_pricing
    """
    try:
        min_req_msat = max(1, int(getattr(settings, "min_request_msat", 1)))
        min_req_sats = float(min_req_msat) / 1000.0

        sats = Pricing.parse_obj(
            {k: v / sats_to_usd for k, v in model.pricing.dict().items()}
        )

        if sats.request <= 0.0:
            sats.request = min_req_sats
        if (sats.max_cost or 0.0) < min_req_sats:
            sats.max_cost = min_req_sats

        return Model(
            id=model.id,
            name=model.name,
            created=model.created,
            description=model.description,
            context_length=model.context_length,
            architecture=model.architecture,
            pricing=model.pricing,
            sats_pricing=sats,
            per_request_limits=model.per_request_limits,
            top_provider=model.top_provider,
            enabled=model.enabled,
            upstream_provider_id=model.upstream_provider_id,
            canonical_slug=model.canonical_slug,
            alias_ids=model.alias_ids,
            forwarded_model_id=model.forwarded_model_id,
            supported_endpoints=model.supported_endpoints,
            confidentiality=model.confidentiality,
            last_seen_at=model.last_seen_at,
            availability_status=model.availability_status,
            capabilities_json=model.capabilities_json,
        )
    except Exception as e:
        logger.error(
            "Failed to update sats pricing for model",
            extra={
                "model_id": model.id,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        return model


async def _update_sats_pricing_once() -> None:
    """Update sats pricing once for all provider models (in-memory only)."""
    from ..proxy import get_upstreams, refresh_model_maps

    upstreams = get_upstreams()
    if not upstreams:
        return

    sats_to_usd = sats_usd_price()

    updated_count = 0
    for upstream in upstreams:
        updated_models = [
            _update_model_sats_pricing(m, sats_to_usd)
            for m in upstream.get_cached_models()
        ]
        upstream._models_cache = updated_models
        upstream._models_by_id = {
            m.forwarded_model_id or m.id: m for m in updated_models
        }
        updated_count += len(updated_models)

    if updated_count > 0:
        logger.info(
            f"Updated sats pricing for {updated_count} models",
            extra={"models_updated": updated_count},
        )
        await refresh_model_maps()


async def update_sats_pricing() -> None:
    """Periodically update sats pricing for all provider models and database overrides."""
    try:
        if not settings.enable_pricing_refresh:
            return
    except Exception:
        pass

    try:
        await _update_sats_pricing_once()
    except Exception as e:
        logger.warning(
            "Initial sats pricing update failed (will retry in loop)",
            extra={"error": str(e)},
        )

    while True:
        try:
            interval = getattr(settings, "pricing_refresh_interval_seconds", 120)
            jitter = max(0.0, float(interval) * 0.1)
            await asyncio.sleep(interval + random.uniform(0, jitter))
        except asyncio.CancelledError:
            break

        try:
            try:
                if not settings.enable_pricing_refresh:
                    return
            except Exception:
                pass

            await _update_sats_pricing_once()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error updating sats pricing: {e}")


class ModelTestRequest(V2BaseModel):
    model_id: str
    endpoint_type: str
    request_data: dict


async def _require_model_test_admin(request: Request) -> None:
    from ..core.admin import require_admin_api

    await require_admin_api(request)


@models_router.post("/api/models/test", response_model=None)
async def test_model(
    payload: ModelTestRequest,
    session: AsyncSession = Depends(get_session),
    _: None = Depends(_require_model_test_admin),
) -> Any:
    """Test a model by sending a request through its configured upstream provider."""
    from sqlmodel import select

    from ..proxy import confidential_routing_required

    if confidential_routing_required():
        return JSONResponse(
            status_code=403,
            content={
                "error": {
                    "type": "invalid_request_error",
                    "code": "confidential_route_required",
                    "message": (
                        "Direct upstream model testing is disabled when "
                        "confidential routing is required"
                    ),
                }
            },
        )

    result = await session.execute(
        select(ModelRow).where(ModelRow.id == payload.model_id)
    )
    model_row = result.scalars().first()

    if not model_row:
        return {
            "success": False,
            "error": f"Model '{payload.model_id}' not found in database",
            "status_code": 404,
        }

    provider = await session.get(UpstreamProviderRow, model_row.upstream_provider_id)
    if not provider:
        return {
            "success": False,
            "error": "Upstream provider not found",
            "status_code": 404,
        }

    base_url = provider.base_url.rstrip("/")
    if payload.endpoint_type == "chat-completions":
        url = f"{base_url}/chat/completions"
    else:
        url = f"{base_url}/{payload.endpoint_type}"

    actual_model_id = model_row.forwarded_model_id or model_row.id
    request_data = dict(payload.request_data)
    request_data["model"] = actual_model_id

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {provider.api_key}",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=request_data, headers=headers)
            try:
                response_data = response.json()
            except Exception:
                response_data = {"raw": response.text}

            return {
                "success": response.status_code < 400,
                "data": response_data,
                "status_code": response.status_code,
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "status_code": 500,
        }


@models_router.get("/v1/models")
@models_router.get("/v1/models/", include_in_schema=False)
@models_router.get("/models")
@models_router.get("/models/", include_in_schema=False)
async def models(session: AsyncSession = Depends(get_session)) -> dict:
    """Get verified TEE-routable models with database overrides applied."""
    from ..proxy import (
        _safe_public_routstr_tee_status,
        get_unique_models,
        is_routable_confidential_model,
    )

    items = get_unique_models()
    data = []
    advertises_confidentiality = False
    routstr_tee_status: dict[str, Any] | None = None
    for model in items:
        m = model.dict()
        model_id = (
            model.forwarded_model_id
            if isinstance(model.forwarded_model_id, str)
            and model.forwarded_model_id.strip()
            else model.id
        )
        confidentiality = _public_model_confidentiality(
            m.get("confidentiality"),
            model_id=model_id,
        )
        if not confidentiality:
            continue

        advertises_confidentiality = True
        if routstr_tee_status is None:
            from ..core.attestation import get_public_routstr_tee_status

            routstr_tee_status = _safe_public_routstr_tee_status(
                get_public_routstr_tee_status()
            ) or {"required": True, "ready": False}
        provider_verified = confidentiality.get("verified") is True
        local_tee_ready = (
            routstr_tee_status.get("ready") is True
            if routstr_tee_status.get("required") is True
            else True
        )
        routeable = (
            isinstance(model_id, str)
            and bool(model_id.strip())
            and is_routable_confidential_model(model_id, model)
        )
        verified = provider_verified and local_tee_ready and routeable
        if not verified:
            continue

        m["confidentiality"] = confidentiality
        m["confidential"] = True
        m["attestation_provider"] = confidentiality.get(
            "provider_type"
        ) or confidentiality.get("mode")
        m["provider_attestation_status"] = "verified"
        m["attestation_status"] = "verified"
        m["attestation_evidence_digest"] = confidentiality.get("evidence_digest")
        if model.forwarded_model_id:
            m["id"] = model.forwarded_model_id
        data.append(m)
    response: dict[str, Any] = {"data": data}
    if advertises_confidentiality:
        if routstr_tee_status is None:
            from ..core.attestation import get_public_routstr_tee_status

            routstr_tee_status = _safe_public_routstr_tee_status(
                get_public_routstr_tee_status()
            ) or {"required": True, "ready": False}
        response["routstr_tee"] = routstr_tee_status
    return response
