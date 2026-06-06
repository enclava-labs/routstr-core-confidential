from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import secrets
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, cast
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from ..upstream.ehbp import parse_ehbp_key_config
from .confidentiality_public import (
    public_confidentiality_policy_binds_provider_proof,
    public_provider_proof_claims_cover_model_selectors,
    public_verified_model_selectors_satisfy_provider,
    verified_public_provider_proof_claims,
)
from .policy_secrets import inline_policy_secret_violations
from .settings import settings
from .version import __version__

REQUIRED_ROUTSTR_TEE_VERIFICATION_STEPS = (
    "tee_attestation_report",
    "tee_certificate_chain",
    "measurement_match",
    "runtime_policy_binding",
    "hpke_key_binding",
    "public_key_binding",
    "freshness",
)
CAP_ROUTSTR_TEE_VERIFICATION_STEPS = (
    "cap_status_verified",
    "cap_claims_verified",
    "cap_state_unlocked",
    "tls_terminates_in_attested_tee",
    "freshness",
)
PUBLIC_ROUTSTR_TEE_VERIFICATION_STEPS = (
    *REQUIRED_ROUTSTR_TEE_VERIFICATION_STEPS,
    *CAP_ROUTSTR_TEE_VERIFICATION_STEPS,
)

REQUIRED_ROUTSTR_TEE_PROOF_CLAIMS = (
    "hpke_key_config_digest",
    "tee_attestation_report_digest",
    "tee_certificate_chain_digest",
    "tee_report_data_hex",
    "tee_report_data_digest",
    "tee_report_nonce",
    "tee_report_nonce_digest",
)
PUBLIC_ROUTSTR_TEE_PROOF_CLAIMS = (
    "attested_local_artifacts",
    "attestation_document_format",
    "cap_attestation_url",
    "cap_claims_instance_id",
    "cap_claims_verified",
    "cap_instance_id",
    "cap_mode",
    "cap_state",
    "cap_status_digest",
    "cap_status_url",
    "cap_tee_domain",
    "cap_tenant_id",
    "client_confidentiality_boundary",
    "hpke_key_config_digest",
    "hpke_public_key_digest",
    "public_key_digest",
    "routstr_code_measurement",
    "routstr_config_measurement",
    "tee_attestation_report_digest",
    "tee_certificate_chain_digest",
    "tee_report_data_digest",
    "tee_report_data_hex",
    "tee_report_nonce",
    "tee_report_nonce_digest",
    "tls_terminates_in_attested_tee",
    "verification_steps",
)
DEFAULT_ROUTSTR_TEE_VERIFIER_MAX_AGE_SECONDS = 300
CONFIDENTIAL_PROVIDER_MODES = {
    "tinfoil": "tinfoil",
    "ppq-private": "ppq-private-tee",
    "privatemode": "privatemode",
}
ROUTABLE_FULL_ATTESTATION_PROVIDERS = (
    "tinfoil",
    "ppq-private",
    "privatemode",
)
CLIENT_CONFIDENTIALITY_BOUNDARY_ATTESTED_TLS = "attested-tls-termination"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_constant_rejecter(label: str) -> Callable[[str], None]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} must not contain {value}")

    return reject_constant


def _loads_strict_json(data: str | bytes, label: str) -> Any:
    return json.loads(data, parse_constant=_json_constant_rejecter(label))


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json(value))


def _is_prefixed_sha256_digest_value(value: object) -> bool:
    if not isinstance(value, str):
        return False
    digest = value.strip().lower()
    if not digest.startswith("sha256:"):
        return False
    digest = digest.removeprefix("sha256:")
    return (
        len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
        and len(set(digest)) > 1
    )


def _require_prefixed_sha256_digest_value(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a sha256 digest")
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    if len(digest) == 64 and len(set(digest)) == 1:
        raise ValueError(f"{label} must not be a placeholder digest")
    if not _is_prefixed_sha256_digest_value(value):
        raise ValueError(f"{label} must be a sha256 digest")
    return value.strip()


def _reject_placeholder_policy_digests(value: object, path: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_path = str(key) if not path else f"{path}.{key}"
            _reject_placeholder_policy_digests(item, key_path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_placeholder_policy_digests(item, f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    if len(digest) == 64 and len(set(digest)) == 1:
        label = path or "policy"
        raise ValueError(f"{label} must not be a placeholder digest")


def _public_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _public_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _public_prefixed_sha256_digest(value: object) -> str | None:
    if isinstance(value, str) and _is_prefixed_sha256_digest_value(value):
        return value.strip()
    return None


def _is_hex_string(value: object, *, length: int | None = None) -> bool:
    if not isinstance(value, str):
        return False
    candidate = value.strip().lower()
    if length is not None and len(candidate) != length:
        return False
    return bool(candidate) and all(char in "0123456789abcdef" for char in candidate)


def _stderr_failure_detail(stderr: bytes) -> str:
    if not stderr:
        return ""
    return f": stderr redacted ({len(stderr)} bytes, {_sha256_bytes(stderr)})"


def _new_verification_nonce() -> str:
    return secrets.token_urlsafe(32)


def _safe_provider_confidentiality(
    status: dict[str, Any],
    *,
    provider_type: str | None,
    provider_base_url: str | None = None,
    public_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mode = _public_string(status.get("mode")) or "none"
    mode_matches_provider = CONFIDENTIAL_PROVIDER_MODES.get(provider_type or "") == mode
    policy_digest = _public_prefixed_sha256_digest(status.get("policy_digest"))
    evidence_digest = _public_prefixed_sha256_digest(status.get("evidence_digest"))
    verified_claims_digest = _public_prefixed_sha256_digest(
        status.get("verified_claims_digest")
    )
    verifier = _public_string(status.get("verifier"))
    verified_at = _public_int(status.get("verified_at"))
    expires_at = _public_int(status.get("expires_at"))
    proof_claim_source = status.get("proof_claims")
    reject_secret_source = isinstance(proof_claim_source, dict)
    if not reject_secret_source:
        proof_claim_source = status.get("verified_claims")
    proof_claims = verified_public_provider_proof_claims(
        mode,
        proof_claim_source,
        public_policy=public_policy,
        reject_secret_source=reject_secret_source,
    )
    model_ids = _public_string_list(status.get("model_ids"))
    model_id_prefixes = _public_string_list(status.get("model_id_prefixes"))
    strict_model_ids = _strict_public_string_list(status.get("model_ids"))
    strict_model_id_prefixes = _strict_public_string_list(
        status.get("model_id_prefixes")
    )
    public_status: dict[str, Any] = {
        "enabled": status.get("enabled", False) is True,
        "verified": False,
        "mode": mode,
        "verifier": None,
        "evidence_digest": None,
        "verified_at": None,
        "expires_at": None,
        "model_ids": model_ids,
        "model_id_prefixes": model_id_prefixes,
    }
    now = int(time.time())
    if (
        status.get("verified", False) is True
        and mode != "none"
        and mode_matches_provider
        and verifier is not None
        and policy_digest is not None
        and evidence_digest is not None
        and verified_claims_digest is not None
        and proof_claims is not None
        and proof_claims.get("payload_policy_digest") == policy_digest
        and proof_claims.get("payload_evidence_digest") == evidence_digest
        and verified_at is not None
        and expires_at is not None
        and verified_at <= now
        and expires_at > now
        and strict_model_ids is not None
        and strict_model_id_prefixes is not None
        and public_verified_model_selectors_satisfy_provider(
            provider_type,
            strict_model_ids,
            strict_model_id_prefixes,
        )
        and public_provider_proof_claims_cover_model_selectors(
            provider_type,
            mode,
            proof_claims,
            strict_model_ids,
            public_policy=public_policy,
        )
        and public_confidentiality_policy_binds_provider_proof(
            provider_type,
            mode,
            public_policy,
            proof_claims,
            strict_model_ids,
        )
    ):
        public_status.update(
            {
                "verified": True,
                "verifier": verifier,
                "policy_digest": policy_digest,
                "evidence_digest": evidence_digest,
                "verified_claims_digest": verified_claims_digest,
                "proof_claims": proof_claims,
                "verified_at": verified_at,
                "expires_at": expires_at,
            }
        )
    return public_status


def _empty_routable_with_full_attestation() -> dict[str, list[str]]:
    return {provider_type: [] for provider_type in ROUTABLE_FULL_ATTESTATION_PROVIDERS}


def _safe_routable_with_full_attestation(value: object) -> dict[str, list[str]]:
    routable = _empty_routable_with_full_attestation()
    if not isinstance(value, dict):
        return routable
    for provider_type in ROUTABLE_FULL_ATTESTATION_PROVIDERS:
        model_ids = _strict_public_string_list(value.get(provider_type, []))
        if model_ids is None:
            return _empty_routable_with_full_attestation()
        routable[provider_type] = sorted(model_ids)
    return routable


def _public_string_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    values = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if len({item.lower() for item in values}) != len(values):
        return []
    return values


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
    if len({item.lower() for item in values}) != len(values):
        return None
    return values


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


def _normalized_base_url(value: str) -> str:
    return value.strip().rstrip("/")


def _public_provider_base_url(
    value: object,
    *,
    provider_type: str | None = None,
    confidentiality_mode: str | None = None,
) -> str | None:
    if not isinstance(value, str):
        return None
    base_url = value.strip()
    if not base_url:
        return None

    try:
        parsed = urlsplit(base_url)
    except ValueError:
        return None
    if not parsed.scheme or not parsed.netloc:
        return None
    if not parsed.hostname:
        return None
    try:
        parsed.port
    except ValueError:
        return None
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    for query_key, query_value in query_pairs:
        if inline_policy_secret_violations(
            {query_key: query_value},
            policy_name="provider base_url query",
        ):
            return None
    is_privatemode_loopback = (
        parsed.scheme in {"http", "https"}
        and _is_loopback_hostname(parsed.hostname)
        and (provider_type == "privatemode" or confidentiality_mode == "privatemode")
    )
    if parsed.scheme != "https" and not is_privatemode_loopback:
        return None
    if "@" in parsed.netloc:
        netloc = parsed.netloc.rsplit("@", 1)[-1]
        base_url = urlunsplit(
            (
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )
    if (
        provider_type == "tinfoil"
        or confidentiality_mode == "tinfoil"
    ) and _normalized_base_url(base_url) != "https://inference.tinfoil.sh/v1":
        return None
    if (
        provider_type == "privatemode"
        or confidentiality_mode == "privatemode"
    ) and not is_privatemode_loopback:
        return None
    return base_url


def _routing_policy_snapshot(
    *,
    routstr_tee_ready: bool | None = None,
) -> dict[str, Any]:
    from .. import proxy as proxy_module

    status_kwargs: dict[str, Any] = {
        "include_provider_urls": True,
        "include_routstr_tee": False,
    }
    if routstr_tee_ready is True:
        status_kwargs["include_verified_claims"] = True
        status_kwargs["routstr_tee_ready_override"] = True
    status = proxy_module.get_confidentiality_status(**status_kwargs)
    routable_with_full_attestation = _safe_routable_with_full_attestation(
        status.get("routable_with_full_attestation")
    )
    providers = []
    raw_providers = status.get("providers", [])
    if not isinstance(raw_providers, list):
        raw_providers = []
    for provider in raw_providers:
        if not isinstance(provider, dict):
            continue
        provider_type = _public_string(provider.get("provider_type"))
        if provider_type is None:
            continue
        upstream_name = _public_string(provider.get("upstream_name"))
        confidentiality_policy = provider.get("confidentiality_policy")
        if not isinstance(confidentiality_policy, dict):
            confidentiality_policy = None
        confidentiality = provider.get("confidentiality") or {}
        if not isinstance(confidentiality, dict):
            confidentiality = {}
        confidentiality_mode = _public_string(confidentiality.get("mode"))
        db_id = provider.get("db_id")
        provider_base_url = _public_provider_base_url(
            provider.get("base_url"),
            provider_type=provider_type,
            confidentiality_mode=confidentiality_mode,
        )
        provider_snapshot = {
            "provider_type": provider_type,
            "upstream_name": upstream_name,
            "base_url": provider_base_url,
            "db_id": db_id if isinstance(db_id, int) else None,
            "confidentiality": _safe_provider_confidentiality(
                confidentiality,
                provider_type=provider_type,
                provider_base_url=provider_base_url,
                public_policy=confidentiality_policy,
            ),
        }
        if confidentiality_policy is not None:
            provider_snapshot["confidentiality_policy"] = confidentiality_policy
        providers.append(provider_snapshot)

    return {
        "mode": _public_string(status.get("mode")) or "disabled",
        "required": status.get("required", False) is True,
        "client_confidentiality": _client_confidentiality_snapshot(),
        "routable_with_full_attestation": routable_with_full_attestation,
        "providers": providers,
    }


def _client_confidentiality_snapshot() -> dict[str, Any]:
    boundary = _public_string(
        getattr(settings, "routstr_tee_client_confidentiality_boundary", "")
    )
    if boundary is not None:
        boundary = boundary.lower()
    if boundary == CLIENT_CONFIDENTIALITY_BOUNDARY_ATTESTED_TLS:
        public_key, public_key_error = _read_public_key()
        attested_tls_public_key_digest = (
            _sha256_bytes(public_key.encode("utf-8"))
            if public_key is not None and public_key_error is None
            else None
        )
        return {
            "mode": CLIENT_CONFIDENTIALITY_BOUNDARY_ATTESTED_TLS,
            "tls_terminates_in_attested_tee": True,
            "inbound_ehbp_ohttp_request_decryption": False,
            "attested_tls_public_key_digest": attested_tls_public_key_digest,
        }
    return {
        "mode": boundary or "unspecified",
        "tls_terminates_in_attested_tee": False,
        "inbound_ehbp_ohttp_request_decryption": False,
    }


def _configured_attested_tls_public_key_digest_failure(
    public_key_digest: str | None,
) -> str | None:
    configured_digest = str(
        getattr(settings, "routstr_tee_attested_tls_public_key_digest", "") or ""
    ).strip()
    if not configured_digest:
        return None
    try:
        expected_digest = _require_prefixed_sha256_digest_value(
            configured_digest,
            "ROUTSTR_TEE_ATTESTED_TLS_PUBLIC_KEY_DIGEST",
        )
    except ValueError as exc:
        return str(exc)
    if public_key_digest is None:
        return (
            "ROUTSTR_TEE_PUBLIC_KEY or ROUTSTR_TEE_PUBLIC_KEY_PATH is required "
            "to match ROUTSTR_TEE_ATTESTED_TLS_PUBLIC_KEY_DIGEST"
        )
    if not hmac.compare_digest(expected_digest.lower(), public_key_digest.lower()):
        return (
            "ROUTSTR_TEE_ATTESTED_TLS_PUBLIC_KEY_DIGEST does not match "
            "ROUTSTR_TEE_PUBLIC_KEY"
        )
    return None


def _timeout_seconds() -> float:
    raw_timeout = getattr(settings, "routstr_tee_verifier_timeout_seconds", 10.0)
    try:
        timeout = float(raw_timeout)
    except (TypeError, ValueError):
        return 10.0
    return max(0.1, min(timeout, 60.0))


def _cap_status_timeout_seconds() -> float:
    raw_timeout = getattr(settings, "routstr_tee_cap_status_timeout_seconds", 2.0)
    try:
        timeout = float(raw_timeout)
    except (TypeError, ValueError):
        return 2.0
    return max(0.1, min(timeout, 10.0))


def _cap_attestation_enabled() -> bool:
    return bool(getattr(settings, "routstr_tee_cap_attestation_enabled", False))


def _validated_cap_status_url() -> str:
    raw_url = str(
        getattr(settings, "routstr_tee_cap_status_url", "")
        or "http://127.0.0.1:8081/status"
    ).strip()
    parsed = urlsplit(raw_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("ROUTSTR_TEE_CAP_STATUS_URL must use http or https")
    host = parsed.hostname
    if not host:
        raise ValueError("ROUTSTR_TEE_CAP_STATUS_URL must include a host")
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError
        except ValueError as exc:
            raise ValueError(
                "ROUTSTR_TEE_CAP_STATUS_URL must point at loopback; use the "
                "CAP attestation proxy inside the same TEE pod"
            ) from exc
    return raw_url


def _public_cap_base_url() -> str | None:
    raw_url = str(getattr(settings, "routstr_tee_cap_public_base_url", "") or "").strip()
    if raw_url:
        parsed = urlsplit(raw_url)
        if parsed.scheme != "https" or not parsed.hostname:
            return None
        return raw_url.rstrip("/")

    tee_domain = str(getattr(settings, "routstr_tee_cap_tee_domain", "") or "").strip()
    if not tee_domain:
        return None
    return f"https://{tee_domain}/.well-known/confidential"


def _cap_tee_domain() -> str | None:
    tee_domain = str(getattr(settings, "routstr_tee_cap_tee_domain", "") or "").strip()
    if tee_domain:
        return tee_domain
    base_url = _public_cap_base_url()
    if not base_url:
        return None
    host = urlsplit(base_url).hostname
    return host or None


def _fetch_cap_attestation_status() -> dict[str, Any]:
    url = _validated_cap_status_url()
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=_cap_status_timeout_seconds()) as response:
            payload = response.read(64 * 1024)
    except HTTPError as exc:
        raise ValueError(f"CAP attestation status returned HTTP {exc.code}") from exc
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise ValueError(f"CAP attestation status request failed: {reason}") from exc
    except Exception as exc:
        raise ValueError(
            f"CAP attestation status request failed: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        status = _loads_strict_json(payload, "CAP attestation status JSON")
    except Exception as exc:
        raise ValueError(f"CAP attestation status is not valid JSON: {exc}") from exc
    if not isinstance(status, dict):
        raise ValueError("CAP attestation status must be a JSON object")
    return status


def _safe_cap_attestation_status(status: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in (
        "auto_unlock_enabled",
        "ciphertext_backend",
        "claims_error",
        "claims_instance_id",
        "claims_verified",
        "config_ready",
        "error",
        "instance_id",
        "mode",
        "state",
        "tenant_id",
        "tenant_instance_identity_hash",
    ):
        value = status.get(key)
        if isinstance(value, (str, bool, int)) or value is None:
            safe[key] = value
    return safe


def _cap_status_failure(status: dict[str, Any]) -> str | None:
    if status.get("claims_verified") is not True:
        return "CAP attestation proxy claims are not verified"
    if status.get("state") != "unlocked":
        return "CAP attestation proxy state is not unlocked"
    if status.get("error") not in (None, ""):
        return "CAP attestation proxy reports an error"
    if status.get("claims_error") not in (None, ""):
        return "CAP attestation proxy reports a claims error"
    if not _public_string(status.get("tenant_id")):
        return "CAP attestation proxy tenant_id is missing"
    if not _public_string(status.get("claims_instance_id")):
        return "CAP attestation proxy claims_instance_id is missing"
    return None


def _cap_attestation_status_evidence() -> dict[str, Any]:
    base_url = _public_cap_base_url()
    tee_domain = _cap_tee_domain()
    status_url = f"{base_url}/status" if base_url else None
    attestation_url = f"{base_url}/attestation" if base_url else None
    base: dict[str, Any] = {
        "available": False,
        "status_digest": None,
        "status_url": status_url,
        "attestation_url": attestation_url,
        "tee_domain": tee_domain,
        "failure_reason": None,
    }

    if not base_url:
        base["failure_reason"] = (
            "ROUTSTR_TEE_CAP_PUBLIC_BASE_URL must be HTTPS or "
            "ROUTSTR_TEE_CAP_TEE_DOMAIN is required"
        )
        return base
    if not tee_domain:
        base["failure_reason"] = "ROUTSTR_TEE_CAP_TEE_DOMAIN is required"
        return base

    try:
        status = _fetch_cap_attestation_status()
    except Exception as exc:
        base["failure_reason"] = str(exc)
        return base

    safe_status = _safe_cap_attestation_status(status)
    base["status"] = safe_status
    base["status_digest"] = _sha256_json(safe_status)
    if failure := _cap_status_failure(safe_status):
        base["failure_reason"] = failure
        return base

    base["available"] = True
    return base


def _verify_cap_routstr_tee_evidence(
    *,
    tee: dict[str, Any],
    routing_policy: dict[str, Any],
) -> dict[str, Any]:
    cap = tee.get("cap_attestation")
    if not isinstance(cap, dict):
        return _unverified_routstr_tee_verification(
            "CAP attestation proxy status is unavailable"
        )
    if cap.get("available") is not True:
        return _unverified_routstr_tee_verification(
            str(cap.get("failure_reason") or "CAP attestation proxy is not verified")
        )
    status = cap.get("status")
    if not isinstance(status, dict):
        return _unverified_routstr_tee_verification(
            "CAP attestation proxy status is unavailable"
        )
    status_digest = cap.get("status_digest")
    if not _is_prefixed_sha256_digest_value(status_digest):
        return _unverified_routstr_tee_verification(
            "CAP attestation proxy status digest must be a sha256 digest"
        )
    tee_domain = _public_string(cap.get("tee_domain"))
    if tee_domain is None:
        return _unverified_routstr_tee_verification(
            "ROUTSTR_TEE_CAP_TEE_DOMAIN is required"
        )
    for claim, label in (
        ("status_url", "CAP attestation status URL"),
        ("attestation_url", "CAP attestation URL"),
    ):
        claim_url = _public_string(cap.get(claim))
        if claim_url is None or urlsplit(claim_url).scheme != "https":
            return _unverified_routstr_tee_verification(f"{label} must be HTTPS")
    boundary = _public_string(
        getattr(settings, "routstr_tee_client_confidentiality_boundary", "")
    )
    if boundary is not None:
        boundary = boundary.lower()
    if boundary != CLIENT_CONFIDENTIALITY_BOUNDARY_ATTESTED_TLS:
        return _unverified_routstr_tee_verification(
            "CAP attestation requires attested TLS termination as the "
            "client confidentiality boundary"
        )

    now = int(time.time())
    max_age = getattr(settings, "routstr_tee_cap_verifier_max_age_seconds", 300)
    try:
        max_age = int(max_age)
    except (TypeError, ValueError):
        max_age = 300
    max_age = max(1, min(max_age, 3600))
    routing_policy_digest = _sha256_json(routing_policy)
    verified_claims = {
        "attestation_document_format": "cap-attestation-proxy-status",
        "cap_attestation_url": cap.get("attestation_url"),
        "cap_claims_instance_id": status.get("claims_instance_id"),
        "cap_claims_verified": status.get("claims_verified") is True,
        "cap_instance_id": status.get("instance_id"),
        "cap_mode": status.get("mode"),
        "cap_state": status.get("state"),
        "cap_status_digest": status_digest,
        "cap_status_url": cap.get("status_url"),
        "cap_tee_domain": tee_domain,
        "cap_tenant_id": status.get("tenant_id"),
        "client_confidentiality_boundary": boundary,
        "routstr_config_measurement": routing_policy_digest,
        "tls_terminates_in_attested_tee": True,
        "verification_steps": {
            "cap_status_verified": True,
            "cap_claims_verified": True,
            "cap_state_unlocked": True,
            "tls_terminates_in_attested_tee": True,
            "freshness": True,
        },
    }
    return {
        "verified": True,
        "verifier": "cap-attestation-proxy",
        "verified_at": now,
        "expires_at": now + max_age,
        "failure_reason": None,
        "evidence_digest": cast(str, status_digest),
        "verified_claims": verified_claims,
    }


def _routstr_tee_required_for_public_status() -> bool:
    mode = str(getattr(settings, "confidential_routing_mode", "disabled") or "")
    return mode.strip().lower() in {
        "required",
        "require",
        "strict",
        "enforced",
    } and bool(getattr(settings, "routstr_tee_attestation_required", False))


def _command_argv(
    command: str,
    *,
    label: str = "Routstr TEE verifier command",
) -> list[str]:
    command = command.strip()
    if not command:
        raise ValueError(f"{label} is not configured")
    if command.startswith("["):
        raw_argv = _loads_strict_json(command, f"{label} JSON")
        if not isinstance(raw_argv, list):
            raise ValueError(f"{label} JSON must be an array")
        if any(not isinstance(item, str) or not item.strip() for item in raw_argv):
            raise ValueError(f"{label} JSON entries must be non-empty strings")
        argv = [item.strip() for item in raw_argv]
    else:
        argv = shlex.split(command)
    if not argv:
        raise ValueError(f"{label} is empty")
    violations = inline_policy_secret_violations({"argv": argv}, policy_name=label)
    if violations:
        raise ValueError("; ".join(violations))
    return argv


def _require_artifact_path_bound_to_command(
    *,
    argv: list[str],
    artifact_path: str,
    setting_name: str,
    command_label: str,
) -> None:
    if artifact_path not in argv:
        raise ValueError(
            f"{setting_name} must match {command_label} executable or argument"
        )


def _routstr_tee_verifier_policy() -> dict[str, Any]:
    raw_policy = str(getattr(settings, "routstr_tee_verifier_policy_json", "") or "")
    if not raw_policy.strip():
        return {}
    try:
        policy = _loads_strict_json(raw_policy, "Routstr TEE verifier policy JSON")
    except Exception as exc:
        raise ValueError(f"Routstr TEE verifier policy JSON is invalid: {exc}") from exc
    if not isinstance(policy, dict):
        raise ValueError("Routstr TEE verifier policy JSON must be an object")
    violations = inline_policy_secret_violations(
        policy,
        policy_name="Routstr TEE verifier policy",
    )
    if violations:
        raise ValueError("; ".join(violations))
    _reject_placeholder_policy_digests(policy)
    return policy


def _routstr_code_measurement_policy_values(policy: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "expected_routstr_code_measurement",
        "allowed_routstr_code_measurements",
    ):
        if key not in policy:
            continue
        value = policy[key]
        if isinstance(value, str):
            candidate = value.strip()
            if not _is_prefixed_sha256_digest_value(candidate):
                raise ValueError(
                    "routstr_code_measurement policy value must be a sha256 digest"
                )
            values.append(candidate)
            continue
        if isinstance(value, list):
            if not value:
                raise ValueError(
                    "routstr_code_measurement policy value must be a sha256 digest"
                )
            for item in value:
                if not isinstance(item, str) or not _is_prefixed_sha256_digest_value(
                    item
                ):
                    raise ValueError(
                        "routstr_code_measurement policy value must be a sha256 digest"
                    )
                values.append(item.strip())
            continue
        raise ValueError(
            "routstr_code_measurement policy value must be a sha256 digest"
        )
    return values


def _int_result_value(value: object, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"{label} must be an integer")


def _int_policy_value(policy: dict[str, Any], *keys: str) -> int | None:
    selected_key: str | None = None
    selected_value: int | None = None
    for key in keys:
        if key not in policy:
            continue
        value = policy[key]
        if isinstance(value, int) and not isinstance(value, bool):
            if selected_value is None:
                selected_key = key
                selected_value = value
                continue
            if value != selected_value:
                raise ValueError(f"{selected_key} and {key} aliases must match")
            continue
        raise ValueError(f"{key} must be an integer")
    return selected_value


def _bounded_routstr_tee_expires_at(
    *,
    verifier_result: dict[str, Any],
    verified_at: int,
    verifier_policy: dict[str, Any],
) -> int:
    explicit_expires_at = _int_result_value(
        verifier_result.get("expires_at"),
        "expires_at",
    )
    now = int(time.time())
    if verified_at > now:
        raise ValueError("Routstr TEE verifier result verified_at is in the future")
    if explicit_expires_at is not None and explicit_expires_at <= now:
        raise ValueError("Routstr TEE verifier result is expired")

    max_age_seconds = _int_policy_value(
        verifier_policy,
        "max_verifier_age_seconds",
        "max_evidence_age_seconds",
    )
    if max_age_seconds is None:
        max_age_seconds = DEFAULT_ROUTSTR_TEE_VERIFIER_MAX_AGE_SECONDS
    if max_age_seconds <= 0:
        raise ValueError("max_verifier_age_seconds must be positive")

    bounded_expires_at = verified_at + max_age_seconds
    if bounded_expires_at <= now:
        raise ValueError("Routstr TEE verifier result is stale")
    if explicit_expires_at is None:
        return bounded_expires_at
    return min(explicit_expires_at, bounded_expires_at)


def _verified_routstr_tee_command() -> tuple[list[str], str, str]:
    command = str(getattr(settings, "routstr_tee_verifier_command", "") or "")
    argv = _command_argv(command, label="Routstr TEE verifier command")
    expected_digest = str(
        getattr(settings, "routstr_tee_verifier_command_digest", "") or ""
    ).strip()
    if not expected_digest:
        raise ValueError("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST is required")
    if not _is_prefixed_sha256_digest_value(expected_digest):
        raise ValueError("ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST must be a sha256 digest")

    artifact_path = (
        str(getattr(settings, "routstr_tee_verifier_artifact_path", "") or "").strip()
        or argv[0]
    )
    _require_artifact_path_bound_to_command(
        argv=argv,
        artifact_path=artifact_path,
        setting_name="ROUTSTR_TEE_VERIFIER_ARTIFACT_PATH",
        command_label="ROUTSTR_TEE_VERIFIER_COMMAND",
    )
    try:
        actual_digest = _sha256_bytes(Path(artifact_path).read_bytes())
    except Exception as exc:
        raise ValueError(
            f"failed to hash Routstr TEE verifier artifact: {type(exc).__name__}: {exc}"
        ) from exc
    if actual_digest != expected_digest:
        raise ValueError(
            "Routstr TEE verifier command digest mismatch: "
            f"expected {expected_digest}, got {actual_digest}"
        )
    return argv, actual_digest, artifact_path


def _routstr_tee_attestation_command_configured() -> bool:
    command = str(getattr(settings, "routstr_tee_attestation_command", "") or "")
    return bool(command.strip())


def _verified_routstr_tee_attestation_command() -> tuple[list[str], str, str]:
    command = str(getattr(settings, "routstr_tee_attestation_command", "") or "")
    argv = _command_argv(command, label="Routstr TEE attestation command")
    expected_digest = str(
        getattr(settings, "routstr_tee_attestation_command_digest", "") or ""
    ).strip()
    if not expected_digest:
        raise ValueError("ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST is required")
    if not _is_prefixed_sha256_digest_value(expected_digest):
        raise ValueError(
            "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST must be a sha256 digest"
        )

    artifact_path = (
        str(
            getattr(settings, "routstr_tee_attestation_artifact_path", "") or ""
        ).strip()
        or argv[0]
    )
    _require_artifact_path_bound_to_command(
        argv=argv,
        artifact_path=artifact_path,
        setting_name="ROUTSTR_TEE_ATTESTATION_ARTIFACT_PATH",
        command_label="ROUTSTR_TEE_ATTESTATION_COMMAND",
    )
    try:
        actual_digest = _sha256_bytes(Path(artifact_path).read_bytes())
    except Exception as exc:
        raise ValueError(
            f"failed to hash Routstr TEE attestation artifact: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if actual_digest != expected_digest:
        raise ValueError(
            "Routstr TEE attestation command digest mismatch: "
            f"expected {expected_digest}, got {actual_digest}"
        )
    return argv, actual_digest, artifact_path


def _read_public_key() -> tuple[str | None, str | None]:
    public_key = str(getattr(settings, "routstr_attestation_public_key", "") or "")
    if public_key:
        return public_key, None

    path = str(getattr(settings, "routstr_attestation_public_key_path", "") or "")
    if not path:
        return None, None

    try:
        key_bytes = Path(path).read_bytes()
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return key_bytes.decode("utf-8", errors="replace").strip(), None


def read_routstr_hpke_key_config() -> bytes:
    """Return the configured public EHBP/OHTTP HPKE key config bytes."""
    encoded = str(
        getattr(settings, "routstr_attestation_hpke_key_config_b64", "") or ""
    ).strip()
    if encoded:
        try:
            key_config = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError(
                f"ROUTSTR_TEE_HPKE_KEY_CONFIG_B64 is not valid base64: {exc}"
            ) from exc
    else:
        path = str(
            getattr(settings, "routstr_attestation_hpke_key_config_path", "") or ""
        )
        if not path:
            raise ValueError("Routstr TEE HPKE key config is not configured")
        try:
            key_config = Path(path).read_bytes()
        except Exception as exc:
            raise ValueError(
                f"failed to read Routstr TEE HPKE key config: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    try:
        parse_ehbp_key_config(key_config)
    except Exception as exc:
        raise ValueError(f"invalid Routstr TEE HPKE key config: {exc}") from exc
    return key_config


def _hpke_key_config_evidence() -> dict[str, Any]:
    base: dict[str, Any] = {
        "available": False,
        "key_config_b64": None,
        "key_config_digest": None,
        "public_key_hex": None,
        "public_key_digest": None,
        "failure_reason": None,
    }
    try:
        key_config = read_routstr_hpke_key_config()
        parsed = parse_ehbp_key_config(key_config)
    except Exception as exc:
        base["failure_reason"] = str(exc)
        return base

    public_key_hex = parsed.public_key.hex()
    base.update(
        {
            "available": True,
            "key_config_b64": base64.b64encode(key_config).decode("ascii"),
            "key_config_digest": _sha256_bytes(key_config),
            "public_key_hex": public_key_hex,
            "public_key_digest": _sha256_bytes(parsed.public_key),
            "failure_reason": None,
        }
    )
    return base


def _unverified_routstr_tee_verification(failure_reason: str) -> dict[str, Any]:
    return {
        "verified": False,
        "verifier": None,
        "verified_at": None,
        "expires_at": None,
        "failure_reason": failure_reason,
        "evidence_digest": None,
        "verified_claims": {},
    }


def _current_routstr_tee_verification_failure(status: dict[str, Any]) -> str | None:
    verified = status.get("verified")
    if verified is False:
        return None
    if verified is not True:
        return "Routstr TEE local verification verified must be true"
    verifier = _public_string(status.get("verifier"))
    if verifier is None:
        return "Routstr TEE local verification verifier must be a non-empty string"
    if verifier == "cap-attestation-proxy":
        return _cap_routstr_tee_verification_failure(status)
    if not _is_prefixed_sha256_digest_value(status.get("evidence_digest")):
        return "Routstr TEE local verification evidence_digest must be a sha256 digest"
    verified_claims = status.get("verified_claims")
    if not isinstance(verified_claims, dict) or not verified_claims:
        return "Routstr TEE local verification must include proof claims"
    for digest_claim in (
        "hpke_key_config_digest",
        "hpke_public_key_digest",
        "public_key_digest",
        "routstr_code_measurement",
        "routstr_config_measurement",
        "tee_attestation_report_digest",
        "tee_certificate_chain_digest",
        "tee_report_data_digest",
    ):
        if not _is_prefixed_sha256_digest_value(verified_claims.get(digest_claim)):
            return (
                "Routstr TEE local verification "
                f"{digest_claim} claim must be a sha256 digest"
            )
    if not _public_string(verified_claims.get("attestation_document_format")):
        return (
            "Routstr TEE local verification "
            "attestation_document_format claim must be a non-empty string"
        )
    if not _is_hex_string(verified_claims.get("tee_report_data_hex"), length=128):
        return "Routstr TEE local verification tee_report_data_hex claim must be 128 hex characters"
    tee_report_nonce = _public_string(verified_claims.get("tee_report_nonce"))
    if tee_report_nonce is None:
        return (
            "Routstr TEE local verification "
            "tee_report_nonce claim must be a non-empty string"
        )
    tee_report_nonce_digest = verified_claims.get("tee_report_nonce_digest")
    if not _is_prefixed_sha256_digest_value(tee_report_nonce_digest):
        return (
            "Routstr TEE local verification "
            "tee_report_nonce_digest claim must be a sha256 digest"
        )
    if tee_report_nonce_digest != _sha256_bytes(tee_report_nonce.encode("utf-8")):
        return (
            "Routstr TEE local verification "
            "tee_report_nonce_digest claim does not match tee_report_nonce"
        )
    verification_steps = verified_claims.get("verification_steps")
    if verification_steps_failure := _routstr_tee_verification_steps_failure(
        verification_steps,
        label="Routstr TEE local verification",
    ):
        return verification_steps_failure
    try:
        _sha256_json(verified_claims)
    except (TypeError, ValueError):
        return "Routstr TEE local verification verified_claims must be canonical JSON"
    try:
        verified_at = _int_result_value(status.get("verified_at"), "verified_at")
    except ValueError as exc:
        return f"Routstr TEE local verification {exc}"
    if verified_at is None:
        return "Routstr TEE local verification must include verified_at"
    now = int(time.time())
    if verified_at > now:
        return "Routstr TEE local verification verified_at is in the future"
    try:
        expires_at = _int_result_value(status.get("expires_at"), "expires_at")
    except ValueError as exc:
        return f"Routstr TEE local verification {exc}"
    if expires_at is None:
        return "Routstr TEE local verification must include a future expires_at"
    if expires_at <= now:
        return "Routstr TEE local verification has expired"
    return None


def _cap_routstr_tee_verification_failure(status: dict[str, Any]) -> str | None:
    if not _is_prefixed_sha256_digest_value(status.get("evidence_digest")):
        return "CAP Routstr TEE verification evidence_digest must be a sha256 digest"
    verified_claims = status.get("verified_claims")
    if not isinstance(verified_claims, dict) or not verified_claims:
        return "CAP Routstr TEE verification must include proof claims"
    if verified_claims.get("cap_claims_verified") is not True:
        return "CAP Routstr TEE verification cap_claims_verified must be true"
    if verified_claims.get("cap_state") != "unlocked":
        return "CAP Routstr TEE verification cap_state must be unlocked"
    if (
        verified_claims.get("client_confidentiality_boundary")
        != CLIENT_CONFIDENTIALITY_BOUNDARY_ATTESTED_TLS
    ):
        return (
            "CAP Routstr TEE verification client_confidentiality_boundary must "
            "be attested-tls-termination"
        )
    if verified_claims.get("tls_terminates_in_attested_tee") is not True:
        return (
            "CAP Routstr TEE verification must attest TLS termination in the TEE"
        )
    for claim in (
        "cap_attestation_url",
        "cap_claims_instance_id",
        "cap_status_url",
        "cap_tee_domain",
        "cap_tenant_id",
    ):
        if not _public_string(verified_claims.get(claim)):
            return f"CAP Routstr TEE verification {claim} must be a non-empty string"
    for url_claim in ("cap_attestation_url", "cap_status_url"):
        claim_url = _public_string(verified_claims.get(url_claim))
        if claim_url is None or urlsplit(claim_url).scheme != "https":
            return f"CAP Routstr TEE verification {url_claim} must be HTTPS"
    for digest_claim in ("cap_status_digest", "routstr_config_measurement"):
        if not _is_prefixed_sha256_digest_value(verified_claims.get(digest_claim)):
            return (
                "CAP Routstr TEE verification "
                f"{digest_claim} claim must be a sha256 digest"
            )
    if verified_claims.get("cap_status_digest") != status.get("evidence_digest"):
        return (
            "CAP Routstr TEE verification cap_status_digest must match "
            "evidence_digest"
        )
    if verification_steps_failure := _routstr_tee_verification_steps_failure(
        verified_claims.get("verification_steps"),
        label="CAP Routstr TEE verification",
        required_steps=CAP_ROUTSTR_TEE_VERIFICATION_STEPS,
    ):
        return verification_steps_failure
    try:
        _sha256_json(verified_claims)
    except (TypeError, ValueError):
        return "CAP Routstr TEE verification verified_claims must be canonical JSON"
    try:
        verified_at = _int_result_value(status.get("verified_at"), "verified_at")
    except ValueError as exc:
        return f"CAP Routstr TEE verification {exc}"
    if verified_at is None:
        return "CAP Routstr TEE verification must include verified_at"
    now = int(time.time())
    if verified_at > now:
        return "CAP Routstr TEE verification verified_at is in the future"
    try:
        expires_at = _int_result_value(status.get("expires_at"), "expires_at")
    except ValueError as exc:
        return f"CAP Routstr TEE verification {exc}"
    if expires_at is None:
        return "CAP Routstr TEE verification must include a future expires_at"
    if expires_at <= now:
        return "CAP Routstr TEE verification has expired"
    return None


def _accepted_routstr_tee_verification(
    status: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    failure_reason = _current_routstr_tee_verification_failure(status)
    if failure_reason is None:
        return status, None
    return _unverified_routstr_tee_verification(failure_reason), failure_reason


def _downgrade_public_verification(
    public_status: dict[str, Any],
    failure_reason: str,
) -> None:
    public_status["verified"] = False
    public_status["verifier"] = None
    public_status["verified_at"] = None
    public_status["expires_at"] = None
    public_status["evidence_digest"] = None
    public_status["failure_reason_digest"] = _sha256_json(failure_reason)
    public_status.pop("failure_reason", None)
    public_status.pop("verified_claims", None)
    public_status.pop("verified_claims_digest", None)
    public_status.pop("proof_claims", None)


def _public_status_without_raw_details(
    status: dict[str, Any],
    *,
    public_claim_keys: tuple[str, ...] = (),
    public_verification_step_keys: tuple[str, ...] = (),
    verified_status_failure: Callable[[dict[str, Any]], str | None] | None = None,
) -> dict[str, Any]:
    public_status = dict(status)
    failure_reason = public_status.get("failure_reason")
    if isinstance(failure_reason, str) and failure_reason.strip():
        public_status["failure_reason_digest"] = _sha256_json(failure_reason)
        public_status.pop("failure_reason", None)

    if public_status.get("verified") is True and verified_status_failure is not None:
        verification_failure = verified_status_failure(status)
        if verification_failure:
            _downgrade_public_verification(public_status, verification_failure)
            return public_status

    verified_claims = public_status.pop("verified_claims", None)
    if public_status.get("verified") is True and isinstance(verified_claims, dict):
        if verified_claims:
            try:
                public_status["verified_claims_digest"] = _sha256_json(verified_claims)
            except (TypeError, ValueError):
                _downgrade_public_verification(
                    public_status,
                    "public verified claims must be canonical JSON",
                )
            else:
                proof_claims = {
                    key: verified_claims[key]
                    for key in public_claim_keys
                    if key != "verification_steps"
                    if key in verified_claims
                }
                verification_steps = verified_claims.get("verification_steps")
                if isinstance(verification_steps, dict):
                    public_steps = {
                        step: verification_steps[step]
                        for step in public_verification_step_keys
                        if isinstance(verification_steps.get(step), bool)
                    }
                    if public_steps:
                        proof_claims["verification_steps"] = public_steps
                if proof_claims:
                    public_status["proof_claims"] = proof_claims
    return public_status


def _public_tee_statement(tee: dict[str, Any]) -> dict[str, Any]:
    public_tee = dict(tee)
    failure_reason = public_tee.get("failure_reason")
    if isinstance(failure_reason, str) and failure_reason.strip():
        public_tee["failure_reason_digest"] = _sha256_json(failure_reason)
        public_tee.pop("failure_reason", None)

    hpke_key_config = public_tee.get("hpke_key_config")
    if isinstance(hpke_key_config, dict):
        public_tee["hpke_key_config"] = _public_status_without_raw_details(
            hpke_key_config
        )

    local_verification = public_tee.get("local_verification")
    if isinstance(local_verification, dict):
        public_tee["local_verification"] = _public_status_without_raw_details(
            local_verification,
            public_claim_keys=PUBLIC_ROUTSTR_TEE_PROOF_CLAIMS,
            public_verification_step_keys=PUBLIC_ROUTSTR_TEE_VERIFICATION_STEPS,
            verified_status_failure=_current_routstr_tee_verification_failure,
        )
    return public_tee


def _bound_result_values(
    verifier_result: dict[str, Any],
    claims: dict[str, Any],
    key: str,
) -> list[object]:
    values: list[object] = []
    if key in verifier_result:
        values.append(verifier_result[key])
    if key in claims:
        values.append(claims[key])
    return values


def _runtime_evidence_digest(claims: dict[str, Any]) -> str:
    volatile_claims = {
        "payload_verification_nonce",
        "runtime_evidence_digest",
        "tee_report_nonce",
        "tee_report_nonce_digest",
    }
    evidence_claims = {
        key: value for key, value in claims.items() if key not in volatile_claims
    }
    return _sha256_json(evidence_claims)


def _required_local_proxy_binary_artifacts(
    routing_policy: dict[str, Any] | None,
) -> dict[str, set[str]]:
    if not isinstance(routing_policy, dict):
        return {}
    if routing_policy.get("mode") != "required" or routing_policy.get("required") is not True:
        return {}
    providers = routing_policy.get("providers")
    if not isinstance(providers, list):
        return {}

    required: dict[str, set[str]] = {}
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        provider_type = _public_string(provider.get("provider_type"))
        confidentiality = provider.get("confidentiality")
        if not isinstance(confidentiality, dict):
            confidentiality = {}
        confidentiality_mode = _public_string(confidentiality.get("mode"))
        if not (
            provider_type == "privatemode" or confidentiality_mode == "privatemode"
        ):
            continue
        if confidentiality.get("verified") is not True:
            continue
        public_policy = provider.get("confidentiality_policy")
        if not isinstance(public_policy, dict):
            continue
        for key in ("proxy_binary_digest", "proxyBinaryDigest"):
            if key not in public_policy:
                continue
            digest = _require_prefixed_sha256_digest_value(
                public_policy.get(key),
                f"confidentiality_policy.{key}",
            )
            required.setdefault("privatemode_proxy_binary", set()).add(digest)
    return required


def _private_local_artifacts_snapshot() -> dict[str, dict[str, str]]:
    try:
        from ..proxy import _upstreams
    except Exception:
        return {}

    artifacts: dict[str, dict[str, str]] = {}
    for provider in _upstreams:
        provider_type = _public_string(getattr(provider, "provider_type", None))
        status_getter = getattr(provider, "confidentiality_status", None)
        try:
            status = status_getter() if callable(status_getter) else None
        except Exception:
            continue
        if getattr(status, "verified", False) is not True:
            continue
        mode = _public_string(getattr(status, "mode", None))
        if not (provider_type == "privatemode" or mode == "privatemode"):
            continue

        policy_getter = getattr(provider, "confidentiality_policy", None)
        try:
            policy = policy_getter() if callable(policy_getter) else None
        except Exception:
            continue
        raw_policy = getattr(policy, "policy", None)
        if not isinstance(raw_policy, dict):
            continue

        digest = None
        for key in ("proxy_binary_digest", "proxyBinaryDigest"):
            if key in raw_policy:
                digest = _require_prefixed_sha256_digest_value(
                    raw_policy.get(key),
                    f"confidentiality_policy.{key}",
                )
                break
        if digest is None:
            continue

        path = None
        for key in ("proxy_binary_path", "proxyBinaryPath"):
            value = raw_policy.get(key)
            if isinstance(value, str) and value.strip():
                path = value.strip()
                break
        if path is None:
            continue

        existing = artifacts.get("privatemode_proxy_binary")
        if existing is not None and existing != {"digest": digest, "path": path}:
            raise ValueError(
                "local_artifacts.privatemode_proxy_binary has conflicting provider "
                "policy values"
            )
        artifacts["privatemode_proxy_binary"] = {"digest": digest, "path": path}
    return artifacts


def _validate_required_local_artifacts(
    *,
    claims: dict[str, Any],
    routing_policy: dict[str, Any] | None,
) -> None:
    required_local_proxy_digests = _required_local_proxy_binary_artifacts(routing_policy)
    if not required_local_proxy_digests:
        return

    local_artifacts = claims.get("attested_local_artifacts")
    if not isinstance(local_artifacts, dict):
        first_artifact_claim = sorted(required_local_proxy_digests)[0]
        raise ValueError(f"attested_local_artifacts.{first_artifact_claim} claim is required")
    for artifact_claim, required_digests in sorted(required_local_proxy_digests.items()):
        if len(required_digests) != 1:
            raise ValueError(
                f"attested_local_artifacts.{artifact_claim} has conflicting routing policy digests"
            )
        claimed_digest = local_artifacts.get(artifact_claim)
        if not _is_prefixed_sha256_digest_value(claimed_digest):
            raise ValueError(
                f"attested_local_artifacts.{artifact_claim} claim is required"
            )
        claimed_digest = cast(str, claimed_digest).strip()
        required_digest = next(iter(required_digests))
        if claimed_digest != required_digest:
            raise ValueError(
                f"attested_local_artifacts.{artifact_claim} claim does not match routing policy"
            )


def _routstr_tee_verification_steps_failure(
    value: object,
    *,
    label: str,
    required_steps: tuple[str, ...] = REQUIRED_ROUTSTR_TEE_VERIFICATION_STEPS,
) -> str | None:
    if not isinstance(value, dict):
        return f"{label} verification_steps must be a JSON object"
    if any(not isinstance(step, str) or not step.strip() for step in value):
        return f"{label} verification_steps keys must be non-empty strings"
    if any(not isinstance(step_value, bool) for step_value in value.values()):
        return f"{label} verification_steps values must be JSON booleans"
    for step in required_steps:
        if value.get(step) is not True:
            return f"{label} {step} step is required"
    return None


def _routstr_tee_report_data_digest(
    *,
    routing_policy_digest: str,
    hpke_key_config_digest: str,
    hpke_public_key_digest: str,
    public_key_digest: str,
    verification_nonce: str,
) -> str:
    return _sha256_json(
        {
            "schema_version": "routstr-tee-report-data-v1",
            "service": "routstr",
            "routing_policy_digest": routing_policy_digest,
            "hpke_key_config_digest": hpke_key_config_digest,
            "hpke_public_key_digest": hpke_public_key_digest,
            "public_key_digest": public_key_digest,
            "verification_nonce": verification_nonce,
        }
    )


def _routstr_tee_report_data_hex(
    *,
    routing_policy_digest: str,
    hpke_key_config_digest: str,
    hpke_public_key_digest: str,
    public_key_digest: str,
    verification_nonce: str,
) -> str:
    report_data_input = _canonical_json(
        {
            "schema_version": "routstr-tee-report-data-v1",
            "service": "routstr",
            "routing_policy_digest": routing_policy_digest,
            "hpke_key_config_digest": hpke_key_config_digest,
            "hpke_public_key_digest": hpke_public_key_digest,
            "public_key_digest": public_key_digest,
            "verification_nonce": verification_nonce,
        }
    )
    return hashlib.sha512(report_data_input).hexdigest()


def _validate_routstr_tee_verifier_result(
    *,
    verifier_result: dict[str, Any],
    verifier_policy: dict[str, Any],
    routing_policy_digest: str,
    routing_policy: dict[str, Any] | None = None,
    attestation_evidence_digest: str,
    attestation_document_format: str,
    hpke_key_config_digest: str,
    hpke_public_key_digest: str,
    public_key_digest: str | None,
    verification_nonce: str,
) -> dict[str, Any]:
    if verifier_result.get("verified") is not True:
        raise ValueError("Routstr TEE verifier did not return verified=true")

    verifier_value = verifier_result.get("verifier")
    if not isinstance(verifier_value, str):
        raise ValueError("Routstr TEE verifier identity must be a string")
    verifier = verifier_value.strip()
    if not verifier:
        raise ValueError("Routstr TEE verifier identity is required")

    raw_claims = verifier_result.get("claims")
    if not isinstance(raw_claims, dict):
        raise ValueError("Routstr TEE verifier claims must be a JSON object")
    claims = dict(raw_claims)

    expected_format = str(attestation_document_format or "").strip()
    if not expected_format:
        raise ValueError("Routstr TEE attestation document format is required")

    for label, value in (
        ("routing_policy_digest", routing_policy_digest),
        ("attestation_evidence_digest", attestation_evidence_digest),
        ("hpke_key_config_digest", hpke_key_config_digest),
        ("hpke_public_key_digest", hpke_public_key_digest),
    ):
        _require_prefixed_sha256_digest_value(value, label)
    if public_key_digest:
        _require_prefixed_sha256_digest_value(public_key_digest, "public_key_digest")
    else:
        raise ValueError("Routstr TEE public key digest is required")

    expected_values = {
        "routing_policy_digest": routing_policy_digest,
        "attestation_evidence_digest": attestation_evidence_digest,
        "attestation_document_format": expected_format,
        "hpke_key_config_digest": hpke_key_config_digest,
        "hpke_public_key_digest": hpke_public_key_digest,
        "public_key_digest": public_key_digest,
        "verification_nonce": verification_nonce,
    }
    for key, expected_value in expected_values.items():
        values = _bound_result_values(verifier_result, claims, key)
        if not values or any(value != expected_value for value in values):
            raise ValueError(f"{key} does not match Routstr TEE verifier payload")

    payload_claims = {
        "payload_routing_policy_digest": routing_policy_digest,
        "payload_verification_nonce": verification_nonce,
    }
    for key, expected_value in payload_claims.items():
        values = _bound_result_values(verifier_result, claims, key)
        if any(value != expected_value for value in values):
            raise ValueError(f"{key} does not match Routstr TEE verifier payload")

    _require_prefixed_sha256_digest_value(
        claims.get("hpke_public_key_digest"),
        "hpke_public_key_digest claim",
    )
    _require_prefixed_sha256_digest_value(
        claims.get("public_key_digest"),
        "public_key_digest claim",
    )

    expected_report_data_digest = _routstr_tee_report_data_digest(
        routing_policy_digest=routing_policy_digest,
        hpke_key_config_digest=hpke_key_config_digest,
        hpke_public_key_digest=hpke_public_key_digest,
        public_key_digest=public_key_digest,
        verification_nonce=verification_nonce,
    )
    expected_report_data_hex = _routstr_tee_report_data_hex(
        routing_policy_digest=routing_policy_digest,
        hpke_key_config_digest=hpke_key_config_digest,
        hpke_public_key_digest=hpke_public_key_digest,
        public_key_digest=public_key_digest,
        verification_nonce=verification_nonce,
    )

    raw_steps = verifier_result.get("verification_steps")
    if raw_steps is None:
        raw_steps = claims.get("verification_steps")
    if verification_steps_failure := _routstr_tee_verification_steps_failure(
        raw_steps,
        label="Routstr TEE",
    ):
        raise ValueError(verification_steps_failure)
    claims["verification_steps"] = dict(raw_steps)

    for required_claim in REQUIRED_ROUTSTR_TEE_PROOF_CLAIMS:
        if required_claim == "tee_report_data_hex":
            continue
        if not _public_string(claims.get(required_claim)):
            raise ValueError(f"{required_claim} claim is required")
    tee_report_data_hex_value = claims.get("tee_report_data_hex")
    if tee_report_data_hex_value is None or (
        isinstance(tee_report_data_hex_value, str)
        and not tee_report_data_hex_value.strip()
    ):
        raise ValueError("tee_report_data_hex claim is required")
    if not _is_hex_string(tee_report_data_hex_value, length=128):
        raise ValueError("tee_report_data_hex claim must be 128 hex characters")
    tee_report_data_hex = cast(str, tee_report_data_hex_value).strip().lower()
    for digest_claim in (
        "tee_attestation_report_digest",
        "tee_certificate_chain_digest",
        "tee_report_data_digest",
    ):
        _require_prefixed_sha256_digest_value(
            claims.get(digest_claim),
            f"{digest_claim} claim",
        )
    if not hmac.compare_digest(tee_report_data_hex, expected_report_data_hex):
        raise ValueError("tee_report_data_hex claim does not match policy and keys")
    if claims.get("tee_report_data_digest") != expected_report_data_digest:
        raise ValueError("tee_report_data_digest claim does not match policy and keys")
    expected_nonce_digest = _sha256_bytes(verification_nonce.encode("utf-8"))
    if claims.get("tee_report_nonce") != verification_nonce:
        raise ValueError("tee_report_nonce does not match Routstr TEE verifier payload")
    if claims.get("tee_report_nonce_digest") != expected_nonce_digest:
        raise ValueError(
            "tee_report_nonce_digest does not match Routstr TEE verifier payload"
        )

    for required_claim in (
        "routstr_code_measurement",
        "routstr_config_measurement",
    ):
        if not str(claims.get(required_claim) or "").strip():
            raise ValueError(f"{required_claim} claim is required")
        _require_prefixed_sha256_digest_value(
            claims.get(required_claim),
            f"{required_claim} claim",
        )
    allowed_code_measurements = _routstr_code_measurement_policy_values(verifier_policy)
    if not allowed_code_measurements:
        raise ValueError(
            "expected_routstr_code_measurement or "
            "allowed_routstr_code_measurements is required"
        )
    for measurement in allowed_code_measurements:
        _require_prefixed_sha256_digest_value(
            measurement,
            "routstr_code_measurement policy value",
        )
    if claims.get("routstr_code_measurement") not in allowed_code_measurements:
        raise ValueError("routstr_code_measurement claim does not match policy")
    if claims.get("routstr_config_measurement") != routing_policy_digest:
        raise ValueError(
            "routstr_config_measurement claim does not match routing policy"
        )
    _validate_required_local_artifacts(
        claims=claims,
        routing_policy=routing_policy,
    )

    verified_at = _int_result_value(verifier_result.get("verified_at"), "verified_at")
    if verified_at is None:
        verified_at = int(time.time())
    expires_at = _bounded_routstr_tee_expires_at(
        verifier_result=verifier_result,
        verified_at=verified_at,
        verifier_policy=verifier_policy,
    )

    claims["runtime_evidence_digest"] = _runtime_evidence_digest(claims)
    return {
        "verified": True,
        "verifier": verifier,
        "verified_at": verified_at,
        "expires_at": expires_at,
        "failure_reason": None,
        "evidence_digest": claims["runtime_evidence_digest"],
        "verified_claims": claims,
    }


def _generate_routstr_tee_attestation_document(
    *,
    tee: dict[str, Any],
    routing_policy: dict[str, Any],
    routing_policy_digest: str,
    verifier_policy: dict[str, Any],
    hpke_key_config: dict[str, Any],
    public_key_digest: str,
    verification_nonce: str,
    tee_report_data_digest: str,
    tee_report_data_hex: str,
) -> None:
    argv, command_digest, artifact_path = _verified_routstr_tee_attestation_command()
    payload = {
        "schema_version": "routstr-tee-attestation-request-v1",
        "service": "routstr",
        "version": __version__,
        "routing_policy": routing_policy,
        "routing_policy_digest": routing_policy_digest,
        "policy": verifier_policy,
        "attestation_document_format": tee.get("evidence_format"),
        "public_key": tee.get("public_key"),
        "public_key_digest": public_key_digest,
        "hpke_key_config_b64": hpke_key_config.get("key_config_b64"),
        "hpke_key_config_digest": hpke_key_config.get("key_config_digest"),
        "hpke_public_key_hex": hpke_key_config.get("public_key_hex"),
        "hpke_public_key_digest": hpke_key_config.get("public_key_digest"),
        "verification_nonce": verification_nonce,
        "tee_report_data_digest": tee_report_data_digest,
        "tee_report_data_hex": tee_report_data_hex,
        "attestation_command_digest": command_digest,
        "attestation_artifact_path": artifact_path,
    }

    try:
        proc = subprocess.run(
            argv,
            input=json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_timeout_seconds(),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Routstr TEE attestation command timed out") from exc
    except Exception as exc:
        raise ValueError(
            f"Routstr TEE attestation command failed to run: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if proc.returncode != 0:
        raise ValueError(
            "Routstr TEE attestation command failed with exit code "
            f"{proc.returncode}{_stderr_failure_detail(proc.stderr)}"
        )

    try:
        result = _loads_strict_json(
            proc.stdout.decode("utf-8"),
            "Routstr TEE attestation command output JSON",
        )
    except Exception as exc:
        raise ValueError(
            f"Routstr TEE attestation command did not return JSON: {exc}"
        ) from exc
    if not isinstance(result, dict):
        raise ValueError("Routstr TEE attestation command output must be a JSON object")

    schema_version = result.get("schema_version")
    if schema_version not in (None, "routstr-tee-attestation-document-v1"):
        raise ValueError("Routstr TEE attestation command returned unexpected schema")

    raw_result_format = result.get("attestation_document_format")
    if raw_result_format is None:
        raw_result_format = tee.get("evidence_format")
    elif not isinstance(raw_result_format, str):
        raise ValueError(
            "Routstr TEE attestation command attestation_document_format "
            "must be a string"
        )
    raw_expected_format = tee.get("evidence_format")
    if not isinstance(raw_expected_format, str):
        expected_format = ""
    else:
        expected_format = raw_expected_format.strip()
    result_format = raw_result_format.strip() if isinstance(raw_result_format, str) else ""
    if not expected_format:
        raise ValueError("Routstr TEE attestation document format is not configured")
    if result_format != expected_format:
        raise ValueError(
            "Routstr TEE attestation command returned a different document format"
        )
    if result.get("tee_report_data_digest") not in (None, tee_report_data_digest):
        raise ValueError(
            "Routstr TEE attestation command returned a different report data digest"
        )
    if result.get("tee_report_data_hex") not in (None, tee_report_data_hex):
        raise ValueError(
            "Routstr TEE attestation command returned a different report data value"
        )

    raw_document_b64 = result.get("attestation_document_b64")
    if not isinstance(raw_document_b64, str):
        raise ValueError(
            "Routstr TEE attestation command attestation_document_b64 must be a string"
        )
    document_b64 = raw_document_b64.strip()
    if not document_b64:
        raise ValueError(
            "Routstr TEE attestation command did not return attestation_document_b64"
        )
    try:
        evidence = base64.b64decode(document_b64, validate=True)
    except Exception as exc:
        raise ValueError(
            f"Routstr TEE attestation command returned invalid base64: {exc}"
        ) from exc
    if not evidence:
        raise ValueError("Routstr TEE attestation command returned empty evidence")

    tee.update(
        {
            "available": True,
            "evidence_format": result_format,
            "attestation_document_b64": document_b64,
            "attestation_evidence_digest": _sha256_bytes(evidence),
            "failure_reason": None,
        }
    )


def _verify_routstr_tee_evidence(
    *,
    tee: dict[str, Any],
    routing_policy: dict[str, Any],
) -> dict[str, Any]:
    hpke_key_config = tee.get("hpke_key_config")
    if not isinstance(hpke_key_config, dict):
        return _unverified_routstr_tee_verification(
            "Routstr TEE HPKE key config is unavailable"
        )

    try:
        argv, verifier_digest, verifier_artifact_path = _verified_routstr_tee_command()
        verifier_policy = _routstr_tee_verifier_policy()
    except Exception as exc:
        return _unverified_routstr_tee_verification(str(exc))

    routing_policy_digest = _sha256_json(routing_policy)
    verification_nonce = _new_verification_nonce()
    public_key_digest = tee.get("public_key_digest")
    if not _is_prefixed_sha256_digest_value(public_key_digest):
        return _unverified_routstr_tee_verification(
            "Routstr TEE public key digest is required"
        )

    attestation_document_format = tee.get("evidence_format")
    if not _public_string(attestation_document_format):
        return _unverified_routstr_tee_verification(
            "Routstr TEE attestation document format is required"
        )
    hpke_key_config_digest = hpke_key_config.get("key_config_digest")
    if not _is_prefixed_sha256_digest_value(hpke_key_config_digest):
        return _unverified_routstr_tee_verification(
            "Routstr TEE HPKE key config digest must be a sha256 digest"
        )
    hpke_public_key_digest = hpke_key_config.get("public_key_digest")
    if not _is_prefixed_sha256_digest_value(hpke_public_key_digest):
        return _unverified_routstr_tee_verification(
            "Routstr TEE HPKE public key digest must be a sha256 digest"
        )

    public_key_digest = cast(str, public_key_digest)
    attestation_document_format = cast(str, attestation_document_format).strip()
    hpke_key_config_digest = cast(str, hpke_key_config_digest).strip()
    hpke_public_key_digest = cast(str, hpke_public_key_digest).strip()
    tee_report_data_digest = _routstr_tee_report_data_digest(
        routing_policy_digest=routing_policy_digest,
        hpke_key_config_digest=hpke_key_config_digest,
        hpke_public_key_digest=hpke_public_key_digest,
        public_key_digest=public_key_digest,
        verification_nonce=verification_nonce,
    )
    tee_report_data_hex = _routstr_tee_report_data_hex(
        routing_policy_digest=routing_policy_digest,
        hpke_key_config_digest=hpke_key_config_digest,
        hpke_public_key_digest=hpke_public_key_digest,
        public_key_digest=public_key_digest,
        verification_nonce=verification_nonce,
    )

    if _routstr_tee_attestation_command_configured():
        try:
            _generate_routstr_tee_attestation_document(
                tee=tee,
                routing_policy=routing_policy,
                routing_policy_digest=routing_policy_digest,
                verifier_policy=verifier_policy,
                hpke_key_config=hpke_key_config,
                public_key_digest=public_key_digest,
                verification_nonce=verification_nonce,
                tee_report_data_digest=tee_report_data_digest,
                tee_report_data_hex=tee_report_data_hex,
            )
        except Exception as exc:
            return _unverified_routstr_tee_verification(str(exc))

    attestation_evidence_digest = tee.get("attestation_evidence_digest")
    if not _is_prefixed_sha256_digest_value(attestation_evidence_digest):
        return _unverified_routstr_tee_verification(
            "Routstr TEE attestation evidence digest must be a sha256 digest"
        )
    attestation_evidence_digest = cast(str, attestation_evidence_digest).strip()
    attestation_document_b64 = tee.get("attestation_document_b64")
    if (
        not isinstance(attestation_document_b64, str)
        or not attestation_document_b64.strip()
    ):
        return _unverified_routstr_tee_verification(
            "Routstr TEE attestation document must be base64 evidence"
        )
    attestation_document_b64 = attestation_document_b64.strip()
    try:
        attestation_document = base64.b64decode(
            attestation_document_b64,
            validate=True,
        )
    except Exception:
        return _unverified_routstr_tee_verification(
            "Routstr TEE attestation document must be base64 evidence"
        )
    if not attestation_document:
        return _unverified_routstr_tee_verification(
            "Routstr TEE attestation document must be base64 evidence"
        )
    if _sha256_bytes(attestation_document) != attestation_evidence_digest:
        return _unverified_routstr_tee_verification(
            "Routstr TEE attestation evidence digest does not match document"
        )

    try:
        local_artifacts = _private_local_artifacts_snapshot()
    except Exception as exc:
        return _unverified_routstr_tee_verification(str(exc))

    payload = {
        "schema_version": "routstr-tee-verifier-v1",
        "service": "routstr",
        "version": __version__,
        "routing_policy": routing_policy,
        "local_artifacts": local_artifacts,
        "routing_policy_digest": routing_policy_digest,
        "policy": verifier_policy,
        "attestation_document_format": attestation_document_format,
        "attestation_document_b64": attestation_document_b64,
        "attestation_evidence_digest": attestation_evidence_digest,
        "public_key": tee.get("public_key"),
        "public_key_digest": public_key_digest,
        "hpke_key_config_b64": hpke_key_config.get("key_config_b64"),
        "hpke_key_config_digest": hpke_key_config_digest,
        "hpke_public_key_hex": hpke_key_config.get("public_key_hex"),
        "hpke_public_key_digest": hpke_public_key_digest,
        "verification_nonce": verification_nonce,
        "verifier_command_digest": verifier_digest,
        "verifier_artifact_path": verifier_artifact_path,
        "tee_report_data_digest": tee_report_data_digest,
        "tee_report_data_hex": tee_report_data_hex,
    }

    try:
        proc = subprocess.run(
            argv,
            input=json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_timeout_seconds(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _unverified_routstr_tee_verification(
            "Routstr TEE verifier command timed out"
        )
    except Exception as exc:
        return _unverified_routstr_tee_verification(
            f"Routstr TEE verifier command failed to run: {type(exc).__name__}: {exc}"
        )

    if proc.returncode != 0:
        return _unverified_routstr_tee_verification(
            "Routstr TEE verifier command failed with exit code "
            f"{proc.returncode}{_stderr_failure_detail(proc.stderr)}"
        )

    try:
        verifier_result = _loads_strict_json(
            proc.stdout.decode("utf-8"),
            "Routstr TEE verifier command output JSON",
        )
    except Exception as exc:
        return _unverified_routstr_tee_verification(
            f"Routstr TEE verifier command did not return JSON: {exc}"
        )
    if not isinstance(verifier_result, dict):
        return _unverified_routstr_tee_verification(
            "Routstr TEE verifier command output must be a JSON object"
        )

    try:
        return _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy=verifier_policy,
            routing_policy_digest=routing_policy_digest,
            routing_policy=routing_policy,
            attestation_evidence_digest=attestation_evidence_digest,
            attestation_document_format=attestation_document_format,
            hpke_key_config_digest=hpke_key_config_digest,
            hpke_public_key_digest=hpke_public_key_digest,
            public_key_digest=public_key_digest,
            verification_nonce=verification_nonce,
        )
    except Exception as exc:
        return _unverified_routstr_tee_verification(
            f"Routstr TEE verifier failed: {exc}"
        )


def _tee_evidence() -> dict[str, Any]:
    path = str(getattr(settings, "routstr_attestation_document_path", "") or "")
    evidence_format = str(
        getattr(settings, "routstr_attestation_document_format", "") or ""
    ).strip()
    attestation_command_configured = _routstr_tee_attestation_command_configured()
    public_key, public_key_error = _read_public_key()
    public_key_digest = (
        _sha256_bytes(public_key.encode("utf-8")) if public_key is not None else None
    )
    configured_public_key_digest_failure = (
        _configured_attested_tls_public_key_digest_failure(public_key_digest)
    )

    base = {
        "available": False,
        "evidence_format": evidence_format or None,
        "attestation_document_b64": None,
        "attestation_evidence_digest": None,
        "attestation_command_configured": attestation_command_configured,
        "public_key": public_key,
        "public_key_digest": public_key_digest,
        "self_verified": False,
        "client_verification_required": True,
        "failure_reason": None,
        "hpke_key_config": _hpke_key_config_evidence(),
    }

    if _cap_attestation_enabled():
        cap_attestation = _cap_attestation_status_evidence()
        base["cap_attestation"] = cap_attestation
        if cap_attestation.get("available") is True:
            base.update(
                {
                    "available": True,
                    "evidence_format": "cap-attestation-proxy-status",
                    "attestation_evidence_digest": cap_attestation.get(
                        "status_digest"
                    ),
                    "attestation_source": "cap-attestation-proxy",
                    "self_verified": True,
                    "failure_reason": None,
                }
            )
        else:
            base["attestation_source"] = "cap-attestation-proxy"
            base["failure_reason"] = str(
                cap_attestation.get("failure_reason")
                or "CAP attestation proxy status is unavailable"
            )
        return base

    if configured_public_key_digest_failure:
        base["failure_reason"] = configured_public_key_digest_failure
        return base

    if public_key_error:
        base["failure_reason"] = (
            f"failed to read Routstr TEE public key: {public_key_error}"
        )

    if not path:
        if attestation_command_configured:
            if not evidence_format and not base["failure_reason"]:
                base["failure_reason"] = (
                    "Routstr TEE attestation document format is not configured"
                )
            return base
        if not base["failure_reason"]:
            base["failure_reason"] = (
                "Routstr TEE attestation document path is not configured"
            )
        return base

    try:
        evidence = Path(path).read_bytes()
    except Exception as exc:
        base["failure_reason"] = (
            f"failed to read Routstr TEE attestation document: "
            f"{type(exc).__name__}: {exc}"
        )
        return base

    base.update(
        {
            "evidence_format": evidence_format or None,
            "attestation_document_b64": base64.b64encode(evidence).decode("ascii"),
            "attestation_evidence_digest": _sha256_bytes(evidence),
            "failure_reason": public_key_error,
        }
    )
    if not evidence_format:
        base["failure_reason"] = (
            "Routstr TEE attestation document format is not configured"
        )
        return base

    base["available"] = True
    return base


def _live_routstr_tee_attestation_command_failure(tee: dict[str, Any]) -> str | None:
    if not _routstr_tee_required_for_public_status():
        return None
    if tee.get("attestation_command_configured") is True:
        return None
    if tee.get("attestation_source") == "cap-attestation-proxy":
        return None
    if tee.get("available") is True:
        return (
            "Routstr TEE attestation command is required for live confidential "
            "routing; static evidence is only suitable for fixtures or offline "
            "validation"
        )
    return None


def _client_confidentiality_boundary_failure() -> str | None:
    if not _routstr_tee_required_for_public_status():
        return None
    boundary = _public_string(
        getattr(settings, "routstr_tee_client_confidentiality_boundary", "")
    )
    if boundary is not None:
        boundary = boundary.lower()
    if boundary == CLIENT_CONFIDENTIALITY_BOUNDARY_ATTESTED_TLS:
        return None
    return (
        "ROUTSTR_TEE_CLIENT_CONFIDENTIALITY_BOUNDARY must be "
        "attested-tls-termination for confidential routing; inbound EHBP/OHTTP "
        "request decryption is not implemented"
    )


def get_routstr_tee_readiness() -> dict[str, Any]:
    """Return whether local Routstr TEE evidence is verified enough to route."""
    tee = _tee_evidence()
    hpke_key_config = tee.get("hpke_key_config")
    if not isinstance(hpke_key_config, dict):
        hpke_key_config = {}
    cap_attestation_mode = tee.get("attestation_source") == "cap-attestation-proxy"

    failure_reasons: list[str] = []
    has_evidence_source = bool(tee.get("available")) or bool(
        tee.get("attestation_command_configured")
    )
    if not has_evidence_source:
        failure_reasons.append(
            str(tee.get("failure_reason") or "Routstr TEE evidence is unavailable")
        )
    elif not tee.get("evidence_format"):
        failure_reasons.append(
            str(
                tee.get("failure_reason")
                or "Routstr TEE attestation document format is not configured"
            )
        )
    elif tee.get("failure_reason"):
        failure_reasons.append(str(tee["failure_reason"]))
    elif live_command_failure := _live_routstr_tee_attestation_command_failure(tee):
        failure_reasons.append(live_command_failure)
    if not cap_attestation_mode and not bool(hpke_key_config.get("available")):
        failure_reasons.append(
            str(
                hpke_key_config.get("failure_reason")
                or "Routstr TEE HPKE key config is unavailable"
            )
        )
    if boundary_failure := _client_confidentiality_boundary_failure():
        failure_reasons.append(boundary_failure)
    local_verification = _unverified_routstr_tee_verification(
        "Routstr TEE evidence prerequisites are unavailable"
    )
    if not failure_reasons:
        routing_policy = _routing_policy_snapshot(routstr_tee_ready=True)
        if cap_attestation_mode:
            local_verification = _verify_cap_routstr_tee_evidence(
                tee=tee,
                routing_policy=routing_policy,
            )
        else:
            local_verification = _verify_routstr_tee_evidence(
                tee=tee,
                routing_policy=routing_policy,
            )
        if local_verification.get("verified") is not True:
            malformed_verification_failure = _current_routstr_tee_verification_failure(
                local_verification
            )
            verification_failure = str(
                malformed_verification_failure
                or local_verification.get("failure_reason")
                or "Routstr TEE evidence is not verified"
            )
            failure_reasons.append(verification_failure)
            local_verification = _unverified_routstr_tee_verification(
                verification_failure
            )
        else:
            local_verification, freshness_failure = _accepted_routstr_tee_verification(
                local_verification
            )
            if freshness_failure:
                failure_reasons.append(freshness_failure)

    return {
        "ready": not failure_reasons,
        "failure_reason": "; ".join(failure_reasons) if failure_reasons else None,
        "attestation_evidence_digest": tee.get("attestation_evidence_digest"),
        "hpke_key_config_digest": hpke_key_config.get("key_config_digest"),
        "hpke_public_key_digest": hpke_key_config.get("public_key_digest"),
        "local_verification": local_verification,
    }


def get_public_routstr_tee_status() -> dict[str, Any]:
    """Return local Routstr TEE readiness without raw diagnostics or claims."""
    readiness = get_routstr_tee_readiness()
    public_status: dict[str, Any] = {
        "required": _routstr_tee_required_for_public_status(),
        "ready": readiness.get("ready", False) is True,
        "client_confidentiality": _client_confidentiality_snapshot(),
        "attestation_evidence_digest": readiness.get("attestation_evidence_digest"),
        "hpke_key_config_digest": readiness.get("hpke_key_config_digest"),
        "hpke_public_key_digest": readiness.get("hpke_public_key_digest"),
    }

    failure_reason = readiness.get("failure_reason")
    if isinstance(failure_reason, str) and failure_reason.strip():
        public_status["failure_reason_digest"] = _sha256_json(failure_reason)

    local_verification = readiness.get("local_verification")
    if isinstance(local_verification, dict):
        public_status["local_verification"] = _public_status_without_raw_details(
            local_verification,
            public_claim_keys=PUBLIC_ROUTSTR_TEE_PROOF_CLAIMS,
            public_verification_step_keys=PUBLIC_ROUTSTR_TEE_VERIFICATION_STEPS,
            verified_status_failure=_current_routstr_tee_verification_failure,
        )
        if public_status["local_verification"].get("verified") is not True:
            public_status["ready"] = False
    elif public_status["ready"] is True:
        public_status["ready"] = False
    return public_status


def get_routstr_attestation_statement() -> dict[str, Any]:
    routing_policy = _routing_policy_snapshot()
    tee = _tee_evidence()
    cap_attestation_mode = tee.get("attestation_source") == "cap-attestation-proxy"
    has_evidence_source = bool(tee.get("available")) or bool(
        tee.get("attestation_command_configured")
    )
    live_command_failure = _live_routstr_tee_attestation_command_failure(tee)
    boundary_failure = _client_confidentiality_boundary_failure()
    if live_command_failure:
        tee["local_verification"] = _unverified_routstr_tee_verification(
            live_command_failure
        )
    elif boundary_failure:
        tee["local_verification"] = _unverified_routstr_tee_verification(
            boundary_failure
        )
    elif (
        has_evidence_source
        and bool(tee.get("evidence_format"))
        and (
            cap_attestation_mode
            or bool((tee.get("hpke_key_config") or {}).get("available"))
        )
    ):
        ready_routing_policy = _routing_policy_snapshot(routstr_tee_ready=True)
        if cap_attestation_mode:
            local_verification = _verify_cap_routstr_tee_evidence(
                tee=tee,
                routing_policy=ready_routing_policy,
            )
        else:
            local_verification = _verify_routstr_tee_evidence(
                tee=tee,
                routing_policy=ready_routing_policy,
            )
        tee["local_verification"], _ = _accepted_routstr_tee_verification(
            local_verification
        )
        if tee["local_verification"].get("verified") is True:
            routing_policy = ready_routing_policy
    else:
        tee["local_verification"] = _unverified_routstr_tee_verification(
            str(tee.get("failure_reason") or "Routstr TEE evidence is unavailable")
        )
    return {
        "schema_version": "routstr-attestation-v1",
        "service": "routstr",
        "version": __version__,
        "generated_at": int(time.time()),
        "routing_policy": routing_policy,
        "routing_policy_digest": _sha256_json(routing_policy),
        "tee": _public_tee_statement(tee),
    }
