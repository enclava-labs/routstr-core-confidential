from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import io
import ipaddress
import json
import secrets
import shlex
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import urljoin, urlparse

import httpx

from ..core.logging import redact_sensitive_text
from ..core.policy_secrets import inline_policy_secret_violations
from .base import ConfidentialityStatus
from .ehbp import parse_ehbp_key_config
from .strict_json import require_canonical_json, sha256_json_payload

if TYPE_CHECKING:
    from .base import BaseUpstreamProvider, ConfidentialVerifierPolicy


SUPPORTED_TINFOIL_ATTESTATION_FORMATS = {
    "https://tinfoil.sh/predicate/sev-snp-guest/v2",
    "https://tinfoil.sh/predicate/tdx-guest/v2",
    "https://tinfoil.sh/predicate/snp-tdx-multiplatform/v1",
    "https://tinfoil.sh/predicate/hardware-measurements/v1",
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
REQUIRED_PRIVATEMODE_PROOF_CLAIMS = (
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
MAX_TINFOIL_ATTESTATION_REPORT_BYTES = 4 * 1024 * 1024
DEFAULT_VERIFIER_MAX_AGE_SECONDS = 300
RELEASE_DIGEST_EXACT_POLICY_KEYS = (
    "expected_release_digest",
    "release_digest",
)
RELEASE_DIGEST_ALLOWED_POLICY_KEYS = (
    "allowed_release_digest",
    "allowed_release_digests",
)
RELEASE_DIGEST_POLICY_KEYS = (
    RELEASE_DIGEST_EXACT_POLICY_KEYS + RELEASE_DIGEST_ALLOWED_POLICY_KEYS
)
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
TINFOIL_TRANSPORT_SECURITY_POLICY_KEYS = (
    "transport_security",
    "tinfoil_transport_security",
)
KNOWN_TINFOIL_MODEL_TARGETS = {
    "llama3-3-70b.tinfoil.containers.tinfoil.dev": {
        "models": {"llama3-3-70b", "llama-3.3-70b"},
        "repos": {"tinfoilsh/confidential-llama3-3-70b"},
    },
    "deepseek-v4-pro-inf12.tinfoil.containers.tinfoil.dev": {
        "models": {"deepseek-v4-pro"},
        "repos": {"tinfoilsh/confidential-deepseek-v4-pro"},
    },
    "kimi-k2-6.inf13.tinfoil.sh": {
        "models": {"kimi-k2-6"},
        "repos": {"tinfoilsh/confidential-kimi-k2-6-b200"},
    },
    "gemma4-31b-1.inf10.tinfoil.sh": {
        "models": {"gemma4-31b", "gemma-4-31b"},
        "repos": {"tinfoilsh/confidential-gemma4-31b"},
    },
    "qwen3-vl-30b.inf10.tinfoil.sh": {
        "models": {"qwen3-vl-30b"},
        "repos": {"tinfoilsh/confidential-qwen3-vl-30b"},
    },
    "glm-5-1.tinfoil.containers.tinfoil.dev": {
        "models": {"glm-5-1"},
        "repos": {"tinfoilsh/confidential-glm5-1"},
    },
    "gpt-oss-120b-1.inf10.tinfoil.sh": {
        "models": {"gpt-oss-120b"},
        "repos": {"tinfoilsh/confidential-gpt-oss-120b"},
    },
    "gpt-oss-safeguard-120b.tinfoil.containers.tinfoil.dev": {
        "models": {"gpt-oss-safeguard-120b"},
        "repos": {"tinfoilsh/confidential-gpt-oss-safeguard-120b"},
    },
}
EHBP_ENCLAVE_MEASUREMENT_EXACT_POLICY_KEYS = (
    "expected_enclave_measurement_fingerprint",
    "enclave_measurement_fingerprint",
)
EHBP_ENCLAVE_MEASUREMENT_ALLOWED_POLICY_KEYS = (
    "allowed_enclave_measurement_fingerprints",
    "allowed_enclave_measurement_fingerprint",
    "allowed_enclave_measurements",
)
EHBP_CODE_MEASUREMENT_EXACT_POLICY_KEYS = (
    "expected_code_measurement_fingerprint",
    "code_measurement_fingerprint",
)
EHBP_CODE_MEASUREMENT_ALLOWED_POLICY_KEYS = (
    "allowed_code_measurement_fingerprints",
    "allowed_code_measurement_fingerprint",
    "allowed_code_measurements",
)
EHBP_MEASUREMENT_DIGEST_POLICY_KEYS = (
    EHBP_ENCLAVE_MEASUREMENT_EXACT_POLICY_KEYS
    + EHBP_ENCLAVE_MEASUREMENT_ALLOWED_POLICY_KEYS
    + EHBP_CODE_MEASUREMENT_EXACT_POLICY_KEYS
    + EHBP_CODE_MEASUREMENT_ALLOWED_POLICY_KEYS
)
EHBP_MEASUREMENT_DIGEST_POLICY_GROUPS = (
    (
        "enclave_measurement_fingerprint",
        EHBP_ENCLAVE_MEASUREMENT_EXACT_POLICY_KEYS,
        EHBP_ENCLAVE_MEASUREMENT_ALLOWED_POLICY_KEYS,
    ),
    (
        "code_measurement_fingerprint",
        EHBP_CODE_MEASUREMENT_EXACT_POLICY_KEYS,
        EHBP_CODE_MEASUREMENT_ALLOWED_POLICY_KEYS,
    ),
)
VERIFIER_DIGEST_POLICY_KEYS = (
    "verifier_command_digest",
    "verifier_binary_digest",
)
PRIVATEMODE_MANIFEST_DIGEST_POLICY_KEYS = ("manifest_digest", "manifestDigest")
PRIVATEMODE_PROXY_IMAGE_DIGEST_POLICY_KEYS = (
    "proxy_image_digest",
    "proxyImageDigest",
)
PRIVATEMODE_PROXY_BINARY_DIGEST_POLICY_KEYS = (
    "proxy_binary_digest",
    "proxyBinaryDigest",
)
PRIVATEMODE_ARTIFACT_DIGEST_POLICY_KEY_GROUPS = (
    PRIVATEMODE_MANIFEST_DIGEST_POLICY_KEYS,
    PRIVATEMODE_PROXY_IMAGE_DIGEST_POLICY_KEYS,
    PRIVATEMODE_PROXY_BINARY_DIGEST_POLICY_KEYS,
)
PRIVATEMODE_MANIFEST_LOG_DIR_POLICY_KEYS = ("manifest_log_dir", "manifestLogDir")
PRIVATEMODE_MANIFEST_IDENTITY_POLICY_KEYS = (
    PRIVATEMODE_MANIFEST_DIGEST_POLICY_KEYS + PRIVATEMODE_MANIFEST_LOG_DIR_POLICY_KEYS
)
PRIVATEMODE_PROXY_BINARY_PATH_POLICY_KEYS = (
    "proxy_binary_path",
    "proxyBinaryPath",
)
PRIVATEMODE_EXPECTED_WORKLOAD_SAN_POLICY_KEYS = (
    "expected_workload_sans",
    "expectedWorkloadSANs",
)
PRIVATEMODE_EXPECTED_WORKLOAD_ID_POLICY_KEYS = (
    "expected_workload_ids",
    "expectedWorkloadIDs",
)
PRIVATEMODE_EXPECTED_WORKLOAD_POLICY_KEYS = (
    PRIVATEMODE_EXPECTED_WORKLOAD_SAN_POLICY_KEYS
    + PRIVATEMODE_EXPECTED_WORKLOAD_ID_POLICY_KEYS
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
PRIVATEMODE_PROXY_ARTIFACT_POLICY_KEYS = (
    PRIVATEMODE_PROXY_IMAGE_DIGEST_POLICY_KEYS
    + PRIVATEMODE_PROXY_BINARY_DIGEST_POLICY_KEYS
)
PRIVATEMODE_COMPONENT_DIGEST_POLICY_KEYS = {
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
}
PRIVATEMODE_COMPONENT_STRING_POLICY_KEYS = {
    "trust_tier": (
        "expected_trust_tier",
        "allowed_trust_tiers",
    ),
    "gpu_attestation_policy": (
        "expected_gpu_attestation_policy",
        "allowed_gpu_attestation_policies",
    ),
}
PRIVATEMODE_TRUST_TIER = "app-e2ee"
PRIVATEMODE_GPU_ATTESTATION_POLICY = "nvidia-ocsp-good-only"
PRIVATEMODE_REQUIRED_COMPONENT_DIGEST_CLAIMS = (
    "coordinator_measurement",
    "secret_service_measurement",
    "ai_worker_measurement",
    "key_release_binding",
)
PRIVATEMODE_KEY_RELEASE_BINDING_CLAIMS = (
    "attested_workload_policy_digest",
    "expected_workload_identity_digest",
    "model_workload_binding_digest",
    "manifest_digest",
    "mesh_ca_digest",
    "secret_service_certificate_digest",
    "inference_secret_id_digest",
    "nvidia_ocsp_policy_mac_digest",
)


def _sha256_bytes_digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_file_digest(path: str) -> str:
    return _sha256_bytes_digest(Path(path).read_bytes())


def _stderr_failure_detail(stderr: bytes) -> str:
    if not stderr:
        return ""
    return f": stderr redacted ({len(stderr)} bytes, {_sha256_bytes_digest(stderr)})"


def _exception_failure_reason(prefix: str, exc: BaseException) -> str:
    detail = str(redact_sensitive_text(str(exc))).strip()
    return f"{prefix}: {detail}" if detail else prefix


def _sha256_json_digest(value: object) -> str:
    return _sha256_bytes_digest(sha256_json_payload(value))


def _require_canonical_json(value: object, label: str) -> None:
    require_canonical_json(value, label)


def _json_constant_rejecter(label: str) -> Callable[[str], None]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} must not contain {value}")

    return reject_constant


def _loads_strict_json(data: str | bytes, label: str) -> Any:
    return json.loads(data, parse_constant=_json_constant_rejecter(label))


def _new_verification_nonce() -> str:
    return secrets.token_urlsafe(32)


def _url_userinfo_violation(
    value: object,
    label: str,
    *,
    require_https: bool = True,
) -> str | None:
    if not isinstance(value, str):
        return None
    url = value.strip()
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except Exception:
        return f"{label} must be an absolute URL"
    if not parsed.scheme or not parsed.netloc:
        return f"{label} must be an absolute URL"
    if "@" in parsed.netloc:
        return f"{label} must not include credentials"
    if require_https and parsed.scheme != "https":
        return f"{label} must use https"
    if not parsed.hostname:
        return f"{label} must include a host"
    try:
        parsed.port
    except ValueError:
        return f"{label} must include a valid port"
    return None


def _provider_base_url_violation(
    provider: "BaseUpstreamProvider",
    *,
    require_https: bool = True,
) -> str | None:
    violation = _url_userinfo_violation(
        provider.base_url,
        "provider base_url",
        require_https=require_https,
    )
    if violation:
        return violation
    parsed = urlparse(provider.base_url)
    if not parsed.scheme or not parsed.netloc:
        return "provider base_url must be an absolute URL"
    if require_https and parsed.scheme != "https":
        return "provider base_url must use https"
    return None


def _is_ppq_owned_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    normalized = hostname.rstrip(".").lower()
    return normalized == "ppq.ai" or normalized.endswith(".ppq.ai")


def _ppq_private_base_url_violation(provider: "BaseUpstreamProvider") -> str | None:
    violation = _provider_base_url_violation(provider)
    if violation:
        return violation
    parsed = urlparse(provider.base_url)
    if not _is_ppq_owned_host(parsed.hostname):
        return "PPQ private base_url host must be ppq.ai or a ppq.ai subdomain"
    path_segments = {segment.lower() for segment in parsed.path.split("/") if segment}
    if "private" not in path_segments:
        return "PPQ private base_url must include a private path segment"
    return None


def _ppq_private_policy_url_violations(
    policy: dict[str, Any],
    *keys: str,
) -> list[str]:
    violations: list[str] = []
    for key in keys:
        value = policy.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        parsed = urlparse(value.strip())
        if (
            not parsed.scheme
            or not parsed.netloc
            or parsed.scheme != "https"
            or parsed.username
            or parsed.password
            or not parsed.hostname
        ):
            continue
        path_segments = {
            segment.lower() for segment in parsed.path.split("/") if segment
        }
        if "private" not in path_segments:
            violations.append(f"{key} must include a private path segment")
    return violations


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


def _privatemode_proxy_base_url_violation(
    provider: "BaseUpstreamProvider",
) -> str | None:
    violation = _provider_base_url_violation(provider, require_https=False)
    if violation:
        return violation.replace("provider base_url", "Privatemode proxy base_url")
    parsed = urlparse(provider.base_url)
    if parsed.scheme not in {"http", "https"}:
        return "Privatemode proxy base_url must use http or https"
    if not _is_loopback_hostname(parsed.hostname):
        return "Privatemode proxy base_url must use a loopback host"
    return None


def _policy_url_userinfo_violations(
    policy: dict[str, Any],
    *keys: str,
) -> list[str]:
    violations: list[str] = []
    for key in keys:
        violation = _url_userinfo_violation(policy.get(key), key)
        if violation:
            violations.append(violation)
    return violations


def _origin_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("provider base_url must be an absolute URL")
    hostname = _normalized_hostname(parsed.hostname)
    if not hostname:
        raise ValueError("provider base_url must include a host")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("provider base_url must include a valid port") from exc
    if port is None or (parsed.scheme == "https" and port == 443):
        return f"{parsed.scheme}://{hostname}"
    return f"{parsed.scheme}://{hostname}:{port}"


def _normalized_hostname(value: str | None) -> str:
    return (value or "").strip().rstrip(".").lower()


def _public_attestation_target_host_violation(
    value: object,
    label: str,
    *,
    absolute_url: bool = False,
) -> str | None:
    if absolute_url:
        host, issue = _absolute_https_hostname(value, label)
    else:
        host, issue = _host_identity_hostname(value, label)
    if issue or not host:
        return None
    if host == "localhost":
        return f"{label} must not use localhost or a non-global IP address"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return None
    if not address.is_global:
        return f"{label} must not use localhost or a non-global IP address"
    return None


def _absolute_https_hostname(
    value: object, label: str
) -> tuple[str | None, str | None]:
    if not isinstance(value, str):
        return None, None
    raw_url = value.strip()
    if not raw_url:
        return None, None
    parsed = urlparse(raw_url)
    if not parsed.scheme or not parsed.netloc:
        return None, f"{label} must be an absolute URL"
    if "@" in parsed.netloc:
        return None, f"{label} must not include credentials"
    if parsed.scheme != "https":
        return None, f"{label} must use https"
    hostname = _normalized_hostname(parsed.hostname)
    if not hostname:
        return None, f"{label} must include a host"
    try:
        parsed.port
    except ValueError:
        return None, f"{label} must include a valid port"
    return hostname, None


def _absolute_https_origin(
    value: object,
    label: str,
) -> tuple[str | None, str | None]:
    if not isinstance(value, str):
        return None, None
    raw_url = value.strip()
    if not raw_url:
        return None, None
    parsed = urlparse(raw_url)
    if not parsed.scheme or not parsed.netloc:
        return None, f"{label} must be an absolute URL"
    if "@" in parsed.netloc:
        return None, f"{label} must not include credentials"
    if parsed.scheme != "https":
        return None, f"{label} must use https"
    hostname = _normalized_hostname(parsed.hostname)
    if not hostname:
        return None, f"{label} must include a host"
    try:
        port = parsed.port
    except ValueError:
        return None, f"{label} must include a valid port"
    if port is None or port == 443:
        return f"https://{hostname}", None
    return f"https://{hostname}:{port}", None


def _host_identity_hostname(
    value: object,
    label: str,
) -> tuple[str | None, str | None]:
    if not isinstance(value, str):
        return None, None
    raw_host = value.strip()
    if not raw_host:
        return None, None
    if "://" in raw_host:
        return _absolute_https_hostname(raw_host, label)
    if any(separator in raw_host for separator in "/?#"):
        return None, f"{label} must be a host or absolute URL"
    parsed = urlparse(f"https://{raw_host}")
    if "@" in parsed.netloc:
        return None, f"{label} must not include credentials"
    hostname = _normalized_hostname(parsed.hostname)
    if not hostname:
        return None, f"{label} must include a host"
    try:
        parsed.port
    except ValueError:
        return None, f"{label} must include a valid port"
    return hostname, None


def _policy_url_host_violations(
    policy: dict[str, Any],
    provider_base_url: str,
    *keys: str,
) -> list[str]:
    provider_host, provider_host_issue = _absolute_https_hostname(
        provider_base_url,
        "provider base_url",
    )
    if provider_host_issue or not provider_host:
        return []
    provider_origin, provider_origin_issue = _absolute_https_origin(
        provider_base_url,
        "provider base_url",
    )

    violations: list[str] = []
    for key in keys:
        value = policy.get(key)
        host, issue = _absolute_https_hostname(value, key)
        if issue or not host:
            continue
        if host != provider_host:
            violations.append(f"{key} host must match provider base_url host")
            continue
        origin, origin_issue = _absolute_https_origin(value, key)
        if (
            not provider_origin_issue
            and provider_origin
            and not origin_issue
            and origin
            and origin != provider_origin
        ):
            violations.append(f"{key} origin must match provider base_url origin")
    return violations


def _policy_host_identity_violations(
    policy: dict[str, Any],
    provider_base_url: str,
    *keys: str,
) -> list[str]:
    provider_host, provider_host_issue = _absolute_https_hostname(
        provider_base_url,
        "provider base_url",
    )
    if provider_host_issue or not provider_host:
        return []

    violations: list[str] = []
    for key in keys:
        value = policy.get(key)
        host, issue = _host_identity_hostname(value, key)
        if issue:
            violations.append(issue)
            continue
        if host and host != provider_host:
            violations.append(f"{key} must match provider base_url host")
    return violations


def _provider_policy(
    provider: "BaseUpstreamProvider",
) -> "ConfidentialVerifierPolicy | None":
    policy_getter = getattr(provider, "confidentiality_policy", None)
    if not callable(policy_getter):
        return None
    return policy_getter()


def _policy_mode_issue(
    policy: "ConfidentialVerifierPolicy",
    *,
    expected_mode: str,
    provider_label: str,
) -> str | None:
    mode = str(policy.mode or "").strip()
    if mode != expected_mode:
        return (
            f"confidentiality mode must be {expected_mode} for "
            f"{provider_label} provider"
        )
    return None


def _status_with_policy(
    provider: "BaseUpstreamProvider",
    *,
    mode: str,
    failure_reason: str,
    evidence_digest: str | None = None,
) -> ConfidentialityStatus:
    current = provider.confidentiality_status()
    policy = _provider_policy(provider)
    if not policy and (policy_error := provider.confidentiality_policy_error()):
        failure_reason = policy_error
    return current.copy(
        update={
            "enabled": True,
            "verified": False,
            "mode": mode,
            "verified_at": None,
            "expires_at": None,
            "failure_reason": failure_reason,
            "verifier": None,
            "policy_digest": policy.digest if policy else None,
            "model_ids": list(policy.model_ids) if policy else current.model_ids,
            "model_id_prefixes": (
                list(policy.model_id_prefixes)
                if policy
                else current.model_id_prefixes
            ),
            "evidence_digest": evidence_digest,
            "verified_claims": {},
        }
    )


def _verified_status_with_policy(
    provider: "BaseUpstreamProvider",
    *,
    mode: str,
    verifier: str,
    evidence_digest: str,
    verified_claims: dict[str, Any],
    verified_at: int,
    expires_at: int | None = None,
) -> ConfidentialityStatus:
    current = provider.confidentiality_status()
    policy = _provider_policy(provider)
    return current.copy(
        update={
            "enabled": True,
            "verified": True,
            "mode": mode,
            "verified_at": verified_at,
            "expires_at": expires_at,
            "failure_reason": None,
            "verifier": verifier,
            "policy_digest": policy.digest if policy else current.policy_digest,
            "model_ids": list(policy.model_ids) if policy else current.model_ids,
            "model_id_prefixes": (
                list(policy.model_id_prefixes)
                if policy
                else current.model_id_prefixes
            ),
            "evidence_digest": evidence_digest,
            "verified_claims": verified_claims,
        }
    )


def _timeout_seconds(policy: "ConfidentialVerifierPolicy") -> float:
    raw_timeout = policy.policy.get("timeout_seconds", 10.0)
    try:
        timeout = float(raw_timeout)
    except (TypeError, ValueError):
        return 10.0
    return max(0.1, min(timeout, 60.0))


def _policy_string(policy: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = policy.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _policy_string_values(policy: dict[str, Any], *keys: str) -> list[str]:
    for key in keys:
        value = policy.get(key)
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        if isinstance(value, list):
            values = [
                item.strip() for item in value if isinstance(item, str) and item.strip()
            ]
            if values:
                return values
    return []


def _normalized_string_policy_values(
    policy: dict[str, Any],
    *keys: str,
) -> list[str]:
    values: list[str] = []
    for value in _policy_values_for_present_keys(policy, *keys):
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if normalized and normalized not in values:
            values.append(normalized)
    return sorted(values)


def _strict_normalized_string_policy_values(
    policy: dict[str, Any],
    *keys: str,
) -> list[str] | None:
    values: list[str] = []
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
        values.extend(normalized)
    deduplicated: dict[str, str] = {}
    for value in values:
        deduplicated.setdefault(value.lower(), value)
    return sorted(deduplicated.values())


def _string_list_policy_violations(
    policy: dict[str, Any],
    *keys: str,
) -> list[str]:
    violations: list[str] = []
    for key in keys:
        if key not in policy:
            continue
        value = policy[key]
        if not isinstance(value, list):
            violations.append(f"{key} must be a list of non-empty strings")
            continue
        if not value:
            violations.append(f"{key} must be a list of non-empty strings")
            continue
        if any(not isinstance(item, str) or not item.strip() for item in value):
            violations.append(f"{key} must be a list of non-empty strings")
            continue
        normalized = [item.strip() for item in value]
        if len({item.lower() for item in normalized}) != len(normalized):
            violations.append(f"{key} must not contain duplicates")
    return violations


def _string_list_policy_alias_violations(
    policy: dict[str, Any],
    label: str,
    *keys: str,
) -> list[str]:
    expected_values: list[list[str]] = []
    for key in keys:
        if key not in policy:
            continue
        values = _normalized_string_policy_values(policy, key)
        if values:
            expected_values.append(values)
    if not expected_values:
        return []
    first = expected_values[0]
    if any(values != first for values in expected_values[1:]):
        return [f"{label} aliases must match"]
    return []


def _non_empty_string_policy_violations(
    policy: dict[str, Any],
    *keys: str,
) -> list[str]:
    violations: list[str] = []
    for key in keys:
        if key not in policy:
            continue
        value = policy[key]
        if not isinstance(value, str) or not value.strip():
            violations.append(f"{key} must be a non-empty string")
    return violations


def _string_policy_field_violations(
    policy: dict[str, Any],
    *keys: str,
) -> list[str]:
    violations: list[str] = []
    for key in keys:
        if key not in policy:
            continue
        value = policy[key]
        if not isinstance(value, str):
            violations.append(f"{key} must be a string")
            continue
        if not value.strip():
            violations.append(f"{key} must be a non-empty string")
    return violations


def _policy_raw_values(policy: dict[str, Any], *keys: str) -> list[object]:
    for key in keys:
        if key not in policy:
            continue
        value = policy[key]
        if isinstance(value, list):
            return list(value)
        return [value]
    return []


def _policy_values_for_present_keys(policy: dict[str, Any], *keys: str) -> list[object]:
    values: list[object] = []
    for key in keys:
        if key not in policy:
            continue
        value = policy[key]
        if isinstance(value, list):
            values.extend(value)
        else:
            values.append(value)
    return values


def _is_sha256_digest_value(value: str) -> bool:
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def _is_template_placeholder_digest_value(value: object) -> bool:
    if not isinstance(value, str):
        return False
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    return (
        len(digest) in {64, 96}
        and all(char in "0123456789abcdef" for char in digest)
        and len(set(digest)) == 1
    )


def _sha256_digest_hex(value: str) -> str:
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    return digest


def _is_native_sev_snp_measurement_value(value: str) -> bool:
    digest = value.strip().lower()
    return (
        not digest.startswith("sha256:")
        and len(digest) == 96
        and all(char in "0123456789abcdef" for char in digest)
    )


def _is_measurement_fingerprint_value(value: str) -> bool:
    return _is_sha256_digest_value(value) or _is_native_sev_snp_measurement_value(value)


def _measurement_fingerprint_hex(value: str) -> str:
    digest = value.strip().lower()
    if _is_sha256_digest_value(digest):
        return _sha256_digest_hex(digest)
    if _is_native_sev_snp_measurement_value(digest):
        return hashlib.sha256(bytes.fromhex(digest)).hexdigest()
    return digest


def _normalized_sha256_digest_policy_values(
    policy: dict[str, Any],
    *keys: str,
) -> list[str]:
    values: list[str] = []
    for value in _policy_values_for_present_keys(policy, *keys):
        if not isinstance(value, str) or not _is_sha256_digest_value(value):
            continue
        digest = _sha256_digest_hex(value)
        if digest not in values:
            values.append(digest)
    return values


def _normalized_measurement_fingerprint_policy_values(
    policy: dict[str, Any],
    *keys: str,
) -> list[str]:
    values: list[str] = []
    for value in _policy_values_for_present_keys(policy, *keys):
        if not isinstance(value, str) or not _is_measurement_fingerprint_value(value):
            continue
        digest = _measurement_fingerprint_hex(value)
        if digest not in values:
            values.append(digest)
    return values


def _sha256_digest_policy_consistency_issue(
    policy: dict[str, Any],
    *,
    label: str,
    exact_keys: tuple[str, ...],
    allowed_keys: tuple[str, ...],
) -> str | None:
    exact_values = _normalized_sha256_digest_policy_values(policy, *exact_keys)
    allowed_values = _normalized_sha256_digest_policy_values(policy, *allowed_keys)
    if len(exact_values) > 1:
        return f"{label} expected aliases must match"
    if (
        exact_values
        and allowed_values
        and not set(exact_values).issubset(allowed_values)
    ):
        return f"{label} expected value must be allowed"
    return None


def _measurement_fingerprint_policy_consistency_issue(
    policy: dict[str, Any],
    *,
    label: str,
    exact_keys: tuple[str, ...],
    allowed_keys: tuple[str, ...],
) -> str | None:
    exact_values = _normalized_measurement_fingerprint_policy_values(
        policy,
        *exact_keys,
    )
    allowed_values = _normalized_measurement_fingerprint_policy_values(
        policy,
        *allowed_keys,
    )
    if len(exact_values) > 1:
        return f"{label} expected aliases must match"
    if (
        exact_values
        and allowed_values
        and not set(exact_values).issubset(allowed_values)
    ):
        return f"{label} expected value must be allowed"
    return None


def _is_prefixed_sha256_digest_value(value: str) -> bool:
    digest = value.strip().lower()
    if not digest.startswith("sha256:"):
        return False
    digest = digest.removeprefix("sha256:")
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def _release_identity_policy_issue(
    policy: "ConfidentialVerifierPolicy",
    *,
    missing_reason: str | None = None,
) -> str | None:
    return _release_identity_policy_dict_issue(
        policy.policy,
        missing_reason=missing_reason,
    )


def _release_identity_policy_dict_issue(
    policy: dict[str, Any],
    *,
    missing_reason: str | None = None,
) -> str | None:
    raw_values = _policy_values_for_present_keys(
        policy,
        *RELEASE_DIGEST_POLICY_KEYS,
    )
    if not raw_values:
        return missing_reason
    if any(
        not isinstance(value, str) or not _is_sha256_digest_value(value)
        for value in raw_values
    ):
        return "release digest policy values must be sha256 digests"
    if any(_is_template_placeholder_digest_value(value) for value in raw_values):
        return "release digest policy values must not be placeholder digests"
    return _sha256_digest_policy_consistency_issue(
        policy,
        label="release_digest",
        exact_keys=RELEASE_DIGEST_EXACT_POLICY_KEYS,
        allowed_keys=RELEASE_DIGEST_ALLOWED_POLICY_KEYS,
    )


def _measurement_identity_policy_issue(
    policy: "ConfidentialVerifierPolicy",
) -> str | None:
    return _measurement_identity_policy_dict_issue(policy.policy)


def _measurement_identity_policy_dict_issue(policy: dict[str, Any]) -> str | None:
    raw_values: list[object] = []
    for key in EHBP_MEASUREMENT_DIGEST_POLICY_KEYS:
        raw_values.extend(_policy_raw_values(policy, key))
    if any(
        not isinstance(value, str) or not _is_measurement_fingerprint_value(value)
        for value in raw_values
    ):
        return (
            "measurement fingerprint policy values must be sha256 digests "
            "or native SEV-SNP measurements"
        )
    if any(_is_template_placeholder_digest_value(value) for value in raw_values):
        return "measurement fingerprint policy values must not be placeholder digests"
    for label, exact_keys, allowed_keys in EHBP_MEASUREMENT_DIGEST_POLICY_GROUPS:
        if issue := _measurement_fingerprint_policy_consistency_issue(
            policy,
            label=label,
            exact_keys=exact_keys,
            allowed_keys=allowed_keys,
        ):
            return issue
    return None


def _policy_bool_issue(policy: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key in policy and not isinstance(policy[key], bool):
            return f"{key} must be a boolean"
    return None


def _policy_bool_enabled(policy: dict[str, Any], *keys: str) -> bool:
    return any(policy.get(key) is True for key in keys)


def _has_ehbp_artifact_identity_pin(policy: dict[str, Any]) -> bool:
    return bool(
        _policy_values_for_present_keys(policy, *RELEASE_DIGEST_POLICY_KEYS)
        or _policy_values_for_present_keys(
            policy,
            *(
                EHBP_CODE_MEASUREMENT_EXACT_POLICY_KEYS
                + EHBP_CODE_MEASUREMENT_ALLOWED_POLICY_KEYS
            ),
        )
    )


def _tinfoil_model_attestation_target_values(
    policy: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    present = [
        (key, policy[key])
        for key in TINFOIL_MODEL_ATTESTATION_TARGET_POLICY_KEYS
        if key in policy
    ]
    if not present:
        return {}, ["model_attestation_targets is required"]

    parsed_values: list[dict[str, dict[str, Any]]] = []
    violations: list[str] = []
    for key, raw_targets in present:
        if not isinstance(raw_targets, dict) or not raw_targets:
            violations.append(f"{key} must be a non-empty object")
            continue

        parsed: dict[str, dict[str, Any]] = {}
        seen_model_ids: set[str] = set()
        for raw_model_id, raw_target in raw_targets.items():
            if not isinstance(raw_model_id, str) or not raw_model_id.strip():
                violations.append(f"{key} keys must be non-empty model strings")
                continue
            model_id = raw_model_id.strip()
            normalized_model_id = model_id.lower()
            if normalized_model_id in seen_model_ids:
                violations.append(f"{key} contains duplicate model target {model_id}")
                continue
            seen_model_ids.add(normalized_model_id)
            if not isinstance(raw_target, dict):
                violations.append(f"{key} for {model_id} must be an object")
                continue
            target = dict(raw_target)
            target_violations = _string_policy_field_violations(
                target,
                "host",
                "expected_enclave_host",
                "attestation_url",
                "hpke_keys_url",
                "repo",
                "expected_repo",
            )
            target_violations.extend(
                _string_list_policy_alias_violations(
                    target,
                    "enclave host",
                    "host",
                    "expected_enclave_host",
                )
            )
            target_violations.extend(
                _string_list_policy_alias_violations(
                    target,
                    "repo",
                    "repo",
                    "expected_repo",
                )
            )
            target_violations.extend(
                _policy_url_userinfo_violations(
                    target,
                    "attestation_url",
                    "hpke_keys_url",
                )
            )
            for host_key in ("host", "expected_enclave_host"):
                _host, issue = _host_identity_hostname(target.get(host_key), host_key)
                if issue:
                    target_violations.append(issue)
                elif public_host_violation := _public_attestation_target_host_violation(
                    target.get(host_key),
                    host_key,
                ):
                    target_violations.append(public_host_violation)
            for url_key in ("attestation_url", "hpke_keys_url"):
                if public_url_host_violation := _public_attestation_target_host_violation(
                    target.get(url_key),
                    url_key,
                    absolute_url=True,
                ):
                    target_violations.append(public_url_host_violation)
            if not (
                _policy_string(target, "host", "expected_enclave_host")
                or _policy_string(target, "attestation_url")
            ):
                target_violations.append("host or attestation_url is required")
            if not _policy_string(target, "repo", "expected_repo"):
                target_violations.append("repo or expected_repo is required")
            if release_issue := _release_identity_policy_dict_issue(target):
                target_violations.append(release_issue)
            if not _release_identity_policy_values_for_policy_dict(target):
                target_violations.append(
                    "expected_release_digest or allowed_release_digests is required"
                )
            if measurement_issue := _measurement_identity_policy_dict_issue(target):
                target_violations.append(measurement_issue)
            if not _has_ehbp_artifact_identity_pin(target):
                target_violations.append(
                    "expected_release_digest, allowed_release_digests, or "
                    "code measurement identity pin is required"
                )
            if target_violations:
                violations.extend(
                    f"{key} for {model_id}: {violation}"
                    for violation in target_violations
                )
                continue
            parsed[model_id] = target
        if parsed:
            parsed_values.append(dict(sorted(parsed.items())))

    if len(parsed_values) > 1:
        first = parsed_values[0]
        if any(value != first for value in parsed_values[1:]):
            violations.append("model_attestation_targets aliases must match")
    return (parsed_values[0] if parsed_values else {}, violations)


def _tinfoil_model_target_origin(
    target_policy: dict[str, Any],
) -> tuple[str | None, str | None]:
    for key in ("attestation_url", "hpke_keys_url"):
        origin, issue = _absolute_https_origin(target_policy.get(key), key)
        if issue:
            return None, issue
        if origin:
            return origin, None
    host, issue = _host_identity_hostname(
        _policy_string(target_policy, "host", "expected_enclave_host"),
        "host",
    )
    if issue:
        return None, issue
    if host:
        return f"https://{host}", None
    return None, "host or attestation_url is required"


def _tinfoil_router_origin_reuse_violations(
    policy: "ConfidentialVerifierPolicy",
    targets: dict[str, dict[str, Any]],
) -> list[str]:
    router_origin, provider_issue = _absolute_https_origin(
        policy.base_url,
        "provider base_url",
    )
    if provider_issue or not router_origin:
        return []

    violations: list[str] = []
    for model_id, target_policy in targets.items():
        target_origin, target_issue = _tinfoil_model_target_origin(target_policy)
        if target_issue or not target_origin:
            continue
        if target_origin == router_origin:
            violations.append(
                f"model_attestation_targets for {model_id} must use a distinct "
                "model enclave origin, not the Tinfoil router origin"
            )
    return violations


def _normalized_model_identity(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    normalized = normalized.removeprefix("tinfoil/")
    normalized = normalized.removeprefix("private/")
    return "".join(char for char in normalized if char.isalnum() or char == "-")


def _tinfoil_known_target_violations(
    targets: dict[str, dict[str, Any]],
) -> list[str]:
    violations: list[str] = []
    for model_id, target in targets.items():
        target_origin, target_issue = _tinfoil_model_target_origin(target)
        if target_issue or not target_origin:
            continue
        parsed = urlparse(target_origin)
        host = parsed.hostname.lower() if parsed.hostname else ""
        known_target = KNOWN_TINFOIL_MODEL_TARGETS.get(host)
        if not known_target:
            continue
        selected_model = _normalized_model_identity(model_id)
        known_models = {
            _normalized_model_identity(model)
            for model in known_target.get("models", set())
        }
        if selected_model not in known_models:
            violations.append(
                f"model_attestation_targets for {model_id} host {host} is a "
                "known Tinfoil target for a different model"
            )
            continue
        target_repo = _policy_string(target, "repo", "expected_repo")
        known_repos = known_target.get("repos", set())
        if known_repos and target_repo not in known_repos:
            violations.append(
                f"model_attestation_targets for {model_id} host {host} repo "
                f"{target_repo or '<missing>'} does not match known Tinfoil "
                "target repos"
            )
    return violations


def _tinfoil_model_attestation_policy_violations(
    policy: "ConfidentialVerifierPolicy",
) -> list[str]:
    violations: list[str] = []
    if not policy.model_ids and not policy.model_id_prefixes:
        return violations
    if bool_issue := _policy_bool_issue(
        policy.policy,
        *TINFOIL_REQUIRE_MODEL_ATTESTATION_POLICY_KEYS,
    ):
        violations.append(bool_issue)
        return violations
    if not _policy_bool_enabled(
        policy.policy,
        *TINFOIL_REQUIRE_MODEL_ATTESTATION_POLICY_KEYS,
    ):
        violations.append(
            "Tinfoil require_model_attestations must be true for selected model routing"
        )
        return violations
    if policy.model_id_prefixes:
        violations.append(
            "Tinfoil model_id_prefixes are not supported with "
            "full model attestations; use exact model_ids"
        )
    if not policy.model_ids:
        violations.append("Tinfoil model_ids are required with full model attestations")

    targets, target_violations = _tinfoil_model_attestation_target_values(policy.policy)
    violations.extend(target_violations)
    if target_violations:
        return violations

    selected_models = set(policy.model_ids)
    for model_id in sorted(selected_models):
        if model_id not in targets:
            violations.append(f"model_attestation_targets is missing {model_id}")
    for model_id in targets:
        if model_id not in selected_models:
            violations.append(f"model_attestation_targets contains unselected model {model_id}")
    violations.extend(_tinfoil_router_origin_reuse_violations(policy, targets))
    violations.extend(_tinfoil_known_target_violations(targets))
    return violations


def _ppq_private_model_selector_policy_violations(
    policy: "ConfidentialVerifierPolicy",
) -> list[str]:
    violations: list[str] = []
    if policy.model_id_prefixes:
        violations.append(
            "PPQ private model_id_prefixes are not supported; use exact model_ids"
        )
    if not policy.model_ids:
        violations.append("PPQ private model_ids are required")
        return violations

    invalid_model_ids = [
        model_id for model_id in policy.model_ids if not model_id.startswith("private/")
    ]
    if invalid_model_ids:
        violations.append("PPQ private model_ids must start with private/")
    return violations


def _tinfoil_model_attestations_required(policy: dict[str, Any]) -> bool:
    return _policy_bool_enabled(policy, *TINFOIL_REQUIRE_MODEL_ATTESTATION_POLICY_KEYS)


def _tinfoil_target_urls(target_policy: dict[str, Any]) -> tuple[str, str]:
    attestation_url = _policy_string(target_policy, "attestation_url")
    hpke_keys_url = _policy_string(target_policy, "hpke_keys_url")
    origin: str | None = ""
    if attestation_url:
        origin, origin_issue = _absolute_https_origin(
            attestation_url,
            "attestation_url",
        )
        if origin_issue:
            raise ValueError(origin_issue)
    if not origin and hpke_keys_url:
        origin, origin_issue = _absolute_https_origin(
            hpke_keys_url,
            "hpke_keys_url",
        )
        if origin_issue:
            raise ValueError(origin_issue)
    if not origin:
        host, host_issue = _host_identity_hostname(
            _policy_string(target_policy, "host", "expected_enclave_host"),
            "host",
        )
        if host_issue:
            raise ValueError(host_issue)
        if not host:
            raise ValueError("host or attestation_url is required")
        origin = f"https://{host}"

    if not attestation_url:
        attestation_url = urljoin(origin, "/.well-known/tinfoil-attestation")
    if not hpke_keys_url:
        hpke_keys_url = urljoin(origin, "/.well-known/hpke-keys")
    return attestation_url, hpke_keys_url


async def _collect_tinfoil_model_attestations(
    *,
    client: httpx.AsyncClient,
    policy: "ConfidentialVerifierPolicy",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not _tinfoil_model_attestations_required(policy.policy):
        return [], []

    targets, violations = _tinfoil_model_attestation_target_values(policy.policy)
    if violations:
        raise ValueError("; ".join(violations))

    payloads: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for model_id in sorted(policy.model_ids):
        target_policy = targets[model_id]
        attestation_url, hpke_keys_url = _tinfoil_target_urls(target_policy)
        attestation_response = await client.get(attestation_url)
        attestation_response.raise_for_status()
        attestation_json = attestation_response.json()
        _require_canonical_json(
            attestation_json,
            f"Tinfoil model attestation response for {model_id}",
        )
        (
            attestation_format,
            attestation_body_bytes,
            attestation_report_bytes,
            attestation_body,
        ) = _parse_tinfoil_attestation(attestation_json)
        _require_tinfoil_ehbp_attestation_format(attestation_format)

        hpke_response = await client.get(hpke_keys_url)
        hpke_response.raise_for_status()
        hpke_keys = hpke_response.content
        if not hpke_keys:
            raise ValueError(f"Tinfoil model HPKE key response is empty for {model_id}")
        ehbp_config = parse_ehbp_key_config(hpke_keys)
        hpke_key_config_b64 = base64.b64encode(hpke_keys).decode("ascii")
        model_evidence = {
            "model_id": model_id,
            "attestation_format": attestation_format,
            "attestation_body_gzip_digest": _sha256_bytes_digest(
                attestation_body_bytes
            ),
            "attestation_report_digest": _sha256_bytes_digest(
                attestation_report_bytes
            ),
            "attestation_wire_body_digest": _sha256_bytes_digest(
                attestation_body.encode("utf-8")
            ),
            "attestation_url": attestation_url,
            "hpke_keys_digest": _sha256_bytes_digest(hpke_keys),
            "ehbp": ehbp_config.verified_claims(),
            "hpke_keys_url": hpke_keys_url,
        }
        evidence.append(model_evidence)
        payloads.append(
            {
                "model_id": model_id,
                "policy": target_policy,
                "attestation_url": attestation_url,
                "attestation": {
                    "format": attestation_format,
                    "body": attestation_body,
                    "body_encoding": "base64+gzip",
                    "body_gzip_digest": _sha256_bytes_digest(attestation_body_bytes),
                    "report_b64": base64.b64encode(attestation_report_bytes).decode(
                        "ascii"
                    ),
                    "report_digest": _sha256_bytes_digest(attestation_report_bytes),
                },
                "hpke_keys_url": hpke_keys_url,
                "hpke_key_config_b64": hpke_key_config_b64,
                "hpke_public_key_hex": ehbp_config.public_key.hex(),
                "ehbp": ehbp_config.verified_claims(),
                "evidence_digest": _sha256_json_digest(model_evidence),
            }
        )
    return payloads, evidence


def _verifier_command_digest_policy_issue(
    policy: "ConfidentialVerifierPolicy",
) -> str | None:
    digest_values: list[str] = []
    for key in VERIFIER_DIGEST_POLICY_KEYS:
        if key not in policy.policy:
            continue
        value = policy.policy[key]
        if (
            not isinstance(value, str)
            or not value.strip()
            or not _is_prefixed_sha256_digest_value(value)
        ):
            return f"{key} must be a sha256 digest"
        if _is_template_placeholder_digest_value(value):
            return f"{key} must not be a placeholder digest"
        digest_values.append(value.strip().lower())
    if not digest_values:
        return "verifier_command_digest is required"
    if len(set(digest_values)) > 1:
        return "verifier_command_digest aliases must match"
    return None


def _prefixed_digest_policy_violations(
    policy: dict[str, Any],
    *keys: str,
) -> list[str]:
    violations: list[str] = []
    for key in keys:
        if key not in policy:
            continue
        value = policy[key]
        if not isinstance(value, str) or not _is_prefixed_sha256_digest_value(value):
            violations.append(f"{key} must be a sha256 digest")
        elif _is_template_placeholder_digest_value(value):
            violations.append(f"{key} must not be a placeholder digest")
    return violations


def _prefixed_digest_alias_policy_violations(
    policy: dict[str, Any],
    *alias_groups: tuple[str, ...],
) -> list[str]:
    violations: list[str] = []
    for keys in alias_groups:
        label = keys[0]
        values = [policy[key] for key in keys if key in policy]
        if any(
            not isinstance(value, str) or not _is_prefixed_sha256_digest_value(value)
            for value in values
        ):
            violations.append(f"{label} must be a sha256 digest")
            continue
        if any(_is_template_placeholder_digest_value(value) for value in values):
            violations.append(f"{label} must not be a placeholder digest")
            continue
        normalized = {value.strip().lower() for value in values}
        if len(normalized) > 1:
            violations.append(f"{label} aliases must match")
    return violations


def _privatemode_component_policy_violations(policy: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for claim_name, keys in PRIVATEMODE_COMPONENT_DIGEST_POLICY_KEYS.items():
        values = _policy_values_for_present_keys(policy, *keys)
        if any(
            not isinstance(value, str) or not _is_prefixed_sha256_digest_value(value)
            for value in values
        ):
            violations.append(f"{claim_name} policy values must be sha256 digests")
            continue
        if any(_is_template_placeholder_digest_value(value) for value in values):
            violations.append(
                f"{claim_name} policy values must not be placeholder digests"
            )
    for claim_name, keys in PRIVATEMODE_COMPONENT_STRING_POLICY_KEYS.items():
        values = _policy_values_for_present_keys(policy, *keys)
        if any(not isinstance(value, str) or not value.strip() for value in values):
            violations.append(
                f"{claim_name} policy values must contain only non-empty strings"
            )
    trust_tiers = _normalized_string_policy_values(policy, "expected_trust_tier")
    if trust_tiers != [PRIVATEMODE_TRUST_TIER]:
        violations.append(f"expected_trust_tier must be {PRIVATEMODE_TRUST_TIER}")
    gpu_policies = _normalized_string_policy_values(
        policy,
        "expected_gpu_attestation_policy",
    )
    if gpu_policies != [PRIVATEMODE_GPU_ATTESTATION_POLICY]:
        violations.append(
            "expected_gpu_attestation_policy must be "
            f"{PRIVATEMODE_GPU_ATTESTATION_POLICY}"
        )
    violations.extend(_privatemode_component_policy_consistency_violations(policy))
    return violations


def _privatemode_workload_policy_violations(policy: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    violations.extend(
        _string_list_policy_violations(
            policy,
            *PRIVATEMODE_EXPECTED_WORKLOAD_POLICY_KEYS,
        )
    )
    violations.extend(
        _string_list_policy_alias_violations(
            policy,
            "expected_workload_sans",
            *PRIVATEMODE_EXPECTED_WORKLOAD_SAN_POLICY_KEYS,
        )
    )
    violations.extend(
        _string_list_policy_alias_violations(
            policy,
            "expected_workload_ids",
            *PRIVATEMODE_EXPECTED_WORKLOAD_ID_POLICY_KEYS,
        )
    )
    if not _normalized_string_policy_values(
        policy,
        *PRIVATEMODE_EXPECTED_WORKLOAD_POLICY_KEYS,
    ):
        violations.append("expected_workload_sans or expected_workload_ids is required")
    return violations


def _privatemode_expected_workload_identity_digest(
    policy: dict[str, Any],
) -> str | None:
    workload_ids = _strict_normalized_string_policy_values(
        policy,
        *PRIVATEMODE_EXPECTED_WORKLOAD_ID_POLICY_KEYS,
    )
    workload_sans = _strict_normalized_string_policy_values(
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


def _privatemode_model_workload_binding_values(
    policy: dict[str, Any],
) -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    present = [
        (key, policy[key])
        for key in PRIVATEMODE_MODEL_WORKLOAD_BINDING_POLICY_KEYS
        if key in policy
    ]
    if not present:
        return {}, ["model_workload_bindings is required"]

    parsed_values: list[dict[str, dict[str, list[str]]]] = []
    violations: list[str] = []
    for key, raw_bindings in present:
        if not isinstance(raw_bindings, dict) or not raw_bindings:
            violations.append(f"{key} must be a non-empty object")
            continue

        parsed: dict[str, dict[str, list[str]]] = {}
        seen_model_ids: set[str] = set()
        for raw_model_id, raw_binding in raw_bindings.items():
            if not isinstance(raw_model_id, str) or not raw_model_id.strip():
                violations.append(f"{key} keys must be non-empty model strings")
                continue
            model_id = raw_model_id.strip()
            normalized_model_id = model_id.lower()
            if normalized_model_id in seen_model_ids:
                violations.append(f"{key} contains duplicate model binding {model_id}")
                continue
            seen_model_ids.add(normalized_model_id)
            if not isinstance(raw_binding, dict):
                violations.append(f"{key} for {model_id} must be an object")
                continue
            binding = dict(raw_binding)
            binding_violations = _string_list_policy_violations(
                binding,
                *PRIVATEMODE_MODEL_WORKLOAD_SAN_POLICY_KEYS,
                *PRIVATEMODE_MODEL_WORKLOAD_ID_POLICY_KEYS,
            )
            binding_violations.extend(
                _string_list_policy_alias_violations(
                    binding,
                    "workload_sans",
                    *PRIVATEMODE_MODEL_WORKLOAD_SAN_POLICY_KEYS,
                )
            )
            binding_violations.extend(
                _string_list_policy_alias_violations(
                    binding,
                    "workload_ids",
                    *PRIVATEMODE_MODEL_WORKLOAD_ID_POLICY_KEYS,
                )
            )
            if binding_violations:
                violations.extend(
                    f"{key} for {model_id}: {violation}"
                    for violation in binding_violations
                )
                continue

            workload_sans = _normalized_string_policy_values(
                binding,
                *PRIVATEMODE_MODEL_WORKLOAD_SAN_POLICY_KEYS,
            )
            workload_ids = _normalized_string_policy_values(
                binding,
                *PRIVATEMODE_MODEL_WORKLOAD_ID_POLICY_KEYS,
            )
            if not workload_sans and not workload_ids:
                violations.append(
                    f"{key} for {model_id} requires workload_sans or workload_ids"
                )
                continue
            parsed[model_id] = {
                "workload_ids": workload_ids,
                "workload_sans": workload_sans,
            }
        if parsed:
            parsed_values.append(dict(sorted(parsed.items())))

    if len(parsed_values) > 1:
        first = parsed_values[0]
        if any(value != first for value in parsed_values[1:]):
            violations.append("model_workload_bindings aliases must match")
    return (parsed_values[0] if parsed_values else {}, violations)


def _privatemode_model_workload_binding_digest(policy: dict[str, Any]) -> str:
    bindings, _violations = _privatemode_model_workload_binding_values(policy)
    return _sha256_json_digest(bindings)


def _privatemode_key_release_binding_digest(claims: dict[str, Any]) -> str:
    return _sha256_json_digest(
        {key: claims.get(key) for key in PRIVATEMODE_KEY_RELEASE_BINDING_CLAIMS}
    )


def _privatemode_model_workload_policy_violations(
    policy: "ConfidentialVerifierPolicy",
) -> list[str]:
    violations: list[str] = []
    raw_policy = policy.policy
    if policy.model_id_prefixes:
        violations.append(
            "Privatemode model_id_prefixes are not supported; use exact model_ids "
            "with model_workload_bindings"
        )
    if not policy.model_ids:
        violations.append(
            "Privatemode model_ids are required with model_workload_bindings"
        )
        return violations
    invalid_model_ids = [
        model_id for model_id in policy.model_ids if not model_id.startswith("privatemode/")
    ]
    if invalid_model_ids:
        violations.append("Privatemode model_ids must start with privatemode/")

    workload_violations = _privatemode_workload_policy_violations(raw_policy)
    violations.extend(workload_violations)
    if workload_violations:
        return violations

    bindings, binding_violations = _privatemode_model_workload_binding_values(
        raw_policy
    )
    violations.extend(binding_violations)
    if binding_violations:
        return violations

    expected_sans = set(
        _normalized_string_policy_values(
            raw_policy,
            *PRIVATEMODE_EXPECTED_WORKLOAD_SAN_POLICY_KEYS,
        )
    )
    expected_ids = set(
        _normalized_string_policy_values(
            raw_policy,
            *PRIVATEMODE_EXPECTED_WORKLOAD_ID_POLICY_KEYS,
        )
    )
    selected_models = set(policy.model_ids)
    bound_sans: set[str] = set()
    bound_ids: set[str] = set()

    for model_id in sorted(selected_models):
        if model_id not in bindings:
            violations.append(f"model_workload_bindings is missing {model_id}")

    for model_id, binding in bindings.items():
        if model_id not in selected_models:
            violations.append(
                f"model_workload_bindings contains unselected model {model_id}"
            )
        binding_sans = set(binding["workload_sans"])
        binding_ids = set(binding["workload_ids"])
        if binding_sans - expected_sans:
            violations.append(
                f"model_workload_bindings for {model_id} references "
                "workload_sans not listed in expected_workload_sans"
            )
        if binding_ids - expected_ids:
            violations.append(
                f"model_workload_bindings for {model_id} references "
                "workload_ids not listed in expected_workload_ids"
            )
        bound_sans.update(binding_sans)
        bound_ids.update(binding_ids)

    if expected_sans - bound_sans:
        violations.append("expected_workload_sans contains unbound workloads")
    if expected_ids - bound_ids:
        violations.append("expected_workload_ids contains unbound workloads")
    return violations


def _privatemode_component_policy_consistency_violations(
    policy: dict[str, Any],
) -> list[str]:
    violations: list[str] = []
    for claim_name, keys in PRIVATEMODE_COMPONENT_DIGEST_POLICY_KEYS.items():
        expected_values = _policy_values_for_present_keys(policy, keys[0])
        allowed_values = _policy_values_for_present_keys(policy, keys[1])
        if not expected_values or not allowed_values:
            continue
        if any(
            not isinstance(value, str) for value in expected_values + allowed_values
        ):
            continue
        expected = {value.strip().lower() for value in expected_values if isinstance(value, str) and value.strip()}
        allowed = {value.strip().lower() for value in allowed_values if isinstance(value, str) and value.strip()}
        if expected and allowed and not expected.issubset(allowed):
            violations.append(f"{claim_name} expected value must be allowed")

    for claim_name, keys in PRIVATEMODE_COMPONENT_STRING_POLICY_KEYS.items():
        expected_values = _policy_values_for_present_keys(policy, keys[0])
        allowed_values = _policy_values_for_present_keys(policy, keys[1])
        if not expected_values or not allowed_values:
            continue
        if any(
            not isinstance(value, str) for value in expected_values + allowed_values
        ):
            continue
        expected = {value.strip() for value in expected_values if isinstance(value, str) and value.strip()}
        allowed = {value.strip() for value in allowed_values if isinstance(value, str) and value.strip()}
        if expected and allowed and not expected.issubset(allowed):
            violations.append(f"{claim_name} expected value must be allowed")
    return violations


def _policy_command(policy: "ConfidentialVerifierPolicy", *keys: str) -> object | None:
    for key in keys:
        value = policy.policy.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            return value
    return None


def _verifier_command_policy_issue(
    policy: "ConfidentialVerifierPolicy",
    *keys: str,
) -> str | None:
    present_commands: list[tuple[str, tuple[str, ...]]] = []
    for key in keys:
        if key not in policy.policy:
            continue
        value = policy.policy[key]
        if not isinstance(value, str | list):
            return f"{key} must be a command string or array"
        try:
            argv = _command_argv(value)
        except ValueError as exc:
            return str(exc)
        present_commands.append((key, tuple(argv)))
    if not present_commands:
        return f"{' or '.join(keys)} is required"
    if len({argv for _, argv in present_commands}) > 1:
        return "verifier_command aliases must match"
    return None


def _command_argv(command: object) -> list[str]:
    if isinstance(command, str):
        argv = shlex.split(command)
    elif isinstance(command, list):
        if any(not isinstance(item, str) or not item.strip() for item in command):
            raise ValueError(
                "confidential verifier command entries must be non-empty strings"
            )
        argv = [item.strip() for item in command]
    else:
        argv = []
    if not argv:
        raise ValueError("confidential verifier command is empty")
    return argv


def _verifier_artifact_path_policy_issue(
    policy: "ConfidentialVerifierPolicy",
    command: object,
) -> str | None:
    try:
        argv = _command_argv(command)
    except ValueError as exc:
        return str(exc)

    artifact_path = policy.policy.get("verifier_artifact_path")
    if artifact_path is None:
        return None
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        return "verifier_artifact_path must be a non-empty string"
    if artifact_path.strip() not in argv:
        return (
            "verifier_artifact_path must match verifier command executable or argument"
        )
    return None


def _verified_command_digest(
    policy: "ConfidentialVerifierPolicy",
    command: object,
) -> tuple[str, str]:
    expected_digest = _policy_string(
        policy.policy,
        "verifier_command_digest",
        "verifier_binary_digest",
    )
    if not expected_digest:
        raise ValueError("verifier_command_digest is required")
    if not _is_prefixed_sha256_digest_value(expected_digest):
        raise ValueError("verifier_command_digest must be a sha256 digest")
    if _is_template_placeholder_digest_value(expected_digest):
        raise ValueError("verifier_command_digest must not be a placeholder digest")

    argv = _command_argv(command)
    configured_artifact_path = _policy_string(policy.policy, "verifier_artifact_path")
    if configured_artifact_path:
        if configured_artifact_path not in argv:
            raise ValueError(
                "verifier_artifact_path must match verifier command executable or argument"
            )
        artifact_path = configured_artifact_path
    else:
        artifact_path = shutil.which(argv[0]) or argv[0]
    actual_digest = _sha256_file_digest(artifact_path)
    if actual_digest != expected_digest:
        raise ValueError(
            "verifier command digest mismatch: "
            f"expected {expected_digest}, got {actual_digest}"
        )
    return actual_digest, artifact_path


async def _run_confidential_verifier_command(
    command: object,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run an isolated verifier command that reads JSON stdin and writes JSON stdout."""
    argv = _command_argv(command)
    try:
        encoded_payload = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "confidential verifier payload JSON must be canonical"
        ) from exc

    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(encoded_payload),
            timeout=timeout_seconds,
        )
    except TimeoutError as exc:
        proc.kill()
        await proc.communicate()
        raise ValueError("confidential verifier command timed out") from exc

    if proc.returncode != 0:
        raise ValueError(
            f"confidential verifier command failed with exit code {proc.returncode}"
            f"{_stderr_failure_detail(stderr)}"
        )
    try:
        result = _loads_strict_json(
            stdout.decode("utf-8"),
            "confidential verifier command output JSON",
        )
    except Exception as exc:
        raise ValueError(
            f"confidential verifier command did not return JSON: {exc}"
        ) from exc
    if not isinstance(result, dict):
        raise ValueError("confidential verifier command output must be a JSON object")
    return result


def _decompress_tinfoil_attestation_report(attestation_body_bytes: bytes) -> bytes:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(attestation_body_bytes)) as gzip_file:
            report = gzip_file.read(MAX_TINFOIL_ATTESTATION_REPORT_BYTES + 1)
    except Exception as exc:
        raise ValueError("Tinfoil attestation body is not valid gzip") from exc

    if not report:
        raise ValueError("Tinfoil attestation report is empty")
    if len(report) > MAX_TINFOIL_ATTESTATION_REPORT_BYTES:
        raise ValueError("Tinfoil attestation report is too large")
    return report


def _parse_tinfoil_attestation(attestation: object) -> tuple[str, bytes, bytes, str]:
    if not isinstance(attestation, dict):
        raise ValueError("Tinfoil attestation response is not a JSON object")

    attestation_format = attestation.get("format")
    if not isinstance(attestation_format, str):
        raise ValueError("Tinfoil attestation response is missing format")
    if attestation_format not in SUPPORTED_TINFOIL_ATTESTATION_FORMATS:
        raise ValueError(
            f"unsupported Tinfoil attestation format: {attestation_format}"
        )

    attestation_body = attestation.get("body")
    if not isinstance(attestation_body, str):
        raise ValueError("Tinfoil attestation response is missing body")
    try:
        attestation_body_bytes = base64.b64decode(attestation_body, validate=True)
    except Exception as exc:
        raise ValueError("Tinfoil attestation body is not valid base64") from exc
    if not attestation_body_bytes:
        raise ValueError("Tinfoil attestation body is empty")

    attestation_report_bytes = _decompress_tinfoil_attestation_report(
        attestation_body_bytes
    )

    return (
        attestation_format,
        attestation_body_bytes,
        attestation_report_bytes,
        attestation_body,
    )


def _require_tinfoil_ehbp_attestation_format(attestation_format: str) -> None:
    if not attestation_format.endswith("/v2"):
        raise ValueError(
            "Tinfoil EHBP requires a v2 attestation predicate with an attested "
            "HPKE public key"
        )


async def refresh_tinfoil_status(
    provider: "BaseUpstreamProvider",
) -> ConfidentialityStatus:
    """Probe Tinfoil evidence endpoints, but do not claim crypto verification."""
    policy = _provider_policy(provider)
    if not policy:
        return _status_with_policy(
            provider,
            mode="tinfoil",
            failure_reason="attestation policy missing for Tinfoil verifier",
        )
    if mode_issue := _policy_mode_issue(
        policy,
        expected_mode="tinfoil",
        provider_label="tinfoil",
    ):
        return _status_with_policy(
            provider,
            mode="tinfoil",
            failure_reason=mode_issue,
        )

    violations = []
    violations.extend(inline_policy_secret_violations(policy.policy))
    if provider_base_url_violation := _provider_base_url_violation(provider):
        violations.append(provider_base_url_violation)
    violations.extend(
        _string_policy_field_violations(
            policy.policy,
            "attestation_url",
            "hpke_keys_url",
            "enclave_host",
            "expected_enclave_host",
        )
    )
    violations.extend(
        _policy_url_userinfo_violations(
            policy.policy,
            "attestation_url",
            "hpke_keys_url",
        )
    )
    violations.extend(
        _policy_url_host_violations(
            policy.policy,
            provider.base_url,
            "attestation_url",
            "hpke_keys_url",
        )
    )
    violations.extend(
        _policy_host_identity_violations(
            policy.policy,
            provider.base_url,
            "enclave_host",
            "expected_enclave_host",
        )
    )
    violations.extend(
        _string_list_policy_alias_violations(
            policy.policy,
            "enclave host",
            "enclave_host",
            "expected_enclave_host",
        )
    )
    violations.extend(
        _non_empty_string_policy_violations(
            policy.policy,
            "repo",
            "expected_repo",
        )
    )
    violations.extend(
        _string_list_policy_alias_violations(
            policy.policy,
            "repo",
            "repo",
            "expected_repo",
        )
    )
    if violations:
        return _status_with_policy(
            provider,
            mode="tinfoil",
            failure_reason="; ".join(violations),
        )
    if not _policy_string(policy.policy, "repo", "expected_repo"):
        return _status_with_policy(
            provider,
            mode="tinfoil",
            failure_reason="repo or expected_repo is required for Tinfoil verifier",
        )
    release_policy_issue = _release_identity_policy_issue(
        policy,
        missing_reason=_missing_tinfoil_release_digest_reason(),
    )
    if release_policy_issue:
        return _status_with_policy(
            provider,
            mode="tinfoil",
            failure_reason=release_policy_issue,
        )
    if measurement_policy_issue := _measurement_identity_policy_issue(policy):
        return _status_with_policy(
            provider,
            mode="tinfoil",
            failure_reason=measurement_policy_issue,
        )
    if verifier_command_issue := _verifier_command_policy_issue(
        policy,
        "verifier_command",
        "tinfoil_verifier_command",
    ):
        return _status_with_policy(
            provider,
            mode="tinfoil",
            failure_reason=verifier_command_issue,
        )
    verifier_command = _policy_command(
        policy,
        "verifier_command",
        "tinfoil_verifier_command",
    )
    if verifier_command is None:
        return _status_with_policy(
            provider,
            mode="tinfoil",
            failure_reason="verifier_command or tinfoil_verifier_command is required",
        )
    if verifier_digest_issue := _verifier_command_digest_policy_issue(policy):
        return _status_with_policy(
            provider,
            mode="tinfoil",
            failure_reason=verifier_digest_issue,
        )
    if artifact_path_issue := _verifier_artifact_path_policy_issue(
        policy,
        verifier_command,
    ):
        return _status_with_policy(
            provider,
            mode="tinfoil",
            failure_reason=artifact_path_issue,
        )
    model_attestation_violations = _tinfoil_model_attestation_policy_violations(policy)
    if model_attestation_violations:
        return _status_with_policy(
            provider,
            mode="tinfoil",
            failure_reason="; ".join(model_attestation_violations),
        )

    origin = _origin_url(provider.base_url)
    attestation_url = str(
        policy.policy.get("attestation_url")
        or urljoin(origin, "/.well-known/tinfoil-attestation")
    )
    hpke_keys_url = str(
        policy.policy.get("hpke_keys_url") or urljoin(origin, "/.well-known/hpke-keys")
    )

    try:
        async with httpx.AsyncClient(timeout=_timeout_seconds(policy)) as client:
            attestation_response = await client.get(attestation_url)
            attestation_response.raise_for_status()
            attestation_json = attestation_response.json()
            _require_canonical_json(
                attestation_json,
                "Tinfoil attestation response",
            )
            (
                attestation_format,
                attestation_body_bytes,
                attestation_report_bytes,
                attestation_body,
            ) = _parse_tinfoil_attestation(attestation_json)
            _require_tinfoil_ehbp_attestation_format(attestation_format)

            hpke_response = await client.get(hpke_keys_url)
            hpke_response.raise_for_status()
            hpke_keys = hpke_response.content
            if not hpke_keys:
                raise ValueError("Tinfoil HPKE key response is empty")
            ehbp_config = parse_ehbp_key_config(hpke_keys)
            hpke_key_config_b64 = base64.b64encode(hpke_keys).decode("ascii")
            (
                model_attestation_payloads,
                model_attestation_evidence,
            ) = await _collect_tinfoil_model_attestations(
                client=client,
                policy=policy,
            )

    except Exception as exc:
        return _status_with_policy(
            provider,
            mode="tinfoil",
            failure_reason=_exception_failure_reason(
                "Tinfoil evidence probe failed",
                exc,
            ),
        )

    evidence_digest = _sha256_json_digest(
        {
            "attestation_format": attestation_format,
            "attestation_body_gzip_digest": _sha256_bytes_digest(
                attestation_body_bytes
            ),
            "attestation_report_digest": _sha256_bytes_digest(attestation_report_bytes),
            "attestation_wire_body_digest": _sha256_bytes_digest(
                attestation_body.encode("utf-8")
            ),
            "attestation_url": attestation_url,
            "hpke_keys_digest": _sha256_bytes_digest(hpke_keys),
            "ehbp": ehbp_config.verified_claims(),
            "hpke_keys_url": hpke_keys_url,
            "model_attestations": model_attestation_evidence,
        }
    )
    try:
        verifier_command_digest, verifier_artifact_path = _verified_command_digest(
            policy,
            verifier_command,
        )
    except Exception as exc:
        return _status_with_policy(
            provider,
            mode="tinfoil",
            failure_reason=_exception_failure_reason("Tinfoil verifier failed", exc),
            evidence_digest=evidence_digest,
        )

    verification_nonce = _new_verification_nonce()
    verifier_payload = {
        "schema_version": "routstr-confidential-verifier-v1",
        "provider_type": provider.provider_type,
        "mode": "tinfoil",
        "base_url": provider.base_url,
        "origin": origin,
        "model_ids": list(policy.model_ids),
        "policy_digest": policy.digest,
        "policy": policy.policy,
        "attestation_url": attestation_url,
        "attestation": {
            "format": attestation_format,
            "body": attestation_body,
            "body_encoding": "base64+gzip",
            "body_gzip_digest": _sha256_bytes_digest(attestation_body_bytes),
            "report_b64": base64.b64encode(attestation_report_bytes).decode("ascii"),
            "report_digest": _sha256_bytes_digest(attestation_report_bytes),
        },
        "hpke_keys_url": hpke_keys_url,
        "hpke_key_config_b64": hpke_key_config_b64,
        "model_attestations": model_attestation_payloads,
        "evidence_digest": evidence_digest,
        "verification_nonce": verification_nonce,
        "verifier_command_digest": verifier_command_digest,
        "verifier_artifact_path": verifier_artifact_path,
    }

    try:
        verifier_result = await _run_confidential_verifier_command(
            verifier_command,
            verifier_payload,
            _timeout_seconds(policy),
        )
        verified_claims = _validate_tinfoil_verifier_result(
            verifier_result=verifier_result,
            policy=policy,
            evidence_digest=evidence_digest,
            verification_nonce=verification_nonce,
            attestation_format=attestation_format,
            attestation_report_digest=_sha256_bytes_digest(attestation_report_bytes),
            ehbp_public_key_hex=ehbp_config.public_key.hex(),
            ehbp_key_config_b64=hpke_key_config_b64,
            ehbp_claims=ehbp_config.verified_claims(),
            model_attestations=model_attestation_payloads,
        )
        verified_at = _int_result_value(
            verifier_result.get("verified_at"), "verified_at"
        )
        if verified_at is None:
            verified_at = int(time.time())
        expires_at = _bounded_verifier_expires_at(
            verifier_result=verifier_result,
            verified_at=verified_at,
            policy=policy,
        )
    except Exception as exc:
        return _status_with_policy(
            provider,
            mode="tinfoil",
            failure_reason=_exception_failure_reason("Tinfoil verifier failed", exc),
            evidence_digest=evidence_digest,
        )

    verifier = str(verifier_result.get("verifier") or "").strip()
    return _verified_status_with_policy(
        provider,
        mode="tinfoil",
        verifier=verifier,
        evidence_digest=verified_claims["runtime_evidence_digest"],
        verified_claims=verified_claims,
        verified_at=verified_at,
        expires_at=expires_at,
    )


def _int_result_value(value: object, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"{label} must be an integer")


def _bounded_verifier_expires_at(
    *,
    verifier_result: dict[str, Any],
    verified_at: int,
    policy: "ConfidentialVerifierPolicy",
) -> int:
    explicit_expires_at = _int_result_value(
        verifier_result.get("expires_at"),
        "expires_at",
    )
    now = int(time.time())
    if verified_at > now:
        raise ValueError("verifier result verified_at is in the future")
    if explicit_expires_at is not None and explicit_expires_at <= now:
        raise ValueError("verifier result is expired")

    max_age_seconds = _int_policy_value(
        policy.policy,
        "max_verifier_age_seconds",
        "max_evidence_age_seconds",
    )
    if max_age_seconds is None:
        max_age_seconds = DEFAULT_VERIFIER_MAX_AGE_SECONDS
    if max_age_seconds <= 0:
        raise ValueError("max_verifier_age_seconds must be positive")

    bounded_expires_at = verified_at + max_age_seconds
    if bounded_expires_at <= now:
        raise ValueError("verifier result is stale")
    if explicit_expires_at is None:
        return bounded_expires_at
    return min(explicit_expires_at, bounded_expires_at)


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


def _require_verifier_payload_binding(
    *,
    verifier_result: dict[str, Any],
    claims: dict[str, Any],
    expected_policy_digest: str,
    expected_evidence_digest: str,
    expected_verification_nonce: str,
) -> None:
    expected_values = {
        "policy_digest": expected_policy_digest,
        "evidence_digest": expected_evidence_digest,
        "verification_nonce": expected_verification_nonce,
    }
    for key, expected_value in expected_values.items():
        values = _bound_result_values(verifier_result, claims, key)
        if not values or any(value != expected_value for value in values):
            raise ValueError(f"{key} does not match verifier payload")

    payload_claims = {
        "payload_policy_digest": expected_policy_digest,
        "payload_evidence_digest": expected_evidence_digest,
        "payload_verification_nonce": expected_verification_nonce,
    }
    for key, expected_value in payload_claims.items():
        values = _bound_result_values(verifier_result, claims, key)
        if any(value != expected_value for value in values):
            raise ValueError(f"{key} does not match verifier payload")
        claims[key] = expected_value


def _expected_ehbp_claim_values(
    policy: "ConfidentialVerifierPolicy",
) -> dict[str, list[str]]:
    return _expected_ehbp_claim_values_for_policy_dict(policy.policy)


def _expected_ehbp_claim_values_for_policy_dict(
    raw_policy: dict[str, Any],
) -> dict[str, list[str]]:
    expected: dict[str, list[str]] = {}

    enclave_measurements = _measurement_fingerprint_policy_values(
        _policy_string_values(
            raw_policy,
            *(
                EHBP_ENCLAVE_MEASUREMENT_EXACT_POLICY_KEYS
                + EHBP_ENCLAVE_MEASUREMENT_ALLOWED_POLICY_KEYS
            ),
        )
    )
    if enclave_measurements:
        expected["enclave_measurement_fingerprint"] = enclave_measurements

    code_measurements = _measurement_fingerprint_policy_values(
        _policy_string_values(
            raw_policy,
            *(
                EHBP_CODE_MEASUREMENT_EXACT_POLICY_KEYS
                + EHBP_CODE_MEASUREMENT_ALLOWED_POLICY_KEYS
            ),
        )
    )
    if code_measurements:
        expected["code_measurement_fingerprint"] = code_measurements

    release_digests = _release_identity_policy_values_for_policy_dict(raw_policy)
    if release_digests:
        expected["release_digest"] = _release_digest_policy_values(release_digests)

    return expected


def _release_identity_policy_values(policy: "ConfidentialVerifierPolicy") -> list[str]:
    return _release_identity_policy_values_for_policy_dict(policy.policy)


def _release_identity_policy_values_for_policy_dict(
    raw_policy: dict[str, Any],
) -> list[str]:
    values = _policy_raw_values(raw_policy, *RELEASE_DIGEST_POLICY_KEYS)
    return [
        value.strip()
        for value in values
        if isinstance(value, str) and _is_sha256_digest_value(value)
    ]


def _artifact_identity_policy_values(policy: "ConfidentialVerifierPolicy") -> list[str]:
    release_identity = _release_identity_policy_values(policy)
    if release_identity:
        return release_identity
    return _policy_string_values(
        policy.policy,
        *(
            EHBP_CODE_MEASUREMENT_EXACT_POLICY_KEYS
            + EHBP_CODE_MEASUREMENT_ALLOWED_POLICY_KEYS
        ),
    )


def _missing_artifact_identity_reason(provider_name: str) -> str:
    return (
        "expected_release_digest, allowed_release_digests, "
        "expected_code_measurement_fingerprint, or "
        "allowed_code_measurement_fingerprints is required "
        f"for {provider_name} verifier"
    )


def _missing_tinfoil_release_digest_reason() -> str:
    return (
        "expected_release_digest or allowed_release_digests is required "
        "for Tinfoil verifier"
    )


def _release_digest_policy_values(values: list[str]) -> list[str]:
    return _sha256_digest_policy_values(values)


def _sha256_digest_policy_values(values: list[str]) -> list[str]:
    allowed: list[str] = []
    for value in values:
        digest = value.strip().lower()
        if not digest:
            continue
        if digest not in allowed:
            allowed.append(digest)
        if digest.startswith("sha256:"):
            hex_digest = digest.removeprefix("sha256:")
            if hex_digest and hex_digest not in allowed:
                allowed.append(hex_digest)
        elif len(digest) == 64:
            prefixed_digest = f"sha256:{digest}"
            if prefixed_digest not in allowed:
                allowed.append(prefixed_digest)
    return allowed


def _measurement_fingerprint_policy_values(values: list[str]) -> list[str]:
    allowed: list[str] = []
    for value in values:
        digest = _measurement_fingerprint_hex(value)
        if not digest:
            continue
        if digest not in allowed:
            allowed.append(digest)
        prefixed_digest = f"sha256:{digest}"
        if prefixed_digest not in allowed:
            allowed.append(prefixed_digest)
    return allowed


def _expected_privatemode_claim_values(
    policy: "ConfidentialVerifierPolicy",
) -> dict[str, list[str]]:
    raw_policy = policy.policy
    expected: dict[str, list[str]] = {}

    claim_policy_keys = {
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
        "gpu_attestation_policy": (
            "expected_gpu_attestation_policy",
            "allowed_gpu_attestation_policies",
        ),
        "trust_tier": (
            "expected_trust_tier",
            "allowed_trust_tiers",
        ),
        "key_release_binding": (
            "expected_key_release_binding",
            "allowed_key_release_bindings",
        ),
    }
    for claim_name, policy_keys in claim_policy_keys.items():
        allowed_values = _policy_string_values(raw_policy, *policy_keys)
        if allowed_values:
            expected[claim_name] = allowed_values

    return expected


def _require_allowed_claim_values(
    claims: dict[str, Any],
    expected_claim_values: dict[str, list[str]] | None,
) -> None:
    for claim_name, allowed_values in (expected_claim_values or {}).items():
        if allowed_values and claims.get(claim_name) not in allowed_values:
            raise ValueError(f"{claim_name} claim does not match policy")


def _require_selected_model_claims(
    claims: dict[str, Any],
    selected_model_ids: list[str],
) -> None:
    value = claims.get("selected_model_ids")
    if not isinstance(value, list):
        raise ValueError("selected_model_ids claim is required")
    if not value:
        raise ValueError("selected_model_ids claim is required")
    if not all(isinstance(model_id, str) and model_id.strip() for model_id in value):
        raise ValueError("selected_model_ids claim must contain only non-empty strings")
    normalized_value = [model_id.strip() for model_id in value]
    normalized_selected_model_ids = [model_id.strip() for model_id in selected_model_ids]
    if len({model_id.lower() for model_id in normalized_value}) != len(
        normalized_value
    ):
        raise ValueError("selected_model_ids claim must not contain duplicates")
    if len(
        {model_id.lower() for model_id in normalized_selected_model_ids}
    ) != len(normalized_selected_model_ids):
        raise ValueError("selected model policy must not contain duplicates")
    if set(normalized_value) != set(normalized_selected_model_ids):
        raise ValueError("selected_model_ids claim does not match selected models")
    claims["selected_model_ids"] = list(normalized_value)


def _require_prefixed_sha256_claim(
    claims: dict[str, Any],
    claim_name: str,
    *,
    required: bool = True,
) -> None:
    value = claims.get(claim_name)
    if value is None or value == "":
        if required:
            raise ValueError(f"{claim_name} claim is required")
        return
    if not isinstance(value, str) or not _is_prefixed_sha256_digest_value(value):
        raise ValueError(f"{claim_name} claim must be a sha256 digest")
    if _is_template_placeholder_digest_value(value):
        raise ValueError(f"{claim_name} claim must not be a placeholder digest")


def _require_verification_steps(
    *,
    verifier_result: dict[str, Any],
    claims: dict[str, Any],
    required_steps: tuple[str, ...],
) -> None:
    raw_steps = verifier_result.get("verification_steps")
    if raw_steps is None:
        raw_steps = claims.get("verification_steps")
    if not isinstance(raw_steps, dict):
        raise ValueError("verification_steps must be a JSON object")
    if any(not isinstance(step, str) or not step.strip() for step in raw_steps):
        raise ValueError("verification_steps keys must be non-empty strings")
    if any(not isinstance(value, bool) for value in raw_steps.values()):
        raise ValueError("verification_steps values must be booleans")

    for step in required_steps:
        if raw_steps.get(step) is not True:
            raise ValueError(f"{step} verification step is required")

    claims["verification_steps"] = dict(raw_steps)


def _attested_ehbp_public_key_claim(
    claims: dict[str, Any],
    expected_public_key_hex: str,
) -> str:
    values: list[str] = []
    for claim_name in (
        "attested_hpke_public_key_hex",
        "hpke_public_key_hex",
        "hpke_public_key",
    ):
        if claim_name not in claims or claims.get(claim_name) is None:
            continue
        claim_value = claims.get(claim_name)
        if not isinstance(claim_value, str):
            raise ValueError("attested HPKE public key claim must be a string")
        normalized = claim_value.strip().lower()
        if normalized:
            values.append(normalized)

    if not values:
        raise ValueError("attested HPKE public key claim is required")
    if any(value != expected_public_key_hex for value in values):
        raise ValueError(
            "attested HPKE public key does not match fetched EHBP key config"
        )
    return values[0]


def _require_tls_public_key_binding_claim(claims: dict[str, Any]) -> None:
    normalized_values: list[str] = []
    for claim_name in ("tls_public_key_fingerprint_sha256", "tls_public_key"):
        if claim_name not in claims:
            continue
        claim_value = claims.get(claim_name)
        if not isinstance(claim_value, str) or not claim_value.strip():
            if claim_name == "tls_public_key_fingerprint_sha256":
                raise ValueError(
                    "TLS public key fingerprint claim must be a sha256 digest"
                )
            raise ValueError("TLS public key claim must be a sha256 digest")
        if not _is_sha256_digest_value(claim_value):
            if claim_name == "tls_public_key_fingerprint_sha256":
                raise ValueError(
                    "TLS public key fingerprint claim must be a sha256 digest"
                )
            raise ValueError("TLS public key claim must be a sha256 digest")
        normalized_values.append(_sha256_digest_hex(claim_value))
    if not normalized_values:
        raise ValueError("TLS public key binding claim is required")
    if len(set(normalized_values)) > 1:
        raise ValueError("TLS public key binding aliases must match")


def _tinfoil_tls_public_key_binding_required(
    policy: dict[str, Any],
    *,
    inherited_policy: dict[str, Any] | None = None,
) -> bool:
    for raw_policy in (policy, inherited_policy or {}):
        for key in TINFOIL_TRANSPORT_SECURITY_POLICY_KEYS:
            value = raw_policy.get(key)
            if not isinstance(value, str):
                continue
            if value.strip().lower() == "ehbp":
                return False
    return True


def _validate_tls_public_key_binding_claim(
    claims: dict[str, Any],
    *,
    required: bool,
) -> None:
    if required or any(
        claim_name in claims
        for claim_name in ("tls_public_key_fingerprint_sha256", "tls_public_key")
    ):
        _require_tls_public_key_binding_claim(claims)


def _validate_tinfoil_verifier_result(
    *,
    verifier_result: dict[str, Any],
    policy: "ConfidentialVerifierPolicy",
    evidence_digest: str,
    verification_nonce: str,
    attestation_format: str,
    attestation_report_digest: str,
    ehbp_public_key_hex: str,
    ehbp_key_config_b64: str,
    ehbp_claims: dict[str, object],
    model_attestations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    model_attestation_policy_violations = _tinfoil_model_attestation_policy_violations(
        policy
    )
    if model_attestation_policy_violations:
        raise ValueError("; ".join(model_attestation_policy_violations))

    claims = _validate_confidential_ehbp_verifier_result(
        verifier_result=verifier_result,
        policy_digest=policy.digest,
        evidence_digest=evidence_digest,
        verification_nonce=verification_nonce,
        ehbp_public_key_hex=ehbp_public_key_hex,
        ehbp_key_config_b64=ehbp_key_config_b64,
        ehbp_claims=ehbp_claims,
        expected_claims={
            "repo": _policy_string(policy.policy, "repo", "expected_repo"),
            "attestation_format": attestation_format,
            "attestation_report_digest": attestation_report_digest,
        },
        expected_claim_values=_expected_ehbp_claim_values(policy),
        require_tls_public_key_binding=_tinfoil_tls_public_key_binding_required(
            policy.policy,
        ),
    )
    _require_tinfoil_model_attestation_claims(
        claims=claims,
        policy=policy,
        model_attestations=model_attestations or [],
    )
    claims["runtime_evidence_digest"] = _ehbp_runtime_evidence_digest(claims)
    return claims


def _require_tinfoil_model_attestation_claims(
    *,
    claims: dict[str, Any],
    policy: "ConfidentialVerifierPolicy",
    model_attestations: list[dict[str, Any]],
    claim_key: str = "model_attestations",
) -> None:
    model_attestation_policy_violations = _tinfoil_model_attestation_policy_violations(
        policy
    )
    if model_attestation_policy_violations:
        raise ValueError("; ".join(model_attestation_policy_violations))

    if not _tinfoil_model_attestations_required(policy.policy):
        if claim_key in claims:
            raise ValueError(f"{claim_key} claim is not allowed by policy")
        return

    raw_model_claims = claims.get(claim_key)
    if not isinstance(raw_model_claims, dict) or not raw_model_claims:
        raise ValueError(f"{claim_key} claim is required")
    if len(model_attestations) != len(policy.model_ids):
        raise ValueError(f"{claim_key} evidence does not cover selected models")

    targets, violations = _tinfoil_model_attestation_target_values(policy.policy)
    if violations:
        raise ValueError("; ".join(violations))
    selected_models = set(policy.model_ids)
    if set(raw_model_claims) != selected_models:
        raise ValueError(f"{claim_key} claim does not match selected models")

    evidence_by_model = {
        str(item.get("model_id")): item
        for item in model_attestations
        if isinstance(item.get("model_id"), str)
    }
    if set(evidence_by_model) != selected_models:
        raise ValueError(f"{claim_key} evidence does not match selected models")

    for model_id in sorted(selected_models):
        model_claims = raw_model_claims.get(model_id)
        if not isinstance(model_claims, dict):
            raise ValueError(f"{claim_key} for {model_id} must be an object")
        evidence = evidence_by_model[model_id]
        expected_values = _expected_ehbp_claim_values_for_policy_dict(targets[model_id])
        _require_allowed_claim_values(model_claims, expected_values)
        if model_claims.get("repo") != _policy_string(
            targets[model_id],
            "repo",
            "expected_repo",
        ):
            raise ValueError(f"{claim_key} for {model_id} repo does not match policy")
        for claim_name, evidence_key in (
            ("attestation_format", ("attestation", "format")),
            ("attestation_report_digest", ("attestation", "report_digest")),
        ):
            expected_value: object = evidence
            for key in evidence_key:
                if not isinstance(expected_value, dict):
                    expected_value = None
                    break
                expected_value = expected_value.get(key)
            if model_claims.get(claim_name) != expected_value:
                raise ValueError(
                    f"{claim_key} for {model_id} {claim_name} "
                    "does not match evidence"
                )
        _attested_ehbp_public_key_claim(
            model_claims,
            str(evidence.get("hpke_public_key_hex") or ""),
        )
        for required_claim in (
            "enclave_measurement_fingerprint",
            "code_measurement_fingerprint",
            "release_digest",
        ):
            claim_value = model_claims.get(required_claim)
            if not isinstance(claim_value, str) or not _is_sha256_digest_value(
                claim_value
            ):
                raise ValueError(
                    f"{claim_key} for {model_id} {required_claim} "
                    "must be a sha256 digest"
                )
        _validate_tls_public_key_binding_claim(
            model_claims,
            required=_tinfoil_tls_public_key_binding_required(
                targets[model_id],
                inherited_policy=policy.policy,
            ),
        )
        _require_verification_steps(
            verifier_result={},
            claims=model_claims,
            required_steps=REQUIRED_EHBP_VERIFICATION_STEPS,
        )


def _validate_confidential_ehbp_verifier_result(
    *,
    verifier_result: dict[str, Any],
    policy_digest: str,
    evidence_digest: str,
    verification_nonce: str,
    ehbp_public_key_hex: str,
    ehbp_key_config_b64: str,
    ehbp_claims: dict[str, object],
    expected_claims: dict[str, str] | None = None,
    expected_claim_values: dict[str, list[str]] | None = None,
    require_tls_public_key_binding: bool = True,
) -> dict[str, Any]:
    if verifier_result.get("verified") is not True:
        raise ValueError("verifier did not return verified=true")

    verifier_value = verifier_result.get("verifier")
    if not isinstance(verifier_value, str):
        raise ValueError("verifier identity must be a string")
    verifier = verifier_value.strip()
    if not verifier:
        raise ValueError("verifier identity is required")

    raw_claims = verifier_result.get("claims")
    if not isinstance(raw_claims, dict):
        raise ValueError("verifier claims must be a JSON object")
    claims = dict(raw_claims)

    _require_verifier_payload_binding(
        verifier_result=verifier_result,
        claims=claims,
        expected_policy_digest=policy_digest,
        expected_evidence_digest=evidence_digest,
        expected_verification_nonce=verification_nonce,
    )

    for claim_name, expected_value in (expected_claims or {}).items():
        if expected_value and claims.get(claim_name) != expected_value:
            if claim_name in {"attestation_format", "attestation_report_digest"}:
                raise ValueError(f"{claim_name} claim does not match evidence")
            raise ValueError(f"verifier {claim_name} claim does not match policy")

    _require_allowed_claim_values(claims, expected_claim_values)
    _require_verification_steps(
        verifier_result=verifier_result,
        claims=claims,
        required_steps=REQUIRED_EHBP_VERIFICATION_STEPS,
    )

    attested_hpke_public_key_hex = _attested_ehbp_public_key_claim(
        claims,
        ehbp_public_key_hex,
    )

    for required_claim in (
        "enclave_measurement_fingerprint",
        "code_measurement_fingerprint",
    ):
        claim_value = claims.get(required_claim)
        if not str(claim_value or "").strip():
            raise ValueError(f"{required_claim} claim is required")
        if not isinstance(claim_value, str) or not _is_sha256_digest_value(claim_value):
            raise ValueError(f"{required_claim} claim must be a sha256 digest")

    _validate_tls_public_key_binding_claim(
        claims,
        required=require_tls_public_key_binding,
    )

    expires_at = _int_result_value(verifier_result.get("expires_at"), "expires_at")
    if expires_at is not None and expires_at <= int(time.time()):
        raise ValueError("verifier result is expired")

    claims.update(ehbp_claims)
    claims.update(
        {
            "transport": "ehbp",
            "ehbp_key_config_b64": ehbp_key_config_b64,
            "attested_hpke_public_key_hex": attested_hpke_public_key_hex,
        }
    )
    claims["runtime_evidence_digest"] = _ehbp_runtime_evidence_digest(claims)
    return claims


def _validate_privatemode_verifier_result(
    *,
    verifier_result: dict[str, Any],
    policy: "ConfidentialVerifierPolicy",
    evidence_digest: str,
    verification_nonce: str,
    proxy_base_url: str,
) -> dict[str, Any]:
    component_policy_violations = _privatemode_component_policy_violations(
        policy.policy
    )
    if component_policy_violations:
        raise ValueError("; ".join(component_policy_violations))

    if verifier_result.get("verified") is not True:
        raise ValueError("verifier did not return verified=true")

    verifier_value = verifier_result.get("verifier")
    if not isinstance(verifier_value, str):
        raise ValueError("verifier identity must be a string")
    verifier = verifier_value.strip()
    if not verifier:
        raise ValueError("verifier identity is required")

    raw_claims = verifier_result.get("claims")
    if not isinstance(raw_claims, dict):
        raise ValueError("verifier claims must be a JSON object")
    claims = dict(raw_claims)

    _require_verifier_payload_binding(
        verifier_result=verifier_result,
        claims=claims,
        expected_policy_digest=policy.digest,
        expected_evidence_digest=evidence_digest,
        expected_verification_nonce=verification_nonce,
    )

    manifest_digest = _policy_string(
        policy.policy,
        *PRIVATEMODE_MANIFEST_DIGEST_POLICY_KEYS,
    )
    if manifest_digest and claims.get("manifest_digest") != manifest_digest:
        raise ValueError("manifest_digest claim does not match policy")
    _require_prefixed_sha256_claim(
        claims,
        "manifest_digest",
        required=bool(manifest_digest),
    )

    manifest_log_dir = _policy_string(
        policy.policy,
        *PRIVATEMODE_MANIFEST_LOG_DIR_POLICY_KEYS,
    )
    if manifest_log_dir:
        _require_prefixed_sha256_claim(claims, "manifest_log_digest")
        manifest_log_manifest_digest = str(
            claims.get("manifest_log_manifest_digest") or ""
        ).strip()
        _require_prefixed_sha256_claim(claims, "manifest_log_manifest_digest")
        manifest_log_entry = claims.get("manifest_log_entry")
        if not isinstance(manifest_log_entry, str):
            raise ValueError("manifest_log_entry claim must be a string")
        if not manifest_log_entry.strip():
            raise ValueError("manifest_log_entry claim is required")
        _require_prefixed_sha256_claim(claims, "manifest_digest")
        if manifest_log_manifest_digest != claims.get("manifest_digest"):
            raise ValueError(
                "manifest_log_manifest_digest does not match manifest_digest"
            )

    proxy_image_digest = _policy_string(
        policy.policy,
        *PRIVATEMODE_PROXY_IMAGE_DIGEST_POLICY_KEYS,
    )
    if proxy_image_digest and claims.get("proxy_image_digest") != proxy_image_digest:
        raise ValueError("proxy_image_digest claim does not match policy")
    _require_prefixed_sha256_claim(
        claims,
        "proxy_image_digest",
        required=bool(proxy_image_digest),
    )

    proxy_binary_digest = _policy_string(
        policy.policy,
        *PRIVATEMODE_PROXY_BINARY_DIGEST_POLICY_KEYS,
    )
    if proxy_binary_digest and claims.get("proxy_binary_digest") != proxy_binary_digest:
        raise ValueError("proxy_binary_digest claim does not match policy")
    _require_prefixed_sha256_claim(
        claims,
        "proxy_binary_digest",
        required=bool(proxy_binary_digest),
    )

    for required_claim in (
        "coordinator_measurement",
        "secret_service_measurement",
        "ai_worker_measurement",
        "key_release_binding",
    ):
        if not str(claims.get(required_claim) or "").strip():
            raise ValueError(f"{required_claim} claim is required")
    gpu_attestation_policy = claims.get("gpu_attestation_policy")
    if not isinstance(gpu_attestation_policy, str):
        raise ValueError("gpu_attestation_policy claim must be a string")
    if not gpu_attestation_policy.strip():
        raise ValueError("gpu_attestation_policy claim is required")
    for digest_claim in PRIVATEMODE_REQUIRED_COMPONENT_DIGEST_CLAIMS:
        _require_prefixed_sha256_claim(claims, digest_claim)
    _require_prefixed_sha256_claim(claims, "attested_workload_policy_digest")
    _require_prefixed_sha256_claim(claims, "expected_workload_identity_digest")
    if claims.get("ai_worker_measurement") != claims.get(
        "attested_workload_policy_digest"
    ):
        raise ValueError(
            "ai_worker_measurement claim must match attested_workload_policy_digest"
        )
    expected_workload_identity_digest = _privatemode_expected_workload_identity_digest(
        policy.policy
    )
    if (
        claims.get("expected_workload_identity_digest")
        != expected_workload_identity_digest
    ):
        raise ValueError(
            "expected_workload_identity_digest claim does not match policy"
        )
    model_workload_binding_digest = _privatemode_model_workload_binding_digest(
        policy.policy
    )
    if claims.get("model_workload_binding_digest") != model_workload_binding_digest:
        raise ValueError("model_workload_binding_digest claim does not match policy")

    _require_verification_steps(
        verifier_result=verifier_result,
        claims=claims,
        required_steps=REQUIRED_PRIVATEMODE_VERIFICATION_STEPS,
    )
    for proof_claim in REQUIRED_PRIVATEMODE_PROOF_CLAIMS:
        _require_prefixed_sha256_claim(claims, proof_claim)
    if claims.get("key_release_binding") != _privatemode_key_release_binding_digest(
        claims
    ):
        raise ValueError("key_release_binding claim does not match component binding")
    _require_allowed_claim_values(
        claims,
        {
            "transport": ["privatemode-proxy"],
            "proxy_base_url": [proxy_base_url],
        },
    )
    _require_allowed_claim_values(
        claims,
        _expected_privatemode_claim_values(policy),
    )
    _require_selected_model_claims(claims, policy.model_ids)

    expires_at = _int_result_value(verifier_result.get("expires_at"), "expires_at")
    if expires_at is not None and expires_at <= int(time.time()):
        raise ValueError("verifier result is expired")

    claims["runtime_evidence_digest"] = _privatemode_runtime_evidence_digest(claims)
    return claims


def _privatemode_runtime_evidence_digest(claims: dict[str, Any]) -> str:
    return _runtime_evidence_digest(claims)


def _ehbp_runtime_evidence_digest(claims: dict[str, Any]) -> str:
    return _runtime_evidence_digest(claims)


def _runtime_evidence_digest(claims: dict[str, Any]) -> str:
    volatile_claims = {
        "payload_policy_digest",
        "payload_evidence_digest",
        "payload_verification_nonce",
        "runtime_evidence_digest",
    }
    evidence_claims = {
        key: value for key, value in claims.items() if key not in volatile_claims
    }
    return _sha256_json_digest(evidence_claims)


async def refresh_ppq_private_status(
    provider: "BaseUpstreamProvider",
) -> ConfidentialityStatus:
    """Probe PPQ private EHBP evidence and require an isolated verifier result."""
    policy = _provider_policy(provider)
    if not policy:
        return _status_with_policy(
            provider,
            mode="ppq-private-tee",
            failure_reason="attestation policy missing for PPQ private verifier",
        )
    if mode_issue := _policy_mode_issue(
        policy,
        expected_mode="ppq-private-tee",
        provider_label="ppq-private",
    ):
        return _status_with_policy(
            provider,
            mode="ppq-private-tee",
            failure_reason=mode_issue,
        )

    violations = []
    violations.extend(inline_policy_secret_violations(policy.policy))
    if provider_base_url_violation := _ppq_private_base_url_violation(provider):
        violations.append(provider_base_url_violation)
    violations.extend(
        _string_policy_field_violations(
            policy.policy,
            "attestation_bundle_url",
            "hpke_keys_url",
            "enclave_host",
            "expected_enclave_host",
        )
    )
    violations.extend(
        _policy_url_userinfo_violations(
            policy.policy,
            "attestation_bundle_url",
            "hpke_keys_url",
        )
    )
    violations.extend(
        _ppq_private_policy_url_violations(
            policy.policy,
            "attestation_bundle_url",
            "hpke_keys_url",
        )
    )
    violations.extend(
        _policy_url_host_violations(
            policy.policy,
            provider.base_url,
            "attestation_bundle_url",
            "hpke_keys_url",
        )
    )
    violations.extend(
        _policy_host_identity_violations(
            policy.policy,
            provider.base_url,
            "enclave_host",
            "expected_enclave_host",
        )
    )
    violations.extend(
        _string_list_policy_alias_violations(
            policy.policy,
            "enclave host",
            "enclave_host",
            "expected_enclave_host",
        )
    )
    violations.extend(
        _non_empty_string_policy_violations(
            policy.policy,
            "repo",
            "expected_repo",
        )
    )
    violations.extend(
        _string_list_policy_alias_violations(
            policy.policy,
            "repo",
            "repo",
            "expected_repo",
        )
    )
    violations.extend(_ppq_private_model_selector_policy_violations(policy))
    if violations:
        return _status_with_policy(
            provider,
            mode="ppq-private-tee",
            failure_reason="; ".join(violations),
        )

    transport = _policy_string(policy.policy, "transport") or "ehbp"
    if transport.lower() != "ehbp":
        return _status_with_policy(
            provider,
            mode="ppq-private-tee",
            failure_reason="PPQ private transport must be ehbp",
        )
    if not _policy_string(policy.policy, "attestation_bundle_url"):
        return _status_with_policy(
            provider,
            mode="ppq-private-tee",
            failure_reason="attestation_bundle_url is required for PPQ private verifier",
        )
    if not _policy_string(policy.policy, "repo", "expected_repo"):
        return _status_with_policy(
            provider,
            mode="ppq-private-tee",
            failure_reason="repo or expected_repo is required for PPQ private verifier",
        )
    if release_policy_issue := _release_identity_policy_issue(policy):
        return _status_with_policy(
            provider,
            mode="ppq-private-tee",
            failure_reason=release_policy_issue,
        )
    if measurement_policy_issue := _measurement_identity_policy_issue(policy):
        return _status_with_policy(
            provider,
            mode="ppq-private-tee",
            failure_reason=measurement_policy_issue,
        )
    if not _artifact_identity_policy_values(policy):
        return _status_with_policy(
            provider,
            mode="ppq-private-tee",
            failure_reason=_missing_artifact_identity_reason("PPQ private"),
        )
    if verifier_command_issue := _verifier_command_policy_issue(
        policy,
        "verifier_command",
        "ppq_private_verifier_command",
    ):
        return _status_with_policy(
            provider,
            mode="ppq-private-tee",
            failure_reason=verifier_command_issue,
        )
    verifier_command = _policy_command(
        policy,
        "verifier_command",
        "ppq_private_verifier_command",
    )
    if verifier_command is None:
        return _status_with_policy(
            provider,
            mode="ppq-private-tee",
            failure_reason="verifier_command or ppq_private_verifier_command is required",
        )
    if verifier_digest_issue := _verifier_command_digest_policy_issue(policy):
        return _status_with_policy(
            provider,
            mode="ppq-private-tee",
            failure_reason=verifier_digest_issue,
        )
    if artifact_path_issue := _verifier_artifact_path_policy_issue(
        policy,
        verifier_command,
    ):
        return _status_with_policy(
            provider,
            mode="ppq-private-tee",
            failure_reason=artifact_path_issue,
        )
    model_attestation_violations = [
        violation.replace("Tinfoil ", "PPQ private ", 1)
        for violation in _tinfoil_model_attestation_policy_violations(policy)
    ]
    if model_attestation_violations:
        return _status_with_policy(
            provider,
            mode="ppq-private-tee",
            failure_reason="; ".join(model_attestation_violations),
        )

    attestation_bundle_url = _policy_string(policy.policy, "attestation_bundle_url")
    hpke_keys_url = str(
        policy.policy.get("hpke_keys_url")
        or urljoin(attestation_bundle_url.rstrip("/") + "/", ".well-known/hpke-keys")
    )

    try:
        async with httpx.AsyncClient(timeout=_timeout_seconds(policy)) as client:
            hpke_response = await client.get(hpke_keys_url)
            hpke_response.raise_for_status()
            hpke_keys = hpke_response.content
            if not hpke_keys:
                raise ValueError("PPQ private HPKE key response is empty")
            ehbp_config = parse_ehbp_key_config(hpke_keys)
            hpke_key_config_b64 = base64.b64encode(hpke_keys).decode("ascii")
            (
                backend_model_attestation_payloads,
                backend_model_attestation_evidence,
            ) = await _collect_tinfoil_model_attestations(
                client=client,
                policy=policy,
            )
    except Exception as exc:
        return _status_with_policy(
            provider,
            mode="ppq-private-tee",
            failure_reason=_exception_failure_reason(
                "PPQ private evidence probe failed",
                exc,
            ),
        )

    evidence_digest = _sha256_json_digest(
        {
            "attestation_bundle_url": attestation_bundle_url,
            "hpke_keys_url": hpke_keys_url,
            "hpke_keys_digest": _sha256_bytes_digest(hpke_keys),
            "ehbp": ehbp_config.verified_claims(),
            "backend_model_attestations": backend_model_attestation_evidence,
        }
    )
    try:
        verifier_command_digest, verifier_artifact_path = _verified_command_digest(
            policy,
            verifier_command,
        )
    except Exception as exc:
        return _status_with_policy(
            provider,
            mode="ppq-private-tee",
            failure_reason=_exception_failure_reason(
                "PPQ private verifier failed",
                exc,
            ),
            evidence_digest=evidence_digest,
        )

    verification_nonce = _new_verification_nonce()
    verifier_payload = {
        "schema_version": "routstr-confidential-verifier-v1",
        "provider_type": provider.provider_type,
        "mode": "ppq-private-tee",
        "base_url": provider.base_url,
        "model_ids": list(policy.model_ids),
        "policy_digest": policy.digest,
        "policy": policy.policy,
        "attestation_bundle_url": attestation_bundle_url,
        "hpke_keys_url": hpke_keys_url,
        "hpke_key_config_b64": hpke_key_config_b64,
        "model_attestations": backend_model_attestation_payloads,
        "evidence_digest": evidence_digest,
        "verification_nonce": verification_nonce,
        "verifier_command_digest": verifier_command_digest,
        "verifier_artifact_path": verifier_artifact_path,
    }

    try:
        verifier_result = await _run_confidential_verifier_command(
            verifier_command,
            verifier_payload,
            _timeout_seconds(policy),
        )
        verified_claims = _validate_confidential_ehbp_verifier_result(
            verifier_result=verifier_result,
            policy_digest=policy.digest,
            evidence_digest=evidence_digest,
            verification_nonce=verification_nonce,
            ehbp_public_key_hex=ehbp_config.public_key.hex(),
            ehbp_key_config_b64=hpke_key_config_b64,
            ehbp_claims=ehbp_config.verified_claims(),
            expected_claims={
                "attestation_bundle_url": attestation_bundle_url,
                "client_encryption_boundary": "routstr-tee-ehbp-proxy",
                "repo": _policy_string(policy.policy, "repo", "expected_repo"),
            },
            expected_claim_values=_expected_ehbp_claim_values(policy),
        )
        _require_selected_model_claims(verified_claims, policy.model_ids)
        _require_tinfoil_model_attestation_claims(
            claims=verified_claims,
            policy=policy,
            model_attestations=backend_model_attestation_payloads,
            claim_key="backend_model_attestations",
        )
        verified_claims["client_encryption_boundary"] = "routstr-tee-ehbp-proxy"
        verified_claims["attestation_bundle_url_digest"] = _sha256_json_digest(
            attestation_bundle_url
        )
        verified_claims["runtime_evidence_digest"] = _ehbp_runtime_evidence_digest(
            verified_claims
        )
        verified_at = _int_result_value(
            verifier_result.get("verified_at"), "verified_at"
        )
        if verified_at is None:
            verified_at = int(time.time())
        expires_at = _bounded_verifier_expires_at(
            verifier_result=verifier_result,
            verified_at=verified_at,
            policy=policy,
        )
    except Exception as exc:
        return _status_with_policy(
            provider,
            mode="ppq-private-tee",
            failure_reason=_exception_failure_reason(
                "PPQ private verifier failed",
                exc,
            ),
            evidence_digest=evidence_digest,
        )

    verifier = str(verifier_result.get("verifier") or "").strip()
    return _verified_status_with_policy(
        provider,
        mode="ppq-private-tee",
        verifier=verifier,
        evidence_digest=verified_claims["runtime_evidence_digest"],
        verified_claims=verified_claims,
        verified_at=verified_at,
        expires_at=expires_at,
    )


def _required_false_bool_policy_issue(
    policy: dict[str, Any],
    label: str,
    *keys: str,
) -> str | None:
    values = [policy[key] for key in keys if key in policy]
    if not values:
        return f"{label} must be explicitly false"
    if any(value is not False for value in values):
        return f"{label} must be explicitly false"
    return None


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


def _required_zero_int_policy_issue(
    policy: dict[str, Any],
    label: str,
    *keys: str,
) -> str | None:
    values = [policy[key] for key in keys if key in policy]
    if not values:
        return f"{label} must be 0"
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value != 0
        for value in values
    ):
        return f"{label} must be 0"
    return None


async def refresh_privatemode_status(
    provider: "BaseUpstreamProvider",
) -> ConfidentialityStatus:
    """Validate strict Privatemode proxy policy and isolated verifier output."""
    policy = _provider_policy(provider)
    if not policy:
        return _status_with_policy(
            provider,
            mode="privatemode",
            failure_reason="attestation policy missing for Privatemode verifier",
        )
    if mode_issue := _policy_mode_issue(
        policy,
        expected_mode="privatemode",
        provider_label="privatemode",
    ):
        return _status_with_policy(
            provider,
            mode="privatemode",
            failure_reason=mode_issue,
        )

    raw_policy = policy.policy
    violations: list[str] = []

    violations.extend(inline_policy_secret_violations(raw_policy))
    violations.extend(
        _prefixed_digest_alias_policy_violations(
            raw_policy,
            *PRIVATEMODE_ARTIFACT_DIGEST_POLICY_KEY_GROUPS,
        )
    )
    violations.extend(_privatemode_component_policy_violations(raw_policy))
    violations.extend(_privatemode_workload_policy_violations(raw_policy))
    violations.extend(_privatemode_model_workload_policy_violations(policy))
    if proxy_base_url_violation := _privatemode_proxy_base_url_violation(provider):
        violations.append(proxy_base_url_violation)
    violations.extend(
        _string_policy_field_violations(
            raw_policy,
            "api_base_url",
            "apiBaseURL",
            "cdn_base_url",
            "cdnBaseURL",
            "manifest_url",
            "manifestURL",
            "api_key_env",
            "apiKeyEnv",
            "api_key_file",
            "apiKeyFile",
            "manifest_path",
            "manifestPath",
            "manifest_b64",
            "manifestB64",
            "manifest_log_dir",
            "manifestLogDir",
            "proxy_binary_path",
            "proxyBinaryPath",
        )
    )
    violations.extend(
        _policy_url_userinfo_violations(
            raw_policy,
            "api_base_url",
            "apiBaseURL",
            "cdn_base_url",
            "cdnBaseURL",
            "manifest_url",
            "manifestURL",
        )
    )
    for issue in (
        _required_false_bool_policy_issue(
            raw_policy,
            "dump_requests",
            "dump_requests",
            "dumpRequests",
        ),
        _required_false_bool_policy_issue(
            raw_policy,
            "shared_prompt_cache",
            "shared_prompt_cache",
            "sharedPromptCache",
        ),
        _required_false_bool_policy_issue(
            raw_policy,
            "nvidia_ocsp_allow_unknown",
            "nvidia_ocsp_allow_unknown",
            "nvidiaOCSPAllowUnknown",
        ),
        _required_zero_int_policy_issue(
            raw_policy,
            "nvidia_ocsp_revoked_grace_period_hours",
            "nvidia_ocsp_revoked_grace_period_hours",
            "nvidiaOCSPRevokedGracePeriod",
        ),
    ):
        if issue:
            violations.append(issue)

    if not _policy_string(
        raw_policy,
        *PRIVATEMODE_MANIFEST_IDENTITY_POLICY_KEYS,
    ):
        violations.append("manifest_digest or manifest_log_dir is required")
    if not _policy_string(
        raw_policy,
        *PRIVATEMODE_PROXY_BINARY_DIGEST_POLICY_KEYS,
    ):
        violations.append(
            "proxy_binary_digest is required for full Privatemode proxy artifact "
            "verification"
        )
    if _policy_string(
        raw_policy,
        *PRIVATEMODE_PROXY_BINARY_DIGEST_POLICY_KEYS,
    ) and not _policy_string(
        raw_policy,
        *PRIVATEMODE_PROXY_BINARY_PATH_POLICY_KEYS,
    ):
        violations.append(
            "proxy_binary_path is required when proxy_binary_digest is set"
        )

    preflight_digest = _sha256_json_digest({"policy": raw_policy})
    if violations:
        return _status_with_policy(
            provider,
            mode="privatemode",
            failure_reason="; ".join(violations),
        )

    if verifier_command_issue := _verifier_command_policy_issue(
        policy,
        "verifier_command",
        "privatemode_verifier_command",
    ):
        return _status_with_policy(
            provider,
            mode="privatemode",
            failure_reason=verifier_command_issue,
        )
    verifier_command = _policy_command(
        policy,
        "verifier_command",
        "privatemode_verifier_command",
    )
    if verifier_command is None:
        return _status_with_policy(
            provider,
            mode="privatemode",
            failure_reason="verifier_command or privatemode_verifier_command is required",
        )
    if verifier_digest_issue := _verifier_command_digest_policy_issue(policy):
        return _status_with_policy(
            provider,
            mode="privatemode",
            failure_reason=verifier_digest_issue,
        )
    if artifact_path_issue := _verifier_artifact_path_policy_issue(
        policy,
        verifier_command,
    ):
        return _status_with_policy(
            provider,
            mode="privatemode",
            failure_reason=artifact_path_issue,
        )
    try:
        verifier_command_digest, verifier_artifact_path = _verified_command_digest(
            policy,
            verifier_command,
        )
    except Exception as exc:
        return _status_with_policy(
            provider,
            mode="privatemode",
            failure_reason=_exception_failure_reason(
                "Privatemode verifier failed", exc
            ),
        )

    verification_nonce = _new_verification_nonce()
    verifier_payload = {
        "schema_version": "routstr-confidential-verifier-v1",
        "provider_type": provider.provider_type,
        "mode": "privatemode",
        "proxy_base_url": provider.base_url,
        "model_ids": list(policy.model_ids),
        "policy_digest": policy.digest,
        "policy": raw_policy,
        "evidence_digest": preflight_digest,
        "verification_nonce": verification_nonce,
        "verifier_command_digest": verifier_command_digest,
        "verifier_artifact_path": verifier_artifact_path,
    }

    try:
        verifier_result = await _run_confidential_verifier_command(
            verifier_command,
            verifier_payload,
            _timeout_seconds(policy),
        )
        verified_claims = _validate_privatemode_verifier_result(
            verifier_result=verifier_result,
            policy=policy,
            evidence_digest=preflight_digest,
            verification_nonce=verification_nonce,
            proxy_base_url=provider.base_url,
        )
        verified_at = _int_result_value(
            verifier_result.get("verified_at"), "verified_at"
        )
        if verified_at is None:
            verified_at = int(time.time())
        expires_at = _bounded_verifier_expires_at(
            verifier_result=verifier_result,
            verified_at=verified_at,
            policy=policy,
        )
    except Exception as exc:
        return _status_with_policy(
            provider,
            mode="privatemode",
            failure_reason=_exception_failure_reason(
                "Privatemode verifier failed", exc
            ),
        )

    verifier = str(verifier_result.get("verifier") or "").strip()
    return _verified_status_with_policy(
        provider,
        mode="privatemode",
        verifier=verifier,
        evidence_digest=verified_claims["runtime_evidence_digest"],
        verified_claims=verified_claims,
        verified_at=verified_at,
        expires_at=expires_at,
    )
