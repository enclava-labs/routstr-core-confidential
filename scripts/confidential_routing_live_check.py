#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import ipaddress
import json
import os
import re
import shlex
import sys
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

FORBIDDEN_PUBLIC_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "bearertoken",
    "client_secret",
    "clientsecret",
    "failure_reason",
    "raw_prompt",
    "upstream_base_url",
    "verified_claims",
}

FORBIDDEN_RUNTIME_PUBLIC_KEYS = {
    "api_base_url",
    "apibaseurl",
    "base_url",
    "baseurl",
    "cdn_base_url",
    "cdnbaseurl",
    "manifest_url",
    "manifesturl",
    "provider_base_url",
    "providerbaseurl",
    "upstream_url",
    "upstreamurl",
}
FORBIDDEN_PUBLIC_KEY_PARTS = {
    "apikey",
    "authorization",
    "bearer",
    "credential",
    "password",
    "private",
    "secret",
    "token",
}
PUBLIC_SECRET_PROOF_ALLOWLIST = {
    "inference_secret_id_digest",
    "secret_service_certificate_digest",
    "secret_service_measurement",
    "secret_service_tls",
}
REQUIRED_ROUTSTR_TEE_VERIFICATION_STEPS = (
    "tee_attestation_report",
    "tee_certificate_chain",
    "measurement_match",
    "runtime_policy_binding",
    "hpke_key_binding",
    "public_key_binding",
    "freshness",
)
REQUIRED_ROUTSTR_TEE_PROOF_DIGEST_CLAIMS = (
    "hpke_key_config_digest",
    "hpke_public_key_digest",
    "public_key_digest",
    "tee_attestation_report_digest",
    "tee_certificate_chain_digest",
    "tee_report_data_digest",
    "tee_report_nonce_digest",
)
REQUIRED_ROUTSTR_TEE_PROOF_STRING_CLAIMS = (
    "attestation_document_format",
    "routstr_code_measurement",
    "routstr_config_measurement",
    "tee_report_nonce",
)
REQUIRED_EHBP_PROVIDER_VERIFICATION_STEPS = (
    "hardware_attestation_report",
    "hardware_certificate_chain",
    "code_transparency",
    "measurement_match",
    "attested_transport_key_binding",
    "freshness",
)
RFC3339_EXTRA_FRACTION_RE = re.compile(r"(\.\d{6})\d+(?=Z|[+-]\d\d:\d\d$)")
KNOWN_TINFOIL_MODEL_HOST_MODELS = {
    "llama3-3-70b.tinfoil.containers.tinfoil.dev": {
        "tinfoil/llama3-3-70b",
        "tinfoil/llama-3.3-70b",
    },
    "deepseek-v4-pro-inf12.tinfoil.containers.tinfoil.dev": {
        "tinfoil/deepseek-v4-pro",
    },
    "kimi-k2-6.inf13.tinfoil.sh": {"tinfoil/kimi-k2-6"},
    "gemma4-31b-1.inf10.tinfoil.sh": {
        "tinfoil/gemma4-31b",
        "tinfoil/gemma-4-31b",
    },
    "qwen3-vl-30b.inf10.tinfoil.sh": {"tinfoil/qwen3-vl-30b"},
    "glm-5-1.tinfoil.containers.tinfoil.dev": {"tinfoil/glm-5-1"},
    "gpt-oss-120b-1.inf10.tinfoil.sh": {"tinfoil/gpt-oss-120b"},
    "gpt-oss-safeguard-120b.tinfoil.containers.tinfoil.dev": {
        "tinfoil/gpt-oss-safeguard-120b",
    },
}
REQUIRED_PRIVATEMODE_PROVIDER_VERIFICATION_STEPS = (
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
REQUIRED_PRIVATEMODE_PROVIDER_PROOF_DIGEST_CLAIMS = (
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
PRIVATEMODE_ATTESTATION_RESULT_DETAIL_KEYS = (
    "transport",
    "trust_tier",
    "manifest_digest",
    "manifest_log_digest",
    "manifest_log_manifest_digest",
    "proxy_image_digest",
    "proxy_binary_digest",
    "coordinator_measurement",
    "secret_service_measurement",
    "ai_worker_measurement",
    "key_release_binding",
    "gpu_attestation_policy",
    "selected_model_ids",
    "verification_steps",
    *REQUIRED_PRIVATEMODE_PROVIDER_PROOF_DIGEST_CLAIMS,
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
PRIVATEMODE_GPU_ATTESTATION_POLICY = "nvidia-ocsp-good-only"
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
TINFOIL_ATTESTATION_RESULT_REQUIRED_TRUE_FIELDS = (
    "sigstore_match",
    "sigstore_bundle_verified",
    "images_verified",
)
PPQ_PRIVATE_BACKEND_ATTESTATION_RESULT_REQUIRED_TRUE_FIELDS = (
    "backend_tls_matches",
    "backend_sigstore_match",
    "backend_sigstore_bundle_verified",
    "backend_images_verified",
)
TINFOIL_TLS_BINDING_NOT_VERIFIED_ERROR = (
    "TLS certificate binding was not verified in "
    "confidential-inference attestation results"
)
PRIVATEMODE_APP_E2EE_TIER_REQUIRED_ERROR = (
    "Privatemode verified attestation results must have trust_tier=app-e2ee"
)
TINFOIL_TRANSPORT_SECURITY_POLICY_KEYS = (
    "transport_security",
    "tinfoil_transport_security",
)


def is_tinfoil_router_identity(
    *,
    check_id: object = None,
    host: str | None = None,
    label: object = None,
) -> bool:
    normalized_check_id = (
        check_id.strip().lower() if isinstance(check_id, str) else ""
    )
    normalized_label = label.strip().lower() if isinstance(label, str) else ""
    return (
        host == "inference.tinfoil.sh"
        or "router" in normalized_check_id
        or "router" in normalized_label
    )


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    message: str


@dataclass(frozen=True)
class ExpectedInferenceRoute:
    provider: str
    model: str
    endpoint: str


@dataclass(frozen=True)
class ProviderModelCatalog:
    provider_models: dict[str, set[str]]


@dataclass(frozen=True)
class AttestationTargetsDirectory:
    provider_check_hosts: dict[str, dict[str, str]]
    provider_check_models: dict[str, dict[str, str]] = field(default_factory=dict)
    provider_target_models: dict[str, set[str]] = field(default_factory=dict)
    provider_check_backend_hosts: dict[str, dict[str, str]] = field(
        default_factory=dict
    )
    provider_check_backend_repos: dict[str, dict[str, str]] = field(
        default_factory=dict
    )
    provider_model_backend_hosts: dict[str, dict[str, str]] = field(
        default_factory=dict
    )
    provider_model_backend_repos: dict[str, dict[str, str]] = field(
        default_factory=dict
    )
    provider_model_hosts: dict[str, dict[str, str]] = field(default_factory=dict)
    provider_ambiguous_target_models: dict[str, set[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class AttestationResultsDirectory:
    provider_host_statuses: dict[str, dict[str, str]]
    provider_host_errors: dict[str, dict[str, str]]
    provider_model_statuses: dict[str, dict[str, str]] = field(default_factory=dict)
    provider_model_errors: dict[str, dict[str, str]] = field(default_factory=dict)
    provider_model_details: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=dict
    )


INFERENCE_ENDPOINT_ALIASES = {
    "chat": "/v1/chat/completions",
    "chat-completion": "/v1/chat/completions",
    "chat-completions": "/v1/chat/completions",
    "/chat/completions": "/v1/chat/completions",
    "/v1/chat/completions": "/v1/chat/completions",
    "completion": "/v1/completions",
    "completions": "/v1/completions",
    "legacy-completion": "/v1/completions",
    "legacy-completions": "/v1/completions",
    "/completions": "/v1/completions",
    "/v1/completions": "/v1/completions",
    "embeddings": "/v1/embeddings",
    "embedding": "/v1/embeddings",
    "/embeddings": "/v1/embeddings",
    "/v1/embeddings": "/v1/embeddings",
    "messages": "/v1/messages",
    "anthropic-messages": "/v1/messages",
    "/messages": "/v1/messages",
    "/v1/messages": "/v1/messages",
    "response": "/v1/responses",
    "responses": "/v1/responses",
    "responses-api": "/v1/responses",
    "/responses": "/v1/responses",
    "/v1/responses": "/v1/responses",
    "audio": "/v1/audio/transcriptions",
    "audio-transcription": "/v1/audio/transcriptions",
    "audio-transcriptions": "/v1/audio/transcriptions",
    "audio-translation": "/v1/audio/translations",
    "audio-translations": "/v1/audio/translations",
    "/v1/audio": "/v1/audio/transcriptions",
    "/v1/audio/transcriptions": "/v1/audio/transcriptions",
    "/v1/audio/translations": "/v1/audio/translations",
}


def reject_json_constant(label: str) -> Callable[[str], None]:
    def reject(value: str) -> None:
        raise ValueError(f"{label} must not contain {value}")

    return reject


def loads_strict_json(data: str, label: str) -> Any:
    return json.loads(data, parse_constant=reject_json_constant(label))


def load_json_document(path: Path, label: str) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_json_constant(label))


def normalized_model_identity(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or None


def model_identity_candidate_variants(value: object) -> set[str]:
    candidate = normalized_model_identity(value)
    if candidate is None:
        return set()
    candidates = {candidate}
    for suffix in ("-enclave", "-model"):
        if candidate.endswith(suffix):
            stripped = candidate[: -len(suffix)].strip("-")
            if stripped:
                candidates.add(stripped)
    return candidates


def provider_catalog_identity_candidate_variants(
    value: object,
    *,
    provider_slug: str,
) -> set[str]:
    candidates = model_identity_candidate_variants(value)
    if provider_slug != "ppq":
        return candidates
    expanded = set(candidates)
    for candidate in candidates:
        for suffix in ("-a3b", "-a10b", "-a17b", "-a22b"):
            if candidate.endswith(suffix):
                stripped = candidate[: -len(suffix)].strip("-")
                if stripped:
                    expanded.add(stripped)
        family_number = re.match(r"^([a-z]+)-([0-9].*)$", candidate)
        if family_number:
            expanded.add(f"{family_number.group(1)}{family_number.group(2)}")
        compact_family_number = re.match(r"^([a-z]+)([0-9].*)$", candidate)
        if compact_family_number:
            expanded.add(
                f"{compact_family_number.group(1)}-{compact_family_number.group(2)}"
            )
    return expanded


def load_provider_model_catalog(path: Path) -> ProviderModelCatalog:
    """Load provider model slugs from confidential-inference providers.json."""
    document = load_json_document(path, "provider catalog JSON")
    if isinstance(document, dict):
        providers = document.get("providers")
    else:
        providers = document
    if not isinstance(providers, list):
        raise ValueError("provider catalog JSON providers must be a list")

    provider_models: dict[str, set[str]] = {}
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        slug = provider.get("slug")
        if not isinstance(slug, str) or not slug.strip():
            continue
        normalized_slug = slug.strip().lower()
        models = provider.get("models")
        if not isinstance(models, list):
            continue
        selected = provider_models.setdefault(normalized_slug, set())
        for model in models:
            if not isinstance(model, dict):
                continue
            for key in ("slug", "id", "name"):
                for candidate in provider_catalog_identity_candidate_variants(
                    model.get(key),
                    provider_slug=normalized_slug,
                ):
                    selected.add(candidate.split("/")[-1])
    return ProviderModelCatalog(provider_models=provider_models)


def hostname_from_https_url(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = urlsplit(value.strip())
    if parsed.scheme != "https" or not parsed.hostname:
        return None
    try:
        parsed.port
    except ValueError:
        return None
    return parsed.hostname.rstrip(".").lower()


def hostname_from_identity(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = urlsplit(f"//{value.strip()}")
    if not parsed.hostname:
        return None
    try:
        parsed.port
    except ValueError:
        return None
    return parsed.hostname.rstrip(".").lower()


def load_attestation_targets_directory(path: Path) -> AttestationTargetsDirectory:
    """Load provider check ID to attestation host mappings from confidential-inference."""
    document = load_json_document(path, "attestation targets JSON")
    providers = document.get("providers") if isinstance(document, dict) else None
    if not isinstance(providers, list):
        raise ValueError("attestation targets JSON providers must be a list")

    provider_check_hosts: dict[str, dict[str, str]] = {}
    provider_check_models: dict[str, dict[str, str]] = {}
    provider_target_models: dict[str, set[str]] = {}
    provider_check_backend_hosts: dict[str, dict[str, str]] = {}
    provider_check_backend_repos: dict[str, dict[str, str]] = {}
    provider_model_backend_hosts: dict[str, dict[str, str]] = {}
    provider_model_backend_repos: dict[str, dict[str, str]] = {}
    provider_model_hosts: dict[str, dict[str, str]] = {}
    provider_model_target_counts: dict[str, dict[str, int]] = {}
    provider_ambiguous_target_models: dict[str, set[str]] = {}
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        slug = public_string_value(provider.get("slug"))
        if slug is None:
            continue
        normalized_provider = slug.lower()
        checks = provider.get("checks")
        if not isinstance(checks, list):
            continue
        check_id_counts: dict[str, int] = {}
        for check in checks:
            if not isinstance(check, dict):
                continue
            check_id = public_string_value(check.get("id"))
            if check_id is not None:
                normalized_check_id = check_id.lower()
                check_id_counts[normalized_check_id] = (
                    check_id_counts.get(normalized_check_id, 0) + 1
                )
        duplicate_check_ids = {
            check_id
            for check_id, count in check_id_counts.items()
            if count > 1
        }
        check_hosts = provider_check_hosts.setdefault(normalized_provider, {})
        check_models = provider_check_models.setdefault(normalized_provider, {})
        target_models = provider_target_models.setdefault(normalized_provider, set())
        check_backend_hosts = provider_check_backend_hosts.setdefault(
            normalized_provider, {}
        )
        check_backend_repos = provider_check_backend_repos.setdefault(
            normalized_provider, {}
        )
        model_backend_hosts = provider_model_backend_hosts.setdefault(
            normalized_provider, {}
        )
        model_backend_repos = provider_model_backend_repos.setdefault(
            normalized_provider, {}
        )
        model_hosts = provider_model_hosts.setdefault(normalized_provider, {})
        model_target_counts = provider_model_target_counts.setdefault(
            normalized_provider, {}
        )
        ambiguous_target_models = provider_ambiguous_target_models.setdefault(
            normalized_provider, set()
        )
        for check in checks:
            if not isinstance(check, dict):
                continue
            check_id = public_string_value(check.get("id"))
            if check_id is None:
                continue
            normalized_check_id = check_id.lower()
            if normalized_check_id in duplicate_check_ids:
                continue
            host = hostname_from_identity(check.get("host"))
            if host is None:
                host = hostname_from_https_url(check.get("url"))
            is_tinfoil_router = normalized_provider == "tinfoil" and (
                is_tinfoil_router_identity(
                    check_id=check_id,
                    host=host,
                    label=check.get("label"),
                )
            )
            model = public_string_value(check.get("model"))
            if model is not None and not is_tinfoil_router:
                normalized_model = model.lower()
                check_models[normalized_check_id] = normalized_model
                target_models.add(normalized_model)
                model_target_counts[normalized_model] = (
                    model_target_counts.get(normalized_model, 0) + 1
                )
            if model is None and normalized_provider == "tinfoil" and host is not None:
                inferred_models = KNOWN_TINFOIL_MODEL_HOST_MODELS.get(host)
                if inferred_models:
                    canonical_model = sorted(inferred_models)[0]
                    check_models[normalized_check_id] = canonical_model
                    target_models.update(inferred_models)
            if host is not None:
                check_hosts[normalized_check_id] = host
                if model is not None and not is_tinfoil_router:
                    model_hosts[model.lower()] = host
            backend_host = hostname_from_identity(check.get("backend_host"))
            if backend_host is not None:
                check_backend_hosts[normalized_check_id] = backend_host
                if model is not None and not is_tinfoil_router:
                    model_backend_hosts[model.lower()] = backend_host
            backend_repo = public_string_value(check.get("backend_repo"))
            if backend_repo is not None:
                check_backend_repos[normalized_check_id] = backend_repo
                if model is not None and not is_tinfoil_router:
                    model_backend_repos[model.lower()] = backend_repo
        for normalized_model, count in model_target_counts.items():
            if count > 1:
                ambiguous_target_models.add(normalized_model)
                model_hosts.pop(normalized_model, None)
                model_backend_hosts.pop(normalized_model, None)
                model_backend_repos.pop(normalized_model, None)
    return AttestationTargetsDirectory(
        provider_check_hosts=provider_check_hosts,
        provider_check_models=provider_check_models,
        provider_target_models=provider_target_models,
        provider_check_backend_hosts=provider_check_backend_hosts,
        provider_check_backend_repos=provider_check_backend_repos,
        provider_model_backend_hosts=provider_model_backend_hosts,
        provider_model_backend_repos=provider_model_backend_repos,
        provider_model_hosts=provider_model_hosts,
        provider_ambiguous_target_models=provider_ambiguous_target_models,
    )


def load_attestation_results_directory(
    path: Path,
    *,
    attestation_targets_directory: AttestationTargetsDirectory | None = None,
    max_result_age_seconds: int | None = None,
    now: datetime | None = None,
) -> AttestationResultsDirectory:
    """Load latest provider attestation statuses from confidential-inference."""
    document = load_json_document(path, "attestation results JSON")
    checks = document.get("checks") if isinstance(document, dict) else None
    if not isinstance(checks, list):
        raise ValueError("attestation results JSON checks must be a list")
    document_timestamp = rfc3339_timestamp(document.get("last_run"))
    if max_result_age_seconds is not None and max_result_age_seconds < 0:
        raise ValueError("max attestation result age must be non-negative")
    freshness_now = now or datetime.now(timezone.utc)
    if freshness_now.tzinfo is None:
        freshness_now = freshness_now.replace(tzinfo=timezone.utc)
    else:
        freshness_now = freshness_now.astimezone(timezone.utc)

    provider_host_statuses: dict[str, dict[str, str]] = {}
    provider_host_errors: dict[str, dict[str, str]] = {}
    provider_model_statuses: dict[str, dict[str, str]] = {}
    provider_model_errors: dict[str, dict[str, str]] = {}
    provider_model_details: dict[str, dict[str, dict[str, Any]]] = {}
    provider_check_id_counts: dict[str, dict[str, int]] = {}
    for check in checks:
        if not isinstance(check, dict):
            continue
        provider = public_string_value(check.get("provider"))
        check_id = public_string_value(check.get("id"))
        if provider is None or check_id is None:
            continue
        normalized_provider = provider.lower()
        normalized_check_id = check_id.lower()
        check_id_counts = provider_check_id_counts.setdefault(normalized_provider, {})
        check_id_counts[normalized_check_id] = (
            check_id_counts.get(normalized_check_id, 0) + 1
        )
    for check in checks:
        if not isinstance(check, dict):
            continue
        provider = public_string_value(check.get("provider"))
        status = public_string_value(check.get("status"))
        if provider is None or status is None:
            continue
        normalized_provider = provider.lower()
        host = hostname_from_identity(check.get("host"))
        explicit_host = host
        if host is None:
            for key in ("source_url", "url"):
                host = hostname_from_https_url(check.get(key))
                if host is not None:
                    break
            explicit_host = host
        check_id = public_string_value(check.get("id"))
        explicit_model = public_string_value(check.get("model"))
        model = explicit_model
        target_host = None
        target_model = None
        if attestation_targets_directory is not None and check_id is not None:
            target_host = (
                attestation_targets_directory.provider_check_hosts.get(
                    normalized_provider, {}
                ).get(check_id.lower())
            )
            target_model = (
                attestation_targets_directory.provider_check_models.get(
                    normalized_provider, {}
                ).get(check_id.lower())
            )
        if (
            target_host is None
            and attestation_targets_directory is not None
            and explicit_model is not None
        ):
            target_host = (
                attestation_targets_directory.provider_model_hosts.get(
                    normalized_provider, {}
                ).get(explicit_model.lower())
            )
        if model is None and attestation_targets_directory is not None:
            model = target_model
        if host is None and attestation_targets_directory is not None:
            host = target_host
        if host is None:
            continue
        if normalized_provider == "tinfoil" and is_tinfoil_router_identity(
            check_id=check_id,
            host=host,
            label=check.get("label"),
        ):
            model = None
        result_error = public_string_value(check.get("error"))
        result_model_target_mismatch = (
            explicit_model is not None
            and target_model is not None
            and explicit_model.lower() != target_model
        )
        if result_model_target_mismatch:
            status = "failed"
            result_error = (
                "model does not match confidential-inference attestation target"
            )
        result_host_target_mismatch = (
            explicit_host is not None
            and target_host is not None
            and (
                target_model is not None
                or (
                    explicit_model is not None
                    and attestation_targets_directory is not None
                    and explicit_model.lower()
                    in attestation_targets_directory.provider_model_hosts.get(
                        normalized_provider, {}
                    )
                )
            )
            and explicit_host != target_host
        )
        if result_host_target_mismatch:
            status = "failed"
            result_error = (
                "host does not match confidential-inference attestation target"
            )
        duplicate_result_check_id = (
            check_id is not None
            and provider_check_id_counts.get(normalized_provider, {}).get(
                check_id.lower(), 0
            )
            > 1
        )
        if duplicate_result_check_id:
            status = "failed"
            result_error = (
                f"duplicate attestation result rows for provider "
                f"{normalized_provider} check {check_id.lower()}"
            )
        if normalized_provider == "tinfoil" and status.lower() == "verified":
            if check.get("tls_matches") is not True:
                status = "failed"
                result_error = TINFOIL_TLS_BINDING_NOT_VERIFIED_ERROR
            else:
                if _normalized_sha256_digest(check.get("release_digest")) is None:
                    status = "failed"
                    result_error = (
                        "release_digest is missing or invalid in "
                        "confidential-inference attestation results"
                    )
                for field in TINFOIL_ATTESTATION_RESULT_REQUIRED_TRUE_FIELDS:
                    if status.lower() != "verified":
                        break
                    if field not in check:
                        status = "failed"
                        result_error = (
                            f"{field} is missing from "
                            "confidential-inference attestation results"
                        )
                        break
                    if check.get(field) is not True:
                        status = "failed"
                        result_error = (
                            f"{field} was not verified in "
                            "confidential-inference attestation results"
                        )
                        break
        if (
            normalized_provider == "privatemode"
            and status.lower() == "verified"
            and str(check.get("trust_tier", "")).strip().lower() != "app-e2ee"
        ):
            status = "failed"
            result_error = PRIVATEMODE_APP_E2EE_TIER_REQUIRED_ERROR
        if (
            normalized_provider == "privatemode"
            and status.lower() == "verified"
            and model is not None
        ):
            proof_shape_issue = privatemode_selected_model_result_shape_issue(
                check,
                model_id=model.strip().lower(),
            )
            if proof_shape_issue is not None:
                status = "failed"
                result_error = proof_shape_issue
        if (
            normalized_provider == "ppq"
            and status.lower() == "verified"
            and model is not None
        ):
            if attestation_targets_directory is not None:
                if (
                    check_id is None
                    and model.strip().lower()
                    in attestation_targets_directory.provider_ambiguous_target_models.get(
                        normalized_provider, set()
                    )
                ):
                    status = "failed"
                    result_error = (
                        "model has multiple confidential-inference attestation "
                        "targets; result row must include a check id"
                    )
                expected_backend_host = (
                    attestation_targets_directory.provider_check_backend_hosts.get(
                        normalized_provider, {}
                    ).get(check_id.lower() if check_id is not None else "")
                )
                if expected_backend_host is None:
                    expected_backend_host = (
                        attestation_targets_directory.provider_model_backend_hosts.get(
                            normalized_provider, {}
                        ).get(model.strip().lower())
                    )
                if expected_backend_host is not None:
                    result_backend_host = hostname_from_identity(
                        check.get("backend_host")
                    )
                    if result_backend_host != expected_backend_host:
                        status = "failed"
                        result_error = (
                            "backend_host does not match confidential-inference "
                            "attestation target"
                        )
                expected_backend_repo = (
                    attestation_targets_directory.provider_check_backend_repos.get(
                        normalized_provider, {}
                    ).get(check_id.lower() if check_id is not None else "")
                )
                if expected_backend_repo is None:
                    expected_backend_repo = (
                        attestation_targets_directory.provider_model_backend_repos.get(
                            normalized_provider, {}
                        ).get(model.strip().lower())
                    )
                if status.lower() == "verified" and expected_backend_repo is not None:
                    result_backend_repo = public_string_value(check.get("backend_repo"))
                    if result_backend_repo != expected_backend_repo:
                        status = "failed"
                        result_error = (
                            "backend_repo does not match confidential-inference "
                            "attestation target"
                        )
            for field in ("release_digest", "backend_release_digest"):
                if status.lower() != "verified":
                    break
                if _normalized_sha256_digest(check.get(field)) is None:
                    status = "failed"
                    result_error = (
                        f"{field} is missing or invalid in "
                        "confidential-inference attestation results"
                    )
                    break
            if status.lower() == "verified":
                for field in PPQ_PRIVATE_BACKEND_ATTESTATION_RESULT_REQUIRED_TRUE_FIELDS:
                    if field not in check:
                        status = "failed"
                        result_error = (
                            f"{field} is missing from "
                            "confidential-inference attestation results"
                        )
                        break
                    if check.get(field) is not True:
                        status = "failed"
                        result_error = (
                            f"{field} was not verified in "
                            "confidential-inference attestation results"
                        )
                        break
        if (
            normalized_provider in {"ppq", "privatemode"}
            and status.lower() == "verified"
            and model is not None
            and model.strip()
            and attestation_targets_directory is not None
        ):
            target_models = set(
                attestation_targets_directory.provider_target_models.get(
                    normalized_provider
                )
                or set()
            )
            target_models.update(
                attestation_targets_directory.provider_check_models.get(
                    normalized_provider,
                    {},
                ).values()
            )
            if model.strip().lower() not in target_models:
                status = "failed"
                result_error = (
                    "selected model is missing from confidential-inference "
                    "attestation targets"
                )
        if (
            status.lower() == "verified"
            and max_result_age_seconds is not None
        ):
            raw_timestamp = check.get("ts")
            timestamp = rfc3339_timestamp(raw_timestamp)
            if raw_timestamp is not None and timestamp is None:
                timestamp = None
            else:
                timestamp = timestamp or document_timestamp
            freshness_issue = attestation_result_freshness_issue(
                timestamp,
                max_age_seconds=max_result_age_seconds,
                now=freshness_now,
            )
            if freshness_issue:
                status = "failed"
                result_error = freshness_issue
        normalized_status = status.lower()
        host_statuses = provider_host_statuses.setdefault(normalized_provider, {})
        model_scoped_ppq_result = normalized_provider == "ppq" and model is not None
        if host in host_statuses and not model_scoped_ppq_result:
            normalized_status = "failed"
            result_error = (
                f"duplicate attestation result rows for provider "
                f"{normalized_provider} host {host}"
            )
        if model_scoped_ppq_result:
            if normalized_status == "verified":
                host_statuses[host] = normalized_status
                provider_host_errors.get(normalized_provider, {}).pop(host, None)
            elif host not in host_statuses:
                host_statuses[host] = normalized_status
        else:
            host_statuses[host] = normalized_status
        if result_error is not None and (
            not model_scoped_ppq_result
            or duplicate_result_check_id
            or result_model_target_mismatch
            or result_host_target_mismatch
        ):
            host_errors = provider_host_errors.setdefault(normalized_provider, {})
            host_errors[host] = result_error
        if model is not None:
            normalized_model = model.strip().lower()
            model_statuses = provider_model_statuses.setdefault(
                normalized_provider, {}
            )
            if normalized_model in model_statuses:
                normalized_status = "failed"
                result_error = (
                    f"duplicate attestation result rows for provider "
                    f"{normalized_provider} model {normalized_model}"
                )
                host_statuses[host] = normalized_status
                host_errors = provider_host_errors.setdefault(normalized_provider, {})
                host_errors[host] = result_error
            model_statuses[normalized_model] = normalized_status
            detail_keys: tuple[str, ...] = (
                "release_digest",
                "backend_release_digest",
                "backend_host",
                "backend_repo",
                "backend_tls_matches",
                "backend_sigstore_match",
                "backend_sigstore_bundle_verified",
                "backend_images_verified",
            )
            if normalized_provider == "privatemode":
                detail_keys = (*detail_keys, *PRIVATEMODE_ATTESTATION_RESULT_DETAIL_KEYS)
            details = {key: check[key] for key in detail_keys if key in check}
            if details:
                provider_model_details.setdefault(normalized_provider, {})[
                    normalized_model
                ] = details
            if result_error is not None:
                model_errors = provider_model_errors.setdefault(normalized_provider, {})
                model_errors[normalized_model] = result_error
    return AttestationResultsDirectory(
        provider_host_statuses=provider_host_statuses,
        provider_host_errors=provider_host_errors,
        provider_model_statuses=provider_model_statuses,
        provider_model_errors=provider_model_errors,
        provider_model_details=provider_model_details,
    )


def privatemode_selected_model_result_shape_issue(
    check: dict[str, Any],
    *,
    model_id: str,
) -> str | None:
    label = "Privatemode selected model evidence"
    required_core_keys = {
        "transport",
        "manifest_digest",
        "proxy_binary_digest",
        "coordinator_measurement",
        "secret_service_measurement",
        "ai_worker_measurement",
        "key_release_binding",
        "selected_model_ids",
        "verification_steps",
    }
    if any(key not in check for key in required_core_keys):
        return f"{label} is missing policy-bound proof details"
    if check.get("transport") != "privatemode-proxy":
        return f"{label} transport must be privatemode-proxy"
    selected_model_ids = check.get("selected_model_ids")
    if not (
        isinstance(selected_model_ids, list)
        and len(selected_model_ids) == 1
        and isinstance(selected_model_ids[0], str)
        and selected_model_ids[0].strip().lower() == model_id
    ):
        return f"{label} selected_model_ids must exactly match the selected model"
    digest_keys = (
        "manifest_digest",
        "proxy_binary_digest",
        "coordinator_measurement",
        "secret_service_measurement",
        "ai_worker_measurement",
        "key_release_binding",
        *REQUIRED_PRIVATEMODE_PROVIDER_PROOF_DIGEST_CLAIMS,
    )
    for key in digest_keys:
        if _normalized_sha256_digest(check.get(key)) is None:
            return f"{label} {key} must be a sha256 digest"
    verification_steps = check.get("verification_steps")
    if not isinstance(verification_steps, dict):
        return f"{label} verification_steps are required"
    for step in REQUIRED_PRIVATEMODE_PROVIDER_VERIFICATION_STEPS:
        if verification_steps.get(step) is not True:
            return f"{label} verification step {step} must be true"
    return None


def rfc3339_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    normalized = RFC3339_EXTRA_FRACTION_RE.sub(r"\1", normalized)
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def attestation_result_freshness_issue(
    timestamp: datetime | None,
    *,
    max_age_seconds: int,
    now: datetime,
) -> str | None:
    if timestamp is None:
        return "attestation result timestamp is missing or invalid"
    age_seconds = (now - timestamp).total_seconds()
    if age_seconds < 0:
        return "attestation result timestamp is in the future"
    if age_seconds > max_age_seconds:
        return (
            "attestation result is stale: "
            f"age_seconds={int(age_seconds)} max_age_seconds={max_age_seconds}"
        )
    return None


def fetch_json(base_url: str, path: str, *, timeout: float) -> Any:
    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("content-type", "")
            data = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"{path} returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"{path} request failed: {exc.reason}") from exc

    if "json" not in content_type.lower():
        raise RuntimeError(f"{path} returned non-JSON content type {content_type!r}")
    try:
        return loads_strict_json(data.decode("utf-8"), f"{path} response JSON")
    except ValueError as exc:
        raise RuntimeError(f"{path} returned invalid JSON: {exc}") from exc


def fetch_bytes(base_url: str, path: str, *, timeout: float) -> tuple[bytes, str]:
    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    request = Request(url, headers={"Accept": "application/ohttp-keys"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read(), response.headers.get("content-type", "")
    except HTTPError as exc:
        raise RuntimeError(f"{path} returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"{path} request failed: {exc.reason}") from exc


def post_json(
    base_url: str,
    path: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> Any:
    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    request_headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        **headers,
    }
    request = Request(
        url,
        data=json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        ),
        headers=request_headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("content-type", "")
            data = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"{path} returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"{path} request failed: {exc.reason}") from exc

    if "json" not in content_type.lower():
        raise RuntimeError(f"{path} returned non-JSON content type {content_type!r}")
    try:
        return loads_strict_json(data.decode("utf-8"), f"{path} response JSON")
    except ValueError as exc:
        raise RuntimeError(f"{path} returned invalid JSON: {exc}") from exc


def post_multipart(
    base_url: str,
    path: str,
    *,
    fields: dict[str, str],
    files: list[tuple[str, str, bytes, str]],
    headers: dict[str, str],
    timeout: float,
) -> Any:
    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    boundary = "routstr-live-check-" + hashlib.sha256(os.urandom(16)).hexdigest()
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        )
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")
    for field_name, filename, content, content_type in files:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    request_headers = {
        "Accept": "application/json",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        **headers,
    }
    request = Request(
        url,
        data=bytes(body),
        headers=request_headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("content-type", "")
            data = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"{path} returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"{path} request failed: {exc.reason}") from exc

    if "json" not in content_type.lower():
        raise RuntimeError(f"{path} returned non-JSON content type {content_type!r}")
    try:
        return loads_strict_json(data.decode("utf-8"), f"{path} response JSON")
    except ValueError as exc:
        raise RuntimeError(f"{path} returned invalid JSON: {exc}") from exc


def public_string_looks_secret(value: str) -> bool:
    stripped = value.strip()
    lowered = stripped.lower()
    if stripped.startswith("sk-") or lowered.startswith("bearer "):
        return True
    if stripped.startswith("cashuA"):
        return True
    if "-----begin private key-----" in lowered:
        return True
    if "-----begin rsa private key-----" in lowered:
        return True
    if "-----begin ec private key-----" in lowered:
        return True
    return False


def forbidden_public_paths(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            key_normalized = key_text.replace("-", "_").lower()
            item_path = f"{path}.{key_text}"
            key_parts = [part for part in key_normalized.split("_") if part]
            route_map_provider_key = (
                path.endswith(".routable_with_full_attestation")
                and key_text in ROUTABLE_FULL_ATTESTATION_PROVIDERS
            )
            forbidden_secret_like_key = (
                key_normalized not in PUBLIC_SECRET_PROOF_ALLOWLIST
                and not key_normalized.endswith("_digest")
                and not route_map_provider_key
                and any(part in FORBIDDEN_PUBLIC_KEY_PARTS for part in key_parts)
            )
            if key_normalized in FORBIDDEN_PUBLIC_KEYS or forbidden_secret_like_key:
                findings.append(item_path)
            findings.extend(forbidden_public_paths(item, item_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(forbidden_public_paths(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        if public_string_looks_secret(value):
            findings.append(path)
    return findings


def forbidden_runtime_public_paths(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            key_normalized = key_text.replace("-", "_").lower()
            item_path = f"{path}.{key_text}"
            if key_normalized in FORBIDDEN_RUNTIME_PUBLIC_KEYS:
                findings.append(item_path)
            findings.extend(forbidden_runtime_public_paths(item, item_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(forbidden_runtime_public_paths(item, f"{path}[{index}]"))
    return findings


def require_digest(value: Any, *, label: str) -> CheckResult:
    if isinstance(value, str) and is_template_placeholder_digest_value(value):
        return CheckResult(False, f"{label} must not be a template placeholder")
    if (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(char in "0123456789abcdefABCDEF" for char in value[7:])
    ):
        return CheckResult(True, f"{label} present")
    return CheckResult(False, f"{label} must be a full sha256 digest")


def require_sha256_digest_value(value: Any, *, label: str) -> CheckResult:
    if not isinstance(value, str) or not value.strip():
        return CheckResult(False, f"{label} is required")
    if is_template_placeholder_digest_value(value):
        return CheckResult(False, f"{label} must not be a template placeholder")
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    if len(digest) == 64 and all(char in "0123456789abcdef" for char in digest):
        return CheckResult(True, f"{label} present")
    return CheckResult(False, f"{label} must be a sha256 digest")


def is_template_placeholder_digest_value(value: str) -> bool:
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    return bool(digest) and len(digest) in {64, 96} and len(set(digest)) == 1


def require_digest_alias_group(
    claims: dict[str, Any],
    aliases: tuple[str, ...],
    *,
    label: str,
) -> list[CheckResult]:
    results = [
        require_digest(claims.get(alias), label=f"{label}.{alias}")
        for alias in aliases
        if alias in claims
    ]
    if results:
        return results
    return [
        CheckResult(
            False,
            f"{label}.{aliases[-1]} must be a full sha256 digest",
        )
    ]


def tinfoil_tls_public_key_binding_required(
    public_policy: dict[str, Any] | None,
    target_policy: dict[str, Any] | None = None,
) -> bool:
    for raw_policy in (target_policy or {}, public_policy or {}):
        for key in TINFOIL_TRANSPORT_SECURITY_POLICY_KEYS:
            value = raw_policy.get(key)
            if isinstance(value, str) and value.strip().lower() == "ehbp":
                return False
    return True


def require_optional_tls_digest_alias_group(
    claims: dict[str, Any],
    *,
    label: str,
    required: bool,
) -> list[CheckResult]:
    has_tls_claim = (
        "tls_public_key_fingerprint_sha256" in claims
        or "tls_public_key" in claims
    )
    if required or has_tls_claim:
        return require_digest_alias_group(
            claims,
            ("tls_public_key_fingerprint_sha256", "tls_public_key"),
            label=label,
        )
    return [CheckResult(True, f"{label}.tls_public_key binding omitted by EHBP policy")]


def require_matching_digest_alias_group(
    claims: dict[str, Any],
    aliases: tuple[str, ...],
    *,
    label: str,
) -> CheckResult:
    values = [
        claims.get(alias).strip().lower()
        for alias in aliases
        if isinstance(claims.get(alias), str)
    ]
    normalized = [
        value.removeprefix("sha256:")
        for value in values
        if value.startswith("sha256:")
        and len(value.removeprefix("sha256:")) == 64
        and all(char in "0123456789abcdef" for char in value.removeprefix("sha256:"))
    ]
    return CheckResult(
        len(normalized) == len(values) and len(set(normalized)) <= 1,
        f"{label}.{aliases[0]} aliases match",
    )


def require_non_empty_string(value: Any, *, label: str) -> CheckResult:
    if isinstance(value, str) and value.strip():
        return CheckResult(True, f"{label} present")
    return CheckResult(False, f"{label} is required")


def require_public_identity_string(value: Any, *, label: str) -> CheckResult:
    if isinstance(value, str) and value.strip():
        return CheckResult(True, f"{label} is a non-empty string")
    return CheckResult(False, f"{label} must be a non-empty string")


def require_true_boolean(value: Any, *, label: str) -> CheckResult:
    return CheckResult(value is True, label)


def require_json_boolean(value: Any, *, label: str) -> CheckResult:
    return CheckResult(isinstance(value, bool), f"{label} is a JSON boolean")


def public_string_value(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def required_local_proxy_binary_artifacts(
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
        provider_type = public_string_value(provider.get("provider_type"))
        confidentiality = provider.get("confidentiality")
        confidentiality_mode = (
            public_string_value(confidentiality.get("mode"))
            if isinstance(confidentiality, dict)
            else None
        )
        if not (
            provider_type == "privatemode" or confidentiality_mode == "privatemode"
        ):
            continue
        if not isinstance(confidentiality, dict) or confidentiality.get("verified") is not True:
            continue
        public_policy = provider.get("confidentiality_policy")
        if not isinstance(public_policy, dict):
            continue
        for key in ("proxy_binary_digest", "proxyBinaryDigest"):
            value = public_policy.get(key)
            if require_digest(value, label="Privatemode proxy binary digest").ok:
                required.setdefault("privatemode_proxy_binary", set()).add(
                    value.strip()
                )
    return required


def ppq_private_base_url_has_private_path(value: str) -> bool:
    parsed = urlsplit(value)
    path_segments = {segment.lower() for segment in parsed.path.split("/") if segment}
    return "private" in path_segments


def ppq_private_base_url_has_ppq_host(value: str) -> bool:
    parsed = urlsplit(value)
    hostname = (parsed.hostname or "").rstrip(".").lower()
    return hostname == "ppq.ai" or hostname.endswith(".ppq.ai")


def normalized_base_url(value: str) -> str:
    return value.strip().rstrip("/")


def tinfoil_base_url_is_default(value: str) -> bool:
    return normalized_base_url(value) == "https://inference.tinfoil.sh/v1"


def is_loopback_hostname(hostname: str | None) -> bool:
    if not hostname:
        return False
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def privatemode_base_url_is_loopback_proxy(value: str) -> bool:
    parsed = urlsplit(value)
    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.hostname)
        and is_loopback_hostname(parsed.hostname)
    )


def provider_base_url_scheme_allowed(
    value: str,
    *,
    provider_type: str | None,
    confidentiality_mode: str | None,
) -> bool:
    parsed = urlsplit(value)
    try:
        parsed.port
    except ValueError:
        return False
    if not parsed.hostname:
        return False
    if parsed.scheme == "https":
        return True
    if (
        parsed.scheme == "http"
        and is_loopback_hostname(parsed.hostname)
        and (provider_type == "privatemode" or confidentiality_mode == "privatemode")
    ):
        return True
    return False


def require_hex_string(value: Any, *, label: str, length: int) -> CheckResult:
    if not isinstance(value, str) or not value.strip():
        return CheckResult(False, f"{label} is required")
    if len(value.strip()) == length and all(
        char in "0123456789abcdefABCDEF" for char in value.strip()
    ):
        return CheckResult(True, f"{label} present")
    return CheckResult(False, f"{label} must be {length} hex characters")


def require_integer_timestamp(value: Any, *, label: str) -> CheckResult:
    if not isinstance(value, int) or isinstance(value, bool):
        return CheckResult(False, f"{label} must be an integer Unix timestamp")
    return CheckResult(True, f"{label} is an integer Unix timestamp")


def require_not_future_timestamp(value: Any, *, label: str) -> CheckResult:
    timestamp_check = require_integer_timestamp(value, label=label)
    if not timestamp_check.ok:
        return timestamp_check
    return CheckResult(value <= time.time(), f"{label} is not in the future")


def require_future_timestamp(value: Any, *, label: str) -> CheckResult:
    timestamp_check = require_integer_timestamp(value, label=label)
    if not timestamp_check.ok:
        return timestamp_check
    return CheckResult(value > time.time(), f"{label} is in the future")


def sha256_bytes_digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_file_digest(path: Path) -> str:
    return sha256_bytes_digest(path.read_bytes())


def sha256_json_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes_digest(payload)


def routstr_tee_report_data_input(
    *,
    routing_policy_digest: str,
    hpke_key_config_digest: str,
    hpke_public_key_digest: str,
    public_key_digest: str,
    verification_nonce: str,
) -> dict[str, str]:
    return {
        "schema_version": "routstr-tee-report-data-v1",
        "service": "routstr",
        "routing_policy_digest": routing_policy_digest,
        "hpke_key_config_digest": hpke_key_config_digest,
        "hpke_public_key_digest": hpke_public_key_digest,
        "public_key_digest": public_key_digest,
        "verification_nonce": verification_nonce,
    }


def routstr_tee_report_data_digest(
    *,
    routing_policy_digest: str,
    hpke_key_config_digest: str,
    hpke_public_key_digest: str,
    public_key_digest: str,
    verification_nonce: str,
) -> str:
    return sha256_json_digest(
        routstr_tee_report_data_input(
            routing_policy_digest=routing_policy_digest,
            hpke_key_config_digest=hpke_key_config_digest,
            hpke_public_key_digest=hpke_public_key_digest,
            public_key_digest=public_key_digest,
            verification_nonce=verification_nonce,
        )
    )


def routstr_tee_report_data_hex(
    *,
    routing_policy_digest: str,
    hpke_key_config_digest: str,
    hpke_public_key_digest: str,
    public_key_digest: str,
    verification_nonce: str,
) -> str:
    payload = json.dumps(
        routstr_tee_report_data_input(
            routing_policy_digest=routing_policy_digest,
            hpke_key_config_digest=hpke_key_config_digest,
            hpke_public_key_digest=hpke_public_key_digest,
            public_key_digest=public_key_digest,
            verification_nonce=verification_nonce,
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha512(payload).hexdigest()


def privatemode_key_release_binding_digest(proof_claims: dict[str, Any]) -> str:
    return sha256_json_digest(
        {key: proof_claims.get(key) for key in PRIVATEMODE_KEY_RELEASE_BINDING_CLAIMS}
    )


def public_string_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item.strip() for item in value if isinstance(item, str) and item.strip()}


def public_routable_with_full_attestation_map(
    value: Any,
) -> dict[str, list[str]] | None:
    if not isinstance(value, dict):
        return None
    routable: dict[str, list[str]] = {}
    for provider_type in ROUTABLE_FULL_ATTESTATION_PROVIDERS:
        raw_model_ids = value.get(provider_type, [])
        if not isinstance(raw_model_ids, list):
            return None
        model_ids: list[str] = []
        seen: set[str] = set()
        for raw_model_id in raw_model_ids:
            if not isinstance(raw_model_id, str) or not raw_model_id.strip():
                return None
            model_id = raw_model_id.strip()
            normalized = model_id.lower()
            if normalized in seen:
                return None
            seen.add(normalized)
            model_ids.append(model_id)
        routable[provider_type] = sorted(model_ids)
    return routable


def strict_public_string_set(value: Any, *, allow_empty: bool = False) -> set[str] | None:
    if not isinstance(value, list):
        return None
    if not value:
        return set() if allow_empty else None
    if not all(isinstance(item, str) and item.strip() for item in value):
        return None
    normalized = [item.strip() for item in value]
    if len({item.lower() for item in normalized}) != len(normalized):
        return None
    return set(normalized)


def strict_public_string_sets_match(left: Any, right: Any) -> bool:
    left_set = strict_public_string_set(left, allow_empty=True)
    right_set = strict_public_string_set(right, allow_empty=True)
    return left_set is not None and right_set is not None and left_set == right_set


def validate_known_provider_exact_selectors(
    confidentiality: dict[str, Any],
    *,
    provider_type: str | None,
    label: str,
) -> list[CheckResult]:
    if confidentiality.get("verified") is not True:
        return []
    if provider_type not in CONFIDENTIAL_PROVIDER_MODES:
        return []
    model_ids = strict_public_string_set(confidentiality.get("model_ids"))
    model_prefixes = strict_public_string_set(
        confidentiality.get("model_id_prefixes"),
        allow_empty=True,
    )
    exact_selectors = (
        model_ids is not None
        and model_prefixes is not None
        and bool(model_ids)
        and not model_prefixes
    )
    if provider_type == "ppq-private":
        exact_selectors = exact_selectors and all(
            model_id.startswith("private/") for model_id in model_ids
        )
    if provider_type == "privatemode":
        exact_selectors = exact_selectors and all(
            model_id.startswith("privatemode/") for model_id in model_ids
        )
    return [
        CheckResult(
            exact_selectors,
            f"{label} verified selectors are exact for {provider_type}",
        )
    ]


def endpoint_covers(provider_endpoint: str, required_endpoint: str) -> bool:
    if provider_endpoint == required_endpoint:
        return True
    return provider_endpoint == "/v1/audio" and required_endpoint.startswith(
        "/v1/audio/"
    )


def endpoints_cover(
    provider_supported_endpoints: set[str],
    required_endpoints: set[str],
) -> bool:
    return bool(required_endpoints) and all(
        any(
            endpoint_covers(provider_endpoint, required_endpoint)
            for provider_endpoint in provider_supported_endpoints
        )
        for required_endpoint in required_endpoints
    )


def validate_public_string_list(value: Any, *, label: str) -> list[CheckResult]:
    if value is None:
        return []
    if not isinstance(value, list):
        return [CheckResult(False, f"{label} must be a list of non-empty strings")]

    invalid_paths: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            invalid_paths.append(f"{label}[{index}]")
    results: list[CheckResult] = []
    if invalid_paths:
        results.append(
            CheckResult(
                False,
                f"{label} must contain only non-empty strings: "
                + ", ".join(invalid_paths[:10]),
            )
        )
    normalized = [item.strip().lower() for item in value if isinstance(item, str)]
    if len(set(normalized)) != len(normalized):
        results.append(CheckResult(False, f"{label} must not contain duplicates"))
    if results:
        return results
    return []


def digest_match_result(value: Any, expected: str, *, label: str) -> CheckResult:
    return CheckResult(
        value == expected,
        f"{label} matches /.well-known/hpke-keys digest",
    )


def attestation_hpke_key_config_digest(attestation: dict[str, Any]) -> Any:
    tee = attestation.get("tee")
    if not isinstance(tee, dict):
        return None
    hpke_key_config = tee.get("hpke_key_config")
    if isinstance(hpke_key_config, dict):
        return hpke_key_config.get("key_config_digest")
    return tee.get("hpke_key_config_digest")


def status_local_verification_evidence_digest(status: dict[str, Any]) -> Any:
    tee = status.get("routstr_tee")
    if not isinstance(tee, dict):
        return None
    local_verification = tee.get("local_verification")
    if isinstance(local_verification, dict):
        return local_verification.get("evidence_digest")
    return None


def attestation_local_verification_evidence_digest(attestation: dict[str, Any]) -> Any:
    tee = attestation.get("tee")
    if not isinstance(tee, dict):
        return None
    local_verification = tee.get("local_verification")
    if isinstance(local_verification, dict):
        return local_verification.get("evidence_digest")
    return None


def status_local_verification_claims_digest(status: dict[str, Any]) -> Any:
    tee = status.get("routstr_tee")
    if not isinstance(tee, dict):
        return None
    local_verification = tee.get("local_verification")
    if isinstance(local_verification, dict):
        return local_verification.get("verified_claims_digest")
    return None


def status_local_verification_proof_claims(status: dict[str, Any]) -> Any:
    tee = status.get("routstr_tee")
    if not isinstance(tee, dict):
        return None
    local_verification = tee.get("local_verification")
    if isinstance(local_verification, dict):
        return local_verification.get("proof_claims")
    return None


def attestation_local_verification_claims_digest(attestation: dict[str, Any]) -> Any:
    tee = attestation.get("tee")
    if not isinstance(tee, dict):
        return None
    local_verification = tee.get("local_verification")
    if isinstance(local_verification, dict):
        return local_verification.get("verified_claims_digest")
    return None


def attestation_local_verification_proof_claims(attestation: dict[str, Any]) -> Any:
    tee = attestation.get("tee")
    if not isinstance(tee, dict):
        return None
    local_verification = tee.get("local_verification")
    if isinstance(local_verification, dict):
        return local_verification.get("proof_claims")
    return None


def hpke_public_key_digest_from_attestation(attestation: dict[str, Any]) -> Any:
    tee = attestation.get("tee")
    if not isinstance(tee, dict):
        return None
    hpke_key_config = tee.get("hpke_key_config")
    if isinstance(hpke_key_config, dict):
        return hpke_key_config.get("public_key_digest")
    return tee.get("hpke_public_key_digest")


def validate_local_tee_proof_claims(
    local_verification: dict[str, Any],
    *,
    label: str,
    routing_policy_digest: Any = None,
    routing_policy: dict[str, Any] | None = None,
    hpke_key_config_digest: Any = None,
    hpke_public_key_digest: Any = None,
) -> list[CheckResult]:
    proof_claims = local_verification.get("proof_claims")
    if not isinstance(proof_claims, dict):
        return [CheckResult(False, f"{label} proof_claims are present")]

    results: list[CheckResult] = []
    for claim in REQUIRED_ROUTSTR_TEE_PROOF_STRING_CLAIMS:
        results.append(
            require_non_empty_string(
                proof_claims.get(claim),
                label=f"{label} proof_claims.{claim}",
            )
        )
    for claim in REQUIRED_ROUTSTR_TEE_PROOF_DIGEST_CLAIMS:
        results.append(
            require_digest(
                proof_claims.get(claim),
                label=f"{label} proof_claims.{claim}",
            )
        )
    results.append(
        require_hex_string(
            proof_claims.get("tee_report_data_hex"),
            label=f"{label} proof_claims.tee_report_data_hex",
            length=128,
        )
    )

    verification_steps = proof_claims.get("verification_steps")
    if not isinstance(verification_steps, dict):
        results.append(
            CheckResult(False, f"{label} proof_claims.verification_steps is required")
        )
    else:
        malformed_keys = [
            key
            for key in verification_steps
            if not isinstance(key, str) or not key.strip()
        ]
        results.append(
            CheckResult(
                not malformed_keys,
                f"{label} proof_claims.verification_steps keys are non-empty strings",
            )
        )
        results.append(
            CheckResult(
                all(isinstance(value, bool) for value in verification_steps.values()),
                f"{label} proof_claims.verification_steps values are JSON booleans",
            )
        )
        for step in REQUIRED_ROUTSTR_TEE_VERIFICATION_STEPS:
            results.append(
                CheckResult(
                    verification_steps.get(step) is True,
                    f"{label} proof_claims.verification_steps.{step} is true",
                )
            )

    if routing_policy_digest is not None:
        results.append(
            CheckResult(
                proof_claims.get("routstr_config_measurement") == routing_policy_digest,
                f"{label} proof_claims.routstr_config_measurement matches routing policy",
            )
        )
    if hpke_public_key_digest is not None:
        results.append(
            CheckResult(
                proof_claims.get("hpke_public_key_digest") == hpke_public_key_digest,
                f"{label} proof_claims.hpke_public_key_digest matches HPKE key",
            )
        )
    if hpke_key_config_digest is not None:
        results.append(
            CheckResult(
                proof_claims.get("hpke_key_config_digest") == hpke_key_config_digest,
                (
                    f"{label} proof_claims.hpke_key_config_digest "
                    "matches HPKE key config"
                ),
            )
        )
    tee_report_nonce = proof_claims.get("tee_report_nonce")
    if isinstance(tee_report_nonce, str) and tee_report_nonce.strip():
        results.append(
            CheckResult(
                proof_claims.get("tee_report_nonce_digest")
                == sha256_bytes_digest(tee_report_nonce.encode("utf-8")),
                f"{label} proof_claims.tee_report_nonce_digest matches tee_report_nonce",
            )
        )
    required_proxy_artifacts = required_local_proxy_binary_artifacts(routing_policy)
    if required_proxy_artifacts:
        attested_local_artifacts = proof_claims.get("attested_local_artifacts")
        for artifact_claim, required_proxy_digests in sorted(
            required_proxy_artifacts.items()
        ):
            proxy_binary_digest = (
                attested_local_artifacts.get(artifact_claim)
                if isinstance(attested_local_artifacts, dict)
                else None
            )
            digest_check = require_digest(
                proxy_binary_digest,
                label=(
                    f"{label} proof_claims.attested_local_artifacts."
                    f"{artifact_claim}"
                ),
            )
            results.append(digest_check)
            results.append(
                CheckResult(
                    digest_check.ok and proxy_binary_digest in required_proxy_digests,
                    (
                        f"{label} proof_claims.attested_local_artifacts."
                        f"{artifact_claim} matches Privatemode proxy binary digest"
                    ),
                )
            )
    if (
        isinstance(routing_policy_digest, str)
        and isinstance(hpke_key_config_digest, str)
        and isinstance(hpke_public_key_digest, str)
        and isinstance(proof_claims.get("public_key_digest"), str)
        and isinstance(tee_report_nonce, str)
        and tee_report_nonce.strip()
    ):
        expected_report_data_digest = routstr_tee_report_data_digest(
            routing_policy_digest=routing_policy_digest,
            hpke_key_config_digest=hpke_key_config_digest,
            hpke_public_key_digest=hpke_public_key_digest,
            public_key_digest=proof_claims["public_key_digest"],
            verification_nonce=tee_report_nonce,
        )
        expected_report_data_hex = routstr_tee_report_data_hex(
            routing_policy_digest=routing_policy_digest,
            hpke_key_config_digest=hpke_key_config_digest,
            hpke_public_key_digest=hpke_public_key_digest,
            public_key_digest=proof_claims["public_key_digest"],
            verification_nonce=tee_report_nonce,
        )
        results.append(
            CheckResult(
                proof_claims.get("tee_report_data_digest")
                == expected_report_data_digest,
                f"{label} proof_claims.tee_report_data_digest binds policy, keys, and nonce",
            )
        )
        results.append(
            CheckResult(
                proof_claims.get("tee_report_data_hex") == expected_report_data_hex,
                f"{label} proof_claims.tee_report_data_hex binds policy, keys, and nonce",
            )
        )
    return results


def validate_provider_verification_steps(
    proof_claims: dict[str, Any],
    *,
    label: str,
    required_steps: tuple[str, ...],
) -> list[CheckResult]:
    verification_steps = proof_claims.get("verification_steps")
    if not isinstance(verification_steps, dict):
        return [
            CheckResult(False, f"{label} proof_claims.verification_steps is required")
        ]
    malformed_keys = [
        key for key in verification_steps if not isinstance(key, str) or not key.strip()
    ]
    results = [
        CheckResult(
            not malformed_keys,
            f"{label} proof_claims.verification_steps keys are non-empty strings",
        ),
        CheckResult(
            all(isinstance(value, bool) for value in verification_steps.values()),
            f"{label} proof_claims.verification_steps values are JSON booleans",
        ),
    ]
    results.extend(
        CheckResult(
            verification_steps.get(step) is True,
            f"{label} proof_claims.verification_steps.{step} is true",
        )
        for step in required_steps
    )
    return results


def validate_tinfoil_model_attestation_proof_claims(
    proof_claims: dict[str, Any],
    *,
    model_ids: set[str],
    label: str,
    claim_key: str = "model_attestations",
    public_policy: dict[str, Any] | None = None,
) -> list[CheckResult]:
    if not model_ids:
        return []
    model_attestations = proof_claims.get(claim_key)
    if not isinstance(model_attestations, dict):
        return [
            CheckResult(
                False,
                f"{label} provider proof_claims.{claim_key} contains selected model",
            )
        ]

    results: list[CheckResult] = []
    attested_models: set[str] = set()
    attestation_keys_are_unique = True
    for raw_model_id in model_attestations:
        if not isinstance(raw_model_id, str) or not raw_model_id.strip():
            attestation_keys_are_unique = False
            continue
        model_id = raw_model_id.strip()
        if model_id in attested_models:
            attestation_keys_are_unique = False
        attested_models.add(model_id)
    results.append(
        CheckResult(
            attestation_keys_are_unique,
            f"{label} provider proof_claims.{claim_key} keys are unique after trimming",
        )
    )
    for model_id in sorted(model_ids):
        model_claims = model_attestations.get(model_id)
        results.append(
            CheckResult(
                isinstance(model_claims, dict),
                (
                    f"{label} provider proof_claims.{claim_key} contains "
                    f"selected model {model_id}"
                ),
            )
        )
        if not isinstance(model_claims, dict):
            continue
        results.append(
            require_non_empty_string(
                model_claims.get("repo"),
                label=f"{label} provider proof_claims.{claim_key}.{model_id}.repo",
            )
        )
        results.append(
            require_non_empty_string(
                model_claims.get("attestation_format"),
                label=(
                    f"{label} provider proof_claims.{claim_key}."
                    f"{model_id}.attestation_format"
                ),
            )
        )
        for claim in ("attestation_report_digest", "release_digest"):
            results.append(
                require_digest(
                    model_claims.get(claim),
                    label=(
                        f"{label} provider proof_claims.{claim_key}."
                        f"{model_id}.{claim}"
                    ),
                )
            )
        for claim in (
            "enclave_measurement_fingerprint",
            "code_measurement_fingerprint",
        ):
            results.append(
                require_sha256_digest_value(
                    model_claims.get(claim),
                    label=(
                        f"{label} provider proof_claims.{claim_key}."
                        f"{model_id}.{claim}"
                    ),
                )
            )
        results.append(
            require_hex_string(
                model_claims.get("attested_hpke_public_key_hex"),
                label=(
                    f"{label} provider proof_claims.{claim_key}."
                    f"{model_id}.attested_hpke_public_key_hex"
                ),
                length=64,
            )
        )
        target_policy = None
        if isinstance(public_policy, dict):
            targets = _policy_targets(public_policy)
            maybe_target = targets.get(model_id) if isinstance(targets, dict) else None
            if isinstance(maybe_target, dict):
                target_policy = maybe_target
        results.extend(
            require_optional_tls_digest_alias_group(
                model_claims,
                label=(
                    f"{label} provider proof_claims.{claim_key}."
                    f"{model_id}"
                ),
                required=tinfoil_tls_public_key_binding_required(
                    public_policy,
                    target_policy,
                ),
            )
        )
        results.extend(
            validate_provider_verification_steps(
                model_claims,
                label=(
                    f"{label} provider proof_claims.{claim_key}.{model_id}"
                ),
                required_steps=REQUIRED_EHBP_PROVIDER_VERIFICATION_STEPS,
            )
        )
    results.append(
        CheckResult(
            attested_models == model_ids,
            f"{label} provider proof_claims.{claim_key} exactly match selected models",
        )
    )
    return results


def _policy_targets(policy: dict[str, Any]) -> dict[str, Any] | None:
    for key in (
        "model_attestation_targets",
        "modelAttestationTargets",
        "model_enclave_bindings",
        "modelEnclaveBindings",
    ):
        value = policy.get(key)
        if isinstance(value, dict):
            return value
    return None


def _policy_string(policy: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = policy.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalized_sha256_digest(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    if len(digest) == 64 and all(char in "0123456789abcdef" for char in digest):
        return f"sha256:{digest}"
    return None


def _policy_string_set(policy: dict[str, Any], *keys: str) -> set[str]:
    values: set[str] = set()
    for key in keys:
        value = policy.get(key)
        if isinstance(value, str) and value.strip():
            values.add(value.strip())
        elif isinstance(value, list):
            values.update(
                item.strip()
                for item in value
                if isinstance(item, str) and item.strip()
            )
    return values


def _policy_sorted_strings(policy: dict[str, Any], *keys: str) -> list[str]:
    return sorted(_policy_string_set(policy, *keys))


def _strict_policy_sorted_strings(
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


def _policy_digest_set(policy: dict[str, Any], *keys: str) -> set[str]:
    return {
        normalized
        for value in _policy_string_set(policy, *keys)
        if (normalized := _normalized_sha256_digest(value)) is not None
    }


def _policy_digest_alias_values(
    policy: dict[str, Any],
    *keys: str,
) -> set[str] | None:
    values: set[str] | None = None
    for key in keys:
        if key not in policy:
            continue
        key_values = _policy_string_set(policy, key)
        normalized = {
            digest
            for value in key_values
            if (digest := _normalized_sha256_digest(value)) is not None
        }
        if len(normalized) != len(key_values):
            return None
        if values is None:
            values = normalized
        elif values != normalized:
            return None
    return values or set()


def _claim_matches_policy_digest_pin(
    claims: dict[str, Any],
    target: dict[str, Any],
    *,
    claim_key: str,
    policy_keys: tuple[str, ...],
) -> bool:
    expected = _policy_string(target, *policy_keys)
    if expected is None:
        return True
    return (
        _normalized_sha256_digest(claims.get(claim_key))
        == _normalized_sha256_digest(expected)
    )


def _policy_digest_binding_results(
    policy: dict[str, Any],
    proof_claims: dict[str, Any],
    *,
    claim_key: str,
    exact_keys: tuple[str, ...],
    allowed_keys: tuple[str, ...] = (),
    label: str,
    description: str,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    claim_digest = _normalized_sha256_digest(proof_claims.get(claim_key))
    exact_values = _policy_digest_set(policy, *exact_keys)
    if exact_values:
        results.append(
            CheckResult(
                len(exact_values) == 1 and claim_digest in exact_values,
                f"{label}.confidentiality_policy {description} matches provider proof",
            )
        )
    allowed_values = _policy_digest_set(policy, *allowed_keys)
    if allowed_values:
        results.append(
            CheckResult(
                claim_digest in allowed_values,
                (
                    f"{label}.confidentiality_policy allowed {description} "
                    "includes provider proof"
                ),
            )
        )
    return results


def _policy_digest_alias_binding_results(
    policy: dict[str, Any],
    proof_claims: dict[str, Any],
    *,
    claim_key: str,
    alias_keys: tuple[str, ...],
    label: str,
    description: str,
) -> list[CheckResult]:
    alias_values = _policy_digest_alias_values(policy, *alias_keys)
    if alias_values is None:
        return [
            CheckResult(
                False,
                (
                    f"{label}.confidentiality_policy {description} aliases "
                    "are valid and consistent"
                ),
            )
        ]
    if not alias_values:
        return []
    claim_digest = _normalized_sha256_digest(proof_claims.get(claim_key))
    return [
        CheckResult(
            len(alias_values) == 1 and claim_digest in alias_values,
            f"{label}.confidentiality_policy {description} matches provider proof",
        )
    ]


def _policy_has_digest_pin(policy: dict[str, Any], *keys: str) -> bool:
    return bool(_policy_digest_set(policy, *keys))


def _policy_has_digest_alias_pin(policy: dict[str, Any], *keys: str) -> bool:
    alias_values = _policy_digest_alias_values(policy, *keys)
    return bool(alias_values)


def _privatemode_expected_workload_identity_digest(
    policy: dict[str, Any],
) -> str | None:
    workload_ids = _strict_policy_sorted_strings(
        policy,
        "expected_workload_ids",
        "expectedWorkloadIDs",
    )
    workload_sans = _strict_policy_sorted_strings(
        policy,
        "expected_workload_sans",
        "expectedWorkloadSANs",
    )
    if workload_ids is None or workload_sans is None:
        return None
    if not workload_ids and not workload_sans:
        return None
    return sha256_json_digest(
        {
            "ids": workload_ids,
            "sans": workload_sans,
        }
    )


def _privatemode_expected_workload_sets(
    policy: dict[str, Any],
) -> tuple[set[str], set[str]] | None:
    workload_ids = _strict_policy_sorted_strings(
        policy,
        "expected_workload_ids",
        "expectedWorkloadIDs",
    )
    workload_sans = _strict_policy_sorted_strings(
        policy,
        "expected_workload_sans",
        "expectedWorkloadSANs",
    )
    if workload_ids is None or workload_sans is None:
        return None
    if not workload_ids and not workload_sans:
        return None
    return set(workload_ids), set(workload_sans)


def _privatemode_model_workload_bindings(
    policy: dict[str, Any],
) -> dict[str, dict[str, list[str]]] | None:
    raw_bindings = None
    for key in ("model_workload_bindings", "modelWorkloadBindings"):
        value = policy.get(key)
        if value is not None:
            raw_bindings = value
            break
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
        workload_ids = _strict_policy_sorted_strings(
            raw_binding,
            "workload_ids",
            "workloadIDs",
        )
        workload_sans = _strict_policy_sorted_strings(
            raw_binding,
            "workload_sans",
            "workloadSANs",
        )
        if workload_ids is None or workload_sans is None:
            return None
        if not workload_ids and not workload_sans:
            return None
        bindings[model_id] = {
            "workload_ids": workload_ids,
            "workload_sans": workload_sans,
        }
    return dict(sorted(bindings.items()))


def _privatemode_model_workload_binding_digest(
    policy: dict[str, Any],
) -> str | None:
    bindings = _privatemode_model_workload_bindings(policy)
    if bindings is None:
        return None
    return sha256_json_digest(bindings)


def validate_tinfoil_policy_proof_binding(
    policy: dict[str, Any],
    proof_claims: dict[str, Any],
    *,
    model_ids: set[str],
    label: str,
) -> list[CheckResult]:
    results: list[CheckResult] = [
        CheckResult(
            _policy_string(policy, "repo", "expected_repo") is not None,
            f"{label}.confidentiality_policy repo is pinned",
        ),
        CheckResult(
            _policy_has_digest_pin(
                policy,
                "expected_release_digest",
                "release_digest",
                "allowed_release_digest",
                "allowed_release_digests",
            ),
            f"{label}.confidentiality_policy release digest is pinned",
        ),
    ]
    expected_repo = _policy_string(policy, "repo", "expected_repo")
    if expected_repo is not None:
        results.append(
            CheckResult(
                proof_claims.get("repo") == expected_repo,
                f"{label}.confidentiality_policy repo matches provider proof",
            )
        )
    results.extend(
        _policy_digest_binding_results(
            policy,
            proof_claims,
            claim_key="release_digest",
            exact_keys=("expected_release_digest", "release_digest"),
            allowed_keys=("allowed_release_digest", "allowed_release_digests"),
            label=label,
            description="release digest",
        )
    )
    for claim_key, exact_keys, allowed_keys, description in (
        (
            "code_measurement_fingerprint",
            (
                "expected_code_measurement_fingerprint",
                "code_measurement_fingerprint",
            ),
            (
                "allowed_code_measurement_fingerprint",
                "allowed_code_measurement_fingerprints",
                "allowed_code_measurements",
            ),
            "code measurement",
        ),
        (
            "enclave_measurement_fingerprint",
            (
                "expected_enclave_measurement_fingerprint",
                "enclave_measurement_fingerprint",
            ),
            (
                "allowed_enclave_measurement_fingerprint",
                "allowed_enclave_measurement_fingerprints",
                "allowed_enclave_measurements",
            ),
            "enclave measurement",
        ),
    ):
        results.extend(
            _policy_digest_binding_results(
                policy,
                proof_claims,
                claim_key=claim_key,
                exact_keys=exact_keys,
                allowed_keys=allowed_keys,
                label=label,
                description=description,
            )
        )
    if not model_ids:
        return results
    targets = _policy_targets(policy)
    model_attestations = proof_claims.get("model_attestations")
    results.extend(
        [
            CheckResult(
                policy.get("require_model_attestations") is True
                or policy.get("requireModelAttestations") is True,
                f"{label}.confidentiality_policy requires model attestations",
            ),
            CheckResult(
                isinstance(targets, dict)
                and {key for key in targets if isinstance(key, str)} == model_ids,
                f"{label}.confidentiality_policy model targets exactly match selected models",
            ),
        ]
    )
    if not isinstance(targets, dict) or not isinstance(model_attestations, dict):
        return results

    for model_id in sorted(model_ids):
        raw_target = targets.get(model_id)
        raw_claims = model_attestations.get(model_id)
        target = raw_target if isinstance(raw_target, dict) else {}
        claims = raw_claims if isinstance(raw_claims, dict) else {}
        results.extend(
            [
                CheckResult(
                    _policy_string(target, "repo", "expected_repo") is not None,
                    (
                        f"{label}.confidentiality_policy target {model_id} "
                        "repo is pinned"
                    ),
                ),
                CheckResult(
                    _policy_has_digest_pin(
                        target,
                        "expected_release_digest",
                        "release_digest",
                        "allowed_release_digest",
                        "allowed_release_digests",
                        "allowedReleaseDigests",
                    )
                    or _policy_has_digest_pin(
                        target,
                        "code_measurement_fingerprint",
                        "expected_code_measurement_fingerprint",
                        "code_measurement",
                        "expected_code_measurement",
                        "allowed_code_measurement_fingerprint",
                        "allowed_code_measurement_fingerprints",
                        "allowed_code_measurements",
                    )
                    or _policy_has_digest_pin(
                        target,
                        "enclave_measurement_fingerprint",
                        "expected_enclave_measurement_fingerprint",
                        "enclave_measurement",
                        "expected_enclave_measurement",
                        "allowed_enclave_measurement_fingerprint",
                        "allowed_enclave_measurement_fingerprints",
                        "allowed_enclave_measurements",
                    ),
                    (
                        f"{label}.confidentiality_policy target {model_id} "
                        "artifact identity is pinned"
                    ),
                ),
            ]
        )
        expected_repo = _policy_string(target, "repo", "expected_repo")
        if expected_repo is not None:
            results.append(
                CheckResult(
                    claims.get("repo") == expected_repo,
                    (
                        f"{label}.confidentiality_policy target {model_id} "
                        "repo matches model attestation"
                    ),
                )
            )
        expected_release = _policy_string(
            target,
            "expected_release_digest",
            "release_digest",
        )
        allowed_releases = {
            normalized
            for release in _policy_string_set(
                target,
                "allowed_release_digest",
                "allowed_release_digests",
                "allowedReleaseDigests",
            )
            if (normalized := _normalized_sha256_digest(release)) is not None
        }
        if expected_release is not None:
            results.append(
                CheckResult(
                    _normalized_sha256_digest(claims.get("release_digest"))
                    == _normalized_sha256_digest(expected_release),
                    (
                        f"{label}.confidentiality_policy target {model_id} "
                        "release digest matches model attestation"
                    ),
                )
            )
        if allowed_releases:
            results.append(
                CheckResult(
                    _normalized_sha256_digest(claims.get("release_digest"))
                    in allowed_releases,
                    (
                        f"{label}.confidentiality_policy target {model_id} "
                        "allowed release digests include model attestation"
                    ),
                )
            )
        for claim_key, policy_keys, description in (
            (
                "code_measurement_fingerprint",
                (
                    "code_measurement_fingerprint",
                    "expected_code_measurement_fingerprint",
                    "code_measurement",
                    "expected_code_measurement",
                ),
                "code measurement matches model attestation",
            ),
            (
                "enclave_measurement_fingerprint",
                (
                    "enclave_measurement_fingerprint",
                    "expected_enclave_measurement_fingerprint",
                    "enclave_measurement",
                    "expected_enclave_measurement",
                ),
                "enclave measurement matches model attestation",
            ),
        ):
            if _policy_string(target, *policy_keys) is None:
                continue
            results.append(
                CheckResult(
                    _claim_matches_policy_digest_pin(
                        claims,
                        target,
                        claim_key=claim_key,
                        policy_keys=policy_keys,
                    ),
                    (
                        f"{label}.confidentiality_policy target {model_id} "
                        f"{description}"
                    ),
                )
            )
        for claim_key, allowed_keys, description in (
            (
                "code_measurement_fingerprint",
                (
                    "allowed_code_measurement_fingerprint",
                    "allowed_code_measurement_fingerprints",
                    "allowed_code_measurements",
                ),
                "allowed code measurements include model attestation",
            ),
            (
                "enclave_measurement_fingerprint",
                (
                    "allowed_enclave_measurement_fingerprint",
                    "allowed_enclave_measurement_fingerprints",
                    "allowed_enclave_measurements",
                ),
                "allowed enclave measurements include model attestation",
            ),
        ):
            allowed_measurements = _policy_digest_set(target, *allowed_keys)
            if not allowed_measurements:
                continue
            results.append(
                CheckResult(
                    _normalized_sha256_digest(claims.get(claim_key))
                    in allowed_measurements,
                    (
                        f"{label}.confidentiality_policy target {model_id} "
                        f"{description}"
                    ),
                )
            )
    return results


def validate_ppq_private_policy_proof_binding(
    policy: dict[str, Any],
    proof_claims: dict[str, Any],
    *,
    model_ids: set[str],
    label: str,
) -> list[CheckResult]:
    results: list[CheckResult] = [
        CheckResult(
            bool(model_ids) and all(model_id.startswith("private/") for model_id in model_ids),
            f"{label}.confidentiality_policy selected models are PPQ private models",
        ),
        CheckResult(
            _policy_has_digest_pin(policy, "attestation_bundle_url_digest"),
            f"{label}.confidentiality_policy attestation bundle URL digest is pinned",
        ),
        CheckResult(
            _policy_has_digest_pin(
                policy,
                "expected_release_digest",
                "release_digest",
                "allowed_release_digest",
                "allowed_release_digests",
            )
            or _policy_has_digest_pin(
                policy,
                "expected_code_measurement_fingerprint",
                "code_measurement_fingerprint",
                "allowed_code_measurement_fingerprint",
                "allowed_code_measurement_fingerprints",
                "allowed_code_measurements",
            ),
            (
                f"{label}.confidentiality_policy release digest or code "
                "measurement is pinned"
            ),
        )
    ]
    expected_repo = _policy_string(policy, "repo", "expected_repo")
    if expected_repo is not None:
        results.append(
            CheckResult(
                proof_claims.get("repo") == expected_repo,
                f"{label}.confidentiality_policy repo matches provider proof",
            )
        )
    results.extend(
        _policy_digest_binding_results(
            policy,
            proof_claims,
            claim_key="release_digest",
            exact_keys=("expected_release_digest", "release_digest"),
            allowed_keys=("allowed_release_digest", "allowed_release_digests"),
            label=label,
            description="release digest",
        )
    )
    for claim_key, exact_keys, allowed_keys, description in (
        (
            "code_measurement_fingerprint",
            (
                "expected_code_measurement_fingerprint",
                "code_measurement_fingerprint",
            ),
            (
                "allowed_code_measurement_fingerprint",
                "allowed_code_measurement_fingerprints",
                "allowed_code_measurements",
            ),
            "code measurement",
        ),
        (
            "enclave_measurement_fingerprint",
            (
                "expected_enclave_measurement_fingerprint",
                "enclave_measurement_fingerprint",
            ),
            (
                "allowed_enclave_measurement_fingerprint",
                "allowed_enclave_measurement_fingerprints",
                "allowed_enclave_measurements",
            ),
            "enclave measurement",
        ),
        (
            "attestation_bundle_url_digest",
            ("attestation_bundle_url_digest",),
            (),
            "attestation bundle URL digest",
        ),
    ):
        results.extend(
            _policy_digest_binding_results(
                policy,
                proof_claims,
                claim_key=claim_key,
                exact_keys=exact_keys,
                allowed_keys=allowed_keys,
                label=label,
                description=description,
            )
        )
    if model_ids:
        targets = _policy_targets(policy)
        backend_model_attestations = proof_claims.get("backend_model_attestations")
        results.extend(
            [
                CheckResult(
                    policy.get("require_model_attestations") is True
                    or policy.get("requireModelAttestations") is True,
                    (
                        f"{label}.confidentiality_policy requires backend "
                        "model attestations"
                    ),
                ),
                CheckResult(
                    isinstance(targets, dict)
                    and {key for key in targets if isinstance(key, str)} == model_ids,
                    (
                        f"{label}.confidentiality_policy backend model targets "
                        "exactly match selected models"
                    ),
                ),
            ]
        )
        if isinstance(targets, dict) and isinstance(backend_model_attestations, dict):
            for model_id in sorted(model_ids):
                raw_target = targets.get(model_id)
                raw_claims = backend_model_attestations.get(model_id)
                target = raw_target if isinstance(raw_target, dict) else {}
                claims = raw_claims if isinstance(raw_claims, dict) else {}
                results.extend(
                    [
                        CheckResult(
                            _policy_string(target, "repo", "expected_repo") is not None,
                            (
                                f"{label}.confidentiality_policy target {model_id} "
                                "repo is pinned"
                            ),
                        ),
                        CheckResult(
                            _policy_has_digest_pin(
                                target,
                                "expected_release_digest",
                                "release_digest",
                                "allowed_release_digest",
                                "allowed_release_digests",
                                "allowedReleaseDigests",
                            )
                            or _policy_has_digest_pin(
                                target,
                                "code_measurement_fingerprint",
                                "expected_code_measurement_fingerprint",
                                "code_measurement",
                                "expected_code_measurement",
                                "allowed_code_measurement_fingerprint",
                                "allowed_code_measurement_fingerprints",
                                "allowed_code_measurements",
                            )
                            or _policy_has_digest_pin(
                                target,
                                "enclave_measurement_fingerprint",
                                "expected_enclave_measurement_fingerprint",
                                "enclave_measurement",
                                "expected_enclave_measurement",
                                "allowed_enclave_measurement_fingerprint",
                                "allowed_enclave_measurement_fingerprints",
                                "allowed_enclave_measurements",
                            ),
                            (
                                f"{label}.confidentiality_policy target {model_id} "
                                "artifact identity is pinned"
                            ),
                        ),
                    ]
                )
                expected_repo = _policy_string(target, "repo", "expected_repo")
                if expected_repo is not None:
                    results.append(
                        CheckResult(
                            claims.get("repo") == expected_repo,
                            (
                                f"{label}.confidentiality_policy target {model_id} "
                                "repo matches model attestation"
                            ),
                        )
                    )
                expected_release = _policy_string(
                    target,
                    "expected_release_digest",
                    "release_digest",
                )
                allowed_releases = {
                    normalized
                    for release in _policy_string_set(
                        target,
                        "allowed_release_digest",
                        "allowed_release_digests",
                        "allowedReleaseDigests",
                    )
                    if (normalized := _normalized_sha256_digest(release)) is not None
                }
                if expected_release is not None:
                    results.append(
                        CheckResult(
                            _normalized_sha256_digest(claims.get("release_digest"))
                            == _normalized_sha256_digest(expected_release),
                            (
                                f"{label}.confidentiality_policy target {model_id} "
                                "release digest matches model attestation"
                            ),
                        )
                    )
                if allowed_releases:
                    results.append(
                        CheckResult(
                            _normalized_sha256_digest(claims.get("release_digest"))
                            in allowed_releases,
                            (
                                f"{label}.confidentiality_policy target {model_id} "
                                "allowed release digests include model attestation"
                            ),
                        )
                    )
                for claim_key, policy_keys, description in (
                    (
                        "code_measurement_fingerprint",
                        (
                            "code_measurement_fingerprint",
                            "expected_code_measurement_fingerprint",
                            "code_measurement",
                            "expected_code_measurement",
                        ),
                        "code measurement matches model attestation",
                    ),
                    (
                        "enclave_measurement_fingerprint",
                        (
                            "enclave_measurement_fingerprint",
                            "expected_enclave_measurement_fingerprint",
                            "enclave_measurement",
                            "expected_enclave_measurement",
                        ),
                        "enclave measurement matches model attestation",
                    ),
                ):
                    if _policy_string(target, *policy_keys) is None:
                        continue
                    results.append(
                        CheckResult(
                            _claim_matches_policy_digest_pin(
                                claims,
                                target,
                                claim_key=claim_key,
                                policy_keys=policy_keys,
                            ),
                            (
                                f"{label}.confidentiality_policy target {model_id} "
                                f"{description}"
                            ),
                        )
                    )
                for claim_key, allowed_keys, description in (
                    (
                        "code_measurement_fingerprint",
                        (
                            "allowed_code_measurement_fingerprint",
                            "allowed_code_measurement_fingerprints",
                            "allowed_code_measurements",
                        ),
                        "allowed code measurements include model attestation",
                    ),
                    (
                        "enclave_measurement_fingerprint",
                        (
                            "allowed_enclave_measurement_fingerprint",
                            "allowed_enclave_measurement_fingerprints",
                            "allowed_enclave_measurements",
                        ),
                        "allowed enclave measurements include model attestation",
                    ),
                ):
                    allowed_measurements = _policy_digest_set(target, *allowed_keys)
                    if not allowed_measurements:
                        continue
                    results.append(
                        CheckResult(
                            _normalized_sha256_digest(claims.get(claim_key))
                            in allowed_measurements,
                            (
                                f"{label}.confidentiality_policy target {model_id} "
                                f"{description}"
                            ),
                        )
                    )
    return results


def validate_ppq_private_selected_model_proof_claims(
    proof_claims: dict[str, Any],
    *,
    model_ids: set[str],
    label: str,
) -> list[CheckResult]:
    selected_model_ids = strict_public_string_set(proof_claims.get("selected_model_ids"))
    return [
        CheckResult(
            bool(model_ids)
            and selected_model_ids is not None
            and selected_model_ids == model_ids,
            f"{label} provider proof_claims.selected_model_ids exactly match selected models",
        )
    ]


def validate_selected_model_proof_claims(
    proof_claims: dict[str, Any],
    *,
    model_ids: set[str],
    label: str,
) -> list[CheckResult]:
    selected_model_ids = strict_public_string_set(proof_claims.get("selected_model_ids"))
    return [
        CheckResult(
            bool(model_ids)
            and selected_model_ids is not None
            and selected_model_ids == model_ids,
            f"{label} provider proof_claims.selected_model_ids exactly match selected models",
        )
    ]


def validate_privatemode_policy_proof_binding(
    policy: dict[str, Any],
    proof_claims: dict[str, Any],
    *,
    model_ids: set[str],
    label: str,
) -> list[CheckResult]:
    results: list[CheckResult] = [
        CheckResult(
            bool(model_ids),
            f"{label}.confidentiality_policy selected models are exact",
        ),
        CheckResult(
            bool(
                _policy_has_digest_alias_pin(
                    policy, "manifest_digest", "manifestDigest"
                )
                or _policy_string_set(policy, "manifest_log_dir", "manifestLogDir")
            ),
            f"{label}.confidentiality_policy manifest identity is pinned",
        ),
        CheckResult(
            _policy_has_digest_alias_pin(
                policy, "proxy_binary_digest", "proxyBinaryDigest"
            ),
            f"{label}.confidentiality_policy proxy binary digest is pinned",
        )
    ]
    for claim_key, exact_keys, allowed_keys, description in (
        (
            "coordinator_measurement",
            ("expected_coordinator_measurement",),
            ("allowed_coordinator_measurements",),
            "coordinator measurement",
        ),
        (
            "secret_service_measurement",
            ("expected_secret_service_measurement",),
            ("allowed_secret_service_measurements",),
            "secret service measurement",
        ),
        (
            "ai_worker_measurement",
            ("expected_ai_worker_measurement",),
            ("allowed_ai_worker_measurements",),
            "AI worker measurement",
        ),
        (
            "key_release_binding",
            ("expected_key_release_binding",),
            ("allowed_key_release_bindings",),
            "key release binding",
        ),
        (
            "expected_workload_identity_digest",
            ("expected_workload_identity_digest",),
            (),
            "expected workload identity digest",
        ),
        (
            "model_workload_binding_digest",
            ("model_workload_binding_digest",),
            (),
            "model workload binding digest",
        ),
    ):
        results.extend(
            _policy_digest_binding_results(
                policy,
                proof_claims,
                claim_key=claim_key,
                exact_keys=exact_keys,
                allowed_keys=allowed_keys,
                label=label,
                description=description,
            )
        )
    for claim_key, alias_keys, description in (
        (
            "manifest_digest",
            ("manifest_digest", "manifestDigest"),
            "manifest digest",
        ),
        (
            "proxy_image_digest",
            ("proxy_image_digest", "proxyImageDigest"),
            "proxy image digest",
        ),
        (
            "proxy_binary_digest",
            ("proxy_binary_digest", "proxyBinaryDigest"),
            "proxy binary digest",
        ),
    ):
        results.extend(
            _policy_digest_alias_binding_results(
                policy,
                proof_claims,
                claim_key=claim_key,
                alias_keys=alias_keys,
                label=label,
                description=description,
            )
        )
    expected_gpu_policies = _policy_string_set(
        policy,
        "expected_gpu_attestation_policy",
        "allowed_gpu_attestation_policies",
    )
    results.append(
        CheckResult(
            expected_gpu_policies == {PRIVATEMODE_GPU_ATTESTATION_POLICY},
            (
                f"{label}.confidentiality_policy expected GPU attestation policy "
                f"is {PRIVATEMODE_GPU_ATTESTATION_POLICY}"
            ),
        )
    )
    results.append(
        CheckResult(
            proof_claims.get("gpu_attestation_policy")
            == PRIVATEMODE_GPU_ATTESTATION_POLICY,
            f"{label}.confidentiality_policy GPU attestation policy matches provider proof",
        )
    )
    expected_trust_tiers = _policy_string_set(policy, "expected_trust_tier")
    results.append(
        CheckResult(
            expected_trust_tiers == {"app-e2ee"}
            and proof_claims.get("trust_tier") == "app-e2ee",
            f"{label}.confidentiality_policy trust tier matches provider proof",
        )
    )
    if _policy_string_set(policy, "manifest_log_dir", "manifestLogDir"):
        results.append(
            require_digest(
                proof_claims.get("manifest_log_digest"),
                label=f"{label}.confidentiality_policy manifest log digest",
            )
        )
        results.append(
            require_digest(
                proof_claims.get("manifest_log_manifest_digest"),
                label=(
                    f"{label}.confidentiality_policy manifest log manifest digest"
                ),
            )
        )
    if any(
        key in policy
        for key in (
            "expected_workload_ids",
            "expectedWorkloadIDs",
            "expected_workload_sans",
            "expectedWorkloadSANs",
        )
    ):
        results.append(
            CheckResult(
                proof_claims.get("expected_workload_identity_digest")
                == _privatemode_expected_workload_identity_digest(policy),
                (
                    f"{label}.confidentiality_policy "
                    "expected workload identity digest matches provider proof"
                ),
            )
        )
    has_model_workload_bindings = any(
        key in policy for key in ("model_workload_bindings", "modelWorkloadBindings")
    )
    model_workload_digest = _privatemode_model_workload_binding_digest(policy)
    if has_model_workload_bindings:
        results.append(
            CheckResult(
                model_workload_digest is not None,
                f"{label}.confidentiality_policy model workload bindings are valid",
            )
        )
    if model_workload_digest is not None:
        bindings = _privatemode_model_workload_bindings(policy) or {}
        results.append(
            CheckResult(
                set(bindings) == model_ids,
                (
                    f"{label}.confidentiality_policy model workload bindings "
                    "exactly match selected models"
                ),
            )
        )
        expected_workloads = _privatemode_expected_workload_sets(policy)
        if expected_workloads is None:
            results.append(
                CheckResult(
                    False,
                    (
                        f"{label}.confidentiality_policy expected workloads are "
                        "valid"
                    ),
                )
            )
        else:
            expected_ids, expected_sans = expected_workloads
            bound_ids = {
                workload_id
                for binding in bindings.values()
                for workload_id in binding.get("workload_ids", [])
            }
            bound_sans = {
                workload_san
                for binding in bindings.values()
                for workload_san in binding.get("workload_sans", [])
            }
            results.append(
                CheckResult(
                    not (expected_ids - bound_ids or expected_sans - bound_sans),
                    (
                        f"{label}.confidentiality_policy expected workloads are "
                        "bound to selected models"
                    ),
                )
            )
        results.append(
            CheckResult(
                proof_claims.get("model_workload_binding_digest")
                == model_workload_digest,
                (
                    f"{label}.confidentiality_policy "
                    "model workload binding digest matches provider proof"
                ),
            )
        )
    elif has_model_workload_bindings:
        results.append(
            CheckResult(
                False,
                (
                    f"{label}.confidentiality_policy "
                    "model workload binding digest matches provider proof"
                ),
            )
        )
    return results


def validate_provider_policy_proof_binding(
    policy: dict[str, Any],
    confidentiality: dict[str, Any],
    *,
    mode: str | None,
    label: str,
) -> list[CheckResult]:
    proof_claims = confidentiality.get("proof_claims")
    if not isinstance(proof_claims, dict):
        return []
    model_ids = strict_public_string_set(confidentiality.get("model_ids")) or set()
    if mode == "tinfoil":
        return validate_tinfoil_policy_proof_binding(
            policy,
            proof_claims,
            model_ids=model_ids,
            label=label,
        )
    if mode == "ppq-private-tee":
        return validate_ppq_private_policy_proof_binding(
            policy,
            proof_claims,
            model_ids=model_ids,
            label=label,
        )
    if mode == "privatemode":
        return validate_privatemode_policy_proof_binding(
            policy,
            proof_claims,
            model_ids=model_ids,
            label=label,
        )
    return []


def validate_provider_proof_claims(
    confidentiality: dict[str, Any],
    *,
    label: str,
    public_policy: dict[str, Any] | None = None,
) -> list[CheckResult]:
    if confidentiality.get("verified") is not True:
        return []

    proof_claims = confidentiality.get("proof_claims")
    if not isinstance(proof_claims, dict):
        return [CheckResult(False, f"{label} provider proof_claims are present")]

    mode = str(confidentiality.get("mode") or "").strip().lower()
    results: list[CheckResult] = [
        require_digest(
            proof_claims.get("payload_policy_digest"),
            label=f"{label} provider proof_claims.payload_policy_digest",
        ),
        require_digest(
            proof_claims.get("payload_evidence_digest"),
            label=f"{label} provider proof_claims.payload_evidence_digest",
        ),
        CheckResult(
            proof_claims.get("payload_policy_digest")
            == confidentiality.get("policy_digest"),
            (
                f"{label} provider proof_claims.payload_policy_digest matches "
                "provider policy digest"
            ),
        ),
    ]
    if mode in {"tinfoil", "ppq-private-tee"}:
        results.append(
            CheckResult(
                proof_claims.get("transport") == "ehbp",
                f"{label} provider proof_claims.transport is ehbp",
            )
        )
        results.append(
            require_non_empty_string(
                proof_claims.get("repo"),
                label=f"{label} provider proof_claims.repo",
            )
        )
        for claim in (
            "enclave_measurement_fingerprint",
            "code_measurement_fingerprint",
        ):
            results.append(
                require_sha256_digest_value(
                    proof_claims.get(claim),
                    label=f"{label} provider proof_claims.{claim}",
                )
            )
        results.append(
            require_hex_string(
                proof_claims.get("attested_hpke_public_key_hex"),
                label=f"{label} provider proof_claims.attested_hpke_public_key_hex",
                length=64,
            )
        )
        results.extend(
            require_optional_tls_digest_alias_group(
                proof_claims,
                label=f"{label} provider proof_claims",
                required=(
                    mode != "tinfoil"
                    or tinfoil_tls_public_key_binding_required(public_policy)
                ),
            )
        )
        if mode == "tinfoil":
            for claim in ("attestation_format", "attestation_report_digest"):
                checker = (
                    require_digest
                    if claim == "attestation_report_digest"
                    else require_non_empty_string
                )
                results.append(
                    checker(
                        proof_claims.get(claim),
                        label=f"{label} provider proof_claims.{claim}",
                    )
                )
            results.append(
                require_digest(
                    proof_claims.get("release_digest"),
                    label=f"{label} provider proof_claims.release_digest",
                )
            )
            results.extend(
                validate_tinfoil_model_attestation_proof_claims(
                    proof_claims,
                    model_ids=strict_public_string_set(
                        confidentiality.get("model_ids")
                    )
                    or set(),
                    label=label,
                    public_policy=public_policy,
                )
            )
        if mode == "ppq-private-tee":
            results.append(
                CheckResult(
                    proof_claims.get("client_encryption_boundary")
                    == "routstr-tee-ehbp-proxy",
                    (
                        f"{label} provider proof_claims.client_encryption_boundary "
                        "is routstr-tee-ehbp-proxy"
                    ),
                )
            )
            results.append(
                require_digest(
                    proof_claims.get("attestation_bundle_url_digest"),
                    label=(
                        f"{label} provider proof_claims.attestation_bundle_url_digest"
                    ),
                )
            )
            results.extend(
                validate_ppq_private_selected_model_proof_claims(
                    proof_claims,
                    model_ids=strict_public_string_set(
                        confidentiality.get("model_ids")
                    )
                    or set(),
                    label=label,
                )
            )
            results.extend(
                validate_tinfoil_model_attestation_proof_claims(
                    proof_claims,
                    model_ids=strict_public_string_set(
                        confidentiality.get("model_ids")
                    )
                    or set(),
                    label=label,
                    claim_key="backend_model_attestations",
                )
            )
        results.extend(
            validate_provider_verification_steps(
                proof_claims,
                label=f"{label} provider",
                required_steps=REQUIRED_EHBP_PROVIDER_VERIFICATION_STEPS,
            )
        )
        return results

    if mode == "privatemode":
        results.append(
            CheckResult(
                proof_claims.get("transport") == "privatemode-proxy",
                f"{label} provider proof_claims.transport is privatemode-proxy",
            )
        )
        results.append(
            CheckResult(
                proof_claims.get("trust_tier") == "app-e2ee",
                f"{label} provider proof_claims.trust_tier is app-e2ee",
            )
        )
        results.extend(
            require_digest_alias_group(
                proof_claims,
                ("manifest_digest", "manifest_log_manifest_digest"),
                label=f"{label} provider proof_claims",
            )
        )
        results.append(
            require_matching_digest_alias_group(
                proof_claims,
                ("manifest_digest", "manifest_log_manifest_digest"),
                label=f"{label} provider proof_claims",
            )
        )
        for claim in (
            "coordinator_measurement",
            "secret_service_measurement",
            "ai_worker_measurement",
            "key_release_binding",
        ):
            results.append(
                require_digest(
                    proof_claims.get(claim),
                    label=f"{label} provider proof_claims.{claim}",
                )
            )
        results.append(
            CheckResult(
                proof_claims.get("gpu_attestation_policy")
                == PRIVATEMODE_GPU_ATTESTATION_POLICY,
                (
                    f"{label} provider proof_claims.gpu_attestation_policy "
                    f"is {PRIVATEMODE_GPU_ATTESTATION_POLICY}"
                ),
            )
        )
        if "proxy_image_digest" in proof_claims:
            results.append(
                require_digest(
                    proof_claims.get("proxy_image_digest"),
                    label=f"{label} provider proof_claims.proxy_image_digest",
                )
            )
        results.append(
            require_digest(
                proof_claims.get("proxy_binary_digest"),
                label=f"{label} provider proof_claims.proxy_binary_digest",
            )
        )
        for claim in REQUIRED_PRIVATEMODE_PROVIDER_PROOF_DIGEST_CLAIMS:
            results.append(
                require_digest(
                    proof_claims.get(claim),
                    label=f"{label} provider proof_claims.{claim}",
                )
            )
        results.append(
            CheckResult(
                proof_claims.get("key_release_binding")
                == privatemode_key_release_binding_digest(proof_claims),
                f"{label} provider proof_claims.key_release_binding binds components",
            )
        )
        results.extend(
            validate_selected_model_proof_claims(
                proof_claims,
                model_ids=strict_public_string_set(confidentiality.get("model_ids"))
                or set(),
                label=label,
            )
        )
        results.extend(
            validate_provider_verification_steps(
                proof_claims,
                label=f"{label} provider",
                required_steps=REQUIRED_PRIVATEMODE_PROVIDER_VERIFICATION_STEPS,
            )
        )
        return results

    return [
        CheckResult(
            False, f"{label} provider proof mode {mode or '<missing>'} is supported"
        )
    ]


def validate_client_confidentiality_boundary(
    value: Any,
    *,
    label: str,
    proof_claims: dict[str, Any] | None = None,
    proof_label: str = "local TEE proof",
) -> list[CheckResult]:
    if not isinstance(value, dict):
        return [CheckResult(False, f"{label} is present")]
    results = [
        require_non_empty_string(
            value.get("mode"),
            label=f"{label}.mode",
        ),
        CheckResult(
            value.get("mode") == "attested-tls-termination",
            f"{label}.mode is attested-tls-termination",
        ),
        require_true_boolean(
            value.get("tls_terminates_in_attested_tee"),
            label=f"{label}.tls_terminates_in_attested_tee",
        ),
        CheckResult(
            value.get("inbound_ehbp_ohttp_request_decryption") is False,
            f"{label}.inbound_ehbp_ohttp_request_decryption is false",
        ),
        require_digest(
            value.get("attested_tls_public_key_digest"),
            label=f"{label}.attested_tls_public_key_digest",
        ),
    ]
    if proof_claims is not None:
        results.append(
            CheckResult(
                value.get("attested_tls_public_key_digest")
                == proof_claims.get("public_key_digest"),
                f"{label}.attested_tls_public_key_digest matches {proof_label}",
            )
        )
    return results


def validate_routstr_tee(status: dict[str, Any]) -> list[CheckResult]:
    results: list[CheckResult] = []
    tee = status.get("routstr_tee")
    if not isinstance(tee, dict):
        return [CheckResult(False, "routstr_tee status is missing")]

    results.append(
        require_true_boolean(
            tee.get("required"),
            label="local Routstr TEE attestation is required",
        )
    )
    results.append(
        require_true_boolean(
            tee.get("ready"),
            label="local Routstr TEE is ready",
        )
    )
    results.extend(
        validate_client_confidentiality_boundary(
            tee.get("client_confidentiality"),
            label="local Routstr TEE client_confidentiality",
        )
    )
    results.append(
        require_digest(
            tee.get("attestation_evidence_digest"),
            label="local TEE evidence digest",
        )
    )
    results.append(
        require_digest(tee.get("hpke_key_config_digest"), label="local HPKE key digest")
    )
    local_verification = tee.get("local_verification")
    if not isinstance(local_verification, dict):
        results.append(CheckResult(False, "local TEE verification status is missing"))
        return results
    proof_claims = local_verification.get("proof_claims")
    if isinstance(proof_claims, dict):
        results.extend(
            validate_client_confidentiality_boundary(
                tee.get("client_confidentiality"),
                label="local Routstr TEE client_confidentiality",
                proof_claims=proof_claims,
            )
        )
    results.append(
        require_true_boolean(
            local_verification.get("verified"),
            label="local TEE verifier reports verified",
        )
    )
    results.append(
        require_not_future_timestamp(
            local_verification.get("verified_at"),
            label="local TEE verifier verified_at",
        )
    )
    results.append(
        require_future_timestamp(
            local_verification.get("expires_at"),
            label="local TEE verifier expiry",
        )
    )
    results.append(
        require_digest(
            local_verification.get("verified_claims_digest"),
            label="local TEE verified claims digest",
        )
    )
    results.append(
        require_digest(
            local_verification.get("evidence_digest"),
            label="local TEE runtime evidence digest",
        )
    )
    results.extend(
        validate_local_tee_proof_claims(
            local_verification,
            label="local TEE status",
            hpke_key_config_digest=tee.get("hpke_key_config_digest"),
            hpke_public_key_digest=tee.get("hpke_public_key_digest"),
        )
    )
    return results


def validate_models_routstr_tee(
    *,
    status: dict[str, Any],
    models: dict[str, Any],
) -> list[CheckResult]:
    data = models.get("data")
    has_confidential_models = (
        isinstance(data, list)
        and any(
            isinstance(item, dict)
            and (
                item.get("confidential") is True
                or isinstance(item.get("confidentiality"), dict)
            )
            for item in data
        )
    )
    if status.get("required") is not True and not has_confidential_models:
        return []

    tee = models.get("routstr_tee")
    if not isinstance(tee, dict):
        return [CheckResult(False, "/v1/models routstr_tee status is present")]

    results: list[CheckResult] = [
        require_true_boolean(
            tee.get("required"),
            label="/v1/models local Routstr TEE attestation is required",
        ),
        require_true_boolean(
            tee.get("ready"),
            label="/v1/models local Routstr TEE is ready",
        ),
    ]
    results.extend(
        validate_client_confidentiality_boundary(
            tee.get("client_confidentiality"),
            label="/v1/models local Routstr TEE client_confidentiality",
        )
    )

    status_tee = status.get("routstr_tee")
    if isinstance(status_tee, dict):
        for field in (
            "client_confidentiality",
            "attestation_evidence_digest",
            "hpke_key_config_digest",
            "hpke_public_key_digest",
        ):
            results.append(
                CheckResult(
                    tee.get(field) == status_tee.get(field),
                    f"/v1/models routstr_tee.{field} matches status",
                )
            )

    local_verification = tee.get("local_verification")
    if not isinstance(local_verification, dict):
        results.append(
            CheckResult(False, "/v1/models local TEE verification status is missing")
        )
        return results
    results.append(
        require_true_boolean(
            local_verification.get("verified"),
            label="/v1/models local TEE verifier reports verified",
        )
    )
    if isinstance(status_tee, dict):
        status_local_verification = status_tee.get("local_verification")
        if isinstance(status_local_verification, dict):
            for field, label, verb in (
                ("verified_at", "verified_at", "matches"),
                ("expires_at", "expires_at", "matches"),
                ("evidence_digest", "evidence digest", "matches"),
                ("verified_claims_digest", "verified claims digest", "matches"),
                ("proof_claims", "proof claims", "match"),
            ):
                results.append(
                    CheckResult(
                        local_verification.get(field)
                        == status_local_verification.get(field),
                        f"/v1/models local TEE {label} {verb} status",
                    )
                )
    return results


def provider_matches(
    provider: dict[str, Any],
    *,
    expected_provider: str,
    expected_model: str,
) -> bool:
    if not provider_confidentiality_mode_matches(provider):
        return False
    provider_type = public_string_value(provider.get("provider_type"))
    upstream_name = public_string_value(provider.get("upstream_name"))
    confidentiality = provider.get("confidentiality")
    if not isinstance(confidentiality, dict):
        return False
    strict_model_ids = strict_public_string_set(confidentiality.get("model_ids"))
    model_ids = strict_model_ids or public_string_set(confidentiality.get("model_ids"))
    model_prefixes = public_string_set(confidentiality.get("model_id_prefixes"))
    provider_names = {
        provider_name
        for provider_name in (provider_type, upstream_name)
        if provider_name is not None
    }
    if expected_provider not in provider_names:
        return False
    if provider_type in CONFIDENTIAL_PROVIDER_MODES:
        return strict_model_ids is not None and expected_model in strict_model_ids
    return expected_model in model_ids or any(
        expected_model.startswith(prefix) for prefix in model_prefixes
    )


def confidentiality_snapshot_matches(
    confidentiality: dict[str, Any],
    status_confidentiality: dict[str, Any],
) -> bool:
    return (
        confidentiality.get("enabled") == status_confidentiality.get("enabled")
        and confidentiality.get("verified") == status_confidentiality.get("verified")
        and confidentiality.get("mode") == status_confidentiality.get("mode")
        and confidentiality.get("verifier") == status_confidentiality.get("verifier")
        and confidentiality.get("verified_at")
        == status_confidentiality.get("verified_at")
        and confidentiality.get("expires_at")
        == status_confidentiality.get("expires_at")
        and confidentiality.get("policy_digest")
        == status_confidentiality.get("policy_digest")
        and confidentiality.get("evidence_digest")
        == status_confidentiality.get("evidence_digest")
        and confidentiality.get("verified_claims_digest")
        == status_confidentiality.get("verified_claims_digest")
        and strict_public_string_sets_match(
            confidentiality.get("model_ids"),
            status_confidentiality.get("model_ids"),
        )
        and strict_public_string_sets_match(
            confidentiality.get("model_id_prefixes"),
            status_confidentiality.get("model_id_prefixes"),
        )
        and isinstance(confidentiality.get("proof_claims"), dict)
        and confidentiality.get("proof_claims")
        == status_confidentiality.get("proof_claims")
    )


def model_confidentiality_matches_provider_status(
    *,
    provider: dict[str, Any],
    model_confidentiality: dict[str, Any],
    model_evidence_digest: Any,
) -> bool:
    status_confidentiality = provider.get("confidentiality")
    if not isinstance(status_confidentiality, dict):
        return False
    provider_supported_endpoints = public_string_set(
        provider.get("supported_endpoints")
    )
    model_supported_endpoints = public_string_set(
        model_confidentiality.get("supported_endpoints")
    )
    return (
        model_confidentiality.get("evidence_digest") == model_evidence_digest
        and endpoints_cover(provider_supported_endpoints, model_supported_endpoints)
        and confidentiality_snapshot_matches(
            model_confidentiality,
            status_confidentiality,
        )
    )


def provider_confidentiality_mode_matches(provider: dict[str, Any]) -> bool:
    provider_type = public_string_value(provider.get("provider_type"))
    confidentiality = provider.get("confidentiality")
    if provider_type is None or not isinstance(confidentiality, dict):
        return False
    mode = public_string_value(confidentiality.get("mode"))
    return CONFIDENTIAL_PROVIDER_MODES.get(provider_type) == mode


def model_confidentiality_mode_matches(confidentiality: dict[str, Any]) -> bool:
    provider_type = public_string_value(confidentiality.get("provider_type"))
    mode = public_string_value(confidentiality.get("mode"))
    if provider_type is None or mode is None:
        return False
    return CONFIDENTIAL_PROVIDER_MODES.get(provider_type) == mode


def validate_provider_mode_contracts(status: dict[str, Any]) -> list[CheckResult]:
    providers = status.get("providers")
    if not isinstance(providers, list):
        return []
    results: list[CheckResult] = []
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        confidentiality = provider.get("confidentiality")
        if not isinstance(confidentiality, dict):
            continue
        if confidentiality.get("verified") is not True:
            continue
        provider_type = public_string_value(provider.get("provider_type"))
        mode = public_string_value(confidentiality.get("mode"))
        results.append(
            CheckResult(
                provider_confidentiality_mode_matches(provider),
                "provider confidentiality mode "
                f"{mode or '<missing>'} matches provider_type "
                f"{provider_type or '<missing>'}",
            )
        )
    return results


def provider_status_label(provider: dict[str, Any], index: int) -> str:
    upstream_name = public_string_value(provider.get("upstream_name"))
    if upstream_name is not None:
        return upstream_name
    provider_type = public_string_value(provider.get("provider_type"))
    if provider_type is not None:
        return provider_type
    return f"providers[{index}]"


def provider_policy_required_for_public_verification(mode: str | None) -> bool:
    return mode in {"tinfoil", "ppq-private-tee", "privatemode"}


def validate_provider_confidentiality_statuses(
    status: dict[str, Any],
) -> list[CheckResult]:
    providers = status.get("providers")
    if not isinstance(providers, list):
        return []

    results: list[CheckResult] = []
    for index, provider in enumerate(providers):
        if not isinstance(provider, dict):
            continue
        confidentiality = provider.get("confidentiality")
        if not isinstance(confidentiality, dict):
            continue
        label = provider_status_label(provider, index)
        results.append(
            require_json_boolean(
                confidentiality.get("enabled"),
                label=f"{label} provider confidentiality.enabled",
            )
        )
        results.append(
            require_json_boolean(
                confidentiality.get("verified"),
                label=f"{label} provider confidentiality.verified",
            )
        )
        if confidentiality.get("verified") is not True:
            continue
        provider_type = public_string_value(provider.get("provider_type"))
        results.extend(
            validate_known_provider_exact_selectors(
                confidentiality,
                provider_type=provider_type,
                label=f"{label} provider",
            )
        )
        results.append(
            require_true_boolean(
                confidentiality.get("enabled"),
                label=f"{label} provider confidentiality is enabled",
            )
        )
        results.append(
            require_digest(
                confidentiality.get("policy_digest"),
                label=f"{label} provider policy digest",
            )
        )
        results.append(
            require_digest(
                confidentiality.get("evidence_digest"),
                label=f"{label} provider evidence digest",
            )
        )
        results.append(
            require_public_identity_string(
                confidentiality.get("verifier"),
                label=f"{label} provider verifier",
            )
        )
        results.append(
            require_not_future_timestamp(
                confidentiality.get("verified_at"),
                label=f"{label} provider verifier verified_at",
            )
        )
        results.append(
            require_future_timestamp(
                confidentiality.get("expires_at"),
                label=f"{label} provider verifier expiry",
            )
        )
        results.append(
            require_digest(
                confidentiality.get("verified_claims_digest"),
                label=f"{label} provider verified claims digest",
            )
        )
        confidentiality_policy = provider.get("confidentiality_policy")
        public_policy = (
            confidentiality_policy if isinstance(confidentiality_policy, dict) else None
        )
        results.extend(
            validate_provider_proof_claims(
                confidentiality,
                label=label,
                public_policy=public_policy,
            )
        )
        confidentiality_mode = public_string_value(confidentiality.get("mode"))
        if confidentiality_policy is None and provider_policy_required_for_public_verification(
            confidentiality_mode
        ):
            results.append(
                CheckResult(
                    False,
                    (
                        f"{label}.confidentiality_policy is required for verified "
                        f"{confidentiality_mode}"
                    ),
                )
            )
        elif confidentiality_policy is not None:
            if not isinstance(confidentiality_policy, dict):
                results.append(
                    CheckResult(False, f"{label}.confidentiality_policy is an object")
                )
            else:
                results.extend(
                    validate_provider_policy_proof_binding(
                        confidentiality_policy,
                        confidentiality,
                        mode=confidentiality_mode,
                        label=label,
                    )
                )
    return results


def validate_expected_provider(
    status: dict[str, Any],
    *,
    expected_provider: str,
    expected_model: str,
) -> list[CheckResult]:
    providers = status.get("providers")
    if not isinstance(providers, list):
        return [CheckResult(False, "providers status list is missing")]

    matching = [
        provider
        for provider in providers
        if isinstance(provider, dict)
        and provider_matches(
            provider,
            expected_provider=expected_provider,
            expected_model=expected_model,
        )
    ]
    if not matching:
        return [
            CheckResult(
                False,
                f"{expected_provider} has no status covering {expected_model}",
            )
        ]

    results: list[CheckResult] = []
    for provider in matching:
        confidentiality = provider.get("confidentiality")
        if not isinstance(confidentiality, dict):
            results.append(CheckResult(False, f"{expected_provider} status is invalid"))
            continue
        public_policy = provider.get("confidentiality_policy")
        if not isinstance(public_policy, dict):
            public_policy = None
        label = f"{expected_provider}/{expected_model}"
        results.append(
            require_true_boolean(
                confidentiality.get("enabled"),
                label=f"{label} confidentiality is enabled",
            )
        )
        results.append(
            require_true_boolean(
                confidentiality.get("verified"),
                label=f"{label} provider verifier reports verified",
            )
        )
        results.append(
            require_digest(
                confidentiality.get("policy_digest"),
                label=f"{label} policy digest",
            )
        )
        results.append(
            require_digest(
                confidentiality.get("evidence_digest"),
                label=f"{label} evidence digest",
            )
        )
        results.append(
            require_public_identity_string(
                confidentiality.get("verifier"),
                label=f"{label} provider verifier",
            )
        )
        results.append(
            require_not_future_timestamp(
                confidentiality.get("verified_at"),
                label=f"{label} provider verifier verified_at",
            )
        )
        results.append(
            require_future_timestamp(
                confidentiality.get("expires_at"),
                label=f"{label} provider verifier expiry",
            )
        )
        results.append(
            require_digest(
                confidentiality.get("verified_claims_digest"),
                label=f"{label} verified claims digest",
            )
        )
        results.extend(
            validate_provider_proof_claims(
                confidentiality,
                label=label,
                public_policy=public_policy,
            )
        )
    return results


def _model_provider_names(model: dict[str, Any]) -> set[str]:
    provider_names = {
        provider_name
        for provider_name in (public_string_value(model.get("attestation_provider")),)
        if provider_name is not None
    }
    confidentiality = model.get("confidentiality")
    if isinstance(confidentiality, dict):
        for provider_name in (
            public_string_value(confidentiality.get("provider_type")),
        ):
            if provider_name is not None:
                provider_names.add(provider_name)
    return provider_names


def model_attestation_provider_matches_provider_type(
    model: dict[str, Any],
    confidentiality: dict[str, Any],
) -> bool:
    return public_string_value(
        model.get("attestation_provider")
    ) == public_string_value(confidentiality.get("provider_type"))


def validate_public_identity_fields(
    *,
    status: dict[str, Any],
    models: dict[str, Any],
) -> list[CheckResult]:
    results: list[CheckResult] = []

    providers = status.get("providers")
    if isinstance(providers, list):
        for provider in providers:
            if not isinstance(provider, dict):
                continue
            results.append(
                require_public_identity_string(
                    provider.get("provider_type"),
                    label="provider provider_type",
                )
            )
            if provider.get("upstream_name") is not None:
                results.append(
                    require_public_identity_string(
                        provider.get("upstream_name"),
                        label="provider upstream_name",
                    )
                )
            confidentiality = provider.get("confidentiality")
            if isinstance(confidentiality, dict):
                results.append(
                    require_public_identity_string(
                        confidentiality.get("mode"),
                        label="provider confidentiality.mode",
                    )
                )

    data = models.get("data")
    if isinstance(data, list):
        for index, model in enumerate(data):
            if not isinstance(model, dict):
                continue
            label = f"/v1/models data[{index}]"
            results.append(
                require_public_identity_string(
                    model.get("id"),
                    label=f"{label}.id",
                )
            )
            confidentiality = model.get("confidentiality")
            has_confidential_metadata = (
                model.get("confidential") is True
                or model.get("attestation_provider") is not None
                or isinstance(confidentiality, dict)
            )
            if has_confidential_metadata:
                results.append(
                    require_public_identity_string(
                        model.get("attestation_provider"),
                        label=f"{label}.attestation_provider",
                    )
                )
            if isinstance(confidentiality, dict):
                results.append(
                    require_public_identity_string(
                        confidentiality.get("provider_type"),
                        label=f"{label}.confidentiality.provider_type",
                    )
                )
                results.append(
                    require_public_identity_string(
                        confidentiality.get("mode"),
                        label=f"{label}.confidentiality.mode",
                    )
                )

    if not results:
        return [CheckResult(True, "public identity metadata is well formed")]
    return results


def validate_public_metadata_lists(
    *,
    status: dict[str, Any],
    models: dict[str, Any],
) -> list[CheckResult]:
    results: list[CheckResult] = []

    providers = status.get("providers")
    if isinstance(providers, list):
        for provider in providers:
            if not isinstance(provider, dict):
                continue
            results.extend(
                validate_public_string_list(
                    provider.get("supported_endpoints"),
                    label="provider supported_endpoints",
                )
            )
            confidentiality = provider.get("confidentiality")
            if not isinstance(confidentiality, dict):
                continue
            results.extend(
                validate_public_string_list(
                    confidentiality.get("model_ids"),
                    label="provider model_ids",
                )
            )
            results.extend(
                validate_public_string_list(
                    confidentiality.get("model_id_prefixes"),
                    label="provider model_id_prefixes",
                )
            )

    data = models.get("data")
    if isinstance(data, list):
        for model in data:
            if not isinstance(model, dict):
                continue
            confidentiality = model.get("confidentiality")
            if not isinstance(confidentiality, dict):
                continue
            results.extend(
                validate_public_string_list(
                    confidentiality.get("supported_endpoints"),
                    label="model supported_endpoints",
                )
            )

    if not results:
        results.append(CheckResult(True, "public string-list metadata is well formed"))
    return results


def provider_policy_snapshots_by_digest(status: dict[str, Any]) -> dict[str, dict[str, Any]]:
    policies: dict[str, dict[str, Any]] = {}
    providers = status.get("providers")
    if not isinstance(providers, list):
        return policies
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        confidentiality = provider.get("confidentiality")
        policy = provider.get("confidentiality_policy")
        if not isinstance(confidentiality, dict) or not isinstance(policy, dict):
            continue
        policy_digest = confidentiality.get("policy_digest")
        if isinstance(policy_digest, str) and policy_digest.strip():
            policies[policy_digest.strip()] = policy
    return policies


def validate_model_confidentiality_statuses(
    models: dict[str, Any],
    *,
    provider_policies_by_digest: dict[str, dict[str, Any]] | None = None,
) -> list[CheckResult]:
    data = models.get("data")
    if not isinstance(data, list):
        return []

    results: list[CheckResult] = []
    for index, model in enumerate(data):
        if not isinstance(model, dict):
            continue
        model_id = public_string_value(model.get("id")) or f"/v1/models data[{index}]"
        confidentiality = model.get("confidentiality")
        if not isinstance(confidentiality, dict):
            continue

        results.append(
            require_json_boolean(
                confidentiality.get("enabled"),
                label=f"{model_id} nested confidentiality.enabled",
            )
        )
        results.append(
            require_json_boolean(
                confidentiality.get("verified"),
                label=f"{model_id} nested confidentiality.verified",
            )
        )
        if model.get("confidential") is True:
            results.append(
                require_true_boolean(
                    confidentiality.get("enabled"),
                    label=f"{model_id} nested confidentiality is enabled",
                )
            )
            results.append(
                require_true_boolean(
                    confidentiality.get("verified"),
                    label=f"{model_id} nested confidentiality is verified",
                )
            )
        if confidentiality.get("verified") is not True:
            continue

        results.append(
            CheckResult(
                model.get("attestation_status") == "verified",
                f"{model_id} model attestation status is verified",
            )
        )
        results.append(
            require_digest(
                model.get("attestation_evidence_digest"),
                label=f"{model_id} model evidence digest",
            )
        )
        results.append(
            require_digest(
                confidentiality.get("policy_digest"),
                label=f"{model_id} nested policy digest",
            )
        )
        nested_evidence_digest = confidentiality.get("evidence_digest")
        results.append(
            require_digest(
                nested_evidence_digest,
                label=f"{model_id} nested evidence digest",
            )
        )
        results.append(
            CheckResult(
                nested_evidence_digest == model.get("attestation_evidence_digest"),
                f"{model_id} nested model evidence digest matches top-level metadata",
            )
        )
        results.append(
            require_digest(
                confidentiality.get("verified_claims_digest"),
                label=f"{model_id} nested verified claims digest",
            )
        )
        results.append(
            require_public_identity_string(
                confidentiality.get("verifier"),
                label=f"{model_id} nested verifier",
            )
        )
        results.append(
            require_not_future_timestamp(
                confidentiality.get("verified_at"),
                label=f"{model_id} nested verifier verified_at",
            )
        )
        results.append(
            require_future_timestamp(
                confidentiality.get("expires_at"),
                label=f"{model_id} nested verifier expiry",
            )
        )
        attestation_provider = public_string_value(model.get("attestation_provider"))
        results.append(
            CheckResult(
                attestation_provider is not None
                and model_attestation_provider_matches_provider_type(
                    model,
                    confidentiality,
                ),
                (
                    f"{model_id} model attestation provider matches nested "
                    "confidentiality provider_type"
                ),
            )
        )
        provider_type = public_string_value(confidentiality.get("provider_type"))
        mode = public_string_value(confidentiality.get("mode"))
        results.append(
            CheckResult(
                model_confidentiality_mode_matches(confidentiality),
                f"{model_id} model confidentiality mode "
                f"{mode or '<missing>'} matches provider_type "
                f"{provider_type or '<missing>'}",
            )
        )
        results.extend(
            validate_provider_proof_claims(
                confidentiality,
                label=f"{model_id} nested",
                public_policy=(
                    (provider_policies_by_digest or {}).get(
                        str(confidentiality.get("policy_digest") or "").strip()
                    )
                ),
            )
        )
    return results


def validate_models(
    models: dict[str, Any],
    expected: list[tuple[str, str]],
    *,
    require_all_confidential: bool = False,
) -> list[CheckResult]:
    data = models.get("data")
    if not isinstance(data, list):
        return [CheckResult(False, "/v1/models response is missing data list")]

    by_id = {
        model_id: item
        for item in data
        if isinstance(item, dict)
        for model_id in [public_string_value(item.get("id"))]
        if model_id is not None
    }
    results: list[CheckResult] = []
    if require_all_confidential:
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = public_string_value(item.get("id")) or "<unknown>"
            confidentiality = item.get("confidentiality")
            is_verified_confidential = (
                item.get("confidential") is True
                and item.get("attestation_status") == "verified"
                and item.get("provider_attestation_status") == "verified"
                and isinstance(confidentiality, dict)
                and confidentiality.get("enabled") is True
                and confidentiality.get("verified") is True
            )
            results.append(
                CheckResult(
                    is_verified_confidential,
                    f"{model_id} is not a verified confidential model",
                )
            )
    for expected_provider, model_id in expected:
        model = by_id.get(model_id)
        if model is None:
            results.append(CheckResult(False, f"{model_id} is missing from /v1/models"))
            continue
        results.append(
            require_true_boolean(
                model.get("confidential"),
                label=f"{model_id} is confidential",
            )
        )
        results.append(
            CheckResult(
                model.get("attestation_status") == "verified",
                f"{model_id} model attestation status is verified",
            )
        )
        results.append(
            CheckResult(
                model.get("provider_attestation_status") == "verified",
                f"{model_id} model provider attestation status is verified",
            )
        )
        results.append(
            require_digest(
                model.get("attestation_evidence_digest"),
                label=f"{model_id} model evidence digest",
            )
        )
        results.append(
            CheckResult(
                expected_provider in _model_provider_names(model),
                f"{model_id} model attestation provider matches {expected_provider}",
            )
        )
        confidentiality = model.get("confidentiality")
        if not isinstance(confidentiality, dict):
            results.append(
                CheckResult(
                    False,
                    f"{model_id} nested confidentiality metadata is present",
                )
            )
            continue
        attestation_provider = public_string_value(model.get("attestation_provider"))
        results.append(
            CheckResult(
                attestation_provider is not None
                and model_attestation_provider_matches_provider_type(
                    model,
                    confidentiality,
                ),
                (
                    f"{model_id} model attestation provider matches nested "
                    "confidentiality provider_type"
                ),
            )
        )
        provider_type = public_string_value(confidentiality.get("provider_type"))
        mode = public_string_value(confidentiality.get("mode"))
        results.append(
            CheckResult(
                model_confidentiality_mode_matches(confidentiality),
                f"{model_id} model confidentiality mode "
                f"{mode or '<missing>'} matches provider_type "
                f"{provider_type or '<missing>'}",
            )
        )
        results.append(
            require_true_boolean(
                confidentiality.get("enabled"),
                label=f"{model_id} nested confidentiality is enabled",
            )
        )
        results.append(
            require_true_boolean(
                confidentiality.get("verified"),
                label=f"{model_id} nested confidentiality is verified",
            )
        )
        results.append(
            require_digest(
                confidentiality.get("policy_digest"),
                label=f"{model_id} nested policy digest",
            )
        )
        results.append(
            require_digest(
                confidentiality.get("evidence_digest"),
                label=f"{model_id} nested evidence digest",
            )
        )
        results.append(
            require_digest(
                confidentiality.get("verified_claims_digest"),
                label=f"{model_id} nested verified claims digest",
            )
        )
        results.append(
            require_public_identity_string(
                confidentiality.get("verifier"),
                label=f"{model_id} nested verifier",
            )
        )
        results.append(
            require_not_future_timestamp(
                confidentiality.get("verified_at"),
                label=f"{model_id} nested verifier verified_at",
            )
        )
        results.append(
            require_future_timestamp(
                confidentiality.get("expires_at"),
                label=f"{model_id} nested verifier expiry",
            )
        )
    return results


def validate_attestation_routing_policy_confidentiality(
    confidentiality: dict[str, Any],
    *,
    label: str,
    public_policy: dict[str, Any] | None = None,
) -> list[CheckResult]:
    results: list[CheckResult] = [
        require_json_boolean(
            confidentiality.get("enabled"),
            label=f"{label}.enabled",
        ),
        require_json_boolean(
            confidentiality.get("verified"),
            label=f"{label}.verified",
        ),
        require_public_identity_string(
            confidentiality.get("mode"),
            label=f"{label}.mode",
        ),
    ]

    verified = confidentiality.get("verified") is True
    verifier = confidentiality.get("verifier")
    if verified or verifier is not None:
        results.append(
            require_public_identity_string(
                verifier,
                label=f"{label}.verifier",
            )
        )

    for digest_field in (
        "policy_digest",
        "evidence_digest",
        "verified_claims_digest",
    ):
        value = confidentiality.get(digest_field)
        if verified or value is not None:
            results.append(
                require_digest(
                    value,
                    label=f"{label}.{digest_field}",
                )
            )

    verified_at = confidentiality.get("verified_at")
    if verified or verified_at is not None:
        results.append(
            require_not_future_timestamp(
                verified_at,
                label=f"{label}.verified_at",
            )
        )

    expires_at = confidentiality.get("expires_at")
    if verified:
        results.append(
            require_future_timestamp(
                expires_at,
                label=f"{label}.expires_at",
            )
        )
    elif expires_at is not None:
        results.append(
            require_integer_timestamp(
                expires_at,
                label=f"{label}.expires_at",
            )
        )

    if verified:
        results.extend(
            validate_provider_proof_claims(
                confidentiality,
                label=label,
                public_policy=public_policy,
            )
        )

    results.extend(
        validate_public_string_list(
            confidentiality.get("model_ids"),
            label=f"{label}.model_ids",
        )
    )
    results.extend(
        validate_public_string_list(
            confidentiality.get("model_id_prefixes"),
            label=f"{label}.model_id_prefixes",
        )
    )
    return results


def validate_attestation_routing_policy(
    attestation: dict[str, Any],
) -> list[CheckResult]:
    routing_policy = attestation.get("routing_policy")
    if not isinstance(routing_policy, dict):
        return [CheckResult(False, "attestation routing_policy is present")]

    try:
        routing_policy_digest = sha256_json_digest(routing_policy)
    except (TypeError, ValueError):
        routing_policy_digest = None

    results: list[CheckResult] = [
        CheckResult(
            routing_policy_digest is not None,
            "attestation routing_policy is canonical JSON",
        ),
        CheckResult(
            attestation.get("routing_policy_digest") == routing_policy_digest,
            "attestation routing_policy_digest matches routing_policy",
        ),
        require_public_identity_string(
            routing_policy.get("mode"),
            label="attestation routing_policy.mode",
        ),
        CheckResult(
            routing_policy.get("mode") == "required",
            "attestation routing_policy.mode is required",
        ),
        require_true_boolean(
            routing_policy.get("required"),
            label="attestation routing_policy.required",
        ),
    ]
    results.extend(
        validate_client_confidentiality_boundary(
            routing_policy.get("client_confidentiality"),
            label="attestation routing_policy.client_confidentiality",
        )
    )
    results.append(
        CheckResult(
            public_routable_with_full_attestation_map(
                routing_policy.get("routable_with_full_attestation")
            )
            is not None,
            (
                "attestation routing_policy.routable_with_full_attestation "
                "is an exact provider model map"
            ),
        )
    )

    providers = routing_policy.get("providers")
    if not isinstance(providers, list):
        results.append(
            CheckResult(False, "attestation routing_policy.providers is a list")
        )
        return results

    for index, provider in enumerate(providers):
        label = f"attestation routing_policy.providers[{index}]"
        if not isinstance(provider, dict):
            results.append(CheckResult(False, f"{label} is an object"))
            continue
        results.append(
            require_public_identity_string(
                provider.get("provider_type"),
                label=f"{label}.provider_type",
            )
        )
        if provider.get("upstream_name") is not None:
            results.append(
                require_public_identity_string(
                    provider.get("upstream_name"),
                    label=f"{label}.upstream_name",
                )
            )
        if provider.get("db_id") is not None:
            results.append(
                CheckResult(
                    isinstance(provider.get("db_id"), int)
                    and not isinstance(provider.get("db_id"), bool),
                    f"{label}.db_id is an integer",
                )
            )
        base_url = provider.get("base_url")
        if base_url is not None:
            base_url_check = require_non_empty_string(
                base_url,
                label=f"{label}.base_url",
            )
            results.append(base_url_check)
            if base_url_check.ok:
                parsed = urlsplit(base_url)
                provider_type = public_string_value(provider.get("provider_type"))
                confidentiality = provider.get("confidentiality")
                confidentiality_mode = (
                    public_string_value(confidentiality.get("mode"))
                    if isinstance(confidentiality, dict)
                    else None
                )
                absolute_url_check = CheckResult(
                    bool(parsed.scheme and parsed.netloc),
                    f"{label}.base_url is an absolute URL",
                )
                results.append(absolute_url_check)
                host_check = CheckResult(
                    absolute_url_check.ok and bool(parsed.hostname),
                    f"{label}.base_url includes a host",
                )
                results.append(host_check)
                try:
                    parsed.port
                    valid_port = True
                except ValueError:
                    valid_port = False
                port_check = CheckResult(
                    absolute_url_check.ok and valid_port,
                    f"{label}.base_url includes a valid port",
                )
                results.append(port_check)
                scheme_check = CheckResult(
                    absolute_url_check.ok
                    and host_check.ok
                    and port_check.ok
                    and provider_base_url_scheme_allowed(
                        base_url,
                        provider_type=provider_type,
                        confidentiality_mode=confidentiality_mode,
                    ),
                    f"{label}.base_url uses an allowed scheme",
                )
                results.append(scheme_check)
                results.append(
                    CheckResult(
                        "@" not in parsed.netloc,
                        f"{label}.base_url omits URL userinfo",
                    )
                )
                if (
                    provider_type == "ppq-private"
                    or confidentiality_mode == "ppq-private-tee"
                ):
                    results.append(
                        CheckResult(
                            absolute_url_check.ok
                            and scheme_check.ok
                            and ppq_private_base_url_has_private_path(base_url),
                            f"{label}.base_url uses PPQ private path",
                        )
                    )
                    results.append(
                        CheckResult(
                            absolute_url_check.ok
                            and scheme_check.ok
                            and ppq_private_base_url_has_ppq_host(base_url),
                            f"{label}.base_url host is ppq.ai or a ppq.ai subdomain",
                        )
                    )
                if provider_type == "tinfoil" or confidentiality_mode == "tinfoil":
                    results.append(
                        CheckResult(
                            absolute_url_check.ok
                            and scheme_check.ok
                            and tinfoil_base_url_is_default(base_url),
                            f"{label}.base_url is Tinfoil default router URL",
                        )
                    )
                if (
                    provider_type == "privatemode"
                    or confidentiality_mode == "privatemode"
                ):
                    results.append(
                        CheckResult(
                            absolute_url_check.ok
                            and scheme_check.ok
                            and privatemode_base_url_is_loopback_proxy(base_url),
                            f"{label}.base_url is Privatemode loopback proxy URL",
                        )
                    )
        confidentiality = provider.get("confidentiality")
        if not isinstance(confidentiality, dict):
            results.append(CheckResult(False, f"{label}.confidentiality is an object"))
            continue
        provider_type = public_string_value(provider.get("provider_type"))
        confidentiality_mode = public_string_value(confidentiality.get("mode"))
        results.append(
            CheckResult(
                CONFIDENTIAL_PROVIDER_MODES.get(provider_type or "")
                == confidentiality_mode,
                f"{label} confidentiality mode "
                f"{confidentiality_mode or '<missing>'} matches provider_type "
                f"{provider_type or '<missing>'}",
            )
        )
        results.extend(
            validate_known_provider_exact_selectors(
                confidentiality,
                provider_type=provider_type,
                label=f"{label}.confidentiality",
            )
        )
        confidentiality_policy = provider.get("confidentiality_policy")
        public_policy = (
            confidentiality_policy if isinstance(confidentiality_policy, dict) else None
        )
        results.extend(
            validate_attestation_routing_policy_confidentiality(
                confidentiality,
                label=f"{label}.confidentiality",
                public_policy=public_policy,
            )
        )
        if confidentiality_policy is None and provider_policy_required_for_public_verification(
            confidentiality_mode
        ):
            results.append(
                CheckResult(
                    False,
                    (
                        f"{label}.confidentiality_policy is required for verified "
                        f"{confidentiality_mode}"
                    ),
                )
            )
        elif confidentiality_policy is not None:
            if not isinstance(confidentiality_policy, dict):
                results.append(
                    CheckResult(False, f"{label}.confidentiality_policy is an object")
                )
            else:
                results.extend(
                    validate_provider_policy_proof_binding(
                        confidentiality_policy,
                        confidentiality,
                        mode=confidentiality_mode,
                        label=label,
                    )
                )
    return results


def tinfoil_result_hosts_from_policy(
    *,
    provider: dict[str, Any],
    policy: dict[str, Any] | None,
) -> set[str]:
    hosts: set[str] = set()
    if host := hostname_from_https_url(provider.get("base_url")):
        hosts.add(host)
    if isinstance(policy, dict):
        for key in ("attestation_url", "hpke_keys_url"):
            if host := hostname_from_https_url(policy.get(key)):
                hosts.add(host)
        for key in ("enclave_host", "expected_enclave_host"):
            if host := hostname_from_identity(policy.get(key)):
                hosts.add(host)
        targets = _policy_targets(policy)
        if isinstance(targets, dict):
            for target in targets.values():
                if not isinstance(target, dict):
                    continue
                for key in ("attestation_url", "hpke_keys_url"):
                    if host := hostname_from_https_url(target.get(key)):
                        hosts.add(host)
                for key in ("host", "expected_enclave_host"):
                    if host := hostname_from_identity(target.get(key)):
                        hosts.add(host)
    return hosts


def tinfoil_result_hosts_from_attestation(attestation: dict[str, Any]) -> set[str]:
    routing_policy = attestation.get("routing_policy")
    if not isinstance(routing_policy, dict):
        return set()
    providers = routing_policy.get("providers")
    if not isinstance(providers, list):
        return set()
    hosts: set[str] = set()
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        provider_type = public_string_value(provider.get("provider_type"))
        confidentiality = provider.get("confidentiality")
        confidentiality_mode = (
            public_string_value(confidentiality.get("mode"))
            if isinstance(confidentiality, dict)
            else None
        )
        if provider_type != "tinfoil" and confidentiality_mode != "tinfoil":
            continue
        policy = provider.get("confidentiality_policy")
        hosts.update(
            tinfoil_result_hosts_from_policy(
                provider=provider,
                policy=policy if isinstance(policy, dict) else None,
            )
        )
    return hosts


def tinfoil_tls_optional_result_hosts_from_attestation(
    attestation: dict[str, Any],
) -> set[str]:
    routing_policy = attestation.get("routing_policy")
    if not isinstance(routing_policy, dict):
        return set()
    providers = routing_policy.get("providers")
    if not isinstance(providers, list):
        return set()
    hosts: set[str] = set()
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        provider_type = public_string_value(provider.get("provider_type"))
        confidentiality = provider.get("confidentiality")
        confidentiality_mode = (
            public_string_value(confidentiality.get("mode"))
            if isinstance(confidentiality, dict)
            else None
        )
        if provider_type != "tinfoil" and confidentiality_mode != "tinfoil":
            continue
        policy = provider.get("confidentiality_policy")
        if not isinstance(policy, dict):
            continue
        if tinfoil_tls_public_key_binding_required(policy):
            continue
        hosts.update(tinfoil_result_hosts_from_policy(provider=provider, policy=policy))
    return hosts


def validate_attestation_results_directory(
    attestation: dict[str, Any],
    attestation_results_directory: AttestationResultsDirectory | None,
) -> list[CheckResult]:
    if attestation_results_directory is None:
        return []
    routing_policy = attestation.get("routing_policy")
    if not isinstance(routing_policy, dict):
        return []
    providers = routing_policy.get("providers")
    if not isinstance(providers, list):
        return []

    results: list[CheckResult] = []
    tinfoil_statuses = (
        attestation_results_directory.provider_host_statuses.get("tinfoil") or {}
    )
    tinfoil_errors = (
        attestation_results_directory.provider_host_errors.get("tinfoil") or {}
    )
    for index, provider in enumerate(providers):
        if not isinstance(provider, dict):
            continue
        provider_type = public_string_value(provider.get("provider_type"))
        confidentiality = provider.get("confidentiality")
        confidentiality_mode = (
            public_string_value(confidentiality.get("mode"))
            if isinstance(confidentiality, dict)
            else None
        )
        if provider_type != "tinfoil" and confidentiality_mode != "tinfoil":
            continue
        label = f"attestation routing_policy.providers[{index}]"
        if not tinfoil_statuses:
            results.append(
                CheckResult(
                    False,
                    "confidential-inference attestation results include tinfoil checks",
                )
            )
            continue
        policy = provider.get("confidentiality_policy")
        public_policy = policy if isinstance(policy, dict) else None
        hosts = tinfoil_result_hosts_from_policy(
            provider=provider,
            policy=public_policy,
        )
        if not hosts:
            results.append(
                CheckResult(
                    False,
                    f"{label} Tinfoil host is present for confidential-inference attestation results",
                )
            )
            continue
        for host in sorted(hosts):
            status = tinfoil_statuses.get(host)
            if status == "verified":
                results.append(
                    CheckResult(
                        True,
                        f"{label} Tinfoil host {host} is verified in confidential-inference attestation results",
                    )
                )
                continue
            if status is None:
                results.append(
                    CheckResult(
                        False,
                        f"{label} Tinfoil host {host} is missing from confidential-inference attestation results",
                    )
                )
                continue
            error = tinfoil_errors.get(host)
            if (
                status == "failed"
                and error == TINFOIL_TLS_BINDING_NOT_VERIFIED_ERROR
                and not tinfoil_tls_public_key_binding_required(public_policy)
            ):
                results.append(
                    CheckResult(
                        True,
                        (
                            f"{label} Tinfoil host {host} TLS binding is omitted "
                            "by EHBP policy"
                        ),
                    )
                )
                continue
            suffix = f": {error}" if error else ""
            results.append(
                CheckResult(
                    False,
                    f"{label} Tinfoil host {host} is not verified in confidential-inference attestation results (status={status}){suffix}",
                )
            )
    return results


def attestation_has_tinfoil_provider(attestation: dict[str, Any]) -> bool:
    routing_policy = attestation.get("routing_policy")
    if not isinstance(routing_policy, dict):
        return False
    providers = routing_policy.get("providers")
    if not isinstance(providers, list):
        return False
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        provider_type = public_string_value(provider.get("provider_type"))
        confidentiality = provider.get("confidentiality")
        confidentiality_mode = (
            public_string_value(confidentiality.get("mode"))
            if isinstance(confidentiality, dict)
            else None
        )
        if provider_type == "tinfoil" or confidentiality_mode == "tinfoil":
            return True
    return False


def attestation_has_ppq_private_provider(attestation: dict[str, Any]) -> bool:
    routing_policy = attestation.get("routing_policy")
    if not isinstance(routing_policy, dict):
        return False
    providers = routing_policy.get("providers")
    if not isinstance(providers, list):
        return False
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        provider_type = public_string_value(provider.get("provider_type"))
        confidentiality = provider.get("confidentiality")
        confidentiality_mode = (
            public_string_value(confidentiality.get("mode"))
            if isinstance(confidentiality, dict)
            else None
        )
        if provider_type == "ppq-private" or confidentiality_mode == "ppq-private-tee":
            return True
    return False


def expected_provider_needs_ppq_results(expected_provider: str) -> bool:
    return expected_provider in {"ppq", "ppq-private", "ppq-private-tee"}


def attestation_results_include_verified_ppq(
    attestation_results_directory: AttestationResultsDirectory | None,
) -> bool:
    if attestation_results_directory is None:
        return False
    ppq_statuses = (
        attestation_results_directory.provider_host_statuses.get("ppq") or {}
    )
    return any(status == "verified" for status in ppq_statuses.values())


def expected_tinfoil_models(expected: list[tuple[str, str]]) -> set[str]:
    models: set[str] = set()
    for expected_provider, model in expected:
        if expected_provider.strip().lower() != "tinfoil":
            continue
        normalized_model = model.strip().lower()
        if normalized_model:
            models.add(normalized_model)
    return models


def attested_tinfoil_models(attestation: dict[str, Any]) -> set[str]:
    routing_policy = attestation.get("routing_policy")
    if not isinstance(routing_policy, dict):
        return set()
    providers = routing_policy.get("providers")
    if not isinstance(providers, list):
        return set()

    models: set[str] = set()
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        provider_type = public_string_value(provider.get("provider_type"))
        confidentiality = provider.get("confidentiality")
        confidentiality_mode = (
            public_string_value(confidentiality.get("mode"))
            if isinstance(confidentiality, dict)
            else None
        )
        if provider_type != "tinfoil" and confidentiality_mode != "tinfoil":
            continue
        if not isinstance(confidentiality, dict):
            continue
        for model_id in strict_public_string_set(confidentiality.get("model_ids")) or set():
            normalized_model = model_id.strip().lower()
            if normalized_model:
                models.add(normalized_model)
    return models


def target_directory_tinfoil_models(
    attestation_targets_directory: AttestationTargetsDirectory | None,
) -> set[str]:
    if attestation_targets_directory is None:
        return set()
    models = set(
        attestation_targets_directory.provider_target_models.get("tinfoil") or set()
    )
    models.update(
        attestation_targets_directory.provider_check_models.get("tinfoil", {}).values()
    )
    return models


def tinfoil_model_identity_matches(model: str, candidate: str) -> bool:
    normalized_model = model.strip().lower()
    normalized_candidate = candidate.strip().lower()
    if normalized_model == normalized_candidate:
        return True
    for aliases in KNOWN_TINFOIL_MODEL_HOST_MODELS.values():
        normalized_aliases = {alias.strip().lower() for alias in aliases}
        if normalized_model in normalized_aliases and normalized_candidate in normalized_aliases:
            return True
    selected = normalized_model_identity(model.split("/")[-1])
    candidate_identity = normalized_model_identity(candidate.split("/")[-1])
    return selected is not None and selected == candidate_identity


def tinfoil_model_set_contains(model: str, candidates: set[str]) -> bool:
    return any(tinfoil_model_identity_matches(model, candidate) for candidate in candidates)


def tinfoil_model_status_is_verified(
    model: str,
    model_statuses: dict[str, str],
) -> bool:
    return any(
        status == "verified" and tinfoil_model_identity_matches(model, candidate)
        for candidate, status in model_statuses.items()
    )


def tinfoil_model_details_for_model(
    model: str,
    model_details: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    direct = model_details.get(model)
    if isinstance(direct, dict):
        return direct
    matches = [
        details
        for candidate, details in model_details.items()
        if tinfoil_model_identity_matches(model, candidate)
        and isinstance(details, dict)
    ]
    if len(matches) == 1:
        return matches[0]
    return {}


def target_directory_ppq_models(
    attestation_targets_directory: AttestationTargetsDirectory | None,
) -> set[str]:
    if attestation_targets_directory is None:
        return set()
    models = set(attestation_targets_directory.provider_target_models.get("ppq") or set())
    models.update(
        attestation_targets_directory.provider_check_models.get("ppq", {}).values()
    )
    return models


def target_directory_provider_models(
    attestation_targets_directory: AttestationTargetsDirectory | None,
    provider: str,
) -> set[str]:
    if attestation_targets_directory is None:
        return set()
    normalized_provider = provider.strip().lower()
    models = set(
        attestation_targets_directory.provider_target_models.get(normalized_provider)
        or set()
    )
    models.update(
        attestation_targets_directory.provider_check_models.get(
            normalized_provider,
            {},
        ).values()
    )
    return models


def expected_ppq_private_models(expected: list[tuple[str, str]]) -> set[str]:
    models: set[str] = set()
    for expected_provider, model in expected:
        if not expected_provider_needs_ppq_results(expected_provider):
            continue
        normalized_model = model.strip().lower()
        if normalized_model:
            models.add(normalized_model)
    return models


def attested_ppq_private_models(attestation: dict[str, Any]) -> set[str]:
    routing_policy = attestation.get("routing_policy")
    if not isinstance(routing_policy, dict):
        return set()
    providers = routing_policy.get("providers")
    if not isinstance(providers, list):
        return set()

    models: set[str] = set()
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        provider_type = public_string_value(provider.get("provider_type"))
        confidentiality = provider.get("confidentiality")
        confidentiality_mode = (
            public_string_value(confidentiality.get("mode"))
            if isinstance(confidentiality, dict)
            else None
        )
        if provider_type != "ppq-private" and confidentiality_mode != "ppq-private-tee":
            continue
        if not isinstance(confidentiality, dict):
            continue
        for model_id in strict_public_string_set(confidentiality.get("model_ids")) or set():
            normalized_model = model_id.strip().lower()
            if normalized_model:
                models.add(normalized_model)
    return models


def expected_privatemode_models(expected: list[tuple[str, str]]) -> set[str]:
    models: set[str] = set()
    for expected_provider, model in expected:
        if expected_provider.strip().lower() != "privatemode":
            continue
        normalized_model = model.strip().lower()
        if normalized_model:
            models.add(normalized_model)
    return models


def attested_privatemode_models(attestation: dict[str, Any]) -> set[str]:
    routing_policy = attestation.get("routing_policy")
    if not isinstance(routing_policy, dict):
        return set()
    providers = routing_policy.get("providers")
    if not isinstance(providers, list):
        return set()

    models: set[str] = set()
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        provider_type = public_string_value(provider.get("provider_type"))
        confidentiality = provider.get("confidentiality")
        confidentiality_mode = (
            public_string_value(confidentiality.get("mode"))
            if isinstance(confidentiality, dict)
            else None
        )
        if provider_type != "privatemode" and confidentiality_mode != "privatemode":
            continue
        if not isinstance(confidentiality, dict):
            continue
        for model_id in strict_public_string_set(confidentiality.get("model_ids")) or set():
            normalized_model = model_id.strip().lower()
            if normalized_model:
                models.add(normalized_model)
    return models


def ppq_private_policy_for_model(
    attestation: dict[str, Any],
    model_id: str,
) -> dict[str, Any] | None:
    routing_policy = attestation.get("routing_policy")
    if not isinstance(routing_policy, dict):
        return None
    providers = routing_policy.get("providers")
    if not isinstance(providers, list):
        return None
    normalized_model = model_id.strip().lower()
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        provider_type = public_string_value(provider.get("provider_type"))
        confidentiality = provider.get("confidentiality")
        confidentiality_mode = (
            public_string_value(confidentiality.get("mode"))
            if isinstance(confidentiality, dict)
            else None
        )
        if provider_type != "ppq-private" and confidentiality_mode != "ppq-private-tee":
            continue
        model_ids = (
            strict_public_string_set(confidentiality.get("model_ids"))
            if isinstance(confidentiality, dict)
            else None
        )
        if model_ids is not None and normalized_model not in {
            model.lower() for model in model_ids
        }:
            continue
        policy = provider.get("confidentiality_policy")
        if isinstance(policy, dict):
            return policy
    return None


def privatemode_policy_for_model(
    attestation: dict[str, Any],
    model_id: str,
) -> dict[str, Any] | None:
    routing_policy = attestation.get("routing_policy")
    if not isinstance(routing_policy, dict):
        return None
    providers = routing_policy.get("providers")
    if not isinstance(providers, list):
        return None
    normalized_model = model_id.strip().lower()
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        provider_type = public_string_value(provider.get("provider_type"))
        confidentiality = provider.get("confidentiality")
        confidentiality_mode = (
            public_string_value(confidentiality.get("mode"))
            if isinstance(confidentiality, dict)
            else None
        )
        if provider_type != "privatemode" and confidentiality_mode != "privatemode":
            continue
        model_ids = (
            strict_public_string_set(confidentiality.get("model_ids"))
            if isinstance(confidentiality, dict)
            else None
        )
        if model_ids is not None and normalized_model not in {
            model.lower() for model in model_ids
        }:
            continue
        policy = provider.get("confidentiality_policy")
        if isinstance(policy, dict):
            return policy
    return None


def tinfoil_policy_for_model(
    attestation: dict[str, Any],
    model_id: str,
) -> dict[str, Any] | None:
    routing_policy = attestation.get("routing_policy")
    if not isinstance(routing_policy, dict):
        return None
    providers = routing_policy.get("providers")
    if not isinstance(providers, list):
        return None
    normalized_model = model_id.strip().lower()
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        provider_type = public_string_value(provider.get("provider_type"))
        confidentiality = provider.get("confidentiality")
        confidentiality_mode = (
            public_string_value(confidentiality.get("mode"))
            if isinstance(confidentiality, dict)
            else None
        )
        if provider_type != "tinfoil" and confidentiality_mode != "tinfoil":
            continue
        model_ids = (
            strict_public_string_set(confidentiality.get("model_ids"))
            if isinstance(confidentiality, dict)
            else None
        )
        if model_ids is not None and normalized_model not in {
            model.lower() for model in model_ids
        }:
            continue
        policy = provider.get("confidentiality_policy")
        if isinstance(policy, dict):
            return policy
    return None


def tinfoil_result_release_checks(
    *,
    attestation: dict[str, Any],
    model_id: str,
    details: dict[str, Any],
) -> list[CheckResult]:
    policy = tinfoil_policy_for_model(attestation, model_id)
    if policy is None:
        return [
            CheckResult(
                False,
                "confidential-inference Tinfoil attestation results "
                f"bind model {model_id} to deployed policy release digests",
            )
        ]
    targets = policy.get("model_attestation_targets")
    target = targets.get(model_id) if isinstance(targets, dict) else None
    target_release_pins = (
        _policy_digest_set(
            target,
            "expected_release_digest",
            "release_digest",
            "allowed_release_digest",
            "allowed_release_digests",
        )
        if isinstance(target, dict)
        else set()
    )
    release_digest = _normalized_sha256_digest(details.get("release_digest"))
    return [
        CheckResult(
            bool(target_release_pins) and release_digest in target_release_pins,
            "confidential-inference Tinfoil attestation results "
            f"bind model {model_id} release digest to deployed policy",
        )
    ]


def privatemode_result_policy_checks(
    *,
    attestation: dict[str, Any],
    model_id: str,
    details: dict[str, Any],
) -> list[CheckResult]:
    policy = privatemode_policy_for_model(attestation, model_id)
    if policy is None:
        return [
            CheckResult(
                False,
                "confidential-inference Privatemode attestation results "
                f"bind model {model_id} to deployed policy",
            )
        ]

    label = f"confidential-inference Privatemode attestation results model {model_id}"
    results: list[CheckResult] = [
        CheckResult(
            details.get("transport") == "privatemode-proxy",
            f"{label} provider proof_claims.transport is privatemode-proxy",
        ),
        CheckResult(
            details.get("trust_tier") == "app-e2ee",
            f"{label} provider proof_claims.trust_tier is app-e2ee",
        ),
    ]
    results.extend(
        validate_privatemode_policy_proof_binding(
            policy,
            details,
            model_ids={model_id},
            label=label,
        )
    )
    results.extend(
        validate_provider_verification_steps(
            details,
            label=f"{label} provider",
            required_steps=REQUIRED_PRIVATEMODE_PROVIDER_VERIFICATION_STEPS,
        )
    )
    return results


def ppq_private_result_release_checks(
    *,
    attestation: dict[str, Any],
    model_id: str,
    details: dict[str, Any],
) -> list[CheckResult]:
    policy = ppq_private_policy_for_model(attestation, model_id)
    if policy is None:
        return [
            CheckResult(
                False,
                "confidential-inference PPQ private attestation results "
                f"bind model {model_id} to deployed policy release digests",
            )
        ]

    results: list[CheckResult] = []
    router_release_pins = _policy_digest_set(
        policy,
        "expected_release_digest",
        "release_digest",
        "allowed_release_digest",
        "allowed_release_digests",
    )
    router_digest = _normalized_sha256_digest(details.get("release_digest"))
    if router_release_pins:
        results.append(
            CheckResult(
                router_digest in router_release_pins,
                "confidential-inference PPQ private attestation results "
                f"bind model {model_id} router release digest to deployed policy",
            )
        )

    targets = policy.get("model_attestation_targets")
    target = targets.get(model_id) if isinstance(targets, dict) else None
    target_release_pins = (
        _policy_digest_set(
            target,
            "expected_release_digest",
            "release_digest",
            "allowed_release_digest",
            "allowed_release_digests",
        )
        if isinstance(target, dict)
        else set()
    )
    backend_digest = _normalized_sha256_digest(details.get("backend_release_digest"))
    results.append(
        CheckResult(
            bool(target_release_pins) and backend_digest in target_release_pins,
            "confidential-inference PPQ private attestation results "
            f"bind model {model_id} backend release digest to deployed policy",
        )
    )
    for field_name in PPQ_PRIVATE_BACKEND_ATTESTATION_RESULT_REQUIRED_TRUE_FIELDS:
        results.append(
            CheckResult(
                details.get(field_name) is True,
                "confidential-inference PPQ private attestation results "
                f"prove model {model_id} {field_name}",
            )
        )
    return results


def ppq_private_result_hosts_from_attestation(attestation: dict[str, Any]) -> set[str]:
    routing_policy = attestation.get("routing_policy")
    if not isinstance(routing_policy, dict):
        return set()
    providers = routing_policy.get("providers")
    if not isinstance(providers, list):
        return set()
    hosts: set[str] = set()
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        provider_type = public_string_value(provider.get("provider_type"))
        confidentiality = provider.get("confidentiality")
        confidentiality_mode = (
            public_string_value(confidentiality.get("mode"))
            if isinstance(confidentiality, dict)
            else None
        )
        if provider_type != "ppq-private" and confidentiality_mode != "ppq-private-tee":
            continue
        if host := hostname_from_https_url(provider.get("base_url")):
            hosts.add(host)
        policy = provider.get("confidentiality_policy")
        if isinstance(policy, dict):
            for key in ("attestation_bundle_url", "hpke_keys_url"):
                if host := hostname_from_https_url(policy.get(key)):
                    hosts.add(host)
            for key in ("enclave_host", "expected_enclave_host"):
                if host := hostname_from_identity(policy.get(key)):
                    hosts.add(host)
    return hosts


def privatemode_result_hosts_from_attestation(attestation: dict[str, Any]) -> set[str]:
    routing_policy = attestation.get("routing_policy")
    if not isinstance(routing_policy, dict):
        return set()
    providers = routing_policy.get("providers")
    if not isinstance(providers, list):
        return set()
    hosts: set[str] = set()
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        provider_type = public_string_value(provider.get("provider_type"))
        confidentiality = provider.get("confidentiality")
        confidentiality_mode = (
            public_string_value(confidentiality.get("mode"))
            if isinstance(confidentiality, dict)
            else None
        )
        if provider_type != "privatemode" and confidentiality_mode != "privatemode":
            continue
        policy = provider.get("confidentiality_policy")
        if not isinstance(policy, dict):
            continue
        for key in (
            "api_base_url",
            "cdn_base_url",
            "manifest_url",
            "proxy_configuration_url",
            "source_verification_url",
            "model_verification_url",
        ):
            if host := hostname_from_https_url(policy.get(key)):
                hosts.add(host)
    return hosts


def strict_external_guardrail_checks(
    *,
    attestation: dict[str, Any],
    expected: list[tuple[str, str]],
    provider_model_catalog: ProviderModelCatalog | None,
    attestation_targets_directory: AttestationTargetsDirectory | None,
    attestation_results_directory: AttestationResultsDirectory | None,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    results.append(
        CheckResult(
            provider_model_catalog is not None,
            "confidential-inference provider catalog guardrail is supplied",
        )
    )
    needs_tinfoil_directory = attestation_has_tinfoil_provider(attestation) or any(
        expected_provider == "tinfoil" for expected_provider, _model in expected
    )
    if needs_tinfoil_directory:
        results.append(
            CheckResult(
                attestation_targets_directory is not None,
                "confidential-inference Tinfoil attestation targets guardrail is supplied",
            )
        )
        results.append(
            CheckResult(
                attestation_results_directory is not None,
                "confidential-inference Tinfoil attestation results guardrail is supplied",
            )
        )
        required_tinfoil_models = expected_tinfoil_models(expected)
        required_tinfoil_models.update(attested_tinfoil_models(attestation))
        if attestation_targets_directory is not None:
            tinfoil_target_models = target_directory_tinfoil_models(
                attestation_targets_directory
            )
            tinfoil_router_hosts = {
                host
                for check_id, host in (
                    attestation_targets_directory.provider_check_hosts.get("tinfoil")
                    or {}
                ).items()
                if "router" in check_id
            }
            results.append(
                CheckResult(
                    bool(tinfoil_router_hosts),
                    "confidential-inference Tinfoil attestation targets include provider-level router",
                )
            )
            for model in sorted(required_tinfoil_models):
                results.append(
                    CheckResult(
                        tinfoil_model_set_contains(model, tinfoil_target_models),
                        "confidential-inference Tinfoil attestation targets include "
                        f"model {model}",
                    )
                )
        if attestation_results_directory is not None:
            tinfoil_model_statuses = (
                attestation_results_directory.provider_model_statuses.get("tinfoil")
                or {}
            )
            tinfoil_model_details = (
                attestation_results_directory.provider_model_details.get("tinfoil")
                or {}
            )
            tinfoil_router_hosts = set()
            if attestation_targets_directory is not None:
                tinfoil_router_hosts = {
                    host
                    for check_id, host in (
                        attestation_targets_directory.provider_check_hosts.get(
                            "tinfoil"
                        )
                        or {}
                    ).items()
                    if "router" in check_id
                }
            tinfoil_host_statuses = (
                attestation_results_directory.provider_host_statuses.get("tinfoil")
                or {}
            )
            results.append(
                CheckResult(
                    bool(tinfoil_router_hosts)
                    and any(
                        tinfoil_host_statuses.get(host) == "verified"
                        for host in tinfoil_router_hosts
                    ),
                    "confidential-inference Tinfoil attestation results include verified provider-level router",
                )
            )
            for model in sorted(required_tinfoil_models):
                results.append(
                    CheckResult(
                        tinfoil_model_status_is_verified(model, tinfoil_model_statuses),
                        "confidential-inference Tinfoil attestation results include "
                        f"verified model {model}",
                    )
                )
                if tinfoil_model_status_is_verified(model, tinfoil_model_statuses):
                    results.extend(
                        tinfoil_result_release_checks(
                            attestation=attestation,
                            model_id=model,
                            details=tinfoil_model_details_for_model(
                                model,
                                tinfoil_model_details,
                            ),
                        )
                    )
    needs_ppq_results = attestation_has_ppq_private_provider(attestation) or any(
        expected_provider_needs_ppq_results(expected_provider)
        for expected_provider, _model in expected
    )
    if needs_ppq_results:
        results.append(
            CheckResult(
                attestation_targets_directory is not None,
                "confidential-inference PPQ private attestation targets guardrail is supplied",
            )
        )
        results.append(
            CheckResult(
                attestation_results_directory is not None,
                "confidential-inference PPQ private attestation results guardrail is supplied",
            )
        )
        required_ppq_models = expected_ppq_private_models(expected)
        required_ppq_models.update(attested_ppq_private_models(attestation))
        if attestation_targets_directory is not None:
            ppq_target_models = target_directory_ppq_models(
                attestation_targets_directory
            )
            for model in sorted(required_ppq_models):
                results.append(
                    CheckResult(
                        model in ppq_target_models,
                        "confidential-inference PPQ private attestation targets "
                        f"include model {model}",
                    )
                )
        if attestation_results_directory is not None:
            required_ppq_hosts = ppq_private_result_hosts_from_attestation(attestation)
            ppq_host_statuses = (
                attestation_results_directory.provider_host_statuses.get("ppq") or {}
            )
            if required_ppq_hosts:
                for host in sorted(required_ppq_hosts):
                    results.append(
                        CheckResult(
                            ppq_host_statuses.get(host) == "verified",
                            "confidential-inference PPQ private attestation results "
                            f"include verified provider host {host}",
                        )
                    )
            else:
                results.append(
                    CheckResult(
                        False,
                        "confidential-inference PPQ private attestation results identify provider host",
                    )
                )
            ppq_model_statuses = (
                attestation_results_directory.provider_model_statuses.get("ppq") or {}
            )
            ppq_model_details = (
                attestation_results_directory.provider_model_details.get("ppq") or {}
            )
            if not required_ppq_models:
                results.append(
                    CheckResult(
                        False,
                        "confidential-inference PPQ private attestation results identify selected models",
                    )
                )
                results.append(
                    CheckResult(
                        False,
                        "confidential-inference PPQ private attestation results include selected model evidence",
                    )
                )
            for model in sorted(required_ppq_models):
                results.append(
                    CheckResult(
                        ppq_model_statuses.get(model) == "verified",
                        "confidential-inference PPQ private attestation results "
                        f"include verified model {model}",
                    )
                )
                if ppq_model_statuses.get(model) == "verified":
                    results.extend(
                        ppq_private_result_release_checks(
                            attestation=attestation,
                            model_id=model,
                            details=ppq_model_details.get(model) or {},
                        )
                    )
    privatemode_hosts = privatemode_result_hosts_from_attestation(attestation)
    required_privatemode_models = expected_privatemode_models(expected)
    required_privatemode_models.update(attested_privatemode_models(attestation))
    needs_privatemode_results = bool(privatemode_hosts or required_privatemode_models)
    if needs_privatemode_results:
        results.append(
            CheckResult(
                attestation_targets_directory is not None,
                "confidential-inference Privatemode attestation targets guardrail is supplied",
            )
        )
        results.append(
            CheckResult(
                attestation_results_directory is not None,
                "confidential-inference Privatemode attestation results guardrail is supplied",
            )
        )
        if attestation_targets_directory is not None:
            privatemode_target_models = target_directory_provider_models(
                attestation_targets_directory,
                "privatemode",
            )
            for model in sorted(required_privatemode_models):
                results.append(
                    CheckResult(
                        model in privatemode_target_models,
                        "confidential-inference Privatemode attestation targets "
                        f"include model {model}",
                    )
                )
        if attestation_results_directory is not None:
            privatemode_statuses = (
                attestation_results_directory.provider_host_statuses.get("privatemode")
                or {}
            )
            for host in sorted(privatemode_hosts):
                results.append(
                    CheckResult(
                        privatemode_statuses.get(host) == "verified",
                        "confidential-inference Privatemode attestation results "
                        f"include verified host {host}",
                    )
                )
            privatemode_model_statuses = (
                attestation_results_directory.provider_model_statuses.get(
                    "privatemode"
                )
                or {}
            )
            privatemode_model_details = (
                attestation_results_directory.provider_model_details.get(
                    "privatemode"
                )
                or {}
            )
            for model in sorted(required_privatemode_models):
                results.append(
                    CheckResult(
                        privatemode_model_statuses.get(model) == "verified",
                        "confidential-inference Privatemode attestation results "
                        f"include verified model {model}",
                    )
                )
                if privatemode_model_statuses.get(model) == "verified":
                    results.extend(
                        privatemode_result_policy_checks(
                            attestation=attestation,
                            model_id=model,
                            details=privatemode_model_details.get(model) or {},
                        )
                    )
    return results


def validate_attestation_statement(attestation: dict[str, Any]) -> list[CheckResult]:
    results = [
        CheckResult(
            attestation.get("schema_version") == "routstr-attestation-v1",
            "Routstr attestation schema is routstr-attestation-v1",
        ),
        require_digest(
            attestation.get("routing_policy_digest"),
            label="Routstr routing policy digest",
        ),
    ]
    results.extend(validate_attestation_routing_policy(attestation))
    tee = attestation.get("tee")
    if isinstance(tee, dict):
        results.append(
            require_digest(
                attestation_hpke_key_config_digest(attestation),
                label="attestation HPKE key config digest",
            )
        )
        results.append(
            require_digest(
                hpke_public_key_digest_from_attestation(attestation),
                label="attestation HPKE public key digest",
            )
        )
    else:
        results.append(CheckResult(False, "attestation tee section is missing"))
        return results

    local_verification = tee.get("local_verification")
    if isinstance(local_verification, dict):
        results.append(
            require_true_boolean(
                local_verification.get("verified"),
                label="attestation local TEE verifier reports verified",
            )
        )
        results.append(
            require_not_future_timestamp(
                local_verification.get("verified_at"),
                label="attestation local TEE verifier verified_at",
            )
        )
        results.append(
            require_future_timestamp(
                local_verification.get("expires_at"),
                label="attestation local TEE verifier expiry",
            )
        )
        results.append(
            require_digest(
                local_verification.get("verified_claims_digest"),
                label="attestation local TEE verified claims digest",
            )
        )
        results.append(
            require_digest(
                local_verification.get("evidence_digest"),
                label="attestation local TEE runtime evidence digest",
            )
        )
        proof_claims = local_verification.get("proof_claims")
        if isinstance(proof_claims, dict):
            routing_policy = attestation.get("routing_policy")
            client_confidentiality = (
                routing_policy.get("client_confidentiality")
                if isinstance(routing_policy, dict)
                else None
            )
            results.extend(
                validate_client_confidentiality_boundary(
                    client_confidentiality,
                    label="attestation routing_policy.client_confidentiality",
                    proof_claims=proof_claims,
                    proof_label="attestation local TEE proof",
                )
            )
        results.extend(
            validate_local_tee_proof_claims(
                local_verification,
                label="attestation local TEE",
                routing_policy_digest=attestation.get("routing_policy_digest"),
                routing_policy=(
                    attestation.get("routing_policy")
                    if isinstance(attestation.get("routing_policy"), dict)
                    else None
                ),
                hpke_key_config_digest=attestation_hpke_key_config_digest(
                    attestation
                ),
                hpke_public_key_digest=hpke_public_key_digest_from_attestation(
                    attestation
                ),
            )
        )
    else:
        results.append(
            CheckResult(False, "attestation local TEE verification status is missing")
        )
    return results


def validate_hpke_keys(key_config: bytes, content_type: str) -> list[CheckResult]:
    return [
        CheckResult(bool(key_config), "Routstr HPKE key config is non-empty"),
        CheckResult(
            "application/ohttp-keys" in content_type.lower(),
            "Routstr HPKE key config uses application/ohttp-keys",
        ),
    ]


def provider_matches_any(
    provider: dict[str, Any],
    *,
    expected_providers: set[str],
    expected_model: str,
) -> bool:
    return any(
        provider_matches(
            provider,
            expected_provider=expected_provider,
            expected_model=expected_model,
        )
        for expected_provider in expected_providers
    )


def matching_provider_confidentialities(
    providers: list[Any],
    *,
    expected_provider: str,
    expected_model: str,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        if not provider_matches(
            provider,
            expected_provider=expected_provider,
            expected_model=expected_model,
        ):
            continue
        confidentiality = provider.get("confidentiality")
        if isinstance(confidentiality, dict):
            matches.append(confidentiality)
    return matches


def matching_provider_policy_snapshots(
    providers: list[Any],
    *,
    expected_provider: str,
    expected_model: str,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        if not provider_matches(
            provider,
            expected_provider=expected_provider,
            expected_model=expected_model,
        ):
            continue
        policy = provider.get("confidentiality_policy")
        if isinstance(policy, dict):
            matches.append(policy)
    return matches


def validate_attested_provider_consistency(
    *,
    status_providers: list[Any],
    attestation_providers: list[Any],
    expected_provider: str,
    expected_model: str,
) -> list[CheckResult]:
    attested_confidentialities = matching_provider_confidentialities(
        attestation_providers,
        expected_provider=expected_provider,
        expected_model=expected_model,
    )
    results: list[CheckResult] = [
        CheckResult(
            bool(attested_confidentialities),
            f"attestation routing policy includes {expected_provider} for {expected_model}",
        )
    ]
    if not attested_confidentialities:
        return results

    status_confidentialities = matching_provider_confidentialities(
        status_providers,
        expected_provider=expected_provider,
        expected_model=expected_model,
    )
    status_policy_digests = {
        confidentiality.get("policy_digest")
        for confidentiality in status_confidentialities
        if isinstance(confidentiality.get("policy_digest"), str)
    }
    status_evidence_digests = {
        confidentiality.get("evidence_digest")
        for confidentiality in status_confidentialities
        if isinstance(confidentiality.get("evidence_digest"), str)
    }
    status_claims_digests = {
        confidentiality.get("verified_claims_digest")
        for confidentiality in status_confidentialities
        if isinstance(confidentiality.get("verified_claims_digest"), str)
    }
    status_proof_claims = [
        confidentiality.get("proof_claims")
        for confidentiality in status_confidentialities
        if isinstance(confidentiality.get("proof_claims"), dict)
    ]
    status_policy_snapshots = matching_provider_policy_snapshots(
        status_providers,
        expected_provider=expected_provider,
        expected_model=expected_model,
    )
    attested_policy_snapshots = matching_provider_policy_snapshots(
        attestation_providers,
        expected_provider=expected_provider,
        expected_model=expected_model,
    )
    if status_policy_snapshots or attested_policy_snapshots:
        results.append(
            CheckResult(
                bool(attested_policy_snapshots)
                and all(
                    policy_snapshot in status_policy_snapshots
                    for policy_snapshot in attested_policy_snapshots
                ),
                (
                    f"{expected_model} attested provider policy snapshot "
                    "matches provider status"
                ),
            )
        )

    for confidentiality in attested_confidentialities:
        results.append(
            CheckResult(
                confidentiality.get("policy_digest") in status_policy_digests,
                f"{expected_model} attested provider policy digest matches provider status",
            )
        )
        results.append(
            CheckResult(
                confidentiality.get("evidence_digest") in status_evidence_digests,
                f"{expected_model} attested provider evidence digest matches provider status",
            )
        )
        results.append(
            CheckResult(
                confidentiality.get("verified_claims_digest") in status_claims_digests,
                f"{expected_model} attested provider verified claims digest matches provider status",
            )
        )
        results.append(
            CheckResult(
                confidentiality.get("proof_claims") in status_proof_claims,
                f"{expected_model} attested provider proof claims match provider status",
            )
        )
        results.append(
            CheckResult(
                any(
                    confidentiality_snapshot_matches(
                        confidentiality,
                        status_confidentiality,
                    )
                    for status_confidentiality in status_confidentialities
                ),
                (
                    f"{expected_model} attested provider confidentiality snapshot "
                    "matches a single provider status"
                ),
            )
        )
    return results


def validate_model_provider_consistency(
    *,
    providers: list[Any],
    model: dict[str, Any],
    expected_providers: set[str],
    model_id: str,
) -> list[CheckResult]:
    matching_provider_digests: list[str] = []
    matching_provider_policy_digests: list[str] = []
    matching_provider_claims_digests: list[str] = []
    matching_provider_endpoints: list[set[str]] = []
    matching_provider_proof_claims: list[dict[str, Any]] = []
    for provider in providers:
        if not isinstance(provider, dict) or not provider_matches_any(
            provider,
            expected_providers=expected_providers,
            expected_model=model_id,
        ):
            continue
        supported_endpoints = public_string_set(provider.get("supported_endpoints"))
        if supported_endpoints:
            matching_provider_endpoints.append(supported_endpoints)
        confidentiality = provider.get("confidentiality")
        if not isinstance(confidentiality, dict):
            continue
        policy_digest = confidentiality.get("policy_digest")
        if isinstance(policy_digest, str):
            matching_provider_policy_digests.append(policy_digest)
        evidence_digest = confidentiality.get("evidence_digest")
        if isinstance(evidence_digest, str):
            matching_provider_digests.append(evidence_digest)
        claims_digest = confidentiality.get("verified_claims_digest")
        if isinstance(claims_digest, str):
            matching_provider_claims_digests.append(claims_digest)
        proof_claims = confidentiality.get("proof_claims")
        if isinstance(proof_claims, dict):
            matching_provider_proof_claims.append(proof_claims)

    results: list[CheckResult] = []
    model_evidence_digest = model.get("attestation_evidence_digest")
    results.append(
        CheckResult(
            isinstance(model_evidence_digest, str)
            and model_evidence_digest in matching_provider_digests,
            f"{model_id} model evidence digest matches provider status",
        )
    )
    model_confidentiality = model.get("confidentiality")
    if isinstance(model_confidentiality, dict):
        model_policy_digest = model_confidentiality.get("policy_digest")
        results.append(
            CheckResult(
                isinstance(model_policy_digest, str)
                and model_policy_digest in matching_provider_policy_digests,
                f"{model_id} nested model policy digest matches provider status",
            )
        )
        results.append(
            CheckResult(
                model_confidentiality.get("evidence_digest") == model_evidence_digest,
                f"{model_id} nested model evidence digest matches top-level metadata",
            )
        )
        model_claims_digest = model_confidentiality.get("verified_claims_digest")
        results.append(
            CheckResult(
                isinstance(model_claims_digest, str)
                and model_claims_digest in matching_provider_claims_digests,
                f"{model_id} nested model verified claims digest matches provider status",
            )
        )
        model_supported_endpoints = public_string_set(
            model_confidentiality.get("supported_endpoints")
        )
        results.append(
            CheckResult(
                any(
                    endpoints_cover(provider_endpoints, model_supported_endpoints)
                    for provider_endpoints in matching_provider_endpoints
                ),
                (
                    f"{model_id} nested model supported endpoints are covered "
                    "by provider status"
                ),
            )
        )
        model_proof_claims = model_confidentiality.get("proof_claims")
        results.append(
            CheckResult(
                isinstance(model_proof_claims, dict)
                and model_proof_claims in matching_provider_proof_claims,
                f"{model_id} nested model proof claims match provider status",
            )
        )
        results.append(
            CheckResult(
                any(
                    model_confidentiality_matches_provider_status(
                        provider=provider,
                        model_confidentiality=model_confidentiality,
                        model_evidence_digest=model_evidence_digest,
                    )
                    for provider in providers
                    if isinstance(provider, dict)
                    and provider_matches_any(
                        provider,
                        expected_providers=expected_providers,
                        expected_model=model_id,
                    )
                ),
                (
                    f"{model_id} nested model confidentiality snapshot matches "
                    "a single provider status"
                ),
            )
        )
    return results


def validate_cross_surface_consistency(
    *,
    status: dict[str, Any],
    attestation: dict[str, Any],
    models: dict[str, Any],
    hpke_key_config: bytes,
    expected: list[tuple[str, str]],
) -> list[CheckResult]:
    results: list[CheckResult] = []
    hpke_key_config_digest = sha256_bytes_digest(hpke_key_config)

    tee = status.get("routstr_tee")
    status_hpke_public_key_digest = (
        tee.get("hpke_public_key_digest") if isinstance(tee, dict) else None
    )
    if isinstance(tee, dict):
        results.append(
            digest_match_result(
                tee.get("hpke_key_config_digest"),
                hpke_key_config_digest,
                label="status local HPKE key config digest",
            )
        )
        local_verification = tee.get("local_verification")
        if isinstance(local_verification, dict):
            results.extend(
                validate_local_tee_proof_claims(
                    local_verification,
                    label="local TEE status",
                    routing_policy_digest=attestation.get("routing_policy_digest"),
                    routing_policy=(
                        attestation.get("routing_policy")
                        if isinstance(attestation.get("routing_policy"), dict)
                        else None
                    ),
                    hpke_key_config_digest=tee.get("hpke_key_config_digest"),
                    hpke_public_key_digest=tee.get("hpke_public_key_digest"),
                )
            )
    results.append(
        digest_match_result(
            attestation_hpke_key_config_digest(attestation),
            hpke_key_config_digest,
            label="attestation HPKE key config digest",
        )
    )

    results.append(
        CheckResult(
            status_local_verification_evidence_digest(status)
            == attestation_local_verification_evidence_digest(attestation),
            "local TEE evidence digest matches status and attestation",
        )
    )
    results.append(
        CheckResult(
            status_local_verification_claims_digest(status)
            == attestation_local_verification_claims_digest(attestation),
            "local TEE claims digest matches status and attestation",
        )
    )
    results.append(
        CheckResult(
            status_hpke_public_key_digest
            == hpke_public_key_digest_from_attestation(attestation),
            "local TEE HPKE public key digest matches status and attestation",
        )
    )
    results.append(
        CheckResult(
            status_local_verification_proof_claims(status)
            == attestation_local_verification_proof_claims(attestation),
            "local TEE proof claims match status and attestation",
        )
    )

    providers = status.get("providers")
    routing_policy = attestation.get("routing_policy")
    attestation_routable = (
        public_routable_with_full_attestation_map(
            routing_policy.get("routable_with_full_attestation")
        )
        if isinstance(routing_policy, dict)
        else None
    )
    status_routable = public_routable_with_full_attestation_map(
        status.get("routable_with_full_attestation")
    )
    results.append(
        CheckResult(
            status_routable is not None
            and attestation_routable is not None
            and status_routable == attestation_routable,
            "routable_with_full_attestation matches status and attestation",
        )
    )
    results.extend(
        validate_expected_routable_with_full_attestation(
            routable_with_full_attestation=status_routable,
            expected=expected,
        )
    )
    results.extend(
        validate_public_verified_models_routable_with_full_attestation(
            routable_with_full_attestation=status_routable,
            models=models,
        )
    )
    results.extend(
        validate_routable_with_full_attestation_models_are_public_verified(
            routable_with_full_attestation=status_routable,
            models=models,
        )
    )
    attestation_providers = (
        routing_policy.get("providers") if isinstance(routing_policy, dict) else None
    )
    model_data = models.get("data")
    if not isinstance(providers, list) or not isinstance(model_data, list):
        return results

    models_by_id = {
        model_id: item
        for item in model_data
        if isinstance(item, dict)
        for model_id in [public_string_value(item.get("id"))]
        if model_id is not None
    }
    for expected_provider, expected_model in expected:
        model = models_by_id.get(expected_model)
        if not isinstance(model, dict):
            continue

        results.extend(
            validate_model_provider_consistency(
                providers=providers,
                model=model,
                expected_providers={expected_provider},
                model_id=expected_model,
            )
        )
        if isinstance(attestation_providers, list):
            results.extend(
                validate_attested_provider_consistency(
                    status_providers=providers,
                    attestation_providers=attestation_providers,
                    expected_provider=expected_provider,
                    expected_model=expected_model,
                )
            )

    expected_model_ids = {model_id for _, model_id in expected}
    for model in model_data:
        if not isinstance(model, dict):
            continue
        model_id = public_string_value(model.get("id"))
        if model_id is None or model_id in expected_model_ids:
            continue
        model_confidentiality = model.get("confidentiality")
        if not isinstance(model_confidentiality, dict):
            continue
        if model_confidentiality.get("verified") is not True:
            continue
        expected_providers = _model_provider_names(model)
        results.extend(
            validate_model_provider_consistency(
                providers=providers,
                model=model,
                expected_providers=expected_providers,
                model_id=model_id,
            )
        )
        if isinstance(attestation_providers, list):
            for expected_provider in sorted(expected_providers):
                results.extend(
                    validate_attested_provider_consistency(
                        status_providers=providers,
                        attestation_providers=attestation_providers,
                        expected_provider=expected_provider,
                        expected_model=model_id,
                    )
                )

    return results


def inference_auth_headers(
    *,
    bearer_token_env: str | None,
    cashu_token_env: str | None,
) -> dict[str, str]:
    if bearer_token_env and cashu_token_env:
        raise ValueError(
            "choose only one inference auth mode: bearer token env or Cashu token env"
        )
    if bearer_token_env:
        token = os.environ.get(bearer_token_env, "").strip()
        if not token:
            raise ValueError(f"{bearer_token_env} is not set")
        authorization = (
            token if token.lower().startswith("bearer ") else f"Bearer {token}"
        )
        return {"Authorization": authorization}
    if cashu_token_env:
        token = os.environ.get(cashu_token_env, "").strip()
        if not token:
            raise ValueError(f"{cashu_token_env} is not set")
        return {"X-Cashu": token}
    return {}


def validate_chat_completion_response(
    model_id: str,
    response: Any,
) -> list[CheckResult]:
    if not isinstance(response, dict):
        return [CheckResult(False, f"{model_id} chat completion response is not JSON")]
    results: list[CheckResult] = []
    response_model = response.get("model")
    if response_model != model_id:
        results.append(
            CheckResult(
                False,
                f"{model_id} chat completion response model matches requested model",
            )
        )
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        results.append(
            CheckResult(
                False, f"{model_id} chat completion response is missing choices"
            )
        )
        return results
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        results.append(
            CheckResult(
                False,
                f"{model_id} chat completion response choice is not an object",
            )
        )
        return results
    message = first_choice.get("message")
    if not isinstance(message, dict):
        results.append(
            CheckResult(
                False,
                f"{model_id} chat completion response is missing message",
            )
        )
        return results
    content = message.get("content")
    if content is None:
        results.append(
            CheckResult(
                False,
                f"{model_id} chat completion response is missing message content",
            )
        )
        return results
    if results:
        return results
    return [CheckResult(True, f"{model_id} confidential chat completion succeeded")]


def validate_legacy_completion_response(
    model_id: str,
    response: Any,
) -> list[CheckResult]:
    if not isinstance(response, dict):
        return [CheckResult(False, f"{model_id} completion response is not JSON")]
    results: list[CheckResult] = []
    response_model = response.get("model")
    if response_model != model_id:
        results.append(
            CheckResult(
                False,
                f"{model_id} completion response model matches requested model",
            )
        )
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        results.append(
            CheckResult(False, f"{model_id} completion response is missing choices")
        )
        return results
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        results.append(
            CheckResult(False, f"{model_id} completion response choice is not an object")
        )
        return results
    if not isinstance(first_choice.get("text"), str):
        results.append(
            CheckResult(False, f"{model_id} completion response is missing text")
        )
        return results
    if results:
        return results
    return [CheckResult(True, f"{model_id} confidential completion succeeded")]


def validate_embedding_response(
    model_id: str,
    response: Any,
) -> list[CheckResult]:
    if not isinstance(response, dict):
        return [CheckResult(False, f"{model_id} embedding response is not JSON")]
    results: list[CheckResult] = []
    response_model = response.get("model")
    if response_model != model_id:
        results.append(
            CheckResult(
                False,
                f"{model_id} embedding response model matches requested model",
            )
        )
    data = response.get("data")
    if not isinstance(data, list) or not data:
        results.append(
            CheckResult(False, f"{model_id} embedding response is missing data")
        )
        return results
    first_embedding = data[0]
    if not isinstance(first_embedding, dict) or not isinstance(
        first_embedding.get("embedding"),
        list,
    ):
        results.append(
            CheckResult(
                False,
                f"{model_id} embedding response is missing embedding vector",
            )
        )
        return results
    if results:
        return results
    return [CheckResult(True, f"{model_id} confidential embedding succeeded")]


def validate_messages_response(
    model_id: str,
    response: Any,
) -> list[CheckResult]:
    if not isinstance(response, dict):
        return [CheckResult(False, f"{model_id} messages response is not JSON")]
    results: list[CheckResult] = []
    response_model = response.get("model")
    if response_model != model_id:
        results.append(
            CheckResult(
                False,
                f"{model_id} messages response model matches requested model",
            )
        )
    content = response.get("content")
    if not isinstance(content, list) or not content:
        results.append(
            CheckResult(False, f"{model_id} messages response is missing content")
        )
        return results
    if results:
        return results
    return [CheckResult(True, f"{model_id} confidential messages request succeeded")]


def validate_responses_api_response(
    model_id: str,
    response: Any,
) -> list[CheckResult]:
    if not isinstance(response, dict):
        return [CheckResult(False, f"{model_id} Responses API response is not JSON")]
    results: list[CheckResult] = []
    if response.get("object") != "response":
        results.append(
            CheckResult(False, f"{model_id} Responses API response object is response")
        )
    if response.get("model") != model_id:
        results.append(
            CheckResult(
                False,
                f"{model_id} Responses API response model matches requested model",
            )
        )
    if response.get("status") != "completed":
        results.append(
            CheckResult(
                False,
                f"{model_id} Responses API response status is completed",
            )
        )
    output_text = response.get("output_text")
    output = response.get("output")
    has_text_output = isinstance(output_text, str) and bool(output_text.strip())
    has_output_items = isinstance(output, list) and bool(output)
    if not has_text_output and not has_output_items:
        results.append(
            CheckResult(
                False,
                f"{model_id} Responses API response is missing output",
            )
        )
    if results:
        return results
    return [
        CheckResult(True, f"{model_id} confidential Responses API request succeeded")
    ]


def validate_audio_transcription_response(
    model_id: str,
    response: Any,
) -> list[CheckResult]:
    if not isinstance(response, dict):
        return [
            CheckResult(False, f"{model_id} audio transcription response is not JSON")
        ]
    if not isinstance(response.get("text"), str):
        return [
            CheckResult(
                False,
                f"{model_id} audio transcription response is missing text",
            )
        ]
    return [CheckResult(True, f"{model_id} confidential audio transcription succeeded")]


def default_audio_fixture() -> tuple[str, bytes, str]:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 1_600)
    return ("routstr-live-check.wav", buffer.getvalue(), "audio/wav")


def content_type_for_audio_path(path: str) -> str:
    normalized = path.lower()
    if normalized.endswith(".wav"):
        return "audio/wav"
    if normalized.endswith(".mp3"):
        return "audio/mpeg"
    if normalized.endswith(".webm"):
        return "audio/webm"
    if normalized.endswith(".m4a"):
        return "audio/mp4"
    return "application/octet-stream"


def load_audio_fixture(path: str | None) -> tuple[str, bytes, str]:
    if not path:
        return default_audio_fixture()
    with open(path, "rb") as handle:
        data = handle.read()
    filename = os.path.basename(path) or "routstr-live-check-audio"
    return (filename, data, content_type_for_audio_path(filename))


def inference_payload(route: ExpectedInferenceRoute, prompt: str) -> dict[str, Any]:
    if route.endpoint == "/v1/chat/completions":
        return {
            "model": route.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 8,
            "stream": False,
        }
    if route.endpoint == "/v1/completions":
        return {
            "model": route.model,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": 8,
            "stream": False,
        }
    if route.endpoint == "/v1/embeddings":
        return {
            "model": route.model,
            "input": prompt,
        }
    if route.endpoint == "/v1/messages":
        return {
            "model": route.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
            "stream": False,
        }
    if route.endpoint == "/v1/responses":
        return {
            "model": route.model,
            "input": prompt,
            "stream": False,
            "store": False,
        }
    raise ValueError(f"unsupported inference endpoint: {route.endpoint}")


def validate_inference_response(
    route: ExpectedInferenceRoute,
    response: Any,
) -> list[CheckResult]:
    if route.endpoint == "/v1/chat/completions":
        return validate_chat_completion_response(route.model, response)
    if route.endpoint == "/v1/completions":
        return validate_legacy_completion_response(route.model, response)
    if route.endpoint == "/v1/embeddings":
        return validate_embedding_response(route.model, response)
    if route.endpoint == "/v1/messages":
        return validate_messages_response(route.model, response)
    if route.endpoint == "/v1/responses":
        return validate_responses_api_response(route.model, response)
    if route.endpoint in {"/v1/audio/transcriptions", "/v1/audio/translations"}:
        return validate_audio_transcription_response(route.model, response)
    return [CheckResult(False, f"{route.endpoint} inference endpoint is supported")]


def run_inference_checks(
    *,
    base_url: str,
    routes: list[ExpectedInferenceRoute],
    auth_headers: dict[str, str],
    prompt: str,
    timeout: float,
    audio_file: tuple[str, bytes, str] | None = None,
) -> list[CheckResult]:
    if not auth_headers:
        return [
            CheckResult(
                False,
                "confidential inference check requires --bearer-token-env or --cashu-token-env",
            )
        ]

    results: list[CheckResult] = []
    for route in routes:
        try:
            if route.endpoint in {
                "/v1/audio/transcriptions",
                "/v1/audio/translations",
            }:
                filename, content, content_type = audio_file or default_audio_fixture()
                response = post_multipart(
                    base_url,
                    route.endpoint,
                    fields={"model": route.model},
                    files=[
                        ("file", filename, content, content_type),
                    ],
                    headers=auth_headers,
                    timeout=timeout,
                )
            else:
                response = post_json(
                    base_url,
                    route.endpoint,
                    payload=inference_payload(route, prompt),
                    headers=auth_headers,
                    timeout=timeout,
                )
        except Exception as exc:
            results.append(
                CheckResult(
                    False,
                    f"{route.model} confidential {route.endpoint} failed: {exc}",
                )
            )
            continue
        results.extend(validate_inference_response(route, response))
    return results


def gate_inference_checks(surface_results: list[CheckResult]) -> list[CheckResult]:
    if all(result.ok for result in surface_results):
        return []
    return [
        CheckResult(
            False,
            "confidential inference check skipped because public attestation checks failed",
        )
    ]


def collect_results(
    *,
    status: dict[str, Any],
    attestation: dict[str, Any],
    models: dict[str, Any],
    hpke_key_config: bytes,
    hpke_content_type: str,
    expected: list[tuple[str, str]],
    inference_routes: list[ExpectedInferenceRoute] | None = None,
    provider_model_catalog: ProviderModelCatalog | None = None,
    attestation_results_directory: AttestationResultsDirectory | None = None,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    results.append(
        CheckResult(
            status.get("mode") == "required",
            "confidential routing mode is required",
        )
    )
    results.append(
        CheckResult(status.get("required") is True, "confidential routing is required")
    )
    results.append(
        require_true_boolean(
            status.get("end_to_end_ready"),
            label="/v1/confidentiality/status end_to_end_ready",
        )
    )
    results.extend(validate_public_identity_fields(status=status, models=models))
    results.extend(validate_provider_mode_contracts(status))
    results.extend(validate_provider_confidentiality_statuses(status))
    results.extend(validate_public_metadata_lists(status=status, models=models))
    results.extend(
        validate_model_confidentiality_statuses(
            models,
            provider_policies_by_digest=provider_policy_snapshots_by_digest(status),
        )
    )
    results.extend(validate_routstr_tee(status))
    results.extend(validate_models_routstr_tee(status=status, models=models))
    for expected_provider, expected_model in expected:
        results.extend(
            validate_expected_provider(
                status,
                expected_provider=expected_provider,
                expected_model=expected_model,
            )
        )
    results.extend(
        validate_expected_provider_catalog_models(
            expected=expected,
            provider_model_catalog=provider_model_catalog,
        )
    )
    results.extend(
        validate_public_verified_models_in_provider_catalog(
            models=models,
            provider_model_catalog=provider_model_catalog,
        )
    )
    results.extend(
        validate_attestation_results_directory(
            attestation=attestation,
            attestation_results_directory=attestation_results_directory,
        )
    )
    results.extend(
        validate_models(
            models,
            expected,
            require_all_confidential=status.get("required") is True,
        )
    )
    if inference_routes:
        results.extend(
            validate_expected_inference_support(
                models=models,
                routes=inference_routes,
            )
        )
    results.extend(validate_attestation_statement(attestation))
    results.extend(validate_hpke_keys(hpke_key_config, hpke_content_type))
    results.extend(
        validate_cross_surface_consistency(
            status=status,
            attestation=attestation,
            models=models,
            hpke_key_config=hpke_key_config,
            expected=expected,
        )
    )

    public_payload = {
        "status": status,
        "attestation": attestation,
        "models": models,
    }
    forbidden_paths = forbidden_public_paths(public_payload)
    if forbidden_paths:
        results.append(
            CheckResult(
                False,
                "public confidentiality surfaces expose forbidden fields: "
                + ", ".join(forbidden_paths[:10]),
            )
        )
    else:
        results.append(
            CheckResult(True, "public confidentiality surfaces are redacted")
        )
    runtime_forbidden_paths = forbidden_runtime_public_paths(
        {"status": status, "models": models}
    )
    if runtime_forbidden_paths:
        results.append(
            CheckResult(
                False,
                "runtime public confidentiality surfaces expose forbidden fields: "
                + ", ".join(runtime_forbidden_paths[:10]),
            )
        )
    else:
        results.append(
            CheckResult(
                True, "runtime public confidentiality surfaces omit provider URLs"
            )
        )
    return results


def parse_expected_provider(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "expected provider must use provider:model format"
        )
    provider, model = value.split(":", 1)
    provider = provider.strip()
    model = model.strip()
    if not provider or not model:
        raise argparse.ArgumentTypeError(
            "expected provider must include non-empty provider and model"
        )
    return provider, model


def normalize_inference_endpoint(value: str) -> str:
    endpoint = value.strip().lower().replace("_", "-")
    if endpoint in INFERENCE_ENDPOINT_ALIASES:
        return INFERENCE_ENDPOINT_ALIASES[endpoint]
    raise argparse.ArgumentTypeError(
        "inference endpoint must be one of chat-completions, completions, responses, embeddings, messages, audio, or audio-translations"
    )


def parse_expected_inference_route(value: str) -> ExpectedInferenceRoute:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "expected inference must use provider:model:endpoint format"
        )
    provider, model, endpoint = (part.strip() for part in parts)
    if not provider or not model or not endpoint:
        raise argparse.ArgumentTypeError(
            "expected inference must include non-empty provider, model, and endpoint"
        )
    return ExpectedInferenceRoute(
        provider=provider,
        model=model,
        endpoint=normalize_inference_endpoint(endpoint),
    )


DEFAULT_INFERENCE_ENDPOINT_PRIORITY = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/v1/messages",
    "/v1/embeddings",
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
)


def _model_by_id(models: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    data = models.get("data")
    if not isinstance(data, list):
        return None
    for item in data:
        if not isinstance(item, dict):
            continue
        if public_string_value(item.get("id")) == model_id:
            return item
    return None


def default_inference_endpoint_for_model(
    *,
    provider: str,
    model_id: str,
    models: dict[str, Any] | None,
) -> str:
    if models is None:
        return "/v1/chat/completions"
    model = _model_by_id(models, model_id)
    if not isinstance(model, dict) or provider not in _model_provider_names(model):
        return "/v1/chat/completions"
    confidentiality = model.get("confidentiality")
    supported_endpoints = (
        public_string_set(confidentiality.get("supported_endpoints"))
        if isinstance(confidentiality, dict)
        else set()
    )
    for endpoint in DEFAULT_INFERENCE_ENDPOINT_PRIORITY:
        if endpoints_cover(supported_endpoints, {endpoint}):
            return endpoint
    return "/v1/chat/completions"


def default_inference_routes(
    expected: list[tuple[str, str]],
    *,
    models: dict[str, Any] | None = None,
) -> list[ExpectedInferenceRoute]:
    return [
        ExpectedInferenceRoute(
            provider=provider,
            model=model,
            endpoint=default_inference_endpoint_for_model(
                provider=provider,
                model_id=model,
                models=models,
            ),
        )
        for provider, model in expected
    ]


def complete_inference_routes(
    expected: list[tuple[str, str]],
    explicit_routes: list[ExpectedInferenceRoute],
    *,
    models: dict[str, Any] | None = None,
    routable_with_full_attestation: dict[str, list[str]] | None = None,
) -> list[ExpectedInferenceRoute]:
    routes = list(explicit_routes)
    covered_pairs = {(route.provider, route.model) for route in routes}
    expected_pairs = list(expected)
    seen_expected = set(expected_pairs)
    if routable_with_full_attestation is not None:
        for provider, model_ids in sorted(routable_with_full_attestation.items()):
            for model_id in model_ids:
                pair = (provider, model_id)
                if pair not in seen_expected:
                    expected_pairs.append(pair)
                    seen_expected.add(pair)
    missing_expected = [pair for pair in expected_pairs if pair not in covered_pairs]
    routes.extend(default_inference_routes(missing_expected, models=models))
    return routes


def expected_pairs_from_routes(
    expected: list[tuple[str, str]],
    routes: list[ExpectedInferenceRoute],
) -> list[tuple[str, str]]:
    pairs = list(expected)
    seen = set(pairs)
    for route in routes:
        pair = (route.provider, route.model)
        if pair not in seen:
            pairs.append(pair)
            seen.add(pair)
    return pairs


def validate_expected_inference_support(
    *,
    models: dict[str, Any],
    routes: list[ExpectedInferenceRoute],
) -> list[CheckResult]:
    data = models.get("data")
    if not isinstance(data, list):
        return [CheckResult(False, "/v1/models response is missing data list")]

    by_id = {
        model_id: item
        for item in data
        if isinstance(item, dict)
        for model_id in [public_string_value(item.get("id"))]
        if model_id is not None
    }
    results: list[CheckResult] = []
    for route in routes:
        model = by_id.get(route.model)
        if not isinstance(model, dict):
            results.append(
                CheckResult(False, f"{route.model} is missing from /v1/models")
            )
            continue
        if route.provider not in _model_provider_names(model):
            results.append(
                CheckResult(
                    False,
                    f"{route.model} inference provider does not match {route.provider}",
                )
            )
            continue
        confidentiality = model.get("confidentiality")
        supported_endpoints = (
            public_string_set(confidentiality.get("supported_endpoints"))
            if isinstance(confidentiality, dict)
            else set()
        )
        if endpoints_cover(supported_endpoints, {route.endpoint}):
            results.append(
                CheckResult(
                    True,
                    f"{route.model} advertises {route.endpoint} for live inference",
                )
            )
        else:
            results.append(
                CheckResult(
                    False,
                    f"{route.model} does not advertise {route.endpoint} for live inference",
                )
            )
    return results


def validate_expected_routable_with_full_attestation(
    *,
    routable_with_full_attestation: dict[str, list[str]] | None,
    expected: list[tuple[str, str]],
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for expected_provider, expected_model in expected:
        models = (
            routable_with_full_attestation.get(expected_provider)
            if routable_with_full_attestation is not None
            else None
        )
        results.append(
            CheckResult(
                isinstance(models, list) and expected_model in models,
                (
                    f"{expected_model} is listed in "
                    f"routable_with_full_attestation[{expected_provider}]"
                ),
            )
        )
    return results


def validate_public_verified_models_routable_with_full_attestation(
    *,
    routable_with_full_attestation: dict[str, list[str]] | None,
    models: dict[str, Any],
) -> list[CheckResult]:
    data = models.get("data")
    if not isinstance(data, list):
        return []
    results: list[CheckResult] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if not (
            item.get("confidential") is True
            and item.get("attestation_status") == "verified"
            and item.get("provider_attestation_status") == "verified"
        ):
            continue
        model_id = public_string_value(item.get("id")) or "<unknown>"
        confidentiality = item.get("confidentiality")
        nested_model_ids = (
            strict_public_string_set(confidentiality.get("model_ids"), allow_empty=False)
            if isinstance(confidentiality, dict)
            else None
        )
        results.append(
            CheckResult(
                nested_model_ids is not None and model_id in nested_model_ids,
                f"{model_id} model id matches nested confidentiality model_ids",
            )
        )
        provider_names = _model_provider_names(item)
        if len(provider_names) != 1:
            results.append(
                CheckResult(
                    False,
                    f"{model_id} has exactly one confidential provider identity",
                )
            )
            continue
        provider_name = next(iter(provider_names))
        provider_models = (
            routable_with_full_attestation.get(provider_name)
            if routable_with_full_attestation is not None
            else None
        )
        results.append(
            CheckResult(
                isinstance(provider_models, list) and model_id in provider_models,
                (
                    f"{model_id} is listed in "
                    f"routable_with_full_attestation[{provider_name}]"
                ),
            )
        )
    return results


def validate_routable_with_full_attestation_models_are_public_verified(
    *,
    routable_with_full_attestation: dict[str, list[str]] | None,
    models: dict[str, Any],
) -> list[CheckResult]:
    data = models.get("data")
    if routable_with_full_attestation is None or not isinstance(data, list):
        return []
    verified_models_by_provider: dict[str, set[str]] = {
        provider: set() for provider in routable_with_full_attestation
    }
    for item in data:
        if not isinstance(item, dict):
            continue
        if not (
            item.get("confidential") is True
            and item.get("attestation_status") == "verified"
            and item.get("provider_attestation_status") == "verified"
        ):
            continue
        model_id = public_string_value(item.get("id"))
        if model_id is None:
            continue
        for provider_name in _model_provider_names(item):
            if provider_name in verified_models_by_provider:
                verified_models_by_provider[provider_name].add(model_id)

    results: list[CheckResult] = []
    for provider_name, model_ids in routable_with_full_attestation.items():
        verified_models = verified_models_by_provider.get(provider_name, set())
        for model_id in model_ids:
            results.append(
                CheckResult(
                    model_id in verified_models,
                    (
                        f"routable_with_full_attestation[{provider_name}] model "
                        f"{model_id} is fully verified in /v1/models"
                    ),
                )
            )
    return results


def validate_external_guardrail_routable_with_full_attestation_matches_status(
    *,
    status_routable_with_full_attestation: dict[str, list[str]] | None,
    external_guardrails: dict[str, Any],
) -> list[CheckResult]:
    external_routable = public_routable_with_full_attestation_map(
        external_guardrails.get("routable_with_full_attestation")
    )
    return [
        CheckResult(
            status_routable_with_full_attestation is not None
            and external_routable is not None
            and external_routable == status_routable_with_full_attestation,
            "external guardrail routable_with_full_attestation matches public status",
        )
    ]


def provider_catalog_slug_for_expected_provider(provider: str) -> str | None:
    if provider in {"ppq", "ppq-private", "ppq-private-tee"}:
        return "ppq"
    if provider in {"tinfoil", "privatemode"}:
        return provider
    return None


def provider_catalog_contains_model(
    catalog_models: set[str],
    model_id: str,
    *,
    provider_slug: str,
) -> bool:
    selected = provider_catalog_identity_candidate_variants(
        model_id.split("/")[-1],
        provider_slug=provider_slug,
    )
    catalog_candidates = {
        candidate
        for catalog_model in catalog_models
        for candidate in provider_catalog_identity_candidate_variants(
            catalog_model.split("/")[-1],
            provider_slug=provider_slug,
        )
    }
    return bool(selected.intersection(catalog_candidates))


def validate_expected_provider_catalog_models(
    *,
    expected: list[tuple[str, str]],
    provider_model_catalog: ProviderModelCatalog | None,
) -> list[CheckResult]:
    if provider_model_catalog is None:
        return []

    results: list[CheckResult] = []
    for expected_provider, expected_model in expected:
        provider_slug = provider_catalog_slug_for_expected_provider(expected_provider)
        if provider_slug is None:
            continue
        catalog_models = provider_model_catalog.provider_models.get(provider_slug)
        if not catalog_models:
            results.append(
                CheckResult(
                    False,
                    "confidential-inference provider catalog does not include "
                    f"models for {provider_slug}",
                )
            )
            continue
        if provider_catalog_contains_model(
            catalog_models,
            expected_model,
            provider_slug=provider_slug,
        ):
            results.append(
                CheckResult(
                    True,
                    f"selected model {expected_model} is present in "
                    f"confidential-inference provider catalog for {provider_slug}",
                )
            )
            continue
        results.append(
            CheckResult(
                False,
                f"selected model {expected_model} is not present in "
                f"confidential-inference provider catalog for {provider_slug}",
            )
        )
    return results


def is_public_provider_verified_confidential_model(model: dict[str, Any]) -> bool:
    confidentiality = model.get("confidentiality")
    return (
        model.get("provider_attestation_status") == "verified"
        and isinstance(confidentiality, dict)
        and confidentiality.get("enabled") is True
        and confidentiality.get("verified") is True
    )


def provider_catalog_slug_for_public_model(
    model: dict[str, Any],
    confidentiality: dict[str, Any],
) -> str | None:
    for value in (
        public_string_value(confidentiality.get("provider_type")),
        public_string_value(model.get("attestation_provider")),
        public_string_value(confidentiality.get("mode")),
    ):
        if value is None:
            continue
        provider_slug = provider_catalog_slug_for_expected_provider(value)
        if provider_slug is not None:
            return provider_slug
    return None


def validate_public_verified_models_in_provider_catalog(
    *,
    models: dict[str, Any],
    provider_model_catalog: ProviderModelCatalog | None,
) -> list[CheckResult]:
    if provider_model_catalog is None:
        return []

    data = models.get("data")
    if not isinstance(data, list):
        return []

    results: list[CheckResult] = []
    for item in data:
        if not isinstance(item, dict) or not is_public_provider_verified_confidential_model(item):
            continue
        model_id = public_string_value(item.get("id"))
        confidentiality = item.get("confidentiality")
        if model_id is None or not isinstance(confidentiality, dict):
            continue
        provider_slug = provider_catalog_slug_for_public_model(item, confidentiality)
        if provider_slug is None:
            continue
        catalog_models = provider_model_catalog.provider_models.get(provider_slug)
        if not catalog_models:
            results.append(
                CheckResult(
                    False,
                    "confidential-inference provider catalog does not include "
                    f"models for {provider_slug}",
                )
            )
            continue
        if provider_catalog_contains_model(
            catalog_models,
            model_id,
            provider_slug=provider_slug,
        ):
            results.append(
                CheckResult(
                    True,
                    f"public provider-verified model {model_id} is present in "
                    f"confidential-inference provider catalog for {provider_slug}",
                )
            )
            continue
        results.append(
            CheckResult(
                False,
                f"public provider-verified model {model_id} is not present in "
                f"confidential-inference provider catalog for {provider_slug}",
            )
        )
    return results


def print_results(results: list[CheckResult]) -> None:
    for result in results:
        status = "ok" if result.ok else "missing"
        print(f"[{status}] {result.message}")


def confidential_routes_ready_value(
    *,
    end_to_end_ready: bool,
    external_guardrails_ready: bool,
    inference_exercised: bool,
    expected_inference_route_count: int,
    inference_attempt_count: int,
    expected_inference_routes: list[dict[str, str]] | None,
    attempted_inference_routes: list[dict[str, str]] | None,
) -> bool:
    expected_routes = expected_inference_routes or []
    attempted_routes = attempted_inference_routes or []
    return (
        end_to_end_ready
        and external_guardrails_ready
        and inference_exercised
        and expected_inference_route_count > 0
        and inference_attempt_count == expected_inference_route_count
        and bool(expected_routes)
        and expected_routes == attempted_routes
    )


def json_report(
    results: list[CheckResult],
    *,
    inference_exercised: bool = False,
    expected_inference_route_count: int = 0,
    inference_attempt_count: int = 0,
    expected_inference_routes: list[dict[str, str]] | None = None,
    attempted_inference_routes: list[dict[str, str]] | None = None,
    external_guardrails: dict[str, Any] | None = None,
    public_surfaces: dict[str, Any] | None = None,
) -> dict[str, Any]:
    external_guardrails = external_guardrails or {}
    routable_with_full_attestation = external_guardrails.get(
        "routable_with_full_attestation"
    )
    if not isinstance(routable_with_full_attestation, dict):
        routable_with_full_attestation = {
            "tinfoil": [],
            "ppq-private": [],
            "privatemode": [],
        }
    end_to_end_ready = all(result.ok for result in results)
    external_guardrails_ready = (
        external_guardrails.get("strict_external_guardrails") is True
        and
        external_guardrails.get("external_guardrails_ready") is True
    )
    confidential_routes_ready = confidential_routes_ready_value(
        end_to_end_ready=end_to_end_ready,
        external_guardrails_ready=external_guardrails_ready,
        inference_exercised=inference_exercised,
        expected_inference_route_count=expected_inference_route_count,
        inference_attempt_count=inference_attempt_count,
        expected_inference_routes=expected_inference_routes,
        attempted_inference_routes=attempted_inference_routes,
    )
    return {
        "confidential_routes_ready": confidential_routes_ready,
        "end_to_end_ready": end_to_end_ready,
        "external_guardrails_ready": external_guardrails_ready,
        "routable_with_full_attestation": routable_with_full_attestation,
        "external_guardrails": external_guardrails,
        "public_surfaces": public_surfaces or {},
        "inference_exercised": inference_exercised,
        "expected_inference_route_count": expected_inference_route_count,
        "inference_attempt_count": inference_attempt_count,
        "expected_inference_routes": expected_inference_routes or [],
        "attempted_inference_routes": attempted_inference_routes or [],
        "checks": [{"ok": result.ok, "message": result.message} for result in results],
    }


def inference_route_report(route: ExpectedInferenceRoute) -> dict[str, str]:
    return {
        "provider": route.provider,
        "model": route.model,
        "endpoint": route.endpoint,
    }


def public_surface_report(
    *,
    status: Any,
    attestation: Any,
    models: Any,
    hpke_key_config: bytes,
    hpke_content_type: str,
) -> dict[str, Any]:
    return {
        "status": {
            "path": "/v1/confidentiality/status",
            "sha256": sha256_json_digest(status),
        },
        "attestation": {
            "path": "/.well-known/routstr-attestation",
            "sha256": sha256_json_digest(attestation),
        },
        "models": {
            "path": "/v1/models",
            "sha256": sha256_json_digest(models),
        },
        "hpke_keys": {
            "path": "/.well-known/hpke-keys",
            "content_type": hpke_content_type,
            "sha256": sha256_bytes_digest(hpke_key_config),
        },
    }


def external_guardrail_report(
    *,
    strict_external_guardrails: bool,
    needs_tinfoil_directory: bool,
    needs_ppq_results: bool,
    attestation: dict[str, Any],
    provider_model_catalog: ProviderModelCatalog | None,
    attestation_targets_directory: AttestationTargetsDirectory | None,
    attestation_results_directory: AttestationResultsDirectory | None,
    expected_tinfoil_models: set[str] | None = None,
    expected_ppq_models: set[str] | None = None,
    expected_privatemode_models: set[str] | None = None,
    provider_catalog_json: Path | None = None,
    attestation_targets_json: Path | None = None,
    attestation_results_json: Path | None = None,
    max_attestation_result_age_seconds: int | None = None,
) -> dict[str, Any]:
    tinfoil_required_hosts = sorted(tinfoil_result_hosts_from_attestation(attestation))
    tinfoil_statuses = (
        attestation_results_directory.provider_host_statuses.get("tinfoil")
        if attestation_results_directory is not None
        else None
    ) or {}
    tinfoil_errors = (
        attestation_results_directory.provider_host_errors.get("tinfoil")
        if attestation_results_directory is not None
        else None
    ) or {}
    tinfoil_tls_optional_hosts = tinfoil_tls_optional_result_hosts_from_attestation(
        attestation
    )
    tinfoil_verified_hosts = [
        host
        for host in tinfoil_required_hosts
        if tinfoil_statuses.get(host) == "verified"
        or (
            host in tinfoil_tls_optional_hosts
            and tinfoil_statuses.get(host) == "failed"
            and tinfoil_errors.get(host) == TINFOIL_TLS_BINDING_NOT_VERIFIED_ERROR
        )
    ]
    tinfoil_results_verified = bool(tinfoil_required_hosts) and (
        tinfoil_verified_hosts == tinfoil_required_hosts
    )
    tinfoil_required_model_set = set(expected_tinfoil_models or set())
    tinfoil_required_model_set.update(attested_tinfoil_models(attestation))
    tinfoil_required_models = sorted(tinfoil_required_model_set)
    tinfoil_target_models = target_directory_tinfoil_models(
        attestation_targets_directory
    )
    tinfoil_target_models_verified = not tinfoil_required_models or all(
        tinfoil_model_set_contains(model, tinfoil_target_models)
        for model in tinfoil_required_models
    )
    tinfoil_model_statuses = (
        attestation_results_directory.provider_model_statuses.get("tinfoil")
        if attestation_results_directory is not None
        else None
    ) or {}
    tinfoil_model_details = (
        attestation_results_directory.provider_model_details.get("tinfoil")
        if attestation_results_directory is not None
        else None
    ) or {}
    tinfoil_model_release_checks = {
        model: tinfoil_result_release_checks(
            attestation=attestation,
            model_id=model,
            details=tinfoil_model_details_for_model(model, tinfoil_model_details),
        )
        for model in tinfoil_required_models
        if tinfoil_model_status_is_verified(model, tinfoil_model_statuses)
    }
    tinfoil_verified_models = [
        model
        for model in tinfoil_required_models
        if tinfoil_model_status_is_verified(model, tinfoil_model_statuses)
        and all(check.ok for check in tinfoil_model_release_checks.get(model, []))
    ]
    tinfoil_models_verified = not tinfoil_required_models or (
        tinfoil_verified_models == tinfoil_required_models
    )
    tinfoil_results_verified = (
        tinfoil_results_verified
        and tinfoil_target_models_verified
        and tinfoil_models_verified
    )
    ppq_required_hosts = sorted(ppq_private_result_hosts_from_attestation(attestation))
    ppq_statuses = (
        attestation_results_directory.provider_host_statuses.get("ppq")
        if attestation_results_directory is not None
        else None
    ) or {}
    ppq_verified_hosts = [
        host for host in ppq_required_hosts if ppq_statuses.get(host) == "verified"
    ]
    ppq_required_model_set = set(expected_ppq_models or set())
    ppq_required_model_set.update(attested_ppq_private_models(attestation))
    ppq_required_models = sorted(ppq_required_model_set)
    ppq_model_statuses = (
        attestation_results_directory.provider_model_statuses.get("ppq")
        if attestation_results_directory is not None
        else None
    ) or {}
    ppq_model_details = (
        attestation_results_directory.provider_model_details.get("ppq")
        if attestation_results_directory is not None
        else None
    ) or {}
    ppq_target_models = (
        target_directory_ppq_models(attestation_targets_directory)
    )
    ppq_target_models_verified = not ppq_required_models or all(
        model in ppq_target_models for model in ppq_required_models
    )
    ppq_model_release_checks = {
        model: ppq_private_result_release_checks(
            attestation=attestation,
            model_id=model,
            details=ppq_model_details.get(model) or {},
        )
        for model in ppq_required_models
        if ppq_model_statuses.get(model) == "verified"
    }
    ppq_verified_models = [
        model
        for model in ppq_required_models
        if ppq_model_statuses.get(model) == "verified"
        and model in ppq_target_models
        and all(check.ok for check in ppq_model_release_checks.get(model, []))
    ]
    ppq_hosts_verified = bool(ppq_required_hosts) and (
        ppq_verified_hosts == ppq_required_hosts
    )
    ppq_targets_supplied = (
        not needs_ppq_results or attestation_targets_directory is not None
    )
    ppq_model_rows_required = bool(
        ppq_required_models or ppq_model_statuses or ppq_target_models
    )
    ppq_models_verified = not ppq_model_rows_required or (
        bool(ppq_required_models) and ppq_verified_models == ppq_required_models
    )
    ppq_results_verified = (
        ppq_targets_supplied
        and ppq_target_models_verified
        and ppq_hosts_verified
        and ppq_models_verified
    )
    privatemode_required_hosts = sorted(
        privatemode_result_hosts_from_attestation(attestation)
    )
    privatemode_statuses = (
        attestation_results_directory.provider_host_statuses.get("privatemode")
        if attestation_results_directory is not None
        else None
    ) or {}
    privatemode_verified_hosts = [
        host
        for host in privatemode_required_hosts
        if privatemode_statuses.get(host) == "verified"
    ]
    privatemode_required_model_set = set(expected_privatemode_models or set())
    privatemode_required_model_set.update(attested_privatemode_models(attestation))
    privatemode_required_models = sorted(privatemode_required_model_set)
    privatemode_model_statuses = (
        attestation_results_directory.provider_model_statuses.get("privatemode")
        if attestation_results_directory is not None
        else None
    ) or {}
    privatemode_model_details = (
        attestation_results_directory.provider_model_details.get("privatemode")
        if attestation_results_directory is not None
        else None
    ) or {}
    privatemode_target_models = target_directory_provider_models(
        attestation_targets_directory,
        "privatemode",
    )
    privatemode_target_models_verified = not privatemode_required_models or all(
        model in privatemode_target_models for model in privatemode_required_models
    )
    privatemode_model_policy_checks = {
        model: privatemode_result_policy_checks(
            attestation=attestation,
            model_id=model,
            details=privatemode_model_details.get(model) or {},
        )
        for model in privatemode_required_models
        if privatemode_model_statuses.get(model) == "verified"
    }
    privatemode_verified_models = [
        model
        for model in privatemode_required_models
        if privatemode_model_statuses.get(model) == "verified"
        and model in privatemode_target_models
        and all(check.ok for check in privatemode_model_policy_checks.get(model, []))
    ]
    privatemode_hosts_verified = not privatemode_required_hosts or (
        privatemode_verified_hosts == privatemode_required_hosts
    )
    privatemode_targets_supplied = not (
        privatemode_required_hosts or privatemode_required_models
    ) or attestation_targets_directory is not None
    privatemode_models_verified = not privatemode_required_models or (
        privatemode_verified_models == privatemode_required_models
    )
    privatemode_results_required = bool(
        privatemode_required_hosts or privatemode_required_models
    )
    privatemode_results_verified = privatemode_results_required and (
        privatemode_targets_supplied
        and privatemode_target_models_verified
        and privatemode_hosts_verified
        and privatemode_models_verified
    )
    tinfoil_guardrails_ready = (
        not needs_tinfoil_directory
        or (
            attestation_targets_directory is not None
            and attestation_results_directory is not None
            and tinfoil_results_verified
        )
    )
    ppq_guardrails_ready = not needs_ppq_results or ppq_results_verified
    privatemode_guardrails_ready = not (
        privatemode_required_hosts or privatemode_required_models
    ) or privatemode_results_verified
    external_guardrails_ready = (
        provider_model_catalog is not None
        and tinfoil_guardrails_ready
        and ppq_guardrails_ready
        and privatemode_guardrails_ready
    )
    tinfoil_routable_models = tinfoil_verified_models if tinfoil_results_verified else []
    ppq_routable_models = ppq_verified_models if ppq_results_verified else []
    privatemode_routable_models = (
        privatemode_verified_models if privatemode_results_verified else []
    )
    report: dict[str, Any] = {
        "strict_external_guardrails": strict_external_guardrails,
        "external_guardrails_ready": external_guardrails_ready,
        "provider_catalog_supplied": provider_model_catalog is not None,
        "tinfoil_directory_required": needs_tinfoil_directory,
        "tinfoil_results_verified": tinfoil_results_verified,
        "tinfoil_required_hosts": tinfoil_required_hosts,
        "tinfoil_verified_hosts": tinfoil_verified_hosts,
        "tinfoil_required_models": tinfoil_required_models,
        "tinfoil_verified_models": tinfoil_verified_models,
        "tinfoil_routable_with_full_attestation": tinfoil_routable_models,
        "tinfoil_target_models_verified": tinfoil_target_models_verified,
        "tinfoil_model_release_checks": {
            model: [
                {"ok": check.ok, "message": check.message}
                for check in checks
            ]
            for model, checks in sorted(tinfoil_model_release_checks.items())
        },
        "ppq_results_required": needs_ppq_results,
        "ppq_attestation_targets_supplied": ppq_targets_supplied,
        "ppq_results_verified": ppq_results_verified,
        "ppq_required_hosts": ppq_required_hosts,
        "ppq_verified_hosts": ppq_verified_hosts,
        "ppq_required_models": ppq_required_models,
        "ppq_verified_models": ppq_verified_models,
        "ppq_routable_with_full_attestation": ppq_routable_models,
        "ppq_target_models_verified": ppq_target_models_verified,
        "ppq_model_release_checks": {
            model: [
                {"ok": check.ok, "message": check.message}
                for check in checks
            ]
            for model, checks in sorted(ppq_model_release_checks.items())
        },
        "privatemode_results_verified": privatemode_results_verified,
        "privatemode_attestation_targets_supplied": privatemode_targets_supplied,
        "privatemode_required_hosts": privatemode_required_hosts,
        "privatemode_verified_hosts": privatemode_verified_hosts,
        "privatemode_required_models": privatemode_required_models,
        "privatemode_verified_models": privatemode_verified_models,
        "privatemode_routable_with_full_attestation": privatemode_routable_models,
        "privatemode_target_models_verified": privatemode_target_models_verified,
        "privatemode_model_policy_checks": {
            model: [
                {"ok": check.ok, "message": check.message}
                for check in checks
            ]
            for model, checks in privatemode_model_policy_checks.items()
        },
        "attestation_targets_supplied": attestation_targets_directory is not None,
        "attestation_results_supplied": attestation_results_directory is not None,
        "max_attestation_result_age_seconds": max_attestation_result_age_seconds,
        "routable_with_full_attestation": {
            "tinfoil": tinfoil_routable_models,
            "ppq-private": ppq_routable_models,
            "privatemode": privatemode_routable_models,
        },
    }
    for prefix, path in (
        ("provider_catalog", provider_catalog_json),
        ("attestation_targets", attestation_targets_json),
        ("attestation_results", attestation_results_json),
    ):
        if path is None:
            continue
        report[f"{prefix}_path"] = str(path)
        report[f"{prefix}_sha256"] = sha256_file_digest(path)
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    def env_flag_enabled(name: str) -> bool:
        value = os.getenv(name)
        if value is None:
            return False
        return value.strip().lower() not in {"", "0", "false", "no", "off"}

    def env_path(name: str) -> Path | None:
        value = os.getenv(name)
        if value is None or not value.strip():
            return None
        return Path(value)

    def env_expected_providers() -> list[tuple[str, str]]:
        value = os.getenv("EXPECT_PROVIDER")
        if value is None or not value.strip():
            return []
        return [parse_expected_provider(item) for item in shlex.split(value)]

    def env_expected_inference_routes() -> list[ExpectedInferenceRoute]:
        value = os.getenv("EXPECT_INFERENCE")
        if value is None or not value.strip():
            return []
        return [parse_expected_inference_route(item) for item in shlex.split(value)]

    parser = argparse.ArgumentParser(
        description=(
            "Validate a deployed Routstr node's public confidentiality surfaces "
            "before trusting verified-only confidential routing."
        )
    )
    parser.add_argument("base_url", help="deployed Routstr base URL")
    parser.add_argument(
        "--expect-provider",
        action="append",
        type=parse_expected_provider,
        default=None,
        metavar="PROVIDER:MODEL",
        help="expected verified provider/model pair, for example tinfoil:tinfoil/gpt-secure",
    )
    parser.add_argument(
        "--expect-inference",
        action="append",
        type=parse_expected_inference_route,
        default=None,
        metavar="PROVIDER:MODEL:ENDPOINT",
        help=(
            "expected provider/model/endpoint to validate and optionally exercise; "
            "endpoint may be chat-completions, completions, responses, embeddings, messages, audio, or audio-translations"
        ),
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--json",
        action="store_true",
        default=env_flag_enabled("LIVE_CHECK_JSON"),
        help="emit JSON results",
    )
    parser.add_argument(
        "--run-inference",
        action="store_true",
        default=env_flag_enabled("RUN_INFERENCE"),
        help=(
            "send a real request for each expected inference route; defaults to "
            "/v1/chat/completions for --expect-provider routes"
        ),
    )
    parser.add_argument(
        "--strict-inference-exercised",
        action="store_true",
        default=env_flag_enabled("STRICT_INFERENCE_EXERCISED"),
        help=(
            "fail unless --run-inference actually exercises the protected request path"
        ),
    )
    parser.add_argument(
        "--strict-confidential-routes-ready",
        action="store_true",
        default=env_flag_enabled("STRICT_CONFIDENTIAL_ROUTES_READY"),
        help=(
            "fail unless public proof, external guardrails, and exercised inference "
            "prove every expected confidential route"
        ),
    )
    parser.add_argument(
        "--bearer-token-env",
        default=os.getenv("BEARER_TOKEN_ENV"),
        help="environment variable containing a Routstr sk-* token for inference",
    )
    parser.add_argument(
        "--cashu-token-env",
        default=os.getenv("CASHU_TOKEN_ENV"),
        help="environment variable containing a Cashu token for inference",
    )
    parser.add_argument(
        "--inference-prompt",
        default=os.getenv("INFERENCE_PROMPT", "Return the word ok."),
        help="non-secret prompt used only when --run-inference is set",
    )
    parser.add_argument(
        "--inference-audio-file",
        default=os.getenv("INFERENCE_AUDIO_FILE"),
        help=(
            "optional non-secret audio file for /v1/audio/transcriptions or /v1/audio/translations live checks; "
            "defaults to a tiny generated WAV"
        ),
    )
    parser.add_argument(
        "--provider-catalog-json",
        type=Path,
        default=env_path("PROVIDER_CATALOG_JSON"),
        help=(
            "optional confidential-inference providers.json file used to "
            "cross-check expected provider model IDs"
        ),
    )
    parser.add_argument(
        "--strict-external-guardrails",
        action="store_true",
        default=env_flag_enabled("STRICT_EXTERNAL_GUARDRAILS"),
        help=(
            "fail unless external confidential-inference guardrail inputs are "
            "supplied for provider catalog checks, Tinfoil directory results, "
            "PPQ selected model results, and Privatemode evidence"
        ),
    )
    parser.add_argument(
        "--attestation-targets-json",
        type=Path,
        default=env_path("ATTESTATION_TARGETS_JSON"),
        help=(
            "optional confidential-inference attestation-targets.json file used "
            "to map attestation result check IDs to Tinfoil router/model hosts"
        ),
    )
    parser.add_argument(
        "--attestation-results-json",
        type=Path,
        default=env_path("ATTESTATION_RESULTS_JSON"),
        help=(
            "optional confidential-inference attestation-results.json file used "
            "to reject deployed Tinfoil routes whose attested routing-policy "
            "hosts are missing, failed, or unreachable in the latest external run"
        ),
    )
    parser.add_argument(
        "--max-attestation-result-age-seconds",
        type=int,
        help=(
            "maximum accepted age for verified confidential-inference result rows; "
            "stale or timestamp-less rows are downgraded before guardrail checks"
        ),
    )
    args = parser.parse_args(argv)
    if args.expect_provider is None:
        args.expect_provider = env_expected_providers()
    if args.expect_inference is None:
        args.expect_inference = env_expected_inference_routes()
    max_age_env = os.getenv("MAX_ATTESTATION_RESULT_AGE_SECONDS")
    if (
        args.max_attestation_result_age_seconds is None
        and max_age_env is not None
        and max_age_env.strip()
    ):
        try:
            args.max_attestation_result_age_seconds = int(max_age_env)
        except ValueError:
            parser.error(
                "MAX_ATTESTATION_RESULT_AGE_SECONDS must be an integer when set"
            )
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.expect_provider and not args.expect_inference:
        raise SystemExit(
            "--expect-provider or --expect-inference is required for live validation"
        )

    status = fetch_json(
        args.base_url, "/v1/confidentiality/status", timeout=args.timeout
    )
    attestation = fetch_json(
        args.base_url,
        "/.well-known/routstr-attestation",
        timeout=args.timeout,
    )
    models = fetch_json(args.base_url, "/v1/models", timeout=args.timeout)
    hpke_key_config, hpke_content_type = fetch_bytes(
        args.base_url,
        "/.well-known/hpke-keys",
        timeout=args.timeout,
    )
    public_surfaces = public_surface_report(
        status=status,
        attestation=attestation,
        models=models,
        hpke_key_config=hpke_key_config,
        hpke_content_type=hpke_content_type,
    )
    status_routable_with_full_attestation = public_routable_with_full_attestation_map(
        status.get("routable_with_full_attestation")
    )
    inference_routes = complete_inference_routes(
        args.expect_provider,
        args.expect_inference,
        models=models if args.run_inference else None,
        routable_with_full_attestation=status_routable_with_full_attestation,
    )
    expected_pairs = expected_pairs_from_routes(args.expect_provider, inference_routes)
    provider_model_catalog = (
        load_provider_model_catalog(args.provider_catalog_json)
        if args.provider_catalog_json is not None
        else None
    )
    attestation_targets_directory = (
        load_attestation_targets_directory(args.attestation_targets_json)
        if args.attestation_targets_json is not None
        else None
    )
    attestation_results_directory = (
        load_attestation_results_directory(
            args.attestation_results_json,
            attestation_targets_directory=attestation_targets_directory,
            max_result_age_seconds=args.max_attestation_result_age_seconds,
        )
        if args.attestation_results_json is not None
        else None
    )

    results = collect_results(
        status=status,
        attestation=attestation,
        models=models,
        hpke_key_config=hpke_key_config,
        hpke_content_type=hpke_content_type,
        expected=expected_pairs,
        inference_routes=inference_routes
        if (args.expect_inference or args.run_inference)
        else None,
        provider_model_catalog=provider_model_catalog,
        attestation_results_directory=attestation_results_directory,
    )
    if args.strict_external_guardrails:
        results.extend(
            strict_external_guardrail_checks(
                attestation=attestation,
                expected=expected_pairs,
                provider_model_catalog=provider_model_catalog,
                attestation_targets_directory=attestation_targets_directory,
                attestation_results_directory=attestation_results_directory,
            )
        )
    needs_tinfoil_directory = attestation_has_tinfoil_provider(attestation) or any(
        expected_provider == "tinfoil" for expected_provider, _model in expected_pairs
    )
    needs_ppq_results = attestation_has_ppq_private_provider(attestation) or any(
        expected_provider_needs_ppq_results(expected_provider)
        for expected_provider, _model in expected_pairs
    )
    external_guardrails = external_guardrail_report(
        strict_external_guardrails=args.strict_external_guardrails,
        needs_tinfoil_directory=needs_tinfoil_directory,
        needs_ppq_results=needs_ppq_results,
        attestation=attestation,
        provider_model_catalog=provider_model_catalog,
        attestation_targets_directory=attestation_targets_directory,
        attestation_results_directory=attestation_results_directory,
        expected_tinfoil_models=expected_tinfoil_models(expected_pairs),
        expected_ppq_models=expected_ppq_private_models(expected_pairs),
        expected_privatemode_models=expected_privatemode_models(expected_pairs),
        provider_catalog_json=args.provider_catalog_json,
        attestation_targets_json=args.attestation_targets_json,
        attestation_results_json=args.attestation_results_json,
        max_attestation_result_age_seconds=args.max_attestation_result_age_seconds,
    )
    if args.strict_external_guardrails:
        results.extend(
            validate_external_guardrail_routable_with_full_attestation_matches_status(
                status_routable_with_full_attestation=(
                    status_routable_with_full_attestation
                ),
                external_guardrails=external_guardrails,
            )
        )
    inference_exercised = False
    inference_attempt_count = 0
    attempted_inference_routes: list[ExpectedInferenceRoute] = []
    if args.run_inference:
        gated_results = gate_inference_checks(results)
        if gated_results:
            results.extend(gated_results)
        else:
            try:
                auth_headers = inference_auth_headers(
                    bearer_token_env=args.bearer_token_env,
                    cashu_token_env=args.cashu_token_env,
                )
            except ValueError as exc:
                results.append(CheckResult(False, str(exc)))
            else:
                audio_file = (
                    load_audio_fixture(args.inference_audio_file)
                    if any(
                        route.endpoint
                        in {"/v1/audio/transcriptions", "/v1/audio/translations"}
                        for route in inference_routes
                    )
                    else None
                )
                attempted_inference_routes = list(inference_routes)
                results.extend(
                    run_inference_checks(
                        base_url=args.base_url,
                        routes=inference_routes,
                        auth_headers=auth_headers,
                        prompt=args.inference_prompt,
                        timeout=args.timeout,
                        audio_file=audio_file,
                    )
                )
                inference_attempt_count = len(attempted_inference_routes)
                inference_exercised = inference_attempt_count == len(
                    inference_routes
                ) and inference_attempt_count > 0
    if args.strict_inference_exercised and (
        not inference_exercised or inference_attempt_count != len(inference_routes)
    ):
        results.append(
            CheckResult(
                False,
                "confidential inference check did not exercise every expected route; pass --run-inference with valid auth after public proof passes",
            )
        )
    expected_inference_route_reports = [
        inference_route_report(route) for route in inference_routes
    ]
    attempted_inference_route_reports = [
        inference_route_report(route) for route in attempted_inference_routes
    ]
    confidential_routes_ready = confidential_routes_ready_value(
        end_to_end_ready=all(result.ok for result in results),
        external_guardrails_ready=external_guardrails.get("external_guardrails_ready")
        is True,
        inference_exercised=inference_exercised,
        expected_inference_route_count=len(inference_routes),
        inference_attempt_count=inference_attempt_count,
        expected_inference_routes=expected_inference_route_reports,
        attempted_inference_routes=attempted_inference_route_reports,
    )
    missing_strict_external_guardrails = (
        args.strict_confidential_routes_ready and not args.strict_external_guardrails
    )
    if missing_strict_external_guardrails:
        results.append(
            CheckResult(
                False,
                "strict confidential route readiness requires --strict-external-guardrails",
            )
        )
        confidential_routes_ready = False
    if (
        args.strict_confidential_routes_ready
        and not confidential_routes_ready
        and not missing_strict_external_guardrails
    ):
        results.append(
            CheckResult(
                False,
                "confidential routes were not fully proven; require public proof, external guardrails, and exercised inference for every expected route",
            )
        )
    if args.json:
        report = json_report(
            results,
            inference_exercised=inference_exercised,
            expected_inference_route_count=len(inference_routes),
            inference_attempt_count=inference_attempt_count,
            expected_inference_routes=expected_inference_route_reports,
            attempted_inference_routes=attempted_inference_route_reports,
            external_guardrails=external_guardrails,
            public_surfaces=public_surfaces,
        )
        try:
            payload = json.dumps(
                report,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        except ValueError as exc:
            print(
                f"error: failed to serialize live-check JSON report: {exc}",
                file=sys.stderr,
            )
            return 1
        print(payload)
    else:
        print_results(results)
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
