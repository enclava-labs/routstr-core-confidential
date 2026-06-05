#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from confidential_policy_runtime_validation import (  # noqa: E402
    RuntimePolicyValidationError,
    validate_runtime_policy_document,
)

DEFAULT_VERIFIER_MANIFEST = (
    ROOT / "build" / "confidential-verifiers" / "confidential-verifier-manifest.json"
)
DEFAULT_VERIFIER_COMMAND = "build/confidential-verifiers/routstr-tinfoil-go-verifier"
DEFAULT_BASE_URL = "https://inference.tinfoil.sh/v1"
DEFAULT_ROUTER_REPO = "tinfoilsh/confidential-model-router"
DEFAULT_MAX_ATTESTATION_RESULT_AGE_SECONDS = 86_400
DEFAULT_ROUTER_HOST = "inference.tinfoil.sh"
REQUIRED_TRUE_MODEL_FIELDS = (
    "sigstore_match",
    "sigstore_bundle_verified",
    "images_verified",
)
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
    "qwen3-vl-30b.inf10.tinfoil.sh": {
        "tinfoil/qwen3-vl-30b",
        "tinfoil/qwen3-vl-30b-a3b",
    },
    "glm-5-1.tinfoil.containers.tinfoil.dev": {"tinfoil/glm-5-1"},
    "gpt-oss-120b-1.inf10.tinfoil.sh": {"tinfoil/gpt-oss-120b"},
    "gpt-oss-safeguard-120b.tinfoil.containers.tinfoil.dev": {
        "tinfoil/gpt-oss-safeguard-120b",
    },
}
KNOWN_TINFOIL_MODEL_HOST_REPOS = {
    "llama3-3-70b.tinfoil.containers.tinfoil.dev": "tinfoilsh/confidential-llama3-3-70b",
    "deepseek-v4-pro-inf12.tinfoil.containers.tinfoil.dev": "tinfoilsh/confidential-deepseek-v4-pro",
    "kimi-k2-6.inf13.tinfoil.sh": "tinfoilsh/confidential-kimi-k2-6-b200",
    "gemma4-31b-1.inf10.tinfoil.sh": "tinfoilsh/confidential-gemma4-31b",
    "qwen3-vl-30b.inf10.tinfoil.sh": "tinfoilsh/confidential-qwen3-vl-30b",
    "glm-5-1.tinfoil.containers.tinfoil.dev": "tinfoilsh/confidential-glm5-1",
    "gpt-oss-120b-1.inf10.tinfoil.sh": "tinfoilsh/confidential-gpt-oss-120b",
    "gpt-oss-safeguard-120b.tinfoil.containers.tinfoil.dev": "tinfoilsh/confidential-gpt-oss-safeguard-120b",
}


@dataclass(frozen=True)
class TinfoilTarget:
    check_id: str
    model_id: str
    host: str
    repo: str
    attestation_url: str


@dataclass(frozen=True)
class VerifiedTinfoilModel:
    model_id: str
    host: str
    repo: str
    attestation_url: str
    release_digest: str


class PolicyGenerationError(ValueError):
    pass


def load_json_file(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=json_constant_rejecter("policy generator JSON"),
        )
    except Exception as exc:
        raise PolicyGenerationError(
            f"failed to read JSON from {path}: {type(exc).__name__}: {exc}"
        ) from exc


def json_constant_rejecter(label: str):
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} must not contain {value}")

    return reject_constant


def rfc3339_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def ensure_fresh_attestation_results(
    document: dict[str, Any],
    *,
    max_age_seconds: int,
) -> datetime:
    if max_age_seconds < 0:
        raise PolicyGenerationError("max attestation result age must be non-negative")
    document_timestamp = rfc3339_timestamp(document.get("last_run"))
    if document_timestamp is None:
        raise PolicyGenerationError("attestation results last_run timestamp is required")
    age_seconds = (datetime.now(UTC) - document_timestamp).total_seconds()
    if age_seconds < 0:
        raise PolicyGenerationError("attestation results timestamp is in the future")
    if age_seconds > max_age_seconds:
        raise PolicyGenerationError(
            "attestation results are stale: "
            f"age_seconds={int(age_seconds)} max_age_seconds={max_age_seconds}"
        )
    return document_timestamp


def ensure_fresh_attestation_check(
    check: dict[str, Any],
    *,
    fallback_timestamp: datetime,
    label: str,
    max_age_seconds: int,
) -> None:
    raw_timestamp = check.get("ts")
    timestamp = rfc3339_timestamp(raw_timestamp)
    if raw_timestamp is not None and timestamp is None:
        raise PolicyGenerationError(f"{label} attestation result timestamp is invalid")
    timestamp = timestamp or fallback_timestamp
    age_seconds = (datetime.now(UTC) - timestamp).total_seconds()
    if age_seconds < 0:
        raise PolicyGenerationError(f"{label} attestation result timestamp is in the future")
    if age_seconds > max_age_seconds:
        raise PolicyGenerationError(
            f"{label} attestation result is stale: "
            f"age_seconds={int(age_seconds)} max_age_seconds={max_age_seconds}"
        )


def is_sha256_digest(value: object) -> bool:
    if not isinstance(value, str):
        return False
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def is_template_placeholder_digest(value: object) -> bool:
    if not isinstance(value, str):
        return False
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    return len(digest) == 64 and len(set(digest)) == 1


def normalize_sha256_digest(value: object, *, label: str) -> str:
    if not is_sha256_digest(value):
        raise PolicyGenerationError(f"{label} must be a sha256 digest")
    if is_template_placeholder_digest(value):
        raise PolicyGenerationError(f"{label} must not be a placeholder digest")
    assert isinstance(value, str)
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        return digest
    return f"sha256:{digest}"


def normalize_model_ids(values: list[str]) -> list[str]:
    model_ids: list[str] = []
    seen: set[str] = set()
    for value in values:
        model_id = value.strip().lower()
        if not model_id:
            raise PolicyGenerationError("--model-id values must be non-empty")
        if not model_id.startswith("tinfoil/"):
            raise PolicyGenerationError(
                "Tinfoil selected models must use exact tinfoil/* model_ids"
            )
        if model_id in seen:
            raise PolicyGenerationError(f"duplicate --model-id value {model_id}")
        seen.add(model_id)
        model_ids.append(model_id)
    return model_ids


def normalized_model_identity(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or None


def provider_catalog_identity_candidate_variants(value: object) -> set[str]:
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


def load_tinfoil_provider_catalog_models(path: Path) -> set[str]:
    document = load_json_file(path)
    providers = document.get("providers") if isinstance(document, dict) else document
    if not isinstance(providers, list):
        raise PolicyGenerationError("provider catalog JSON providers must be a list")
    models: set[str] = set()
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        if str(provider.get("slug", "")).strip().lower() != "tinfoil":
            continue
        provider_models = provider.get("models")
        if not isinstance(provider_models, list):
            continue
        for model in provider_models:
            if not isinstance(model, dict):
                continue
            for key in ("slug", "id", "name"):
                for candidate in provider_catalog_identity_candidate_variants(
                    model.get(key)
                ):
                    models.add(candidate.split("/")[-1])
    return models


def require_selected_models_in_provider_catalog(
    selected_models: list[str],
    catalog_models: set[str],
) -> None:
    if not catalog_models:
        raise PolicyGenerationError(
            "provider catalog JSON does not include Tinfoil models"
        )
    catalog_candidates = {
        candidate
        for catalog_model in catalog_models
        for candidate in provider_catalog_identity_candidate_variants(
            catalog_model.split("/")[-1]
        )
    }
    for model_id in selected_models:
        selected = provider_catalog_identity_candidate_variants(model_id.split("/")[-1])
        if selected.intersection(catalog_candidates):
            continue
        raise PolicyGenerationError(
            f"selected Tinfoil model {model_id} is not present in provider catalog"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def https_host(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolicyGenerationError(f"{label} is required")
    parsed = urlparse(value.strip())
    if parsed.scheme != "https" or not parsed.hostname:
        raise PolicyGenerationError(f"{label} must be an absolute HTTPS URL")
    if parsed.username or parsed.password or "@" in parsed.netloc:
        raise PolicyGenerationError(f"{label} must not contain credentials")
    try:
        parsed.port
    except ValueError:
        raise PolicyGenerationError(f"{label} must include a valid port") from None
    return parsed.hostname.rstrip(".").lower()


def absolute_https_url(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolicyGenerationError(f"{label} must be an absolute HTTPS URL")
    url = value.strip()
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise PolicyGenerationError(f"{label} must be an absolute HTTPS URL")
    if parsed.username or parsed.password or "@" in parsed.netloc:
        raise PolicyGenerationError(f"{label} must not contain credentials")
    if not parsed.hostname:
        raise PolicyGenerationError(f"{label} must include a host")
    try:
        parsed.port
    except ValueError:
        raise PolicyGenerationError(f"{label} must include a valid port") from None
    return url


def tinfoil_router_url(value: object, *, label: str) -> str:
    url = absolute_https_url(value, label=label)
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").rstrip(".").lower()
    if hostname != DEFAULT_ROUTER_HOST:
        raise PolicyGenerationError(f"{label} host must be {DEFAULT_ROUTER_HOST}")
    return url


def normalize_command(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise PolicyGenerationError(f"{label} must be a command string")
    command = value.strip()
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise PolicyGenerationError(f"{label} is invalid: {exc}") from exc
    if not argv:
        raise PolicyGenerationError(f"{label} is empty")
    return command


def non_empty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolicyGenerationError(f"{label} must be a non-empty string")
    return value.strip()


def target_host(check: dict[str, Any]) -> tuple[str, str]:
    selected_host = ""
    attestation_url = ""
    raw_host = check.get("host")
    if isinstance(raw_host, str) and raw_host.strip():
        selected_host = raw_host.strip().rstrip(".").lower()
    for key in ("url", "attestation_url"):
        raw_url = check.get(key)
        if not isinstance(raw_url, str) or not raw_url.strip():
            continue
        url_host = https_host(raw_url, label=f"Tinfoil attestation target {key}")
        if selected_host and url_host != selected_host:
            raise PolicyGenerationError(
                "Tinfoil attestation target host aliases must match"
            )
        selected_host = url_host
        if attestation_url and raw_url.strip() != attestation_url:
            raise PolicyGenerationError(
                "Tinfoil attestation target URL aliases must match"
            )
        attestation_url = raw_url.strip()
    if not selected_host:
        raise PolicyGenerationError("Tinfoil attestation target host is required")
    return selected_host, attestation_url or (
        f"https://{selected_host}/.well-known/tinfoil-attestation"
    )


def model_from_target(
    check: dict[str, Any],
    host: str,
    *,
    selected_models: set[str],
) -> str | None:
    raw_model = check.get("model")
    known_models = KNOWN_TINFOIL_MODEL_HOST_MODELS.get(host)
    if isinstance(raw_model, str) and raw_model.strip():
        model_id = raw_model.strip().lower()
        if known_models and model_id not in known_models:
            raise PolicyGenerationError(
                f"Tinfoil target {host} has known host/model mismatch"
            )
        return model_id
    if known_models and len(known_models) == 1:
        return next(iter(known_models))
    if known_models:
        selected_known_models = sorted(known_models & selected_models)
        if len(selected_known_models) == 1:
            return selected_known_models[0]
    return None


def tinfoil_targets_by_check_id(
    attestation_targets: dict[str, Any],
    selected_models: set[str],
) -> tuple[dict[str, TinfoilTarget], set[str], dict[str, str]]:
    providers = attestation_targets.get("providers")
    if not isinstance(providers, list):
        raise PolicyGenerationError("attestation targets JSON must contain providers[]")
    targets: dict[str, TinfoilTarget] = {}
    router_check_ids: set[str] = set()
    router_repos: dict[str, str] = {}
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        if str(provider.get("slug", "")).strip().lower() != "tinfoil":
            continue
        checks = provider.get("checks")
        if not isinstance(checks, list):
            continue
        for check in checks:
            if not isinstance(check, dict):
                continue
            check_id = str(check.get("id", "")).strip().lower()
            if not check_id:
                continue
            label = str(check.get("label", "")).strip().lower()
            if "router" in check_id or "router" in label:
                router_check_ids.add(check_id)
                repo = str(check.get("repo", "")).strip()
                if repo:
                    router_repos[check_id] = repo
                continue
            host, attestation_url = target_host(check)
            model_id = model_from_target(
                check,
                host,
                selected_models=selected_models,
            )
            if model_id is None:
                continue
            repo = str(check.get("repo", "")).strip()
            known_repo = KNOWN_TINFOIL_MODEL_HOST_REPOS.get(host, "")
            if repo and known_repo and repo != known_repo:
                raise PolicyGenerationError(
                    f"Tinfoil target {check_id} has known host/repo mismatch"
                )
            repo = repo or known_repo
            if not repo:
                raise PolicyGenerationError(
                    f"Tinfoil target {check_id} host {host} has no known repo"
                )
            target = TinfoilTarget(
                check_id=check_id,
                model_id=model_id,
                host=host,
                repo=repo,
                attestation_url=attestation_url,
            )
            existing = targets.get(check_id)
            if existing and existing != target:
                raise PolicyGenerationError(
                    f"Tinfoil attestation target {check_id} has conflicting entries"
                )
            targets[check_id] = target
    return targets, router_check_ids, router_repos


def require_router_repo_matches_targets(
    router_repo: str,
    router_target_repos: dict[str, str],
) -> None:
    expected_repos = sorted(set(router_target_repos.values()))
    if len(expected_repos) > 1:
        raise PolicyGenerationError(
            "Tinfoil router attestation targets have conflicting repos: "
            + ", ".join(expected_repos)
        )
    if expected_repos and router_repo != expected_repos[0]:
        raise PolicyGenerationError(
            "Tinfoil router_repo does not match attestation target"
        )


def selected_tinfoil_checks(
    attestation_results: dict[str, Any],
    selected_models: list[str],
    *,
    targets_by_check_id: dict[str, TinfoilTarget],
    router_check_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    checks = attestation_results.get("checks")
    if not isinstance(checks, list):
        raise PolicyGenerationError("attestation results JSON must contain checks[]")
    selected = {model.strip().lower() for model in selected_models if model.strip()}
    rows: dict[str, dict[str, Any]] = {}
    router_row: dict[str, Any] | None = None
    for raw_check in checks:
        if not isinstance(raw_check, dict):
            continue
        if str(raw_check.get("provider", "")).strip().lower() != "tinfoil":
            continue
        check_id = str(raw_check.get("id", "")).strip().lower()
        if check_id in router_check_ids:
            if router_row is not None:
                raise PolicyGenerationError("duplicate Tinfoil router result rows")
            router_row = raw_check
            continue
        target = targets_by_check_id.get(check_id)
        raw_model = raw_check.get("model")
        model_id = (
            raw_model.strip().lower()
            if isinstance(raw_model, str) and raw_model.strip()
            else target.model_id
            if target
            else ""
        )
        if target and model_id != target.model_id:
            known_models = KNOWN_TINFOIL_MODEL_HOST_MODELS.get(target.host) or set()
            model_aliases = {model_id}
            if "/" not in model_id:
                model_aliases.add(f"tinfoil/{model_id}")
            if target.model_id in known_models and model_aliases & known_models:
                model_id = target.model_id
            else:
                raise PolicyGenerationError(
                    f"Tinfoil result {check_id} model {model_id} does not match target"
                )
        if model_id not in selected:
            continue
        if model_id in rows:
            raise PolicyGenerationError(
                f"duplicate Tinfoil result rows for selected model {model_id}"
            )
        rows[model_id] = raw_check
    missing = sorted(selected - set(rows))
    if missing:
        raise PolicyGenerationError(
            "selected Tinfoil models are missing from attestation results: "
            + ", ".join(missing)
        )
    if router_row is None:
        raise PolicyGenerationError("Tinfoil router is missing from attestation results")
    return rows, router_row


def require_verified_router(
    check: dict[str, Any],
    *,
    transport_security: str,
) -> str:
    status = str(check.get("status", "")).strip().lower()
    tls_binding_failure = (
        transport_security == "ehbp"
        and status == "failed"
        and check.get("tls_matches") is False
        and "tls certificate binding" in str(check.get("error", "")).strip().lower()
    )
    if status != "verified" and not tls_binding_failure:
        raise PolicyGenerationError(f"Tinfoil router is not verified (status={status})")
    if transport_security == "tls" and check.get("tls_matches") is not True:
        raise PolicyGenerationError("Tinfoil router requires tls_matches=true")
    return normalize_sha256_digest(
        check.get("release_digest"),
        label="Tinfoil router release_digest",
    )


def verified_model_from_check(
    *,
    model_id: str,
    check: dict[str, Any],
    target: TinfoilTarget,
    transport_security: str,
) -> VerifiedTinfoilModel:
    status = str(check.get("status", "")).strip().lower()
    if status != "verified":
        raise PolicyGenerationError(
            f"selected Tinfoil model {model_id} is not verified (status={status})"
        )
    if transport_security == "tls" and check.get("tls_matches") is not True:
        raise PolicyGenerationError(
            f"selected Tinfoil model {model_id} requires tls_matches=true"
        )
    for field in REQUIRED_TRUE_MODEL_FIELDS:
        if check.get(field) is not True:
            raise PolicyGenerationError(
                f"selected Tinfoil model {model_id} requires {field}=true"
            )
    return VerifiedTinfoilModel(
        model_id=model_id,
        host=target.host,
        repo=target.repo,
        attestation_url=target.attestation_url,
        release_digest=normalize_sha256_digest(
            check.get("release_digest"),
            label=f"{model_id} release_digest",
        ),
    )


def verifier_digest_from_manifest(path: Path) -> str:
    manifest = load_json_file(path)
    entries = manifest.get("verifiers") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        raise PolicyGenerationError("verifier manifest must contain verifiers[]")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and "tinfoil" in [str(mode) for mode in entry.get("modes", [])]
    ]
    if len(matches) != 1:
        raise PolicyGenerationError("verifier manifest must contain exactly one Tinfoil verifier")
    return normalize_sha256_digest(matches[0].get("sha256"), label="Tinfoil verifier sha256")


def build_policy_document(
    *,
    models: list[VerifiedTinfoilModel],
    router_release_digest: str,
    verifier_command_digest: str,
    verifier_command: str,
    base_url: str,
    router_repo: str,
    transport_security: str,
) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "repo": router_repo,
        "expected_release_digest": router_release_digest,
        "require_model_attestations": True,
        "model_attestation_targets": {
            model.model_id: {
                "attestation_url": model.attestation_url,
                "host": model.host,
                "repo": model.repo,
                "expected_release_digest": model.release_digest,
            }
            for model in sorted(models, key=lambda item: item.model_id)
        },
        "verifier_command": verifier_command,
        "verifier_command_digest": verifier_command_digest,
    }
    if transport_security == "ehbp":
        policy["transport_security"] = "ehbp"
    return {
        "base_url": base_url,
        "confidentiality": {
            "enabled": True,
            "mode": "tinfoil",
            "model_ids": [model.model_id for model in sorted(models, key=lambda item: item.model_id)],
            "policy": policy,
        },
    }


def generate_policy(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model_id:
        raise PolicyGenerationError("at least one --model-id is required")
    selected_models = normalize_model_ids(args.model_id)
    if args.provider_catalog_json:
        require_selected_models_in_provider_catalog(
            selected_models,
            load_tinfoil_provider_catalog_models(args.provider_catalog_json),
        )
    verifier_command_digest = (
        normalize_sha256_digest(
            args.verifier_command_digest,
            label="verifier_command_digest",
        )
        if args.verifier_command_digest
        else verifier_digest_from_manifest(args.verifier_manifest_json)
    )
    targets_by_check_id, router_check_ids, router_target_repos = tinfoil_targets_by_check_id(
        load_json_file(args.attestation_targets_json),
        set(selected_models),
    )
    attestation_results = load_json_file(args.attestation_results_json)
    if not isinstance(attestation_results, dict):
        raise PolicyGenerationError("attestation results JSON must be an object")
    document_timestamp = ensure_fresh_attestation_results(
        attestation_results,
        max_age_seconds=args.max_attestation_result_age_seconds,
    )
    rows, router_row = selected_tinfoil_checks(
        attestation_results,
        selected_models,
        targets_by_check_id=targets_by_check_id,
        router_check_ids=router_check_ids,
    )
    targets_by_model = {target.model_id: target for target in targets_by_check_id.values()}
    router_release_digest = require_verified_router(
        router_row,
        transport_security=args.transport_security,
    )
    router_repo = non_empty_string(args.router_repo, label="router_repo")
    require_router_repo_matches_targets(router_repo, router_target_repos)
    ensure_fresh_attestation_check(
        router_row,
        fallback_timestamp=document_timestamp,
        label="Tinfoil router",
        max_age_seconds=args.max_attestation_result_age_seconds,
    )
    models = [
        verified_model_from_check(
            model_id=model_id,
            check=rows[model_id],
            target=targets_by_model[model_id],
            transport_security=args.transport_security,
        )
        for model_id in sorted(rows)
    ]
    for model_id, check in sorted(rows.items()):
        ensure_fresh_attestation_check(
            check,
            fallback_timestamp=document_timestamp,
            label=f"selected Tinfoil model {model_id}",
            max_age_seconds=args.max_attestation_result_age_seconds,
        )
    document = build_policy_document(
        models=models,
        router_release_digest=router_release_digest,
        verifier_command_digest=verifier_command_digest,
        verifier_command=normalize_command(
            args.verifier_command,
            label="verifier_command",
        ),
        base_url=tinfoil_router_url(args.base_url, label="base_url"),
        router_repo=router_repo,
        transport_security=args.transport_security,
    )
    try:
        return validate_runtime_policy_document(document, provider_type="tinfoil")
    except RuntimePolicyValidationError as exc:
        raise PolicyGenerationError(str(exc)) from exc


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a fail-closed Tinfoil Routstr policy from verified "
            "confidential-inference attestation results."
        )
    )
    parser.add_argument("--attestation-results-json", type=Path, required=True)
    parser.add_argument("--attestation-targets-json", type=Path, required=True)
    parser.add_argument("--provider-catalog-json", type=Path)
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--verifier-command-digest")
    parser.add_argument(
        "--verifier-manifest-json",
        type=Path,
        default=DEFAULT_VERIFIER_MANIFEST,
    )
    parser.add_argument("--verifier-command", default=DEFAULT_VERIFIER_COMMAND)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--router-repo", default=DEFAULT_ROUTER_REPO)
    parser.add_argument(
        "--max-attestation-result-age-seconds",
        type=int,
        default=DEFAULT_MAX_ATTESTATION_RESULT_AGE_SECONDS,
        help=(
            "maximum accepted age for confidential-inference result evidence; "
            "defaults to 86400"
        ),
    )
    parser.add_argument(
        "--transport-security",
        choices=("tls", "ehbp"),
        default="tls",
        help="tls requires directory TLS binding rows; ehbp emits an EHBP-only policy",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        document = generate_policy(args)
    except PolicyGenerationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        payload = json.dumps(
            document,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
    except ValueError as exc:
        print(f"error: failed to serialize policy JSON: {exc}", file=sys.stderr)
        return 1
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
