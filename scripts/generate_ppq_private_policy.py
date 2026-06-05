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
DEFAULT_BASE_URL = "https://api.ppq.ai/private/v1"
DEFAULT_ATTESTATION_BUNDLE_URL = "https://api.ppq.ai/private"
DEFAULT_ROUTER_REPO = "tinfoilsh/confidential-model-router"
DEFAULT_MAX_ATTESTATION_RESULT_AGE_SECONDS = 86_400
REQUIRED_TRUE_BACKEND_FIELDS = (
    "backend_tls_matches",
    "backend_sigstore_match",
    "backend_sigstore_bundle_verified",
    "backend_images_verified",
)
KNOWN_BACKEND_HOST_REPOS = {
    "deepseek-v4-pro-inf12.tinfoil.containers.tinfoil.dev": (
        "tinfoilsh/confidential-deepseek-v4-pro"
    ),
    "gemma4-31b-1.inf10.tinfoil.sh": "tinfoilsh/confidential-gemma4-31b",
    "glm-5-1.tinfoil.containers.tinfoil.dev": "tinfoilsh/confidential-glm5-1",
    "gpt-oss-120b-1.inf10.tinfoil.sh": "tinfoilsh/confidential-gpt-oss-120b",
    "kimi-k2-6.inf13.tinfoil.sh": "tinfoilsh/confidential-kimi-k2-6-b200",
    "llama3-3-70b.tinfoil.containers.tinfoil.dev": (
        "tinfoilsh/confidential-llama3-3-70b"
    ),
    "qwen3-vl-30b.inf10.tinfoil.sh": "tinfoilsh/confidential-qwen3-vl-30b",
}


@dataclass(frozen=True)
class VerifiedPpqModel:
    model_id: str
    router_release_digest: str
    backend_host: str
    backend_release_digest: str
    backend_repo: str


@dataclass(frozen=True)
class PpqAttestationTarget:
    model_id: str
    router_repo: str = ""
    backend_host: str = ""
    backend_repo: str = ""


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
        if not model_id.startswith("private/"):
            raise PolicyGenerationError("PPQ private model_ids must start with private/")
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


def ppq_catalog_identity_candidate_variants(value: object) -> set[str]:
    candidate = normalized_model_identity(value)
    if candidate is None:
        return set()
    candidates = {candidate}
    for suffix in ("-enclave", "-model", "-a3b", "-a10b", "-a17b", "-a22b"):
        if candidate.endswith(suffix):
            stripped = candidate[: -len(suffix)].strip("-")
            if stripped:
                candidates.add(stripped)
    family_number = re.match(r"^([a-z]+)-([0-9].*)$", candidate)
    if family_number:
        candidates.add(f"{family_number.group(1)}{family_number.group(2)}")
    compact_family_number = re.match(r"^([a-z]+)([0-9].*)$", candidate)
    if compact_family_number:
        candidates.add(
            f"{compact_family_number.group(1)}-{compact_family_number.group(2)}"
        )
    return candidates


def load_ppq_provider_catalog_models(path: Path) -> set[str]:
    document = load_json_file(path)
    providers = document.get("providers") if isinstance(document, dict) else document
    if not isinstance(providers, list):
        raise PolicyGenerationError("provider catalog JSON providers must be a list")
    models: set[str] = set()
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        if str(provider.get("slug", "")).strip().lower() != "ppq":
            continue
        provider_models = provider.get("models")
        if not isinstance(provider_models, list):
            continue
        for model in provider_models:
            if not isinstance(model, dict):
                continue
            for key in ("slug", "id", "name"):
                for candidate in ppq_catalog_identity_candidate_variants(model.get(key)):
                    models.add(candidate.split("/")[-1])
    return models


def require_selected_models_in_provider_catalog(
    selected_models: list[str],
    catalog_models: set[str],
) -> None:
    if not catalog_models:
        raise PolicyGenerationError(
            "provider catalog JSON does not include PPQ models"
        )
    for model_id in selected_models:
        selected = ppq_catalog_identity_candidate_variants(model_id.split("/")[-1])
        catalog_candidates = {
            candidate
            for catalog_model in catalog_models
            for candidate in ppq_catalog_identity_candidate_variants(
                catalog_model.split("/")[-1]
            )
        }
        if selected.intersection(catalog_candidates):
            continue
        raise PolicyGenerationError(
            f"selected PPQ model {model_id} is not present in provider catalog"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PolicyGenerationError(
            f"failed to hash proxy binary {path}: {type(exc).__name__}: {exc}"
        ) from exc
    return f"sha256:{digest.hexdigest()}"


def parse_model_repo_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise PolicyGenerationError(
                "--backend-repo values must be formatted as HOST=owner/repo"
            )
        host, repo = value.split("=", 1)
        host = host.strip().lower()
        repo = repo.strip()
        if not host or not repo:
            raise PolicyGenerationError(
                "--backend-repo values must be formatted as HOST=owner/repo"
            )
        overrides[host] = repo
    return overrides


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


def absolute_https_origin(value: str, *, label: str) -> str:
    url = absolute_https_url(value, label=label)
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").rstrip(".").lower()
    port = parsed.port
    if port is None or port == 443:
        return f"https://{hostname}"
    return f"https://{hostname}:{port}"


def ppq_private_url(value: object, *, label: str) -> str:
    url = absolute_https_url(value, label=label)
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").rstrip(".").lower()
    if hostname != "ppq.ai" and not hostname.endswith(".ppq.ai"):
        raise PolicyGenerationError(f"{label} host must be ppq.ai or a ppq.ai subdomain")
    path_segments = {segment.lower() for segment in parsed.path.split("/") if segment}
    if "private" not in path_segments:
        raise PolicyGenerationError(f"{label} must include a private path segment")
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


def non_empty_alias_value(
    data: dict[str, Any],
    *keys: str,
    label: str,
    lowercase: bool = False,
) -> str:
    selected = ""
    for key in keys:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        normalized = value.strip()
        if lowercase:
            normalized = normalized.lower()
        if selected and normalized != selected:
            raise PolicyGenerationError(f"{label} aliases must match")
        selected = normalized
    return selected


def ppq_targets_by_check_id(
    attestation_targets: dict[str, Any],
) -> dict[str, PpqAttestationTarget]:
    providers = attestation_targets.get("providers")
    if not isinstance(providers, list):
        raise PolicyGenerationError("attestation targets JSON must contain providers[]")
    targets_by_check_id: dict[str, PpqAttestationTarget] = {}
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        if str(provider.get("slug", "")).strip().lower() != "ppq":
            continue
        checks = provider.get("checks")
        if not isinstance(checks, list):
            continue
        for check in checks:
            if not isinstance(check, dict):
                continue
            check_id = str(check.get("id", "")).strip().lower()
            model_id = str(check.get("model", "")).strip().lower()
            if not check_id or not model_id:
                continue
            target = PpqAttestationTarget(
                model_id=model_id,
                router_repo=non_empty_alias_value(
                    check,
                    "repo",
                    "router_repo",
                    "routerRepo",
                    label="repo",
                ),
                backend_host=non_empty_alias_value(
                    check,
                    "backend_host",
                    "backendHost",
                    label="backend_host",
                    lowercase=True,
                ),
                backend_repo=non_empty_alias_value(
                    check,
                    "backend_repo",
                    "backendRepo",
                    label="backend_repo",
                ),
            )
            existing = targets_by_check_id.get(check_id)
            if existing and existing != target:
                raise PolicyGenerationError(
                    f"PPQ attestation target {check_id} has conflicting entries"
                )
            targets_by_check_id[check_id] = target
    return targets_by_check_id


def ppq_targets_by_model(
    targets_by_check_id: dict[str, PpqAttestationTarget],
) -> dict[str, PpqAttestationTarget]:
    targets_by_model: dict[str, PpqAttestationTarget] = {}
    for check_id, target in targets_by_check_id.items():
        existing = targets_by_model.get(target.model_id)
        if existing and existing != target:
            raise PolicyGenerationError(
                f"PPQ model {target.model_id} has conflicting attestation targets "
                f"including {check_id}"
            )
        targets_by_model[target.model_id] = target
    return targets_by_model


def require_selected_model_targets(
    selected_models: list[str],
    targets_by_model: dict[str, PpqAttestationTarget],
) -> None:
    missing = sorted(
        model_id for model_id in selected_models if model_id not in targets_by_model
    )
    if missing:
        raise PolicyGenerationError(
            "PPQ attestation targets are required for selected models: "
            + ", ".join(missing)
        )


def require_router_repo_matches_targets(
    router_repo: str,
    targets: list[PpqAttestationTarget | None],
) -> None:
    expected_repos = sorted(
        {target.router_repo for target in targets if target and target.router_repo}
    )
    if len(expected_repos) > 1:
        raise PolicyGenerationError(
            "PPQ attestation targets have conflicting router repos: "
            + ", ".join(expected_repos)
        )
    if expected_repos and router_repo != expected_repos[0]:
        raise PolicyGenerationError("PPQ router_repo does not match attestation target")


def ppq_check_model_id(
    raw_check: dict[str, Any],
    targets_by_check_id: dict[str, PpqAttestationTarget],
) -> str:
    model_id = str(raw_check.get("model", "")).strip().lower()
    check_id = str(raw_check.get("id", "")).strip().lower()
    target = targets_by_check_id.get(check_id) if check_id else None
    if model_id and target and model_id != target.model_id:
        raise PolicyGenerationError(
            f"PPQ attestation result {check_id} model {model_id} does not match "
            f"attestation target model {target.model_id}"
        )
    if model_id:
        return model_id
    if target is None:
        return ""
    return target.model_id


def selected_ppq_checks(
    attestation_results: dict[str, Any],
    selected_models: list[str],
    targets_by_check_id: dict[str, PpqAttestationTarget] | None = None,
) -> dict[str, dict[str, Any]]:
    checks = attestation_results.get("checks")
    if not isinstance(checks, list):
        raise PolicyGenerationError("attestation results JSON must contain checks[]")
    selected = {model.strip().lower() for model in selected_models if model.strip()}
    targets_by_check_id = targets_by_check_id or {}
    rows: dict[str, dict[str, Any]] = {}
    for raw_check in checks:
        if not isinstance(raw_check, dict):
            continue
        if str(raw_check.get("provider", "")).strip().lower() != "ppq":
            continue
        model_id = ppq_check_model_id(raw_check, targets_by_check_id)
        if model_id not in selected:
            continue
        if model_id in rows:
            raise PolicyGenerationError(
                f"duplicate PPQ result rows for selected model {model_id}"
            )
        rows[model_id] = raw_check
    missing = sorted(selected - set(rows))
    if missing:
        raise PolicyGenerationError(
            "selected PPQ models are missing from attestation results: "
            + ", ".join(missing)
        )
    return rows


def verified_model_from_check(
    *,
    model_id: str,
    check: dict[str, Any],
    backend_repo_overrides: dict[str, str],
    attestation_target: PpqAttestationTarget | None = None,
) -> VerifiedPpqModel:
    status = str(check.get("status", "")).strip().lower()
    if status != "verified":
        error = str(check.get("error", "")).strip()
        suffix = f": {error}" if error else ""
        raise PolicyGenerationError(
            f"selected PPQ model {model_id} is not verified (status={status}){suffix}"
        )
    for field in REQUIRED_TRUE_BACKEND_FIELDS:
        if check.get(field) is not True:
            raise PolicyGenerationError(
                f"selected PPQ model {model_id} requires {field}=true"
            )
    backend_host = non_empty_alias_value(
        check,
        "backend_host",
        "backendHost",
        label="backend_host",
        lowercase=True,
    )
    if not backend_host:
        raise PolicyGenerationError(
            f"selected PPQ model {model_id} is missing backend_host"
        )
    if attestation_target and attestation_target.backend_host:
        if backend_host != attestation_target.backend_host:
            raise PolicyGenerationError(
                f"selected PPQ model {model_id} backend_host does not match "
                "attestation target"
            )
    backend_repo = backend_repo_overrides.get(backend_host) or KNOWN_BACKEND_HOST_REPOS.get(
        backend_host
    )
    if not backend_repo:
        raise PolicyGenerationError(
            f"selected PPQ model {model_id} backend host {backend_host} has no "
            "known repo; pass --backend-repo HOST=owner/repo"
        )
    if attestation_target and attestation_target.backend_repo:
        if backend_repo != attestation_target.backend_repo:
            raise PolicyGenerationError(
                f"selected PPQ model {model_id} backend_repo does not match "
                "attestation target"
            )
    return VerifiedPpqModel(
        model_id=model_id,
        router_release_digest=normalize_sha256_digest(
            check.get("release_digest"),
            label=f"{model_id} release_digest",
        ),
        backend_host=backend_host,
        backend_release_digest=normalize_sha256_digest(
            check.get("backend_release_digest"),
            label=f"{model_id} backend_release_digest",
        ),
        backend_repo=backend_repo,
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
        and "ppq-private-tee" in [str(mode) for mode in entry.get("modes", [])]
    ]
    if len(matches) != 1:
        raise PolicyGenerationError(
            "verifier manifest must contain exactly one ppq-private-tee verifier"
        )
    return normalize_sha256_digest(
        matches[0].get("sha256"),
        label="ppq-private-tee verifier sha256",
    )


def build_policy_document(
    *,
    models: list[VerifiedPpqModel],
    verifier_command_digest: str,
    verifier_command: str,
    base_url: str,
    attestation_bundle_url: str,
    router_repo: str,
) -> dict[str, Any]:
    router_digests = sorted({model.router_release_digest for model in models})
    policy: dict[str, Any] = {
        "attestation_bundle_url": attestation_bundle_url,
        "repo": router_repo,
        "require_model_attestations": True,
        "model_attestation_targets": {
            model.model_id: {
                "host": model.backend_host,
                "repo": model.backend_repo,
                "expected_release_digest": model.backend_release_digest,
            }
            for model in sorted(models, key=lambda item: item.model_id)
        },
        "verifier_command": verifier_command,
        "verifier_command_digest": verifier_command_digest,
    }
    if len(router_digests) == 1:
        policy["expected_release_digest"] = router_digests[0]
    else:
        policy["allowed_release_digests"] = router_digests
    return {
        "base_url": base_url,
        "confidentiality": {
            "enabled": True,
            "mode": "ppq-private-tee",
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
            load_ppq_provider_catalog_models(args.provider_catalog_json),
        )
    base_url = ppq_private_url(args.base_url, label="base_url")
    attestation_bundle_url = ppq_private_url(
        args.attestation_bundle_url,
        label="attestation_bundle_url",
    )
    if absolute_https_origin(
        attestation_bundle_url,
        label="attestation_bundle_url",
    ) != absolute_https_origin(base_url, label="base_url"):
        raise PolicyGenerationError(
            "attestation_bundle_url origin must match base_url origin"
        )
    verifier_command_digest = (
        normalize_sha256_digest(
            args.verifier_command_digest,
            label="verifier_command_digest",
        )
        if args.verifier_command_digest
        else verifier_digest_from_manifest(args.verifier_manifest_json)
    )
    backend_repo_overrides = parse_model_repo_overrides(args.backend_repo)
    attestation_results = load_json_file(args.attestation_results_json)
    if not isinstance(attestation_results, dict):
        raise PolicyGenerationError("attestation results JSON must be an object")
    document_timestamp = ensure_fresh_attestation_results(
        attestation_results,
        max_age_seconds=args.max_attestation_result_age_seconds,
    )
    if not args.attestation_targets_json:
        raise PolicyGenerationError(
            "PPQ attestation targets are required for selected models: "
            + ", ".join(selected_models)
        )
    targets_by_check_id = ppq_targets_by_check_id(
        load_json_file(args.attestation_targets_json)
    )
    targets_by_model = ppq_targets_by_model(targets_by_check_id)
    require_selected_model_targets(selected_models, targets_by_model)
    rows = selected_ppq_checks(
        attestation_results,
        selected_models,
        targets_by_check_id=targets_by_check_id,
    )
    models = [
        verified_model_from_check(
            model_id=model_id,
            check=rows[model_id],
            backend_repo_overrides=backend_repo_overrides,
            attestation_target=targets_by_model.get(model_id),
        )
        for model_id in sorted(rows)
    ]
    router_repo = non_empty_string(args.router_repo, label="router_repo")
    require_router_repo_matches_targets(
        router_repo,
        [targets_by_model.get(model_id) for model_id in sorted(rows)],
    )
    for model_id, check in sorted(rows.items()):
        ensure_fresh_attestation_check(
            check,
            fallback_timestamp=document_timestamp,
            label=f"selected PPQ model {model_id}",
            max_age_seconds=args.max_attestation_result_age_seconds,
        )
    document = build_policy_document(
        models=models,
        verifier_command_digest=verifier_command_digest,
        verifier_command=normalize_command(
            args.verifier_command,
            label="verifier_command",
        ),
        base_url=base_url,
        attestation_bundle_url=attestation_bundle_url,
        router_repo=router_repo,
    )
    try:
        return validate_runtime_policy_document(document, provider_type="ppq-private")
    except RuntimePolicyValidationError as exc:
        raise PolicyGenerationError(str(exc)) from exc


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a fail-closed PPQ private Routstr policy from verified "
            "confidential-inference attestation results."
        )
    )
    parser.add_argument("--attestation-results-json", type=Path, required=True)
    parser.add_argument("--attestation-targets-json", type=Path)
    parser.add_argument("--provider-catalog-json", type=Path)
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--proxy-binary-digest")
    parser.add_argument("--proxy-binary-path", type=Path)
    parser.add_argument("--verifier-command-digest")
    parser.add_argument(
        "--verifier-manifest-json",
        type=Path,
        default=DEFAULT_VERIFIER_MANIFEST,
    )
    parser.add_argument("--verifier-command", default=DEFAULT_VERIFIER_COMMAND)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--attestation-bundle-url", default=DEFAULT_ATTESTATION_BUNDLE_URL)
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
        "--backend-repo",
        action="append",
        default=[],
        help="backend host to repo mapping, formatted as HOST=owner/repo",
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
