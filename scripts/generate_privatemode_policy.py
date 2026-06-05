#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
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
DEFAULT_VERIFIER_COMMAND = "build/confidential-verifiers/routstr-privatemode-go-verifier"
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
VERIFIER_GPU_ATTESTATION_POLICY = "nvidia-ocsp-good-only"
VERIFIER_TRUST_TIER = "app-e2ee"


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


def non_empty_unique_strings(values: list[str], *, label: str) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        value = raw_value.strip()
        if not value:
            raise PolicyGenerationError(f"{label} values must be non-empty")
        key = value.lower()
        if key in seen:
            raise PolicyGenerationError(f"{label} contains duplicate value {value}")
        seen.add(key)
        normalized.append(value)
    return normalized


def normalize_model_ids(values: list[str]) -> list[str]:
    model_ids = non_empty_unique_strings(values, label="model_id")
    for model_id in model_ids:
        if not model_id.lower().startswith("privatemode/"):
            raise PolicyGenerationError(
                "Privatemode selected models must use exact privatemode/* model_ids"
            )
    return [model_id.lower() for model_id in model_ids]


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


def load_privatemode_provider_catalog_models(path: Path) -> set[str]:
    document = load_json_file(path)
    providers = document.get("providers") if isinstance(document, dict) else document
    if not isinstance(providers, list):
        raise PolicyGenerationError("provider catalog JSON providers must be a list")
    models: set[str] = set()
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        if str(provider.get("slug", "")).strip().lower() != "privatemode":
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
            "provider catalog JSON does not include Privatemode models"
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
            f"selected Privatemode model {model_id} is not present in provider catalog"
        )


def require_loopback_url(value: str, *, label: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise PolicyGenerationError(f"{label} must be an absolute loopback URL")
    if parsed.username or parsed.password or "@" in parsed.netloc:
        raise PolicyGenerationError(f"{label} must not contain URL userinfo")
    try:
        parsed.port
    except ValueError:
        raise PolicyGenerationError(f"{label} must include a valid port") from None
    hostname = parsed.hostname.lower()
    if hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise PolicyGenerationError(f"{label} must be a loopback URL")
    return value


def optional_https_url(value: str | None, *, label: str) -> str | None:
    if value is None:
        return None
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise PolicyGenerationError(f"{label} must be an absolute HTTPS URL")
    if parsed.username or parsed.password or "@" in parsed.netloc:
        raise PolicyGenerationError(f"{label} must not contain URL userinfo")
    try:
        parsed.port
    except ValueError:
        raise PolicyGenerationError(f"{label} must include a valid port") from None
    return value


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


def optional_non_empty_string(value: str | None, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise PolicyGenerationError(f"{label} must be a non-empty string")
    return value.strip()


def parse_binding_values(values: list[str], *, label: str) -> dict[str, list[str]]:
    bindings: dict[str, list[str]] = {}
    for value in values:
        if "=" not in value:
            raise PolicyGenerationError(
                f"{label} values must be formatted as MODEL=WORKLOAD"
            )
        raw_model_id, raw_workload = value.split("=", 1)
        model_id = raw_model_id.strip().lower()
        workload = raw_workload.strip()
        if not model_id or not workload:
            raise PolicyGenerationError(
                f"{label} values must be formatted as MODEL=WORKLOAD"
            )
        bindings.setdefault(model_id, []).append(workload)
    return {
        model_id: non_empty_unique_strings(workloads, label=label)
        for model_id, workloads in bindings.items()
    }


def build_model_workload_bindings(
    *,
    model_ids: list[str],
    expected_sans: list[str],
    expected_ids: list[str],
    model_workload_sans: dict[str, list[str]],
    model_workload_ids: dict[str, list[str]],
) -> dict[str, dict[str, list[str]]]:
    explicit_models = set(model_workload_sans) | set(model_workload_ids)
    selected_models = set(model_ids)
    if not explicit_models:
        if len(model_ids) != 1:
            raise PolicyGenerationError(
                "multiple Privatemode models require explicit --model-workload "
                "bindings"
            )
        explicit_models = selected_models
        model_workload_sans = {model_ids[0]: expected_sans} if expected_sans else {}
        model_workload_ids = {model_ids[0]: expected_ids} if expected_ids else {}
    if explicit_models != selected_models:
        missing = sorted(selected_models - explicit_models)
        extra = sorted(explicit_models - selected_models)
        if missing:
            raise PolicyGenerationError(
                "model_workload_bindings is missing selected models: "
                + ", ".join(missing)
            )
        raise PolicyGenerationError(
            "model_workload_bindings contains unselected models: " + ", ".join(extra)
        )

    expected_san_set = set(expected_sans)
    expected_id_set = set(expected_ids)
    bound_sans: set[str] = set()
    bound_ids: set[str] = set()
    bindings: dict[str, dict[str, list[str]]] = {}
    for model_id in sorted(model_ids):
        sans = sorted(model_workload_sans.get(model_id, []))
        workload_ids = sorted(model_workload_ids.get(model_id, []))
        if not sans and not workload_ids:
            raise PolicyGenerationError(
                f"model_workload_bindings for {model_id} requires workload_sans "
                "or workload_ids"
            )
        if set(sans) - expected_san_set:
            raise PolicyGenerationError(
                f"model_workload_bindings for {model_id} references "
                "workload_sans not listed in expected_workload_sans"
            )
        if set(workload_ids) - expected_id_set:
            raise PolicyGenerationError(
                f"model_workload_bindings for {model_id} references "
                "workload_ids not listed in expected_workload_ids"
            )
        binding: dict[str, list[str]] = {}
        if sans:
            binding["workload_sans"] = sans
        if workload_ids:
            binding["workload_ids"] = workload_ids
        bindings[model_id] = binding
        bound_sans.update(sans)
        bound_ids.update(workload_ids)
    if expected_san_set - bound_sans:
        raise PolicyGenerationError("expected_workload_sans contains unbound workloads")
    if expected_id_set - bound_ids:
        raise PolicyGenerationError("expected_workload_ids contains unbound workloads")
    return bindings


def verifier_digest_from_manifest(path: Path) -> str:
    manifest = load_json_file(path)
    entries = manifest.get("verifiers") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        raise PolicyGenerationError("verifier manifest must contain verifiers[]")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and "privatemode" in [str(mode) for mode in entry.get("modes", [])]
    ]
    if len(matches) != 1:
        raise PolicyGenerationError(
            "verifier manifest must contain exactly one privatemode verifier"
        )
    return normalize_sha256_digest(
        matches[0].get("sha256"),
        label="privatemode verifier sha256",
    )


def build_policy_document(
    *,
    model_ids: list[str],
    base_url: str,
    manifest_digest: str,
    manifest_url: str | None,
    manifest_path: str | None,
    manifest_log_dir: str | None,
    proxy_binary_path: Path,
    proxy_binary_digest: str,
    verifier_command: str,
    verifier_command_digest: str,
    api_base_url: str | None,
    cdn_base_url: str | None,
    expected_coordinator_measurement: str,
    expected_secret_service_measurement: str,
    expected_ai_worker_measurement: str,
    expected_gpu_attestation_policy: str,
    expected_key_release_binding: str,
    expected_workload_sans: list[str],
    expected_workload_ids: list[str],
    model_workload_bindings: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "manifest_digest": manifest_digest,
        "proxy_binary_digest": proxy_binary_digest,
        "proxy_binary_path": str(proxy_binary_path),
        "dump_requests": False,
        "shared_prompt_cache": False,
        "nvidia_ocsp_allow_unknown": False,
        "nvidia_ocsp_revoked_grace_period_hours": 0,
        "expected_trust_tier": VERIFIER_TRUST_TIER,
        "expected_coordinator_measurement": expected_coordinator_measurement,
        "expected_secret_service_measurement": expected_secret_service_measurement,
        "expected_ai_worker_measurement": expected_ai_worker_measurement,
        "expected_gpu_attestation_policy": expected_gpu_attestation_policy,
        "expected_key_release_binding": expected_key_release_binding,
        "model_workload_bindings": model_workload_bindings,
        "verifier_command": verifier_command,
        "verifier_command_digest": verifier_command_digest,
    }
    if expected_workload_sans:
        policy["expected_workload_sans"] = expected_workload_sans
    if expected_workload_ids:
        policy["expected_workload_ids"] = expected_workload_ids
    for key, value in (
        ("manifest_url", manifest_url),
        ("manifest_path", manifest_path),
        ("manifest_log_dir", manifest_log_dir),
        ("api_base_url", api_base_url),
        ("cdn_base_url", cdn_base_url),
    ):
        if value:
            policy[key] = value
    return {
        "base_url": base_url,
        "confidentiality": {
            "enabled": True,
            "mode": "privatemode",
            "model_ids": model_ids,
            "policy": policy,
        },
    }


def generate_policy(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model_id:
        raise PolicyGenerationError("at least one --model-id is required")
    model_ids = normalize_model_ids(args.model_id)
    if args.provider_catalog_json:
        require_selected_models_in_provider_catalog(
            model_ids,
            load_privatemode_provider_catalog_models(args.provider_catalog_json),
        )
    base_url = require_loopback_url(args.base_url, label="base_url")
    expected_sans = non_empty_unique_strings(args.workload_san, label="workload_san")
    expected_ids = non_empty_unique_strings(args.workload_id, label="workload_id")
    if not expected_sans and not expected_ids:
        raise PolicyGenerationError(
            "at least one --workload-san or --workload-id is required"
        )
    model_workload_bindings = build_model_workload_bindings(
        model_ids=model_ids,
        expected_sans=expected_sans,
        expected_ids=expected_ids,
        model_workload_sans=parse_binding_values(
            args.model_workload_san,
            label="model_workload_san",
        ),
        model_workload_ids=parse_binding_values(
            args.model_workload_id,
            label="model_workload_id",
        ),
    )
    proxy_binary_digest = sha256_file(args.proxy_binary_path)
    if args.proxy_binary_digest:
        expected_proxy_binary_digest = normalize_sha256_digest(
            args.proxy_binary_digest,
            label="proxy_binary_digest",
        )
        if expected_proxy_binary_digest != proxy_binary_digest:
            raise PolicyGenerationError(
                "proxy_binary_digest does not match --proxy-binary-path contents"
            )
    verifier_command_digest = (
        normalize_sha256_digest(
            args.verifier_command_digest,
            label="verifier_command_digest",
        )
        if args.verifier_command_digest
        else verifier_digest_from_manifest(args.verifier_manifest_json)
    )
    gpu_policy = args.expected_gpu_attestation_policy.strip()
    if gpu_policy != VERIFIER_GPU_ATTESTATION_POLICY:
        raise PolicyGenerationError(
            "expected_gpu_attestation_policy must be "
            f"{VERIFIER_GPU_ATTESTATION_POLICY}"
        )
    document = build_policy_document(
        model_ids=model_ids,
        base_url=base_url,
        manifest_digest=normalize_sha256_digest(
            args.manifest_digest,
            label="manifest_digest",
        ),
        manifest_url=optional_https_url(args.manifest_url, label="manifest_url"),
        manifest_path=optional_non_empty_string(
            args.manifest_path,
            label="manifest_path",
        ),
        manifest_log_dir=optional_non_empty_string(
            args.manifest_log_dir,
            label="manifest_log_dir",
        ),
        proxy_binary_path=args.proxy_binary_path,
        proxy_binary_digest=proxy_binary_digest,
        verifier_command=normalize_command(
            args.verifier_command,
            label="verifier_command",
        ),
        verifier_command_digest=verifier_command_digest,
        api_base_url=optional_https_url(args.api_base_url, label="api_base_url"),
        cdn_base_url=optional_https_url(args.cdn_base_url, label="cdn_base_url"),
        expected_coordinator_measurement=normalize_sha256_digest(
            args.expected_coordinator_measurement,
            label="expected_coordinator_measurement",
        ),
        expected_secret_service_measurement=normalize_sha256_digest(
            args.expected_secret_service_measurement,
            label="expected_secret_service_measurement",
        ),
        expected_ai_worker_measurement=normalize_sha256_digest(
            args.expected_ai_worker_measurement,
            label="expected_ai_worker_measurement",
        ),
        expected_gpu_attestation_policy=gpu_policy,
        expected_key_release_binding=normalize_sha256_digest(
            args.expected_key_release_binding,
            label="expected_key_release_binding",
        ),
        expected_workload_sans=expected_sans,
        expected_workload_ids=expected_ids,
        model_workload_bindings=model_workload_bindings,
    )
    try:
        return validate_runtime_policy_document(document, provider_type="privatemode")
    except RuntimePolicyValidationError as exc:
        raise PolicyGenerationError(str(exc)) from exc


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a fail-closed Privatemode Routstr policy from pinned "
            "manifest, proxy, verifier, component, and workload evidence."
        )
    )
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--provider-catalog-json", type=Path)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--manifest-digest", required=True)
    parser.add_argument("--manifest-url")
    parser.add_argument("--manifest-path")
    parser.add_argument("--manifest-log-dir")
    parser.add_argument("--proxy-binary-path", type=Path, required=True)
    parser.add_argument("--proxy-binary-digest")
    parser.add_argument("--verifier-command-digest")
    parser.add_argument(
        "--verifier-manifest-json",
        type=Path,
        default=DEFAULT_VERIFIER_MANIFEST,
    )
    parser.add_argument("--verifier-command", default=DEFAULT_VERIFIER_COMMAND)
    parser.add_argument("--api-base-url")
    parser.add_argument("--cdn-base-url")
    parser.add_argument("--expected-coordinator-measurement", required=True)
    parser.add_argument("--expected-secret-service-measurement", required=True)
    parser.add_argument("--expected-ai-worker-measurement", required=True)
    parser.add_argument(
        "--expected-gpu-attestation-policy",
        required=True,
        help=(
            "GPU policy claim expected from the current Privatemode verifier; "
            f"must be {VERIFIER_GPU_ATTESTATION_POLICY}"
        ),
    )
    parser.add_argument("--expected-key-release-binding", required=True)
    parser.add_argument("--workload-san", action="append", default=[])
    parser.add_argument("--workload-id", action="append", default=[])
    parser.add_argument("--model-workload-san", action="append", default=[])
    parser.add_argument("--model-workload-id", action="append", default=[])
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
