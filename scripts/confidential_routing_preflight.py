#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT_DIR = ROOT / "build" / "confidential-verifiers"
DEFAULT_MAX_ATTESTATION_RESULT_AGE_SECONDS = 86_400
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
RFC3339_EXTRA_FRACTION_RE = re.compile(r"(\.\d{6})\d+(?=Z|[+-]\d\d:\d\d$)")


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
class VerifierTarget:
    name: str
    module_dir: Path
    output_name: str
    test_args: tuple[str, ...]
    build_args: tuple[str, ...]
    purpose: str
    modes: tuple[str, ...]
    verifier_names: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class AttestationTargetsDirectory:
    provider_hosts: dict[str, set[str]]
    provider_host_model_ids: dict[str, dict[str, set[str]]]
    provider_host_repos: dict[str, dict[str, set[str]]]
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
    provider_check_statuses: dict[str, dict[str, str]] = field(default_factory=dict)
    provider_check_errors: dict[str, dict[str, str]] = field(default_factory=dict)
    provider_host_details: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=dict
    )
    provider_check_details: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=dict
    )
    provider_model_statuses: dict[str, dict[str, str]] = field(default_factory=dict)
    provider_model_errors: dict[str, dict[str, str]] = field(default_factory=dict)
    provider_model_details: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ProviderModelCatalog:
    provider_models: dict[str, set[str]]


TARGETS = (
    VerifierTarget(
        name="tinfoil-ppq",
        module_dir=ROOT / "verifiers" / "tinfoil-go",
        output_name="routstr-tinfoil-go-verifier",
        test_args=("go", "test", "-count=1", "./..."),
        build_args=("go", "build", "-trimpath", "-ldflags=-s -w"),
        purpose="Tinfoil and PPQ private EHBP provider verifier",
        modes=("tinfoil", "ppq-private-tee"),
        verifier_names=(
            ("tinfoil", "routstr-tinfoil-go-verifier"),
            ("ppq-private-tee", "routstr-ppq-private-go-verifier"),
        ),
    ),
    VerifierTarget(
        name="privatemode",
        module_dir=ROOT / "verifiers" / "privatemode-go",
        output_name="routstr-privatemode-go-verifier",
        test_args=("go", "test", "-count=1", "-tags", "contrast_unstable_api", "./..."),
        build_args=(
            "go",
            "build",
            "-tags",
            "contrast_unstable_api",
            "-trimpath",
            "-ldflags=-s -w",
        ),
        purpose="Privatemode proxy and backend verifier",
        modes=("privatemode",),
        verifier_names=(("privatemode", "routstr-privatemode-go-verifier"),),
    ),
    VerifierTarget(
        name="routstr-tee",
        module_dir=ROOT / "verifiers" / "routstr-tee-go",
        output_name="routstr-tee-go-verifier",
        test_args=("go", "test", "-count=1", "./..."),
        build_args=("go", "build", "-trimpath", "-ldflags=-s -w"),
        purpose="Local Routstr TEE quote verifier",
        modes=(),
    ),
    VerifierTarget(
        name="routstr-tee-attest",
        module_dir=ROOT / "verifiers" / "routstr-tee-attest-go",
        output_name="routstr-tee-attest-go",
        test_args=("go", "test", "-count=1", "./..."),
        build_args=("go", "build", "-trimpath", "-ldflags=-s -w"),
        purpose="Local Routstr TEE quote generation command",
        modes=(),
    ),
)
SUPPORTED_CONFIDENTIALITY_MODES = {"tinfoil", "ppq-private-tee", "privatemode"}
RUNTIME_PROVIDER_TYPE_BY_CONFIDENTIALITY_MODE = {
    "tinfoil": "tinfoil",
    "ppq-private-tee": "ppq-private",
    "privatemode": "privatemode",
}
SUPPORTED_PROVIDER_MODEL_SUPPORT_SLUGS = ("tinfoil", "ppq", "privatemode")
KNOWN_TINFOIL_MODEL_HOST_REPOS = {
    "llama3-3-70b.tinfoil.containers.tinfoil.dev": {
        "tinfoilsh/confidential-llama3-3-70b",
    },
    "deepseek-v4-pro-inf12.tinfoil.containers.tinfoil.dev": {
        "tinfoilsh/confidential-deepseek-v4-pro",
    },
    "kimi-k2-6.inf13.tinfoil.sh": {
        "tinfoilsh/confidential-kimi-k2-6-b200",
    },
    "gemma4-31b-1.inf10.tinfoil.sh": {
        "tinfoilsh/confidential-gemma4-31b",
    },
    "qwen3-vl-30b.inf10.tinfoil.sh": {
        "tinfoilsh/confidential-qwen3-vl-30b",
    },
    "glm-5-1.tinfoil.containers.tinfoil.dev": {
        "tinfoilsh/confidential-glm5-1",
    },
    "gpt-oss-120b-1.inf10.tinfoil.sh": {
        "tinfoilsh/confidential-gpt-oss-120b",
    },
    "gpt-oss-safeguard-120b.tinfoil.containers.tinfoil.dev": {
        "tinfoilsh/confidential-gpt-oss-safeguard-120b",
    },
}


SECRET_POLICY_KEYS = {
    "access_token",
    "accesstoken",
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "bearer_token",
    "bearertoken",
    "client_secret",
    "clientsecret",
    "password",
    "refresh_token",
    "refreshtoken",
    "token",
    "secret",
    "upstream_api_key",
    "upstreamapikey",
}
SECRET_LIKE_VALUE_PATTERN = re.compile(
    r"(?i)(?:bearer\s+\S+|(?:^|[=:\s,;])sk-[A-Za-z0-9][A-Za-z0-9._-]*)"
)
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
    "gpu_attestation_policy": (
        "expected_gpu_attestation_policy",
        "allowed_gpu_attestation_policies",
    ),
    "trust_tier": (
        "expected_trust_tier",
        "allowed_trust_tiers",
    ),
}
PRIVATEMODE_GPU_ATTESTATION_POLICY = "nvidia-ocsp-good-only"
REQUIRED_PRIVATEMODE_RESULT_VERIFICATION_STEPS = (
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
REQUIRED_PRIVATEMODE_RESULT_DIGEST_CLAIMS = (
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
    *REQUIRED_PRIVATEMODE_RESULT_DIGEST_CLAIMS,
)
TDX_ATTESTATION_FORMATS = {
    "tdx_quote",
    "tdx_guest_v2",
    "https://tinfoil.sh/predicate/tdx-guest/v2",
}
SEV_SNP_ATTESTATION_FORMATS = {
    "sev_snp_quote",
    "sev_snp_report",
    "sev_guest_v2",
    "https://tinfoil.sh/predicate/sev-snp-guest/v2",
}
ROUTSTR_TEE_ATTESTATION_FORMATS = TDX_ATTESTATION_FORMATS | SEV_SNP_ATTESTATION_FORMATS
KEM_X25519_HKDF_SHA256 = 0x0020
KDF_HKDF_SHA256 = 0x0001
AEAD_AES_256_GCM = 0x0002
HPKE_PUBLIC_KEY_SIZES = {KEM_X25519_HKDF_SHA256: 32}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def sha256_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def run_command(args: list[str], *, cwd: Path, log_file: Any = None) -> None:
    if log_file is None:
        log_file = sys.stdout
    print(f"$ {' '.join(args)}", flush=True, file=log_file)
    subprocess.run(
        args,
        cwd=cwd,
        check=True,
        stdout=log_file if log_file is not sys.stdout else None,
        stderr=log_file if log_file is not sys.stdout else None,
    )


def build_verifiers(
    *,
    output_dir: Path,
    skip_tests: bool,
    skip_build: bool,
    log_file: Any = None,
) -> list[dict[str, Any]]:
    if shutil.which("go") is None:
        raise RuntimeError("go is required to test and build verifier commands")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for target in TARGETS:
        if not target.module_dir.is_dir():
            raise RuntimeError(f"missing verifier module: {target.module_dir}")

        if not skip_tests:
            run_command(list(target.test_args), cwd=target.module_dir, log_file=log_file)

        output_path = output_dir / target.output_name
        if not skip_build:
            build_args = list(target.build_args) + ["-o", str(output_path), "."]
            run_command(build_args, cwd=target.module_dir, log_file=log_file)

        if output_path.exists():
            digest = sha256_file(output_path)
            size_bytes: int | None = output_path.stat().st_size
        else:
            digest = None
            size_bytes = None

        manifest.append(
            {
                "name": target.name,
                "modes": list(target.modes),
                "verifier_names": dict(target.verifier_names),
                "purpose": target.purpose,
                "module_dir": str(target.module_dir.relative_to(ROOT)),
                "artifact_path": str(output_path),
                "sha256": digest,
                "size_bytes": size_bytes,
            }
        )

    manifest_path = output_dir / "confidential-verifier-manifest.json"
    manifest_path.write_text(
        json.dumps({"verifiers": manifest}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def env_value(name: str) -> str:
    return os.environ.get(name, "").strip()


def read_hpke_u16(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 2 > len(data):
        raise ValueError("invalid EHBP HPKE key config: truncated uint16")
    return int.from_bytes(data[offset : offset + 2], "big"), offset + 2


def ehbp_key_config_issue(data: bytes) -> str | None:
    try:
        if len(data) < 1 + 2 + 2:
            raise ValueError("invalid EHBP HPKE key config: too short")

        offset = 0
        key_id = data[offset]
        offset += 1

        kem_id, offset = read_hpke_u16(data, offset)
        public_key_size = HPKE_PUBLIC_KEY_SIZES.get(kem_id)
        if public_key_size is None:
            raise ValueError(
                f"invalid EHBP HPKE key config: unsupported KEM 0x{kem_id:04x}"
            )
        if offset + public_key_size > len(data):
            raise ValueError("invalid EHBP HPKE key config: truncated public key")

        offset += public_key_size
        suites_len, offset = read_hpke_u16(data, offset)
        if suites_len == 0:
            raise ValueError("invalid EHBP HPKE key config: no cipher suites")
        if suites_len % 4 != 0:
            raise ValueError(
                "invalid EHBP HPKE key config: malformed cipher suite list"
            )
        if offset + suites_len > len(data):
            raise ValueError(
                "invalid EHBP HPKE key config: truncated cipher suite list"
            )

        suites: list[tuple[int, int]] = []
        suites_end = offset + suites_len
        while offset < suites_end:
            kdf_id, offset = read_hpke_u16(data, offset)
            aead_id, offset = read_hpke_u16(data, offset)
            suites.append((kdf_id, aead_id))

        if offset != len(data):
            raise ValueError("invalid EHBP HPKE key config: trailing bytes")
        if key_id != 0:
            raise ValueError(
                f"invalid EHBP HPKE key config: unsupported key id {key_id}"
            )
        if not suites or suites[0] != (KDF_HKDF_SHA256, AEAD_AES_256_GCM):
            kdf_id, aead_id = suites[0]
            raise ValueError(
                "invalid EHBP HPKE key config: unsupported EHBP cipher suite "
                f"KDF=0x{kdf_id:04x} AEAD=0x{aead_id:04x}"
            )
    except ValueError as exc:
        return f"invalid Routstr TEE HPKE key config: {exc}"
    return None


def routstr_hpke_key_config_issue() -> str | None:
    encoded = env_value("ROUTSTR_TEE_HPKE_KEY_CONFIG_B64")
    if encoded:
        try:
            key_config = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            return f"ROUTSTR_TEE_HPKE_KEY_CONFIG_B64 is not valid base64: {exc}"
        return ehbp_key_config_issue(key_config)

    path = env_value("ROUTSTR_TEE_HPKE_KEY_CONFIG_PATH")
    if path:
        try:
            key_config = Path(path).read_bytes()
        except Exception as exc:
            return (
                "failed to read Routstr TEE HPKE key config: "
                f"{type(exc).__name__}: {exc}"
            )
        return ehbp_key_config_issue(key_config)

    return "set ROUTSTR_TEE_HPKE_KEY_CONFIG_B64 or ROUTSTR_TEE_HPKE_KEY_CONFIG_PATH"


def routstr_public_key_value() -> tuple[str | None, str | None]:
    public_key = env_value("ROUTSTR_TEE_PUBLIC_KEY")
    if public_key:
        return public_key, None

    path = env_value("ROUTSTR_TEE_PUBLIC_KEY_PATH")
    if path:
        try:
            public_key = Path(path).read_text(encoding="utf-8").strip()
        except Exception as exc:
            return None, (
                f"failed to read Routstr TEE public key: {type(exc).__name__}: {exc}"
            )
        if not public_key:
            return None, "Routstr TEE public key file is empty"
        return public_key, None

    return None, (
        "set ROUTSTR_TEE_PUBLIC_KEY or ROUTSTR_TEE_PUBLIC_KEY_PATH so local TEE "
        "proof binds Routstr's public key"
    )


def routstr_public_key_issue() -> str | None:
    _, issue = routstr_public_key_value()
    return issue


def routstr_attested_tls_public_key_digest_check() -> tuple[str | None, str | None]:
    public_key, issue = routstr_public_key_value()
    if public_key is None:
        if issue and issue.startswith("set ROUTSTR_TEE_PUBLIC_KEY"):
            return None, (
                "set ROUTSTR_TEE_PUBLIC_KEY or ROUTSTR_TEE_PUBLIC_KEY_PATH so "
                "attested TLS boundary can publish Routstr's public key digest"
            )
        return None, issue
    actual_digest = sha256_text(public_key)
    configured_digest = env_value("ROUTSTR_TEE_ATTESTED_TLS_PUBLIC_KEY_DIGEST")
    if not configured_digest:
        return actual_digest, None
    if not is_sha256_digest_value(configured_digest):
        return None, "must be a sha256 digest"
    if is_template_placeholder_digest_value(configured_digest):
        return None, "placeholder digest is not allowed"
    normalized_configured_digest = f"sha256:{sha256_digest_hex(configured_digest)}"
    if normalized_configured_digest != actual_digest:
        return None, (
            "configured digest does not match ROUTSTR_TEE_PUBLIC_KEY; "
            f"expected {actual_digest}"
        )
    return normalized_configured_digest, None


def routstr_attestation_format_issue(value: str) -> str | None:
    if not value:
        return (
            "set the hardware evidence format, for example tdx_quote or sev_snp_report"
        )
    if value.strip().lower() not in ROUTSTR_TEE_ATTESTATION_FORMATS:
        return (
            "unsupported ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT; expected TDX or "
            "SEV-SNP evidence format"
        )
    return None


def tdx_quote_command_value(policy: dict[str, Any]) -> object | None:
    if "tdx_quote_command" in policy:
        return policy["tdx_quote_command"]
    value = policy.get("tdx_quote_command")
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, list) and value:
        return value
    env_command = env_value("ROUTSTR_TDX_QUOTE_COMMAND")
    if env_command:
        return env_command
    return None


def tdx_quote_artifact_path_value(policy: dict[str, Any]) -> str:
    value = policy.get("tdx_quote_artifact_path")
    if value is None and "tdx_quote_artifact_path" not in policy:
        return env_value("ROUTSTR_TDX_QUOTE_ARTIFACT_PATH")
    if not isinstance(value, str) or not value.strip():
        return ""
    return value.strip()


def tdx_quote_artifact_path_issue(policy: dict[str, Any]) -> str | None:
    if "tdx_quote_artifact_path" not in policy:
        return None
    value = policy["tdx_quote_artifact_path"]
    if not isinstance(value, str) or not value.strip():
        return "tdx_quote_artifact_path must be a non-empty string"
    return None


def tdx_quote_policy_issues(
    policy: dict[str, Any],
    evidence_format: str,
) -> list[str]:
    issues: list[str] = []
    policy_digest = ""
    if "tdx_quote_command_digest" in policy:
        raw_policy_digest = policy["tdx_quote_command_digest"]
        if not isinstance(raw_policy_digest, str) or not raw_policy_digest.strip():
            issues.append("tdx_quote_command_digest must be a sha256 digest")
        else:
            policy_digest = raw_policy_digest.strip()
    env_digest = env_value("ROUTSTR_TDX_QUOTE_COMMAND_DIGEST")
    if policy_digest and not is_prefixed_sha256_digest_value(policy_digest):
        issues.append("tdx_quote_command_digest must be a sha256 digest")
    if env_digest and not is_prefixed_sha256_digest_value(env_digest):
        issues.append("ROUTSTR_TDX_QUOTE_COMMAND_DIGEST must be a sha256 digest")
    if artifact_path_issue := tdx_quote_artifact_path_issue(policy):
        issues.append(artifact_path_issue)

    command = tdx_quote_command_value(policy)
    command_argv_values: list[str] = []
    if command is not None:
        try:
            command_argv_values = command_argv(command, label="tdx_quote_command")
        except ValueError as exc:
            issues.append(str(exc))
        else:
            issues.extend(
                inline_secret_violations(
                    command_argv_values,
                    path="tdx_quote_command",
                )
            )

    artifact_path = tdx_quote_artifact_path_value(policy)
    if (
        artifact_path
        and command_argv_values
        and artifact_path not in command_argv_values
    ):
        issues.append(
            "tdx_quote_artifact_path must match tdx quote command executable or argument"
        )

    if evidence_format.strip().lower() in TDX_ATTESTATION_FORMATS:
        if command is None:
            issues.append(
                "tdx_quote_command or ROUTSTR_TDX_QUOTE_COMMAND is required for TDX quote generation"
            )
        if not policy_digest and not env_digest:
            issues.append(
                "tdx_quote_command_digest or ROUTSTR_TDX_QUOTE_COMMAND_DIGEST is required for TDX quote generation"
            )
    return issues


def sev_snp_quote_policy_issues(
    policy: dict[str, Any],
    evidence_format: str,
) -> list[str]:
    if evidence_format.strip().lower() not in SEV_SNP_ATTESTATION_FORMATS:
        return []

    issues: list[str] = []
    if "sev_guest_device_path" in policy:
        value = policy["sev_guest_device_path"]
        if not isinstance(value, str) or not value.strip():
            issues.append("sev_guest_device_path must be a non-empty string")

    vmpl_values: list[tuple[str, int]] = []
    for key in ("sev_snp_vmpl", "vmpl"):
        if key not in policy:
            continue
        value = policy[key]
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 3:
            vmpl_values.append((key, value))
            continue
        issues.append(f"{key} must be an integer from 0 to 3")
    if len(vmpl_values) > 1 and len({value for _, value in vmpl_values}) > 1:
        issues.append("sev_snp_vmpl and vmpl aliases must match")

    return issues


def max_age_policy_issues(policy: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    selected_key: str | None = None
    selected_value: int | None = None
    for key in ("max_verifier_age_seconds", "max_evidence_age_seconds"):
        if key not in policy:
            continue
        value = policy[key]
        if not isinstance(value, int) or isinstance(value, bool):
            issues.append(f"{key} must be an integer")
            continue
        if value <= 0:
            issues.append(f"{key} must be positive")
            continue
        if selected_value is None:
            selected_key = key
            selected_value = value
            continue
        if value != selected_value:
            issues.append(f"{selected_key} and {key} aliases must match")
    return issues


def routstr_tee_max_age_policy_issues(policy: dict[str, Any]) -> list[str]:
    return max_age_policy_issues(policy)


def runtime_command_argv(value: str, *, label: str) -> list[str]:
    stripped = value.strip()
    if stripped.startswith("["):
        try:
            parsed = json.loads(
                stripped,
                parse_constant=json_constant_rejecter(f"{label} JSON"),
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} JSON is invalid: {exc}") from exc
        return command_argv(parsed, label=label)
    return command_argv(stripped, label=label)


def env_command_issue(command_key: str) -> str | None:
    command = env_value(command_key)
    if not command:
        return None
    try:
        argv = runtime_command_argv(command, label=command_key)
    except ValueError as exc:
        return str(exc)
    if violations := inline_secret_violations(argv, path=command_key):
        return "; ".join(violations)
    return None


def env_command_artifact_path_issue(
    *,
    command_key: str,
    artifact_key: str,
) -> str | None:
    command = env_value(command_key)
    artifact_path = env_value(artifact_key)
    if not command or not artifact_path:
        return None
    try:
        argv = runtime_command_argv(command, label=command_key)
    except ValueError as exc:
        return str(exc)
    if artifact_path not in argv:
        return f"must match {command_key} executable or argument"
    return None


def client_confidentiality_boundary_issue() -> str | None:
    boundary = env_value("ROUTSTR_TEE_CLIENT_CONFIDENTIALITY_BOUNDARY")
    if not boundary:
        return (
            "set to attested-tls-termination; inbound EHBP/OHTTP request "
            "decryption is not implemented"
        )
    if boundary.strip().lower() != "attested-tls-termination":
        return (
            "must be attested-tls-termination; inbound EHBP/OHTTP request "
            "decryption is not implemented"
        )
    return None


def env_checks() -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    def add(status: str, key: str, detail: str) -> None:
        checks.append({"status": status, "key": key, "detail": detail})

    routing_mode = env_value("CONFIDENTIAL_ROUTING_MODE")
    if routing_mode == "required":
        add("ok", "CONFIDENTIAL_ROUTING_MODE", "required")
    else:
        add(
            "missing",
            "CONFIDENTIAL_ROUTING_MODE",
            "set to required for verified-only production routing",
        )

    if boundary_issue := client_confidentiality_boundary_issue():
        add(
            "missing",
            "ROUTSTR_TEE_CLIENT_CONFIDENTIALITY_BOUNDARY",
            boundary_issue,
        )
    else:
        add(
            "ok",
            "ROUTSTR_TEE_CLIENT_CONFIDENTIALITY_BOUNDARY",
            "attested TLS termination declared",
        )

    if env_value("ROUTSTR_TEE_ATTESTATION_REQUIRED").lower() == "false":
        add(
            "missing",
            "ROUTSTR_TEE_ATTESTATION_REQUIRED",
            "must not be false for the target deployment",
        )

    attestation_format = env_value("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT")
    if attestation_format_issue := routstr_attestation_format_issue(attestation_format):
        add(
            "missing",
            "ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT",
            attestation_format_issue,
        )
    else:
        add("ok", "ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT", "configured")

    required_env = (
        (
            "ROUTSTR_TEE_VERIFIER_COMMAND",
            "point at the built routstr-tee-go-verifier binary",
        ),
        (
            "ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST",
            "pin the routstr-tee-go-verifier sha256 digest",
        ),
    )
    for key, detail in required_env:
        value = env_value(key)
        if key.endswith("_DIGEST") and value:
            digest_ok = is_prefixed_sha256_digest_value(value)
            placeholder = digest_ok and is_template_placeholder_digest_value(value)
            add(
                "ok" if digest_ok and not placeholder else "missing",
                key,
                "pin configured"
                if digest_ok and not placeholder
                else (
                    "placeholder digest is not allowed"
                    if placeholder
                    else "must be a sha256 digest"
                ),
            )
        else:
            add("ok" if value else "missing", key, detail)

    if verifier_command_issue := env_command_issue("ROUTSTR_TEE_VERIFIER_COMMAND"):
        add("missing", "ROUTSTR_TEE_VERIFIER_COMMAND", verifier_command_issue)
    if verifier_artifact_issue := env_command_artifact_path_issue(
        command_key="ROUTSTR_TEE_VERIFIER_COMMAND",
        artifact_key="ROUTSTR_TEE_VERIFIER_ARTIFACT_PATH",
    ):
        add("missing", "ROUTSTR_TEE_VERIFIER_ARTIFACT_PATH", verifier_artifact_issue)

    if public_key_issue := routstr_public_key_issue():
        add(
            "missing",
            "ROUTSTR_TEE_PUBLIC_KEY",
            public_key_issue,
        )
    else:
        add("ok", "ROUTSTR_TEE_PUBLIC_KEY", "configured")

    public_key_digest, public_key_digest_issue = (
        routstr_attested_tls_public_key_digest_check()
    )
    if public_key_digest is None:
        add(
            "missing",
            "ROUTSTR_TEE_ATTESTED_TLS_PUBLIC_KEY_DIGEST",
            public_key_digest_issue
            or "Routstr public key is required for attested TLS public-key digest",
        )
    else:
        add(
            "ok",
            "ROUTSTR_TEE_ATTESTED_TLS_PUBLIC_KEY_DIGEST",
            public_key_digest,
        )

    if hpke_key_config_issue := routstr_hpke_key_config_issue():
        add(
            "missing",
            "ROUTSTR_TEE_HPKE_KEY_CONFIG",
            hpke_key_config_issue,
        )
    else:
        add("ok", "ROUTSTR_TEE_HPKE_KEY_CONFIG", "configured")

    if env_value("ROUTSTR_TEE_ATTESTATION_COMMAND"):
        add("ok", "ROUTSTR_TEE_ATTESTATION_COMMAND", "configured")
        if attestation_command_issue := env_command_issue(
            "ROUTSTR_TEE_ATTESTATION_COMMAND"
        ):
            add(
                "missing",
                "ROUTSTR_TEE_ATTESTATION_COMMAND",
                attestation_command_issue,
            )
        if attestation_artifact_issue := env_command_artifact_path_issue(
            command_key="ROUTSTR_TEE_ATTESTATION_COMMAND",
            artifact_key="ROUTSTR_TEE_ATTESTATION_ARTIFACT_PATH",
        ):
            add(
                "missing",
                "ROUTSTR_TEE_ATTESTATION_ARTIFACT_PATH",
                attestation_artifact_issue,
            )
        attestation_digest = env_value("ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST")
        attestation_digest_ok = bool(
            attestation_digest and is_prefixed_sha256_digest_value(attestation_digest)
        )
        attestation_digest_placeholder = bool(
            attestation_digest_ok
            and attestation_digest
            and is_template_placeholder_digest_value(attestation_digest)
        )
        add(
            "ok"
            if attestation_digest_ok and not attestation_digest_placeholder
            else "missing",
            "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
            "pin configured"
            if attestation_digest_ok and not attestation_digest_placeholder
            else (
                "placeholder digest is not allowed"
                if attestation_digest_placeholder
                else (
                    "must be a sha256 digest"
                    if attestation_digest
                    else "pin the quote generation command digest"
                )
            ),
        )
    elif env_value("ROUTSTR_TEE_ATTESTATION_DOCUMENT_PATH"):
        add(
            "missing",
            "ROUTSTR_TEE_ATTESTATION_COMMAND",
            (
                "static evidence is only suitable for fixtures or offline validation; "
                "live deployments should configure a digest-pinned "
                "ROUTSTR_TEE_ATTESTATION_COMMAND"
            ),
        )
    else:
        add(
            "missing",
            "ROUTSTR_TEE_ATTESTATION_COMMAND",
            "live deployments should generate fresh quote evidence with a digest-pinned command",
        )

    policy_json = env_value("ROUTSTR_TEE_VERIFIER_POLICY_JSON")
    if not policy_json:
        add(
            "missing",
            "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
            "must pin expected_routstr_code_measurement or allowed_routstr_code_measurements",
        )
    else:
        try:
            policy = json.loads(
                policy_json,
                parse_constant=json_constant_rejecter(
                    "ROUTSTR_TEE_VERIFIER_POLICY_JSON"
                ),
            )
        except json.JSONDecodeError as exc:
            add("missing", "ROUTSTR_TEE_VERIFIER_POLICY_JSON", f"invalid JSON: {exc}")
        except ValueError as exc:
            add("missing", "ROUTSTR_TEE_VERIFIER_POLICY_JSON", str(exc))
        else:
            if not isinstance(policy, dict):
                add(
                    "missing",
                    "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
                    "must be a JSON object",
                )
            else:
                placeholder_violations = placeholder_digest_violations(policy)
                for violation in placeholder_violations:
                    add("missing", "ROUTSTR_TEE_VERIFIER_POLICY_JSON", violation)
                if code_measurement_issue := routstr_code_measurement_policy_issue(
                    policy
                ):
                    add(
                        "missing",
                        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
                        code_measurement_issue,
                    )
                elif not placeholder_violations:
                    add(
                        "ok",
                        "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
                        "code measurement pinned",
                    )
                for issue in tdx_quote_policy_issues(
                    policy,
                    env_value("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT"),
                ):
                    add("missing", "ROUTSTR_TEE_VERIFIER_POLICY_JSON", issue)
                for issue in sev_snp_quote_policy_issues(
                    policy,
                    env_value("ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT"),
                ):
                    add("missing", "ROUTSTR_TEE_VERIFIER_POLICY_JSON", issue)
                for issue in routstr_tee_max_age_policy_issues(policy):
                    add("missing", "ROUTSTR_TEE_VERIFIER_POLICY_JSON", issue)
                for violation in inline_secret_violations(policy):
                    add("missing", "ROUTSTR_TEE_VERIFIER_POLICY_JSON", violation)

    return checks


def json_constant_rejecter(label: str) -> Callable[[str], None]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} must not contain {value}")

    return reject_constant


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            parse_constant=json_constant_rejecter("policy JSON"),
        )
    if not isinstance(value, dict):
        raise ValueError("policy JSON root must be an object")
    return value


def load_json_document(path: Path, label: str) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=json_constant_rejecter(label),
        )


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
    if provider_slug not in {"ppq", "tinfoil"}:
        return candidates
    expanded = set(candidates)
    for candidate in candidates:
        if candidate is None:
            continue
        expanded.add(candidate)
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


def model_identity_candidates_from_check(
    check: dict[str, Any],
    *,
    provider_slug: str,
    host: str,
) -> set[str]:
    candidates: set[str] = set()
    for key in ("model", "model_id", "model_slug", "slug"):
        for candidate in model_identity_candidate_variants(check.get(key)):
            candidates.add(candidate.split("/")[-1])

    for key in ("models", "model_ids", "model_slugs", "slugs"):
        raw_values = check.get(key)
        if not isinstance(raw_values, list):
            continue
        for raw_value in raw_values:
            for candidate in model_identity_candidate_variants(raw_value):
                candidates.add(candidate.split("/")[-1])

    for check_id in model_identity_candidate_variants(check.get("id")):
        prefix = f"{provider_slug}-"
        candidates.add(
            check_id.removeprefix(prefix) if check_id.startswith(prefix) else check_id
        )

    for label in model_identity_candidate_variants(check.get("label")):
        candidates.add(label)

    host_prefix = normalized_model_identity(host.split(".", 1)[0])
    if host_prefix:
        candidates.add(host_prefix)

    return {candidate for candidate in candidates if candidate}


def model_identity_matches_directory_candidate(
    model_id: str,
    candidates: set[str],
) -> bool:
    selected = normalized_model_identity(model_id.split("/")[-1])
    if selected is None:
        return False
    return selected in candidates


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


def load_attestation_targets_directory(path: Path) -> AttestationTargetsDirectory:
    """Load known provider attestation target hosts from confidential-inference."""
    document = load_json_file(path)
    providers = document.get("providers")
    if not isinstance(providers, list):
        raise ValueError("attestation targets JSON providers must be a list")

    provider_hosts: dict[str, set[str]] = {}
    provider_host_model_ids: dict[str, dict[str, set[str]]] = {}
    provider_host_repos: dict[str, dict[str, set[str]]] = {}
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
        slug = provider.get("slug")
        if not isinstance(slug, str) or not slug.strip():
            continue
        normalized_slug = slug.strip().lower()
        checks = provider.get("checks")
        if not isinstance(checks, list):
            continue
        check_id_counts: dict[str, int] = {}
        for check in checks:
            if not isinstance(check, dict):
                continue
            raw_check_id = check.get("id")
            if isinstance(raw_check_id, str) and raw_check_id.strip():
                normalized_check_id = raw_check_id.strip().lower()
                check_id_counts[normalized_check_id] = (
                    check_id_counts.get(normalized_check_id, 0) + 1
                )
        duplicate_check_ids = {
            check_id
            for check_id, count in check_id_counts.items()
            if count > 1
        }
        hosts = provider_hosts.setdefault(normalized_slug, set())
        host_model_ids = provider_host_model_ids.setdefault(normalized_slug, {})
        host_repos = provider_host_repos.setdefault(normalized_slug, {})
        check_hosts = provider_check_hosts.setdefault(normalized_slug, {})
        check_models = provider_check_models.setdefault(normalized_slug, {})
        target_models = provider_target_models.setdefault(normalized_slug, set())
        check_backend_hosts = provider_check_backend_hosts.setdefault(
            normalized_slug, {}
        )
        check_backend_repos = provider_check_backend_repos.setdefault(
            normalized_slug, {}
        )
        model_backend_hosts = provider_model_backend_hosts.setdefault(
            normalized_slug, {}
        )
        model_backend_repos = provider_model_backend_repos.setdefault(
            normalized_slug, {}
        )
        model_hosts = provider_model_hosts.setdefault(normalized_slug, {})
        model_target_counts = provider_model_target_counts.setdefault(
            normalized_slug, {}
        )
        ambiguous_target_models = provider_ambiguous_target_models.setdefault(
            normalized_slug, set()
        )
        for check in checks:
            if not isinstance(check, dict):
                continue
            raw_check_id = check.get("id")
            normalized_check_id = (
                raw_check_id.strip().lower()
                if isinstance(raw_check_id, str) and raw_check_id.strip()
                else None
            )
            if normalized_check_id in duplicate_check_ids:
                continue
            check_host = None
            for key in ("host", "url"):
                value = check.get(key)
                if key == "host":
                    candidate_host, _issue = host_identity_hostname(value, key)
                else:
                    candidate_host, _issue = absolute_https_hostname(value, key)
                if candidate_host:
                    check_host = candidate_host
                    break
            is_tinfoil_router = normalized_slug == "tinfoil" and (
                is_tinfoil_router_identity(
                    check_id=raw_check_id,
                    host=check_host,
                    label=check.get("label"),
                )
            )
            if (
                normalized_check_id is not None
                and isinstance(check.get("model"), str)
                and check["model"].strip()
                and not is_tinfoil_router
            ):
                normalized_model = check["model"].strip().lower()
                check_models[normalized_check_id] = normalized_model
                target_models.add(normalized_model)
                model_target_counts[normalized_model] = (
                    model_target_counts.get(normalized_model, 0) + 1
                )
            for key in ("host", "url"):
                value = check.get(key)
                if key == "host":
                    host, _issue = host_identity_hostname(value, key)
                else:
                    host, _issue = absolute_https_hostname(value, key)
                if (
                    host
                    and normalized_check_id is not None
                ):
                    check_hosts[normalized_check_id] = host
                    if (
                        isinstance(check.get("model"), str)
                        and check["model"].strip()
                        and not is_tinfoil_router
                    ):
                        model_hosts[check["model"].strip().lower()] = host
            if normalized_check_id is not None:
                backend_host, _issue = host_identity_hostname(
                    check.get("backend_host"), "backend_host"
                )
                if backend_host:
                    check_backend_hosts[normalized_check_id] = backend_host
                    if (
                        isinstance(check.get("model"), str)
                        and check["model"].strip()
                        and not is_tinfoil_router
                    ):
                        model_backend_hosts[
                            check["model"].strip().lower()
                        ] = backend_host
                backend_repo = check.get("backend_repo")
                if isinstance(backend_repo, str) and backend_repo.strip():
                    check_backend_repos[normalized_check_id] = backend_repo.strip()
                    if (
                        isinstance(check.get("model"), str)
                        and check["model"].strip()
                        and not is_tinfoil_router
                    ):
                        model_backend_repos[
                            check["model"].strip().lower()
                        ] = backend_repo.strip()
            if check.get("type") != "tinfoil":
                continue
            check_id = str(check.get("id") or "").strip().lower()
            label = str(check.get("label") or "").strip().lower()
            if "router" in check_id or "router" in label:
                continue
            for key in ("host", "url"):
                value = check.get(key)
                if key == "host":
                    host, _issue = host_identity_hostname(value, key)
                else:
                    host, _issue = absolute_https_hostname(value, key)
                if host:
                    hosts.add(host)
                    candidates = model_identity_candidates_from_check(
                        check,
                        provider_slug=normalized_slug,
                        host=host,
                    )
                    if candidates:
                        host_model_ids.setdefault(host, set()).update(candidates)
                    repo = check.get("repo")
                    if isinstance(repo, str) and repo.strip():
                        host_repos.setdefault(host, set()).add(repo.strip())
                    elif normalized_slug == "tinfoil":
                        known_repos = KNOWN_TINFOIL_MODEL_HOST_REPOS.get(host)
                        if known_repos:
                            host_repos.setdefault(host, set()).update(known_repos)
        for normalized_model, count in model_target_counts.items():
            if count > 1:
                ambiguous_target_models.add(normalized_model)
                model_hosts.pop(normalized_model, None)
                model_backend_hosts.pop(normalized_model, None)
                model_backend_repos.pop(normalized_model, None)
    return AttestationTargetsDirectory(
        provider_hosts=provider_hosts,
        provider_host_model_ids=provider_host_model_ids,
        provider_host_repos=provider_host_repos,
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
    document = load_json_file(path)
    checks = document.get("checks")
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
    provider_check_statuses: dict[str, dict[str, str]] = {}
    provider_check_errors: dict[str, dict[str, str]] = {}
    provider_host_details: dict[str, dict[str, dict[str, Any]]] = {}
    provider_check_details: dict[str, dict[str, dict[str, Any]]] = {}
    provider_model_statuses: dict[str, dict[str, str]] = {}
    provider_model_errors: dict[str, dict[str, str]] = {}
    provider_model_details: dict[str, dict[str, dict[str, Any]]] = {}
    provider_check_id_counts: dict[str, dict[str, int]] = {}
    for check in checks:
        if not isinstance(check, dict):
            continue
        provider = check.get("provider")
        raw_check_id = check.get("id")
        if (
            not isinstance(provider, str)
            or not provider.strip()
            or not isinstance(raw_check_id, str)
            or not raw_check_id.strip()
        ):
            continue
        normalized_provider = provider.strip().lower()
        normalized_check_id = raw_check_id.strip().lower()
        check_id_counts = provider_check_id_counts.setdefault(normalized_provider, {})
        check_id_counts[normalized_check_id] = (
            check_id_counts.get(normalized_check_id, 0) + 1
        )
    for check in checks:
        if not isinstance(check, dict):
            continue
        provider = check.get("provider")
        if not isinstance(provider, str) or not provider.strip():
            continue
        normalized_provider = provider.strip().lower()
        status = check.get("status")
        if not isinstance(status, str) or not status.strip():
            continue
        host = None
        host_value = check.get("host")
        if isinstance(host_value, str) and host_value.strip():
            host, _issue = host_identity_hostname(host_value, "host")
        explicit_host = host
        if host is None:
            for key in ("source_url", "url"):
                value = check.get(key)
                if isinstance(value, str) and value.strip():
                    host, _issue = absolute_https_hostname(value, key)
                    if host:
                        break
            explicit_host = host
        raw_check_id = check.get("id")
        explicit_model = check.get("model")
        model = explicit_model
        target_host = None
        target_model = None
        if (
            attestation_targets_directory is not None
            and isinstance(raw_check_id, str)
            and raw_check_id.strip()
        ):
            target_host = (
                attestation_targets_directory.provider_check_hosts.get(
                    normalized_provider, {}
                ).get(raw_check_id.strip().lower())
            )
            target_model = (
                attestation_targets_directory.provider_check_models.get(
                    normalized_provider, {}
                ).get(raw_check_id.strip().lower())
            )
        if (
            target_host is None
            and attestation_targets_directory is not None
            and isinstance(explicit_model, str)
            and explicit_model.strip()
        ):
            target_host = (
                attestation_targets_directory.provider_model_hosts.get(
                    normalized_provider, {}
                ).get(explicit_model.strip().lower())
            )
        if (
            not isinstance(model, str) or not model.strip()
        ) and attestation_targets_directory is not None:
            model = target_model
        if host is None and attestation_targets_directory is not None:
            host = target_host
        if host is None:
            continue
        if normalized_provider == "tinfoil" and is_tinfoil_router_identity(
            check_id=raw_check_id,
            host=host,
            label=check.get("label"),
        ):
            model = None
        if (
            normalized_provider == "ppq"
            and (not isinstance(model, str) or not model.strip())
            and attestation_targets_directory is not None
        ):
            check_hosts = attestation_targets_directory.provider_check_hosts.get(
                normalized_provider,
                {},
            )
            check_models = attestation_targets_directory.provider_check_models.get(
                normalized_provider,
                {},
            )
            candidate_models = {
                check_models[check_id]
                for check_id, check_host in check_hosts.items()
                if check_host == host and check_id in check_models
            }
            if len(candidate_models) == 1:
                model = next(iter(candidate_models))
        result_error = check.get("error")
        result_model_target_mismatch = (
            isinstance(explicit_model, str)
            and explicit_model.strip()
            and target_model is not None
            and explicit_model.strip().lower() != target_model
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
                    isinstance(explicit_model, str)
                    and explicit_model.strip()
                    and attestation_targets_directory is not None
                    and explicit_model.strip().lower()
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
        duplicate_result_check_id = False
        if isinstance(raw_check_id, str) and raw_check_id.strip():
            normalized_check_id = raw_check_id.strip().lower()
            duplicate_result_check_id = (
                provider_check_id_counts.get(normalized_provider, {}).get(
                    normalized_check_id, 0
                )
                > 1
            )
            if duplicate_result_check_id:
                status = "failed"
                result_error = (
                    f"duplicate attestation result rows for provider "
                    f"{normalized_provider} check {normalized_check_id}"
                )
        if normalized_provider == "tinfoil" and status.strip().lower() == "verified":
            if check.get("tls_matches") is not True:
                status = "failed"
                result_error = TINFOIL_TLS_BINDING_NOT_VERIFIED_ERROR
            else:
                release_digest = check.get("release_digest")
                if (
                    not isinstance(release_digest, str)
                    or not is_sha256_digest_value(release_digest)
                ):
                    status = "failed"
                    result_error = (
                        "release_digest is missing or invalid in "
                        "confidential-inference attestation results"
                    )
                for field in TINFOIL_ATTESTATION_RESULT_REQUIRED_TRUE_FIELDS:
                    if status.strip().lower() != "verified":
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
            and status.strip().lower() == "verified"
            and str(check.get("trust_tier", "")).strip().lower() != "app-e2ee"
        ):
            status = "failed"
            result_error = PRIVATEMODE_APP_E2EE_TIER_REQUIRED_ERROR
        if (
            normalized_provider == "privatemode"
            and status.strip().lower() == "verified"
            and isinstance(model, str)
            and model.strip()
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
            and status.strip().lower() == "verified"
            and isinstance(model, str)
            and model.strip()
        ):
            if (
                (not isinstance(raw_check_id, str) or not raw_check_id.strip())
                and attestation_targets_directory is not None
                and model.strip().lower()
                in attestation_targets_directory.provider_ambiguous_target_models.get(
                    normalized_provider, set()
                )
            ):
                status = "failed"
                result_error = (
                    "model has multiple confidential-inference attestation targets; "
                    "result row must include a check id"
                )
            expected_backend_host = None
            expected_backend_repo = None
            if attestation_targets_directory is not None and isinstance(
                raw_check_id, str
            ) and raw_check_id.strip():
                normalized_check_id = raw_check_id.strip().lower()
                expected_backend_host = (
                    attestation_targets_directory.provider_check_backend_hosts.get(
                        normalized_provider, {}
                    ).get(normalized_check_id)
                )
                expected_backend_repo = (
                    attestation_targets_directory.provider_check_backend_repos.get(
                        normalized_provider, {}
                    ).get(normalized_check_id)
                )
            if expected_backend_host is None and attestation_targets_directory is not None:
                expected_backend_host = (
                    attestation_targets_directory.provider_model_backend_hosts.get(
                        normalized_provider, {}
                    ).get(model.strip().lower())
                )
            if expected_backend_repo is None and attestation_targets_directory is not None:
                expected_backend_repo = (
                    attestation_targets_directory.provider_model_backend_repos.get(
                        normalized_provider, {}
                    ).get(model.strip().lower())
                )
            if expected_backend_host is not None:
                result_backend_host, _issue = host_identity_hostname(
                    check.get("backend_host"), "backend_host"
                )
                if result_backend_host != expected_backend_host:
                    status = "failed"
                    result_error = (
                        "backend_host does not match confidential-inference "
                        "attestation target"
                    )
            if status.strip().lower() == "verified" and expected_backend_repo is not None:
                result_backend_repo = check.get("backend_repo")
                if (
                    not isinstance(result_backend_repo, str)
                    or result_backend_repo.strip() != expected_backend_repo
                ):
                    status = "failed"
                    result_error = (
                        "backend_repo does not match confidential-inference "
                        "attestation target"
                    )
            for field in ("release_digest", "backend_release_digest"):
                if status.strip().lower() != "verified":
                    break
                field_value = check.get(field)
                if (
                    not isinstance(field_value, str)
                    or not is_sha256_digest_value(field_value)
                ):
                    status = "failed"
                    result_error = (
                        f"{field} is missing or invalid in "
                        "confidential-inference attestation results"
                    )
                    break
            if status.strip().lower() == "verified":
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
            and status.strip().lower() == "verified"
            and isinstance(model, str)
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
            status.strip().lower() == "verified"
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
        normalized_status = status.strip().lower()
        host_statuses = provider_host_statuses.setdefault(normalized_provider, {})
        model_scoped_ppq_result = (
            normalized_provider == "ppq"
            and isinstance(model, str)
            and bool(model.strip())
        )
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
        if isinstance(raw_check_id, str) and raw_check_id.strip():
            normalized_check_id = raw_check_id.strip().lower()
            check_statuses = provider_check_statuses.setdefault(
                normalized_provider, {}
            )
            if normalized_check_id in check_statuses:
                normalized_status = "failed"
                result_error = (
                    f"duplicate attestation result rows for provider "
                    f"{normalized_provider} check {normalized_check_id}"
                )
                host_statuses[host] = normalized_status
            check_statuses[normalized_check_id] = normalized_status
        if (
            isinstance(result_error, str)
            and result_error.strip()
            and (
                not model_scoped_ppq_result
                or duplicate_result_check_id
                or result_model_target_mismatch
                or result_host_target_mismatch
            )
        ):
            host_errors = provider_host_errors.setdefault(normalized_provider, {})
            host_errors[host] = result_error.strip()
            if isinstance(raw_check_id, str) and raw_check_id.strip():
                normalized_check_id = raw_check_id.strip().lower()
                provider_check_errors.setdefault(normalized_provider, {})[
                    normalized_check_id
                ] = result_error.strip()
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
            provider_host_details.setdefault(normalized_provider, {})[host] = details
            if isinstance(raw_check_id, str) and raw_check_id.strip():
                normalized_check_id = raw_check_id.strip().lower()
                provider_check_details.setdefault(normalized_provider, {})[
                    normalized_check_id
                ] = details
        if isinstance(model, str) and model.strip():
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
            if details:
                provider_model_details.setdefault(normalized_provider, {})[
                    normalized_model
                ] = details
            if isinstance(result_error, str) and result_error.strip():
                model_errors = provider_model_errors.setdefault(normalized_provider, {})
                model_errors[normalized_model] = result_error.strip()
    return AttestationResultsDirectory(
        provider_host_statuses=provider_host_statuses,
        provider_host_errors=provider_host_errors,
        provider_check_statuses=provider_check_statuses,
        provider_check_errors=provider_check_errors,
        provider_host_details=provider_host_details,
        provider_check_details=provider_check_details,
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
    if not privatemode_selected_model_ids_exactly_match(
        check.get("selected_model_ids"),
        model_id,
    ):
        return f"{label} selected_model_ids must exactly match the selected model"
    digest_keys = (
        "manifest_digest",
        "proxy_binary_digest",
        "coordinator_measurement",
        "secret_service_measurement",
        "ai_worker_measurement",
        "key_release_binding",
        *REQUIRED_PRIVATEMODE_RESULT_DIGEST_CLAIMS,
    )
    for key in digest_keys:
        value = check.get(key)
        if not isinstance(value, str) or not is_prefixed_sha256_digest_value(value):
            return f"{label} {key} must be a sha256 digest"
    verification_steps = check.get("verification_steps")
    if not isinstance(verification_steps, dict):
        return f"{label} verification_steps are required"
    for step in REQUIRED_PRIVATEMODE_RESULT_VERIFICATION_STEPS:
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


def provider_settings_source(document: dict[str, Any]) -> dict[str, Any]:
    source = document.get("provider_settings")
    if isinstance(source, str) and source.strip():
        try:
            source = json.loads(
                source,
                parse_constant=json_constant_rejecter("provider_settings JSON"),
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"provider_settings JSON is invalid: {exc}") from exc
    if not isinstance(source, dict):
        source = document
    return source


def provider_base_url_from_document(document: dict[str, Any]) -> str:
    sources = [document]
    provider_settings = provider_settings_source(document)
    if provider_settings is not document:
        sources.append(provider_settings)
    for source in sources:
        for key in ("base_url", "baseUrl"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def confidentiality_policy_from_document(
    document: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    source = provider_settings_source(document)

    confidentiality = source.get("confidentiality")
    if isinstance(confidentiality, dict):
        mode = policy_string(confidentiality, "mode")
        if not mode:
            mode = policy_string(source, "mode")
        policy = confidentiality.get("policy")
        if isinstance(policy, dict):
            return mode, policy
        return mode, confidentiality

    policy = source.get("policy")
    if isinstance(policy, dict):
        return policy_string(source, "mode"), policy
    return policy_string(source, "mode"), source


def confidentiality_mode_issue(document: dict[str, Any]) -> str | None:
    source = provider_settings_source(document)
    confidentiality = source.get("confidentiality")
    mode_source = confidentiality if isinstance(confidentiality, dict) else source
    if "mode" not in mode_source:
        return None
    value = mode_source.get("mode")
    if isinstance(value, str) and value.strip():
        return None
    return "mode must be a non-empty string"


def confidentiality_model_selectors_from_document(
    document: dict[str, Any],
) -> tuple[list[str], list[str]]:
    selector_source = confidentiality_model_selector_source(document)
    return (
        policy_string_values(selector_source, "model_ids"),
        policy_string_values(selector_source, "model_id_prefixes"),
    )


def confidentiality_model_selector_source(document: dict[str, Any]) -> dict[str, Any]:
    source = provider_settings_source(document)
    confidentiality = source.get("confidentiality")
    if isinstance(confidentiality, dict):
        return confidentiality
    return source


def runtime_policy_loadability_issues(
    document: dict[str, Any],
    *,
    mode: str,
    base_url: str,
) -> list[str]:
    provider_type = RUNTIME_PROVIDER_TYPE_BY_CONFIDENTIALITY_MODE.get(mode)
    if provider_type is None:
        return []
    if not base_url:
        return []
    try:
        provider_settings = provider_settings_source(document)
    except ValueError as exc:
        return [str(exc)]

    from routstr.upstream.base import ConfidentialVerifierPolicy

    try:
        policy = ConfidentialVerifierPolicy.from_provider_settings(
            provider_type=provider_type,
            base_url=base_url,
            provider_settings=provider_settings,
        )
    except Exception as exc:
        return [f"provider policy is not runtime-loadable: {exc}"]
    if policy is None:
        return ["provider policy is not runtime-loadable by Routstr runtime"]
    return []


def selector_list_issues(source: dict[str, Any], *keys: str) -> list[str]:
    issues: list[str] = []
    for key in keys:
        if key not in source:
            continue
        value = source[key]
        if isinstance(value, list):
            if any(not isinstance(item, str) or not item.strip() for item in value):
                issues.append(f"{key} must be a list of non-empty strings")
                continue
            normalized = [item.strip().lower() for item in value]
            if len(set(normalized)) != len(normalized):
                issues.append(f"{key} must not contain duplicates")
            continue
        issues.append(f"{key} must be a list of non-empty strings")
    return issues


def inline_secret_violations(value: Any, path: str = "$") -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            key_normalized = key_text.replace("-", "_").lower()
            item_path = f"{path}.{key_text}"
            if key_normalized in SECRET_POLICY_KEYS:
                violations.append(
                    f"inline secret-like policy key is not allowed: {item_path}"
                )
            violations.extend(inline_secret_violations(item, item_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            violations.extend(inline_secret_violations(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        stripped = value.strip()
        try:
            parsed = urlparse(stripped)
        except ValueError:
            parsed = None
        if parsed is not None:
            has_userinfo = (
                parsed.username is not None
                or parsed.password is not None
                or "@" in parsed.netloc
            )
            if parsed.scheme and parsed.netloc and has_userinfo:
                violations.append(
                    f"credential-bearing policy URL is not allowed: {path}"
                )
            if parsed.scheme and parsed.netloc:
                for raw_key, raw_value in parse_qsl(
                    parsed.query,
                    keep_blank_values=True,
                ):
                    if (
                        raw_key.replace("-", "_").lower() in SECRET_POLICY_KEYS
                        and raw_value
                    ):
                        violations.append(
                            "credential-bearing policy URL query is not allowed: "
                            f"{path}"
                        )
                        break
        if SECRET_LIKE_VALUE_PATTERN.search(stripped):
            violations.append(f"inline secret-like policy value is not allowed: {path}")
    return violations


def policy_string(policy: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = policy.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def tinfoil_transport_security_policy_issue(policy: dict[str, Any]) -> str | None:
    values = [
        policy[key]
        for key in TINFOIL_TRANSPORT_SECURITY_POLICY_KEYS
        if key in policy
    ]
    if not values:
        return None
    normalized_values: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            return "Tinfoil transport_security must be tls or ehbp"
        normalized = value.strip().lower()
        if normalized not in {"tls", "ehbp"}:
            return "Tinfoil transport_security must be tls or ehbp"
        normalized_values.append(normalized)
    if len(set(normalized_values)) > 1:
        return "Tinfoil transport_security aliases must match"
    return None


def tinfoil_tls_public_key_binding_required(
    policy: dict[str, Any],
    target_policy: dict[str, Any] | None = None,
) -> bool:
    for raw_policy in (target_policy or {}, policy):
        for key in TINFOIL_TRANSPORT_SECURITY_POLICY_KEYS:
            value = raw_policy.get(key)
            if isinstance(value, str) and value.strip().lower() == "ehbp":
                return False
    return True


def tinfoil_tls_binding_failure_error(error: object) -> bool:
    normalized = str(error or "").strip().lower()
    return (
        normalized == TINFOIL_TLS_BINDING_NOT_VERIFIED_ERROR.lower()
        or "tls certificate binding" in normalized
    )


def policy_string_values(policy: dict[str, Any], *keys: str) -> list[str]:
    for key in keys:
        value = policy.get(key)
        if isinstance(value, list):
            values = [
                item.strip() for item in value if isinstance(item, str) and item.strip()
            ]
            if values:
                return values
    return []


def normalized_string_policy_values(policy: dict[str, Any], *keys: str) -> list[str]:
    values: list[str] = []
    seen_values: set[str] = set()
    for value in policy_values_for_present_keys(policy, *keys):
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        normalized_key = normalized.lower()
        if normalized and normalized_key not in seen_values:
            seen_values.add(normalized_key)
            values.append(normalized)
    return sorted(values)


def string_list_policy_issues(policy: dict[str, Any], *keys: str) -> list[str]:
    issues: list[str] = []
    for key in keys:
        if key not in policy:
            continue
        value = policy[key]
        if not isinstance(value, list):
            issues.append(f"{key} must be a list of non-empty strings")
            continue
        if not value:
            issues.append(f"{key} must be a list of non-empty strings")
            continue
        if any(not isinstance(item, str) or not item.strip() for item in value):
            issues.append(f"{key} must be a list of non-empty strings")
            continue
        normalized = [item.strip() for item in value]
        if len({item.lower() for item in normalized}) != len(normalized):
            issues.append(f"{key} must not contain duplicates")
    return issues


def string_list_policy_alias_issues(
    policy: dict[str, Any],
    label: str,
    *keys: str,
) -> list[str]:
    present_values: list[list[str]] = []
    for key in keys:
        if key not in policy:
            continue
        values = normalized_string_policy_values(policy, key)
        if values:
            present_values.append(values)
    if not present_values:
        return []
    first = present_values[0]
    if any(values != first for values in present_values[1:]):
        return [f"{label} aliases must match"]
    return []


def non_empty_string_policy_issues(policy: dict[str, Any], *keys: str) -> list[str]:
    issues: list[str] = []
    for key in keys:
        if key not in policy:
            continue
        value = policy[key]
        if not isinstance(value, str) or not value.strip():
            issues.append(f"{key} must be a non-empty string")
    return issues


def string_policy_field_issues(policy: dict[str, Any], *keys: str) -> list[str]:
    issues: list[str] = []
    for key in keys:
        if key not in policy:
            continue
        value = policy[key]
        if not isinstance(value, str):
            issues.append(f"{key} must be a string")
            continue
        if not value.strip():
            issues.append(f"{key} must be a non-empty string")
    return issues


def policy_raw_values(policy: dict[str, Any], *keys: str) -> list[object]:
    for key in keys:
        if key not in policy:
            continue
        value = policy[key]
        if isinstance(value, list):
            return list(value)
        return [value]
    return []


def policy_values_for_present_keys(policy: dict[str, Any], *keys: str) -> list[object]:
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


def is_sha256_digest_value(value: str) -> bool:
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def is_template_placeholder_digest_value(value: str) -> bool:
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    return bool(digest) and len(digest) in {64, 96} and len(set(digest)) == 1


def placeholder_digest_violations(value: Any, path: str = "$") -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            violations.extend(placeholder_digest_violations(item, f"{path}.{key}"))
        return violations
    if isinstance(value, list):
        for index, item in enumerate(value):
            violations.extend(placeholder_digest_violations(item, f"{path}[{index}]"))
        return violations
    if isinstance(value, str) and is_template_placeholder_digest_value(value):
        violations.append(f"placeholder digest policy value is not allowed: {path}")
    return violations


def sha256_digest_hex(value: str) -> str:
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    return digest


def is_native_sev_snp_measurement_value(value: str) -> bool:
    digest = value.strip().lower()
    return (
        not digest.startswith("sha256:")
        and len(digest) == 96
        and all(char in "0123456789abcdef" for char in digest)
    )


def is_measurement_fingerprint_value(value: str) -> bool:
    return is_sha256_digest_value(value) or is_native_sev_snp_measurement_value(value)


def measurement_fingerprint_hex(value: str) -> str:
    digest = value.strip().lower()
    if is_sha256_digest_value(digest):
        return sha256_digest_hex(digest)
    if is_native_sev_snp_measurement_value(digest):
        return hashlib.sha256(bytes.fromhex(digest)).hexdigest()
    return digest


def normalized_sha256_digest_policy_values(
    policy: dict[str, Any],
    *keys: str,
) -> list[str]:
    values: list[str] = []
    for value in policy_values_for_present_keys(policy, *keys):
        if not isinstance(value, str) or not is_sha256_digest_value(value):
            continue
        digest = sha256_digest_hex(value)
        if digest not in values:
            values.append(digest)
    return values


def normalized_measurement_fingerprint_policy_values(
    policy: dict[str, Any],
    *keys: str,
) -> list[str]:
    values: list[str] = []
    for value in policy_values_for_present_keys(policy, *keys):
        if not isinstance(value, str) or not is_measurement_fingerprint_value(value):
            continue
        digest = measurement_fingerprint_hex(value)
        if digest not in values:
            values.append(digest)
    return values


def sha256_digest_policy_consistency_issue(
    policy: dict[str, Any],
    *,
    label: str,
    exact_keys: tuple[str, ...],
    allowed_keys: tuple[str, ...],
) -> str | None:
    exact_values = normalized_sha256_digest_policy_values(policy, *exact_keys)
    allowed_values = normalized_sha256_digest_policy_values(policy, *allowed_keys)
    if len(exact_values) > 1:
        return f"{label} expected aliases must match"
    if (
        exact_values
        and allowed_values
        and not set(exact_values).issubset(allowed_values)
    ):
        return f"{label} expected value must be allowed"
    return None


def measurement_fingerprint_policy_consistency_issue(
    policy: dict[str, Any],
    *,
    label: str,
    exact_keys: tuple[str, ...],
    allowed_keys: tuple[str, ...],
) -> str | None:
    exact_values = normalized_measurement_fingerprint_policy_values(
        policy,
        *exact_keys,
    )
    allowed_values = normalized_measurement_fingerprint_policy_values(
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


def is_prefixed_sha256_digest_value(value: str) -> bool:
    digest = value.strip().lower()
    if not digest.startswith("sha256:"):
        return False
    digest = digest.removeprefix("sha256:")
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def release_digest_policy_values(policy: dict[str, Any]) -> list[str]:
    values = policy_raw_values(policy, *RELEASE_DIGEST_POLICY_KEYS)
    return [
        value.strip()
        for value in values
        if isinstance(value, str) and is_sha256_digest_value(value)
    ]


def release_digest_policy_hex_values(policy: dict[str, Any]) -> set[str]:
    return {
        sha256_digest_hex(value)
        for value in release_digest_policy_values(policy)
    }


def result_sha256_digest_hex(value: object) -> str | None:
    if not isinstance(value, str) or not is_sha256_digest_value(value):
        return None
    return sha256_digest_hex(value)


def tinfoil_result_release_digest_issue(
    subject: str,
    details: dict[str, Any],
    policy: dict[str, Any],
) -> str | None:
    expected_digests = release_digest_policy_hex_values(policy)
    release_digest = result_sha256_digest_hex(details.get("release_digest"))
    if not release_digest:
        return f"{subject} attestation result is missing release_digest"
    if expected_digests and release_digest not in expected_digests:
        return (
            f"{subject} release digest sha256:{release_digest} does not match "
            "policy release digests"
        )
    return None


def release_digest_policy_issue(
    policy: dict[str, Any],
    *,
    missing_required: bool = False,
) -> str | None:
    values = policy_values_for_present_keys(policy, *RELEASE_DIGEST_POLICY_KEYS)
    if not values:
        if missing_required:
            return "expected_release_digest or allowed_release_digests is required"
        return None
    if any(
        not isinstance(value, str) or not is_sha256_digest_value(value)
        for value in values
    ):
        return "release digest policy values must be sha256 digests"
    return sha256_digest_policy_consistency_issue(
        policy,
        label="release_digest",
        exact_keys=RELEASE_DIGEST_EXACT_POLICY_KEYS,
        allowed_keys=RELEASE_DIGEST_ALLOWED_POLICY_KEYS,
    )


def measurement_digest_policy_issue(policy: dict[str, Any]) -> str | None:
    raw_values: list[object] = []
    for key in EHBP_MEASUREMENT_DIGEST_POLICY_KEYS:
        raw_values.extend(policy_raw_values(policy, key))
    if any(
        not isinstance(value, str) or not is_measurement_fingerprint_value(value)
        for value in raw_values
    ):
        return (
            "measurement fingerprint policy values must be sha256 digests "
            "or native SEV-SNP measurements"
        )
    for label, exact_keys, allowed_keys in EHBP_MEASUREMENT_DIGEST_POLICY_GROUPS:
        if issue := measurement_fingerprint_policy_consistency_issue(
            policy,
            label=label,
            exact_keys=exact_keys,
            allowed_keys=allowed_keys,
        ):
            return issue
    return None


def policy_bool_issue(policy: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key in policy and not isinstance(policy[key], bool):
            return f"{key} must be a boolean"
    return None


def policy_bool_enabled(policy: dict[str, Any], *keys: str) -> bool:
    return any(policy.get(key) is True for key in keys)


def tinfoil_model_attestation_target_values(
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
    issues: list[str] = []
    for key, raw_targets in present:
        if not isinstance(raw_targets, dict) or not raw_targets:
            issues.append(f"{key} must be a non-empty object")
            continue

        parsed: dict[str, dict[str, Any]] = {}
        seen_model_ids: set[str] = set()
        for raw_model_id, raw_target in raw_targets.items():
            if not isinstance(raw_model_id, str) or not raw_model_id.strip():
                issues.append(f"{key} keys must be non-empty model strings")
                continue
            model_id = raw_model_id.strip()
            normalized_model_id = model_id.lower()
            if normalized_model_id in seen_model_ids:
                issues.append(f"{key} contains duplicate model target {model_id}")
                continue
            seen_model_ids.add(normalized_model_id)
            if not isinstance(raw_target, dict):
                issues.append(f"{key} for {model_id} must be an object")
                continue
            target = dict(raw_target)
            target_issues = string_policy_field_issues(
                target,
                "host",
                "expected_enclave_host",
                "attestation_url",
                "hpke_keys_url",
                "repo",
                "expected_repo",
            )
            target_issues.extend(
                string_list_policy_alias_issues(
                    target,
                    "enclave host",
                    "host",
                    "expected_enclave_host",
                )
            )
            target_issues.extend(
                string_list_policy_alias_issues(
                    target,
                    "repo",
                    "repo",
                    "expected_repo",
                )
            )
            target_issues.extend(policy_url_issues(target, "attestation_url", "hpke_keys_url"))
            for host_key in ("host", "expected_enclave_host"):
                _host, issue = host_identity_hostname(target.get(host_key), host_key)
                if issue:
                    target_issues.append(issue)
                elif public_host_issue := public_attestation_target_host_issue(
                    target.get(host_key),
                    host_key,
                ):
                    target_issues.append(public_host_issue)
            for url_key in ("attestation_url", "hpke_keys_url"):
                if public_url_host_issue := public_attestation_target_host_issue(
                    target.get(url_key),
                    url_key,
                    absolute_url=True,
                ):
                    target_issues.append(public_url_host_issue)
            if not (
                policy_string(target, "host", "expected_enclave_host")
                or policy_string(target, "attestation_url")
            ):
                target_issues.append("host or attestation_url is required")
            if not policy_string(target, "repo", "expected_repo"):
                target_issues.append("repo or expected_repo is required")
            if release_issue := release_digest_policy_issue(target):
                target_issues.append(release_issue)
            if not release_digest_policy_values(target):
                target_issues.append(
                    "expected_release_digest or allowed_release_digests is required"
                )
            if measurement_issue := measurement_digest_policy_issue(target):
                target_issues.append(measurement_issue)
            if not has_artifact_identity_pin(target):
                target_issues.append(
                    "expected_release_digest, allowed_release_digests, or "
                    "code measurement identity pin is required"
                )
            if target_issues:
                issues.extend(
                    f"{key} for {model_id}: {issue}" for issue in target_issues
                )
                continue
            parsed[model_id] = target
        if parsed:
            parsed_values.append(dict(sorted(parsed.items())))

    if len(parsed_values) > 1:
        first = parsed_values[0]
        if any(value != first for value in parsed_values[1:]):
            issues.append("model_attestation_targets aliases must match")
    return (parsed_values[0] if parsed_values else {}, issues)


def tinfoil_model_target_origin(
    target: dict[str, Any],
) -> tuple[str | None, str | None]:
    for key in ("attestation_url", "hpke_keys_url"):
        origin, issue = absolute_https_origin(target.get(key), key)
        if issue:
            return None, issue
        if origin:
            return origin, None
    host, issue = host_identity_hostname(
        policy_string(target, "host", "expected_enclave_host"),
        "host",
    )
    if issue:
        return None, issue
    if host:
        return f"https://{host}", None
    return None, "host or attestation_url is required"


def tinfoil_router_origin_reuse_issues(
    targets: dict[str, dict[str, Any]],
    *,
    provider_base_url: str,
) -> list[str]:
    router_origin, provider_issue = absolute_https_origin(
        provider_base_url,
        "provider base_url",
    )
    if provider_issue or not router_origin:
        return []

    issues: list[str] = []
    for model_id, target in targets.items():
        target_origin, target_issue = tinfoil_model_target_origin(target)
        if target_issue or not target_origin:
            continue
        if target_origin == router_origin:
            issues.append(
                f"model_attestation_targets for {model_id} must use a distinct "
                "model enclave origin, not the Tinfoil router origin"
            )
    return issues


def tinfoil_attestation_directory_target_issues(
    targets: dict[str, dict[str, Any]],
    *,
    attestation_targets_directory: AttestationTargetsDirectory | None,
    strict_attestation_targets: bool,
) -> tuple[list[str], list[str]]:
    if not attestation_targets_directory:
        return [], []
    known_hosts = attestation_targets_directory.provider_hosts.get("tinfoil") or set()
    known_host_models = (
        attestation_targets_directory.provider_host_model_ids.get("tinfoil") or {}
    )
    known_host_repos = (
        attestation_targets_directory.provider_host_repos.get("tinfoil") or {}
    )
    if not known_hosts:
        message = "attestation targets for provider tinfoil are not present"
        return ([message], []) if strict_attestation_targets else ([], [message])

    issues: list[str] = []
    warnings: list[str] = []
    for model_id, target in sorted(targets.items()):
        host, host_issue = tinfoil_model_target_origin(target)
        if host_issue or not host:
            continue
        parsed = urlparse(host)
        normalized_host = normalized_hostname(parsed.hostname)
        if normalized_host in known_hosts:
            model_candidates = known_host_models.get(normalized_host) or set()
            selected_model_aliases = provider_catalog_identity_candidate_variants(
                model_id.split("/")[-1],
                provider_slug="tinfoil",
            )
            if model_candidates and not selected_model_aliases.intersection(
                model_candidates
            ):
                message = (
                    f"model_attestation_targets for {model_id} host {normalized_host} "
                    "is listed for a different model in attestation targets for "
                    "provider tinfoil"
                )
            elif known_host_repos.get(normalized_host):
                target_repo = target.get("repo")
                if not isinstance(target_repo, str) or not target_repo.strip():
                    message = (
                        f"model_attestation_targets for {model_id} host "
                        f"{normalized_host} repo is required by attestation "
                        "targets for provider tinfoil"
                    )
                elif target_repo.strip() not in known_host_repos[normalized_host]:
                    message = (
                        f"model_attestation_targets for {model_id} host "
                        f"{normalized_host} repo {target_repo.strip()} does not "
                        "match attestation targets for provider tinfoil"
                    )
                else:
                    continue
            else:
                continue
        else:
            message = (
                f"model_attestation_targets for {model_id} host {normalized_host} "
                "is not present in attestation targets for provider tinfoil"
            )
        if strict_attestation_targets:
            issues.append(message)
        else:
            warnings.append(message)
    return issues, warnings


def tinfoil_attestation_results_target_issues(
    targets: dict[str, dict[str, Any]],
    *,
    policy: dict[str, Any],
    attestation_results_directory: AttestationResultsDirectory | None,
    strict_attestation_results: bool,
) -> tuple[list[str], list[str]]:
    if not attestation_results_directory:
        return [], []
    host_statuses = (
        attestation_results_directory.provider_host_statuses.get("tinfoil") or {}
    )
    host_errors = (
        attestation_results_directory.provider_host_errors.get("tinfoil") or {}
    )
    host_details = (
        attestation_results_directory.provider_host_details.get("tinfoil") or {}
    )
    if not host_statuses:
        message = "attestation results for provider tinfoil are not present"
        return ([message], []) if strict_attestation_results else ([], [message])

    issues: list[str] = []
    warnings: list[str] = []
    for model_id, target in sorted(targets.items()):
        host, host_issue = tinfoil_model_target_origin(target)
        if host_issue or not host:
            continue
        parsed = urlparse(host)
        normalized_host = normalized_hostname(parsed.hostname)
        status = host_statuses.get(normalized_host)
        if status == "verified":
            release_issue = tinfoil_result_release_digest_issue(
                f"model_attestation_targets for {model_id}",
                host_details.get(normalized_host) or {},
                target,
            )
            if release_issue:
                if strict_attestation_results:
                    issues.append(release_issue)
                else:
                    warnings.append(release_issue)
            continue
        error = host_errors.get(normalized_host)
        if (
            status == "failed"
            and tinfoil_tls_binding_failure_error(error)
            and not tinfoil_tls_public_key_binding_required(policy, target)
        ):
            release_issue = tinfoil_result_release_digest_issue(
                f"model_attestation_targets for {model_id}",
                host_details.get(normalized_host) or {},
                target,
            )
            if release_issue:
                if strict_attestation_results:
                    issues.append(release_issue)
                else:
                    warnings.append(release_issue)
            continue
        if status is None:
            message = (
                f"model_attestation_targets for {model_id} host {normalized_host} "
                "is not present in confidential-inference attestation results"
            )
        else:
            suffix = f": {error}" if error else ""
            message = (
                f"model_attestation_targets for {model_id} host {normalized_host} "
                "is not verified in confidential-inference attestation results"
                f" (status={status}){suffix}"
            )
        if strict_attestation_results:
            issues.append(message)
        else:
            warnings.append(message)
    return issues, warnings


def tinfoil_provider_result_hosts(
    policy: dict[str, Any],
    provider_base_url: str,
) -> set[str]:
    hosts: set[str] = set()
    for key in ("attestation_url", "hpke_keys_url"):
        host, _issue = absolute_https_hostname(policy.get(key), key)
        if host:
            hosts.add(host)
    for key in ("enclave_host", "expected_enclave_host"):
        host, _issue = host_identity_hostname(policy.get(key), key)
        if host:
            hosts.add(host)
    host, _issue = absolute_https_hostname(provider_base_url, "provider base_url")
    if host:
        hosts.add(host)
    return hosts


def tinfoil_provider_attestation_results_issues(
    policy: dict[str, Any],
    *,
    provider_base_url: str,
    attestation_results_directory: AttestationResultsDirectory | None,
    strict_attestation_results: bool,
) -> tuple[list[str], list[str]]:
    if not attestation_results_directory:
        return [], []
    host_statuses = (
        attestation_results_directory.provider_host_statuses.get("tinfoil") or {}
    )
    host_errors = (
        attestation_results_directory.provider_host_errors.get("tinfoil") or {}
    )
    host_details = (
        attestation_results_directory.provider_host_details.get("tinfoil") or {}
    )
    if not host_statuses:
        message = "attestation results for provider tinfoil are not present"
        return ([message], []) if strict_attestation_results else ([], [message])

    issues: list[str] = []
    warnings: list[str] = []
    for host in sorted(tinfoil_provider_result_hosts(policy, provider_base_url)):
        status = host_statuses.get(host)
        if status == "verified":
            release_issue = tinfoil_result_release_digest_issue(
                f"Tinfoil provider host {host}",
                host_details.get(host) or {},
                policy,
            )
            if release_issue:
                if strict_attestation_results:
                    issues.append(release_issue)
                else:
                    warnings.append(release_issue)
            continue
        error = host_errors.get(host)
        if (
            status == "failed"
            and tinfoil_tls_binding_failure_error(error)
            and not tinfoil_tls_public_key_binding_required(policy)
        ):
            release_issue = tinfoil_result_release_digest_issue(
                f"Tinfoil provider host {host}",
                host_details.get(host) or {},
                policy,
            )
            if release_issue:
                if strict_attestation_results:
                    issues.append(release_issue)
                else:
                    warnings.append(release_issue)
            continue
        if status is None:
            message = (
                f"Tinfoil provider host {host} is not present in "
                "confidential-inference attestation results"
            )
        else:
            suffix = f": {error}" if error else ""
            message = (
                f"Tinfoil provider host {host} is not verified in "
                "confidential-inference attestation results"
                f" (status={status}){suffix}"
            )
        if strict_attestation_results:
            issues.append(message)
        else:
            warnings.append(message)
    return issues, warnings


def ppq_private_provider_result_hosts(
    policy: dict[str, Any],
    provider_base_url: str,
) -> set[str]:
    hosts: set[str] = set()
    for key in ("attestation_bundle_url", "hpke_keys_url"):
        host, _issue = absolute_https_hostname(policy.get(key), key)
        if host:
            hosts.add(host)
    for key in ("enclave_host", "expected_enclave_host"):
        host, _issue = host_identity_hostname(policy.get(key), key)
        if host:
            hosts.add(host)
    host, _issue = absolute_https_hostname(provider_base_url, "provider base_url")
    if host:
        hosts.add(host)
    return hosts


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


def ppq_private_attestation_results_issues(
    policy: dict[str, Any],
    *,
    provider_base_url: str,
    model_ids: list[str],
    attestation_targets_directory: AttestationTargetsDirectory | None,
    attestation_results_directory: AttestationResultsDirectory | None,
    strict_attestation_results: bool,
) -> tuple[list[str], list[str]]:
    if not attestation_results_directory:
        return [], []
    host_statuses = (
        attestation_results_directory.provider_host_statuses.get("ppq") or {}
    )
    if not host_statuses:
        message = "attestation results for provider ppq are not present"
        return ([message], []) if strict_attestation_results else ([], [message])

    issues: list[str] = []
    warnings: list[str] = []
    model_statuses = (
        attestation_results_directory.provider_model_statuses.get("ppq") or {}
    )
    model_errors = attestation_results_directory.provider_model_errors.get("ppq") or {}
    target_models = target_directory_ppq_models(attestation_targets_directory)
    selected_models = {
        model.strip().lower() for model in model_ids if model.strip()
    }
    model_rows_are_authoritative = bool(
        selected_models or model_statuses or target_models
    )
    if model_rows_are_authoritative:
        router_release_digests = release_digest_policy_hex_values(policy)
        model_targets, model_target_issues = tinfoil_model_attestation_target_values(
            policy
        )
        model_details = (
            attestation_results_directory.provider_model_details.get("ppq") or {}
        )
        for model_id in sorted(selected_models):
            status = model_statuses.get(model_id)
            if status == "verified":
                details = model_details.get(model_id) or {}
                router_digest = result_sha256_digest_hex(
                    details.get("release_digest")
                )
                if not router_digest:
                    message = (
                        f"PPQ private model {model_id} attestation result is "
                        "missing router release_digest"
                    )
                    if strict_attestation_results:
                        issues.append(message)
                    else:
                        warnings.append(message)
                elif (
                    router_release_digests
                    and router_digest not in router_release_digests
                ):
                    message = (
                        f"PPQ private model {model_id} router release digest "
                        f"sha256:{router_digest} does not match policy release digests"
                    )
                    if strict_attestation_results:
                        issues.append(message)
                    else:
                        warnings.append(message)

                backend_digest = result_sha256_digest_hex(
                    details.get("backend_release_digest")
                )
                target_release_digests = (
                    release_digest_policy_hex_values(model_targets.get(model_id, {}))
                    if not model_target_issues
                    else set()
                )
                if not backend_digest:
                    message = (
                        f"PPQ private model {model_id} attestation result is "
                        "missing backend_release_digest"
                    )
                    if strict_attestation_results:
                        issues.append(message)
                    else:
                        warnings.append(message)
                elif (
                    target_release_digests
                    and backend_digest not in target_release_digests
                ):
                    message = (
                        f"PPQ private model {model_id} backend release digest "
                        f"sha256:{backend_digest} does not match "
                        "model_attestation_targets release digests"
                    )
                    if strict_attestation_results:
                        issues.append(message)
                    else:
                        warnings.append(message)
                for field_name in (
                    PPQ_PRIVATE_BACKEND_ATTESTATION_RESULT_REQUIRED_TRUE_FIELDS
                ):
                    if details.get(field_name) is True:
                        continue
                    message = (
                        f"PPQ private model {model_id} attestation result requires "
                        f"{field_name}=true"
                    )
                    if strict_attestation_results:
                        issues.append(message)
                    else:
                        warnings.append(message)
                continue
            if status is None:
                message = (
                    f"PPQ private model {model_id} is not present in "
                    "confidential-inference attestation results"
                )
            else:
                error = model_errors.get(model_id)
                suffix = f": {error}" if error else ""
                message = (
                    f"PPQ private model {model_id} is not verified in "
                    "confidential-inference attestation results"
                    f" (status={status}){suffix}"
                )
            if strict_attestation_results:
                issues.append(message)
            else:
                warnings.append(message)
    if model_rows_are_authoritative:
        return issues, warnings
    message = (
        "PPQ private selected model evidence is required in "
        "confidential-inference attestation results"
    )
    if strict_attestation_results:
        issues.append(message)
    else:
        warnings.append(message)
    return issues, warnings


def privatemode_provider_result_hosts(policy: dict[str, Any]) -> set[str]:
    hosts: set[str] = set()
    for key in (
        "api_base_url",
        "apiBaseURL",
        "cdn_base_url",
        "cdnBaseURL",
        "manifest_url",
        "manifestURL",
        "proxy_configuration_url",
        "proxyConfigurationURL",
        "source_verification_url",
        "sourceVerificationURL",
        "model_verification_url",
        "modelVerificationURL",
    ):
        host, _issue = absolute_https_hostname(policy.get(key), key)
        if host:
            hosts.add(host)
    return hosts


def sha256_json_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def privatemode_expected_workload_identity_digest(
    policy: dict[str, Any],
) -> str | None:
    workload_sans = normalized_string_policy_values(
        policy,
        *PRIVATEMODE_EXPECTED_WORKLOAD_SAN_POLICY_KEYS,
    )
    workload_ids = normalized_string_policy_values(
        policy,
        *PRIVATEMODE_EXPECTED_WORKLOAD_ID_POLICY_KEYS,
    )
    if not workload_sans and not workload_ids:
        return None
    return sha256_json_digest({"ids": workload_ids, "sans": workload_sans})


def privatemode_model_workload_binding_digest(
    policy: dict[str, Any],
) -> str | None:
    bindings, issues = privatemode_model_workload_binding_values(policy)
    if issues or not bindings:
        return None
    return sha256_json_digest(bindings)


def privatemode_selected_model_ids_exactly_match(
    value: object,
    model_id: str,
) -> bool:
    if not isinstance(value, list) or len(value) != 1:
        return False
    selected = value[0]
    return isinstance(selected, str) and selected.strip().lower() == model_id


def privatemode_result_policy_bound_issues(
    policy: dict[str, Any],
    *,
    model_id: str,
    details: dict[str, Any],
) -> list[str]:
    if not details:
        return [
            f"Privatemode model {model_id} attestation results are missing "
            "policy-bound proof details"
        ]

    issues: list[str] = []
    core_detail_keys = {
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
    if any(key not in details for key in core_detail_keys):
        issues.append(
            f"Privatemode model {model_id} attestation results are missing "
            "policy-bound proof details"
        )
    if details.get("transport") != "privatemode-proxy":
        issues.append(
            f"Privatemode model {model_id} attestation results transport "
            "must be privatemode-proxy"
        )
    if details.get("trust_tier") != "app-e2ee":
        issues.append(
            f"Privatemode model {model_id} attestation results trust_tier "
            "must be app-e2ee"
        )

    for label, policy_keys in (
        ("manifest_digest", PRIVATEMODE_MANIFEST_DIGEST_POLICY_KEYS),
        ("proxy_binary_digest", PRIVATEMODE_PROXY_BINARY_DIGEST_POLICY_KEYS),
    ):
        expected = normalized_sha256_digest_policy_values(policy, *policy_keys)
        actual = details.get(label)
        if expected and (
            not isinstance(actual, str)
            or not is_sha256_digest_value(actual)
            or sha256_digest_hex(actual) not in expected
        ):
            issues.append(
                f"Privatemode model {model_id} attestation results {label} "
                "does not match policy"
            )

    for claim_name, policy_keys in PRIVATEMODE_COMPONENT_DIGEST_POLICY_KEYS.items():
        expected = normalized_sha256_digest_policy_values(policy, *policy_keys)
        actual = details.get(claim_name)
        if expected and (
            not isinstance(actual, str)
            or not is_sha256_digest_value(actual)
            or sha256_digest_hex(actual) not in expected
        ):
            issues.append(
                f"Privatemode model {model_id} attestation results {claim_name} "
                "does not match policy"
            )

    for claim_name, policy_keys in PRIVATEMODE_COMPONENT_STRING_POLICY_KEYS.items():
        expected = normalized_string_policy_values(policy, *policy_keys)
        actual = details.get(claim_name)
        if expected and (
            not isinstance(actual, str) or actual.strip() not in expected
        ):
            issues.append(
                f"Privatemode model {model_id} attestation results {claim_name} "
                "does not match policy"
            )

    if not privatemode_selected_model_ids_exactly_match(
        details.get("selected_model_ids"),
        model_id,
    ):
        issues.append(
            f"Privatemode model {model_id} attestation results selected_model_ids "
            "must exactly match the selected model"
        )

    expected_workload_digest = privatemode_expected_workload_identity_digest(policy)
    if expected_workload_digest is not None and (
        details.get("expected_workload_identity_digest") != expected_workload_digest
    ):
        issues.append(
            f"Privatemode model {model_id} attestation results expected workload "
            "identity digest does not match policy"
        )

    model_workload_digest = privatemode_model_workload_binding_digest(policy)
    if model_workload_digest is not None and (
        details.get("model_workload_binding_digest") != model_workload_digest
    ):
        issues.append(
            f"Privatemode model {model_id} attestation results model workload "
            "binding digest does not match policy"
        )

    for claim_name in REQUIRED_PRIVATEMODE_RESULT_DIGEST_CLAIMS:
        value = details.get(claim_name)
        if not isinstance(value, str) or not is_prefixed_sha256_digest_value(value):
            issues.append(
                f"Privatemode model {model_id} attestation results {claim_name} "
                "must be a sha256 digest"
            )

    verification_steps = details.get("verification_steps")
    if not isinstance(verification_steps, dict):
        issues.append(
            f"Privatemode model {model_id} attestation results verification_steps "
            "are required"
        )
    else:
        for step in REQUIRED_PRIVATEMODE_RESULT_VERIFICATION_STEPS:
            if verification_steps.get(step) is not True:
                issues.append(
                    f"Privatemode model {model_id} attestation results "
                    f"verification step {step} must be true"
                )
    return issues


def privatemode_attestation_results_issues(
    policy: dict[str, Any],
    *,
    model_ids: list[str],
    attestation_results_directory: AttestationResultsDirectory | None,
    strict_attestation_results: bool,
) -> tuple[list[str], list[str]]:
    if not attestation_results_directory:
        return [], []
    host_statuses = (
        attestation_results_directory.provider_host_statuses.get("privatemode") or {}
    )
    host_errors = (
        attestation_results_directory.provider_host_errors.get("privatemode") or {}
    )
    model_statuses = (
        attestation_results_directory.provider_model_statuses.get("privatemode") or {}
    )
    model_errors = (
        attestation_results_directory.provider_model_errors.get("privatemode") or {}
    )
    model_details = (
        attestation_results_directory.provider_model_details.get("privatemode") or {}
    )
    hosts = privatemode_provider_result_hosts(policy)
    if hosts and not host_statuses:
        message = "attestation results for provider privatemode are not present"
        return ([message], []) if strict_attestation_results else ([], [message])

    issues: list[str] = []
    warnings: list[str] = []
    for host in sorted(hosts):
        status = host_statuses.get(host)
        if status == "verified":
            continue
        if status is None:
            message = (
                f"Privatemode provider host {host} is not present in "
                "confidential-inference attestation results"
            )
        else:
            error = host_errors.get(host)
            suffix = f": {error}" if error else ""
            message = (
                f"Privatemode provider host {host} is not verified in "
                "confidential-inference attestation results"
                f" (status={status}){suffix}"
            )
        if strict_attestation_results:
            issues.append(message)
        else:
            warnings.append(message)
    for model_id in sorted({model.strip().lower() for model in model_ids if model.strip()}):
        status = model_statuses.get(model_id)
        if status == "verified":
            proof_issues = privatemode_result_policy_bound_issues(
                policy,
                model_id=model_id,
                details=model_details.get(model_id) or {},
            )
            if strict_attestation_results:
                issues.extend(proof_issues)
            else:
                warnings.extend(proof_issues)
            continue
        if status is None:
            message = (
                f"Privatemode model {model_id} is not present in "
                "confidential-inference attestation results"
            )
        else:
            error = model_errors.get(model_id)
            suffix = f": {error}" if error else ""
            message = (
                f"Privatemode model {model_id} is not verified in "
                "confidential-inference attestation results"
                f" (status={status}){suffix}"
            )
        if strict_attestation_results:
            issues.append(message)
        else:
            warnings.append(message)
    return issues, warnings


def tinfoil_model_attestation_policy_issues(
    policy: dict[str, Any],
    *,
    model_ids: list[str],
    model_id_prefixes: list[str],
    provider_base_url: str = "",
    attestation_targets_directory: AttestationTargetsDirectory | None = None,
    strict_attestation_targets: bool = False,
    attestation_results_directory: AttestationResultsDirectory | None = None,
    strict_attestation_results: bool = False,
    attestation_target_warnings: list[str] | None = None,
) -> list[str]:
    issues: list[str] = []
    if bool_issue := policy_bool_issue(
        policy,
        *TINFOIL_REQUIRE_MODEL_ATTESTATION_POLICY_KEYS,
    ):
        issues.append(bool_issue)
        return issues
    if not policy_bool_enabled(policy, *TINFOIL_REQUIRE_MODEL_ATTESTATION_POLICY_KEYS):
        issues.append(
            "Tinfoil require_model_attestations must be true for selected model routing"
        )
        return issues
    if model_id_prefixes:
        issues.append(
            "Tinfoil model_id_prefixes are not supported with "
            "full model attestations; use exact model_ids"
        )
    if not model_ids:
        issues.append("Tinfoil model_ids are required with full model attestations")

    targets, target_issues = tinfoil_model_attestation_target_values(policy)
    issues.extend(target_issues)
    if target_issues:
        return issues

    selected_models = set(model_ids)
    for model_id in sorted(selected_models):
        if model_id not in targets:
            issues.append(f"model_attestation_targets is missing {model_id}")
    for model_id in targets:
        if model_id not in selected_models:
            issues.append(f"model_attestation_targets contains unselected model {model_id}")
    issues.extend(
        tinfoil_router_origin_reuse_issues(
            targets,
            provider_base_url=provider_base_url,
        )
    )
    directory_issues, _warnings = tinfoil_attestation_directory_target_issues(
        targets,
        attestation_targets_directory=attestation_targets_directory,
        strict_attestation_targets=strict_attestation_targets,
    )
    issues.extend(directory_issues)
    if attestation_target_warnings is not None:
        attestation_target_warnings.extend(_warnings)
    result_issues, result_warnings = tinfoil_attestation_results_target_issues(
        targets,
        policy=policy,
        attestation_results_directory=attestation_results_directory,
        strict_attestation_results=strict_attestation_results,
    )
    issues.extend(result_issues)
    if attestation_target_warnings is not None:
        attestation_target_warnings.extend(result_warnings)
    return issues


def provider_attestation_directory_warnings(
    mode: str,
    attestation_targets_directory: AttestationTargetsDirectory | None,
) -> list[str]:
    if attestation_targets_directory is None:
        return []
    if mode == "privatemode":
        return [
            "confidential-inference attestation targets contain Privatemode "
            "reference metadata only; Privatemode still requires live proxy "
            "Contrast attestation evidence"
        ]
    return []


def provider_catalog_slug_for_mode(mode: str) -> str | None:
    if mode == "ppq-private-tee":
        return "ppq"
    if mode in {"tinfoil", "privatemode"}:
        return mode
    return None


def provider_catalog_model_issues(
    *,
    mode: str,
    model_ids: list[str],
    provider_model_catalog: ProviderModelCatalog | None,
    strict_provider_catalog: bool,
) -> tuple[list[str], list[str]]:
    if provider_model_catalog is None or not model_ids:
        return [], []
    provider_slug = provider_catalog_slug_for_mode(mode)
    if provider_slug is None:
        return [], []
    catalog_models = provider_model_catalog.provider_models.get(provider_slug)
    if not catalog_models:
        message = (
            "confidential-inference provider catalog does not include models for "
            f"{provider_slug}"
        )
        return ([message], []) if strict_provider_catalog else ([], [message])

    issues: list[str] = []
    warnings: list[str] = []
    for model_id in model_ids:
        selected = provider_catalog_identity_candidate_variants(
            model_id.split("/")[-1],
            provider_slug=provider_slug,
        )
        if selected.intersection(catalog_models):
            continue
        message = (
            f"selected model {model_id} is not present in "
            f"confidential-inference provider catalog for {provider_slug}"
        )
        if strict_provider_catalog:
            issues.append(message)
        else:
            warnings.append(message)
    return issues, warnings


def ppq_private_model_selector_policy_issues(
    *,
    model_ids: list[str],
    model_id_prefixes: list[str],
) -> list[str]:
    issues: list[str] = []
    if model_id_prefixes:
        issues.append(
            "PPQ private model_id_prefixes are not supported; use exact model_ids"
        )
    if not model_ids:
        issues.append("PPQ private model_ids are required")
        return issues
    if any(not model_id.startswith("private/") for model_id in model_ids):
        issues.append("PPQ private model_ids must start with private/")
    return issues


def ppq_private_backend_model_attestation_policy_issues(
    policy: dict[str, Any],
    *,
    model_ids: list[str],
) -> list[str]:
    issues: list[str] = []
    if bool_issue := policy_bool_issue(
        policy,
        *TINFOIL_REQUIRE_MODEL_ATTESTATION_POLICY_KEYS,
    ):
        issues.append(bool_issue)
        return issues
    if not policy_bool_enabled(policy, *TINFOIL_REQUIRE_MODEL_ATTESTATION_POLICY_KEYS):
        issues.append(
            "PPQ private require_model_attestations must be true so the "
            "mapped downstream model enclave is verified"
        )
        return issues

    targets, target_issues = tinfoil_model_attestation_target_values(policy)
    issues.extend(target_issues)
    if target_issues:
        return issues

    selected_models = set(model_ids)
    for model_id in sorted(selected_models):
        if model_id not in targets:
            issues.append(f"model_attestation_targets is missing {model_id}")
    for model_id in targets:
        if model_id not in selected_models:
            issues.append(f"model_attestation_targets contains unselected model {model_id}")
    return issues


def verifier_digest_policy_issue(policy: dict[str, Any]) -> str | None:
    valid_digest_found = False
    for key in VERIFIER_DIGEST_POLICY_KEYS:
        if key not in policy:
            continue
        value = policy[key]
        if (
            not isinstance(value, str)
            or not value.strip()
            or not is_prefixed_sha256_digest_value(value)
        ):
            return f"{key} must be a sha256 digest"
        valid_digest_found = True
    if not valid_digest_found:
        return "verifier_command_digest or verifier_binary_digest is required"
    return None


def verifier_digest_policy_value(policy: dict[str, Any]) -> tuple[str | None, str | None]:
    values: list[tuple[str, str]] = []
    for key in VERIFIER_DIGEST_POLICY_KEYS:
        if key not in policy:
            continue
        value = policy[key]
        if (
            not isinstance(value, str)
            or not value.strip()
            or not is_prefixed_sha256_digest_value(value)
        ):
            return None, f"{key} must be a sha256 digest"
        values.append((key, value.strip().lower()))
    if not values:
        return None, "verifier_command_digest or verifier_binary_digest is required"
    unique = {value for _, value in values}
    if len(unique) > 1:
        return None, "verifier_command_digest aliases must match"
    return values[0][1], None


def resolve_verifier_artifact_path(
    *,
    policy: dict[str, Any],
    command_argv: list[str],
    policy_path: Path,
) -> Path:
    configured_artifact_path = policy.get("verifier_artifact_path")
    raw_artifact_path = (
        configured_artifact_path.strip()
        if isinstance(configured_artifact_path, str) and configured_artifact_path.strip()
        else command_argv[0]
    )
    artifact_path = Path(raw_artifact_path)
    if artifact_path.is_absolute():
        return artifact_path
    if os.sep in raw_artifact_path or (os.altsep and os.altsep in raw_artifact_path):
        policy_relative = policy_path.parent / artifact_path
        if policy_relative.exists():
            return policy_relative
        root_relative = ROOT / artifact_path
        if root_relative.exists():
            return root_relative
        return policy_relative
    resolved = shutil.which(raw_artifact_path)
    if resolved:
        return Path(resolved)
    return policy_path.parent / artifact_path


def verifier_artifact_digest_policy_issue(
    policy: dict[str, Any],
    policy_path: Path,
    *command_keys: str,
) -> str | None:
    present_commands: list[tuple[str, tuple[str, ...]]] = []
    for key in command_keys:
        if key not in policy:
            continue
        try:
            present_commands.append(
                (key, tuple(command_argv(policy[key], label=key)))
            )
        except ValueError as exc:
            return str(exc)
    if not present_commands:
        return f"{' or '.join(command_keys)} is required"
    if len({argv for _, argv in present_commands}) > 1:
        return "verifier_command aliases must match"
    expected_digest, digest_issue = verifier_digest_policy_value(policy)
    if digest_issue:
        return digest_issue
    assert expected_digest is not None

    artifact_path = resolve_verifier_artifact_path(
        policy=policy,
        command_argv=list(present_commands[0][1]),
        policy_path=policy_path,
    )
    try:
        actual_digest = "sha256:" + hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    except Exception as exc:
        return (
            f"failed to hash verifier artifact: {type(exc).__name__}: {exc}"
        )
    if actual_digest != expected_digest:
        return (
            "verifier command digest mismatch: "
            f"expected {expected_digest}, got {actual_digest}"
        )
    return None


def prefixed_digest_policy_issues(
    policy: dict[str, Any],
    *keys: str,
) -> list[str]:
    issues: list[str] = []
    for key in keys:
        if key not in policy:
            continue
        value = policy[key]
        if not isinstance(value, str) or not is_prefixed_sha256_digest_value(value):
            issues.append(f"{key} must be a sha256 digest")
    return issues


def prefixed_digest_alias_policy_issues(
    policy: dict[str, Any],
    *alias_groups: tuple[str, ...],
) -> list[str]:
    issues: list[str] = []
    for keys in alias_groups:
        label = keys[0]
        values = [policy[key] for key in keys if key in policy]
        if any(
            not isinstance(value, str) or not is_prefixed_sha256_digest_value(value)
            for value in values
        ):
            issues.append(f"{label} must be a sha256 digest")
            continue
        normalized = {value.strip().lower() for value in values}
        if len(normalized) > 1:
            issues.append(f"{label} aliases must match")
    return issues


def privatemode_component_policy_issues(policy: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for claim_name, keys in PRIVATEMODE_COMPONENT_DIGEST_POLICY_KEYS.items():
        values = policy_values_for_present_keys(policy, *keys)
        if any(
            not isinstance(value, str) or not is_prefixed_sha256_digest_value(value)
            for value in values
        ):
            issues.append(f"{claim_name} policy values must be sha256 digests")
    for claim_name, keys in PRIVATEMODE_COMPONENT_STRING_POLICY_KEYS.items():
        values = policy_values_for_present_keys(policy, *keys)
        if any(not isinstance(value, str) or not value.strip() for value in values):
            issues.append(
                f"{claim_name} policy values must contain only non-empty strings"
            )
    trust_tiers = normalized_string_policy_values(policy, "expected_trust_tier")
    if trust_tiers != ["app-e2ee"]:
        issues.append("expected_trust_tier must be app-e2ee")
    gpu_policies = normalized_string_policy_values(
        policy,
        "expected_gpu_attestation_policy",
    )
    if gpu_policies != [PRIVATEMODE_GPU_ATTESTATION_POLICY]:
        issues.append(
            "expected_gpu_attestation_policy must be "
            f"{PRIVATEMODE_GPU_ATTESTATION_POLICY}"
        )
    issues.extend(privatemode_component_policy_consistency_issues(policy))
    return issues


def privatemode_workload_policy_issues(policy: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    issues.extend(
        string_list_policy_issues(
            policy,
            *PRIVATEMODE_EXPECTED_WORKLOAD_POLICY_KEYS,
        )
    )
    issues.extend(
        string_list_policy_alias_issues(
            policy,
            "expected_workload_sans",
            *PRIVATEMODE_EXPECTED_WORKLOAD_SAN_POLICY_KEYS,
        )
    )
    issues.extend(
        string_list_policy_alias_issues(
            policy,
            "expected_workload_ids",
            *PRIVATEMODE_EXPECTED_WORKLOAD_ID_POLICY_KEYS,
        )
    )
    if not normalized_string_policy_values(
        policy,
        *PRIVATEMODE_EXPECTED_WORKLOAD_POLICY_KEYS,
    ):
        issues.append("expected_workload_sans or expected_workload_ids is required")
    return issues


def privatemode_model_workload_binding_values(
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
    issues: list[str] = []
    for key, raw_bindings in present:
        if not isinstance(raw_bindings, dict) or not raw_bindings:
            issues.append(f"{key} must be a non-empty object")
            continue

        parsed: dict[str, dict[str, list[str]]] = {}
        seen_model_ids: set[str] = set()
        for raw_model_id, raw_binding in raw_bindings.items():
            if not isinstance(raw_model_id, str) or not raw_model_id.strip():
                issues.append(f"{key} keys must be non-empty model strings")
                continue
            model_id = raw_model_id.strip()
            normalized_model_id = model_id.lower()
            if normalized_model_id in seen_model_ids:
                issues.append(f"{key} contains duplicate model binding {model_id}")
                continue
            seen_model_ids.add(normalized_model_id)
            if not isinstance(raw_binding, dict):
                issues.append(f"{key} for {model_id} must be an object")
                continue
            binding = dict(raw_binding)
            binding_issues = string_list_policy_issues(
                binding,
                *PRIVATEMODE_MODEL_WORKLOAD_SAN_POLICY_KEYS,
                *PRIVATEMODE_MODEL_WORKLOAD_ID_POLICY_KEYS,
            )
            binding_issues.extend(
                string_list_policy_alias_issues(
                    binding,
                    "workload_sans",
                    *PRIVATEMODE_MODEL_WORKLOAD_SAN_POLICY_KEYS,
                )
            )
            binding_issues.extend(
                string_list_policy_alias_issues(
                    binding,
                    "workload_ids",
                    *PRIVATEMODE_MODEL_WORKLOAD_ID_POLICY_KEYS,
                )
            )
            if binding_issues:
                issues.extend(
                    f"{key} for {model_id}: {issue}" for issue in binding_issues
                )
                continue

            workload_sans = normalized_string_policy_values(
                binding,
                *PRIVATEMODE_MODEL_WORKLOAD_SAN_POLICY_KEYS,
            )
            workload_ids = normalized_string_policy_values(
                binding,
                *PRIVATEMODE_MODEL_WORKLOAD_ID_POLICY_KEYS,
            )
            if not workload_sans and not workload_ids:
                issues.append(
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
            issues.append("model_workload_bindings aliases must match")
    return (parsed_values[0] if parsed_values else {}, issues)


def privatemode_model_workload_policy_issues(
    policy: dict[str, Any],
    *,
    model_ids: list[str],
    model_id_prefixes: list[str],
) -> list[str]:
    issues: list[str] = []
    if model_id_prefixes:
        issues.append(
            "Privatemode model_id_prefixes are not supported; use exact model_ids "
            "with model_workload_bindings"
        )
    if not model_ids:
        issues.append("Privatemode model_ids are required")
    elif any(not model_id.startswith("privatemode/") for model_id in model_ids):
        issues.append("Privatemode model_ids must start with privatemode/")

    bindings, binding_issues = privatemode_model_workload_binding_values(policy)
    issues.extend(binding_issues)
    if binding_issues:
        return issues

    expected_sans = set(
        normalized_string_policy_values(
            policy,
            *PRIVATEMODE_EXPECTED_WORKLOAD_SAN_POLICY_KEYS,
        )
    )
    expected_ids = set(
        normalized_string_policy_values(
            policy,
            *PRIVATEMODE_EXPECTED_WORKLOAD_ID_POLICY_KEYS,
        )
    )
    selected_models = set(model_ids)
    bound_sans: set[str] = set()
    bound_ids: set[str] = set()

    for model_id in sorted(selected_models):
        if model_id not in bindings:
            issues.append(f"model_workload_bindings is missing {model_id}")

    for model_id, binding in bindings.items():
        if model_id not in selected_models:
            issues.append(
                f"model_workload_bindings contains unselected model {model_id}"
            )
        binding_sans = set(binding["workload_sans"])
        binding_ids = set(binding["workload_ids"])
        if binding_sans - expected_sans:
            issues.append(
                f"model_workload_bindings for {model_id} references "
                "workload_sans not listed in expected_workload_sans"
            )
        if binding_ids - expected_ids:
            issues.append(
                f"model_workload_bindings for {model_id} references "
                "workload_ids not listed in expected_workload_ids"
            )
        bound_sans.update(binding_sans)
        bound_ids.update(binding_ids)

    if expected_sans - bound_sans:
        issues.append("expected_workload_sans contains unbound workloads")
    if expected_ids - bound_ids:
        issues.append("expected_workload_ids contains unbound workloads")
    return issues


def privatemode_component_policy_consistency_issues(
    policy: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    for claim_name, keys in PRIVATEMODE_COMPONENT_DIGEST_POLICY_KEYS.items():
        expected_values = policy_values_for_present_keys(policy, keys[0])
        allowed_values = policy_values_for_present_keys(policy, keys[1])
        if not expected_values or not allowed_values:
            continue
        if any(
            not isinstance(value, str) for value in expected_values + allowed_values
        ):
            continue
        expected = {value.strip().lower() for value in expected_values if value.strip()}
        allowed = {value.strip().lower() for value in allowed_values if value.strip()}
        if expected and allowed and not expected.issubset(allowed):
            issues.append(f"{claim_name} expected value must be allowed")

    for claim_name, keys in PRIVATEMODE_COMPONENT_STRING_POLICY_KEYS.items():
        expected_values = policy_values_for_present_keys(policy, keys[0])
        allowed_values = policy_values_for_present_keys(policy, keys[1])
        if not expected_values or not allowed_values:
            continue
        if any(
            not isinstance(value, str) for value in expected_values + allowed_values
        ):
            continue
        expected = {value.strip() for value in expected_values if value.strip()}
        allowed = {value.strip() for value in allowed_values if value.strip()}
        if expected and allowed and not expected.issubset(allowed):
            issues.append(f"{claim_name} expected value must be allowed")
    return issues


def routstr_code_measurement_policy_issue(policy: dict[str, Any]) -> str | None:
    values: list[object] = []
    found = False
    for key in (
        "expected_routstr_code_measurement",
        "allowed_routstr_code_measurements",
    ):
        if key not in policy:
            continue
        found = True
        value = policy[key]
        if isinstance(value, list):
            if not value:
                return "routstr code measurement policy values must be sha256 digests"
            values.extend(value)
        else:
            values.append(value)
    if not found:
        return (
            "missing expected_routstr_code_measurement or "
            "allowed_routstr_code_measurements"
        )
    if any(
        not isinstance(value, str) or not is_prefixed_sha256_digest_value(value)
        for value in values
    ):
        return "routstr code measurement policy values must be sha256 digests"
    return None


def verifier_command_policy_issue(policy: dict[str, Any], *keys: str) -> str | None:
    present_commands: list[tuple[str, tuple[str, ...]]] = []
    for key in keys:
        if key not in policy:
            continue
        value = policy[key]
        if not isinstance(value, (str, list)):
            return f"{key} must be a command string or array"
        try:
            argv = command_argv(value, label=key)
        except ValueError as exc:
            return str(exc)
        present_commands.append((key, tuple(argv)))
    if present_commands:
        if len({argv for _, argv in present_commands}) > 1:
            return "verifier_command aliases must match"
        return verifier_artifact_path_policy_issue(policy, list(present_commands[0][1]))
    return f"{' or '.join(keys)} is required"


def command_argv(value: object, *, label: str) -> list[str]:
    if isinstance(value, str):
        argv = shlex.split(value)
    elif isinstance(value, list):
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(f"{label} entries must be non-empty strings")
        argv = [item.strip() for item in value]
    else:
        argv = []
    if not argv:
        raise ValueError(f"{label} is empty")
    return argv


def verifier_artifact_path_policy_issue(
    policy: dict[str, Any],
    command: object,
) -> str | None:
    try:
        argv = command_argv(command, label="verifier_command")
    except ValueError as exc:
        return str(exc)

    artifact_path = policy.get("verifier_artifact_path")
    if artifact_path is None:
        return None
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        return "verifier_artifact_path must be a non-empty string"
    if artifact_path.strip() not in argv:
        return (
            "verifier_artifact_path must match verifier command executable or argument"
        )
    return None


def has_artifact_identity_pin(policy: dict[str, Any]) -> bool:
    return bool(
        has_release_digest_pin(policy)
        or policy_string(policy, *EHBP_CODE_MEASUREMENT_EXACT_POLICY_KEYS)
        or policy_string_values(policy, *EHBP_CODE_MEASUREMENT_ALLOWED_POLICY_KEYS)
    )


def has_release_digest_pin(policy: dict[str, Any]) -> bool:
    return bool(release_digest_policy_values(policy))


def absolute_url_issue(
    value: str,
    label: str,
    *,
    require_https: bool = True,
) -> str | None:
    if not value:
        return None
    parsed = urlparse(value)
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


def provider_base_url_issue(
    base_url: str,
    *,
    require_https: bool = True,
) -> str | None:
    return absolute_url_issue(
        base_url,
        "provider base_url",
        require_https=require_https,
    )


def normalized_hostname(value: str | None) -> str:
    return (value or "").strip().rstrip(".").lower()


def public_attestation_target_host_issue(
    value: object,
    label: str,
    *,
    absolute_url: bool = False,
) -> str | None:
    if absolute_url:
        host, issue = absolute_https_hostname(value, label)
    else:
        host, issue = host_identity_hostname(value, label)
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


def absolute_https_hostname(value: object, label: str) -> tuple[str | None, str | None]:
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
    hostname = normalized_hostname(parsed.hostname)
    if not hostname:
        return None, f"{label} must include a host"
    try:
        parsed.port
    except ValueError:
        return None, f"{label} must include a valid port"
    return hostname, None


def absolute_https_origin(value: object, label: str) -> tuple[str | None, str | None]:
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
    hostname = normalized_hostname(parsed.hostname)
    if not hostname:
        return None, f"{label} must include a host"
    try:
        port = parsed.port
    except ValueError:
        return None, f"{label} must include a valid port"
    if port is None or port == 443:
        return f"https://{hostname}", None
    return f"https://{hostname}:{port}", None


def host_identity_hostname(value: object, label: str) -> tuple[str | None, str | None]:
    if not isinstance(value, str):
        return None, None
    raw_host = value.strip()
    if not raw_host:
        return None, None
    if "://" in raw_host:
        return absolute_https_hostname(raw_host, label)
    if any(separator in raw_host for separator in "/?#"):
        return None, f"{label} must be a host or absolute URL"
    parsed = urlparse(f"https://{raw_host}")
    if "@" in parsed.netloc:
        return None, f"{label} must not include credentials"
    hostname = normalized_hostname(parsed.hostname)
    if not hostname:
        return None, f"{label} must include a host"
    try:
        parsed.port
    except ValueError:
        return None, f"{label} must include a valid port"
    return hostname, None


def policy_url_host_issues(
    policy: dict[str, Any],
    provider_base_url: str,
    *keys: str,
) -> list[str]:
    provider_host, provider_issue = absolute_https_hostname(
        provider_base_url,
        "provider base_url",
    )
    if provider_issue or not provider_host:
        return []
    provider_origin, provider_origin_issue = absolute_https_origin(
        provider_base_url,
        "provider base_url",
    )

    issues: list[str] = []
    for key in keys:
        host, issue = absolute_https_hostname(policy.get(key), key)
        if issue or not host:
            continue
        if host != provider_host:
            issues.append(f"{key} host must match provider base_url host")
            continue
        origin, origin_issue = absolute_https_origin(policy.get(key), key)
        if (
            not provider_origin_issue
            and provider_origin
            and not origin_issue
            and origin
            and origin != provider_origin
        ):
            issues.append(f"{key} origin must match provider base_url origin")
    return issues


def policy_host_identity_issues(
    policy: dict[str, Any],
    provider_base_url: str,
    *keys: str,
) -> list[str]:
    provider_host, provider_issue = absolute_https_hostname(
        provider_base_url,
        "provider base_url",
    )
    if provider_issue or not provider_host:
        return []

    issues: list[str] = []
    for key in keys:
        host, issue = host_identity_hostname(policy.get(key), key)
        if issue:
            issues.append(issue)
            continue
        if host and host != provider_host:
            issues.append(f"{key} must match provider base_url host")
    return issues


def ppq_private_base_url_issue(base_url: str) -> str | None:
    if not base_url:
        return None
    if issue := provider_base_url_issue(base_url):
        return issue
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").rstrip(".").lower()
    if hostname != "ppq.ai" and not hostname.endswith(".ppq.ai"):
        return "PPQ private base_url host must be ppq.ai or a ppq.ai subdomain"
    path_segments = {segment.lower() for segment in parsed.path.split("/") if segment}
    if "private" not in path_segments:
        return "PPQ private base_url must include a private path segment"
    return None


def ppq_private_policy_url_issues(policy: dict[str, Any], *keys: str) -> list[str]:
    issues: list[str] = []
    for key in keys:
        value = policy.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        url = value.strip()
        if absolute_url_issue(url, key):
            continue
        parsed = urlparse(url)
        path_segments = {
            segment.lower() for segment in parsed.path.split("/") if segment
        }
        if "private" not in path_segments:
            issues.append(f"{key} must include a private path segment")
    return issues


def policy_url_issues(policy: dict[str, Any], *keys: str) -> list[str]:
    issues: list[str] = []
    for key in keys:
        value = policy.get(key)
        if isinstance(value, str) and value.strip():
            if issue := absolute_url_issue(value.strip(), key):
                issues.append(issue)
    return issues


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


def privatemode_proxy_base_url_issue(base_url: str) -> str | None:
    if not base_url:
        return None
    if issue := provider_base_url_issue(base_url, require_https=False):
        return issue.replace("provider base_url", "Privatemode proxy base_url")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        return "Privatemode proxy base_url must use http or https"
    if not is_loopback_hostname(parsed.hostname):
        return "Privatemode proxy base_url must use a loopback host"
    return None


def required_false_bool_policy_issue(
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


def required_zero_int_policy_issue(
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


def validate_provider_policy(
    path: Path,
    *,
    attestation_targets_directory: AttestationTargetsDirectory | None = None,
    strict_attestation_targets: bool = False,
    attestation_results_directory: AttestationResultsDirectory | None = None,
    strict_attestation_results: bool = False,
    provider_model_catalog: ProviderModelCatalog | None = None,
    strict_provider_catalog: bool = False,
    verify_artifacts: bool = False,
    reject_placeholder_digests: bool = False,
) -> dict[str, Any]:
    try:
        document = load_json_file(path)
    except ValueError as exc:
        return {
            "path": str(path),
            "mode": "",
            "ok": False,
            "issues": [str(exc)],
            "warnings": [],
        }
    try:
        mode, policy = confidentiality_policy_from_document(document)
        model_ids, model_id_prefixes = confidentiality_model_selectors_from_document(
            document
        )
        base_url = provider_base_url_from_document(document)
    except ValueError as exc:
        return {
            "path": str(path),
            "mode": "",
            "ok": False,
            "issues": [str(exc)],
            "warnings": [],
        }
    issues: list[str] = []
    warnings: list[str] = []

    mode_issue = confidentiality_mode_issue(document)
    if mode_issue:
        issues.append(mode_issue)
    elif not mode:
        issues.append(
            "mode is required; expected tinfoil, ppq-private-tee, or privatemode"
        )
    elif mode not in SUPPORTED_CONFIDENTIALITY_MODES:
        issues.append(
            f"unknown confidentiality mode: {mode}; expected tinfoil, "
            "ppq-private-tee, or privatemode"
        )
    if not model_ids and not model_id_prefixes:
        issues.append("model_ids or model_id_prefixes is required")
    issues.extend(
        selector_list_issues(
            confidentiality_model_selector_source(document),
            "model_ids",
            "model_id_prefixes",
        )
    )
    if mode in SUPPORTED_CONFIDENTIALITY_MODES:
        issues.extend(
            runtime_policy_loadability_issues(
                document,
                mode=mode,
                base_url=base_url,
            )
        )

    for violation in inline_secret_violations(policy):
        issues.append(violation)
    if reject_placeholder_digests:
        issues.extend(placeholder_digest_violations(policy))
    issues.extend(max_age_policy_issues(policy))
    warnings.extend(
        provider_attestation_directory_warnings(
            mode,
            attestation_targets_directory,
        )
    )
    catalog_issues, catalog_warnings = provider_catalog_model_issues(
        mode=mode,
        model_ids=model_ids,
        provider_model_catalog=provider_model_catalog,
        strict_provider_catalog=strict_provider_catalog,
    )
    issues.extend(catalog_issues)
    warnings.extend(catalog_warnings)

    if mode != "privatemode" and (issue := provider_base_url_issue(base_url)):
        issues.append(issue)

    if mode == "tinfoil":
        issues.extend(
            string_policy_field_issues(
                policy,
                "attestation_url",
                "hpke_keys_url",
                "enclave_host",
                "expected_enclave_host",
                "transport_security",
                "tinfoil_transport_security",
            )
        )
        if transport_issue := tinfoil_transport_security_policy_issue(policy):
            issues.append(transport_issue)
        issues.extend(policy_url_issues(policy, "attestation_url", "hpke_keys_url"))
        issues.extend(
            policy_url_host_issues(
                policy,
                base_url,
                "attestation_url",
                "hpke_keys_url",
            )
        )
        issues.extend(
            policy_host_identity_issues(
                policy,
                base_url,
                "enclave_host",
                "expected_enclave_host",
            )
        )
        issues.extend(
            string_list_policy_alias_issues(
                policy,
                "enclave host",
                "enclave_host",
                "expected_enclave_host",
            )
        )
        issues.extend(non_empty_string_policy_issues(policy, "repo", "expected_repo"))
        issues.extend(
            string_list_policy_alias_issues(
                policy,
                "repo",
                "repo",
                "expected_repo",
            )
        )
        if not policy_string(policy, "repo", "expected_repo"):
            issues.append("repo or expected_repo is required")
        if release_issue := release_digest_policy_issue(
            policy,
            missing_required=True,
        ):
            issues.append(release_issue)
        if measurement_issue := measurement_digest_policy_issue(policy):
            issues.append(measurement_issue)
        result_issues, result_warnings = tinfoil_provider_attestation_results_issues(
            policy,
            provider_base_url=base_url,
            attestation_results_directory=attestation_results_directory,
            strict_attestation_results=strict_attestation_results,
        )
        issues.extend(result_issues)
        warnings.extend(result_warnings)
        issues.extend(
            tinfoil_model_attestation_policy_issues(
                policy,
                model_ids=model_ids,
                model_id_prefixes=model_id_prefixes,
                provider_base_url=base_url,
                attestation_targets_directory=attestation_targets_directory,
                strict_attestation_targets=strict_attestation_targets,
                attestation_results_directory=attestation_results_directory,
                strict_attestation_results=strict_attestation_results,
                attestation_target_warnings=warnings,
            )
        )
        verifier_command_issue = verifier_command_policy_issue(
            policy,
            "verifier_command",
            "tinfoil_verifier_command",
        )
        if verifier_command_issue:
            issues.append(verifier_command_issue)
        verifier_digest_issue = verifier_digest_policy_issue(policy)
        if verifier_digest_issue:
            issues.append(verifier_digest_issue)
        if verify_artifacts and not verifier_command_issue and not verifier_digest_issue:
            if verifier_artifact_issue := verifier_artifact_digest_policy_issue(
                policy,
                path,
                "verifier_command",
                "tinfoil_verifier_command",
            ):
                issues.append(verifier_artifact_issue)
    elif mode == "ppq-private-tee":
        issues.extend(
            ppq_private_model_selector_policy_issues(
                model_ids=model_ids,
                model_id_prefixes=model_id_prefixes,
            )
        )
        issues.extend(
            ppq_private_backend_model_attestation_policy_issues(
                policy,
                model_ids=model_ids,
            )
        )
        if ppq_base_issue := ppq_private_base_url_issue(base_url):
            if ppq_base_issue not in issues:
                issues.append(ppq_base_issue)
        issues.extend(
            string_policy_field_issues(
                policy,
                "attestation_bundle_url",
                "hpke_keys_url",
                "enclave_host",
                "expected_enclave_host",
            )
        )
        issues.extend(
            policy_url_issues(policy, "attestation_bundle_url", "hpke_keys_url")
        )
        issues.extend(
            ppq_private_policy_url_issues(
                policy,
                "attestation_bundle_url",
                "hpke_keys_url",
            )
        )
        issues.extend(
            policy_url_host_issues(
                policy,
                base_url,
                "attestation_bundle_url",
                "hpke_keys_url",
            )
        )
        issues.extend(
            policy_host_identity_issues(
                policy,
                base_url,
                "enclave_host",
                "expected_enclave_host",
            )
        )
        issues.extend(
            string_list_policy_alias_issues(
                policy,
                "enclave host",
                "enclave_host",
                "expected_enclave_host",
            )
        )
        issues.extend(non_empty_string_policy_issues(policy, "repo", "expected_repo"))
        issues.extend(
            string_list_policy_alias_issues(
                policy,
                "repo",
                "repo",
                "expected_repo",
            )
        )
        if not policy_string(policy, "attestation_bundle_url"):
            issues.append("attestation_bundle_url is required")
        if not policy_string(policy, "repo", "expected_repo"):
            issues.append("repo or expected_repo is required")
        if release_issue := release_digest_policy_issue(policy):
            issues.append(release_issue)
        if measurement_issue := measurement_digest_policy_issue(policy):
            issues.append(measurement_issue)
        if not has_artifact_identity_pin(policy):
            issues.append("release digest or code measurement identity pin is required")
        result_issues, result_warnings = ppq_private_attestation_results_issues(
            policy,
            provider_base_url=base_url,
            model_ids=model_ids,
            attestation_targets_directory=attestation_targets_directory,
            attestation_results_directory=attestation_results_directory,
            strict_attestation_results=strict_attestation_results,
        )
        issues.extend(result_issues)
        warnings.extend(result_warnings)
        verifier_command_issue = verifier_command_policy_issue(
            policy,
            "verifier_command",
            "ppq_private_verifier_command",
        )
        if verifier_command_issue:
            issues.append(verifier_command_issue)
        verifier_digest_issue = verifier_digest_policy_issue(policy)
        if verifier_digest_issue:
            issues.append(verifier_digest_issue)
        if verify_artifacts and not verifier_command_issue and not verifier_digest_issue:
            if verifier_artifact_issue := verifier_artifact_digest_policy_issue(
                policy,
                path,
                "verifier_command",
                "ppq_private_verifier_command",
            ):
                issues.append(verifier_artifact_issue)
    elif mode == "privatemode":
        issues.extend(
            prefixed_digest_alias_policy_issues(
                policy,
                *PRIVATEMODE_ARTIFACT_DIGEST_POLICY_KEY_GROUPS,
            )
        )
        issues.extend(privatemode_component_policy_issues(policy))
        issues.extend(privatemode_workload_policy_issues(policy))
        issues.extend(
            privatemode_model_workload_policy_issues(
                policy,
                model_ids=model_ids,
                model_id_prefixes=model_id_prefixes,
            )
        )
        issues.extend(
            string_policy_field_issues(
                policy,
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
        issues.extend(
            policy_url_issues(
                policy,
                "api_base_url",
                "apiBaseURL",
                "cdn_base_url",
                "cdnBaseURL",
                "manifest_url",
                "manifestURL",
            )
        )
        privatemode_base_issue = privatemode_proxy_base_url_issue(base_url)
        if privatemode_base_issue and privatemode_base_issue not in issues:
            issues.append(privatemode_base_issue)
        result_issues, result_warnings = privatemode_attestation_results_issues(
            policy,
            model_ids=model_ids,
            attestation_results_directory=attestation_results_directory,
            strict_attestation_results=strict_attestation_results,
        )
        issues.extend(result_issues)
        warnings.extend(result_warnings)
        for issue in (
            required_false_bool_policy_issue(
                policy,
                "dump_requests",
                "dump_requests",
                "dumpRequests",
            ),
            required_false_bool_policy_issue(
                policy,
                "shared_prompt_cache",
                "shared_prompt_cache",
                "sharedPromptCache",
            ),
            required_false_bool_policy_issue(
                policy,
                "nvidia_ocsp_allow_unknown",
                "nvidia_ocsp_allow_unknown",
                "nvidiaOCSPAllowUnknown",
            ),
            required_zero_int_policy_issue(
                policy,
                "nvidia_ocsp_revoked_grace_period_hours",
                "nvidia_ocsp_revoked_grace_period_hours",
                "nvidiaOCSPRevokedGracePeriod",
            ),
        ):
            if issue:
                issues.append(issue)
        if not policy_string(
            policy,
            *PRIVATEMODE_MANIFEST_IDENTITY_POLICY_KEYS,
        ):
            issues.append("manifest_digest or manifest_log_dir is required")
        if not policy_string(
            policy,
            *PRIVATEMODE_PROXY_BINARY_DIGEST_POLICY_KEYS,
        ):
            issues.append(
                "proxy_binary_digest is required for full Privatemode proxy artifact "
                "verification"
            )
        if policy_string(
            policy,
            *PRIVATEMODE_PROXY_BINARY_DIGEST_POLICY_KEYS,
        ) and not policy_string(
            policy,
            *PRIVATEMODE_PROXY_BINARY_PATH_POLICY_KEYS,
        ):
            issues.append(
                "proxy_binary_path is required when proxy_binary_digest is set"
            )
        verifier_command_issue = verifier_command_policy_issue(
            policy,
            "verifier_command",
            "privatemode_verifier_command",
        )
        if verifier_command_issue:
            issues.append(verifier_command_issue)
        verifier_digest_issue = verifier_digest_policy_issue(policy)
        if verifier_digest_issue:
            issues.append(verifier_digest_issue)
        if verify_artifacts and not verifier_command_issue and not verifier_digest_issue:
            if verifier_artifact_issue := verifier_artifact_digest_policy_issue(
                policy,
                path,
                "verifier_command",
                "privatemode_verifier_command",
            ):
                issues.append(verifier_artifact_issue)
        if policy_string(policy, "api_key", "apiKey"):
            issues.append("api_key must not be embedded in verifier policy")
    result = {
        "path": str(path),
        "mode": mode,
        "model_ids": model_ids,
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
    }
    if mode == "tinfoil":
        result["transport_security"] = (
            policy_string(policy, *TINFOIL_TRANSPORT_SECURITY_POLICY_KEYS) or "tls"
        ).strip().lower()
    return result


def print_report(
    *,
    verifier_manifest: list[dict[str, Any]],
    checks: list[dict[str, str]],
    policies: list[dict[str, Any]],
    manifest_path: Path,
) -> None:
    print("\nVerifier artifacts")
    for entry in verifier_manifest:
        digest = entry.get("sha256") or "not built"
        print(f"- {entry['name']}: {digest}")
        print(f"  path: {entry['artifact_path']}")
        print(f"  purpose: {entry['purpose']}")
    print(f"- manifest: {manifest_path}")

    print("\nLocal Routstr TEE readiness")
    for check in checks:
        print(f"- [{check['status']}] {check['key']}: {check['detail']}")

    if policies:
        print("\nProvider policy checks")
        for policy in policies:
            status = "ok" if policy["ok"] else "failed"
            print(f"- [{status}] {policy['path']} ({policy['mode'] or 'unknown mode'})")
            for issue in policy["issues"]:
                print(f"  issue: {issue}")
            for warning in policy["warnings"]:
                print(f"  warning: {warning}")


def provider_attestation_target_models(
    slug: str,
    attestation_targets_directory: AttestationTargetsDirectory | None,
) -> set[str]:
    if attestation_targets_directory is None:
        return set()
    models = set(attestation_targets_directory.provider_target_models.get(slug) or set())
    host_model_ids = (
        attestation_targets_directory.provider_host_model_ids.get(slug) or {}
    )
    for candidates in host_model_ids.values():
        models.update(candidates)
    return models


def provider_verified_model_statuses(
    slug: str,
    *,
    attestation_targets_directory: AttestationTargetsDirectory | None,
    attestation_results_directory: AttestationResultsDirectory | None,
) -> tuple[dict[str, str], dict[str, str]]:
    if attestation_results_directory is None:
        return {}, {}
    model_statuses = dict(
        attestation_results_directory.provider_model_statuses.get(slug) or {}
    )
    model_errors = dict(
        attestation_results_directory.provider_model_errors.get(slug) or {}
    )
    if slug == "privatemode":
        model_details = (
            attestation_results_directory.provider_model_details.get(slug) or {}
        )
        for model_id, status in list(model_statuses.items()):
            if status != "verified":
                continue
            details = model_details.get(model_id) or {}
            required_detail_keys = {
                "transport",
                "trust_tier",
                "manifest_digest",
                "proxy_binary_digest",
                "coordinator_measurement",
                "secret_service_measurement",
                "ai_worker_measurement",
                "key_release_binding",
                "gpu_attestation_policy",
                "selected_model_ids",
                "verification_steps",
                *REQUIRED_PRIVATEMODE_RESULT_DIGEST_CLAIMS,
            }
            verification_steps = details.get("verification_steps")
            if (
                any(key not in details for key in required_detail_keys)
                or not privatemode_selected_model_ids_exactly_match(
                    details.get("selected_model_ids"),
                    model_id,
                )
                or not isinstance(verification_steps, dict)
                or any(
                    verification_steps.get(step) is not True
                    for step in REQUIRED_PRIVATEMODE_RESULT_VERIFICATION_STEPS
                )
            ):
                model_statuses[model_id] = "failed"
                model_errors[model_id] = (
                    "Privatemode selected model evidence is missing policy-bound "
                    "proof details"
                )
    if attestation_targets_directory is None:
        return model_statuses, model_errors

    if slug in {"ppq", "privatemode"}:
        target_models = provider_attestation_target_models(
            slug,
            attestation_targets_directory,
        )
        for model_id, status in list(model_statuses.items()):
            if status != "verified":
                continue
            if model_id not in target_models:
                model_statuses[model_id] = "failed"
                model_errors[model_id] = (
                    "selected model is missing from confidential-inference "
                    "attestation targets"
                )

    host_statuses = (
        attestation_results_directory.provider_host_statuses.get(slug) or {}
    )
    host_errors = attestation_results_directory.provider_host_errors.get(slug) or {}
    host_model_ids = attestation_targets_directory.provider_host_model_ids.get(slug) or {}
    for host, status in host_statuses.items():
        for model_id in host_model_ids.get(host, set()):
            existing_status = model_statuses.get(model_id)
            if existing_status == "verified" and status != "verified":
                continue
            model_statuses[model_id] = status
            if host_errors.get(host):
                model_errors[model_id] = host_errors[host]
    return model_statuses, model_errors


def provider_level_attestation_results(
    slug: str,
    *,
    attestation_targets_directory: AttestationTargetsDirectory | None,
    attestation_results_directory: AttestationResultsDirectory | None,
) -> list[dict[str, str]]:
    if (
        attestation_targets_directory is None
        or attestation_results_directory is None
    ):
        return []
    check_hosts = attestation_targets_directory.provider_check_hosts.get(slug) or {}
    model_check_ids = set(
        attestation_targets_directory.provider_check_models.get(slug) or {}
    )
    if slug in {"ppq", "privatemode"}:
        model_statuses, model_errors = provider_verified_model_statuses(
            slug,
            attestation_targets_directory=attestation_targets_directory,
            attestation_results_directory=attestation_results_directory,
        )
        host_models: dict[str, set[str]] = {}
        for check_id, host in check_hosts.items():
            model_id = (
                attestation_targets_directory.provider_check_models.get(slug) or {}
            ).get(check_id)
            if model_id:
                host_models.setdefault(host, set()).add(model_id)
        if slug == "privatemode" and not host_models:
            pass
        else:
            results: list[dict[str, str]] = []
            for host, models in sorted(host_models.items()):
                failed_models = sorted(
                    model for model in models if model_statuses.get(model) == "failed"
                )
                missing_models = sorted(
                    model
                    for model in models
                    if model_statuses.get(model) not in {"verified", "failed"}
                )
                if failed_models:
                    error = "; ".join(
                        model_errors.get(model)
                        or (
                            f"{'PPQ private' if slug == 'ppq' else 'Privatemode'} "
                            f"selected model {model} failed verification"
                        )
                        for model in failed_models
                    )
                    results.append({"host": host, "status": "failed", "error": error})
                elif missing_models:
                    provider_label = (
                        "PPQ private" if slug == "ppq" else "Privatemode"
                    )
                    results.append(
                        {
                            "host": host,
                            "status": "missing",
                            "error": (
                                f"{provider_label} provider host {host} is missing verified "
                                "results for selected models: "
                                + ", ".join(missing_models)
                            ),
                        }
                    )
                else:
                    results.append({"host": host, "status": "verified", "error": ""})
            return results
    provider_hosts = {
        host
        for check_id, host in check_hosts.items()
        if check_id not in model_check_ids
        and (slug != "tinfoil" or "router" in check_id)
    }
    if slug == "tinfoil" and not provider_hosts:
        has_model_targets = bool(
            attestation_targets_directory.provider_target_models.get(slug)
            or attestation_targets_directory.provider_check_models.get(slug)
        )
        has_model_results = bool(
            attestation_results_directory.provider_model_statuses.get(slug)
        )
        if has_model_targets or has_model_results:
            return [
                {
                    "host": "inference.tinfoil.sh",
                    "status": "missing",
                    "error": "Tinfoil provider-level router attestation target is missing",
                }
            ]
    host_statuses = (
        attestation_results_directory.provider_host_statuses.get(slug) or {}
    )
    host_errors = attestation_results_directory.provider_host_errors.get(slug) or {}
    check_statuses = (
        attestation_results_directory.provider_check_statuses.get(slug) or {}
    )
    check_errors = attestation_results_directory.provider_check_errors.get(slug) or {}
    return [
        {
            "host": host,
            "status": check_statuses.get(check_id)
            or host_statuses.get(host)
            or "missing",
            "error": check_errors.get(check_id) or host_errors.get(host, ""),
        }
        for check_id, host in sorted(check_hosts.items(), key=lambda item: item[1])
        if host in provider_hosts
    ]


def provider_model_support_summary(
    *,
    provider_model_catalog: ProviderModelCatalog | None = None,
    attestation_targets_directory: AttestationTargetsDirectory | None = None,
    attestation_results_directory: AttestationResultsDirectory | None = None,
) -> dict[str, dict[str, Any]]:
    provider_slugs = set(SUPPORTED_PROVIDER_MODEL_SUPPORT_SLUGS)

    summary: dict[str, dict[str, Any]] = {}
    for slug in sorted(provider_slugs):
        catalog_models = (
            set(provider_model_catalog.provider_models.get(slug) or set())
            if provider_model_catalog is not None
            else set()
        )
        target_models = provider_attestation_target_models(
            slug,
            attestation_targets_directory,
        )
        model_statuses, model_errors = provider_verified_model_statuses(
            slug,
            attestation_targets_directory=attestation_targets_directory,
            attestation_results_directory=attestation_results_directory,
        )
        verified_models = {
            model for model, status in model_statuses.items() if status == "verified"
        }
        provider_level_results = provider_level_attestation_results(
            slug,
            attestation_targets_directory=attestation_targets_directory,
            attestation_results_directory=attestation_results_directory,
        )
        provider_level_required = bool(provider_level_results)
        provider_level_ready = provider_level_required and all(
            result["status"] == "verified" for result in provider_level_results
        )
        routable_models = (
            target_models.intersection(verified_models)
            if not provider_level_required or provider_level_ready
            else set()
        )

        target_model_keys = {
            candidate
            for model in target_models
            for candidate in provider_catalog_identity_candidate_variants(
                model.split("/")[-1],
                provider_slug=slug,
            )
        }
        target_model_keys.discard(None)
        catalog_without_targets = [
            model
            for model in sorted(catalog_models)
            if not provider_catalog_identity_candidate_variants(
                model.split("/")[-1],
                provider_slug=slug,
            ).intersection(target_model_keys)
        ]
        unverified_targets = []
        for model in sorted(target_models - verified_models):
            status = model_statuses.get(model) or "missing"
            unverified_targets.append(
                {
                    "model": model,
                    "status": status,
                    "error": model_errors.get(model, ""),
                }
            )

        entry = {
            "catalog_models": sorted(catalog_models),
            "attestation_target_models": sorted(target_models),
            "verified_attestation_models": sorted(verified_models),
            "provider_level_required": provider_level_required,
            "provider_level_ready": provider_level_ready,
            "provider_level_results": provider_level_results,
            "routable_with_full_attestation": sorted(routable_models),
            "attestation_targets_without_verified_results": unverified_targets,
            "catalog_models_without_attestation_targets": catalog_without_targets,
        }
        if any(entry.values()):
            summary[slug] = entry
    return summary


def provider_support_slug_for_policy_mode(mode: object) -> str | None:
    if not isinstance(mode, str):
        return None
    normalized = mode.strip().lower()
    if normalized == "ppq-private-tee":
        return "ppq"
    if normalized in {"tinfoil", "privatemode"}:
        return normalized
    return None


def tinfoil_ehbp_provider_level_exception_allowed(entry: dict[str, Any]) -> bool:
    if entry.get("provider_level_ready") is True:
        return True
    results = entry.get("provider_level_results")
    if not isinstance(results, list) or not results:
        return False
    for result in results:
        if not isinstance(result, dict):
            return False
        if result.get("status") != "failed":
            return False
        if result.get("error") != TINFOIL_TLS_BINDING_NOT_VERIFIED_ERROR:
            return False
    return True


def provider_model_support_ready_for_policies(
    policies: list[dict[str, Any]],
    provider_support: dict[str, dict[str, Any]],
) -> bool:
    ok_policies = [policy for policy in policies if policy.get("ok") is True]
    if not ok_policies:
        return False
    checked_policy_count = 0
    for policy in ok_policies:
        slug = provider_support_slug_for_policy_mode(policy.get("mode"))
        if slug is None:
            continue
        entry = provider_support.get(slug)
        if not isinstance(entry, dict):
            return False
        routable = entry.get("routable_with_full_attestation")
        if not isinstance(routable, list):
            return False
        routable_models = {
            model.strip().lower()
            for model in routable
            if isinstance(model, str) and model.strip()
        }
        if (
            slug == "tinfoil"
            and str(policy.get("transport_security", "")).strip().lower() == "ehbp"
        ):
            if not tinfoil_ehbp_provider_level_exception_allowed(entry):
                return False
            verified = entry.get("verified_attestation_models")
            if not isinstance(verified, list):
                return False
            targets = entry.get("attestation_target_models")
            if not isinstance(targets, list):
                return False
            routable_models = {
                model.strip().lower()
                for model in verified
                if isinstance(model, str) and model.strip()
            }
            verified_model_aliases = {
                alias
                for model in routable_models
                for alias in provider_catalog_identity_candidate_variants(
                    model.split("/")[-1],
                    provider_slug="tinfoil",
                )
            }
            target_model_aliases = {
                alias
                for model in targets
                if isinstance(model, str) and model.strip()
                for alias in provider_catalog_identity_candidate_variants(
                    model.split("/")[-1],
                    provider_slug="tinfoil",
                )
            }
            routable_model_aliases = verified_model_aliases.intersection(
                target_model_aliases
            )
        else:
            routable_model_aliases = set()
        raw_model_ids = policy.get("model_ids", [])
        raw_model_id_prefixes = policy.get("model_id_prefixes", [])
        if not isinstance(raw_model_ids, list) or not isinstance(
            raw_model_id_prefixes,
            list,
        ):
            return False
        if any(not isinstance(model, str) or not model.strip() for model in raw_model_ids):
            return False
        if any(
            not isinstance(prefix, str) or not prefix.strip()
            for prefix in raw_model_id_prefixes
        ):
            return False
        model_ids = [model.strip().lower() for model in raw_model_ids]
        model_id_prefixes = [
            prefix.strip().lower() for prefix in raw_model_id_prefixes
        ]
        if model_id_prefixes:
            return False
        if not model_ids:
            return False
        if len(set(model_ids)) != len(model_ids):
            return False
        checked_policy_count += 1
        if slug == "tinfoil" and policy.get("transport_security") == "ehbp":
            for model in model_ids:
                model_aliases = provider_catalog_identity_candidate_variants(
                    model.split("/")[-1],
                    provider_slug="tinfoil",
                )
                if not model_aliases.intersection(routable_model_aliases):
                    return False
        elif not all(model in routable_models for model in model_ids):
            return False
    return checked_policy_count > 0


def preflight_json_report(
    *,
    verifier_manifest: list[dict[str, str]],
    checks: list[dict[str, str]],
    policies: list[dict[str, Any]],
    manifest_path: Path,
    provider_catalog_json: Path | None = None,
    attestation_targets_json: Path | None = None,
    attestation_results_json: Path | None = None,
    max_attestation_result_age_seconds: int | None = None,
    provider_model_catalog: ProviderModelCatalog | None = None,
    attestation_targets_directory: AttestationTargetsDirectory | None = None,
    attestation_results_directory: AttestationResultsDirectory | None = None,
    require_guardrail_inputs: bool = False,
) -> dict[str, Any]:
    local_tee_ready = bool(checks) and not any(
        check["status"] == "missing" for check in checks
    )
    provider_policies_ready = bool(policies) and not any(
        (not policy["ok"]) or bool(policy.get("warnings")) for policy in policies
    )
    provider_support_inputs_present = any(
        input_value is not None
        for input_value in (
            provider_model_catalog,
            attestation_targets_directory,
            attestation_results_directory,
        )
    )
    provider_support = (
        provider_model_support_summary(
            provider_model_catalog=provider_model_catalog,
            attestation_targets_directory=attestation_targets_directory,
            attestation_results_directory=attestation_results_directory,
        )
        if provider_support_inputs_present
        else {}
    )
    provider_model_support_ready = False if provider_support_inputs_present else None
    if provider_support:
        provider_model_support_ready = provider_model_support_ready_for_policies(
            policies,
            provider_support,
        )
    guardrail_inputs: dict[str, Any] = {
        "max_attestation_result_age_seconds": max_attestation_result_age_seconds,
    }
    for prefix, path in (
        ("provider_catalog", provider_catalog_json),
        ("attestation_targets", attestation_targets_json),
        ("attestation_results", attestation_results_json),
    ):
        if path is None:
            continue
        guardrail_inputs[f"{prefix}_path"] = str(path)
        guardrail_inputs[f"{prefix}_sha256"] = sha256_file(path)
    guardrail_inputs_ready = not require_guardrail_inputs or all(
        path is not None
        for path in (
            provider_catalog_json,
            attestation_targets_json,
            attestation_results_json,
        )
    )
    deployment_ready = local_tee_ready and provider_policies_ready
    if provider_model_support_ready is not None:
        deployment_ready = deployment_ready and provider_model_support_ready
    deployment_ready = deployment_ready and guardrail_inputs_ready
    report = {
        "deployment_ready": deployment_ready,
        "local_tee_ready": local_tee_ready,
        "provider_policies_ready": provider_policies_ready,
        "provider_policy_count": len(policies),
        "verifiers": verifier_manifest,
        "local_tee_checks": checks,
        "provider_policy_checks": policies,
        "manifest_path": str(manifest_path),
    }
    if require_guardrail_inputs:
        report["guardrail_inputs_ready"] = guardrail_inputs_ready
    if any(value is not None for value in guardrail_inputs.values()):
        report["guardrail_inputs"] = guardrail_inputs
    if provider_support_inputs_present:
        report["provider_model_support"] = provider_support
        report["provider_model_support_ready"] = provider_model_support_ready
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

    def env_path_list(name: str) -> list[Path]:
        value = os.getenv(name)
        if value is None or not value.strip():
            return []
        return [Path(item) for item in shlex.split(value)]

    parser = argparse.ArgumentParser(
        description=(
            "Build/digest confidential-routing verifier commands and check the "
            "deployment inputs needed before live Tinfoil, PPQ private, "
            "Privatemode, and Routstr TEE validation."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for built verifier artifacts",
    )
    parser.add_argument("--skip-tests", action="store_true", help="skip go test")
    parser.add_argument("--skip-build", action="store_true", help="skip go build")
    parser.add_argument(
        "--policy-json",
        type=Path,
        action="append",
        default=None,
        help="provider settings or confidentiality policy JSON file to validate",
    )
    parser.add_argument(
        "--attestation-targets-json",
        type=Path,
        default=env_path("ATTESTATION_TARGETS_JSON"),
        help=(
            "optional confidential-inference attestation-targets.json file used "
            "to cross-check selected provider model targets"
        ),
    )
    parser.add_argument(
        "--strict-attestation-targets",
        action="store_true",
        default=env_flag_enabled("STRICT_ATTESTATION_TARGETS"),
        help=(
            "fail provider policy checks when selected model targets are absent "
            "from --attestation-targets-json"
        ),
    )
    parser.add_argument(
        "--attestation-results-json",
        type=Path,
        default=env_path("ATTESTATION_RESULTS_JSON"),
        help=(
            "optional confidential-inference attestation-results.json file used "
            "to cross-check latest provider target verification statuses"
        ),
    )
    parser.add_argument(
        "--strict-attestation-results",
        action="store_true",
        default=env_flag_enabled("STRICT_ATTESTATION_RESULTS"),
        help=(
            "fail provider policy checks when selected Tinfoil model targets are "
            "missing, failed, or unreachable in --attestation-results-json"
        ),
    )
    parser.add_argument(
        "--max-attestation-result-age-seconds",
        type=int,
        default=DEFAULT_MAX_ATTESTATION_RESULT_AGE_SECONDS,
        help=(
            "maximum accepted age for verified confidential-inference result rows; "
            "stale or timestamp-less rows are downgraded before strict checks; "
            f"defaults to {DEFAULT_MAX_ATTESTATION_RESULT_AGE_SECONDS}"
        ),
    )
    parser.add_argument(
        "--provider-catalog-json",
        type=Path,
        default=env_path("PROVIDER_CATALOG_JSON"),
        help=(
            "optional confidential-inference providers.json file used to "
            "cross-check exact selected provider model IDs"
        ),
    )
    parser.add_argument(
        "--strict-provider-catalog",
        action="store_true",
        default=env_flag_enabled("STRICT_PROVIDER_CATALOG"),
        help=(
            "fail provider policy checks when selected model IDs are absent "
            "from --provider-catalog-json"
        ),
    )
    parser.add_argument(
        "--verify-artifacts",
        action="store_true",
        default=env_flag_enabled("VERIFY_ARTIFACTS"),
        help=(
            "hash configured provider verifier artifacts and compare them with "
            "verifier_command_digest or verifier_binary_digest"
        ),
    )
    parser.add_argument(
        "--strict-env",
        action="store_true",
        default=env_flag_enabled("STRICT_ENV"),
        help="exit nonzero if local TEE environment readiness checks are missing",
    )
    parser.add_argument(
        "--strict-deployment-ready",
        action="store_true",
        default=env_flag_enabled("STRICT_DEPLOYMENT_READY"),
        help=(
            "exit nonzero unless local TEE readiness is complete, at least one "
            "supplied provider policy passes validation, required guardrail "
            "files are present, and at least one routable selected model is in "
            "routable_with_full_attestation"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=env_flag_enabled("PREFLIGHT_JSON"),
        help="emit JSON report",
    )
    args = parser.parse_args(argv)
    if args.policy_json is None:
        args.policy_json = env_path_list("POLICY_JSON")
    max_age_env = os.getenv("MAX_ATTESTATION_RESULT_AGE_SECONDS")
    if (
        "--max-attestation-result-age-seconds" not in argv
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
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    )
    verifier_manifest = build_verifiers(
        output_dir=output_dir,
        skip_tests=args.skip_tests,
        skip_build=args.skip_build,
        log_file=sys.stderr if args.json else sys.stdout,
    )
    checks = env_checks()
    attestation_targets_directory = None
    if args.attestation_targets_json is not None:
        attestation_targets_directory = load_attestation_targets_directory(
            args.attestation_targets_json
        )
    attestation_results_directory = None
    if args.attestation_results_json is not None:
        attestation_results_directory = load_attestation_results_directory(
            args.attestation_results_json,
            attestation_targets_directory=attestation_targets_directory,
            max_result_age_seconds=args.max_attestation_result_age_seconds,
        )
    provider_model_catalog = None
    if args.provider_catalog_json is not None:
        provider_model_catalog = load_provider_model_catalog(args.provider_catalog_json)
    policies = [
        validate_provider_policy(
            path,
            attestation_targets_directory=attestation_targets_directory,
            strict_attestation_targets=args.strict_attestation_targets,
            attestation_results_directory=attestation_results_directory,
            strict_attestation_results=args.strict_attestation_results,
            provider_model_catalog=provider_model_catalog,
            strict_provider_catalog=args.strict_provider_catalog,
            verify_artifacts=args.verify_artifacts,
            reject_placeholder_digests=args.strict_deployment_ready,
        )
        for path in args.policy_json
    ]
    manifest_path = output_dir / "confidential-verifier-manifest.json"

    report = preflight_json_report(
        verifier_manifest=verifier_manifest,
        checks=checks,
        policies=policies,
        manifest_path=manifest_path,
        provider_catalog_json=args.provider_catalog_json,
        attestation_targets_json=args.attestation_targets_json,
        attestation_results_json=args.attestation_results_json,
        max_attestation_result_age_seconds=args.max_attestation_result_age_seconds,
        provider_model_catalog=provider_model_catalog,
        attestation_targets_directory=attestation_targets_directory,
        attestation_results_directory=attestation_results_directory,
        require_guardrail_inputs=args.strict_deployment_ready,
    )
    if args.json:
        try:
            payload = json.dumps(
                report,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        except ValueError as exc:
            print(
                f"error: failed to serialize preflight JSON report: {exc}",
                file=sys.stderr,
            )
            return 1
        print(payload)
    else:
        print_report(
            verifier_manifest=verifier_manifest,
            checks=checks,
            policies=policies,
            manifest_path=manifest_path,
        )

    has_missing_env = any(check["status"] == "missing" for check in checks)
    has_policy_issue = any(not policy["ok"] for policy in policies)
    missing_strict_guardrails = bool(
        args.strict_deployment_ready
        and (
            args.provider_catalog_json is None
            or args.attestation_targets_json is None
            or args.attestation_results_json is None
        )
    )
    if args.strict_deployment_ready and not report["deployment_ready"]:
        return 1
    if missing_strict_guardrails:
        return 1
    if has_policy_issue or (args.strict_env and has_missing_env):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
