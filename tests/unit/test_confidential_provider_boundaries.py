from __future__ import annotations

import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from routstr.algorithm import (
    provider_endpoint_requirement_for_path,
    provider_supports_required_endpoint,
    public_supported_endpoints_for_provider,
)
from routstr.payment.models import _update_model_sats_pricing
from routstr.upstream import upstream_provider_classes
from routstr.upstream.base import ConfidentialityStatus, ConfidentialVerifierPolicy
from routstr.upstream.confidential_verifiers import (
    _expected_ehbp_claim_values,
    _measurement_identity_policy_issue,
    _run_confidential_verifier_command,
    _validate_privatemode_verifier_result,
    _validate_tinfoil_verifier_result,
    _verified_command_digest,
    _verified_status_with_policy,
)
from routstr.upstream.ppqai import PPQPrivateUpstreamProvider
from routstr.upstream.privatemode import PrivatemodeUpstreamProvider
from routstr.upstream.tinfoil import TinfoilUpstreamProvider

VERIFIER_STUB_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "confidential_verifier_stub.py"
)
TINFOIL_EHBP_ATTESTATION_FORMAT = "https://tinfoil.sh/predicate/sev-snp-guest/v2"
VALID_VERIFIER_DIGEST = (
    "sha256:" + hashlib.sha256(b"valid-verifier").hexdigest()
)
VALID_VERIFIER_DIGEST_MISMATCH = (
    "sha256:" + hashlib.sha256(b"valid-verifier-mismatch").hexdigest()
)


def _test_digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


def _sha256_json_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


PRIVATEMODE_EXPECTED_WORKLOAD_SANS = ["workload-kimi-k2-6"]
PRIVATEMODE_TRUST_TIER = "app-e2ee"
PRIVATEMODE_GPU_ATTESTATION_POLICY = "nvidia-ocsp-good-only"
PRIVATEMODE_EXPECTED_WORKLOAD_IDENTITY_DIGEST = _sha256_json_digest(
    {"ids": [], "sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS}
)
PRIVATEMODE_MODEL_WORKLOAD_BINDINGS = {
    "privatemode/kimi-k2-6": {
        "workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
    }
}
PRIVATEMODE_MODEL_IDS = ["privatemode/kimi-k2-6"]
PRIVATEMODE_EMPTY_MODEL_WORKLOAD_BINDING_DIGEST = _sha256_json_digest({})
PRIVATEMODE_MODEL_WORKLOAD_BINDING_DIGEST = _sha256_json_digest(
    {
        "privatemode/kimi-k2-6": {
            "workload_ids": [],
            "workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
        }
    }
)
PRIVATEMODE_MESH_CA_DIGEST = _test_digest("mesh-ca")
PRIVATEMODE_SECRET_SERVICE_CERTIFICATE_DIGEST = _test_digest("secret-service-cert")
PRIVATEMODE_NVIDIA_OCSP_POLICY_MAC_DIGEST = _test_digest("nvidia-ocsp-mac")
PRIVATEMODE_INFERENCE_SECRET_ID_DIGEST = _test_digest("inference-secret-id")
EHBP_CODE_MEASUREMENT = _test_digest("ehbp-code-measurement")
EHBP_ENCLAVE_MEASUREMENT = _test_digest("ehbp-enclave-measurement")
EHBP_OTHER_CODE_MEASUREMENT = _test_digest("ehbp-other-code-measurement")
EHBP_OTHER_ENCLAVE_MEASUREMENT = _test_digest("ehbp-other-enclave-measurement")
EHBP_FIXTURE_CODE_MEASUREMENT = _test_digest("fixture-code-measurement")
TINFOIL_SELECTED_MODEL_ID = "tinfoil/kimi-k2-6"
TINFOIL_SELECTED_MODEL_HOST = "kimi-k2-6.inf13.tinfoil.sh"
TINFOIL_SELECTED_MODEL_REPO = "tinfoilsh/confidential-kimi-k2-6-b200"
TINFOIL_ROUTER_RELEASE_HEX = hashlib.sha256(b"tinfoil-router-release").hexdigest()
TINFOIL_ROUTER_RELEASE_DIGEST = f"sha256:{TINFOIL_ROUTER_RELEASE_HEX}"
TINFOIL_MODEL_RELEASE_DIGEST = _test_digest("tinfoil-kimi-k2-6-release")
TINFOIL_TLS_PUBLIC_KEY_DIGEST = _test_digest("tinfoil-tls-public-key")
PRIVATEMODE_MANIFEST_DIGEST = _test_digest("privatemode-manifest")
PRIVATEMODE_MANIFEST_LOG_DIGEST = _test_digest("privatemode-manifest-log")
PRIVATEMODE_PROXY_IMAGE_DIGEST = _test_digest("privatemode-proxy-image")
PRIVATEMODE_PROXY_BINARY_DIGEST = _test_digest("privatemode-proxy-binary")
PRIVATEMODE_PROXY_BINARY_PATH = "/opt/privatemode/privatemode-proxy"
PRIVATEMODE_OTHER_MANIFEST_DIGEST = _test_digest("privatemode-other-manifest")
PRIVATEMODE_COORDINATOR_MEASUREMENT = _test_digest("privatemode-coordinator")
PRIVATEMODE_SECRET_SERVICE_MEASUREMENT = _test_digest("privatemode-secret-service")
PRIVATEMODE_OTHER_SECRET_SERVICE_MEASUREMENT = _test_digest(
    "privatemode-other-secret-service"
)
PRIVATEMODE_AI_WORKER_MEASUREMENT = _test_digest("privatemode-attested-workload-policy")


def _privatemode_key_release_binding_digest(
    *,
    model_workload_binding_digest: str = PRIVATEMODE_EMPTY_MODEL_WORKLOAD_BINDING_DIGEST,
) -> str:
    return _sha256_json_digest(
        {
            "attested_workload_policy_digest": PRIVATEMODE_AI_WORKER_MEASUREMENT,
            "expected_workload_identity_digest": (
                PRIVATEMODE_EXPECTED_WORKLOAD_IDENTITY_DIGEST
            ),
            "model_workload_binding_digest": model_workload_binding_digest,
            "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
            "mesh_ca_digest": PRIVATEMODE_MESH_CA_DIGEST,
            "secret_service_certificate_digest": (
                PRIVATEMODE_SECRET_SERVICE_CERTIFICATE_DIGEST
            ),
            "inference_secret_id_digest": PRIVATEMODE_INFERENCE_SECRET_ID_DIGEST,
            "nvidia_ocsp_policy_mac_digest": PRIVATEMODE_NVIDIA_OCSP_POLICY_MAC_DIGEST,
        }
    )


PRIVATEMODE_KEY_RELEASE_BINDING = _privatemode_key_release_binding_digest()
PRIVATEMODE_MODEL_WORKLOAD_KEY_RELEASE_BINDING = (
    _privatemode_key_release_binding_digest(
        model_workload_binding_digest=PRIVATEMODE_MODEL_WORKLOAD_BINDING_DIGEST
    )
)


def _sha256_file_digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def test_verified_command_digest_resolves_bare_command_from_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = tmp_path / "routstr-test-verifier"
    verifier.write_bytes(b"test verifier binary")
    verifier.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    policy = ConfidentialVerifierPolicy(
        provider_type="tinfoil",
        mode="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        policy={
            "verifier_command": ["routstr-test-verifier"],
            "verifier_command_digest": _sha256_file_digest(verifier),
        },
    )

    digest, artifact_path = _verified_command_digest(
        policy,
        ["routstr-test-verifier"],
    )

    assert digest == _sha256_file_digest(verifier)
    assert artifact_path == str(verifier)


def _tinfoil_attestation_body(report: bytes = b"attestation-report") -> str:
    return base64.b64encode(gzip.compress(report)).decode("ascii")


def _ehbp_verification_steps() -> dict[str, bool]:
    return {
        "hardware_attestation_report": True,
        "hardware_certificate_chain": True,
        "code_transparency": True,
        "measurement_match": True,
        "attested_transport_key_binding": True,
        "freshness": True,
    }


def _tinfoil_full_model_attestation_policy(
    *,
    code_measurement: str = EHBP_CODE_MEASUREMENT,
) -> dict[str, Any]:
    return {
        "require_model_attestations": True,
        "model_attestation_targets": {
            TINFOIL_SELECTED_MODEL_ID: {
                "host": TINFOIL_SELECTED_MODEL_HOST,
                "repo": TINFOIL_SELECTED_MODEL_REPO,
                "expected_code_measurement_fingerprint": code_measurement,
                "expected_release_digest": TINFOIL_MODEL_RELEASE_DIGEST,
            }
        },
    }


def _tinfoil_model_attestation_claims(payload: dict[str, Any]) -> dict[str, Any]:
    model_attestations = payload.get("model_attestations")
    if not isinstance(model_attestations, list) or not model_attestations:
        return {}
    first = model_attestations[0]
    if not isinstance(first, dict):
        return {}
    attestation = first.get("attestation")
    if not isinstance(attestation, dict):
        attestation = {}
    return {
        TINFOIL_SELECTED_MODEL_ID: {
            "repo": TINFOIL_SELECTED_MODEL_REPO,
            "attestation_format": attestation.get("format")
            or TINFOIL_EHBP_ATTESTATION_FORMAT,
            "attestation_report_digest": attestation.get("report_digest")
            or _test_digest("model-attestation-report"),
            "attested_hpke_public_key_hex": "11" * 32,
            "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
            "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
            "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
            "release_digest": TINFOIL_MODEL_RELEASE_DIGEST,
            "verification_steps": _ehbp_verification_steps(),
        }
    }


def _privatemode_verification_steps() -> dict[str, bool]:
    return {
        "contrast_manifest": True,
        "coordinator_attestation": True,
        "mesh_ca_binding": True,
        "secret_service_tls": True,
        "ai_worker_attestation": True,
        "gpu_attestation": True,
        "key_release_binding": True,
        "prompt_encryption": True,
        "nvidia_ocsp_revocation": True,
    }


def _privatemode_proof_claims(
    *,
    model_workload_binding_digest: str = PRIVATEMODE_EMPTY_MODEL_WORKLOAD_BINDING_DIGEST,
    selected_model_ids: list[str] | None = ["privatemode/kimi-k2-6"],
) -> dict[str, object]:
    claims: dict[str, object] = {
        "coordinator_attestation_doc_digest": _test_digest("coordinator-doc"),
        "mesh_ca_digest": PRIVATEMODE_MESH_CA_DIGEST,
        "secret_service_certificate_digest": (
            PRIVATEMODE_SECRET_SERVICE_CERTIFICATE_DIGEST
        ),
        "ai_worker_manifest_digest": _test_digest("ai-worker-manifest"),
        "attested_workload_identity_digest": _test_digest("attested-workload-identity"),
        "trust_tier": PRIVATEMODE_TRUST_TIER,
        "attested_workload_policy_digest": PRIVATEMODE_AI_WORKER_MEASUREMENT,
        "expected_workload_identity_digest": PRIVATEMODE_EXPECTED_WORKLOAD_IDENTITY_DIGEST,
        "model_workload_binding_digest": model_workload_binding_digest,
        "nvidia_ocsp_policy_header_digest": _test_digest("nvidia-ocsp-header"),
        "nvidia_ocsp_policy_mac_digest": PRIVATEMODE_NVIDIA_OCSP_POLICY_MAC_DIGEST,
        "prompt_encryption_ciphertext_digest": _test_digest("prompt-ciphertext"),
        "inference_secret_id_digest": PRIVATEMODE_INFERENCE_SECRET_ID_DIGEST,
    }
    if selected_model_ids is not None:
        claims["selected_model_ids"] = selected_model_ids
    return claims


@pytest.mark.asyncio
async def test_confidential_verifier_command_rejects_non_standard_json_constants(
    tmp_path: Path,
) -> None:
    verifier = tmp_path / "nan_verifier.py"
    verifier.write_text(
        (
            "import sys\n"
            'sys.stdout.write(\'{"verified": true, "claims": {"non_finite": NaN}}\')\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not contain NaN"):
        await _run_confidential_verifier_command(
            [sys.executable, str(verifier)],
            {"schema_version": "unit-test"},
            2.0,
        )


def test_tinfoil_policy_accepts_native_sev_snp_measurement_pin() -> None:
    native_measurement = "b1" * 48
    normalized_measurement = hashlib.sha256(
        bytes.fromhex(native_measurement)
    ).hexdigest()
    policy = ConfidentialVerifierPolicy.from_provider_settings(
        provider_type="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        provider_settings={
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "expected_code_measurement_fingerprint": native_measurement,
                    "allowed_code_measurement_fingerprints": [
                        "sha256:" + normalized_measurement
                    ],
                },
            }
        },
    )
    assert policy is not None

    assert _measurement_identity_policy_issue(policy) is None
    expected_values = _expected_ehbp_claim_values(policy)
    assert normalized_measurement in expected_values["code_measurement_fingerprint"]
    assert (
        "sha256:" + normalized_measurement
        in expected_values["code_measurement_fingerprint"]
    )


def test_verified_status_rebuilds_model_selectors_from_active_policy() -> None:
    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": _test_digest("release"),
                },
            }
        }
    )
    policy = provider.confidentiality_policy()
    assert policy is not None
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=False,
            mode="tinfoil",
            model_id_prefixes=["tinfoil/"],
            policy_digest=policy.digest,
        )
    )

    status = _verified_status_with_policy(
        provider,
        mode="tinfoil",
        verifier="unit-test-verifier",
        evidence_digest=_test_digest("evidence"),
        verified_claims={"transport": "ehbp"},
        verified_at=1_700_000_000,
        expires_at=1_800_000_000,
    )

    assert status.model_ids == ["tinfoil/gpt-secure"]
    assert status.model_id_prefixes == []


@pytest.mark.asyncio
async def test_confidential_verifier_command_rejects_non_canonical_payload_before_execution(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "executed"
    verifier = tmp_path / "payload_verifier.py"
    verifier.write_text(
        (
            "import pathlib, sys\n"
            f"pathlib.Path({str(marker)!r}).write_text('ran')\n"
            "sys.stdout.write('{\"verified\": true}')\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="confidential verifier payload JSON must be canonical",
    ):
        await _run_confidential_verifier_command(
            [sys.executable, str(verifier)],
            {"schema_version": "unit-test", "non_finite": float("nan")},
            2.0,
        )
    assert not marker.exists()


def test_tinfoil_verifier_result_rejects_non_string_verifier_identity() -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="tinfoil",
        mode="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
        },
    )
    evidence_digest = _test_digest("tinfoil-evidence")
    nonce = "nonce"
    attestation_report_digest = _test_digest("tinfoil-report")

    with pytest.raises(ValueError, match="verifier identity must be a string"):
        _validate_tinfoil_verifier_result(
            verifier_result={
                "verified": True,
                "verifier": 123,
                "policy_digest": policy.digest,
                "evidence_digest": evidence_digest,
                "verification_nonce": nonce,
                "claims": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "attestation_format": TINFOIL_EHBP_ATTESTATION_FORMAT,
                    "attestation_report_digest": attestation_report_digest,
                    "attested_hpke_public_key_hex": "11" * 32,
                    "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                    "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                    "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verification_steps": _ehbp_verification_steps(),
                },
            },
            policy=policy,
            evidence_digest=evidence_digest,
            verification_nonce=nonce,
            attestation_format=TINFOIL_EHBP_ATTESTATION_FORMAT,
            attestation_report_digest=attestation_report_digest,
            ehbp_public_key_hex="11" * 32,
            ehbp_key_config_b64=base64.b64encode(b"hpke-key-config").decode("ascii"),
            ehbp_claims={},
        )


def test_tinfoil_verifier_result_rejects_non_canonical_claims() -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="tinfoil",
        mode="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
        },
    )
    evidence_digest = _test_digest("tinfoil-evidence")
    nonce = "nonce"
    attestation_report_digest = _test_digest("tinfoil-report")

    with pytest.raises(ValueError):
        _validate_tinfoil_verifier_result(
            verifier_result={
                "verified": True,
                "verifier": "unit-test-tinfoil-verifier",
                "policy_digest": policy.digest,
                "evidence_digest": evidence_digest,
                "verification_nonce": nonce,
                "claims": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "attestation_format": TINFOIL_EHBP_ATTESTATION_FORMAT,
                    "attestation_report_digest": attestation_report_digest,
                    "attested_hpke_public_key_hex": "11" * 32,
                    "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                    "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                    "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verification_steps": _ehbp_verification_steps(),
                    "non_canonical": float("nan"),
                },
            },
            policy=policy,
            evidence_digest=evidence_digest,
            verification_nonce=nonce,
            attestation_format=TINFOIL_EHBP_ATTESTATION_FORMAT,
            attestation_report_digest=attestation_report_digest,
            ehbp_public_key_hex="11" * 32,
            ehbp_key_config_b64=base64.b64encode(b"hpke-key-config").decode("ascii"),
            ehbp_claims={},
        )


def test_tinfoil_verifier_result_rejects_non_boolean_verification_step() -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="tinfoil",
        mode="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
        },
    )
    evidence_digest = _test_digest("tinfoil-evidence")
    nonce = "nonce"
    attestation_report_digest = _test_digest("tinfoil-report")
    verification_steps: dict[str, object] = dict(_ehbp_verification_steps())
    verification_steps["Authorization"] = "SECRET_STEP"

    with pytest.raises(
        ValueError,
        match="verification_steps values must be booleans",
    ):
        _validate_tinfoil_verifier_result(
            verifier_result={
                "verified": True,
                "verifier": "unit-test-tinfoil-verifier",
                "policy_digest": policy.digest,
                "evidence_digest": evidence_digest,
                "verification_nonce": nonce,
                "claims": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "attestation_format": TINFOIL_EHBP_ATTESTATION_FORMAT,
                    "attestation_report_digest": attestation_report_digest,
                    "attested_hpke_public_key_hex": "11" * 32,
                    "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                    "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                    "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verification_steps": verification_steps,
                },
            },
            policy=policy,
            evidence_digest=evidence_digest,
            verification_nonce=nonce,
            attestation_format=TINFOIL_EHBP_ATTESTATION_FORMAT,
            attestation_report_digest=attestation_report_digest,
            ehbp_public_key_hex="11" * 32,
            ehbp_key_config_b64=base64.b64encode(b"hpke-key-config").decode("ascii"),
            ehbp_claims={},
        )


def test_tinfoil_verifier_result_rejects_conflicting_tls_key_aliases() -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="tinfoil",
        mode="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
        },
    )
    evidence_digest = _test_digest("tinfoil-evidence")
    nonce = "nonce"
    attestation_report_digest = _test_digest("tinfoil-report")

    with pytest.raises(ValueError, match="TLS public key binding aliases must match"):
        _validate_tinfoil_verifier_result(
            verifier_result={
                "verified": True,
                "verifier": "unit-test-tinfoil-verifier",
                "policy_digest": policy.digest,
                "evidence_digest": evidence_digest,
                "verification_nonce": nonce,
                "claims": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "attestation_format": TINFOIL_EHBP_ATTESTATION_FORMAT,
                    "attestation_report_digest": attestation_report_digest,
                    "attested_hpke_public_key_hex": "11" * 32,
                    "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                    "tls_public_key": "33" * 32,
                    "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                    "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verification_steps": _ehbp_verification_steps(),
                },
            },
            policy=policy,
            evidence_digest=evidence_digest,
            verification_nonce=nonce,
            attestation_format=TINFOIL_EHBP_ATTESTATION_FORMAT,
            attestation_report_digest=attestation_report_digest,
            ehbp_public_key_hex="11" * 32,
            ehbp_key_config_b64=base64.b64encode(b"hpke-key-config").decode("ascii"),
            ehbp_claims={},
        )


def test_tinfoil_verifier_result_rejects_string_expiry_claim() -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="tinfoil",
        mode="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
        },
    )
    evidence_digest = _test_digest("tinfoil-evidence")
    nonce = "nonce"
    attestation_report_digest = _test_digest("tinfoil-report")

    with pytest.raises(ValueError, match="expires_at must be an integer"):
        _validate_tinfoil_verifier_result(
            verifier_result={
                "verified": True,
                "verifier": "unit-test-tinfoil-verifier",
                "policy_digest": policy.digest,
                "evidence_digest": evidence_digest,
                "verification_nonce": nonce,
                "expires_at": "4102444800",
                "claims": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "attestation_format": TINFOIL_EHBP_ATTESTATION_FORMAT,
                    "attestation_report_digest": attestation_report_digest,
                    "attested_hpke_public_key_hex": "11" * 32,
                    "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                    "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                    "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verification_steps": _ehbp_verification_steps(),
                },
            },
            policy=policy,
            evidence_digest=evidence_digest,
            verification_nonce=nonce,
            attestation_format=TINFOIL_EHBP_ATTESTATION_FORMAT,
            attestation_report_digest=attestation_report_digest,
            ehbp_public_key_hex="11" * 32,
            ehbp_key_config_b64=base64.b64encode(b"hpke-key-config").decode("ascii"),
            ehbp_claims={},
        )


def test_tinfoil_verifier_result_rejects_non_string_attested_hpke_key() -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="tinfoil",
        mode="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
        },
    )
    evidence_digest = _test_digest("tinfoil-evidence")
    nonce = "nonce"
    attestation_report_digest = _test_digest("tinfoil-report")

    with pytest.raises(
        ValueError,
        match="attested HPKE public key claim must be a string",
    ):
        _validate_tinfoil_verifier_result(
            verifier_result={
                "verified": True,
                "verifier": "unit-test-tinfoil-verifier",
                "policy_digest": policy.digest,
                "evidence_digest": evidence_digest,
                "verification_nonce": nonce,
                "claims": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "attestation_format": TINFOIL_EHBP_ATTESTATION_FORMAT,
                    "attestation_report_digest": attestation_report_digest,
                    "attested_hpke_public_key_hex": int("1" * 64),
                    "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                    "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                    "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verification_steps": _ehbp_verification_steps(),
                },
            },
            policy=policy,
            evidence_digest=evidence_digest,
            verification_nonce=nonce,
            attestation_format=TINFOIL_EHBP_ATTESTATION_FORMAT,
            attestation_report_digest=attestation_report_digest,
            ehbp_public_key_hex="11" * 32,
            ehbp_key_config_b64=base64.b64encode(b"hpke-key-config").decode("ascii"),
            ehbp_claims={},
        )


@pytest.mark.parametrize(
    ("claim_name", "claim_value", "match_text"),
    (
        (
            "payload_policy_digest",
            _test_digest("wrong-policy"),
            "payload_policy_digest",
        ),
        (
            "payload_evidence_digest",
            _test_digest("wrong-evidence"),
            "payload_evidence_digest",
        ),
        ("payload_verification_nonce", "wrong-nonce", "payload_verification_nonce"),
    ),
)
def test_tinfoil_verifier_result_rejects_mismatched_payload_binding_claims(
    claim_name: str,
    claim_value: str,
    match_text: str,
) -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="tinfoil",
        mode="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
        },
    )
    evidence_digest = _test_digest("tinfoil-evidence")
    nonce = "nonce"
    attestation_report_digest = _test_digest("tinfoil-report")

    with pytest.raises(ValueError, match=match_text):
        _validate_tinfoil_verifier_result(
            verifier_result={
                "verified": True,
                "verifier": "unit-test-tinfoil-verifier",
                "policy_digest": policy.digest,
                "evidence_digest": evidence_digest,
                "verification_nonce": nonce,
                "claims": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "attestation_format": TINFOIL_EHBP_ATTESTATION_FORMAT,
                    "attestation_report_digest": attestation_report_digest,
                    "attested_hpke_public_key_hex": "11" * 32,
                    "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                    "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                    "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verification_steps": _ehbp_verification_steps(),
                    claim_name: claim_value,
                },
            },
            policy=policy,
            evidence_digest=evidence_digest,
            verification_nonce=nonce,
            attestation_format=TINFOIL_EHBP_ATTESTATION_FORMAT,
            attestation_report_digest=attestation_report_digest,
            ehbp_public_key_hex="11" * 32,
            ehbp_key_config_b64=base64.b64encode(b"hpke-key-config").decode("ascii"),
            ehbp_claims={},
        )


def test_tinfoil_verifier_result_rejects_model_attestation_without_release_digest() -> (
    None
):
    policy = ConfidentialVerifierPolicy(
        provider_type="tinfoil",
        mode="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        model_ids=[TINFOIL_SELECTED_MODEL_ID],
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
            "require_model_attestations": True,
            "model_attestation_targets": {
                TINFOIL_SELECTED_MODEL_ID: {
                    "host": TINFOIL_SELECTED_MODEL_HOST,
                    "repo": TINFOIL_SELECTED_MODEL_REPO,
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_MODEL_RELEASE_DIGEST,
                }
            },
        },
    )
    evidence_digest = _test_digest("tinfoil-evidence")
    nonce = "nonce"
    attestation_report_digest = _test_digest("tinfoil-report")
    model_report_digest = _test_digest("model-attestation-report")

    with pytest.raises(
        ValueError,
        match="release_digest claim does not match policy",
    ):
        _validate_tinfoil_verifier_result(
            verifier_result={
                "verified": True,
                "verifier": "unit-test-tinfoil-verifier",
                "policy_digest": policy.digest,
                "evidence_digest": evidence_digest,
                "verification_nonce": nonce,
                "claims": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "attestation_format": TINFOIL_EHBP_ATTESTATION_FORMAT,
                    "attestation_report_digest": attestation_report_digest,
                    "attested_hpke_public_key_hex": "11" * 32,
                    "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                    "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                    "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verification_steps": _ehbp_verification_steps(),
                    "model_attestations": {
                        TINFOIL_SELECTED_MODEL_ID: {
                            "repo": TINFOIL_SELECTED_MODEL_REPO,
                            "attestation_format": TINFOIL_EHBP_ATTESTATION_FORMAT,
                            "attestation_report_digest": model_report_digest,
                            "attested_hpke_public_key_hex": "11" * 32,
                            "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                            "enclave_measurement_fingerprint": (
                                EHBP_ENCLAVE_MEASUREMENT
                            ),
                            "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                            "verification_steps": _ehbp_verification_steps(),
                        }
                    },
                },
            },
            policy=policy,
            evidence_digest=evidence_digest,
            verification_nonce=nonce,
            attestation_format=TINFOIL_EHBP_ATTESTATION_FORMAT,
            attestation_report_digest=attestation_report_digest,
            ehbp_public_key_hex="11" * 32,
            ehbp_key_config_b64=base64.b64encode(b"hpke-key-config").decode("ascii"),
            ehbp_claims={},
            model_attestations=[
                {
                    "model_id": TINFOIL_SELECTED_MODEL_ID,
                    "attestation": {
                        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
                        "report_digest": model_report_digest,
                    },
                    "hpke_public_key_hex": "11" * 32,
                }
            ],
        )


def test_privatemode_verifier_result_rejects_non_string_verifier_identity() -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="privatemode",
        mode="privatemode",
        base_url="http://127.0.0.1:8080/v1",
        policy={
            "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
            "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
            "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
        },
    )
    evidence_digest = _test_digest("privatemode-evidence")
    nonce = "nonce"

    with pytest.raises(ValueError, match="verifier identity must be a string"):
        _validate_privatemode_verifier_result(
            verifier_result={
                "verified": True,
                "verifier": 123,
                "policy_digest": policy.digest,
                "evidence_digest": evidence_digest,
                "verification_nonce": nonce,
                "claims": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
                    "secret_service_measurement": (
                        PRIVATEMODE_SECRET_SERVICE_MEASUREMENT
                    ),
                    "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                    "gpu_attestation_policy": "nvidia-ocsp-good-only",
                    "key_release_binding": PRIVATEMODE_KEY_RELEASE_BINDING,
                    **_privatemode_proof_claims(),
                    "verification_steps": _privatemode_verification_steps(),
                },
            },
            policy=policy,
            evidence_digest=evidence_digest,
            verification_nonce=nonce,
            proxy_base_url="http://127.0.0.1:8080/v1",
        )


def test_privatemode_verifier_result_rejects_missing_proxy_boundary_claims() -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="privatemode",
        mode="privatemode",
        base_url="http://127.0.0.1:8080/v1",
        model_ids=["privatemode/kimi-k2-6"],
        policy={
            "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
            "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
            "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
            "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
            "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
            "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
        },
    )
    evidence_digest = _test_digest("privatemode-evidence")
    nonce = "nonce"

    with pytest.raises(
        ValueError,
        match="transport claim does not match policy",
    ):
        _validate_privatemode_verifier_result(
            verifier_result={
                "verified": True,
                "verifier": "unit-test-privatemode-verifier",
                "policy_digest": policy.digest,
                "evidence_digest": evidence_digest,
                "verification_nonce": nonce,
                "claims": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
                    "secret_service_measurement": (
                        PRIVATEMODE_SECRET_SERVICE_MEASUREMENT
                    ),
                    "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                    "gpu_attestation_policy": "nvidia-ocsp-good-only",
                    "key_release_binding": PRIVATEMODE_MODEL_WORKLOAD_KEY_RELEASE_BINDING,
                    **_privatemode_proof_claims(
                        model_workload_binding_digest=(
                            PRIVATEMODE_MODEL_WORKLOAD_BINDING_DIGEST
                        )
                    ),
                    "verification_steps": _privatemode_verification_steps(),
                },
            },
            policy=policy,
            evidence_digest=evidence_digest,
            verification_nonce=nonce,
            proxy_base_url="http://127.0.0.1:8080/v1",
        )


def test_privatemode_verifier_result_rejects_non_string_gpu_policy() -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="privatemode",
        mode="privatemode",
        base_url="http://127.0.0.1:8080/v1",
        policy={
            "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
            "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
            "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
        },
    )
    evidence_digest = _test_digest("privatemode-evidence")
    nonce = "nonce"

    with pytest.raises(
        ValueError,
        match="gpu_attestation_policy claim must be a string",
    ):
        _validate_privatemode_verifier_result(
            verifier_result={
                "verified": True,
                "verifier": "unit-test-privatemode-verifier",
                "policy_digest": policy.digest,
                "evidence_digest": evidence_digest,
                "verification_nonce": nonce,
                "claims": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
                    "secret_service_measurement": (
                        PRIVATEMODE_SECRET_SERVICE_MEASUREMENT
                    ),
                    "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                    "gpu_attestation_policy": 123,
                    "key_release_binding": PRIVATEMODE_KEY_RELEASE_BINDING,
                    **_privatemode_proof_claims(),
                    "verification_steps": _privatemode_verification_steps(),
                },
            },
            policy=policy,
            evidence_digest=evidence_digest,
            verification_nonce=nonce,
            proxy_base_url="http://127.0.0.1:8080/v1",
        )


def test_privatemode_verifier_result_rejects_non_boolean_verification_step() -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="privatemode",
        mode="privatemode",
        base_url="http://127.0.0.1:8080/v1",
        model_ids=["privatemode/kimi-k2-6"],
        policy={
            "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
            "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
            "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
            "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
            "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
            "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
        },
    )
    evidence_digest = _test_digest("privatemode-evidence")
    nonce = "nonce"
    verification_steps: dict[str, object] = dict(_privatemode_verification_steps())
    verification_steps["Authorization"] = "SECRET_STEP"

    with pytest.raises(
        ValueError,
        match="verification_steps values must be booleans",
    ):
        _validate_privatemode_verifier_result(
            verifier_result={
                "verified": True,
                "verifier": "unit-test-privatemode-verifier",
                "policy_digest": policy.digest,
                "evidence_digest": evidence_digest,
                "verification_nonce": nonce,
                "claims": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
                    "secret_service_measurement": (
                        PRIVATEMODE_SECRET_SERVICE_MEASUREMENT
                    ),
                    "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                    "gpu_attestation_policy": "nvidia-ocsp-good-only",
                    "key_release_binding": (
                        PRIVATEMODE_MODEL_WORKLOAD_KEY_RELEASE_BINDING
                    ),
                    **_privatemode_proof_claims(
                        model_workload_binding_digest=(
                            PRIVATEMODE_MODEL_WORKLOAD_BINDING_DIGEST
                        )
                    ),
                    "verification_steps": verification_steps,
                },
            },
            policy=policy,
            evidence_digest=evidence_digest,
            verification_nonce=nonce,
            proxy_base_url="http://127.0.0.1:8080/v1",
        )


def test_privatemode_verifier_result_rejects_malformed_component_digest() -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="privatemode",
        mode="privatemode",
        base_url="http://127.0.0.1:8080/v1",
        policy={
            "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
            "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
            "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
        },
    )
    evidence_digest = _test_digest("privatemode-evidence")
    nonce = "nonce"

    with pytest.raises(
        ValueError,
        match="coordinator_measurement claim must be a sha256 digest",
    ):
        _validate_privatemode_verifier_result(
            verifier_result={
                "verified": True,
                "verifier": "unit-test-privatemode-verifier",
                "policy_digest": policy.digest,
                "evidence_digest": evidence_digest,
                "verification_nonce": nonce,
                "claims": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "coordinator_measurement": "coordinator-placeholder",
                    "secret_service_measurement": (
                        PRIVATEMODE_SECRET_SERVICE_MEASUREMENT
                    ),
                    "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                    "gpu_attestation_policy": "nvidia-ocsp-good-only",
                    "key_release_binding": PRIVATEMODE_KEY_RELEASE_BINDING,
                    **_privatemode_proof_claims(),
                    "verification_steps": _privatemode_verification_steps(),
                },
            },
            policy=policy,
            evidence_digest=evidence_digest,
            verification_nonce=nonce,
            proxy_base_url="http://127.0.0.1:8080/v1",
        )


def test_privatemode_verifier_result_rejects_workload_digest_mismatch() -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="privatemode",
        mode="privatemode",
        base_url="http://127.0.0.1:8080/v1",
        policy={
            "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
            "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
            "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
        },
    )
    evidence_digest = _test_digest("privatemode-evidence")
    nonce = "nonce"
    proof_claims = _privatemode_proof_claims()
    proof_claims["expected_workload_identity_digest"] = _test_digest(
        "wrong-workload-policy"
    )

    with pytest.raises(
        ValueError,
        match="expected_workload_identity_digest claim does not match policy",
    ):
        _validate_privatemode_verifier_result(
            verifier_result={
                "verified": True,
                "verifier": "unit-test-privatemode-verifier",
                "policy_digest": policy.digest,
                "evidence_digest": evidence_digest,
                "verification_nonce": nonce,
                "claims": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
                    "secret_service_measurement": (
                        PRIVATEMODE_SECRET_SERVICE_MEASUREMENT
                    ),
                    "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                    "gpu_attestation_policy": "nvidia-ocsp-good-only",
                    "key_release_binding": PRIVATEMODE_KEY_RELEASE_BINDING,
                    **proof_claims,
                    "verification_steps": _privatemode_verification_steps(),
                },
            },
            policy=policy,
            evidence_digest=evidence_digest,
            verification_nonce=nonce,
            proxy_base_url="http://127.0.0.1:8080/v1",
        )


def test_privatemode_verifier_result_rejects_scalar_workload_policy_shape() -> None:
    workload_san = "workload-kimi-k2-6"
    expected_workload_digest = _sha256_json_digest(
        {"ids": [], "sans": [workload_san]}
    )
    model_workload_digest = _sha256_json_digest(
        {
            "privatemode/kimi-k2-6": {
                "workload_ids": [],
                "workload_sans": [workload_san],
            }
        }
    )
    key_release_binding = _sha256_json_digest(
        {
            "attested_workload_policy_digest": PRIVATEMODE_AI_WORKER_MEASUREMENT,
            "expected_workload_identity_digest": expected_workload_digest,
            "model_workload_binding_digest": model_workload_digest,
            "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
            "mesh_ca_digest": PRIVATEMODE_MESH_CA_DIGEST,
            "secret_service_certificate_digest": (
                PRIVATEMODE_SECRET_SERVICE_CERTIFICATE_DIGEST
            ),
            "inference_secret_id_digest": PRIVATEMODE_INFERENCE_SECRET_ID_DIGEST,
            "nvidia_ocsp_policy_mac_digest": PRIVATEMODE_NVIDIA_OCSP_POLICY_MAC_DIGEST,
        }
    )
    policy = ConfidentialVerifierPolicy(
        provider_type="privatemode",
        mode="privatemode",
        base_url="http://127.0.0.1:8080/v1",
        model_ids=["privatemode/kimi-k2-6"],
        policy={
            "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
            "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
            "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
            "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
            "expected_workload_sans": workload_san,
            "model_workload_bindings": {
                "privatemode/kimi-k2-6": {
                    "workload_sans": [workload_san],
                }
            },
        },
    )
    evidence_digest = _test_digest("privatemode-evidence")
    nonce = "nonce"
    proof_claims = _privatemode_proof_claims(
        model_workload_binding_digest=model_workload_digest
    )
    proof_claims["expected_workload_identity_digest"] = expected_workload_digest

    with pytest.raises(
        ValueError,
        match="expected_workload_identity_digest claim does not match policy",
    ):
        _validate_privatemode_verifier_result(
            verifier_result={
                "verified": True,
                "verifier": "unit-test-privatemode-verifier",
                "policy_digest": policy.digest,
                "evidence_digest": evidence_digest,
                "verification_nonce": nonce,
                "claims": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
                    "secret_service_measurement": (
                        PRIVATEMODE_SECRET_SERVICE_MEASUREMENT
                    ),
                    "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                    "gpu_attestation_policy": "nvidia-ocsp-good-only",
                    "key_release_binding": key_release_binding,
                    **proof_claims,
                    "verification_steps": _privatemode_verification_steps(),
                },
            },
            policy=policy,
            evidence_digest=evidence_digest,
            verification_nonce=nonce,
            proxy_base_url="http://127.0.0.1:8080/v1",
        )


def test_privatemode_verifier_result_rejects_non_string_manifest_log_entry() -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="privatemode",
        mode="privatemode",
        base_url="http://127.0.0.1:8080/v1",
        policy={
            "manifest_log_dir": "/var/lib/privatemode/manifests",
            "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
            "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
        },
    )
    evidence_digest = _test_digest("privatemode-evidence")
    nonce = "nonce"

    with pytest.raises(
        ValueError,
        match="manifest_log_entry claim must be a string",
    ):
        _validate_privatemode_verifier_result(
            verifier_result={
                "verified": True,
                "verifier": "unit-test-privatemode-verifier",
                "policy_digest": policy.digest,
                "evidence_digest": evidence_digest,
                "verification_nonce": nonce,
                "claims": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "manifest_log_digest": PRIVATEMODE_MANIFEST_LOG_DIGEST,
                    "manifest_log_manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "manifest_log_entry": {"entry": "2026-05-29 manifest.json"},
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
                    "secret_service_measurement": (
                        PRIVATEMODE_SECRET_SERVICE_MEASUREMENT
                    ),
                    "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                    "gpu_attestation_policy": "nvidia-ocsp-good-only",
                    "key_release_binding": PRIVATEMODE_KEY_RELEASE_BINDING,
                    **_privatemode_proof_claims(),
                    "verification_steps": _privatemode_verification_steps(),
                },
            },
            policy=policy,
            evidence_digest=evidence_digest,
            verification_nonce=nonce,
            proxy_base_url="http://127.0.0.1:8080/v1",
        )


class DummyResponse:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._data


class DummyAsyncClient:
    calls: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def __aenter__(self) -> "DummyAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> DummyResponse:
        self.calls.append({"url": url, "headers": headers or {}})
        return DummyResponse(
            {
                "data": [
                    {
                        "id": "secure-model",
                        "created": 1_700_000_000,
                        "owned_by": "provider",
                    }
                ]
            }
        )


class DummyTinfoilCatalogClient:
    calls: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def __aenter__(self) -> "DummyTinfoilCatalogClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> DummyResponse:
        self.calls.append({"url": url, "headers": headers or {}})
        return DummyResponse(
            {
                "data": [
                    {
                        "id": "nomic-embed-text",
                        "created": 1_700_000_000,
                        "owned_by": "tinfoil",
                        "type": "embedding",
                        "context_window": 8192,
                        "endpoints": ["/v1/embeddings"],
                        "pricing": {
                            "inputTokenPricePer1M": 0.05,
                            "outputTokenPricePer1M": 0,
                            "requestPrice": 0,
                        },
                    },
                    {
                        "id": "voxtral-small-24b",
                        "created": 1_700_000_001,
                        "owned_by": "tinfoil",
                        "type": "audio",
                        "context_window": 32000,
                        "endpoints": [
                            "/v1/audio/transcriptions",
                            "/v1/audio/translations",
                            "/v1/chat/completions",
                        ],
                        "pricing": {
                            "inputTokenPricePer1M": 0.2,
                            "outputTokenPricePer1M": 0.6,
                            "requestPrice": 0,
                        },
                    },
                    {
                        "id": "qwen3-tts",
                        "created": 1_700_000_002,
                        "owned_by": "tinfoil",
                        "type": "tts",
                        "endpoints": ["/v1/audio/speech"],
                        "pricing": {
                            "inputTokenPricePer1M": 0,
                            "outputTokenPricePer1M": 0,
                            "requestPrice": 0.01,
                        },
                    },
                    {
                        "id": "websearch",
                        "created": 1_700_000_003,
                        "owned_by": "tinfoil",
                        "type": "tool",
                        "context_window": 8192,
                        "endpoints": [
                            "/v1/chat/completions",
                            "/v1/responses",
                        ],
                        "pricing": {
                            "inputTokenPricePer1M": 0,
                            "outputTokenPricePer1M": 0,
                            "requestPrice": 0,
                        },
                    },
                    {
                        "id": "malformed-tinfoil-endpoints",
                        "created": 1_700_000_004,
                        "owned_by": "tinfoil",
                        "type": "chat",
                        "context_window": 8192,
                        "endpoints": [
                            "/v1/chat/completions",
                            {"unexpected": "object"},
                        ],
                        "pricing": {
                            "inputTokenPricePer1M": 0,
                            "outputTokenPricePer1M": 0,
                            "requestPrice": 0,
                        },
                    },
                ]
            }
        )


class NonCanonicalTinfoilCatalogClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def __aenter__(self) -> "NonCanonicalTinfoilCatalogClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> DummyResponse:
        return DummyResponse(
            {
                "data": [
                    {
                        "id": "nan-priced-tinfoil-model",
                        "created": 1_700_000_000,
                        "owned_by": "tinfoil",
                        "type": "chat",
                        "context_window": 8192,
                        "endpoints": ["/v1/chat/completions"],
                        "pricing": {
                            "inputTokenPricePer1M": float("nan"),
                            "outputTokenPricePer1M": 0,
                            "requestPrice": 0,
                        },
                    }
                ]
            }
        )


class DummyPrivatemodeCatalogResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "gpt-oss-120b",
                    "object": "model",
                    "tasks": ["generate", "tool_calling"],
                },
                {
                    "id": "gemma-4-31b",
                    "object": "model",
                    "tasks": ["generate", "tool_calling", "vision"],
                },
                {
                    "id": "qwen3-embedding-4b",
                    "object": "model",
                    "tasks": ["embed"],
                },
                {
                    "id": "whisper-large-v3",
                    "object": "model",
                    "endpoints": ["/v1/audio/transcriptions"],
                },
                {
                    "id": "malformed-privatemode-tasks",
                    "object": "model",
                    "tasks": ["generate", {"unexpected": "object"}],
                },
                {
                    "id": "malformed-privatemode-endpoints",
                    "object": "model",
                    "endpoints": [
                        "/v1/audio/transcriptions",
                        {"unexpected": "object"},
                    ],
                },
            ],
        }


class DummyPrivatemodeCatalogClient:
    calls: list[dict[str, object]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def __aenter__(self) -> "DummyPrivatemodeCatalogClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> DummyPrivatemodeCatalogResponse:
        self.calls.append({"url": url, "headers": headers or {}})
        return DummyPrivatemodeCatalogResponse()


class NonCanonicalPrivatemodeCatalogResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "nan-context-privatemode-model",
                    "object": "model",
                    "tasks": ["generate"],
                    "context_window": float("nan"),
                }
            ],
        }


class NonCanonicalPrivatemodeCatalogClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def __aenter__(self) -> "NonCanonicalPrivatemodeCatalogClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> NonCanonicalPrivatemodeCatalogResponse:
        return NonCanonicalPrivatemodeCatalogResponse()


class DummyVerifierResponse:
    def __init__(
        self,
        *,
        data: dict[str, Any] | None = None,
        content: bytes = b"",
    ) -> None:
        self._data = data
        self.content = content

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        if self._data is None:
            raise ValueError("not json")
        return self._data


class DummyVerifierClient:
    calls: list[str] = []
    attestation: dict[str, Any] = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    hpke_keys: bytes = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def __aenter__(self) -> "DummyVerifierClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def get(self, url: str) -> DummyVerifierResponse:
        self.calls.append(url)
        if url.endswith("/.well-known/tinfoil-attestation"):
            return DummyVerifierResponse(data=self.attestation)
        if url.endswith("/.well-known/hpke-keys"):
            return DummyVerifierResponse(content=self.hpke_keys)
        raise AssertionError(f"unexpected URL: {url}")


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_router_only_policy_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_DIGEST,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "timeout_seconds": 1,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert DummyVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert (
        "Tinfoil require_model_attestations must be true for selected model routing"
        in (status.failure_reason or "")
    )


def test_confidential_provider_types_are_registered() -> None:
    provider_types = {provider.provider_type for provider in upstream_provider_classes}

    assert "tinfoil" in provider_types
    assert "ppq-private" in provider_types
    assert "privatemode" in provider_types


def test_confidential_provider_endpoint_capabilities_match_provider_docs() -> None:
    assert TinfoilUpstreamProvider.supports_audio_api is True
    assert TinfoilUpstreamProvider.supports_audio_transcriptions_api is True
    assert TinfoilUpstreamProvider.supports_audio_translations_api is True
    assert TinfoilUpstreamProvider.supports_audio_speech_api is False
    assert TinfoilUpstreamProvider.supports_completions_api is False
    assert TinfoilUpstreamProvider.supports_images_api is False
    assert TinfoilUpstreamProvider.supports_moderations_api is False
    assert TinfoilUpstreamProvider.supports_responses_api is True
    assert PPQPrivateUpstreamProvider.supports_audio_api is False
    assert PPQPrivateUpstreamProvider.supports_audio_transcriptions_api is False
    assert PPQPrivateUpstreamProvider.supports_audio_translations_api is False
    assert PPQPrivateUpstreamProvider.supports_audio_speech_api is False
    assert PPQPrivateUpstreamProvider.supports_completions_api is False
    assert PPQPrivateUpstreamProvider.supports_images_api is False
    assert PPQPrivateUpstreamProvider.supports_moderations_api is False
    assert PPQPrivateUpstreamProvider.supports_responses_api is False
    assert PrivatemodeUpstreamProvider.supports_audio_api is True
    assert PrivatemodeUpstreamProvider.supports_audio_transcriptions_api is True
    assert PrivatemodeUpstreamProvider.supports_audio_translations_api is False
    assert PrivatemodeUpstreamProvider.supports_audio_speech_api is False
    assert PrivatemodeUpstreamProvider.supports_completions_api is True
    assert PrivatemodeUpstreamProvider.supports_embeddings_api is True
    assert PrivatemodeUpstreamProvider.supports_images_api is False
    assert PrivatemodeUpstreamProvider.supports_moderations_api is False
    assert PrivatemodeUpstreamProvider.supports_responses_api is False


def _ppq_private_policy(
    *,
    verifier_command_digest: object | None = VALID_VERIFIER_DIGEST,
    verifier_command: object = "/usr/local/bin/routstr-ppq-private-verifier",
    include_proxy_binary_path: bool = True,
) -> dict[str, object]:
    policy: dict[str, object] = {
        "attestation_bundle_url": (
            "https://api.ppq.ai/private/v1/attestation-bundle"
        ),
        "repo": "ppq-ai/private-tee",
        "expected_release_digest": _test_digest("ppq-private-release"),
        "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
        "proxy_binary_digest": _test_digest("ppq-private-proxy"),
        "verifier_command": verifier_command,
        "require_model_attestations": True,
        "model_attestation_targets": {
            "private/gpt-oss-120b": {
                "host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                "repo": "tinfoilsh/confidential-gpt-oss-120b",
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "expected_release_digest": TINFOIL_MODEL_RELEASE_DIGEST,
            }
        },
    }
    if include_proxy_binary_path:
        policy["proxy_binary_path"] = "/opt/ppq/ppq-private-mode-proxy"
    if verifier_command_digest is not None:
        policy["verifier_command_digest"] = verifier_command_digest
    return policy


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_unpinned_verifier_command_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": _ppq_private_policy(verifier_command_digest=None),
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert DummyVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert "verifier_command_digest is required" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_malformed_verifier_digest_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": _ppq_private_policy(
                    verifier_command_digest="not-a-sha256-digest"
                ),
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert DummyVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert "verifier_command_digest must be a sha256 digest" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_allows_proxy_digest_without_proxy_path_to_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": _ppq_private_policy(
                    verifier_command_digest=_test_digest("ppq-private-verifier"),
                    include_proxy_binary_path=False,
                ),
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert any(url.endswith("/.well-known/hpke-keys") for url in DummyVerifierClient.calls)
    assert status.verified is False
    assert status.evidence_digest is not None
    assert "proxy_binary_path" not in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_verifier_command_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST_MISMATCH,
        raising=False,
    )

    provider = PPQPrivateUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": _ppq_private_policy(),
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert DummyVerifierClient.calls == [
        "https://api.ppq.ai/private/v1/attestation-bundle/.well-known/hpke-keys",
        "https://gpt-oss-120b-1.inf10.tinfoil.sh/.well-known/tinfoil-attestation",
        "https://gpt-oss-120b-1.inf10.tinfoil.sh/.well-known/hpke-keys",
    ]
    assert status.verified is False
    assert status.verified_claims == {}
    assert status.evidence_digest is not None
    assert "verifier command digest mismatch" in (status.failure_reason or "")


def test_privatemode_provider_forwards_anthropic_messages_natively() -> None:
    assert PrivatemodeUpstreamProvider.supports_anthropic_messages is True


@pytest.mark.asyncio
async def test_privatemode_fetch_models_uses_proxy_task_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPrivatemodeCatalogClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", DummyPrivatemodeCatalogClient)

    provider = PrivatemodeUpstreamProvider(
        api_key="test",
        base_url="http://127.0.0.1:8088/v1",
    )
    models = await provider.fetch_models()

    by_id = {model.id: model for model in models}
    assert by_id["gpt-oss-120b"].architecture.input_modalities == ["text"]
    assert by_id["gpt-oss-120b"].architecture.output_modalities == ["text"]
    assert by_id["gemma-4-31b"].architecture.input_modalities == ["text", "image"]
    assert by_id["gemma-4-31b"].architecture.output_modalities == ["text"]
    assert by_id["qwen3-embedding-4b"].architecture.input_modalities == ["text"]
    assert by_id["qwen3-embedding-4b"].architecture.output_modalities == ["embedding"]
    assert by_id["whisper-large-v3"].architecture.input_modalities == ["audio"]
    assert by_id["whisper-large-v3"].architecture.output_modalities == ["text"]
    assert public_supported_endpoints_for_provider(
        provider,
        model=by_id["gpt-oss-120b"],
    ) == ["/v1/chat/completions", "/v1/completions", "/v1/messages"]
    assert public_supported_endpoints_for_provider(
        provider,
        model=by_id["qwen3-embedding-4b"],
    ) == ["/v1/embeddings"]
    assert public_supported_endpoints_for_provider(
        provider,
        model=by_id["whisper-large-v3"],
    ) == ["/v1/audio/transcriptions"]
    assert (
        public_supported_endpoints_for_provider(
            provider,
            model=by_id["malformed-privatemode-tasks"],
        )
        == []
    )
    assert (
        public_supported_endpoints_for_provider(
            provider,
            model=by_id["malformed-privatemode-endpoints"],
        )
        == []
    )
    assert DummyPrivatemodeCatalogClient.calls == [
        {
            "url": "http://127.0.0.1:8088/v1/models",
            "headers": {"Authorization": "Bearer test"},
        }
    ]


@pytest.mark.asyncio
async def test_privatemode_fetch_models_rejects_non_canonical_catalog_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", NonCanonicalPrivatemodeCatalogClient)

    provider = PrivatemodeUpstreamProvider(
        api_key="test",
        base_url="http://127.0.0.1:8088/v1",
    )

    assert await provider.fetch_models() == []


@pytest.mark.asyncio
async def test_privatemode_fetch_models_rejects_remote_proxy_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPrivatemodeCatalogClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", DummyPrivatemodeCatalogClient)

    provider = PrivatemodeUpstreamProvider(
        api_key="test",
        base_url="https://api.privatemode.ai/v1",
    )

    assert await provider.fetch_models() == []
    assert DummyPrivatemodeCatalogClient.calls == []


@pytest.mark.asyncio
async def test_tinfoil_provider_starts_fail_closed_and_fetches_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyAsyncClient.calls = []
    monkeypatch.setattr("routstr.upstream.generic.httpx.AsyncClient", DummyAsyncClient)

    provider = TinfoilUpstreamProvider(api_key="test")

    status = provider.confidentiality_status()
    assert status.enabled is True
    assert status.verified is False
    assert status.mode == "tinfoil"
    assert status.failure_reason == "runtime Tinfoil verifier has not run"

    models = await provider.fetch_models()

    assert [model.id for model in models] == ["secure-model"]
    assert DummyAsyncClient.calls == [
        {
            "url": "https://inference.tinfoil.sh/v1/models",
            "headers": {"Authorization": "Bearer test"},
        }
    ]


@pytest.mark.asyncio
async def test_tinfoil_fetch_models_uses_catalog_endpoint_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyTinfoilCatalogClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.tinfoil.httpx.AsyncClient", DummyTinfoilCatalogClient
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    models = await provider.fetch_models()

    by_id = {model.id: model for model in models}
    assert by_id["nomic-embed-text"].architecture.output_modalities == ["embedding"]
    assert by_id["voxtral-small-24b"].architecture.input_modalities == ["audio", "text"]
    assert by_id["voxtral-small-24b"].architecture.output_modalities == ["text"]
    assert by_id["qwen3-tts"].architecture.input_modalities == ["text"]
    assert by_id["qwen3-tts"].architecture.output_modalities == ["audio"]
    assert by_id["websearch"].architecture.input_modalities == ["tool"]
    assert by_id["websearch"].architecture.output_modalities == ["tool"]
    assert public_supported_endpoints_for_provider(
        provider,
        model=by_id["nomic-embed-text"],
    ) == ["/v1/embeddings"]
    assert public_supported_endpoints_for_provider(
        provider,
        model=by_id["voxtral-small-24b"],
    ) == [
        "/v1/chat/completions",
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
    ]
    assert (
        public_supported_endpoints_for_provider(
            provider,
            model=by_id["qwen3-tts"],
        )
        == []
    )
    assert (
        public_supported_endpoints_for_provider(
            provider,
            model=by_id["websearch"],
        )
        == []
    )
    assert (
        public_supported_endpoints_for_provider(
            provider,
            model=by_id["malformed-tinfoil-endpoints"],
        )
        == []
    )
    translations_requirement = provider_endpoint_requirement_for_path(
        "v1/audio/translations"
    )
    assert translations_requirement == (
        "supports_audio_translations_api",
        "Audio Translations API",
    )
    translations_attr, _ = translations_requirement
    assert (
        provider_supports_required_endpoint(
            provider,
            translations_attr,
            model=by_id["qwen3-tts"],
        )
        is False
    )


@pytest.mark.asyncio
async def test_tinfoil_fetch_models_rejects_non_canonical_catalog_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.upstream.tinfoil.httpx.AsyncClient",
        NonCanonicalTinfoilCatalogClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")

    assert await provider.fetch_models() == []


@pytest.mark.asyncio
async def test_tinfoil_fetch_models_rejects_unsafe_base_url_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyTinfoilCatalogClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.tinfoil.httpx.AsyncClient", DummyTinfoilCatalogClient
    )

    provider = TinfoilUpstreamProvider(
        api_key="test",
        base_url="https://operator:secret-token@inference.tinfoil.sh/v1",
    )

    assert await provider.fetch_models() == []
    assert DummyTinfoilCatalogClient.calls == []


@pytest.mark.asyncio
async def test_tinfoil_catalog_endpoint_ceiling_survives_sats_pricing_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyTinfoilCatalogClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.tinfoil.httpx.AsyncClient", DummyTinfoilCatalogClient
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    models = await provider.fetch_models()
    voxtral = {model.id: model for model in models}["voxtral-small-24b"]

    priced_voxtral = _update_model_sats_pricing(voxtral, sats_to_usd=0.0001)

    assert priced_voxtral.supported_endpoints == [
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
        "/v1/chat/completions",
    ]
    assert public_supported_endpoints_for_provider(
        provider,
        model=priced_voxtral,
    ) == [
        "/v1/chat/completions",
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
    ]


@pytest.mark.asyncio
async def test_tinfoil_failed_refresh_clears_stale_verifier_metadata() -> None:
    provider = TinfoilUpstreamProvider(api_key="test")
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="tinfoil",
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            verifier="stale-verifier",
            policy_digest="sha256:stale-policy",
            evidence_digest="sha256:stale-evidence",
            verified_claims={"transport": "ehbp"},
        )
    )

    status = await provider.refresh_confidentiality_status()
    stored_status = provider.confidentiality_status()

    assert status.verified is False
    assert status.verifier is None
    assert status.policy_digest is None
    assert status.evidence_digest is None
    assert status.verified_claims == {}
    assert stored_status.verifier is None
    assert stored_status.evidence_digest is None
    assert stored_status.verified_claims == {}
    assert "attestation policy missing" in (stored_status.failure_reason or "")


@pytest.mark.asyncio
async def test_tinfoil_refresh_returns_sanitized_runtime_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_refresh_tinfoil_status(
        provider: TinfoilUpstreamProvider,
    ) -> ConfidentialityStatus:
        return ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="tinfoil",
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            verifier="test-verifier",
            policy_digest=_test_digest("policy"),
            evidence_digest=_test_digest("evidence"),
            verified_claims={"transport": "ehbp", "non_canonical": float("nan")},
        )

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.refresh_tinfoil_status",
        fake_refresh_tinfoil_status,
    )

    provider = TinfoilUpstreamProvider(api_key="test")

    status = await provider.refresh_confidentiality_status()
    stored_status = provider.confidentiality_status()

    assert status is stored_status
    assert status.verified is False
    assert status.verifier is None
    assert status.evidence_digest is None
    assert status.verified_claims == {}
    assert (
        status.failure_reason
        == "confidentiality verifier did not return required evidence"
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_base_url_credentials_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(
        api_key="test",
        base_url="https://operator:secret-token@inference.tinfoil.sh/v1",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert DummyVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert "provider base_url must not include credentials" in (
        status.failure_reason or ""
    )
    assert "secret-token" not in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_non_https_base_url_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(
        api_key="test",
        base_url="http://inference.tinfoil.sh/v1",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert DummyVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert "provider base_url must use https" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_non_https_policy_urls_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(
        api_key="test",
        base_url="https://inference.tinfoil.sh/v1",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": _test_digest("release"),
                    "attestation_url": "http://inference.tinfoil.sh/.well-known/tinfoil-attestation",
                    "hpke_keys_url": "http://inference.tinfoil.sh/.well-known/hpke-keys",
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    failure_reason = status.failure_reason or ""
    assert DummyVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert "attestation_url must use https" in failure_reason
    assert "hpke_keys_url must use https" in failure_reason


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_policy_host_mismatch_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(
        api_key="test",
        base_url="https://inference.tinfoil.sh/v1",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": _test_digest("release"),
                    "expected_enclave_host": "other.tinfoil.sh",
                    "attestation_url": "https://other.tinfoil.sh/.well-known/tinfoil-attestation",
                    "hpke_keys_url": "https://other.tinfoil.sh/.well-known/hpke-keys",
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    failure_reason = status.failure_reason or ""
    assert DummyVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert "expected_enclave_host must match provider base_url host" in failure_reason
    assert "attestation_url host must match provider base_url host" in failure_reason
    assert "hpke_keys_url host must match provider base_url host" in failure_reason


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_non_string_policy_urls_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(
        api_key="test",
        base_url="https://inference.tinfoil.sh/v1",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": _test_digest("release"),
                    "expected_enclave_host": {"host": "inference.tinfoil.sh"},
                    "attestation_url": {
                        "url": "https://inference.tinfoil.sh/.well-known/tinfoil-attestation"
                    },
                    "hpke_keys_url": 123,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    failure_reason = status.failure_reason or ""
    assert DummyVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert "expected_enclave_host must be a string" in failure_reason
    assert "attestation_url must be a string" in failure_reason
    assert "hpke_keys_url must be a string" in failure_reason


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_policy_mode_mismatch_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["tinfoil/secure-model"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": _test_digest("release"),
                    "verifier_command": [sys.executable, str(VERIFIER_STUB_PATH)],
                    "verifier_artifact_path": str(VERIFIER_STUB_PATH),
                    "verifier_command_digest": _sha256_file_digest(VERIFIER_STUB_PATH),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert DummyVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert (
        status.failure_reason
        == "confidentiality mode must be tinfoil for tinfoil provider"
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_requires_artifact_identity_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyVerifierClient.calls == []
    assert "expected_release_digest or allowed_release_digests is required" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_conflicting_repo_aliases_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_repo": "attacker/conflicting-router",
                    "expected_release_digest": "sha256:" + ("a" * 64),
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyVerifierClient.calls == []
    assert "repo aliases must match" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_tinfoil_refresh_requires_release_digest_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyVerifierClient.calls == []
    assert "expected_release_digest or allowed_release_digests is required" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_malformed_release_digest_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "not-a-sha256-digest",
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyVerifierClient.calls == []
    assert "release digest policy values must be sha256 digests" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_malformed_release_digest_alias_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "allowed_release_digests": [{"unexpected": "object"}],
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyVerifierClient.calls == []
    assert "release digest policy values must be sha256 digests" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_release_digest_outside_allowed_policy_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-tinfoil-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "attestation_format": payload["attestation"]["format"],
                "attestation_report_digest": payload["attestation"]["report_digest"],
                "attested_hpke_public_key_hex": "11" * 32,
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "allowed_release_digests": [TINFOIL_MODEL_RELEASE_DIGEST],
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyVerifierClient.calls == []
    assert "release_digest expected value must be allowed" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_blank_release_digest_alias_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "allowed_release_digest": "",
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyVerifierClient.calls == []
    assert "release digest policy values must be sha256 digests" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_evidence_probe_failure_redacts_secret_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingVerifierClient(DummyVerifierClient):
        calls: list[str] = []

        async def get(self, url: str) -> DummyVerifierResponse:
            self.calls.append(url)
            raise RuntimeError(
                'probe failed {"raw_prompt":"SECRET_PROMPT"} '
                "api_key=SECRET_PROVIDER_KEY sk-live-secretvalue "
                "url=https://operator:secret-token@inference.tinfoil.sh/v1"
            )

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        FailingVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    failure_reason = status.failure_reason or ""
    assert status.verified is False
    assert "Tinfoil evidence probe failed" in failure_reason
    assert "api_key: [REDACTED]" in failure_reason
    assert "raw_prompt: [REDACTED]" in failure_reason
    assert "https://inference.tinfoil.sh/v1" in failure_reason
    assert "SECRET_PROVIDER_KEY" not in failure_reason
    assert "SECRET_PROMPT" not in failure_reason
    assert "sk-live-secretvalue" not in failure_reason
    assert "secret-token" not in failure_reason


@pytest.mark.asyncio
async def test_tinfoil_refresh_requires_verifier_command_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.enabled is True
    assert status.verified is False
    assert status.policy_digest is not None
    assert status.evidence_digest is None
    assert (
        status.failure_reason
        == "verifier_command or tinfoil_verifier_command is required"
    )
    assert DummyVerifierClient.calls == []


@pytest.mark.asyncio
async def test_tinfoil_refresh_marks_verified_when_command_verifies_and_key_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.time.time",
        lambda: 1_700_000_010,
    )
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        assert command == ["/usr/local/bin/routstr-tinfoil-verifier"]
        assert timeout_seconds == 1.0
        return {
            "verified": True,
            "verifier": "unit-test-tinfoil-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "verified_at": 1_700_000_000,
            "expires_at": 1_800_000_000,
            "claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "attestation_format": DummyVerifierClient.attestation["format"],
                "attestation_report_digest": payload["attestation"]["report_digest"],
                "attested_hpke_public_key_hex": "11" * 32,
                "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                "verification_steps": _ehbp_verification_steps(),
                "model_attestations": _tinfoil_model_attestation_claims(payload),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": [TINFOIL_SELECTED_MODEL_ID],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    **_tinfoil_full_model_attestation_policy(),
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "timeout_seconds": 1,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is True
    assert status.failure_reason is None
    assert status.verifier == "unit-test-tinfoil-verifier"
    assert status.verified_at == 1_700_000_000
    assert status.expires_at == 1_700_000_300
    assert status.policy_digest is not None
    assert status.evidence_digest is not None
    assert status.verified_claims["transport"] == "ehbp"
    assert status.verified_claims["attested_hpke_public_key_hex"] == "11" * 32
    assert status.verified_claims["ehbp_key_config_b64"] == base64.b64encode(
        DummyVerifierClient.hpke_keys
    ).decode("ascii")
    assert verifier_payloads[0]["policy"]["repo"] == (
        "tinfoilsh/confidential-model-router"
    )
    assert (
        status.verified_claims["payload_policy_digest"]
        == verifier_payloads[0]["policy_digest"]
    )
    assert (
        status.verified_claims["payload_evidence_digest"]
        == verifier_payloads[0]["evidence_digest"]
    )
    assert (
        status.verified_claims["payload_verification_nonce"]
        == verifier_payloads[0]["verification_nonce"]
    )
    assert (
        verifier_payloads[0]["attestation"]["format"]
        == (DummyVerifierClient.attestation["format"])
    )
    assert (
        verifier_payloads[0]["attestation"]["body"]
        == (DummyVerifierClient.attestation["body"])
    )
    assert verifier_payloads[0]["attestation"]["body_encoding"] == "base64+gzip"
    assert verifier_payloads[0]["attestation"]["report_b64"] == base64.b64encode(
        b"attestation-report"
    ).decode("ascii")
    assert verifier_payloads[0]["hpke_key_config_b64"] == base64.b64encode(
        DummyVerifierClient.hpke_keys
    ).decode("ascii")
    assert verifier_payloads[0]["verifier_command_digest"] == VALID_VERIFIER_DIGEST
    assert isinstance(verifier_payloads[0]["verification_nonce"], str)
    assert verifier_payloads[0]["verification_nonce"]


@pytest.mark.asyncio
async def test_tinfoil_refresh_collects_and_requires_model_attestation_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.time.time",
        lambda: 1_700_000_010,
    )
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        model_attestations = payload["model_attestations"]
        assert len(model_attestations) == 1
        return {
            "verified": True,
            "verifier": "unit-test-tinfoil-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "verified_at": 1_700_000_000,
            "claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "attestation_format": DummyVerifierClient.attestation["format"],
                "attestation_report_digest": payload["attestation"]["report_digest"],
                "attested_hpke_public_key_hex": "11" * 32,
                "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                "verification_steps": _ehbp_verification_steps(),
                "model_attestations": {
                    "tinfoil/kimi-k2-6": {
                        "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                        "attestation_format": model_attestations[0]["attestation"][
                            "format"
                        ],
                        "attestation_report_digest": model_attestations[0][
                            "attestation"
                        ]["report_digest"],
                        "attested_hpke_public_key_hex": "11" * 32,
                        "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                        "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                        "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                        "release_digest": TINFOIL_MODEL_RELEASE_DIGEST,
                        "verification_steps": _ehbp_verification_steps(),
                    }
                },
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/kimi-k2-6": {
                            "host": "kimi-k2-6.inf13.tinfoil.sh",
                            "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                            "expected_code_measurement_fingerprint": (
                                EHBP_CODE_MEASUREMENT
                            ),
                            "expected_release_digest": TINFOIL_MODEL_RELEASE_DIGEST,
                        }
                    },
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "timeout_seconds": 1,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is True
    model_claims = status.verified_claims["model_attestations"]
    assert model_claims["tinfoil/kimi-k2-6"]["repo"] == (
        "tinfoilsh/confidential-kimi-k2-6-b200"
    )
    assert verifier_payloads[0]["model_attestations"][0]["model_id"] == (
        "tinfoil/kimi-k2-6"
    )
    assert verifier_payloads[0]["model_ids"] == ["tinfoil/kimi-k2-6"]
    assert verifier_payloads[0]["model_attestations"][0]["attestation_url"] == (
        "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation"
    )
    assert (
        "https://kimi-k2-6.inf13.tinfoil.sh/.well-known/tinfoil-attestation"
        in DummyVerifierClient.calls
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_non_canonical_attestation_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.time.time",
        lambda: 1_700_000_010,
    )
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
        "non_canonical": float("nan"),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-tinfoil-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "verified_at": 1_700_000_000,
            "claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "attestation_format": payload["attestation"]["format"],
                "attestation_report_digest": payload["attestation"]["report_digest"],
                "attested_hpke_public_key_hex": "12" * 32,
                "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                "verification_steps": _ehbp_verification_steps(),
                "model_attestations": _tinfoil_model_attestation_claims(payload),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "timeout_seconds": 1,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert "Tinfoil evidence probe failed" in (status.failure_reason or "")
    assert "canonical JSON" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_stale_verified_at_despite_future_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.time.time",
        lambda: 1_700_000_400,
    )
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-tinfoil-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "verified_at": 1_700_000_000,
            "expires_at": 1_800_000_000,
            "claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "attestation_format": DummyVerifierClient.attestation["format"],
                "attestation_report_digest": payload["attestation"]["report_digest"],
                "attested_hpke_public_key_hex": "11" * 32,
                "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_at is None
    assert status.expires_at is None
    assert "verifier result is stale" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_future_verified_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.time.time",
        lambda: 1_700_000_010,
    )
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-tinfoil-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "verified_at": 1_700_000_020,
            "expires_at": 1_700_000_320,
            "claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "attestation_format": DummyVerifierClient.attestation["format"],
                "attestation_report_digest": payload["attestation"]["report_digest"],
                "attested_hpke_public_key_hex": "11" * 32,
                "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_at is None
    assert status.expires_at is None
    assert "verifier result verified_at is in the future" in (
        status.failure_reason or ""
    )


@pytest.mark.parametrize(
    ("policy_fragment", "expected_failure"),
    (
        (
            {"max_verifier_age_seconds": "300"},
            "max_verifier_age_seconds must be an integer",
        ),
        (
            {"max_verifier_age_seconds": None},
            "max_verifier_age_seconds must be an integer",
        ),
        (
            {"max_verifier_age_seconds": 300, "max_evidence_age_seconds": None},
            "max_evidence_age_seconds must be an integer",
        ),
        (
            {"max_verifier_age_seconds": 300, "max_evidence_age_seconds": 301},
            "max_verifier_age_seconds and max_evidence_age_seconds aliases must match",
        ),
    ),
)
@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_malformed_max_verifier_age_policy(
    monkeypatch: pytest.MonkeyPatch,
    policy_fragment: dict[str, object],
    expected_failure: str,
) -> None:
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.time.time",
        lambda: 1_700_000_010,
    )
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-tinfoil-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "verified_at": 1_700_000_000,
            "claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "attestation_format": DummyVerifierClient.attestation["format"],
                "attestation_report_digest": payload["attestation"]["report_digest"],
                "attested_hpke_public_key_hex": "11" * 32,
                "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **policy_fragment,
                    "timeout_seconds": 1,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert expected_failure in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_tinfoil_refresh_accepts_prefixed_expected_release_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operators may paste release pins as sha256:<hex>; verifier claims use hex."""
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-tinfoil-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "attestation_format": payload["attestation"]["format"],
                "attestation_report_digest": payload["attestation"]["report_digest"],
                "attested_hpke_public_key_hex": "11" * 32,
                "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                "verification_steps": _ehbp_verification_steps(),
                "model_attestations": _tinfoil_model_attestation_claims(payload),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": [TINFOIL_SELECTED_MODEL_ID],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_DIGEST,
                    **_tinfoil_full_model_attestation_policy(),
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is True
    assert status.failure_reason is None
    assert status.verified_claims["release_digest"] == TINFOIL_ROUTER_RELEASE_HEX


@pytest.mark.asyncio
async def test_tinfoil_refresh_accepts_prefixed_measurement_pin_with_raw_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    raw_code_measurement = EHBP_CODE_MEASUREMENT.removeprefix("sha256:")
    raw_enclave_measurement = EHBP_ENCLAVE_MEASUREMENT.removeprefix("sha256:")

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-tinfoil-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "attestation_format": payload["attestation"]["format"],
                "attestation_report_digest": payload["attestation"]["report_digest"],
                "attested_hpke_public_key_hex": "11" * 32,
                "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                "enclave_measurement_fingerprint": raw_enclave_measurement,
                "code_measurement_fingerprint": raw_code_measurement,
                "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                "verification_steps": _ehbp_verification_steps(),
                "model_attestations": _tinfoil_model_attestation_claims(payload),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": [TINFOIL_SELECTED_MODEL_ID],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    **_tinfoil_full_model_attestation_policy(),
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is True
    assert status.failure_reason is None
    assert (
        status.verified_claims["code_measurement_fingerprint"] == raw_code_measurement
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_evidence_digest_tracks_verifier_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )
    call_count = 0

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {
            "verified": True,
            "verifier": "unit-test-tinfoil-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "attestation_format": payload["attestation"]["format"],
                "attestation_report_digest": payload["attestation"]["report_digest"],
                "attested_hpke_public_key_hex": "11" * 32,
                "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                "enclave_measurement_fingerprint": _test_digest(
                    f"ehbp-enclave-measurement-{call_count}"
                ),
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                "verification_steps": _ehbp_verification_steps(),
                "model_attestations": _tinfoil_model_attestation_claims(payload),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": [TINFOIL_SELECTED_MODEL_ID],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    **_tinfoil_full_model_attestation_policy(),
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    first_status = await provider.refresh_confidentiality_status()
    second_status = await provider.refresh_confidentiality_status()

    assert first_status.verified is True
    assert second_status.verified is True
    assert first_status.policy_digest == second_status.policy_digest
    assert first_status.evidence_digest != second_status.evidence_digest
    assert (
        first_status.evidence_digest
        == first_status.verified_claims["runtime_evidence_digest"]
    )
    assert (
        second_status.evidence_digest
        == second_status.verified_claims["runtime_evidence_digest"]
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_verifier_without_full_step_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-tinfoil-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "attestation_format": payload["attestation"]["format"],
                "attestation_report_digest": payload["attestation"]["report_digest"],
                "attested_hpke_public_key_hex": "12" * 32,
                "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "verification_steps must be a JSON object" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_release_digest_outside_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-tinfoil-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "attestation_format": payload["attestation"]["format"],
                "attestation_report_digest": payload["attestation"]["report_digest"],
                "attested_hpke_public_key_hex": "11" * 32,
                "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_MODEL_RELEASE_DIGEST,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "release_digest claim does not match policy" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_tinfoil_refresh_uses_real_digest_pinned_verifier_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.time.time",
        lambda: 1_700_000_010,
    )
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": [TINFOIL_SELECTED_MODEL_ID],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": (
                        EHBP_FIXTURE_CODE_MEASUREMENT
                    ),
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    **_tinfoil_full_model_attestation_policy(
                        code_measurement=EHBP_FIXTURE_CODE_MEASUREMENT
                    ),
                    "timeout_seconds": 2,
                    "verifier_artifact_path": str(VERIFIER_STUB_PATH),
                    "verifier_command": [sys.executable, str(VERIFIER_STUB_PATH)],
                    "verifier_command_digest": _sha256_file_digest(VERIFIER_STUB_PATH),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is True
    assert status.failure_reason is None
    assert status.verifier == "routstr-test-confidential-verifier"
    assert status.verified_at == 1_700_000_000
    assert status.expires_at == 1_700_000_300
    assert status.verified_claims["repo"] == "tinfoilsh/confidential-model-router"
    assert status.verified_claims["payload_policy_digest"] == status.policy_digest
    assert (
        status.verified_claims["payload_evidence_digest"]
        != status.verified_claims["runtime_evidence_digest"]
    )
    assert status.verified_claims["runtime_evidence_digest"] == status.evidence_digest
    assert status.verified_claims["payload_verification_nonce"]
    assert status.verified_claims["attestation_body_encoding"] == "base64+gzip"
    assert status.verified_claims["attestation_report_digest"].startswith("sha256:")
    assert status.verified_claims["attested_hpke_public_key_hex"] == "11" * 32
    assert status.verified_claims["transport"] == "ehbp"


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_verifier_without_tls_binding_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-tinfoil-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "attestation_format": payload["attestation"]["format"],
                "attestation_report_digest": payload["attestation"]["report_digest"],
                "attested_hpke_public_key_hex": "11" * 32,
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "TLS public key binding claim is required" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_placeholder_tls_fingerprint_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-tinfoil-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "attestation_format": payload["attestation"]["format"],
                "attestation_report_digest": payload["attestation"]["report_digest"],
                "attested_hpke_public_key_hex": "11" * 32,
                "tls_public_key_fingerprint_sha256": "tls-fp",
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "TLS public key fingerprint claim must be a sha256 digest" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_placeholder_measurement_fingerprint_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-tinfoil-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "attestation_format": payload["attestation"]["format"],
                "attestation_report_digest": payload["attestation"]["report_digest"],
                "attested_hpke_public_key_hex": "11" * 32,
                "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                "enclave_measurement_fingerprint": "enclave-fp",
                "code_measurement_fingerprint": "code-fp",
                "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "enclave_measurement_fingerprint claim must be a sha256 digest" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_verifier_stderr_is_redacted_from_failure_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )
    monkeypatch.setenv("ROUTSTR_VERIFIER_STUB_BEHAVIOR", "failure_secret")

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "timeout_seconds": 2,
                    "verifier_artifact_path": str(VERIFIER_STUB_PATH),
                    "verifier_command": [sys.executable, str(VERIFIER_STUB_PATH)],
                    "verifier_command_digest": _sha256_file_digest(VERIFIER_STUB_PATH),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert "confidential verifier command failed with exit code 7" in (
        status.failure_reason or ""
    )
    assert "stderr redacted" in (status.failure_reason or "")
    assert "SECRET_PROMPT" not in (status.failure_reason or "")
    assert "sk-live-secret" not in (status.failure_reason or "")
    assert "raw-quote" not in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_verifier_output_bound_to_other_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-tinfoil-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": "sha256:other-evidence",
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "attestation_format": payload["attestation"]["format"],
                "attestation_report_digest": payload["attestation"]["report_digest"],
                "attested_hpke_public_key_hex": "11" * 32,
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "evidence_digest does not match verifier payload" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_verifier_report_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-tinfoil-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "attestation_format": payload["attestation"]["format"],
                "attestation_report_digest": "sha256:other-report",
                "attested_hpke_public_key_hex": "11" * 32,
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "attestation_report_digest claim does not match evidence" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_enclave_measurement_outside_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-tinfoil-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "attestation_format": payload["attestation"]["format"],
                "attestation_report_digest": payload["attestation"]["report_digest"],
                "attested_hpke_public_key_hex": "11" * 32,
                "enclave_measurement_fingerprint": EHBP_OTHER_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "allowed_enclave_measurement_fingerprints": [
                        EHBP_ENCLAVE_MEASUREMENT
                    ],
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "enclave_measurement_fingerprint claim does not match policy" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_code_measurement_outside_allowed_policy_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-tinfoil-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "attestation_format": payload["attestation"]["format"],
                "attestation_report_digest": payload["attestation"]["report_digest"],
                "attested_hpke_public_key_hex": "11" * 32,
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "allowed_code_measurement_fingerprints": [
                        EHBP_OTHER_CODE_MEASUREMENT
                    ],
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyVerifierClient.calls == []
    assert "code_measurement_fingerprint expected value must be allowed" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_verifier_key_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-tinfoil-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "attestation_format": payload["attestation"]["format"],
                "attestation_report_digest": payload["attestation"]["report_digest"],
                "attested_hpke_public_key_hex": "12" * 32,
                "tls_public_key_fingerprint_sha256": TINFOIL_TLS_PUBLIC_KEY_DIGEST,
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "attested HPKE public key does not match" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_unpinned_verifier_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert "verifier_command_digest is required" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_malformed_verifier_command_digest_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": "not-a-sha256-digest",
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyVerifierClient.calls == []
    assert "verifier_command_digest must be a sha256 digest" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_malformed_verifier_digest_alias_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": 123,
                    "verifier_binary_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyVerifierClient.calls == []
    assert "verifier_command_digest must be a sha256 digest" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_conflicting_verifier_digest_aliases_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "verifier_binary_digest": VALID_VERIFIER_DIGEST_MISMATCH,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyVerifierClient.calls == []
    assert "verifier_command_digest aliases must match" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_malformed_verifier_command_alias_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": {"path": "/tmp/ambiguous-verifier"},
                    "tinfoil_verifier_command": [
                        "/usr/local/bin/routstr-tinfoil-verifier"
                    ],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyVerifierClient.calls == []
    assert "verifier_command must be a command string or array" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_unbound_verifier_artifact_path_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": [sys.executable, "/tmp/evil-verifier.py"],
                    "verifier_artifact_path": str(VERIFIER_STUB_PATH),
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyVerifierClient.calls == []
    assert (
        "verifier_artifact_path must match verifier command executable or argument"
        in (status.failure_reason or "")
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_blank_verifier_artifact_path_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": [str(VERIFIER_STUB_PATH)],
                    "verifier_artifact_path": "",
                    "verifier_command_digest": _sha256_file_digest(VERIFIER_STUB_PATH),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyVerifierClient.calls == []
    assert "verifier_artifact_path must be a non-empty string" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_verifier_command_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST_MISMATCH,
        raising=False,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert "verifier command digest mismatch" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_attestation_body_that_is_not_gzip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": base64.b64encode(b"not-gzip").decode("ascii"),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert "Tinfoil attestation body is not valid gzip" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_v1_attestation_format_for_ehbp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": "https://tinfoil.sh/predicate/snp-tdx-multiplatform/v1",
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert "Tinfoil EHBP requires a v2 attestation predicate" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_malformed_ehbp_key_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = b"not-an-ohttp-key-config"
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert "invalid EHBP HPKE key config" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_unsupported_attestation_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    DummyVerifierClient.attestation = {
        "format": "https://example.invalid/not-tinfoil",
        "body": _tinfoil_attestation_body(),
    }
    DummyVerifierClient.hpke_keys = (
        b"\x00\x00\x20" + (b"\x11" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert "unsupported Tinfoil attestation format" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_tinfoil_refresh_requires_expected_measurement_source() -> None:
    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {"timeout_seconds": 1},
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert "repo or expected_repo is required" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_tinfoil_refresh_requires_model_attestation_targets_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": ["tinfoil/kimi-k2-6"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "require_model_attestations": True,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyVerifierClient.calls == []
    assert "model_attestation_targets is required" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_non_string_repo_alias_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_repo": {"name": "tinfoilsh/confidential-model-router"},
                    "expected_release_digest": TINFOIL_ROUTER_RELEASE_HEX,
                    "verifier_command": ["/usr/local/bin/routstr-tinfoil-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyVerifierClient.calls == []
    assert "expected_repo must be a non-empty string" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_inline_api_key_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "api_key": "sk-should-not-enter-policy",
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert DummyVerifierClient.calls == []
    assert "api_key must not be embedded in verifier policy" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_nested_inline_secret_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "probe_headers": {
                        "Authorization": "Bearer sk-should-not-enter-policy"
                    },
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert DummyVerifierClient.calls == []
    assert "authorization must not be embedded in verifier policy" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_refresh_token_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "credentials": {"refreshToken": "refresh-secret"},
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert DummyVerifierClient.calls == []
    assert "refresh_token must not be embedded in verifier policy" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_tinfoil_refresh_rejects_inline_secret_value_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyVerifierClient,
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "probe_headers": {
                        "X-Provider-Credential": "sk-should-not-enter-policy",
                    },
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert DummyVerifierClient.calls == []
    failure_reason = status.failure_reason or ""
    assert "secret-like value must not be embedded in verifier policy" in failure_reason
    assert "sk-should-not-enter-policy" not in failure_reason


@pytest.mark.asyncio
async def test_privatemode_provider_starts_fail_closed_and_fetches_proxy_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyAsyncClient.calls = []
    monkeypatch.setattr("routstr.upstream.generic.httpx.AsyncClient", DummyAsyncClient)

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )

    status = provider.confidentiality_status()
    assert status.enabled is True
    assert status.verified is False
    assert status.mode == "privatemode"
    assert status.failure_reason == "runtime Privatemode verifier has not run"

    models = await provider.fetch_models()

    assert [model.id for model in models] == ["secure-model"]
    assert DummyAsyncClient.calls == [
        {
            "url": "http://127.0.0.1:8080/v1/models",
            "headers": {"Authorization": "Bearer test"},
        }
    ]


@pytest.mark.asyncio
async def test_privatemode_refresh_returns_sanitized_runtime_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_refresh_privatemode_status(
        provider: PrivatemodeUpstreamProvider,
    ) -> ConfidentialityStatus:
        return ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="privatemode",
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            verifier="test-verifier",
            policy_digest=_test_digest("policy"),
            evidence_digest=_test_digest("evidence"),
            verified_claims={
                "transport": "privatemode-local-proxy",
                "non_canonical": float("nan"),
            },
        )

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.refresh_privatemode_status",
        fake_refresh_privatemode_status,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )

    status = await provider.refresh_confidentiality_status()
    stored_status = provider.confidentiality_status()

    assert status is stored_status
    assert status.verified is False
    assert status.verifier is None
    assert status.evidence_digest is None
    assert status.verified_claims == {}
    assert (
        status.failure_reason
        == "confidentiality verifier did not return required evidence"
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_insecure_proxy_policy() -> None:
    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "manifest_digest": "sha256:manifest",
                    "dump_requests": True,
                    "nvidia_ocsp_allow_unknown": True,
                    "nvidia_ocsp_revoked_grace_period_hours": 48,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.policy_digest is not None
    assert "dump_requests must be explicitly false" in (status.failure_reason or "")
    assert "nvidia_ocsp_allow_unknown must be explicitly false" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_requires_pinned_proxy_artifact() -> None:
    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "manifest_digest": "sha256:manifest",
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert "proxy_binary_digest is required for full Privatemode proxy artifact verification" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_requires_verifier_command() -> None:
    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert (
        status.failure_reason
        == "verifier_command or privatemode_verifier_command is required"
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_requires_explicit_proxy_privacy_knobs() -> None:
    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert "dump_requests must be explicitly false" in (status.failure_reason or "")
    assert "shared_prompt_cache must be explicitly false" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_falsy_non_boolean_privacy_knobs() -> None:
    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "dump_requests": "",
                    "shared_prompt_cache": "",
                    "nvidia_ocsp_allow_unknown": "",
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert "dump_requests must be explicitly false" in (status.failure_reason or "")
    assert "shared_prompt_cache must be explicitly false" in (
        status.failure_reason or ""
    )
    assert "nvidia_ocsp_allow_unknown must be explicitly false" in (
        status.failure_reason or ""
    )
    assert status.failure_reason != "cryptographic Privatemode verifier not implemented"


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_conflicting_privacy_knob_aliases() -> None:
    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "dump_requests": False,
                    "dumpRequests": True,
                    "shared_prompt_cache": False,
                    "sharedPromptCache": True,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidiaOCSPAllowUnknown": True,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "nvidiaOCSPRevokedGracePeriod": 48,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert "dump_requests must be explicitly false" in (status.failure_reason or "")
    assert "shared_prompt_cache must be explicitly false" in (
        status.failure_reason or ""
    )
    assert "nvidia_ocsp_allow_unknown must be explicitly false" in (
        status.failure_reason or ""
    )
    assert "nvidia_ocsp_revoked_grace_period_hours must be 0" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_conflicting_artifact_digest_aliases() -> (
    None
):
    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "manifestDigest": PRIVATEMODE_OTHER_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "proxyImageDigest": _test_digest("other-proxy-image"),
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert "manifest_digest aliases must match" in (status.failure_reason or "")
    assert "proxy_image_digest aliases must match" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_boolean_ocsp_grace_period() -> None:
    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": False,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert "nvidia_ocsp_revoked_grace_period_hours must be 0" in (
        status.failure_reason or ""
    )
    assert status.failure_reason != "cryptographic Privatemode verifier not implemented"


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_malformed_verifier_command_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        return {}

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": "not-a-sha256-digest",
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert verifier_payloads == []
    assert "verifier_command_digest must be a sha256 digest" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_conflicting_verifier_digest_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        return {}

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "verifier_binary_digest": VALID_VERIFIER_DIGEST_MISMATCH,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert verifier_payloads == []
    assert "verifier_command_digest aliases must match" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_unbound_verifier_artifact_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        return {}

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": [sys.executable, "/tmp/evil-verifier.py"],
                    "verifier_artifact_path": str(VERIFIER_STUB_PATH),
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert verifier_payloads == []
    assert (
        "verifier_artifact_path must match verifier command executable or argument"
        in (status.failure_reason or "")
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_inline_api_key_before_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        return {
            "verified": True,
            "verifier": "unit-test-privatemode-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                "proxy_base_url": payload["proxy_base_url"],
                "transport": "privatemode-proxy",
                "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
                "secret_service_measurement": PRIVATEMODE_SECRET_SERVICE_MEASUREMENT,
                "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                "gpu_attestation_policy": "nvidia-ocsp-good-only",
                "key_release_binding": PRIVATEMODE_MODEL_WORKLOAD_KEY_RELEASE_BINDING,
                **_privatemode_proof_claims(
                    model_workload_binding_digest=PRIVATEMODE_MODEL_WORKLOAD_BINDING_DIGEST
                ),
                "verification_steps": _privatemode_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "api_key": "sk-should-not-enter-policy",
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert verifier_payloads == []
    assert "api_key must not be embedded in verifier policy" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_base_url_credentials_before_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        return {}

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://operator:secret-token@127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": ["privatemode/kimi-k2-6"],
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert verifier_payloads == []
    assert "Privatemode proxy base_url must not include credentials" in (
        status.failure_reason or ""
    )
    assert "secret-token" not in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_policy_mode_mismatch_before_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        return {}

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": ["privatemode/gpt-secure"],
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert verifier_payloads == []
    assert (
        status.failure_reason
        == "confidentiality mode must be privatemode for privatemode provider"
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_remote_proxy_base_url_before_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        return {
            "verified": True,
            "verifier": "unit-test-privatemode-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                "proxy_base_url": payload["proxy_base_url"],
                "transport": "privatemode-proxy",
                "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
                "secret_service_measurement": PRIVATEMODE_SECRET_SERVICE_MEASUREMENT,
                "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                "gpu_attestation_policy": "nvidia-ocsp-good-only",
                "key_release_binding": PRIVATEMODE_MODEL_WORKLOAD_KEY_RELEASE_BINDING,
                **_privatemode_proof_claims(
                    model_workload_binding_digest=PRIVATEMODE_MODEL_WORKLOAD_BINDING_DIGEST
                ),
                "verification_steps": _privatemode_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://privatemode-proxy:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": ["privatemode/kimi-k2-6"],
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert verifier_payloads == []
    assert "Privatemode proxy base_url must use a loopback host" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_non_http_proxy_base_url_before_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        return {}

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="ws://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert verifier_payloads == []
    assert "Privatemode proxy base_url must use http or https" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_policy_url_credentials_before_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        return {}

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "api_base_url": "https://operator:api-secret@api.privatemode.ai",
                    "cdn_base_url": "https://operator:cdn-secret@cdn.confidential.cloud/privatemode/v2",
                    "manifest_url": "https://operator:manifest-secret@cdn.confidential.cloud/manifest.json",
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    failure_reason = status.failure_reason or ""
    assert status.verified is False
    assert verifier_payloads == []
    assert "api_base_url must not include credentials" in failure_reason
    assert "cdn_base_url must not include credentials" in failure_reason
    assert "manifest_url must not include credentials" in failure_reason
    assert "api-secret" not in failure_reason
    assert "cdn-secret" not in failure_reason
    assert "manifest-secret" not in failure_reason


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_non_https_policy_urls_before_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        return {}

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "api_base_url": "http://api.privatemode.ai",
                    "cdn_base_url": "http://cdn.confidential.cloud/privatemode/v2",
                    "manifest_url": "http://cdn.confidential.cloud/manifest.json",
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    failure_reason = status.failure_reason or ""
    assert status.verified is False
    assert verifier_payloads == []
    assert "api_base_url must use https" in failure_reason
    assert "cdn_base_url must use https" in failure_reason
    assert "manifest_url must use https" in failure_reason


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_invalid_proxy_base_url_port_before_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        return {}

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:bad/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert verifier_payloads == []
    assert "Privatemode proxy base_url must include a valid port" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_non_string_policy_urls_before_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        return {}

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "api_base_url": {"url": "https://api.privatemode.ai"},
                    "cdn_base_url": 123,
                    "manifest_url": ["https://cdn.confidential.cloud/manifest.json"],
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    failure_reason = status.failure_reason or ""
    assert status.verified is False
    assert verifier_payloads == []
    assert "api_base_url must be a string" in failure_reason
    assert "cdn_base_url must be a string" in failure_reason
    assert "manifest_url must be a string" in failure_reason


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_non_string_verifier_input_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        return {}

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "api_key_env": {"env": "PRIVATEMODE_API_KEY"},
                    "api_key_file": 123,
                    "manifest_path": ["/var/lib/privatemode/manifest.json"],
                    "manifest_b64": {"value": "e30="},
                    "manifest_log_dir": ["/var/lib/privatemode/log"],
                    "proxyBinaryPath": {"path": "/opt/privatemode/proxy"},
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    failure_reason = status.failure_reason or ""
    assert status.verified is False
    assert verifier_payloads == []
    assert "api_key_env must be a string" in failure_reason
    assert "api_key_file must be a string" in failure_reason
    assert "manifest_path must be a string" in failure_reason
    assert "manifest_b64 must be a string" in failure_reason
    assert "manifest_log_dir must be a string" in failure_reason
    assert "proxyBinaryPath must be a string" in failure_reason


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_blank_verifier_input_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        return {}

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "api_base_url": " ",
                    "cdn_base_url": "",
                    "manifest_url": " ",
                    "api_key_env": " ",
                    "api_key_file": "",
                    "manifest_path": " ",
                    "manifest_b64": "",
                    "manifest_log_dir": " ",
                    "proxyBinaryPath": "",
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    failure_reason = status.failure_reason or ""
    assert status.verified is False
    assert verifier_payloads == []
    for field in (
        "api_base_url",
        "cdn_base_url",
        "manifest_url",
        "api_key_env",
        "api_key_file",
        "manifest_path",
        "manifest_b64",
        "manifest_log_dir",
        "proxyBinaryPath",
    ):
        assert f"{field} must be a non-empty string" in failure_reason


@pytest.mark.asyncio
async def test_privatemode_verifier_failure_redacts_secret_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        raise RuntimeError(
            'verifier failed {"raw_prompt":"SECRET_PROMPT"} '
            "api_key=SECRET_PROVIDER_KEY sk-live-secretvalue "
            "url=http://operator:secret-token@127.0.0.1:8080/v1"
        )

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    failure_reason = status.failure_reason or ""
    assert status.verified is False
    assert "Privatemode verifier failed" in failure_reason
    assert "api_key: [REDACTED]" in failure_reason
    assert "raw_prompt: [REDACTED]" in failure_reason
    assert "http://127.0.0.1:8080/v1" in failure_reason
    assert "SECRET_PROVIDER_KEY" not in failure_reason
    assert "SECRET_PROMPT" not in failure_reason
    assert "sk-live-secretvalue" not in failure_reason
    assert "secret-token" not in failure_reason


@pytest.mark.asyncio
async def test_privatemode_refresh_marks_verified_when_command_verifies_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.time.time",
        lambda: 1_700_000_010,
    )
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        assert command == ["/usr/local/bin/routstr-privatemode-verifier"]
        return {
            "verified": True,
            "verifier": "unit-test-privatemode-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "verified_at": 1_700_000_000,
            "claims": {
                "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                "proxy_base_url": payload["proxy_base_url"],
                "transport": "privatemode-proxy",
                "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
                "secret_service_measurement": PRIVATEMODE_SECRET_SERVICE_MEASUREMENT,
                "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                "gpu_attestation_policy": "nvidia-ocsp-good-only",
                "key_release_binding": PRIVATEMODE_MODEL_WORKLOAD_KEY_RELEASE_BINDING,
                **_privatemode_proof_claims(
                    model_workload_binding_digest=PRIVATEMODE_MODEL_WORKLOAD_BINDING_DIGEST
                ),
                "verification_steps": _privatemode_verification_steps(),
            },
        }

    from routstr.upstream import confidential_verifiers

    original_runner = confidential_verifiers._run_confidential_verifier_command
    original_digest = confidential_verifiers._sha256_file_digest
    confidential_verifiers._run_confidential_verifier_command = fake_verifier_command
    confidential_verifiers._sha256_file_digest = lambda path: VALID_VERIFIER_DIGEST
    try:
        provider = PrivatemodeUpstreamProvider(
            base_url="http://127.0.0.1:8080/v1",
            api_key="test",
        )
        provider.configure_confidentiality_from_settings(
            {
                "confidentiality": {
                    "enabled": True,
                    "mode": "privatemode",
                    "model_ids": ["privatemode/kimi-k2-6"],
                    "policy": {
                        "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                        "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                        "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                        "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                        "dump_requests": False,
                        "shared_prompt_cache": False,
                        "nvidia_ocsp_allow_unknown": False,
                        "nvidia_ocsp_revoked_grace_period_hours": 0,
                        "verifier_command": [
                            "/usr/local/bin/routstr-privatemode-verifier"
                        ],
                        "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    },
                }
            }
        )

        status = await provider.refresh_confidentiality_status()
    finally:
        confidential_verifiers._run_confidential_verifier_command = original_runner
        confidential_verifiers._sha256_file_digest = original_digest

    assert status.verified is True
    assert status.failure_reason is None
    assert status.verifier == "unit-test-privatemode-verifier"
    assert status.verified_at == 1_700_000_000
    assert status.expires_at == 1_700_000_300
    assert status.verified_claims["manifest_digest"] == PRIVATEMODE_MANIFEST_DIGEST
    assert (
        status.verified_claims["proxy_image_digest"] == PRIVATEMODE_PROXY_IMAGE_DIGEST
    )
    assert verifier_payloads[0]["mode"] == "privatemode"
    assert (
        verifier_payloads[0]["policy"]["manifest_digest"] == PRIVATEMODE_MANIFEST_DIGEST
    )
    assert verifier_payloads[0]["model_ids"] == ["privatemode/kimi-k2-6"]
    assert verifier_payloads[0]["proxy_base_url"] == "http://127.0.0.1:8080/v1"
    assert isinstance(verifier_payloads[0]["verification_nonce"], str)
    assert verifier_payloads[0]["verification_nonce"]
    assert (
        status.verified_claims["payload_policy_digest"]
        == verifier_payloads[0]["policy_digest"]
    )
    assert (
        status.verified_claims["payload_evidence_digest"]
        == verifier_payloads[0]["evidence_digest"]
    )
    assert (
        status.verified_claims["payload_verification_nonce"]
        == verifier_payloads[0]["verification_nonce"]
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_verifier_without_selected_model_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.time.time",
        lambda: 1_700_000_010,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-privatemode-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "verified_at": 1_700_000_000,
            "claims": {
                "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                "proxy_base_url": payload["proxy_base_url"],
                "transport": "privatemode-proxy",
                "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
                "secret_service_measurement": PRIVATEMODE_SECRET_SERVICE_MEASUREMENT,
                "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                "gpu_attestation_policy": "nvidia-ocsp-good-only",
                "key_release_binding": PRIVATEMODE_MODEL_WORKLOAD_KEY_RELEASE_BINDING,
                **_privatemode_proof_claims(
                    model_workload_binding_digest=PRIVATEMODE_MODEL_WORKLOAD_BINDING_DIGEST,
                    selected_model_ids=None,
                ),
                "verification_steps": _privatemode_verification_steps(),
            },
        }

    from routstr.upstream import confidential_verifiers

    original_runner = confidential_verifiers._run_confidential_verifier_command
    original_digest = confidential_verifiers._sha256_file_digest
    confidential_verifiers._run_confidential_verifier_command = fake_verifier_command
    confidential_verifiers._sha256_file_digest = lambda path: VALID_VERIFIER_DIGEST
    try:
        provider = PrivatemodeUpstreamProvider(
            base_url="http://127.0.0.1:8080/v1",
            api_key="test",
        )
        provider.configure_confidentiality_from_settings(
            {
                "confidentiality": {
                    "enabled": True,
                    "mode": "privatemode",
                    "model_ids": ["privatemode/kimi-k2-6"],
                    "policy": {
                        "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                        "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                        "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                        "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                        "dump_requests": False,
                        "shared_prompt_cache": False,
                        "nvidia_ocsp_allow_unknown": False,
                        "nvidia_ocsp_revoked_grace_period_hours": 0,
                        "verifier_command": [
                            "/usr/local/bin/routstr-privatemode-verifier"
                        ],
                        "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    },
                }
            }
        )

        status = await provider.refresh_confidentiality_status()
    finally:
        confidential_verifiers._run_confidential_verifier_command = original_runner
        confidential_verifiers._sha256_file_digest = original_digest

    assert status.verified is False
    assert status.verified_claims == {}
    assert "selected_model_ids claim is required" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_missing_model_workload_bindings_before_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.time.time",
        lambda: 1_700_000_010,
    )
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        return {
            "verified": True,
            "verifier": "unit-test-privatemode-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "verified_at": 1_700_000_000,
            "claims": {
                "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                "proxy_base_url": payload["proxy_base_url"],
                "transport": "privatemode-proxy",
                "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
                "secret_service_measurement": PRIVATEMODE_SECRET_SERVICE_MEASUREMENT,
                "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                "gpu_attestation_policy": "nvidia-ocsp-good-only",
                "key_release_binding": PRIVATEMODE_MODEL_WORKLOAD_KEY_RELEASE_BINDING,
                **_privatemode_proof_claims(
                    model_workload_binding_digest=PRIVATEMODE_MODEL_WORKLOAD_BINDING_DIGEST
                ),
                "verification_steps": _privatemode_verification_steps(),
            },
        }

    from routstr.upstream import confidential_verifiers

    original_runner = confidential_verifiers._run_confidential_verifier_command
    original_digest = confidential_verifiers._sha256_file_digest
    confidential_verifiers._run_confidential_verifier_command = fake_verifier_command
    confidential_verifiers._sha256_file_digest = lambda path: VALID_VERIFIER_DIGEST
    try:
        provider = PrivatemodeUpstreamProvider(
            base_url="http://127.0.0.1:8080/v1",
            api_key="test",
        )
        provider.configure_confidentiality_from_settings(
            {
                "confidentiality": {
                    "enabled": True,
                    "mode": "privatemode",
                    "model_ids": ["privatemode/kimi-k2-6"],
                    "policy": {
                        "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                        "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                        "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                        "dump_requests": False,
                        "shared_prompt_cache": False,
                        "nvidia_ocsp_allow_unknown": False,
                        "nvidia_ocsp_revoked_grace_period_hours": 0,
                        "verifier_command": [
                            "/usr/local/bin/routstr-privatemode-verifier"
                        ],
                        "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    },
                }
            }
        )

        status = await provider.refresh_confidentiality_status()
    finally:
        confidential_verifiers._run_confidential_verifier_command = original_runner
        confidential_verifiers._sha256_file_digest = original_digest

    assert status.verified is False
    assert "model_workload_bindings is required" in (status.failure_reason or "")
    assert verifier_payloads == []


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_unbound_key_release_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.time.time",
        lambda: 1_700_000_010,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-privatemode-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "verified_at": 1_700_000_000,
            "claims": {
                "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
                "secret_service_measurement": PRIVATEMODE_SECRET_SERVICE_MEASUREMENT,
                "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                "gpu_attestation_policy": "nvidia-ocsp-good-only",
                "key_release_binding": _test_digest("unbound-key-release-binding"),
                **_privatemode_proof_claims(
                    model_workload_binding_digest=PRIVATEMODE_MODEL_WORKLOAD_BINDING_DIGEST
                ),
                "verification_steps": _privatemode_verification_steps(),
            },
        }

    from routstr.upstream import confidential_verifiers

    original_runner = confidential_verifiers._run_confidential_verifier_command
    original_digest = confidential_verifiers._sha256_file_digest
    confidential_verifiers._run_confidential_verifier_command = fake_verifier_command
    confidential_verifiers._sha256_file_digest = lambda path: VALID_VERIFIER_DIGEST
    try:
        provider = PrivatemodeUpstreamProvider(
            base_url="http://127.0.0.1:8080/v1",
            api_key="test",
        )
        provider.configure_confidentiality_from_settings(
            {
                "confidentiality": {
                    "enabled": True,
                    "mode": "privatemode",
                    "model_ids": ["privatemode/kimi-k2-6"],
                    "policy": {
                        "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                        "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                        "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                        "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                        "dump_requests": False,
                        "shared_prompt_cache": False,
                        "nvidia_ocsp_allow_unknown": False,
                        "nvidia_ocsp_revoked_grace_period_hours": 0,
                        "verifier_command": [
                            "/usr/local/bin/routstr-privatemode-verifier"
                        ],
                        "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    },
                }
            }
        )

        status = await provider.refresh_confidentiality_status()
    finally:
        confidential_verifiers._run_confidential_verifier_command = original_runner
        confidential_verifiers._sha256_file_digest = original_digest

    assert status.verified is False
    assert "key_release_binding claim does not match component binding" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_evidence_digest_tracks_verifier_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {
            "verified": True,
            "verifier": "unit-test-privatemode-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                "proxy_base_url": payload["proxy_base_url"],
                "transport": "privatemode-proxy",
                "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                "coordinator_measurement": _test_digest(f"coordinator-{call_count}"),
                "secret_service_measurement": PRIVATEMODE_SECRET_SERVICE_MEASUREMENT,
                "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                "gpu_attestation_policy": "nvidia-ocsp-good-only",
                "key_release_binding": PRIVATEMODE_MODEL_WORKLOAD_KEY_RELEASE_BINDING,
                **_privatemode_proof_claims(
                    model_workload_binding_digest=PRIVATEMODE_MODEL_WORKLOAD_BINDING_DIGEST
                ),
                "verification_steps": _privatemode_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": ["privatemode/kimi-k2-6"],
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    first_status = await provider.refresh_confidentiality_status()
    second_status = await provider.refresh_confidentiality_status()

    assert first_status.verified is True
    assert second_status.verified is True
    assert first_status.policy_digest == second_status.policy_digest
    assert first_status.evidence_digest != second_status.evidence_digest
    assert (
        first_status.evidence_digest
        == first_status.verified_claims["runtime_evidence_digest"]
    )
    assert (
        second_status.evidence_digest
        == second_status.verified_claims["runtime_evidence_digest"]
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_verifier_without_full_step_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-privatemode-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
                "secret_service_measurement": PRIVATEMODE_SECRET_SERVICE_MEASUREMENT,
                "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                "gpu_attestation_policy": "nvidia-ocsp-good-only",
                "key_release_binding": PRIVATEMODE_MODEL_WORKLOAD_KEY_RELEASE_BINDING,
                **_privatemode_proof_claims(
                    model_workload_binding_digest=(
                        PRIVATEMODE_MODEL_WORKLOAD_BINDING_DIGEST
                    )
                ),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "verification_steps must be a JSON object" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_verifier_without_proof_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        proof_claims = _privatemode_proof_claims(
            model_workload_binding_digest=PRIVATEMODE_MODEL_WORKLOAD_BINDING_DIGEST
        )
        proof_claims.pop("coordinator_attestation_doc_digest")
        return {
            "verified": True,
            "verifier": "unit-test-privatemode-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
                "secret_service_measurement": PRIVATEMODE_SECRET_SERVICE_MEASUREMENT,
                "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                "gpu_attestation_policy": "nvidia-ocsp-good-only",
                "key_release_binding": PRIVATEMODE_MODEL_WORKLOAD_KEY_RELEASE_BINDING,
                **proof_claims,
                "verification_steps": _privatemode_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "coordinator_attestation_doc_digest claim is required" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_placeholder_proof_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        proof_claims = _privatemode_proof_claims(
            model_workload_binding_digest=PRIVATEMODE_MODEL_WORKLOAD_BINDING_DIGEST
        )
        proof_claims["coordinator_attestation_doc_digest"] = "sha256:coordinator-doc"
        return {
            "verified": True,
            "verifier": "unit-test-privatemode-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
                "secret_service_measurement": PRIVATEMODE_SECRET_SERVICE_MEASUREMENT,
                "ai_worker_measurement": proof_claims[
                    "attested_workload_policy_digest"
                ],
                "gpu_attestation_policy": "nvidia-ocsp-good-only",
                "key_release_binding": PRIVATEMODE_MODEL_WORKLOAD_KEY_RELEASE_BINDING,
                **proof_claims,
                "verification_steps": _privatemode_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert "coordinator_attestation_doc_digest claim must be a sha256 digest" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_requires_manifest_log_inclusion_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-privatemode-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                "manifest_log_digest": PRIVATEMODE_MANIFEST_LOG_DIGEST,
                "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
                "secret_service_measurement": PRIVATEMODE_SECRET_SERVICE_MEASUREMENT,
                "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                "gpu_attestation_policy": "nvidia-ocsp-good-only",
                "key_release_binding": PRIVATEMODE_MODEL_WORKLOAD_KEY_RELEASE_BINDING,
                **_privatemode_proof_claims(
                    model_workload_binding_digest=(
                        PRIVATEMODE_MODEL_WORKLOAD_BINDING_DIGEST
                    )
                ),
                "verification_steps": _privatemode_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "manifest_log_dir": "/var/lib/privatemode/manifests",
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "manifest_log_manifest_digest claim is required" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_verifier_manifest_mismatch() -> None:
    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-privatemode-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "manifest_digest": PRIVATEMODE_OTHER_MANIFEST_DIGEST,
                "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
                "secret_service_measurement": PRIVATEMODE_SECRET_SERVICE_MEASUREMENT,
                "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                "gpu_attestation_policy": "nvidia-ocsp-good-only",
                "key_release_binding": PRIVATEMODE_MODEL_WORKLOAD_KEY_RELEASE_BINDING,
                **_privatemode_proof_claims(
                    model_workload_binding_digest=(
                        PRIVATEMODE_MODEL_WORKLOAD_BINDING_DIGEST
                    )
                ),
                "verification_steps": _privatemode_verification_steps(),
            },
        }

    from routstr.upstream import confidential_verifiers

    original_runner = confidential_verifiers._run_confidential_verifier_command
    original_digest = confidential_verifiers._sha256_file_digest
    confidential_verifiers._run_confidential_verifier_command = fake_verifier_command
    confidential_verifiers._sha256_file_digest = lambda path: VALID_VERIFIER_DIGEST
    try:
        provider = PrivatemodeUpstreamProvider(
            base_url="http://127.0.0.1:8080/v1",
            api_key="test",
        )
        provider.configure_confidentiality_from_settings(
            {
                "confidentiality": {
                    "enabled": True,
                    "mode": "privatemode",
                    "model_ids": PRIVATEMODE_MODEL_IDS,
                    "policy": {
                        "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                        "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                        "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                        "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                        "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                        "dump_requests": False,
                        "shared_prompt_cache": False,
                        "nvidia_ocsp_allow_unknown": False,
                        "nvidia_ocsp_revoked_grace_period_hours": 0,
                        "verifier_command": [
                            "/usr/local/bin/routstr-privatemode-verifier"
                        ],
                        "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    },
                }
            }
        )

        status = await provider.refresh_confidentiality_status()
    finally:
        confidential_verifiers._run_confidential_verifier_command = original_runner
        confidential_verifiers._sha256_file_digest = original_digest

    assert status.verified is False
    assert status.verified_claims == {}
    assert "manifest_digest claim does not match policy" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_malformed_component_policy_before_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        return {
            "verified": True,
            "verifier": "unit-test-privatemode-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
                "secret_service_measurement": PRIVATEMODE_SECRET_SERVICE_MEASUREMENT,
                "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                "gpu_attestation_policy": "nvidia-ocsp-good-only",
                "key_release_binding": PRIVATEMODE_MODEL_WORKLOAD_KEY_RELEASE_BINDING,
                **_privatemode_proof_claims(
                    model_workload_binding_digest=(
                        PRIVATEMODE_MODEL_WORKLOAD_BINDING_DIGEST
                    )
                ),
                "verification_steps": _privatemode_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "allowed_secret_service_measurements": [
                        PRIVATEMODE_SECRET_SERVICE_MEASUREMENT,
                        {"unexpected": "object"},
                    ],
                    "expected_gpu_attestation_policy": {"unexpected": "object"},
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert verifier_payloads == []
    assert "secret_service_measurement policy values must be sha256 digests" in (
        status.failure_reason or ""
    )
    assert (
        "gpu_attestation_policy policy values must contain only non-empty strings"
        in (status.failure_reason or "")
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_conflicting_component_policy_before_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        return {}

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "expected_secret_service_measurement": (
                        PRIVATEMODE_SECRET_SERVICE_MEASUREMENT
                    ),
                    "allowed_secret_service_measurements": [
                        PRIVATEMODE_OTHER_SECRET_SERVICE_MEASUREMENT,
                    ],
                    "expected_gpu_attestation_policy": "nvidia-ocsp-good-only",
                    "allowed_gpu_attestation_policies": ["nvidia-ocsp-relaxed"],
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert verifier_payloads == []
    assert "secret_service_measurement expected value must be allowed" in (
        status.failure_reason or ""
    )
    assert "gpu_attestation_policy expected value must be allowed" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_privatemode_refresh_rejects_component_claim_outside_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-privatemode-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                "proxy_base_url": payload["proxy_base_url"],
                "transport": "privatemode-proxy",
                "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
                "secret_service_measurement": (
                    PRIVATEMODE_OTHER_SECRET_SERVICE_MEASUREMENT
                ),
                "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
                "gpu_attestation_policy": "nvidia-ocsp-good-only",
                "key_release_binding": PRIVATEMODE_MODEL_WORKLOAD_KEY_RELEASE_BINDING,
                **_privatemode_proof_claims(
                    model_workload_binding_digest=(
                        PRIVATEMODE_MODEL_WORKLOAD_BINDING_DIGEST
                    )
                ),
                "verification_steps": _privatemode_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
    )

    provider = PrivatemodeUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "privatemode",
                "model_ids": PRIVATEMODE_MODEL_IDS,
                "policy": {
                    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
                    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
                    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
                    "proxy_binary_path": PRIVATEMODE_PROXY_BINARY_PATH,
                    "expected_workload_sans": PRIVATEMODE_EXPECTED_WORKLOAD_SANS,
                    "expected_trust_tier": PRIVATEMODE_TRUST_TIER,
                    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
                    "model_workload_bindings": PRIVATEMODE_MODEL_WORKLOAD_BINDINGS,
                    "expected_coordinator_measurement": (
                        PRIVATEMODE_COORDINATOR_MEASUREMENT
                    ),
                    "allowed_secret_service_measurements": [
                        PRIVATEMODE_SECRET_SERVICE_MEASUREMENT,
                    ],
                    "expected_ai_worker_measurement": (
                        PRIVATEMODE_AI_WORKER_MEASUREMENT
                    ),
                    "expected_gpu_attestation_policy": "nvidia-ocsp-good-only",
                    "expected_key_release_binding": (
                        PRIVATEMODE_MODEL_WORKLOAD_KEY_RELEASE_BINDING
                    ),
                    "dump_requests": False,
                    "shared_prompt_cache": False,
                    "nvidia_ocsp_allow_unknown": False,
                    "nvidia_ocsp_revoked_grace_period_hours": 0,
                    "verifier_command": ["/usr/local/bin/routstr-privatemode-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "secret_service_measurement claim does not match policy" in (
        status.failure_reason or ""
    )
