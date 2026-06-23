import base64
import hashlib
import json
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException, Response
from httpx import AsyncClient
from httpx import Response as HttpxResponse
from sqlmodel.ext.asyncio.session import AsyncSession

from routstr.core.admin import admin_sessions
from routstr.core.db import ApiKey, ModelRow, UpstreamProviderRow
from routstr.core.exceptions import UpstreamError
from routstr.core.settings import settings
from routstr.payment.models import Architecture, Model, Pricing
from routstr.upstream.base import BaseUpstreamProvider, ConfidentialityStatus
from routstr.upstream.ppqai import PPQPrivateUpstreamProvider
from routstr.upstream.tinfoil import TinfoilUpstreamProvider

VERIFIER_STUB_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "confidential_verifier_stub.py"
)
ATTESTATION_STUB_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "routstr_tee_attestation_stub.py"
)


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


def _hex(label: str, *, length: int = 64) -> str:
    payload = b""
    counter = 0
    while len(payload.hex()) < length:
        payload += hashlib.sha256(f"{label}:{counter}".encode("utf-8")).digest()
        counter += 1
    return payload.hex()[:length]


VALID_POLICY_DIGEST = _digest("valid-provider-policy")
VALID_EVIDENCE_DIGEST = _digest("valid-provider-evidence")
VALID_CLAIMS_DIGEST = _digest("valid-provider-claims")
VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST = _digest("valid-local-runtime-evidence")
VALID_ROUTSTR_CODE_MEASUREMENT = _digest("valid-routstr-code-measurement")
VALID_TEE_REPORT_DIGEST = _digest("valid-tee-report")
VALID_TEE_CERTIFICATE_CHAIN_DIGEST = _digest("valid-tee-certificate-chain")
VALID_HPKE_KEY_CONFIG_DIGEST = _digest("valid-hpke-key-config")
VALID_HPKE_PUBLIC_KEY_DIGEST = _digest("valid-hpke-public-key")
VALID_PUBLIC_KEY_DIGEST = _digest("valid-public-key")
VALID_TEE_REPORT_DATA_DIGEST = _digest("valid-tee-report-data")
VALID_TEE_REPORT_DATA_HEX = _hex("valid-tee-report-data", length=128)
VALID_TEE_REPORT_NONCE = "unit-test-tee-report-nonce"
VALID_TEE_REPORT_NONCE_DIGEST = (
    "sha256:" + hashlib.sha256(VALID_TEE_REPORT_NONCE.encode("utf-8")).hexdigest()
)
FIXTURE_ROUTSTR_CODE_MEASUREMENT = (
    "sha256:" + hashlib.sha256(b"fixture-routstr-code-measurement").hexdigest()
)
VALID_PROVIDER_ATTESTATION_REPORT_DIGEST = _digest("valid-provider-attestation-report")
VALID_PROVIDER_RELEASE_DIGEST = _digest("valid-provider-release")
VALID_PROVIDER_TLS_DIGEST = _digest("valid-provider-tls-key")


def _ehbp_verification_steps() -> dict[str, bool]:
    return {
        "hardware_attestation_report": True,
        "hardware_certificate_chain": True,
        "code_transparency": True,
        "measurement_match": True,
        "attested_transport_key_binding": True,
        "freshness": True,
    }


def _tinfoil_model_attestations_for_model_ids(
    model_ids: list[str],
) -> dict[str, dict[str, object]]:
    return {
        model_id: {
            "repo": f"tinfoilsh/confidential-{model_id.rsplit('/', 1)[-1]}",
            "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
            "attestation_report_digest": _digest(f"{model_id}:attestation-report"),
            "attested_hpke_public_key_hex": _hex(f"{model_id}:hpke-public-key"),
            "enclave_measurement_fingerprint": _digest(
                f"{model_id}:enclave-measurement"
            ),
            "code_measurement_fingerprint": _digest(f"{model_id}:code-measurement"),
            "release_digest": _digest(f"{model_id}:release"),
            "tls_public_key_fingerprint_sha256": _digest(f"{model_id}:tls-key"),
            "verification_steps": _ehbp_verification_steps(),
        }
        for model_id in model_ids
    }


def _tinfoil_policy_for_model_ids(model_ids: list[str]) -> dict[str, object]:
    return {
        "repo": "tinfoilsh/confidential-model-router",
        "expected_release_digest": VALID_PROVIDER_RELEASE_DIGEST,
        "require_model_attestations": True,
        "model_attestation_targets": {
            model_id: {
                "host": f"{model_id.rsplit('/', 1)[-1]}.tinfoil.example",
                "repo": f"tinfoilsh/confidential-{model_id.rsplit('/', 1)[-1]}",
                "expected_release_digest": _digest(f"{model_id}:release"),
            }
            for model_id in model_ids
        },
    }


def _public_tinfoil_proof_claims(model_ids: list[str]) -> dict[str, object]:
    return {
        "transport": "ehbp",
        "repo": "tinfoilsh/confidential-model-router",
        "attested_hpke_public_key_hex": _hex("public-tinfoil-router-hpke"),
        "enclave_measurement_fingerprint": _digest(
            "public-tinfoil-router-enclave-measurement"
        ),
        "code_measurement_fingerprint": _digest(
            "public-tinfoil-router-code-measurement"
        ),
        "payload_policy_digest": VALID_POLICY_DIGEST,
        "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
        "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
        "attestation_report_digest": _digest(
            "public-tinfoil-router-attestation-report"
        ),
        "release_digest": _digest("public-tinfoil-router-release"),
        "tls_public_key_fingerprint_sha256": _digest("public-tinfoil-router-tls-key"),
        "verification_steps": _ehbp_verification_steps(),
        "model_attestations": _tinfoil_model_attestations_for_model_ids(model_ids),
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


def _sha256_json_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _privatemode_key_release_binding_digest(claims: dict[str, object]) -> str:
    return _sha256_json_digest(
        {
            "attested_workload_policy_digest": claims.get(
                "attested_workload_policy_digest"
            ),
            "expected_workload_identity_digest": claims.get(
                "expected_workload_identity_digest"
            ),
            "model_workload_binding_digest": claims.get(
                "model_workload_binding_digest"
            ),
            "manifest_digest": claims.get("manifest_digest"),
            "mesh_ca_digest": claims.get("mesh_ca_digest"),
            "secret_service_certificate_digest": claims.get(
                "secret_service_certificate_digest"
            ),
            "inference_secret_id_digest": claims.get("inference_secret_id_digest"),
            "nvidia_ocsp_policy_mac_digest": claims.get(
                "nvidia_ocsp_policy_mac_digest"
            ),
        }
    )


def _configure_routstr_tee_attestation_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        settings,
        "routstr_tee_attestation_command",
        json.dumps([sys.executable, str(ATTESTATION_STUB_PATH)]),
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_attestation_artifact_path",
        str(ATTESTATION_STUB_PATH),
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_attestation_command_digest",
        _sha256_file_digest(ATTESTATION_STUB_PATH),
    )


def _provider_verified_claims_for_mode(mode: str) -> dict[str, object]:
    if mode == "privatemode":
        claims: dict[str, object] = {
            "transport": "privatemode-proxy",
            "trust_tier": "app-e2ee",
            "payload_policy_digest": VALID_POLICY_DIGEST,
            "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
            "manifest_digest": _digest("privatemode-manifest"),
            "proxy_image_digest": _digest("privatemode-proxy-image"),
            "proxy_binary_digest": _digest("privatemode-proxy-binary"),
            "coordinator_measurement": _digest("privatemode-coordinator"),
            "secret_service_measurement": _digest("privatemode-secret-service"),
            "ai_worker_measurement": _digest("privatemode-ai-worker"),
            "attested_workload_identity_digest": _digest(
                "privatemode-workload-identity"
            ),
            "attested_workload_policy_digest": _digest("privatemode-ai-worker"),
            "expected_workload_identity_digest": _digest(
                "privatemode-expected-workload"
            ),
            "model_workload_binding_digest": _digest(
                "privatemode-model-workload-binding"
            ),
            "gpu_attestation_policy": "nvidia-ocsp-good-only",
            "coordinator_attestation_doc_digest": _digest(
                "privatemode-coordinator-attestation-doc"
            ),
            "mesh_ca_digest": _digest("privatemode-mesh-ca"),
            "secret_service_certificate_digest": _digest(
                "privatemode-secret-service-certificate"
            ),
            "ai_worker_manifest_digest": _digest("privatemode-ai-worker-manifest"),
            "nvidia_ocsp_policy_header_digest": _digest(
                "privatemode-nvidia-ocsp-policy-header"
            ),
            "nvidia_ocsp_policy_mac_digest": _digest(
                "privatemode-nvidia-ocsp-policy-mac"
            ),
            "prompt_encryption_ciphertext_digest": _digest(
                "privatemode-prompt-encryption-ciphertext"
            ),
            "inference_secret_id_digest": _digest("privatemode-inference-secret"),
            "verification_steps": _privatemode_verification_steps(),
        }
        claims["key_release_binding"] = _privatemode_key_release_binding_digest(claims)
        return claims

    claims: dict[str, object] = {
        "transport": "ehbp",
        "repo": "tinfoilsh/confidential-model-router",
        "payload_policy_digest": VALID_POLICY_DIGEST,
        "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
        "attested_hpke_public_key_hex": _hex("provider-hpke-public-key"),
        "enclave_measurement_fingerprint": _digest("provider-enclave-measurement"),
        "code_measurement_fingerprint": _digest("provider-code-measurement"),
        "verification_steps": _ehbp_verification_steps(),
    }
    if mode == "tinfoil":
        claims.update(
            {
                "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
                "attestation_report_digest": VALID_PROVIDER_ATTESTATION_REPORT_DIGEST,
                "release_digest": VALID_PROVIDER_RELEASE_DIGEST,
                "tls_public_key_fingerprint_sha256": VALID_PROVIDER_TLS_DIGEST,
            }
        )
    return claims


def _key(balance: int = 100_000) -> ApiKey:
    return ApiKey(
        hashed_key=f"test_{uuid.uuid4().hex}",
        balance=balance,
        reserved_balance=0,
        total_spent=0,
    )


def _ehbp_key_config(public_key: bytes | None = None) -> bytes:
    public_key = public_key or (b"\x55" * 32)
    return b"\x00" + b"\x00\x20" + public_key + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"


def _sha256_file_digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _bind_local_confidentiality_policy(
    provider: BaseUpstreamProvider,
    policy_digest: str = VALID_POLICY_DIGEST,
    policy: dict[str, object] | None = None,
    mode: str | None = None,
    model_ids: list[str] | None = None,
    model_id_prefixes: list[str] | None = None,
) -> None:
    def policy_getter() -> SimpleNamespace:
        local_policy = policy
        status = provider.confidentiality_status()
        selected_model_ids = (
            list(model_ids)
            if model_ids is not None
            else list(getattr(status, "model_ids", []))
        )
        if (
            local_policy is None
            and getattr(provider, "provider_type", None) == "tinfoil"
        ):
            if isinstance(selected_model_ids, list) and all(
                isinstance(model_id, str) and model_id.strip()
                for model_id in selected_model_ids
            ):
                local_policy = _tinfoil_policy_for_model_ids(selected_model_ids)
        return SimpleNamespace(
            digest=policy_digest,
            mode=mode or getattr(status, "mode", "none"),
            model_ids=selected_model_ids,
            model_id_prefixes=(
                list(model_id_prefixes)
                if model_id_prefixes is not None
                else list(getattr(status, "model_id_prefixes", []))
            ),
            policy=local_policy or {},
        )

    setattr(
        provider,
        "confidentiality_policy",
        policy_getter,
    )


def _privatemode_policy_for_model_ids(model_ids: list[str]) -> dict[str, object]:
    workload_san = "secure-model.default.svc.cluster.local"
    return {
        "manifest_digest": _digest("privatemode-manifest"),
        "proxy_binary_digest": _digest("privatemode-proxy-binary"),
        "expected_trust_tier": "app-e2ee",
        "expected_gpu_attestation_policy": "nvidia-ocsp-good-only",
        "expected_workload_sans": [workload_san],
        "model_workload_bindings": {
            model_id: {"workload_sans": [workload_san]} for model_id in model_ids
        },
    }


def _privatemode_claims_for_model_ids(model_ids: list[str]) -> dict[str, object]:
    claims = _provider_verified_claims_for_mode("privatemode")
    workload_san = "secure-model.default.svc.cluster.local"
    claims["expected_workload_identity_digest"] = _sha256_json_digest(
        {"ids": [], "sans": [workload_san]}
    )
    claims["model_workload_binding_digest"] = _sha256_json_digest(
        {
            model_id: {
                "workload_ids": [],
                "workload_sans": [workload_san],
            }
            for model_id in model_ids
        }
    )
    claims["selected_model_ids"] = list(model_ids)
    claims["key_release_binding"] = _privatemode_key_release_binding_digest(claims)
    return claims


def _mark_provider_verified(
    provider: BaseUpstreamProvider,
    *,
    mode: str = "tinfoil",
    model_ids: list[str] | None = None,
    model_id_prefixes: list[str] | None = None,
) -> None:
    selected_model_ids = model_ids or ["gpt-secure"]
    policy: dict[str, object] | None = None
    verified_claims = _provider_verified_claims_for_mode(mode)
    if mode == "privatemode":
        verified_claims = _privatemode_claims_for_model_ids(selected_model_ids)
        verified_claims["proxy_base_url"] = provider.base_url
        policy = _privatemode_policy_for_model_ids(selected_model_ids)
    elif mode == "tinfoil":
        verified_claims["model_attestations"] = (
            _tinfoil_model_attestations_for_model_ids(selected_model_ids)
        )
        policy = _tinfoil_policy_for_model_ids(selected_model_ids)
    selected_prefixes = model_id_prefixes or []
    _bind_local_confidentiality_policy(
        provider,
        policy=policy,
        mode=mode,
        model_ids=selected_model_ids,
        model_id_prefixes=selected_prefixes,
    )
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode=mode,
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            verifier="unit-test-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_claims=verified_claims,
            model_ids=selected_model_ids,
            model_id_prefixes=selected_prefixes,
        )
    )


def _require_missing_local_tee(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    monkeypatch.setattr(
        settings,
        "routstr_tee_client_confidentiality_boundary",
        "attested-tls-termination",
    )
    monkeypatch.setattr(settings, "routstr_attestation_document_path", "")
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_b64", "")
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_path", "")


def test_routstr_tee_unavailable_message_requires_strict_ready_boolean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    monkeypatch.setattr(
        "routstr.core.attestation.get_routstr_tee_readiness",
        lambda: {
            "ready": "false",
            "failure_reason": "malformed readiness",
        },
    )
    from routstr.proxy import get_routstr_tee_unavailable_message

    message = get_routstr_tee_unavailable_message()

    assert message is not None
    assert "malformed readiness" in message


def _local_tee_verified_claims_with_secrets(
    *,
    hpke_key_config_digest: str = VALID_HPKE_KEY_CONFIG_DIGEST,
    hpke_public_key_digest: str = VALID_HPKE_PUBLIC_KEY_DIGEST,
    public_key_digest: str = VALID_PUBLIC_KEY_DIGEST,
) -> dict[str, object]:
    return {
        "attestation_document_format": "tdx_quote",
        "hpke_key_config_digest": hpke_key_config_digest,
        "hpke_public_key_digest": hpke_public_key_digest,
        "public_key_digest": public_key_digest,
        "routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT,
        "routstr_config_measurement": VALID_POLICY_DIGEST,
        "tee_attestation_report_digest": VALID_TEE_REPORT_DIGEST,
        "tee_certificate_chain_digest": VALID_TEE_CERTIFICATE_CHAIN_DIGEST,
        "tee_report_data_digest": VALID_TEE_REPORT_DATA_DIGEST,
        "tee_report_data_hex": VALID_TEE_REPORT_DATA_HEX,
        "tee_report_nonce": VALID_TEE_REPORT_NONCE,
        "tee_report_nonce_digest": VALID_TEE_REPORT_NONCE_DIGEST,
        "verification_steps": {
            "freshness": True,
            "hpke_key_binding": True,
            "measurement_match": True,
            "public_key_binding": True,
            "runtime_policy_binding": True,
            "tee_attestation_report": True,
            "tee_certificate_chain": True,
        },
        "debug_secret": "SECRET_STEP",
        "api_key": "SECRET_LOCAL_VERIFIER_CLAIM",
        "raw_prompt": "SECRET_PROMPT",
    }


def _expected_local_tee_public_proof_claims(
    *,
    hpke_key_config_digest: str = VALID_HPKE_KEY_CONFIG_DIGEST,
    hpke_public_key_digest: str = VALID_HPKE_PUBLIC_KEY_DIGEST,
    public_key_digest: str = VALID_PUBLIC_KEY_DIGEST,
) -> dict[str, object]:
    return {
        "attestation_document_format": "tdx_quote",
        "hpke_key_config_digest": hpke_key_config_digest,
        "hpke_public_key_digest": hpke_public_key_digest,
        "public_key_digest": public_key_digest,
        "routstr_code_measurement": VALID_ROUTSTR_CODE_MEASUREMENT,
        "routstr_config_measurement": VALID_POLICY_DIGEST,
        "tee_attestation_report_digest": VALID_TEE_REPORT_DIGEST,
        "tee_certificate_chain_digest": VALID_TEE_CERTIFICATE_CHAIN_DIGEST,
        "tee_report_data_digest": VALID_TEE_REPORT_DATA_DIGEST,
        "tee_report_data_hex": VALID_TEE_REPORT_DATA_HEX,
        "tee_report_nonce": VALID_TEE_REPORT_NONCE,
        "tee_report_nonce_digest": VALID_TEE_REPORT_NONCE_DIGEST,
        "verification_steps": {
            "freshness": True,
            "hpke_key_binding": True,
            "measurement_match": True,
            "public_key_binding": True,
            "runtime_policy_binding": True,
            "tee_attestation_report": True,
            "tee_certificate_chain": True,
        },
    }


def _public_routstr_tee_status(*, ready: bool = True) -> dict[str, object]:
    return {
        "required": True,
        "ready": ready,
        "client_confidentiality": {
            "mode": "attested-tls-termination",
            "tls_terminates_in_attested_tee": True,
            "inbound_ehbp_ohttp_request_decryption": False,
            "attested_tls_public_key_digest": VALID_PUBLIC_KEY_DIGEST,
        },
        "attestation_evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
        "local_verification": {
            "verified": ready,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "proof_claims": {
                "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
                "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
                "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            },
        },
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_required_rejects_unavailable_route_before_billing(
    integration_client: AsyncClient,
    integration_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Required confidential routing must fail closed before any reservation."""
    key = _key()
    integration_session.add(key)
    await integration_session.commit()

    monkeypatch.setattr(
        settings, "confidential_routing_mode", "required", raising=False
    )

    model = Model(
        id="gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/chat/completions"],
    )
    mock_pay_for_request = AsyncMock()
    mock_get_max_cost = AsyncMock(return_value=1_000)

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=None),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
    ):
        response = await integration_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer sk-{key.hashed_key}"},
            json={
                "model": "gpt-secure",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "confidential_route_unavailable"
    assert "gpt-secure" in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()

    await integration_session.refresh(key)
    assert key.balance == 100_000
    assert key.reserved_balance == 0
    assert key.total_spent == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_required_rechecks_cached_provider_attestation_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider-map entry must fail closed if attestation expires after publish."""
    monkeypatch.setattr(
        settings, "confidential_routing_mode", "required", raising=False
    )
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)
    monkeypatch.setattr("routstr.algorithm.time.time", lambda: 4_102_444_801)

    model = Model(
        id="gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/chat/completions"],
    )
    provider = BaseUpstreamProvider(base_url="http://127.0.0.1:8080/v1", api_key="test")
    provider.provider_type = "tinfoil"
    _bind_local_confidentiality_policy(
        provider,
        mode="tinfoil",
        model_ids=["gpt-secure"],
    )
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="tinfoil",
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            verifier="unit-test-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_claims={
                **_provider_verified_claims_for_mode("tinfoil"),
                "model_attestations": _tinfoil_model_attestations_for_model_ids(
                    ["gpt-secure"]
                ),
            },
            model_ids=["gpt-secure"],
        )
    )

    mock_get_max_cost = AsyncMock(return_value=1_000)
    mock_get_bearer_key = AsyncMock(
        side_effect=HTTPException(status_code=418, detail="billing reached")
    )

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
        patch("routstr.proxy.get_bearer_token_key", mock_get_bearer_key),
    ):
        response = await integration_client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "gpt-secure",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "confidential_route_unavailable"
    assert "No verified confidential provider found" in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_get_bearer_key.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_required_rejects_provider_without_public_exact_map_metadata_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request-time routing must enforce the same proof shape as the exact map."""
    monkeypatch.setattr(
        settings, "confidential_routing_mode", "required", raising=False
    )
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)

    model = Model(
        id="gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/chat/completions"],
    )
    provider = BaseUpstreamProvider(base_url="http://127.0.0.1:8080/v1", api_key="test")
    provider.provider_type = "tinfoil"
    _mark_provider_verified(provider, mode="tinfoil", model_ids=["gpt-secure"])

    mock_get_max_cost = AsyncMock(return_value=1_000)
    mock_get_bearer_key = AsyncMock(
        side_effect=HTTPException(status_code=418, detail="billing reached")
    )

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.is_confidential_provider_for_model", return_value=True),
        patch("routstr.proxy.public_confidentiality_metadata", return_value=None),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
        patch("routstr.proxy.get_bearer_token_key", mock_get_bearer_key),
    ):
        response = await integration_client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "gpt-secure",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "confidential_route_unavailable"
    assert "No verified confidential provider found" in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_get_bearer_key.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_responses_rejects_conflicting_model_selectors_before_lookup(
    integration_client: AsyncClient,
) -> None:
    with patch(
        "routstr.proxy.get_model_instance",
        side_effect=AssertionError("model lookup must not run"),
    ):
        response = await integration_client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "tinfoil/secure-model",
                "input": {
                    "model": "private/gpt-oss-120b",
                    "content": "SECRET_PROMPT",
                },
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["detail"]["error"] == {
        "type": "invalid_request_error",
        "code": "invalid_model",
    }
    assert "SECRET_PROMPT" not in response.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ppq_private_rejects_responses_api_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PPQ documents private/* models as chat-only; reject Responses before cost."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)

    model = Model(
        id="private/gpt-oss-120b",
        name="Private GPT OSS",
        forwarded_model_id="private/gpt-oss-120b",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/chat/completions"],
    )
    provider = PPQPrivateUpstreamProvider(api_key="test")
    mock_pay_for_request = AsyncMock()
    mock_get_max_cost = AsyncMock(return_value=1_000)

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
    ):
        response = await integration_client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "private/gpt-oss-120b",
                "input": "SECRET_PROMPT",
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "unsupported_endpoint"
    assert "Responses API" in body["error"]["message"]
    assert "private/gpt-oss-120b" in body["error"]["message"]
    assert "SECRET_PROMPT" not in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ppq_private_x_cashu_responses_api_rejects_before_token_redemption(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported PPQ private Responses requests must not redeem cashu tokens."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)

    model = Model(
        id="private/gpt-oss-120b",
        name="Private GPT OSS",
        forwarded_model_id="private/gpt-oss-120b",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
    )
    provider = PPQPrivateUpstreamProvider(api_key="test")
    mock_get_max_cost = AsyncMock(return_value=1_000)
    mock_receive_token = AsyncMock(return_value=(10_000, "msat", None))

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
        patch("routstr.upstream.base.recieve_token", mock_receive_token),
    ):
        response = await integration_client.post(
            "/v1/responses",
            headers={"X-Cashu": "cashuAtest"},
            json={
                "model": "private/gpt-oss-120b",
                "input": "SECRET_PROMPT",
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "unsupported_endpoint"
    assert "Responses API" in body["error"]["message"]
    assert "SECRET_PROMPT" not in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_receive_token.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ppq_private_rejects_embeddings_api_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PPQ private transport only documents chat completions; reject embeddings early."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)

    model = Model(
        id="private/gpt-oss-120b",
        name="Private GPT OSS",
        forwarded_model_id="private/gpt-oss-120b",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
    )
    provider = PPQPrivateUpstreamProvider(api_key="test")
    mock_pay_for_request = AsyncMock()
    mock_get_max_cost = AsyncMock(return_value=1_000)

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
    ):
        response = await integration_client.post(
            "/v1/embeddings",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "private/gpt-oss-120b",
                "input": "SECRET_EMBEDDING_INPUT",
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "unsupported_endpoint"
    assert "Embeddings API" in body["error"]["message"]
    assert "private/gpt-oss-120b" in body["error"]["message"]
    assert "SECRET_EMBEDDING_INPUT" not in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_text_model_rejects_embeddings_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confidential endpoint routing must also match the selected model shape."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)

    model = Model(
        id="tinfoil/gpt-secure",
        name="Tinfoil GPT Secure",
        forwarded_model_id="tinfoil/gpt-secure",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
    )
    provider = TinfoilUpstreamProvider(api_key="test")
    _mark_provider_verified(provider, model_ids=["tinfoil/gpt-secure"])
    key = _key()
    mock_forward = AsyncMock(return_value=Response('{"data":[]}', status_code=200))
    mock_get_key = AsyncMock(return_value=key)
    mock_get_max_cost = AsyncMock(return_value=1_000)
    mock_pay_for_request = AsyncMock()

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.get_bearer_token_key", mock_get_key),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch.object(provider, "forward_request", mock_forward),
    ):
        response = await integration_client.post(
            "/v1/embeddings",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "tinfoil/gpt-secure",
                "input": "SECRET_EMBEDDING_INPUT",
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "unsupported_endpoint"
    assert "Embeddings API" in body["error"]["message"]
    assert "tinfoil/gpt-secure" in body["error"]["message"]
    assert "SECRET_EMBEDDING_INPUT" not in body["error"]["message"]
    mock_get_key.assert_not_awaited()
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()
    mock_forward.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_required_rejects_unrecognized_model_path_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Required confidential mode must not forward model prompts to unknown paths."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)

    model = Model(
        id="tinfoil/gpt-secure",
        name="Tinfoil GPT Secure",
        forwarded_model_id="tinfoil/gpt-secure",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
    )
    provider = TinfoilUpstreamProvider(api_key="test")
    _mark_provider_verified(provider, model_ids=["tinfoil/gpt-secure"])
    mock_forward = AsyncMock(return_value=Response('{"ok":true}', status_code=200))
    mock_get_key = AsyncMock(return_value=_key())
    mock_get_max_cost = AsyncMock(return_value=1_000)
    mock_pay_for_request = AsyncMock()

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.get_bearer_token_key", mock_get_key),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch.object(provider, "forward_request", mock_forward),
    ):
        response = await integration_client.post(
            "/v1/chat/completions/side-channel",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "tinfoil/gpt-secure",
                "messages": [{"role": "user", "content": "SECRET_PROMPT"}],
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "unsupported_endpoint"
    assert "unrecognized API path" in body["error"]["message"]
    assert "tinfoil/gpt-secure" in body["error"]["message"]
    assert "SECRET_PROMPT" not in body["error"]["message"]
    mock_get_key.assert_not_awaited()
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()
    mock_forward.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_required_rejects_unrecognized_model_path_before_cashu_redemption(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown confidential model paths must not redeem Cashu tokens."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)

    model = Model(
        id="tinfoil/gpt-secure",
        name="Tinfoil GPT Secure",
        forwarded_model_id="tinfoil/gpt-secure",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
    )
    provider = TinfoilUpstreamProvider(api_key="test")
    _mark_provider_verified(provider, model_ids=["tinfoil/gpt-secure"])
    mock_forward = AsyncMock(return_value=Response('{"ok":true}', status_code=200))
    mock_get_max_cost = AsyncMock(return_value=1_000)
    mock_receive_token = AsyncMock(return_value=(10_000, "msat", None))

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
        patch("routstr.upstream.base.recieve_token", mock_receive_token),
        patch.object(provider, "forward_request", mock_forward),
    ):
        response = await integration_client.post(
            "/v1/chat/completions/side-channel",
            headers={"X-Cashu": "cashuAtest"},
            json={
                "model": "tinfoil/gpt-secure",
                "messages": [{"role": "user", "content": "SECRET_PROMPT"}],
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "unsupported_endpoint"
    assert "unrecognized API path" in body["error"]["message"]
    assert "SECRET_PROMPT" not in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_receive_token.assert_not_awaited()
    mock_forward.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ppq_private_rejects_legacy_completions_api_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PPQ private transport only documents chat completions; reject legacy completions."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)

    model = Model(
        id="private/gpt-oss-120b",
        name="Private GPT OSS",
        forwarded_model_id="private/gpt-oss-120b",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
    )
    provider = PPQPrivateUpstreamProvider(api_key="test")
    mock_pay_for_request = AsyncMock()
    mock_get_max_cost = AsyncMock(return_value=1_000)

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
    ):
        response = await integration_client.post(
            "/v1/completions",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "private/gpt-oss-120b",
                "prompt": "SECRET_COMPLETION_PROMPT",
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "unsupported_endpoint"
    assert "Completions API" in body["error"]["message"]
    assert "private/gpt-oss-120b" in body["error"]["message"]
    assert "SECRET_COMPLETION_PROMPT" not in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ppq_private_rejects_images_api_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PPQ private transport only documents chat completions; reject image endpoints."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)

    model = Model(
        id="private/gpt-oss-120b",
        name="Private GPT OSS",
        forwarded_model_id="private/gpt-oss-120b",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
    )
    provider = PPQPrivateUpstreamProvider(api_key="test")
    mock_pay_for_request = AsyncMock()
    mock_get_max_cost = AsyncMock(return_value=1_000)

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
    ):
        response = await integration_client.post(
            "/v1/images/generations",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "private/gpt-oss-120b",
                "prompt": "SECRET_IMAGE_PROMPT",
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "unsupported_endpoint"
    assert "Images API" in body["error"]["message"]
    assert "private/gpt-oss-120b" in body["error"]["message"]
    assert "SECRET_IMAGE_PROMPT" not in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ppq_private_rejects_moderations_api_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PPQ private transport only documents chat completions; reject moderation endpoints."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)

    model = Model(
        id="private/gpt-oss-120b",
        name="Private GPT OSS",
        forwarded_model_id="private/gpt-oss-120b",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
    )
    provider = PPQPrivateUpstreamProvider(api_key="test")
    mock_pay_for_request = AsyncMock()
    mock_get_max_cost = AsyncMock(return_value=1_000)

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
    ):
        response = await integration_client.post(
            "/v1/moderations",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "private/gpt-oss-120b",
                "input": "SECRET_MODERATION_INPUT",
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "unsupported_endpoint"
    assert "Moderations API" in body["error"]["message"]
    assert "private/gpt-oss-120b" in body["error"]["message"]
    assert "SECRET_MODERATION_INPUT" not in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ppq_private_rejects_audio_api_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PPQ private transport only documents chat completions; reject audio endpoints."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)

    model = Model(
        id="private/gpt-oss-120b",
        name="Private GPT OSS",
        forwarded_model_id="private/gpt-oss-120b",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
    )
    provider = PPQPrivateUpstreamProvider(api_key="test")
    mock_pay_for_request = AsyncMock()
    mock_get_max_cost = AsyncMock(return_value=1_000)

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
    ):
        response = await integration_client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "private/gpt-oss-120b",
                "prompt": "SECRET_AUDIO_CONTEXT",
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "unsupported_endpoint"
    assert "Audio Transcriptions API" in body["error"]["message"]
    assert "private/gpt-oss-120b" in body["error"]["message"]
    assert "SECRET_AUDIO_CONTEXT" not in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tinfoil_audio_multipart_uses_form_model_for_routing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audio transcription requests are multipart, so routing cannot require JSON."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)

    model = Model(
        id="tinfoil/whisper-large-v3-turbo",
        name="Tinfoil Whisper",
        forwarded_model_id="tinfoil/whisper-large-v3-turbo",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="audio->text",
            input_modalities=["audio"],
            output_modalities=["text"],
            tokenizer="unknown",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.0),
        supported_endpoints=["/v1/audio/transcriptions"],
    )
    provider = TinfoilUpstreamProvider(api_key="test")
    _mark_provider_verified(provider, model_ids=["tinfoil/whisper-large-v3-turbo"])
    key = _key()
    mock_forward = AsyncMock(return_value=Response('{"text":"ok"}', status_code=200))
    mock_get_key = AsyncMock(return_value=key)
    mock_get_max_cost = AsyncMock(return_value=1_000)
    mock_pay_for_request = AsyncMock()

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.get_bearer_token_key", mock_get_key),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch.object(provider, "forward_request", mock_forward),
    ):
        response = await integration_client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer sk-test"},
            data={
                "model": "tinfoil/whisper-large-v3-turbo",
                "prompt": "SECRET_STYLE_HINT",
            },
            files={"file": ("sample.wav", b"SECRET_AUDIO_BYTES", "audio/wav")},
        )

    assert response.status_code == 200
    mock_get_max_cost.assert_awaited_once()
    mock_get_key.assert_awaited_once()
    mock_pay_for_request.assert_awaited_once()
    mock_forward.assert_awaited_once()
    forwarded_body = mock_forward.await_args.args[3]
    assert b"SECRET_AUDIO_BYTES" in forwarded_body
    assert b"SECRET_STYLE_HINT" in forwarded_body


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tinfoil_rejects_audio_family_path_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confidential audio routes must use exact supported audio sub-endpoints."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)

    model = Model(
        id="tinfoil/whisper-large-v3-turbo",
        name="Tinfoil Whisper",
        forwarded_model_id="tinfoil/whisper-large-v3-turbo",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="audio->text",
            input_modalities=["audio"],
            output_modalities=["text"],
            tokenizer="unknown",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.0),
        supported_endpoints=["/v1/audio/transcriptions"],
    )
    provider = TinfoilUpstreamProvider(api_key="test")
    _mark_provider_verified(provider, model_ids=["tinfoil/whisper-large-v3-turbo"])
    mock_forward = AsyncMock(return_value=Response('{"text":"ok"}', status_code=200))
    mock_get_key = AsyncMock(return_value=_key())
    mock_get_max_cost = AsyncMock(return_value=1_000)
    mock_pay_for_request = AsyncMock()

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.get_bearer_token_key", mock_get_key),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch.object(provider, "forward_request", mock_forward),
    ):
        response = await integration_client.post(
            "/v1/audio",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "tinfoil/whisper-large-v3-turbo",
                "input": "SECRET_AUDIO_REQUEST",
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "unsupported_endpoint"
    assert "unrecognized API path" in body["error"]["message"]
    assert "tinfoil/whisper-large-v3-turbo" in body["error"]["message"]
    assert "SECRET_AUDIO_REQUEST" not in body["error"]["message"]
    mock_get_key.assert_not_awaited()
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()
    mock_forward.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_privatemode_audio_speech_rejects_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Privatemode docs expose speech-to-text, not confidential text-to-speech."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)

    model = Model(
        id="privatemode/audio-model",
        name="Privatemode Audio",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text->audio",
            input_modalities=["text"],
            output_modalities=["audio"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/chat/completions"],
    )
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "privatemode"
    provider.supports_audio_api = True
    provider.supports_audio_transcriptions_api = True
    provider.supports_audio_speech_api = False
    _mark_provider_verified(
        provider,
        mode="privatemode",
        model_ids=["privatemode/audio-model"],
    )
    mock_pay_for_request = AsyncMock()
    mock_get_max_cost = AsyncMock(return_value=1_000)

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
    ):
        response = await integration_client.post(
            "/v1/audio/speech",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "privatemode/audio-model",
                "input": "SECRET_SPEECH_INPUT",
                "voice": "default",
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "unsupported_endpoint"
    assert "Audio Speech API" in body["error"]["message"]
    assert "privatemode/audio-model" in body["error"]["message"]
    assert "SECRET_SPEECH_INPUT" not in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_upstream_error_response_redacts_sensitive_details(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider/verifier diagnostics must not leak through caller errors."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)

    model = Model(
        id="gpt-secure",
        name="Secure Model",
        forwarded_model_id="gpt-secure",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/chat/completions"],
    )
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    _mark_provider_verified(provider)
    key = _key()
    sensitive_error = (
        "verifier failed api_key=SECRET_PROVIDER_KEY "
        '"raw_prompt":"SECRET_PROMPT" '
        "Bearer sk-live-secretvalue "
        "https://user:pass@verified.example/v1"
    )
    mock_forward = AsyncMock(
        side_effect=UpstreamError(sensitive_error, status_code=502)
    )
    mock_get_key = AsyncMock(return_value=key)
    mock_get_max_cost = AsyncMock(return_value=1_000)
    mock_pay_for_request = AsyncMock()
    mock_revert_pay_for_request = AsyncMock()

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.get_bearer_token_key", mock_get_key),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.revert_pay_for_request", mock_revert_pay_for_request),
        patch.object(provider, "forward_request", mock_forward),
    ):
        response = await integration_client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "gpt-secure",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 502
    response_text = response.text
    assert "SECRET_PROVIDER_KEY" not in response_text
    assert "SECRET_PROMPT" not in response_text
    assert "sk-live-secretvalue" not in response_text
    assert "user:pass" not in response_text
    assert "api_key: [REDACTED]" in response_text
    assert "raw_prompt: [REDACTED]" in response_text
    assert "https://verified.example/v1" in response_text
    mock_forward.assert_awaited_once()
    mock_revert_pay_for_request.assert_awaited_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_retry_hook_receives_redacted_upstream_error_message(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failover hooks must not receive raw confidential upstream diagnostics."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)

    model = Model(
        id="gpt-secure",
        name="Secure Model",
        forwarded_model_id="gpt-secure",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/chat/completions"],
    )

    class RecordingProvider(BaseUpstreamProvider):
        def __init__(self, provider_type: str) -> None:
            super().__init__(
                base_url="https://inference.tinfoil.sh/v1",
                api_key="test",
            )
            self.provider_type = provider_type
            self.redirect_messages: list[str] = []
            _mark_provider_verified(self)

        async def on_upstream_error_redirect(
            self, status_code: int, error_message: str
        ) -> None:
            self.redirect_messages.append(error_message)

    first_provider = RecordingProvider("tinfoil")
    second_provider = RecordingProvider("tinfoil")
    key = _key()
    upstream_error = json.dumps(
        {
            "error": {
                "message": (
                    "retry failed api_key=SECRET_PROVIDER_KEY "
                    '"raw_prompt":"SECRET_PROMPT" '
                    "https://user:pass@verified.example/v1"
                )
            }
        }
    )
    first_forward = AsyncMock(
        return_value=Response(
            upstream_error,
            status_code=502,
            media_type="application/json",
        )
    )
    second_forward = AsyncMock(
        return_value=Response(
            '{"id":"ok"}', status_code=200, media_type="application/json"
        )
    )

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch(
            "routstr.proxy.get_provider_for_model",
            return_value=[first_provider, second_provider],
        ),
        patch("routstr.proxy.get_bearer_token_key", AsyncMock(return_value=key)),
        patch("routstr.proxy.get_max_cost_for_model", AsyncMock(return_value=1_000)),
        patch("routstr.proxy.pay_for_request", AsyncMock()),
        patch.object(first_provider, "forward_request", first_forward),
        patch.object(second_provider, "forward_request", second_forward),
    ):
        response = await integration_client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "gpt-secure",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert len(first_provider.redirect_messages) == 1
    redirect_message = first_provider.redirect_messages[0]
    assert "api_key: [REDACTED]" in redirect_message
    assert "raw_prompt: [REDACTED]" in redirect_message
    assert "https://verified.example/v1" in redirect_message
    assert "SECRET_PROVIDER_KEY" not in redirect_message
    assert "SECRET_PROMPT" not in redirect_message
    assert "user:pass" not in redirect_message
    first_forward.assert_awaited_once()
    second_forward.assert_awaited_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ppq_private_rejects_messages_api_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PPQ private EHBP must not fall into Anthropic Messages translation after cost."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)

    model = Model(
        id="private/gpt-oss-120b",
        name="Private GPT OSS",
        forwarded_model_id="private/gpt-oss-120b",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
    )
    provider = PPQPrivateUpstreamProvider(api_key="test")
    mock_pay_for_request = AsyncMock()
    mock_get_max_cost = AsyncMock(return_value=1_000)

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
    ):
        response = await integration_client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "private/gpt-oss-120b",
                "messages": [{"role": "user", "content": "SECRET_PROMPT"}],
                "max_tokens": 64,
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "unsupported_endpoint"
    assert "Messages API" in body["error"]["message"]
    assert "private/gpt-oss-120b" in body["error"]["message"]
    assert "SECRET_PROMPT" not in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ppq_private_rejects_messages_count_tokens_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Messages count_tokens is still a Messages API route for PPQ private."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)

    model = Model(
        id="private/gpt-oss-120b",
        name="Private GPT OSS",
        forwarded_model_id="private/gpt-oss-120b",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
    )
    provider = PPQPrivateUpstreamProvider(api_key="test")
    mock_pay_for_request = AsyncMock()
    mock_get_max_cost = AsyncMock(return_value=1_000)

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
    ):
        response = await integration_client.post(
            "/v1/messages/count_tokens",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "private/gpt-oss-120b",
                "messages": [{"role": "user", "content": "SECRET_PROMPT"}],
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "unsupported_endpoint"
    assert "Messages Count Tokens API" in body["error"]["message"]
    assert "private/gpt-oss-120b" in body["error"]["message"]
    assert "SECRET_PROMPT" not in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ppq_private_x_cashu_messages_api_rejects_before_token_redemption(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported PPQ private Messages requests must not redeem cashu tokens."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)

    model = Model(
        id="private/gpt-oss-120b",
        name="Private GPT OSS",
        forwarded_model_id="private/gpt-oss-120b",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
    )
    provider = PPQPrivateUpstreamProvider(api_key="test")
    mock_get_max_cost = AsyncMock(return_value=1_000)
    mock_receive_token = AsyncMock(return_value=(10_000, "msat", None))

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
        patch("routstr.upstream.base.recieve_token", mock_receive_token),
    ):
        response = await integration_client.post(
            "/v1/messages",
            headers={"X-Cashu": "cashuAtest"},
            json={
                "model": "private/gpt-oss-120b",
                "messages": [{"role": "user", "content": "SECRET_PROMPT"}],
                "max_tokens": 64,
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "unsupported_endpoint"
    assert "Messages API" in body["error"]["message"]
    assert "SECRET_PROMPT" not in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_receive_token.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_required_rejects_missing_routstr_tee_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verified provider is not enough if Routstr itself lacks TEE evidence."""
    _require_missing_local_tee(monkeypatch)

    model = Model(
        id="gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/chat/completions"],
    )
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    _mark_provider_verified(provider)
    mock_pay_for_request = AsyncMock()
    mock_get_max_cost = AsyncMock(return_value=1_000)

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
    ):
        response = await integration_client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "gpt-secure",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "routstr_tee_attestation_unavailable"
    assert "Routstr TEE attestation is required" in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_required_rejects_embeddings_without_routstr_tee_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embeddings inputs are sensitive model traffic and must hit the same TEE gate."""
    _require_missing_local_tee(monkeypatch)

    model = Model(
        id="embed-secure",
        name="Secure Embeddings",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["embedding"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.0),
        supported_endpoints=["/v1/embeddings"],
    )
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    _mark_provider_verified(provider, model_ids=["embed-secure"])
    mock_pay_for_request = AsyncMock()
    mock_get_max_cost = AsyncMock(return_value=1_000)

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
    ):
        response = await integration_client.post(
            "/v1/embeddings",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "embed-secure",
                "input": "SECRET_EMBEDDING_INPUT",
            },
        )

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "routstr_tee_attestation_unavailable"
    assert "Routstr TEE attestation is required" in body["error"]["message"]
    assert "SECRET_EMBEDDING_INPUT" not in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_required_rejects_responses_without_routstr_tee_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Responses API prompts must not reach billing unless local TEE evidence is ready."""
    _require_missing_local_tee(monkeypatch)

    model = Model(
        id="response-secure",
        name="Secure Responses",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/responses"],
    )
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    _mark_provider_verified(provider, model_ids=["response-secure"])
    mock_pay_for_request = AsyncMock()
    mock_get_max_cost = AsyncMock(return_value=1_000)

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
    ):
        response = await integration_client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "response-secure",
                "input": "SECRET_RESPONSE_INPUT",
            },
        )

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "routstr_tee_attestation_unavailable"
    assert "Routstr TEE attestation is required" in body["error"]["message"]
    assert "SECRET_RESPONSE_INPUT" not in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_required_rejects_messages_without_routstr_tee_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native Messages traffic must hit the local TEE gate before billing."""
    _require_missing_local_tee(monkeypatch)

    model = Model(
        id="privatemode/message-secure",
        name="Secure Messages",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/messages"],
    )
    provider = BaseUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.provider_type = "privatemode"
    provider.supports_anthropic_messages = True
    _mark_provider_verified(
        provider,
        mode="privatemode",
        model_ids=["privatemode/message-secure"],
    )
    mock_pay_for_request = AsyncMock()
    mock_get_max_cost = AsyncMock(return_value=1_000)

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
    ):
        response = await integration_client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "privatemode/message-secure",
                "messages": [{"role": "user", "content": "SECRET_MESSAGE_INPUT"}],
                "max_tokens": 32,
            },
        )

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "routstr_tee_attestation_unavailable"
    assert "Routstr TEE attestation is required" in body["error"]["message"]
    assert "SECRET_MESSAGE_INPUT" not in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_privatemode_rejects_messages_count_tokens_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Privatemode docs support /v1/messages, not count_tokens as a route."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", False)

    model = Model(
        id="privatemode/message-secure",
        name="Secure Messages",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/messages"],
    )
    provider = BaseUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.provider_type = "privatemode"
    provider.supports_anthropic_messages = True
    _mark_provider_verified(
        provider,
        mode="privatemode",
        model_ids=["privatemode/message-secure"],
    )
    mock_pay_for_request = AsyncMock()
    mock_get_max_cost = AsyncMock(return_value=1_000)

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
    ):
        response = await integration_client.post(
            "/v1/messages/count_tokens",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "privatemode/message-secure",
                "messages": [{"role": "user", "content": "SECRET_MESSAGE_INPUT"}],
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "unsupported_endpoint"
    assert "Messages Count Tokens API" in body["error"]["message"]
    assert "privatemode/message-secure" in body["error"]["message"]
    assert "SECRET_MESSAGE_INPUT" not in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_required_rejects_legacy_completions_without_routstr_tee_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy completions prompts must hit the local TEE gate before billing."""
    _require_missing_local_tee(monkeypatch)

    model = Model(
        id="privatemode/completion-secure",
        name="Secure Completion",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/completions"],
    )
    provider = BaseUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.provider_type = "privatemode"
    provider.supports_completions_api = True
    _mark_provider_verified(
        provider,
        mode="privatemode",
        model_ids=["privatemode/completion-secure"],
    )
    mock_pay_for_request = AsyncMock()
    mock_get_max_cost = AsyncMock(return_value=1_000)

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
    ):
        response = await integration_client.post(
            "/v1/completions",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "privatemode/completion-secure",
                "prompt": "SECRET_COMPLETION_INPUT",
                "max_tokens": 32,
            },
        )

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "routstr_tee_attestation_unavailable"
    assert "Routstr TEE attestation is required" in body["error"]["message"]
    assert "SECRET_COMPLETION_INPUT" not in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_required_rejects_audio_without_routstr_tee_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multipart audio payloads must hit the local TEE gate before billing."""
    _require_missing_local_tee(monkeypatch)

    model = Model(
        id="audio-secure",
        name="Secure Audio",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="audio->text",
            input_modalities=["audio"],
            output_modalities=["text"],
            tokenizer="unknown",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.0),
        supported_endpoints=["/v1/audio/transcriptions"],
    )
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    _mark_provider_verified(provider, model_ids=["audio-secure"])
    mock_pay_for_request = AsyncMock()
    mock_get_max_cost = AsyncMock(return_value=1_000)

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
    ):
        response = await integration_client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer sk-test"},
            data={
                "model": "audio-secure",
                "prompt": "SECRET_AUDIO_PROMPT",
            },
            files={"file": ("sample.wav", b"SECRET_AUDIO_BYTES", "audio/wav")},
        )

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "routstr_tee_attestation_unavailable"
    assert "Routstr TEE attestation is required" in body["error"]["message"]
    assert "SECRET_AUDIO_PROMPT" not in body["error"]["message"]
    assert "SECRET_AUDIO_BYTES" not in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_required_rejects_x_cashu_without_routstr_tee_before_token_redemption(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """X-Cashu confidential traffic must not redeem tokens before local TEE proof."""
    _require_missing_local_tee(monkeypatch)

    model = Model(
        id="gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/chat/completions"],
    )
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    _mark_provider_verified(provider)
    mock_get_max_cost = AsyncMock(return_value=1_000)
    mock_receive_token = AsyncMock(return_value=(10_000, "msat", None))

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
        patch("routstr.upstream.base.recieve_token", mock_receive_token),
    ):
        response = await integration_client.post(
            "/v1/chat/completions",
            headers={"X-Cashu": "cashuAtest"},
            json={
                "model": "gpt-secure",
                "messages": [{"role": "user", "content": "SECRET_CASHU_INPUT"}],
            },
        )

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "routstr_tee_attestation_unavailable"
    assert "Routstr TEE attestation is required" in body["error"]["message"]
    assert "SECRET_CASHU_INPUT" not in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_receive_token.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_required_rejects_unverified_routstr_tee_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)

    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    monkeypatch.setattr(
        settings,
        "routstr_tee_client_confidentiality_boundary",
        "attested-tls-termination",
    )
    monkeypatch.setattr(settings, "routstr_attestation_document_path", "")
    monkeypatch.setattr(settings, "routstr_attestation_document_format", "tdx_quote")
    _configure_routstr_tee_attestation_stub(monkeypatch)
    monkeypatch.setattr(
        settings,
        "routstr_attestation_public_key",
        "-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----",
    )
    monkeypatch.setattr(settings, "routstr_attestation_public_key_path", "")
    monkeypatch.setattr(
        settings,
        "routstr_attestation_hpke_key_config_b64",
        base64.b64encode(hpke_key_config).decode("ascii"),
    )
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_path", "")
    monkeypatch.setattr(settings, "routstr_tee_verifier_command", "")

    model = Model(
        id="gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/chat/completions"],
    )
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    _mark_provider_verified(provider)
    mock_pay_for_request = AsyncMock()
    mock_get_max_cost = AsyncMock(return_value=1_000)
    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
    ):
        response = await integration_client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "gpt-secure",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "routstr_tee_attestation_unavailable"
    assert "Routstr TEE verifier command is not configured" in body["error"]["message"]
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_required_redacts_routstr_tee_failure_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)

    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    monkeypatch.setattr(
        settings,
        "routstr_tee_client_confidentiality_boundary",
        "attested-tls-termination",
    )
    monkeypatch.setattr(settings, "routstr_attestation_document_path", "")
    monkeypatch.setattr(settings, "routstr_attestation_document_format", "tdx_quote")
    _configure_routstr_tee_attestation_stub(monkeypatch)
    monkeypatch.setattr(
        settings,
        "routstr_attestation_public_key",
        "-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----",
    )
    monkeypatch.setattr(settings, "routstr_attestation_public_key_path", "")
    monkeypatch.setattr(
        settings,
        "routstr_attestation_hpke_key_config_b64",
        base64.b64encode(hpke_key_config).decode("ascii"),
    )
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_path", "")

    def fake_verify_routstr_tee_evidence(**kwargs: object) -> dict[str, object]:
        return {
            "verified": False,
            "verifier": None,
            "verified_at": None,
            "expires_at": None,
            "failure_reason": (
                "local verifier failed api_key=SECRET_LOCAL_KEY "
                "raw_prompt=SECRET_PROMPT "
                "url=https://operator:secret-token@local.example/v1"
            ),
            "evidence_digest": None,
            "verified_claims": {},
        }

    monkeypatch.setattr(
        "routstr.core.attestation._verify_routstr_tee_evidence",
        fake_verify_routstr_tee_evidence,
    )

    model = Model(
        id="gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/chat/completions"],
    )
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    _mark_provider_verified(provider)
    mock_pay_for_request = AsyncMock()
    mock_get_max_cost = AsyncMock(return_value=1_000)

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
    ):
        response = await integration_client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "gpt-secure",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 503
    body = response.json()
    message = body["error"]["message"]
    assert body["error"]["type"] == "routstr_tee_attestation_unavailable"
    assert "api_key: [REDACTED]" in message
    assert "raw_prompt: [REDACTED]" in message
    assert "https://local.example/v1" in message
    assert "SECRET_LOCAL_KEY" not in message
    assert "SECRET_PROMPT" not in message
    assert "secret-token" not in message
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_required_allows_cost_check_when_routstr_tee_is_verified(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.core.attestation.time.time",
        lambda: 1_700_000_010,
    )
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)

    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    monkeypatch.setattr(
        settings,
        "routstr_tee_client_confidentiality_boundary",
        "attested-tls-termination",
    )
    monkeypatch.setattr(settings, "routstr_attestation_document_path", "")
    monkeypatch.setattr(settings, "routstr_attestation_document_format", "tdx_quote")
    _configure_routstr_tee_attestation_stub(monkeypatch)
    monkeypatch.setattr(
        settings,
        "routstr_attestation_public_key",
        "-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----",
    )
    monkeypatch.setattr(settings, "routstr_attestation_public_key_path", "")
    monkeypatch.setattr(
        settings,
        "routstr_attestation_hpke_key_config_b64",
        base64.b64encode(hpke_key_config).decode("ascii"),
    )
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_path", "")
    monkeypatch.setattr(
        settings,
        "routstr_tee_verifier_command",
        json.dumps([sys.executable, str(VERIFIER_STUB_PATH)]),
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_verifier_artifact_path",
        str(VERIFIER_STUB_PATH),
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_verifier_command_digest",
        _sha256_file_digest(VERIFIER_STUB_PATH),
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_verifier_policy_json",
        json.dumps(
            {"expected_routstr_code_measurement": (FIXTURE_ROUTSTR_CODE_MEASUREMENT)}
        ),
    )
    monkeypatch.setattr(settings, "routstr_tee_verifier_timeout_seconds", 2.0)

    model = Model(
        id="gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/chat/completions"],
    )
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    _bind_local_confidentiality_policy(
        provider,
        mode="tinfoil",
        model_ids=["gpt-secure"],
    )
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="tinfoil",
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            verifier="unit-test-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_claims={
                **_provider_verified_claims_for_mode("tinfoil"),
                "model_attestations": _tinfoil_model_attestations_for_model_ids(
                    ["gpt-secure"]
                ),
            },
            model_ids=["gpt-secure"],
        )
    )
    mock_get_max_cost = AsyncMock(return_value=1_000)
    mock_get_bearer_key = AsyncMock(
        side_effect=HTTPException(status_code=418, detail="stopped after tee check")
    )

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
        patch("routstr.proxy.get_bearer_token_key", mock_get_bearer_key),
    ):
        response = await integration_client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "gpt-secure",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 418
    mock_get_max_cost.assert_awaited_once()
    mock_get_bearer_key.assert_awaited_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_required_rejects_routstr_tee_public_key_not_bound_before_billing(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)

    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    monkeypatch.setattr(
        settings,
        "routstr_tee_client_confidentiality_boundary",
        "attested-tls-termination",
    )
    monkeypatch.setattr(settings, "routstr_attestation_document_path", "")
    monkeypatch.setattr(settings, "routstr_attestation_document_format", "tdx_quote")
    _configure_routstr_tee_attestation_stub(monkeypatch)
    monkeypatch.setattr(
        settings,
        "routstr_attestation_public_key",
        "-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----",
    )
    monkeypatch.setattr(settings, "routstr_attestation_public_key_path", "")
    monkeypatch.setattr(
        settings,
        "routstr_attestation_hpke_key_config_b64",
        base64.b64encode(hpke_key_config).decode("ascii"),
    )
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_path", "")
    monkeypatch.setattr(
        settings,
        "routstr_tee_verifier_command",
        json.dumps([sys.executable, str(VERIFIER_STUB_PATH)]),
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_verifier_artifact_path",
        str(VERIFIER_STUB_PATH),
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_verifier_command_digest",
        _sha256_file_digest(VERIFIER_STUB_PATH),
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_verifier_policy_json",
        json.dumps(
            {"expected_routstr_code_measurement": (FIXTURE_ROUTSTR_CODE_MEASUREMENT)}
        ),
    )
    monkeypatch.setattr(settings, "routstr_tee_verifier_timeout_seconds", 2.0)
    monkeypatch.setenv("ROUTSTR_VERIFIER_STUB_BEHAVIOR", "omit_public_key_digest")

    model = Model(
        id="gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/chat/completions"],
    )
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    _mark_provider_verified(provider)
    mock_pay_for_request = AsyncMock()
    mock_get_max_cost = AsyncMock(return_value=1_000)

    with (
        patch("routstr.proxy.get_model_instance", return_value=model),
        patch("routstr.proxy.get_provider_for_model", return_value=[provider]),
        patch("routstr.proxy.pay_for_request", mock_pay_for_request),
        patch("routstr.proxy.get_max_cost_for_model", mock_get_max_cost),
    ):
        response = await integration_client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-test"},
            json={
                "model": "gpt-secure",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "routstr_tee_attestation_unavailable"
    assert (
        "public_key_digest does not match Routstr TEE verifier payload"
        in (body["error"]["message"])
    )
    mock_get_max_cost.assert_not_awaited()
    mock_pay_for_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_required_rejects_tee_get_to_unverified_upstream(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-model TEE proxy paths must not bypass verified-provider filtering."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")

    provider = BaseUpstreamProvider(
        base_url="https://plain.example/v1",
        api_key="SECRET_PROVIDER_KEY",
    )
    provider.provider_type = "custom"
    provider.forward_get_request = AsyncMock(  # type: ignore[method-assign]
        return_value=Response(content=b"leaked", status_code=200)
    )

    with patch("routstr.proxy._upstreams", [provider]):
        response = await integration_client.get("/tee/attestation")

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "confidential_route_unavailable"
    provider.forward_get_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidential_required_rejects_tee_get_before_local_tee_ready(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-model TEE proxy paths must still require Routstr's own TEE gate."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    monkeypatch.setattr(settings, "routstr_attestation_document_path", "")
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_b64", "")
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_path", "")

    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="SECRET_PROVIDER_KEY",
    )
    provider.provider_type = "tinfoil"
    _mark_provider_verified(provider)
    provider.forward_get_request = AsyncMock(  # type: ignore[method-assign]
        return_value=Response(content=b"leaked", status_code=200)
    )

    with patch("routstr.proxy._upstreams", [provider]):
        response = await integration_client.get("/tee/attestation")

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "routstr_tee_attestation_unavailable"
    provider.forward_get_request.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_model_test_endpoint_requires_admin_auth(
    integration_client: AsyncClient,
) -> None:
    response = await integration_client.post(
        "/api/models/test",
        json={
            "model_id": "gpt-secure",
            "endpoint_type": "chat-completions",
            "request_data": {"messages": [{"role": "user", "content": "hello"}]},
        },
    )

    assert response.status_code == 403


@pytest.mark.integration
@pytest.mark.asyncio
async def test_model_test_endpoint_disabled_when_confidential_routing_required(
    integration_client: AsyncClient,
    integration_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_row = UpstreamProviderRow(
        provider_type="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        api_key="SECRET_PROVIDER_KEY",
        enabled=True,
        provider_fee=1.01,
    )
    integration_session.add(provider_row)
    await integration_session.commit()
    await integration_session.refresh(provider_row)
    assert provider_row.id is not None

    integration_session.add(
        ModelRow(
            id="gpt-secure",
            name="Secure Model",
            created=1,
            description="desc",
            context_length=8192,
            architecture='{"modality":"text","input_modalities":["text"],"output_modalities":["text"],"tokenizer":"gpt","instruct_type":null}',
            pricing='{"prompt":0.001,"completion":0.002}',
            upstream_provider_id=provider_row.id,
            enabled=True,
        )
    )
    await integration_session.commit()

    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    admin_token = "test-admin-model-test-confidential-routing"
    admin_sessions[admin_token] = 4_102_444_800
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        with patch(
            "httpx.AsyncClient.post",
            new=AsyncMock(return_value=HttpxResponse(200, json={"ok": True})),
        ) as upstream_post:
            response = await integration_client.request(
                "POST",
                "/api/models/test",
                json={
                    "model_id": "gpt-secure",
                    "endpoint_type": "chat-completions",
                    "request_data": {
                        "messages": [{"role": "user", "content": "SECRET_PROMPT"}]
                    },
                },
            )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "confidential_route_required"
        upstream_post.assert_not_awaited()
    finally:
        admin_sessions.pop(admin_token, None)
        integration_client.headers.pop("Authorization", None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_refresh_model_maps_refreshes_confidentiality_before_publish(
    integration_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime attestation refresh must happen before required-mode maps publish."""
    provider_row = UpstreamProviderRow(
        provider_type="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
        enabled=True,
        provider_fee=1.01,
    )
    integration_session.add(provider_row)
    await integration_session.commit()
    await integration_session.refresh(provider_row)
    assert provider_row.id is not None

    integration_session.add(
        ModelRow(
            id="gpt-secure",
            name="Secure Model",
            created=1,
            description="desc",
            context_length=8192,
            architecture='{"modality":"text","input_modalities":["text"],"output_modalities":["text"],"tokenizer":"gpt","instruct_type":null}',
            pricing='{"prompt":0.001,"completion":0.002}',
            upstream_provider_id=provider_row.id,
            enabled=True,
        )
    )
    await integration_session.commit()

    class RefreshingProvider(BaseUpstreamProvider):
        provider_type = "tinfoil"

        def __init__(self) -> None:
            super().__init__(
                base_url="https://inference.tinfoil.sh/v1",
                api_key="test",
                provider_fee=1.01,
            )
            self.db_id = provider_row.id
            self.refresh_confidentiality_calls = 0

        def get_cached_models(self) -> list[Model]:
            return [
                Model(
                    id="gpt-secure",
                    name="Secure Model",
                    created=1,
                    description="desc",
                    context_length=8192,
                    architecture=Architecture(
                        modality="text",
                        input_modalities=["text"],
                        output_modalities=["text"],
                        tokenizer="gpt",
                        instruct_type=None,
                    ),
                    pricing=Pricing(prompt=0.001, completion=0.002),
                    supported_endpoints=["/v1/chat/completions"],
                )
            ]

        async def refresh_confidentiality_status(self) -> ConfidentialityStatus:
            self.refresh_confidentiality_calls += 1
            status = ConfidentialityStatus(
                enabled=True,
                verified=True,
                mode="tinfoil",
                verified_at=1_700_000_000,
                expires_at=4_102_444_800,
                verifier="unit-test-verifier",
                policy_digest=VALID_POLICY_DIGEST,
                evidence_digest=VALID_EVIDENCE_DIGEST,
                verified_claims={
                    **_provider_verified_claims_for_mode("tinfoil"),
                    "model_attestations": _tinfoil_model_attestations_for_model_ids(
                        ["gpt-secure"]
                    ),
                },
                model_ids=["gpt-secure"],
            )
            self.set_confidentiality_status(status)
            return status

    provider = RefreshingProvider()
    _bind_local_confidentiality_policy(
        provider,
        mode="tinfoil",
        model_ids=["gpt-secure"],
    )

    @asynccontextmanager
    async def mock_create_session() -> AsyncGenerator[AsyncSession, None]:
        yield integration_session

    monkeypatch.setattr(settings, "confidential_routing_mode", "required")

    from routstr import proxy

    with (
        patch("routstr.proxy._upstreams", [provider]),
        patch("routstr.proxy.create_session", return_value=mock_create_session()),
    ):
        await proxy.refresh_model_maps()

    assert provider.refresh_confidentiality_calls == 1
    assert proxy.get_provider_for_model("gpt-secure") == [provider]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidentiality_status_endpoint_reports_runtime_provider_state(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "verified-test"
    provider.upstream_name = "verified-test"
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=False,
            mode="test-verifier",
            failure_reason="attestation expired",
            model_ids=["gpt-secure"],
            policy_digest=VALID_POLICY_DIGEST,
        )
    )

    monkeypatch.setattr(settings, "confidential_routing_mode", "required")

    with patch("routstr.proxy._upstreams", [provider]):
        response = await integration_client.get("/v1/confidentiality/status")

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "required"
    assert data["required"] is True
    assert data["providers"][0]["provider_type"] == "verified-test"
    assert data["providers"][0]["supported_endpoints"] == [
        "/v1/chat/completions",
        "/v1/audio",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/images",
        "/v1/moderations",
        "/v1/responses",
    ]
    assert data["providers"][0]["confidentiality"]["verified"] is False
    assert "failure_reason" not in data["providers"][0]["confidentiality"]
    assert data["providers"][0]["confidentiality"]["failure_reason_digest"].startswith(
        "sha256:"
    )
    assert "policy_digest" not in data["providers"][0]["confidentiality"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidentiality_status_endpoint_marks_end_to_end_unready_without_routstr_tee(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_missing_local_tee(monkeypatch)

    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil-prod"
    _mark_provider_verified(provider, model_ids=["tinfoil/gpt-secure"])

    with patch("routstr.proxy._upstreams", [provider]):
        response = await integration_client.get("/v1/confidentiality/status")

    assert response.status_code == 200
    data = response.json()
    assert data["providers"][0]["confidentiality"]["verified"] is True
    assert data["routstr_tee"]["required"] is True
    assert data["routstr_tee"]["ready"] is False
    assert data["end_to_end_ready"] is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidentiality_status_endpoint_marks_end_to_end_ready_with_routstr_tee(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(
        "routstr.core.attestation.get_public_routstr_tee_status",
        _public_routstr_tee_status,
    )

    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil-prod"
    _mark_provider_verified(provider, model_ids=["tinfoil/gpt-secure"])
    secure_model = Model(
        id="tinfoil/gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/chat/completions"],
    )

    with (
        patch("routstr.proxy._upstreams", [provider]),
        patch("routstr.proxy._model_instances", {"tinfoil/gpt-secure": secure_model}),
        patch("routstr.proxy._provider_map", {"tinfoil/gpt-secure": [provider]}),
    ):
        response = await integration_client.get("/v1/confidentiality/status")

    assert response.status_code == 200
    data = response.json()
    assert data["providers"][0]["confidentiality"]["verified"] is True
    assert data["routstr_tee"]["ready"] is True
    assert data["end_to_end_ready"] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidentiality_status_endpoint_requires_local_tee_proof_for_end_to_end_ready(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(
        "routstr.core.attestation.get_public_routstr_tee_status",
        lambda: {
            "required": True,
            "ready": True,
            "client_confidentiality": {
                "mode": "attested-tls-termination",
                "tls_terminates_in_attested_tee": True,
                "inbound_ehbp_ohttp_request_decryption": False,
                "attested_tls_public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            },
        },
    )

    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil-prod"
    _mark_provider_verified(provider, model_ids=["tinfoil/gpt-secure"])
    secure_model = Model(
        id="tinfoil/gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/chat/completions"],
    )

    with (
        patch("routstr.proxy._upstreams", [provider]),
        patch("routstr.proxy._model_instances", {"tinfoil/gpt-secure": secure_model}),
        patch("routstr.proxy._provider_map", {"tinfoil/gpt-secure": [provider]}),
    ):
        response = await integration_client.get("/v1/confidentiality/status")

    assert response.status_code == 200
    data = response.json()
    assert data["providers"][0]["confidentiality"]["verified"] is True
    assert data["routstr_tee"]["ready"] is False
    assert data["end_to_end_ready"] is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidentiality_status_endpoint_rejects_expired_local_tee_proof_for_end_to_end_ready(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")

    expired_status = _public_routstr_tee_status()
    local_verification = expired_status["local_verification"]
    assert isinstance(local_verification, dict)
    local_verification["verified_at"] = 1
    local_verification["expires_at"] = 2
    monkeypatch.setattr(
        "routstr.core.attestation.get_public_routstr_tee_status",
        lambda: expired_status,
    )

    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil-prod"
    _mark_provider_verified(provider, model_ids=["tinfoil/gpt-secure"])
    secure_model = Model(
        id="tinfoil/gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/chat/completions"],
    )

    with (
        patch("routstr.proxy._upstreams", [provider]),
        patch("routstr.proxy._model_instances", {"tinfoil/gpt-secure": secure_model}),
        patch("routstr.proxy._provider_map", {"tinfoil/gpt-secure": [provider]}),
    ):
        response = await integration_client.get("/v1/confidentiality/status")

    assert response.status_code == 200
    data = response.json()
    assert data["providers"][0]["confidentiality"]["verified"] is True
    assert data["routstr_tee"]["ready"] is False
    assert data["end_to_end_ready"] is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidentiality_status_endpoint_requires_routable_confidential_model_for_end_to_end_ready(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(
        "routstr.core.attestation.get_public_routstr_tee_status",
        _public_routstr_tee_status,
    )

    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil-prod"
    _mark_provider_verified(provider, model_ids=["tinfoil/gpt-secure"])

    with (
        patch("routstr.proxy._upstreams", [provider]),
        patch("routstr.proxy._model_instances", {}),
        patch("routstr.proxy._provider_map", {}),
    ):
        response = await integration_client.get("/v1/confidentiality/status")

    assert response.status_code == 200
    data = response.json()
    assert data["providers"][0]["confidentiality"]["verified"] is True
    assert data["routstr_tee"]["ready"] is True
    assert data["end_to_end_ready"] is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidentiality_status_endpoint_requires_routstr_tee_boundary_for_end_to_end_ready(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(
        "routstr.core.attestation.get_public_routstr_tee_status",
        lambda: {
            "required": True,
            "ready": True,
        },
    )

    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil-prod"
    _mark_provider_verified(provider, model_ids=["tinfoil/gpt-secure"])

    with patch("routstr.proxy._upstreams", [provider]):
        response = await integration_client.get("/v1/confidentiality/status")

    assert response.status_code == 200
    data = response.json()
    assert data["providers"][0]["confidentiality"]["verified"] is True
    assert data["routstr_tee"]["ready"] is False
    assert data["end_to_end_ready"] is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidentiality_status_endpoint_requires_attested_tls_key_digest_for_end_to_end_ready(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(
        "routstr.core.attestation.get_public_routstr_tee_status",
        lambda: {
            "required": True,
            "ready": True,
            "client_confidentiality": {
                "mode": "attested-tls-termination",
                "tls_terminates_in_attested_tee": True,
                "inbound_ehbp_ohttp_request_decryption": False,
            },
        },
    )

    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil-prod"
    _mark_provider_verified(provider, model_ids=["tinfoil/gpt-secure"])

    with patch("routstr.proxy._upstreams", [provider]):
        response = await integration_client.get("/v1/confidentiality/status")

    assert response.status_code == 200
    data = response.json()
    assert data["providers"][0]["confidentiality"]["verified"] is True
    assert data["routstr_tee"]["ready"] is False
    assert data["end_to_end_ready"] is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidentiality_status_endpoint_exposes_sanitized_policy_without_urls(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil"
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": _tinfoil_policy_for_model_ids(["tinfoil/gpt-secure"]),
            }
        }
    )
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")

    with patch("routstr.proxy._upstreams", [provider]):
        response = await integration_client.get("/v1/confidentiality/status")

    assert response.status_code == 200
    provider_status = response.json()["providers"][0]
    assert "base_url" not in provider_status
    assert "db_id" not in provider_status
    assert provider_status["confidentiality_policy"]["repo"] == (
        "tinfoilsh/confidential-model-router"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidentiality_status_endpoint_omits_raw_verifier_claims(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BaseUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="SECRET_PROVIDER_KEY",
    )
    provider.provider_type = "privatemode"
    provider.upstream_name = "privatemode-local"
    model_ids = ["privatemode/model"]
    local_policy = _privatemode_policy_for_model_ids(model_ids)
    verified_claims: dict[str, object] = {
        "transport": "privatemode-proxy",
        "trust_tier": "app-e2ee",
        "manifest_digest": _digest("status-privatemode-manifest"),
        "proxy_image_digest": _digest("status-privatemode-proxy-image"),
        "coordinator_measurement": _digest("status-privatemode-coordinator"),
        "secret_service_measurement": _digest("status-privatemode-secret-service"),
        "ai_worker_measurement": _digest("status-privatemode-ai-worker"),
        "attested_workload_identity_digest": _digest(
            "status-privatemode-workload-identity"
        ),
        "attested_workload_policy_digest": _digest("status-privatemode-ai-worker"),
        "gpu_attestation_policy": "nvidia-ocsp-good-only",
        "coordinator_attestation_doc_digest": _digest(
            "status-privatemode-coordinator-attestation-doc"
        ),
        "mesh_ca_digest": _digest("status-privatemode-mesh-ca"),
        "secret_service_certificate_digest": _digest(
            "status-privatemode-secret-service-certificate"
        ),
        "ai_worker_manifest_digest": _digest("status-privatemode-ai-worker-manifest"),
        "nvidia_ocsp_policy_header_digest": _digest(
            "status-privatemode-nvidia-ocsp-policy-header"
        ),
        "nvidia_ocsp_policy_mac_digest": _digest(
            "status-privatemode-nvidia-ocsp-policy-mac"
        ),
        "prompt_encryption_ciphertext_digest": _digest(
            "status-privatemode-prompt-encryption-ciphertext"
        ),
        "inference_secret_id_digest": _digest("status-privatemode-inference-secret"),
        "verification_steps": {
            "contrast_manifest": True,
            "coordinator_attestation": True,
            "mesh_ca_binding": True,
            "secret_service_tls": True,
            "ai_worker_attestation": True,
            "gpu_attestation": True,
            "key_release_binding": True,
            "prompt_encryption": True,
            "nvidia_ocsp_revocation": True,
        },
        "api_key": "SECRET_CLAIM_VALUE",
        "proxy_base_url": "http://127.0.0.1:8080/v1",
        "manifest_log_entry": "internal-manifest-log-entry",
    }
    verified_claims.update(_privatemode_claims_for_model_ids(model_ids))
    verified_claims["api_key"] = "SECRET_CLAIM_VALUE"
    verified_claims["manifest_log_entry"] = "internal-manifest-log-entry"
    verified_claims["key_release_binding"] = _privatemode_key_release_binding_digest(
        verified_claims
    )
    _bind_local_confidentiality_policy(
        provider,
        policy=local_policy,
        mode="privatemode",
        model_ids=model_ids,
    )
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="privatemode",
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            verifier="unit-test-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            model_ids=model_ids,
            verified_claims=verified_claims,
        )
    )

    monkeypatch.setattr(settings, "confidential_routing_mode", "required")

    with patch("routstr.proxy._upstreams", [provider]):
        response = await integration_client.get("/v1/confidentiality/status")

    assert response.status_code == 200
    data = response.json()
    confidentiality = data["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is True
    assert confidentiality["evidence_digest"] == VALID_EVIDENCE_DIGEST
    assert confidentiality["verified_claims_digest"].startswith("sha256:")
    assert "verified_claims" not in confidentiality

    serialized = json.dumps(data)
    assert "SECRET_PROVIDER_KEY" not in serialized
    assert "SECRET_CLAIM_VALUE" not in serialized
    assert "internal-manifest-log-entry" not in serialized
    assert "127.0.0.1:8080" not in serialized
    assert "base_url" not in serialized
    assert "api_key" not in serialized


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidentiality_status_endpoint_exposes_public_provider_proof_claims(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="SECRET_PROVIDER_KEY",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil-prod"
    _bind_local_confidentiality_policy(
        provider,
        mode="tinfoil",
        model_ids=["tinfoil/gpt-secure"],
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": _digest("status-tinfoil-router-release"),
            "expected_code_measurement_fingerprint": _digest(
                "status-tinfoil-router-code-measurement"
            ),
            "expected_enclave_measurement_fingerprint": _digest(
                "status-tinfoil-router-enclave-measurement"
            ),
            "require_model_attestations": True,
            "model_attestation_targets": {
                "tinfoil/gpt-secure": {
                    "repo": "tinfoilsh/confidential-gpt-secure",
                    "expected_release_digest": _digest("tinfoil/gpt-secure:release"),
                    "expected_code_measurement_fingerprint": _digest(
                        "tinfoil/gpt-secure:code-measurement"
                    ),
                    "expected_enclave_measurement_fingerprint": _digest(
                        "tinfoil/gpt-secure:enclave-measurement"
                    ),
                }
            },
        },
    )
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="tinfoil",
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            verifier="unit-test-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            model_ids=["tinfoil/gpt-secure"],
            verified_claims={
                "transport": "ehbp",
                "repo": "tinfoilsh/confidential-model-router",
                "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
                "attestation_report_digest": _digest(
                    "status-tinfoil-router-attestation-report"
                ),
                "attested_hpke_public_key_hex": _hex("status-tinfoil-router-hpke"),
                "enclave_measurement_fingerprint": _digest(
                    "status-tinfoil-router-enclave-measurement"
                ),
                "code_measurement_fingerprint": _digest(
                    "status-tinfoil-router-code-measurement"
                ),
                "payload_policy_digest": VALID_POLICY_DIGEST,
                "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
                "release_digest": _digest("status-tinfoil-router-release"),
                "tls_public_key_fingerprint_sha256": _digest(
                    "status-tinfoil-router-tls-key"
                ),
                "verification_steps": {
                    "hardware_attestation_report": True,
                    "hardware_certificate_chain": True,
                    "code_transparency": True,
                    "measurement_match": True,
                    "attested_transport_key_binding": True,
                    "freshness": True,
                },
                "debug_secret": "SECRET_STEP",
                "model_attestations": _tinfoil_model_attestations_for_model_ids(
                    ["tinfoil/gpt-secure"]
                ),
                "ehbp_key_config_b64": "SECRET_KEY_CONFIG",
                "api_key": "SECRET_CLAIM_VALUE",
            },
        )
    )

    monkeypatch.setattr(settings, "confidential_routing_mode", "required")

    with patch("routstr.proxy._upstreams", [provider]):
        response = await integration_client.get("/v1/confidentiality/status")

    assert response.status_code == 200
    data = response.json()
    confidentiality = data["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is True
    assert confidentiality["verified_claims_digest"].startswith("sha256:")
    assert "verified_claims" not in confidentiality
    assert confidentiality["proof_claims"] == {
        "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
        "attestation_report_digest": _digest(
            "status-tinfoil-router-attestation-report"
        ),
        "attested_hpke_public_key_hex": _hex("status-tinfoil-router-hpke"),
        "code_measurement_fingerprint": _digest(
            "status-tinfoil-router-code-measurement"
        ),
        "enclave_measurement_fingerprint": _digest(
            "status-tinfoil-router-enclave-measurement"
        ),
        "payload_policy_digest": VALID_POLICY_DIGEST,
        "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
        "release_digest": _digest("status-tinfoil-router-release"),
        "repo": "tinfoilsh/confidential-model-router",
        "tls_public_key_fingerprint_sha256": _digest("status-tinfoil-router-tls-key"),
        "transport": "ehbp",
        "model_attestations": _tinfoil_model_attestations_for_model_ids(
            ["tinfoil/gpt-secure"]
        ),
        "verification_steps": {
            "attested_transport_key_binding": True,
            "code_transparency": True,
            "freshness": True,
            "hardware_attestation_report": True,
            "hardware_certificate_chain": True,
            "measurement_match": True,
        },
    }
    serialized = json.dumps(data)
    assert "SECRET_PROVIDER_KEY" not in serialized
    assert "SECRET_CLAIM_VALUE" not in serialized
    assert "SECRET_KEY_CONFIG" not in serialized
    assert "SECRET_STEP" not in serialized
    assert '"verified_claims":' not in serialized


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidentiality_status_endpoint_includes_redacted_routstr_tee_status(
    integration_client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = b"fake-tdx-quote"
    hpke_key_config = _ehbp_key_config()
    evidence_path = tmp_path / "quote.bin"
    evidence_path.write_bytes(evidence)

    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    monkeypatch.setattr(
        settings,
        "routstr_tee_client_confidentiality_boundary",
        "attested-tls-termination",
    )
    monkeypatch.setattr(
        settings, "routstr_attestation_document_path", str(evidence_path)
    )
    monkeypatch.setattr(settings, "routstr_attestation_document_format", "tdx_quote")
    _configure_routstr_tee_attestation_stub(monkeypatch)
    monkeypatch.setattr(
        settings,
        "routstr_attestation_public_key",
        "-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----",
    )
    monkeypatch.setattr(settings, "routstr_attestation_public_key_path", "")
    monkeypatch.setattr(
        settings,
        "routstr_attestation_hpke_key_config_b64",
        base64.b64encode(hpke_key_config).decode("ascii"),
    )
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_path", "")

    def fake_verify_routstr_tee_evidence(**kwargs: object) -> dict[str, object]:
        tee = kwargs["tee"]
        assert isinstance(tee, dict)
        hpke_key_config = tee["hpke_key_config"]
        assert isinstance(hpke_key_config, dict)
        return {
            "verified": True,
            "verifier": "unit-test-routstr-tee-verifier",
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "failure_reason": None,
            "evidence_digest": str(tee["attestation_evidence_digest"]),
            "verified_claims": _local_tee_verified_claims_with_secrets(
                hpke_key_config_digest=str(hpke_key_config["key_config_digest"]),
                hpke_public_key_digest=str(hpke_key_config["public_key_digest"]),
                public_key_digest=str(tee["public_key_digest"]),
            ),
        }

    monkeypatch.setattr(
        "routstr.core.attestation._verify_routstr_tee_evidence",
        fake_verify_routstr_tee_evidence,
    )

    response = await integration_client.get("/v1/confidentiality/status")

    assert response.status_code == 200
    data = response.json()
    routstr_tee = data["routstr_tee"]
    public_key_digest = (
        "sha256:"
        + hashlib.sha256(
            b"-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----"
        ).hexdigest()
    )
    assert routstr_tee["required"] is True
    assert routstr_tee["ready"] is True
    assert routstr_tee["client_confidentiality"] == {
        "mode": "attested-tls-termination",
        "tls_terminates_in_attested_tee": True,
        "inbound_ehbp_ohttp_request_decryption": False,
        "attested_tls_public_key_digest": public_key_digest,
    }
    assert routstr_tee["attestation_evidence_digest"].startswith("sha256:")
    assert routstr_tee["local_verification"]["verified"] is True
    assert routstr_tee["local_verification"]["verified_claims_digest"].startswith(
        "sha256:"
    )
    assert routstr_tee["local_verification"][
        "proof_claims"
    ] == _expected_local_tee_public_proof_claims(
        hpke_key_config_digest=routstr_tee["hpke_key_config_digest"],
        hpke_public_key_digest=routstr_tee["hpke_public_key_digest"],
        public_key_digest=public_key_digest,
    )
    assert "verified_claims" not in routstr_tee["local_verification"]

    serialized = json.dumps(data)
    assert "SECRET_LOCAL_VERIFIER_CLAIM" not in serialized
    assert "SECRET_PROMPT" not in serialized
    assert "SECRET_STEP" not in serialized
    assert "api_key" not in serialized
    assert "raw_prompt" not in serialized


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidentiality_status_accepts_cap_attested_tls_boundary(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.core.attestation.time.time",
        lambda: 1_700_000_010,
    )
    monkeypatch.setattr("routstr.proxy.time.time", lambda: 1_700_000_010)
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil-prod"
    _mark_provider_verified(provider, model_ids=["tinfoil/gpt-secure"])
    secure_model = Model(
        id="tinfoil/gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/chat/completions"],
    )

    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    monkeypatch.setattr(
        settings,
        "routstr_tee_client_confidentiality_boundary",
        "attested-tls-termination",
    )
    monkeypatch.setattr(settings, "routstr_attestation_document_path", "")
    monkeypatch.setattr(settings, "routstr_attestation_document_format", "")
    monkeypatch.setattr(settings, "routstr_attestation_public_key", "")
    monkeypatch.setattr(settings, "routstr_attestation_public_key_path", "")
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_b64", "")
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_path", "")
    monkeypatch.setattr(
        settings,
        "routstr_tee_cap_attestation_enabled",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_cap_tee_domain",
        "routstr-core.example.tee.enclava.dev",
        raising=False,
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_cap_public_base_url",
        "https://routstr-core.example.tee.enclava.dev/.well-known/confidential",
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.core.attestation._fetch_cap_attestation_status",
        lambda: {
            "claims_verified": True,
            "state": "unlocked",
            "mode": "password",
            "tenant_id": "cap-test-org-routstr-core",
            "claims_instance_id": "routstr-core",
            "instance_id": "cap-test-org-routstr-core-routstr-core",
            "error": None,
            "claims_error": None,
        },
        raising=False,
    )

    with (
        patch("routstr.proxy._upstreams", [provider]),
        patch("routstr.proxy._model_instances", {"tinfoil/gpt-secure": secure_model}),
        patch("routstr.proxy._provider_map", {"tinfoil/gpt-secure": [provider]}),
    ):
        response = await integration_client.get("/v1/confidentiality/status")
        attestation_response = await integration_client.get(
            "/.well-known/routstr-attestation"
        )
        info_response = await integration_client.get("/v1/info")

    assert response.status_code == 200
    assert attestation_response.status_code == 200
    assert info_response.status_code == 200
    data = response.json()
    routstr_tee = data["routstr_tee"]
    assert routstr_tee["required"] is True
    assert routstr_tee["ready"] is True
    assert data["end_to_end_ready"] is True
    assert routstr_tee["hpke_key_config_digest"] is None
    assert routstr_tee["local_verification"]["verified"] is True
    assert routstr_tee["local_verification"]["verifier"] == "cap-attestation-proxy"
    proof_claims = routstr_tee["local_verification"]["proof_claims"]
    assert proof_claims["cap_claims_verified"] is True
    assert proof_claims["cap_state"] == "unlocked"
    assert proof_claims["cap_tee_domain"] == "routstr-core.example.tee.enclava.dev"
    assert proof_claims["client_confidentiality_boundary"] == (
        "attested-tls-termination"
    )
    assert proof_claims["verification_steps"] == {
        "cap_status_verified": True,
        "cap_claims_verified": True,
        "cap_state_unlocked": True,
        "tls_terminates_in_attested_tee": True,
        "freshness": True,
    }
    attestation_statement = attestation_response.json()
    routing_policy = attestation_statement["routing_policy"]
    routing_provider = routing_policy["providers"][0]
    assert routing_policy["routable_with_full_attestation"] == {
        "tinfoil": ["tinfoil/gpt-secure"],
        "ppq-private": [],
        "privatemode": [],
    }
    assert routing_provider["confidentiality"]["verified"] is True
    assert routing_provider["confidentiality"]["proof_claims"]["model_attestations"][
        "tinfoil/gpt-secure"
    ]["release_digest"].startswith("sha256:")
    tee_statement = attestation_statement["tee"]
    assert tee_statement["evidence_format"] == "cap-attestation-proxy-status"
    assert tee_statement["attestation_source"] == "cap-attestation-proxy"
    assert (
        tee_statement["attestation_evidence_digest"]
        == (routstr_tee["attestation_evidence_digest"])
    )
    assert tee_statement["cap_attestation"]["available"] is True
    assert tee_statement["cap_attestation"]["attestation_url"] == (
        "https://routstr-core.example.tee.enclava.dev/.well-known/confidential/"
        "attestation"
    )
    assert (
        tee_statement["local_verification"]["proof_claims"]["cap_attestation_url"]
        == tee_statement["cap_attestation"]["attestation_url"]
    )
    info_confidentiality = info_response.json()["confidentiality"]
    assert info_confidentiality["routstr_tee"]["ready"] is True
    assert info_confidentiality["end_to_end_ready"] is True


def test_cap_attestation_status_reuses_fresh_verified_cache_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from routstr.core import attestation as attestation_module

    now = 1_800_000_000
    calls = 0

    def cap_status() -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "claims_verified": True,
                "state": "unlocked",
                "mode": "password",
                "tenant_id": "cap-test-org-routstr-core",
                "claims_instance_id": "routstr-core",
                "instance_id": "cap-test-org-routstr-core-routstr-core",
                "error": None,
                "claims_error": None,
            }
        raise TimeoutError("status endpoint stalled")

    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    monkeypatch.setattr(
        settings,
        "routstr_tee_client_confidentiality_boundary",
        "attested-tls-termination",
    )
    monkeypatch.setattr(settings, "routstr_attestation_document_path", "")
    monkeypatch.setattr(settings, "routstr_attestation_document_format", "")
    monkeypatch.setattr(settings, "routstr_attestation_public_key", "")
    monkeypatch.setattr(settings, "routstr_attestation_public_key_path", "")
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_b64", "")
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_path", "")
    monkeypatch.setattr(settings, "routstr_tee_cap_attestation_enabled", True)
    monkeypatch.setattr(settings, "routstr_tee_cap_verifier_max_age_seconds", 300)
    monkeypatch.setattr(
        settings,
        "routstr_tee_cap_tee_domain",
        "routstr-core.example.tee.enclava.dev",
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_cap_public_base_url",
        "https://routstr-core.example.tee.enclava.dev/.well-known/confidential",
    )
    monkeypatch.setattr(attestation_module, "_fetch_cap_attestation_status", cap_status)
    monkeypatch.setattr(attestation_module.time, "time", lambda: now)
    attestation_module._clear_cap_attestation_status_cache()

    first = attestation_module.get_routstr_tee_readiness()

    assert first["ready"] is True
    assert first["local_verification"]["verified_at"] == now

    now += 5
    second = attestation_module.get_routstr_tee_readiness()

    assert second["ready"] is True
    assert second["failure_reason"] is None
    assert second["local_verification"]["verified_at"] == 1_800_000_000
    assert second["local_verification"]["expires_at"] == 1_800_000_300

    now = 1_800_000_301
    expired = attestation_module.get_routstr_tee_readiness()

    assert expired["ready"] is False
    assert "status endpoint stalled" in expired["failure_reason"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidentiality_status_endpoint_omits_raw_failure_reason(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BaseUpstreamProvider(
        base_url="https://verified.example/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil-prod"
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=False,
            mode="tinfoil",
            failure_reason=(
                "verifier failed with api_key=SECRET_PROVIDER_KEY "
                "and raw_prompt=SECRET_PROMPT"
            ),
            policy_digest=VALID_POLICY_DIGEST,
        )
    )

    monkeypatch.setattr(settings, "confidential_routing_mode", "required")

    with patch("routstr.proxy._upstreams", [provider]):
        response = await integration_client.get("/v1/confidentiality/status")

    assert response.status_code == 200
    data = response.json()
    confidentiality = data["providers"][0]["confidentiality"]
    assert "failure_reason" not in confidentiality
    assert confidentiality["failure_reason_digest"].startswith("sha256:")

    serialized = json.dumps(data)
    assert "SECRET_PROVIDER_KEY" not in serialized
    assert "SECRET_PROMPT" not in serialized
    assert "api_key" not in serialized
    assert "raw_prompt" not in serialized


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_confidentiality_status_exposes_redacted_failure_reason(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BaseUpstreamProvider(
        base_url="https://verified.example/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil-prod"
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=False,
            mode="tinfoil",
            failure_reason=(
                "verifier failed with api_key=SECRET_PROVIDER_KEY "
                "and raw_prompt=SECRET_PROMPT"
            ),
            policy_digest=VALID_POLICY_DIGEST,
            verified_claims={"release_digest": VALID_PROVIDER_RELEASE_DIGEST},
        )
    )

    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    admin_token = "test-admin-confidentiality-diagnostics"
    admin_sessions[admin_token] = 4_102_444_800

    try:
        with patch("routstr.proxy._upstreams", [provider]):
            response = await integration_client.get(
                "/admin/api/confidentiality/status",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
    finally:
        admin_sessions.pop(admin_token, None)

    assert response.status_code == 200
    data = response.json()
    confidentiality = data["providers"][0]["confidentiality"]
    assert confidentiality["failure_reason"] == (
        "verifier failed with api_key: [REDACTED] and raw_prompt: [REDACTED]"
    )
    assert confidentiality["policy_digest"] == VALID_POLICY_DIGEST
    assert confidentiality["verified_claims_present"] is True
    assert confidentiality["verified_claims_keys"] == ["release_digest"]
    assert "verified_claims" not in confidentiality

    serialized = json.dumps(data)
    assert "SECRET_PROVIDER_KEY" not in serialized
    assert "SECRET_PROMPT" not in serialized


@pytest.mark.integration
@pytest.mark.asyncio
async def test_models_endpoint_exposes_public_confidentiality_metadata(
    integration_client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = b"fake-tdx-quote"
    hpke_key_config = _ehbp_key_config()
    evidence_path = tmp_path / "quote.bin"
    evidence_path.write_bytes(evidence)

    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    monkeypatch.setattr(
        settings,
        "routstr_tee_client_confidentiality_boundary",
        "attested-tls-termination",
    )
    monkeypatch.setattr(
        settings, "routstr_attestation_document_path", str(evidence_path)
    )
    monkeypatch.setattr(settings, "routstr_attestation_document_format", "tdx_quote")
    _configure_routstr_tee_attestation_stub(monkeypatch)
    monkeypatch.setattr(
        settings,
        "routstr_attestation_public_key",
        "-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----",
    )
    monkeypatch.setattr(settings, "routstr_attestation_public_key_path", "")
    monkeypatch.setattr(
        settings,
        "routstr_attestation_hpke_key_config_b64",
        base64.b64encode(hpke_key_config).decode("ascii"),
    )
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_path", "")

    def fake_verify_routstr_tee_evidence(**kwargs: object) -> dict[str, object]:
        tee = kwargs["tee"]
        assert isinstance(tee, dict)
        hpke_key_config = tee["hpke_key_config"]
        assert isinstance(hpke_key_config, dict)
        return {
            "verified": True,
            "verifier": "unit-test-routstr-tee-verifier",
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "failure_reason": None,
            "evidence_digest": str(tee["attestation_evidence_digest"]),
            "verified_claims": _local_tee_verified_claims_with_secrets(
                hpke_key_config_digest=str(hpke_key_config["key_config_digest"]),
                hpke_public_key_digest=str(hpke_key_config["public_key_digest"]),
                public_key_digest=str(tee["public_key_digest"]),
            ),
        }

    monkeypatch.setattr(
        "routstr.core.attestation._verify_routstr_tee_evidence",
        fake_verify_routstr_tee_evidence,
    )

    provider_claims = _public_tinfoil_proof_claims(["tinfoil/gpt-secure"])
    provider_policy = {
        **_tinfoil_policy_for_model_ids(["tinfoil/gpt-secure"]),
        "expected_release_digest": _digest("public-tinfoil-router-release"),
    }
    secure_model = Model(
        id="tinfoil/gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        confidentiality={
            "enabled": True,
            "verified": True,
            "attestation_status": "verified",
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 1_800_000_000,
            "model_ids": ["tinfoil/gpt-secure"],
            "model_id_prefixes": [],
            "supported_endpoints": [
                "/v1/chat/completions",
                "/v1/audio",
                "/v1/embeddings",
                "/v1/responses",
            ],
            "metadata_leakage": ["model", "usage"],
            "verified_claims": provider_claims,
            "proof_claims": provider_claims,
            "confidentiality_policy": provider_policy,
        },
    )

    with (
        patch("routstr.proxy._unique_models", {"tinfoil/gpt-secure": secure_model}),
        patch(
            "routstr.proxy.is_routable_confidential_model",
            return_value=True,
            create=True,
        ),
    ):
        response = await integration_client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()
    public_key_digest = (
        "sha256:"
        + hashlib.sha256(
            b"-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----"
        ).hexdigest()
    )
    assert data["routstr_tee"]["required"] is True
    assert data["routstr_tee"]["ready"] is True
    assert data["routstr_tee"]["client_confidentiality"] == {
        "mode": "attested-tls-termination",
        "tls_terminates_in_attested_tee": True,
        "inbound_ehbp_ohttp_request_decryption": False,
        "attested_tls_public_key_digest": public_key_digest,
    }
    assert data["routstr_tee"]["local_verification"]["verified"] is True
    assert "verified_claims" not in data["routstr_tee"]["local_verification"]
    assert len(data["data"]) == 1
    model = data["data"][0]
    assert model["id"] == "tinfoil/gpt-secure"
    assert model["confidential"] is True
    assert model["attestation_provider"] == "tinfoil"
    assert model["attestation_status"] == "verified"
    assert model["attestation_evidence_digest"] == VALID_EVIDENCE_DIGEST
    assert model["confidentiality"]["verified"] is True
    assert model["confidentiality"]["verified_claims_digest"] == _sha256_json_digest(
        provider_claims
    )
    assert model["confidentiality"]["proof_claims"] == provider_claims
    assert "verified_claims" not in model["confidentiality"]
    assert "confidentiality_policy" not in model["confidentiality"]
    serialized = json.dumps(data)
    assert "api_key" not in serialized
    assert "SECRET" not in serialized


@pytest.mark.integration
@pytest.mark.asyncio
async def test_models_endpoint_does_not_mark_unroutable_confidential_model_verified(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(
        "routstr.core.attestation.get_public_routstr_tee_status",
        lambda: {
            "required": True,
            "ready": True,
            "client_confidentiality": {
                "mode": "attested-tls-termination",
                "tls_terminates_in_attested_tee": True,
                "inbound_ehbp_ohttp_request_decryption": False,
                "attested_tls_public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            },
        },
    )
    provider_claims = _public_tinfoil_proof_claims(["tinfoil/gpt-secure"])
    provider_policy = {
        **_tinfoil_policy_for_model_ids(["tinfoil/gpt-secure"]),
        "expected_release_digest": _digest("public-tinfoil-router-release"),
    }
    secure_model = Model(
        id="tinfoil/gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        confidentiality={
            "enabled": True,
            "verified": True,
            "attestation_status": "verified",
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 1_800_000_000,
            "model_ids": ["tinfoil/gpt-secure"],
            "model_id_prefixes": [],
            "supported_endpoints": ["/v1/chat/completions"],
            "metadata_leakage": ["model"],
            "verified_claims": provider_claims,
            "proof_claims": provider_claims,
            "confidentiality_policy": provider_policy,
        },
    )

    with (
        patch("routstr.proxy._unique_models", {"tinfoil/gpt-secure": secure_model}),
        patch(
            "routstr.proxy.is_routable_confidential_model",
            return_value=False,
            create=True,
        ),
    ):
        response = await integration_client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"] == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_models_endpoint_marks_confidential_unavailable_without_routstr_tee(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider proof alone must not advertise end-to-end confidentiality."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    monkeypatch.setattr(settings, "routstr_attestation_document_path", "")
    monkeypatch.setattr(settings, "routstr_attestation_document_format", "")
    monkeypatch.setattr(settings, "routstr_attestation_public_key", "")
    monkeypatch.setattr(settings, "routstr_attestation_public_key_path", "")
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_b64", "")
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_path", "")
    monkeypatch.setattr(settings, "routstr_tee_verifier_command", "")
    provider_claims = _public_tinfoil_proof_claims(["tinfoil/gpt-secure"])
    provider_policy = {
        **_tinfoil_policy_for_model_ids(["tinfoil/gpt-secure"]),
        "expected_release_digest": _digest("public-tinfoil-router-release"),
    }

    secure_model = Model(
        id="tinfoil/gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        confidentiality={
            "enabled": True,
            "verified": True,
            "attestation_status": "verified",
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 1_800_000_000,
            "model_ids": ["tinfoil/gpt-secure"],
            "model_id_prefixes": [],
            "supported_endpoints": ["/v1/chat/completions"],
            "metadata_leakage": ["model"],
            "verified_claims": provider_claims,
            "proof_claims": provider_claims,
            "confidentiality_policy": provider_policy,
        },
    )

    with patch("routstr.proxy._unique_models", {"tinfoil/gpt-secure": secure_model}):
        response = await integration_client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()
    assert data["routstr_tee"]["required"] is True
    assert data["routstr_tee"]["ready"] is False
    assert data["data"] == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_models_endpoint_requires_local_tee_proof_before_confidential_true(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare ready flag must not become end-to-end confidentiality proof."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    monkeypatch.setattr(
        "routstr.core.attestation.get_public_routstr_tee_status",
        lambda: {
            "required": True,
            "ready": True,
            "client_confidentiality": {
                "mode": "attested-tls-termination",
                "tls_terminates_in_attested_tee": True,
                "inbound_ehbp_ohttp_request_decryption": False,
            },
        },
    )
    provider_claims = _public_tinfoil_proof_claims(["tinfoil/gpt-secure"])
    provider_policy = {
        **_tinfoil_policy_for_model_ids(["tinfoil/gpt-secure"]),
        "expected_release_digest": _digest("public-tinfoil-router-release"),
    }
    secure_model = Model(
        id="tinfoil/gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        confidentiality={
            "enabled": True,
            "verified": True,
            "attestation_status": "verified",
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 1_800_000_000,
            "model_ids": ["tinfoil/gpt-secure"],
            "model_id_prefixes": [],
            "supported_endpoints": ["/v1/chat/completions"],
            "metadata_leakage": ["model"],
            "verified_claims": provider_claims,
            "proof_claims": provider_claims,
            "confidentiality_policy": provider_policy,
        },
    )

    with (
        patch("routstr.proxy._unique_models", {"tinfoil/gpt-secure": secure_model}),
        patch(
            "routstr.proxy.is_routable_confidential_model",
            return_value=True,
            create=True,
        ),
    ):
        response = await integration_client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()
    assert data["routstr_tee"]["ready"] is False
    assert data["data"] == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_models_endpoint_redacts_raw_confidentiality_diagnostics(
    integration_client: AsyncClient,
) -> None:
    provider_claims = {
        **_public_tinfoil_proof_claims(["tinfoil/gpt-secure"]),
        "debug_secret": "SECRET_STEP",
        "ehbp_key_config_b64": "SECRET_KEY_CONFIG",
        "api_key": "SECRET_PROVIDER_KEY",
        "raw_prompt": "SECRET_PROMPT",
    }
    provider_policy = {
        **_tinfoil_policy_for_model_ids(["tinfoil/gpt-secure"]),
        "expected_release_digest": _digest("public-tinfoil-router-release"),
    }
    secure_model = Model(
        id="tinfoil/gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        confidentiality={
            "enabled": True,
            "verified": True,
            "attestation_status": "verified",
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 1_800_000_000,
            "model_ids": ["tinfoil/gpt-secure"],
            "model_id_prefixes": [],
            "supported_endpoints": ["/v1/chat/completions"],
            "metadata_leakage": ["model"],
            "verified_claims": provider_claims,
            "confidentiality_policy": provider_policy,
            "failure_reason": "api_key=SECRET_PROVIDER_KEY raw_prompt=SECRET_PROMPT",
            "api_key": "SECRET_PROVIDER_KEY",
            "raw_prompt": "SECRET_PROMPT",
        },
    )

    with (
        patch("routstr.proxy._unique_models", {"tinfoil/gpt-secure": secure_model}),
        patch(
            "routstr.proxy.is_routable_confidential_model",
            return_value=True,
            create=True,
        ),
    ):
        response = await integration_client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()
    confidentiality = data["data"][0]["confidentiality"]
    assert confidentiality["verified"] is True
    assert confidentiality["verified_claims_digest"].startswith("sha256:")
    assert confidentiality["model_ids"] == ["tinfoil/gpt-secure"]
    assert confidentiality["model_id_prefixes"] == []
    assert confidentiality["supported_endpoints"] == ["/v1/chat/completions"]
    assert confidentiality["metadata_leakage"] == ["model"]
    assert confidentiality["proof_claims"] == _public_tinfoil_proof_claims(
        ["tinfoil/gpt-secure"]
    )
    assert "verified_claims" not in confidentiality
    assert "failure_reason" not in confidentiality
    assert "api_key" not in confidentiality
    assert "raw_prompt" not in confidentiality
    serialized = json.dumps(data)
    assert "SECRET_PROVIDER_KEY" not in serialized
    assert "SECRET_PROMPT" not in serialized
    assert "SECRET_KEY_CONFIG" not in serialized
    assert "SECRET_STEP" not in serialized


@pytest.mark.integration
@pytest.mark.asyncio
async def test_models_endpoint_accepts_tinfoil_prefixed_selector_for_bare_catalog_model(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tinfoil publishes bare catalog IDs while verified policy selectors are prefixed."""
    monkeypatch.setattr(
        "routstr.core.attestation.get_public_routstr_tee_status",
        lambda: {"required": False, "ready": False},
    )
    provider_claims = _public_tinfoil_proof_claims(["tinfoil/kimi-k2-6"])
    provider_policy = {
        **_tinfoil_policy_for_model_ids(["tinfoil/kimi-k2-6"]),
        "expected_release_digest": _digest("public-tinfoil-router-release"),
    }
    secure_model = Model(
        id="kimi-k2-6",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        confidentiality={
            "enabled": True,
            "verified": True,
            "attestation_status": "verified",
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 1_800_000_000,
            "model_ids": ["tinfoil/kimi-k2-6"],
            "model_id_prefixes": [],
            "supported_endpoints": ["/v1/chat/completions"],
            "metadata_leakage": ["model"],
            "verified_claims": provider_claims,
            "proof_claims": provider_claims,
            "confidentiality_policy": provider_policy,
        },
    )

    with (
        patch("routstr.proxy._unique_models", {"kimi-k2-6": secure_model}),
        patch(
            "routstr.proxy.is_routable_confidential_model",
            return_value=True,
            create=True,
        ),
    ):
        response = await integration_client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 1
    model = data["data"][0]
    assert model["id"] == "kimi-k2-6"
    assert model["confidential"] is True
    assert model["confidentiality"]["model_ids"] == ["tinfoil/kimi-k2-6"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_models_endpoint_drops_verified_confidentiality_for_unbound_model_row(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proof for one model must not be exposed on a different model row."""
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(
        "routstr.core.attestation.get_public_routstr_tee_status",
        lambda: {
            "required": True,
            "ready": True,
            "client_confidentiality": {
                "mode": "attested-tls-termination",
                "tls_terminates_in_attested_tee": True,
                "inbound_ehbp_ohttp_request_decryption": False,
                "attested_tls_public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            },
        },
    )
    provider_claims = _public_tinfoil_proof_claims(["tinfoil/gpt-secure"])
    provider_policy = {
        **_tinfoil_policy_for_model_ids(["tinfoil/gpt-secure"]),
        "expected_release_digest": _digest("public-tinfoil-router-release"),
    }
    wrong_model = Model(
        id="tinfoil/other-model",
        name="Wrong Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        confidentiality={
            "enabled": True,
            "verified": True,
            "attestation_status": "verified",
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 1_800_000_000,
            "model_ids": ["tinfoil/gpt-secure"],
            "model_id_prefixes": [],
            "supported_endpoints": ["/v1/chat/completions"],
            "metadata_leakage": ["model"],
            "verified_claims": provider_claims,
            "proof_claims": provider_claims,
            "confidentiality_policy": provider_policy,
        },
    )

    with (
        patch("routstr.proxy._unique_models", {"tinfoil/other-model": wrong_model}),
        patch(
            "routstr.proxy.is_routable_confidential_model",
            return_value=True,
            create=True,
        ),
    ):
        response = await integration_client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"] == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_models_endpoint_drops_verified_confidentiality_with_malformed_public_lists(
    integration_client: AsyncClient,
) -> None:
    secure_model = Model(
        id="tinfoil/gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        confidentiality={
            "enabled": True,
            "verified": True,
            "attestation_status": "verified",
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 1_800_000_000,
            "model_ids": [
                "tinfoil/gpt-secure",
                {"api_key": "SECRET_MODEL_SELECTOR"},
            ],
            "model_id_prefixes": ["tinfoil/"],
            "supported_endpoints": ["/v1/chat/completions"],
            "metadata_leakage": ["model"],
            "proof_claims": _public_tinfoil_proof_claims(["tinfoil/gpt-secure"]),
        },
    )

    with patch("routstr.proxy._unique_models", {"tinfoil/gpt-secure": secure_model}):
        response = await integration_client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()
    assert data["data"] == []
    serialized = json.dumps(data)
    assert "SECRET_MODEL_SELECTOR" not in serialized


@pytest.mark.integration
@pytest.mark.asyncio
async def test_models_endpoint_drops_conflicting_nested_attestation_status(
    integration_client: AsyncClient,
) -> None:
    secure_model = Model(
        id="tinfoil/gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        confidentiality={
            "enabled": True,
            "verified": True,
            "attestation_status": "unavailable",
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 1_800_000_000,
            "model_ids": ["tinfoil/gpt-secure"],
            "model_id_prefixes": [],
            "supported_endpoints": ["/v1/chat/completions"],
            "metadata_leakage": ["model"],
            "proof_claims": _public_tinfoil_proof_claims(["tinfoil/gpt-secure"]),
        },
    )

    with patch("routstr.proxy._unique_models", {"tinfoil/gpt-secure": secure_model}):
        response = await integration_client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()
    assert data["data"] == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidentiality_refresh_exception_clears_verified_status() -> None:
    """A verifier exception must fail closed, not preserve stale verified status."""

    class FailingVerifierProvider(BaseUpstreamProvider):
        provider_type = "failing-verifier"

        async def refresh_confidentiality_status(self) -> ConfidentialityStatus:
            raise RuntimeError("verifier unavailable")

    provider = FailingVerifierProvider(
        base_url="https://verified.example/v1",
        api_key="test",
    )
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="test-verifier",
            model_ids=["gpt-secure"],
            verifier="unit-test-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_claims={"tee": "unit-test"},
        )
    )

    from routstr import proxy

    with patch("routstr.proxy._upstreams", [provider]):
        await proxy.refresh_confidentiality_statuses()

    status = provider.confidentiality_status()
    assert status.enabled is True
    assert status.verified is False
    assert status.failure_reason == "verifier unavailable"
    assert status.model_ids == ["gpt-secure"]
    assert status.policy_digest == VALID_POLICY_DIGEST
    assert status.verifier is None
    assert status.evidence_digest is None
    assert status.verified_claims == {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidentiality_refresh_exception_redacts_base_url_credentials() -> None:
    """Verifier refresh logs must not leak credentials embedded in provider URLs."""

    class FailingVerifierProvider(BaseUpstreamProvider):
        provider_type = "failing-verifier"

        async def refresh_confidentiality_status(self) -> ConfidentialityStatus:
            raise RuntimeError("verifier unavailable")

    provider = FailingVerifierProvider(
        base_url="https://operator:secret-token@verified.example/v1",
        api_key="test",
    )
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="test-verifier",
            model_ids=["gpt-secure"],
            verifier="unit-test-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_claims={"tee": "unit-test"},
        )
    )

    from routstr import proxy

    with (
        patch("routstr.proxy._upstreams", [provider]),
        patch("routstr.proxy.logger.warning") as log_warning,
    ):
        await proxy.refresh_confidentiality_statuses()

    log_warning.assert_called_once()
    _, kwargs = log_warning.call_args
    serialized_log = json.dumps(kwargs, sort_keys=True)
    assert "secret-token" not in serialized_log
    assert "operator:" not in serialized_log
    assert kwargs["extra"]["base_url"] == "https://verified.example/v1"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_confidentiality_refresh_exception_redacts_secret_error_text() -> None:
    """Verifier exception messages can contain sensitive provider diagnostics."""

    class FailingVerifierProvider(BaseUpstreamProvider):
        provider_type = "failing-verifier"

        async def refresh_confidentiality_status(self) -> ConfidentialityStatus:
            raise RuntimeError(
                "verifier failed with api_key=SECRET_PROVIDER_KEY "
                "raw_prompt=SECRET_PROMPT "
                "url=https://operator:secret-token@verified.example/v1"
            )

    provider = FailingVerifierProvider(
        base_url="https://verified.example/v1",
        api_key="test",
    )
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="test-verifier",
            model_ids=["gpt-secure"],
            verifier="unit-test-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_claims={"tee": "unit-test"},
        )
    )

    from routstr import proxy

    with (
        patch("routstr.proxy._upstreams", [provider]),
        patch("routstr.proxy.logger.warning") as log_warning,
    ):
        await proxy.refresh_confidentiality_statuses()

    status = provider.confidentiality_status()
    log_warning.assert_called_once()
    _, kwargs = log_warning.call_args
    serialized = json.dumps(
        {
            "failure_reason": status.failure_reason,
            "log": kwargs,
        },
        sort_keys=True,
    )
    assert "SECRET_PROVIDER_KEY" not in serialized
    assert "SECRET_PROMPT" not in serialized
    assert "secret-token" not in serialized
    assert "operator:" not in serialized
    assert "api_key: [REDACTED]" in serialized
    assert "raw_prompt: [REDACTED]" in serialized
    assert "https://verified.example/v1" in serialized


@pytest.mark.integration
@pytest.mark.asyncio
async def test_routstr_attestation_statement_exposes_policy_without_secrets(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BaseUpstreamProvider(
        base_url="https://verified.example/v1",
        api_key="SECRET_PROVIDER_KEY",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil-prod"
    provider.db_id = 42
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=False,
            mode="tinfoil",
            model_ids=["secure-model"],
            policy_digest=VALID_POLICY_DIGEST,
            failure_reason="cryptographic verifier not implemented",
        )
    )

    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_attestation_document_path", "")
    monkeypatch.setattr(settings, "routstr_attestation_document_format", "")
    monkeypatch.setattr(settings, "routstr_attestation_public_key", "")
    monkeypatch.setattr(settings, "routstr_attestation_public_key_path", "")
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_b64", "")
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_path", "")

    with patch("routstr.proxy._upstreams", [provider]):
        response = await integration_client.get("/.well-known/routstr-attestation")

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "routstr"
    assert data["schema_version"] == "routstr-attestation-v1"
    assert data["routing_policy_digest"].startswith("sha256:")
    assert data["routing_policy"]["required"] is True
    assert data["routing_policy"]["providers"][0]["provider_type"] == "tinfoil"
    assert data["routing_policy"]["providers"][0]["confidentiality"] == {
        "enabled": True,
        "verified": False,
        "mode": "tinfoil",
        "verifier": None,
        "evidence_digest": None,
        "verified_at": None,
        "expires_at": None,
        "model_ids": ["secure-model"],
        "model_id_prefixes": [],
    }
    assert data["tee"]["available"] is False

    serialized = json.dumps(data)
    assert "SECRET_PROVIDER_KEY" not in serialized
    assert "api_key" not in serialized


@pytest.mark.integration
@pytest.mark.asyncio
async def test_routstr_attestation_statement_does_not_publish_routable_models_without_local_tee(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_missing_local_tee(monkeypatch)

    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil-prod"
    _mark_provider_verified(provider, model_ids=["tinfoil/gpt-secure"])
    secure_model = Model(
        id="tinfoil/gpt-secure",
        name="Secure Model",
        created=1,
        description="desc",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="gpt",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.001, completion=0.002),
        supported_endpoints=["/v1/chat/completions"],
    )

    with (
        patch("routstr.proxy._upstreams", [provider]),
        patch("routstr.proxy._model_instances", {"tinfoil/gpt-secure": secure_model}),
        patch("routstr.proxy._provider_map", {"tinfoil/gpt-secure": [provider]}),
    ):
        response = await integration_client.get("/.well-known/routstr-attestation")

    assert response.status_code == 200
    data = response.json()
    assert data["tee"]["available"] is False
    assert data["routing_policy"]["routable_with_full_attestation"] == {
        "tinfoil": [],
        "ppq-private": [],
        "privatemode": [],
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_routstr_attestation_statement_includes_configured_tee_evidence(
    integration_client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = b"fake-tdx-quote"
    hpke_key_config = _ehbp_key_config()
    evidence_path = tmp_path / "quote.bin"
    evidence_path.write_bytes(evidence)
    public_key = "-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----"

    monkeypatch.setattr(
        settings, "routstr_attestation_document_path", str(evidence_path)
    )
    monkeypatch.setattr(settings, "routstr_attestation_document_format", "tdx_quote")
    monkeypatch.setattr(settings, "routstr_attestation_public_key", public_key)
    monkeypatch.setattr(settings, "routstr_attestation_public_key_path", "")
    monkeypatch.setattr(
        settings,
        "routstr_attestation_hpke_key_config_b64",
        base64.b64encode(hpke_key_config).decode("ascii"),
    )
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_path", "")

    response = await integration_client.get("/v1/confidentiality/attestation")

    assert response.status_code == 200
    data = response.json()
    assert data["tee"]["available"] is True
    assert data["tee"]["evidence_format"] == "tdx_quote"
    assert (
        data["tee"]["attestation_document_b64"] == base64.b64encode(evidence).decode()
    )
    assert data["tee"]["attestation_evidence_digest"] == (
        "sha256:" + hashlib.sha256(evidence).hexdigest()
    )
    assert data["tee"]["public_key"] == public_key
    assert data["tee"]["public_key_digest"] == (
        "sha256:" + hashlib.sha256(public_key.encode("utf-8")).hexdigest()
    )
    assert data["tee"]["hpke_key_config"] == {
        "available": True,
        "key_config_b64": base64.b64encode(hpke_key_config).decode("ascii"),
        "key_config_digest": "sha256:" + hashlib.sha256(hpke_key_config).hexdigest(),
        "public_key_hex": "55" * 32,
        "public_key_digest": "sha256:" + hashlib.sha256(b"\x55" * 32).hexdigest(),
        "failure_reason": None,
    }
    assert data["tee"]["local_verification"]["verified"] is False
    assert "failure_reason" not in data["tee"]["local_verification"]
    assert data["tee"]["local_verification"]["failure_reason_digest"].startswith(
        "sha256:"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_routstr_attestation_statement_omits_raw_local_verifier_claims(
    integration_client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = b"fake-tdx-quote"
    hpke_key_config = _ehbp_key_config()
    evidence_path = tmp_path / "quote.bin"
    evidence_path.write_bytes(evidence)

    monkeypatch.setattr(
        settings, "routstr_attestation_document_path", str(evidence_path)
    )
    monkeypatch.setattr(settings, "routstr_attestation_document_format", "tdx_quote")
    monkeypatch.setattr(
        settings,
        "routstr_attestation_public_key",
        "-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----",
    )
    monkeypatch.setattr(settings, "routstr_attestation_public_key_path", "")
    monkeypatch.setattr(
        settings,
        "routstr_attestation_hpke_key_config_b64",
        base64.b64encode(hpke_key_config).decode("ascii"),
    )
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_path", "")

    def fake_verify_routstr_tee_evidence(**kwargs: object) -> dict[str, object]:
        tee = kwargs["tee"]
        assert isinstance(tee, dict)
        hpke_key_config = tee["hpke_key_config"]
        assert isinstance(hpke_key_config, dict)
        return {
            "verified": True,
            "verifier": "unit-test-routstr-tee-verifier",
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "failure_reason": None,
            "evidence_digest": str(tee["attestation_evidence_digest"]),
            "verified_claims": _local_tee_verified_claims_with_secrets(
                hpke_key_config_digest=str(hpke_key_config["key_config_digest"]),
                hpke_public_key_digest=str(hpke_key_config["public_key_digest"]),
                public_key_digest=str(tee["public_key_digest"]),
            ),
        }

    monkeypatch.setattr(
        "routstr.core.attestation._verify_routstr_tee_evidence",
        fake_verify_routstr_tee_evidence,
    )

    response = await integration_client.get("/.well-known/routstr-attestation")

    assert response.status_code == 200
    data = response.json()
    local_verification = data["tee"]["local_verification"]
    assert local_verification["verified"] is True
    assert (
        local_verification["evidence_digest"]
        == data["tee"]["attestation_evidence_digest"]
    )
    assert local_verification["verified_claims_digest"].startswith("sha256:")
    assert local_verification[
        "proof_claims"
    ] == _expected_local_tee_public_proof_claims(
        hpke_key_config_digest=data["tee"]["hpke_key_config"]["key_config_digest"],
        hpke_public_key_digest=data["tee"]["hpke_key_config"]["public_key_digest"],
        public_key_digest=data["tee"]["public_key_digest"],
    )
    assert "verified_claims" not in local_verification

    serialized = json.dumps(data)
    assert "SECRET_LOCAL_VERIFIER_CLAIM" not in serialized
    assert "SECRET_PROMPT" not in serialized
    assert "SECRET_STEP" not in serialized
    assert "api_key" not in serialized
    assert "raw_prompt" not in serialized


@pytest.mark.integration
@pytest.mark.asyncio
async def test_routstr_hpke_keys_endpoint_serves_configured_key_config(
    integration_client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hpke_key_config = _ehbp_key_config(public_key=b"\x66" * 32)
    key_config_path = tmp_path / "hpke-keys.bin"
    key_config_path.write_bytes(hpke_key_config)

    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_b64", "")
    monkeypatch.setattr(
        settings,
        "routstr_attestation_hpke_key_config_path",
        str(key_config_path),
    )

    response = await integration_client.get("/.well-known/hpke-keys")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/ohttp-keys"
    assert response.content == hpke_key_config
