import base64
import hashlib
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from routstr.core.db import ApiKey
from routstr.core.exceptions import UpstreamError
from routstr.payment.cost_calculation import CostData
from routstr.payment.models import Architecture, Model, Pricing
from routstr.upstream.base import ConfidentialityStatus
from routstr.upstream.ehbp import (
    AEAD_AES_256_GCM,
    KDF_HKDF_SHA256,
    KEM_X25519_HKDF_SHA256,
    decrypt_ehbp_request_for_test,
    encrypt_ehbp_response_body_for_test,
    parse_ehbp_key_config,
)
from routstr.upstream.ppqai import PPQPrivateUpstreamProvider
from routstr.upstream.privatemode import PrivatemodeUpstreamProvider
from routstr.upstream.tinfoil import TinfoilUpstreamProvider

VALID_POLICY_DIGEST = "sha256:" + hashlib.sha256(b"test-policy").hexdigest()
VALID_EVIDENCE_DIGEST = "sha256:" + hashlib.sha256(b"test-evidence").hexdigest()
VALID_RELEASE_DIGEST = "sha256:" + hashlib.sha256(b"test-release").hexdigest()
VALID_TLS_DIGEST = "sha256:" + hashlib.sha256(b"test-tls-key").hexdigest()
VALID_ENCLAVE_MEASUREMENT = "sha256:" + hashlib.sha256(
    b"test-enclave-measurement"
).hexdigest()
VALID_CODE_MEASUREMENT = "sha256:" + hashlib.sha256(
    b"test-code-measurement"
).hexdigest()
VALID_ATTESTATION_REPORT_DIGEST = "sha256:" + hashlib.sha256(
    b"test-attestation-report"
).hexdigest()
VALID_PROXY_BINARY_DIGEST = "sha256:" + hashlib.sha256(
    b"test-proxy-binary"
).hexdigest()
VALID_PROXY_IMAGE_DIGEST = "sha256:" + hashlib.sha256(
    b"test-proxy-image"
).hexdigest()
VALID_PRIVATEMODE_MANIFEST_DIGEST = "sha256:" + hashlib.sha256(
    b"test-privatemode-manifest"
).hexdigest()
VALID_PRIVATEMODE_COORDINATOR_MEASUREMENT = "sha256:" + hashlib.sha256(
    b"test-privatemode-coordinator"
).hexdigest()
VALID_PRIVATEMODE_SECRET_SERVICE_MEASUREMENT = "sha256:" + hashlib.sha256(
    b"test-privatemode-secret-service"
).hexdigest()
VALID_PRIVATEMODE_AI_WORKER_MEASUREMENT = "sha256:" + hashlib.sha256(
    b"test-privatemode-ai-worker"
).hexdigest()
VALID_PRIVATEMODE_WORKLOAD_IDENTITY_DIGEST = "sha256:" + hashlib.sha256(
    b"test-privatemode-workload-identity"
).hexdigest()
VALID_PRIVATEMODE_COORDINATOR_DOC_DIGEST = "sha256:" + hashlib.sha256(
    b"test-privatemode-coordinator-doc"
).hexdigest()
VALID_PRIVATEMODE_MESH_CA_DIGEST = "sha256:" + hashlib.sha256(
    b"test-privatemode-mesh-ca"
).hexdigest()
VALID_PRIVATEMODE_SECRET_SERVICE_CERT_DIGEST = "sha256:" + hashlib.sha256(
    b"test-privatemode-secret-service-cert"
).hexdigest()
VALID_PRIVATEMODE_AI_WORKER_MANIFEST_DIGEST = "sha256:" + hashlib.sha256(
    b"test-privatemode-ai-worker-manifest"
).hexdigest()
VALID_PRIVATEMODE_NVIDIA_OCSP_HEADER_DIGEST = "sha256:" + hashlib.sha256(
    b"test-privatemode-nvidia-ocsp-header"
).hexdigest()
VALID_PRIVATEMODE_NVIDIA_OCSP_MAC_DIGEST = "sha256:" + hashlib.sha256(
    b"test-privatemode-nvidia-ocsp-mac"
).hexdigest()
VALID_PRIVATEMODE_PROMPT_CIPHERTEXT_DIGEST = "sha256:" + hashlib.sha256(
    b"test-privatemode-prompt-ciphertext"
).hexdigest()
VALID_PRIVATEMODE_INFERENCE_SECRET_DIGEST = "sha256:" + hashlib.sha256(
    b"test-privatemode-inference-secret"
).hexdigest()
VALID_PPQ_ATTESTATION_BUNDLE_URL = "https://api.ppq.ai/private"
VALID_PPQ_ATTESTATION_BUNDLE_URL_DIGEST = (
    "sha256:"
    + hashlib.sha256(
        json.dumps(
            VALID_PPQ_ATTESTATION_BUNDLE_URL,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
)


def _raw_public_key(private_key: x25519.X25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _key_config(public_key: bytes) -> bytes:
    return (
        b"\x00"
        + KEM_X25519_HKDF_SHA256.to_bytes(2, "big")
        + public_key
        + (4).to_bytes(2, "big")
        + KDF_HKDF_SHA256.to_bytes(2, "big")
        + AEAD_AES_256_GCM.to_bytes(2, "big")
    )


def _ehbp_verification_steps() -> dict[str, bool]:
    return {
        "hardware_attestation_report": True,
        "hardware_certificate_chain": True,
        "code_transparency": True,
        "measurement_match": True,
        "attested_transport_key_binding": True,
        "freshness": True,
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


def _verified_ehbp_claims(
    mode: str,
    key_config: bytes,
    model_ids: list[str] | None = None,
) -> dict[str, object]:
    parsed_key_config = parse_ehbp_key_config(key_config)
    claims: dict[str, object] = {
        "transport": "ehbp",
        "repo": "tinfoilsh/confidential-model-router",
        "payload_policy_digest": VALID_POLICY_DIGEST,
        "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
        "ehbp_key_config_b64": base64.b64encode(key_config).decode("ascii"),
        "attested_hpke_public_key_hex": parsed_key_config.public_key.hex(),
        "ehbp_public_key_digest": parsed_key_config.public_key_digest,
        "enclave_measurement_fingerprint": VALID_ENCLAVE_MEASUREMENT,
        "code_measurement_fingerprint": VALID_CODE_MEASUREMENT,
        "tls_public_key_fingerprint_sha256": VALID_TLS_DIGEST,
        "verification_steps": _ehbp_verification_steps(),
    }
    if mode == "tinfoil":
        claims.update(
            {
                "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
                "attestation_report_digest": VALID_ATTESTATION_REPORT_DIGEST,
                "release_digest": VALID_RELEASE_DIGEST,
            }
        )
    if mode == "ppq-private-tee":
        selected_model_ids = model_ids or ["private/gpt-oss-120b"]
        claims.update(
            {
                "attestation_bundle_url": VALID_PPQ_ATTESTATION_BUNDLE_URL,
                "attestation_bundle_url_digest": VALID_PPQ_ATTESTATION_BUNDLE_URL_DIGEST,
                "client_encryption_boundary": "routstr-tee-ehbp-proxy",
                "proxy_binary_digest": VALID_PROXY_BINARY_DIGEST,
                "repo": "ppq-ai/private-tee",
                "release_digest": VALID_RELEASE_DIGEST,
                "selected_model_ids": selected_model_ids,
                "backend_model_attestations": _tinfoil_model_attestations_for_model_ids(
                    selected_model_ids,
                    key_config,
                ),
            }
        )
    return claims


def _tinfoil_model_attestations_for_model_ids(
    model_ids: list[str],
    key_config: bytes,
) -> dict[str, dict[str, object]]:
    parsed_key_config = parse_ehbp_key_config(key_config)
    return {
        model_id: {
            "repo": f"tinfoilsh/confidential-{model_id.rsplit('/', 1)[-1]}",
            "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
            "attestation_report_digest": VALID_ATTESTATION_REPORT_DIGEST,
            "attested_hpke_public_key_hex": parsed_key_config.public_key.hex(),
            "enclave_measurement_fingerprint": VALID_ENCLAVE_MEASUREMENT,
            "code_measurement_fingerprint": VALID_CODE_MEASUREMENT,
            "tls_public_key_fingerprint_sha256": VALID_TLS_DIGEST,
            "release_digest": VALID_RELEASE_DIGEST,
            "verification_steps": _ehbp_verification_steps(),
        }
        for model_id in model_ids
    }


def _tinfoil_policy_for_model_ids(model_ids: list[str]) -> dict[str, object]:
    return {
        "repo": "tinfoilsh/confidential-model-router",
        "expected_release_digest": VALID_RELEASE_DIGEST,
        "transport_security": "ehbp",
        "require_model_attestations": True,
        "model_attestation_targets": {
            model_id: {
                "host": f"{model_id.rsplit('/', 1)[-1]}.tinfoil.example",
                "repo": f"tinfoilsh/confidential-{model_id.rsplit('/', 1)[-1]}",
                "expected_code_measurement_fingerprint": VALID_CODE_MEASUREMENT,
                "expected_release_digest": VALID_RELEASE_DIGEST,
            }
            for model_id in model_ids
        },
    }


def _tinfoil_verified_ehbp_claims(
    key_config: bytes,
    model_ids: list[str],
) -> dict[str, object]:
    claims = _verified_ehbp_claims("tinfoil", key_config)
    claims["model_attestations"] = _tinfoil_model_attestations_for_model_ids(
        model_ids,
        key_config,
    )
    return claims


def _privatemode_policy_for_model_ids(model_ids: list[str]) -> dict[str, object]:
    workload_san = "secure-model.default.svc.cluster.local"
    bindings = {
        model_id: {"workload_sans": [workload_san]}
        for model_id in model_ids
    }
    canonical_bindings = {
        model_id: {"workload_ids": [], "workload_sans": [workload_san]}
        for model_id in model_ids
    }
    expected_workload_identity_digest = _sha256_json_digest(
        {"ids": [], "sans": [workload_san]}
    )
    model_workload_binding_digest = _sha256_json_digest(canonical_bindings)
    key_release_binding = _sha256_json_digest(
        {
            "attested_workload_policy_digest": VALID_PRIVATEMODE_AI_WORKER_MEASUREMENT,
            "expected_workload_identity_digest": expected_workload_identity_digest,
            "model_workload_binding_digest": model_workload_binding_digest,
            "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
            "mesh_ca_digest": VALID_PRIVATEMODE_MESH_CA_DIGEST,
            "secret_service_certificate_digest": VALID_PRIVATEMODE_SECRET_SERVICE_CERT_DIGEST,
            "inference_secret_id_digest": VALID_PRIVATEMODE_INFERENCE_SECRET_DIGEST,
            "nvidia_ocsp_policy_mac_digest": VALID_PRIVATEMODE_NVIDIA_OCSP_MAC_DIGEST,
        }
    )
    return {
        "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
        "proxy_image_digest": VALID_PROXY_IMAGE_DIGEST,
        "proxy_binary_digest": VALID_PROXY_BINARY_DIGEST,
        "expected_coordinator_measurement": VALID_PRIVATEMODE_COORDINATOR_MEASUREMENT,
        "expected_secret_service_measurement": VALID_PRIVATEMODE_SECRET_SERVICE_MEASUREMENT,
        "expected_ai_worker_measurement": VALID_PRIVATEMODE_AI_WORKER_MEASUREMENT,
        "expected_gpu_attestation_policy": "nvidia-ocsp-good-only",
        "expected_trust_tier": "app-e2ee",
        "expected_key_release_binding": key_release_binding,
        "expected_workload_identity_digest": expected_workload_identity_digest,
        "expected_workload_sans": [workload_san],
        "model_workload_binding_digest": model_workload_binding_digest,
        "model_workload_bindings": bindings,
    }


def _privatemode_verified_claims(
    model_ids: list[str] | None = None,
) -> dict[str, object]:
    model_ids = model_ids or ["privatemode/model"]
    workload_san = "secure-model.default.svc.cluster.local"
    claims: dict[str, object] = {
        "transport": "privatemode-proxy",
        "trust_tier": "app-e2ee",
        "proxy_base_url": "http://127.0.0.1:8080/v1",
        "payload_policy_digest": VALID_POLICY_DIGEST,
        "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
        "manifest_digest": VALID_PRIVATEMODE_MANIFEST_DIGEST,
        "proxy_image_digest": VALID_PROXY_IMAGE_DIGEST,
        "proxy_binary_digest": VALID_PROXY_BINARY_DIGEST,
        "coordinator_measurement": VALID_PRIVATEMODE_COORDINATOR_MEASUREMENT,
        "secret_service_measurement": VALID_PRIVATEMODE_SECRET_SERVICE_MEASUREMENT,
        "ai_worker_measurement": VALID_PRIVATEMODE_AI_WORKER_MEASUREMENT,
        "gpu_attestation_policy": "nvidia-ocsp-good-only",
        "attested_workload_identity_digest": VALID_PRIVATEMODE_WORKLOAD_IDENTITY_DIGEST,
        "attested_workload_policy_digest": VALID_PRIVATEMODE_AI_WORKER_MEASUREMENT,
        "expected_workload_identity_digest": _sha256_json_digest(
            {"ids": [], "sans": [workload_san]}
        ),
        "selected_model_ids": model_ids,
        "model_workload_binding_digest": _sha256_json_digest(
            {
                model_id: {
                    "workload_ids": [],
                    "workload_sans": [workload_san],
                }
                for model_id in model_ids
            }
        ),
        "coordinator_attestation_doc_digest": VALID_PRIVATEMODE_COORDINATOR_DOC_DIGEST,
        "mesh_ca_digest": VALID_PRIVATEMODE_MESH_CA_DIGEST,
        "secret_service_certificate_digest": VALID_PRIVATEMODE_SECRET_SERVICE_CERT_DIGEST,
        "ai_worker_manifest_digest": VALID_PRIVATEMODE_AI_WORKER_MANIFEST_DIGEST,
        "nvidia_ocsp_policy_header_digest": VALID_PRIVATEMODE_NVIDIA_OCSP_HEADER_DIGEST,
        "nvidia_ocsp_policy_mac_digest": VALID_PRIVATEMODE_NVIDIA_OCSP_MAC_DIGEST,
        "prompt_encryption_ciphertext_digest": VALID_PRIVATEMODE_PROMPT_CIPHERTEXT_DIGEST,
        "inference_secret_id_digest": VALID_PRIVATEMODE_INFERENCE_SECRET_DIGEST,
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
    }
    claims["key_release_binding"] = _privatemode_key_release_binding_digest(claims)
    return claims


def _key() -> ApiKey:
    return ApiKey(hashed_key="abcdef0123" * 4, balance=1_000_000)


def _model() -> Model:
    return Model(
        id="tinfoil/secure-model",
        name="tinfoil/secure-model",
        forwarded_model_id="tinfoil/secure-model",
        created=0,
        description="",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="x",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.0, completion=0.0),
        supported_endpoints=["/v1/chat/completions", "/v1/responses"],
    )


def _other_tinfoil_model() -> Model:
    return Model(
        id="tinfoil/other-model",
        name="tinfoil/other-model",
        forwarded_model_id="tinfoil/other-model",
        created=0,
        description="",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="x",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.0, completion=0.0),
        supported_endpoints=["/v1/chat/completions"],
    )


def _ppq_private_model() -> Model:
    return Model(
        id="private/gpt-oss-120b",
        name="private/gpt-oss-120b",
        forwarded_model_id="private/gpt-oss-120b",
        created=0,
        description="",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="x",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.0, completion=0.0),
        supported_endpoints=["/v1/chat/completions"],
    )


def _aliased_ppq_private_model() -> Model:
    model = _ppq_private_model()
    model.id = "ppq-secure-gpt"
    model.name = "PPQ Secure GPT"
    model.forwarded_model_id = "private/gpt-oss-120b"
    return model


def _privatemode_model() -> Model:
    return Model(
        id="privatemode/gpt-secure",
        name="privatemode/gpt-secure",
        forwarded_model_id="privatemode/gpt-secure",
        created=0,
        description="",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="x",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.0, completion=0.0),
        supported_endpoints=["/v1/chat/completions"],
    )


def _other_privatemode_model() -> Model:
    return Model(
        id="privatemode/other-model",
        name="privatemode/other-model",
        forwarded_model_id="privatemode/other-model",
        created=0,
        description="",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="x",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.0, completion=0.0),
        supported_endpoints=["/v1/chat/completions"],
    )


def _audio_model() -> Model:
    return Model(
        id="tinfoil/whisper-large-v3-turbo",
        name="tinfoil/whisper-large-v3-turbo",
        forwarded_model_id="tinfoil/whisper-large-v3-turbo",
        created=0,
        description="",
        context_length=8192,
        architecture=Architecture(
            modality="audio->text",
            input_modalities=["audio"],
            output_modalities=["text"],
            tokenizer="x",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.0, completion=0.0),
        supported_endpoints=["/v1/audio/transcriptions"],
    )


def _base_audio_model() -> Model:
    return Model(
        id="whisper-large-v3-turbo",
        name="whisper-large-v3-turbo",
        created=0,
        description="",
        context_length=8192,
        architecture=Architecture(
            modality="audio->text",
            input_modalities=["audio"],
            output_modalities=["text"],
            tokenizer="x",
            instruct_type=None,
        ),
        pricing=Pricing(prompt=0.0, completion=0.0),
        supported_endpoints=["/v1/audio/transcriptions"],
    )


def _request() -> MagicMock:
    request = MagicMock()
    request.method = "POST"
    request.query_params = {}
    request.state.request_id = "req-test"
    return request


def _cashu_request(body: bytes) -> MagicMock:
    request = _request()
    request.body = AsyncMock(return_value=body)
    return request


def _cost_data(total_msats: int) -> CostData:
    return CostData(
        base_msats=0,
        input_msats=total_msats,
        output_msats=0,
        total_msats=total_msats,
        total_usd=0.0,
        input_tokens=1,
        output_tokens=1,
    )


def _verified_ehbp_status(
    mode: str,
    *,
    model_ids: list[str] | None = None,
    model_id_prefixes: list[str] | None = None,
) -> ConfidentialityStatus:
    recipient_private = x25519.X25519PrivateKey.from_private_bytes(bytes(range(32)))
    if model_ids is None and model_id_prefixes is None:
        if mode == "ppq-private-tee":
            model_ids = ["private/gpt-oss-120b"]
        else:
            model_ids = ["tinfoil/secure-model"]
    key_config = _key_config(_raw_public_key(recipient_private))
    verified_claims = (
        _tinfoil_verified_ehbp_claims(key_config, model_ids or [])
        if mode == "tinfoil"
        else _verified_ehbp_claims(
            mode,
            key_config,
            model_ids,
        )
    )
    return ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode=mode,
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_claims=verified_claims,
        model_ids=model_ids or [],
        model_id_prefixes=model_id_prefixes or [],
    )


def _bind_local_confidentiality_policy(
    provider: TinfoilUpstreamProvider
    | PPQPrivateUpstreamProvider
    | PrivatemodeUpstreamProvider,
    policy_digest: str = VALID_POLICY_DIGEST,
    policy: dict[str, object] | None = None,
) -> None:
    def policy_getter() -> SimpleNamespace:
        local_policy = policy
        provider_type = getattr(provider, "provider_type", "")
        mode = {
            "tinfoil": "tinfoil",
            "ppq-private": "ppq-private-tee",
            "privatemode": "privatemode",
        }.get(provider_type, "")
        status = provider.confidentiality_status()
        model_ids = list(getattr(status, "model_ids", []) or [])
        model_id_prefixes = list(getattr(status, "model_id_prefixes", []) or [])
        if not model_ids and isinstance(local_policy, dict):
            raw_model_attestations = local_policy.get("model_attestation_targets")
            raw_workload_bindings = local_policy.get("model_workload_bindings")
            if isinstance(raw_model_attestations, dict) and raw_model_attestations:
                model_ids = [
                    model_id
                    for model_id in raw_model_attestations
                    if isinstance(model_id, str) and model_id.strip()
                ]
            elif isinstance(raw_workload_bindings, dict) and raw_workload_bindings:
                model_ids = [
                    model_id
                    for model_id in raw_workload_bindings
                    if isinstance(model_id, str) and model_id.strip()
                ]
        if isinstance(provider, TinfoilUpstreamProvider) and not model_ids:
            model_ids = ["tinfoil/secure-model"]
            model_id_prefixes = []
        elif isinstance(provider, PPQPrivateUpstreamProvider):
            model_ids = model_ids or ["private/gpt-oss-120b"]
            model_id_prefixes = []
        elif isinstance(provider, PrivatemodeUpstreamProvider) and not model_ids:
            model_ids = ["privatemode/gpt-secure"]
            model_id_prefixes = []
        if local_policy is None and isinstance(provider, TinfoilUpstreamProvider):
            if isinstance(model_ids, list) and all(
                isinstance(model_id, str) and model_id.strip()
                for model_id in model_ids
            ):
                local_policy = _tinfoil_policy_for_model_ids(model_ids)
        if local_policy is None and isinstance(provider, PPQPrivateUpstreamProvider):
            local_policy = {
                "attestation_bundle_url": VALID_PPQ_ATTESTATION_BUNDLE_URL,
                "attestation_bundle_url_digest": VALID_PPQ_ATTESTATION_BUNDLE_URL_DIGEST,
                "client_encryption_boundary": "routstr-tee-ehbp-proxy",
                "repo": "ppq-ai/private-tee",
                "expected_release_digest": VALID_RELEASE_DIGEST,
                "expected_code_measurement_fingerprint": VALID_CODE_MEASUREMENT,
                "proxy_binary_digest": VALID_PROXY_BINARY_DIGEST,
                "require_model_attestations": True,
                "model_attestation_targets": {
                    model_id: {
                        "host": f"{model_id.rsplit('/', 1)[-1]}.tinfoil.example",
                        "repo": f"tinfoilsh/confidential-{model_id.rsplit('/', 1)[-1]}",
                        "expected_code_measurement_fingerprint": VALID_CODE_MEASUREMENT,
                        "expected_release_digest": VALID_RELEASE_DIGEST,
                    }
                    for model_id in model_ids
                },
            }
        if local_policy is None and isinstance(provider, PrivatemodeUpstreamProvider):
            local_policy = _privatemode_policy_for_model_ids(model_ids)
        return SimpleNamespace(
            digest=policy_digest,
            provider_type=provider_type,
            mode=mode,
            model_ids=model_ids,
            model_id_prefixes=model_id_prefixes,
            policy=local_policy or {},
        )

    setattr(
        provider,
        "confidentiality_policy",
        policy_getter,
    )


class DummyEhbpAsyncClient:
    request_body: bytes | None = None
    request_headers: dict[str, str] = {}
    recipient_private_key: x25519.X25519PrivateKey
    response_nonce = b"\x44" * 32
    response_payload: dict[str, Any] = {
        "id": "chatcmpl-secure",
        "model": "secure-model",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> "DummyEhbpAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    def build_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        content: bytes,
        params: Any = None,
    ) -> httpx.Request:
        return httpx.Request(method, url, headers=headers, content=content)

    async def send(
        self, request: httpx.Request, stream: bool = False
    ) -> httpx.Response:
        self.__class__.request_body = await request.aread()
        self.__class__.request_headers = dict(request.headers)
        enc = bytes.fromhex(request.headers["ehbp-encapsulated-key"])
        decrypted = decrypt_ehbp_request_for_test(
            self.recipient_private_key,
            enc,
            self.__class__.request_body,
        )
        assert b"SECRET_PROMPT" in decrypted.plaintext
        response_body = json.dumps(self.response_payload).encode()
        encrypted_response = encrypt_ehbp_response_body_for_test(
            decrypted.exported_secret,
            enc,
            self.response_nonce,
            response_body,
        )
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "Ehbp-Response-Nonce": self.response_nonce.hex(),
            },
            content=encrypted_response,
            request=request,
        )

    async def aclose(self) -> None:
        pass


class DummyPPQEhbpAsyncClient(DummyEhbpAsyncClient):
    decrypted_plaintext: bytes | None = None

    async def send(
        self, request: httpx.Request, stream: bool = False
    ) -> httpx.Response:
        response = await super().send(request, stream)
        enc = bytes.fromhex(request.headers["ehbp-encapsulated-key"])
        decrypted = decrypt_ehbp_request_for_test(
            self.recipient_private_key,
            enc,
            self.__class__.request_body or b"",
        )
        self.__class__.decrypted_plaintext = decrypted.plaintext
        return response


class DummyPlainAsyncClient:
    request_headers: dict[str, str] = {}
    request_body: bytes | None = None
    response_payload: dict[str, Any] = {
        "id": "chatcmpl-secure",
        "model": "secure-model",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def build_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        content: Any,
        params: Any = None,
    ) -> httpx.Request:
        return httpx.Request(method, url, headers=headers, content=content)

    async def send(
        self, request: httpx.Request, stream: bool = False
    ) -> httpx.Response:
        self.__class__.request_headers = dict(request.headers)
        self.__class__.request_body = await request.aread()
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=json.dumps(self.response_payload).encode("utf-8"),
            request=request,
        )

    async def aclose(self) -> None:
        pass


class DummyGetAsyncClient:
    request_headers: dict[str, str] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> "DummyGetAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    def build_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        content: Any,
        params: Any = None,
    ) -> httpx.Request:
        return httpx.Request(method, url, headers=headers, content=b"")

    async def send(
        self, request: httpx.Request, stream: bool = False
    ) -> httpx.Response:
        self.__class__.request_headers = dict(request.headers)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=b'{"ok":true}',
            request=request,
        )

    async def aclose(self) -> None:
        pass


class DummyAudioEhbpAsyncClient(DummyEhbpAsyncClient):
    decrypted_plaintext: bytes | None = None
    response_nonce = b"\x45" * 32

    async def send(
        self, request: httpx.Request, stream: bool = False
    ) -> httpx.Response:
        self.__class__.request_body = await request.aread()
        self.__class__.request_headers = dict(request.headers)
        enc = bytes.fromhex(request.headers["ehbp-encapsulated-key"])
        decrypted = decrypt_ehbp_request_for_test(
            self.recipient_private_key,
            enc,
            self.__class__.request_body,
        )
        self.__class__.decrypted_plaintext = decrypted.plaintext
        assert b"SECRET_AUDIO_BYTES" in decrypted.plaintext
        response_body = b'{"text":"ok","usage":{"type":"duration","seconds":"1"}}'
        encrypted_response = encrypt_ehbp_response_body_for_test(
            decrypted.exported_secret,
            enc,
            self.response_nonce,
            response_body,
        )
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "Ehbp-Response-Nonce": self.response_nonce.hex(),
            },
            content=encrypted_response,
            request=request,
        )


class FailIfNetworkAsyncClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("network send attempted before attestation")


class SecretExceptionEhbpAsyncClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def build_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        content: Any,
        params: Any = None,
    ) -> httpx.Request:
        return httpx.Request(method, url, headers=headers, content=content)

    async def send(
        self, request: httpx.Request, stream: bool = False
    ) -> httpx.Response:
        raise RuntimeError("serializer failed while handling SECRET_PROMPT")

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_tinfoil_forward_request_requires_verified_ehbp_transport() -> None:
    provider = TinfoilUpstreamProvider(api_key="test")

    with pytest.raises(UpstreamError) as exc_info:
        await provider.forward_request(
            request=_request(),
            path="v1/chat/completions",
            headers={"authorization": "Bearer test"},
            request_body=b'{"model":"tinfoil/secure-model","messages":[]}',
            key=_key(),
            max_cost_for_model=10_000,
            session=MagicMock(),
            model_obj=_model(),
        )

    assert "verified EHBP transport is required" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tinfoil_forward_request_requires_verification_for_requested_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = TinfoilUpstreamProvider(api_key="test")
    _bind_local_confidentiality_policy(provider)
    provider.set_confidentiality_status(
        _verified_ehbp_status("tinfoil", model_ids=["tinfoil/secure-model"])
    )
    monkeypatch.setattr(
        "routstr.upstream.base.httpx.AsyncClient",
        FailIfNetworkAsyncClient,
    )

    with pytest.raises(UpstreamError) as exc_info:
        await provider.forward_request(
            request=_request(),
            path="v1/chat/completions",
            headers={"authorization": "Bearer test"},
            request_body=b'{"model":"tinfoil/other-model","messages":[]}',
            key=_key(),
            max_cost_for_model=10_000,
            session=MagicMock(),
            model_obj=_other_tinfoil_model(),
        )

    assert "verified confidential transport does not cover requested model" in str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_tinfoil_forward_request_requires_verification_for_forwarded_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = TinfoilUpstreamProvider(api_key="test")
    _bind_local_confidentiality_policy(
        provider,
        policy=_tinfoil_policy_for_model_ids(["secure-model"]),
    )
    provider.set_confidentiality_status(
        _verified_ehbp_status("tinfoil", model_ids=["secure-model"])
    )
    model = _model()
    model.id = "secure-model"
    model.forwarded_model_id = "tinfoil/different-secure-backend"
    monkeypatch.setattr(
        "routstr.upstream.base.httpx.AsyncClient",
        FailIfNetworkAsyncClient,
    )

    with pytest.raises(UpstreamError) as exc_info:
        await provider.forward_request(
            request=_request(),
            path="v1/chat/completions",
            headers={"authorization": "Bearer test"},
            request_body=b'{"model":"secure-model","messages":[]}',
            key=_key(),
            max_cost_for_model=10_000,
            session=MagicMock(),
            model_obj=model,
        )

    assert "verified confidential transport does not cover requested model" in str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_tinfoil_forward_get_request_requires_verified_ehbp_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = TinfoilUpstreamProvider(api_key="test")
    request = _request()
    request.method = "GET"
    monkeypatch.setattr(
        "routstr.upstream.base.httpx.AsyncClient",
        FailIfNetworkAsyncClient,
    )

    with pytest.raises(UpstreamError) as exc_info:
        await provider.forward_get_request(
            request=request,
            path="tee/attestation",
            headers={"authorization": "Bearer test"},
        )

    assert "verified EHBP transport is required" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tinfoil_forward_get_request_replaces_user_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = TinfoilUpstreamProvider(api_key="tinfoil-upstream-key")
    _bind_local_confidentiality_policy(provider)
    provider.set_confidentiality_status(
        _verified_ehbp_status("tinfoil", model_ids=["tinfoil/secure-model"])
    )
    request = _request()
    request.method = "GET"
    DummyGetAsyncClient.request_headers = {}
    monkeypatch.setattr(
        "routstr.upstream.base.httpx.AsyncClient",
        DummyGetAsyncClient,
    )

    response = await provider.forward_get_request(
        request=request,
        path="tee/attestation",
        headers={"authorization": "Bearer routstr-user-key"},
    )

    assert response.status_code == 200
    assert DummyGetAsyncClient.request_headers["authorization"] == (
        "Bearer tinfoil-upstream-key"
    )
    assert "routstr-user-key" not in str(DummyGetAsyncClient.request_headers)


@pytest.mark.asyncio
async def test_tinfoil_forward_get_request_rejects_non_tee_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = TinfoilUpstreamProvider(api_key="tinfoil-upstream-key")
    _bind_local_confidentiality_policy(provider)
    provider.set_confidentiality_status(
        _verified_ehbp_status("tinfoil", model_ids=["tinfoil/secure-model"])
    )
    request = _request()
    request.method = "GET"
    monkeypatch.setattr(
        "routstr.upstream.base.httpx.AsyncClient",
        FailIfNetworkAsyncClient,
    )

    with pytest.raises(UpstreamError) as exc_info:
        await provider.forward_get_request(
            request=request,
            path="v1/models",
            headers={"authorization": "Bearer routstr-user-key"},
        )

    assert "does not support unrecognized API paths" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tinfoil_forward_request_rejects_verified_status_without_local_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = TinfoilUpstreamProvider(api_key="test")
    provider.set_confidentiality_status(
        _verified_ehbp_status("tinfoil", model_ids=["tinfoil/secure-model"])
    )
    monkeypatch.setattr(
        "routstr.upstream.base.httpx.AsyncClient",
        FailIfNetworkAsyncClient,
    )

    with pytest.raises(UpstreamError) as exc_info:
        await provider.forward_request(
            request=_request(),
            path="v1/chat/completions",
            headers={"authorization": "Bearer test"},
            request_body=b'{"model":"tinfoil/secure-model","messages":[]}',
            key=_key(),
            max_cost_for_model=10_000,
            session=MagicMock(),
            model_obj=_model(),
        )

    assert "verified EHBP transport is required" in str(exc_info.value)


@pytest.mark.asyncio
async def test_privatemode_forward_request_requires_verified_proxy_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = PrivatemodeUpstreamProvider(api_key="test")
    monkeypatch.setattr(
        "routstr.upstream.base.httpx.AsyncClient",
        FailIfNetworkAsyncClient,
    )

    with pytest.raises(UpstreamError) as exc_info:
        await provider.forward_request(
            request=_request(),
            path="v1/chat/completions",
            headers={"authorization": "Bearer test"},
            request_body=b'{"model":"privatemode/gpt-secure","messages":[]}',
            key=_key(),
            max_cost_for_model=10_000,
            session=MagicMock(),
            model_obj=_privatemode_model(),
        )

    assert "verified confidential transport is required" in str(exc_info.value)


@pytest.mark.asyncio
async def test_privatemode_forward_request_requires_verification_for_requested_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = PrivatemodeUpstreamProvider(api_key="test")
    model_ids = ["privatemode/gpt-secure"]
    _bind_local_confidentiality_policy(
        provider,
        policy=_privatemode_policy_for_model_ids(model_ids),
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
            verified_claims=_privatemode_verified_claims(model_ids),
            model_ids=model_ids,
        )
    )
    monkeypatch.setattr(
        "routstr.upstream.base.httpx.AsyncClient",
        FailIfNetworkAsyncClient,
    )

    with pytest.raises(UpstreamError) as exc_info:
        await provider.forward_request(
            request=_request(),
            path="v1/chat/completions",
            headers={"authorization": "Bearer test"},
            request_body=b'{"model":"privatemode/other-model","messages":[]}',
            key=_key(),
            max_cost_for_model=10_000,
            session=MagicMock(),
            model_obj=_other_privatemode_model(),
        )

    assert "verified confidential transport does not cover requested model" in str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_privatemode_forward_get_request_requires_verified_proxy_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = PrivatemodeUpstreamProvider(api_key="test")
    request = _request()
    request.method = "GET"
    monkeypatch.setattr(
        "routstr.upstream.base.httpx.AsyncClient",
        FailIfNetworkAsyncClient,
    )

    with pytest.raises(UpstreamError) as exc_info:
        await provider.forward_get_request(
            request=request,
            path="tee/attestation",
            headers={"authorization": "Bearer test"},
        )

    assert "verified confidential transport is required" in str(exc_info.value)


@pytest.mark.asyncio
async def test_privatemode_forward_get_request_replaces_user_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = PrivatemodeUpstreamProvider(api_key="privatemode-proxy-key")
    model_ids = ["privatemode/gpt-secure"]
    _bind_local_confidentiality_policy(
        provider,
        policy=_privatemode_policy_for_model_ids(model_ids),
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
            verified_claims=_privatemode_verified_claims(model_ids),
            model_ids=model_ids,
        )
    )
    request = _request()
    request.method = "GET"
    DummyGetAsyncClient.request_headers = {}
    monkeypatch.setattr(
        "routstr.upstream.base.httpx.AsyncClient",
        DummyGetAsyncClient,
    )

    response = await provider.forward_get_request(
        request=request,
        path="tee/attestation",
        headers={"authorization": "Bearer routstr-user-key"},
    )

    assert response.status_code == 200
    assert DummyGetAsyncClient.request_headers["authorization"] == (
        "Bearer privatemode-proxy-key"
    )
    assert "routstr-user-key" not in str(DummyGetAsyncClient.request_headers)


def test_privatemode_current_transport_requires_provider_specific_proof() -> None:
    provider = PrivatemodeUpstreamProvider(api_key="test")
    provider._confidentiality_status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="privatemode",
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_claims={"transport": "privatemode-proxy"},
    )

    with pytest.raises(UpstreamError) as exc_info:
        provider._require_current_confidential_transport()

    assert "verified confidential transport evidence is incomplete" in str(
        exc_info.value
    )


def test_privatemode_current_transport_accepts_provider_specific_proof() -> None:
    provider = PrivatemodeUpstreamProvider(api_key="test")
    model_ids = ["privatemode/model"]
    _bind_local_confidentiality_policy(
        provider,
        policy=_privatemode_policy_for_model_ids(model_ids),
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
            verified_claims=_privatemode_verified_claims(model_ids),
            model_ids=model_ids,
        )
    )

    assert provider._require_current_confidential_transport().verified is True


def test_privatemode_current_transport_binds_verified_proxy_base_url() -> None:
    provider = PrivatemodeUpstreamProvider(
        api_key="test",
        base_url="http://127.0.0.1:8080/v1",
    )
    _bind_local_confidentiality_policy(
        provider,
        policy=_privatemode_policy_for_model_ids(["privatemode/model"]),
    )
    verified_claims = _privatemode_verified_claims()
    verified_claims["proxy_base_url"] = "http://127.0.0.1:9090/v1"
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
            verified_claims=verified_claims,
            model_ids=["privatemode/model"],
        )
    )

    with pytest.raises(UpstreamError) as exc_info:
        provider._require_current_confidential_transport()

    assert "verified confidential transport evidence is incomplete" in str(
        exc_info.value
    )


def test_privatemode_current_transport_requires_loopback_proxy_url() -> None:
    provider = PrivatemodeUpstreamProvider(
        api_key="test",
        base_url="https://api.privatemode.ai/v1",
    )
    _bind_local_confidentiality_policy(
        provider,
        policy=_privatemode_policy_for_model_ids(["privatemode/model"]),
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
            verified_claims=_privatemode_verified_claims(),
            model_ids=["privatemode/model"],
        )
    )

    with pytest.raises(UpstreamError) as exc_info:
        provider._require_current_confidential_transport()

    assert "Privatemode proxy base_url must use a loopback host" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tinfoil_forward_request_rejects_invalid_verified_ehbp_key_config() -> (
    None
):
    provider = TinfoilUpstreamProvider(api_key="test")
    _bind_local_confidentiality_policy(provider)
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
                "transport": "ehbp",
                "repo": "tinfoilsh/confidential-model-router",
                "ehbp_key_config_b64": base64.b64encode(b"bad").decode("ascii"),
                "attested_hpke_public_key_hex": "b" * 64,
                "enclave_measurement_fingerprint": VALID_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": VALID_CODE_MEASUREMENT,
                "payload_policy_digest": VALID_POLICY_DIGEST,
                "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
                "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
                "attestation_report_digest": VALID_ATTESTATION_REPORT_DIGEST,
                "release_digest": VALID_RELEASE_DIGEST,
                "tls_public_key_fingerprint_sha256": VALID_TLS_DIGEST,
                "verification_steps": _ehbp_verification_steps(),
                "model_attestations": _tinfoil_model_attestations_for_model_ids(
                    ["tinfoil/secure-model"],
                    _key_config(bytes.fromhex("bb" * 32)),
                ),
            },
            model_ids=["tinfoil/secure-model"],
        )
    )

    with pytest.raises(UpstreamError) as exc_info:
        await provider.forward_request(
            request=_request(),
            path="v1/chat/completions",
            headers={"authorization": "Bearer test"},
            request_body=b'{"model":"tinfoil/secure-model","messages":[]}',
            key=_key(),
            max_cost_for_model=10_000,
            session=MagicMock(),
            model_obj=_model(),
        )

    assert "verified EHBP key config is invalid" in str(exc_info.value)


def test_tinfoil_ehbp_key_config_requires_provider_specific_proof() -> None:
    private_key = x25519.X25519PrivateKey.generate()
    provider = TinfoilUpstreamProvider(api_key="test")
    provider._confidentiality_status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="tinfoil",
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_claims={
            "transport": "ehbp",
            "ehbp_key_config_b64": base64.b64encode(
                _key_config(_raw_public_key(private_key))
            ).decode("ascii"),
        },
    )

    with pytest.raises(UpstreamError) as exc_info:
        provider._verified_ehbp_key_config()

    assert "verified EHBP transport evidence is incomplete" in str(exc_info.value)


def test_tinfoil_ehbp_key_config_requires_attested_key_binding() -> None:
    attested_private_key = x25519.X25519PrivateKey.from_private_bytes(bytes(range(32)))
    routed_private_key = x25519.X25519PrivateKey.generate()
    key_config = _key_config(_raw_public_key(routed_private_key))
    claims = _tinfoil_verified_ehbp_claims(key_config, ["tinfoil/secure-model"])
    claims["attested_hpke_public_key_hex"] = _raw_public_key(attested_private_key).hex()

    provider = TinfoilUpstreamProvider(api_key="test")
    _bind_local_confidentiality_policy(provider)
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
            verified_claims=claims,
            model_ids=["tinfoil/secure-model"],
        )
    )

    with pytest.raises(UpstreamError) as exc_info:
        provider._verified_ehbp_key_config()

    assert "verified EHBP key config does not match attested key" in str(exc_info.value)


def test_tinfoil_ehbp_key_config_rejects_expired_verified_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = TinfoilUpstreamProvider(api_key="test")
    _bind_local_confidentiality_policy(provider)
    provider.set_confidentiality_status(
        _verified_ehbp_status("tinfoil", model_ids=["tinfoil/secure-model"])
    )
    monkeypatch.setattr("routstr.upstream.base.time.time", lambda: 4_102_444_801)

    with pytest.raises(UpstreamError) as exc_info:
        provider._verified_ehbp_key_config()

    assert "verified EHBP transport attestation has expired" in str(exc_info.value)


def test_tinfoil_ehbp_key_config_rejects_truthy_string_verified() -> None:
    private_key = x25519.X25519PrivateKey.generate()
    provider = TinfoilUpstreamProvider(api_key="test")
    provider._confidentiality_status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="tinfoil",
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_claims={
            "transport": "ehbp",
            "ehbp_key_config_b64": base64.b64encode(
                _key_config(_raw_public_key(private_key))
            ).decode("ascii"),
        },
    ).copy(update={"verified": "false"})

    with pytest.raises(UpstreamError) as exc_info:
        provider._verified_ehbp_key_config()

    assert "verified EHBP transport is required" in str(exc_info.value)


def test_tinfoil_ehbp_key_config_rejects_non_string_transport_claim() -> None:
    class CoercibleTransport:
        def __str__(self) -> str:
            return "ehbp"

    private_key = x25519.X25519PrivateKey.generate()
    provider = TinfoilUpstreamProvider(api_key="test")
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
                "transport": CoercibleTransport(),
                "ehbp_key_config_b64": base64.b64encode(
                    _key_config(_raw_public_key(private_key))
                ).decode("ascii"),
            },
        )
    )

    status = provider.confidentiality_status()
    assert status.verified is False
    assert status.failure_reason == (
        "confidentiality verifier did not return required evidence"
    )

    with pytest.raises(UpstreamError) as exc_info:
        provider._verified_ehbp_key_config()

    assert "verified EHBP transport is required" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tinfoil_messages_path_does_not_fall_back_to_plaintext_litellm() -> None:
    provider = TinfoilUpstreamProvider(api_key="test")
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
            verified_claims={"transport": "ehbp", "ehbp_key_config_b64": "AAAA"},
        )
    )

    with patch.object(
        provider,
        "_forward_messages_via_litellm",
        new=AsyncMock(side_effect=AssertionError("litellm would bypass EHBP")),
    ):
        with pytest.raises(UpstreamError) as exc_info:
            await provider.forward_request(
                request=_request(),
                path="v1/messages",
                headers={"authorization": "Bearer test"},
                request_body=b'{"model":"tinfoil/secure-model","messages":[]}',
                key=_key(),
                max_cost_for_model=10_000,
                session=MagicMock(),
                model_obj=_model(),
            )

    assert "does not support the Messages API" in str(exc_info.value)


@pytest.mark.asyncio
async def test_confidential_messages_translation_requires_selected_model_coverage() -> None:
    provider = PrivatemodeUpstreamProvider(api_key="test")
    provider.supports_anthropic_messages = False
    provider._require_supported_endpoint_for_path = MagicMock()
    _bind_local_confidentiality_policy(
        provider,
        policy=_privatemode_policy_for_model_ids(["privatemode/gpt-secure"]),
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
            verified_claims=_privatemode_verified_claims(["privatemode/gpt-secure"]),
            model_ids=["privatemode/gpt-secure"],
        )
    )

    with patch.object(
        provider,
        "_forward_messages_via_litellm",
        new=AsyncMock(side_effect=AssertionError("litellm bypassed model coverage")),
    ):
        with pytest.raises(UpstreamError) as exc_info:
            await provider.forward_request(
                request=_request(),
                path="v1/messages",
                headers={"authorization": "Bearer test"},
                request_body=b'{"model":"privatemode/other-model","messages":[]}',
                key=_key(),
                max_cost_for_model=10_000,
                session=MagicMock(),
                model_obj=_other_privatemode_model(),
            )

    assert "verified confidential transport does not cover requested model" in str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_confidential_x_cashu_messages_translation_requires_selected_model_coverage() -> None:
    provider = PrivatemodeUpstreamProvider(api_key="test")
    provider.supports_anthropic_messages = False
    provider._require_supported_endpoint_for_path = MagicMock()
    _bind_local_confidentiality_policy(
        provider,
        policy=_privatemode_policy_for_model_ids(["privatemode/gpt-secure"]),
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
            verified_claims=_privatemode_verified_claims(["privatemode/gpt-secure"]),
            model_ids=["privatemode/gpt-secure"],
        )
    )
    request = _cashu_request(b'{"model":"privatemode/other-model","messages":[]}')

    with patch.object(
        provider,
        "_forward_x_cashu_messages_via_litellm",
        new=AsyncMock(side_effect=AssertionError("litellm bypassed model coverage")),
    ):
        with pytest.raises(UpstreamError) as exc_info:
            await provider.forward_x_cashu_request(
                request=request,
                path="v1/messages",
                headers={"authorization": "Bearer test"},
                amount=5_000,
                unit="sat",
                max_cost_for_model=10_000,
                model_obj=_other_privatemode_model(),
            )

    assert "verified confidential transport does not cover requested model" in str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_confidential_x_cashu_count_tokens_requires_selected_model_coverage() -> None:
    provider = PrivatemodeUpstreamProvider(api_key="test")
    provider.supports_anthropic_messages = False
    provider._require_supported_endpoint_for_path = MagicMock()
    _bind_local_confidentiality_policy(
        provider,
        policy=_privatemode_policy_for_model_ids(["privatemode/gpt-secure"]),
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
            verified_claims=_privatemode_verified_claims(["privatemode/gpt-secure"]),
            model_ids=["privatemode/gpt-secure"],
        )
    )
    request = _cashu_request(
        b'{"model":"privatemode/other-model","messages":[{"role":"user","content":"secret"}]}'
    )

    with pytest.raises(UpstreamError) as exc_info:
        await provider.forward_x_cashu_request(
            request=request,
            path="v1/messages/count_tokens",
            headers={"authorization": "Bearer test"},
            amount=5_000,
            unit="sat",
            max_cost_for_model=10_000,
            model_obj=_other_privatemode_model(),
        )

    assert "verified confidential transport does not cover requested model" in str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_tinfoil_unrecognized_model_path_rejected_before_ehbp_transport() -> None:
    provider = TinfoilUpstreamProvider(api_key="test")

    with patch.object(
        provider,
        "_prepare_ehbp_request_body",
        side_effect=AssertionError("unknown paths must not reach EHBP forwarding"),
    ):
        with pytest.raises(UpstreamError) as exc_info:
            await provider.forward_request(
                request=_request(),
                path="v1/chat/completions/side-channel",
                headers={"authorization": "Bearer test"},
                request_body=b'{"model":"tinfoil/secure-model","messages":[]}',
                key=_key(),
                max_cost_for_model=10_000,
                session=MagicMock(),
                model_obj=_model(),
            )

    assert "does not support unrecognized API paths" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tinfoil_forward_request_encrypts_body_and_decrypts_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipient_private = x25519.X25519PrivateKey.from_private_bytes(bytes(range(32)))
    key_config = _key_config(_raw_public_key(recipient_private))
    DummyEhbpAsyncClient.request_body = None
    DummyEhbpAsyncClient.request_headers = {}
    DummyEhbpAsyncClient.recipient_private_key = recipient_private
    DummyEhbpAsyncClient.response_payload = {
        "id": "chatcmpl-secure",
        "model": "secure-model",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    monkeypatch.setattr("routstr.upstream.base.httpx.AsyncClient", DummyEhbpAsyncClient)

    provider = TinfoilUpstreamProvider(api_key="test")
    _bind_local_confidentiality_policy(provider)
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
            verified_claims=_tinfoil_verified_ehbp_claims(
                key_config,
                ["tinfoil/secure-model"],
            ),
            model_ids=["tinfoil/secure-model"],
        )
    )
    session = MagicMock()
    session.refresh = AsyncMock()

    with patch(
        "routstr.upstream.base.adjust_payment_for_tokens",
        new=AsyncMock(return_value={"total_msats": 1, "total_usd": 0.0}),
    ):
        response = await provider.forward_request(
            request=_request(),
            path="v1/chat/completions",
            headers={"authorization": "Bearer test"},
            request_body=(
                b'{"model":"tinfoil/secure-model","messages":[{"role":"user",'
                b'"content":"SECRET_PROMPT"}],"stream":false}'
            ),
            key=_key(),
            max_cost_for_model=10_000,
            session=session,
            model_obj=_model(),
        )

    assert response.status_code == 200
    assert DummyEhbpAsyncClient.request_body is not None
    assert b"SECRET_PROMPT" not in DummyEhbpAsyncClient.request_body
    assert "ehbp-encapsulated-key" in DummyEhbpAsyncClient.request_headers
    assert DummyEhbpAsyncClient.request_headers["transfer-encoding"] == "chunked"
    assert (
        DummyEhbpAsyncClient.request_headers["x-tinfoil-request-usage-metrics"]
        == "true"
    )
    assert "content-length" not in DummyEhbpAsyncClient.request_headers

    body = json.loads(bytes(response.body))
    assert body["choices"][0]["message"]["content"] == "ok"
    assert body["model"] == "tinfoil/secure-model"


@pytest.mark.asyncio
async def test_tinfoil_forward_request_uses_public_route_alias_for_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipient_private = x25519.X25519PrivateKey.from_private_bytes(bytes(range(32)))
    key_config = _key_config(_raw_public_key(recipient_private))
    DummyEhbpAsyncClient.request_body = None
    DummyEhbpAsyncClient.request_headers = {}
    DummyEhbpAsyncClient.recipient_private_key = recipient_private
    DummyEhbpAsyncClient.response_payload = {
        "id": "chatcmpl-secure",
        "model": "secure-model",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    monkeypatch.setattr("routstr.upstream.base.httpx.AsyncClient", DummyEhbpAsyncClient)

    provider = TinfoilUpstreamProvider(api_key="test")
    _bind_local_confidentiality_policy(provider)
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
            verified_claims=_tinfoil_verified_ehbp_claims(
                key_config,
                ["tinfoil/secure-model"],
            ),
            model_ids=["tinfoil/secure-model"],
        )
    )
    model = _model()
    model.id = "secure-model"
    model.forwarded_model_id = None
    session = MagicMock()
    session.refresh = AsyncMock()

    with patch(
        "routstr.upstream.base.adjust_payment_for_tokens",
        new=AsyncMock(return_value={"total_msats": 1, "total_usd": 0.0}),
    ):
        response = await provider.forward_request(
            request=_request(),
            path="v1/chat/completions",
            headers={"authorization": "Bearer test"},
            request_body=(
                b'{"model":"tinfoil/secure-model","messages":[{"role":"user",'
                b'"content":"SECRET_PROMPT"}],"stream":false}'
            ),
            key=_key(),
            max_cost_for_model=10_000,
            session=session,
            model_obj=model,
            route_alias="tinfoil/secure-model",
        )

    assert response.status_code == 200
    assert DummyEhbpAsyncClient.request_body is not None


@pytest.mark.asyncio
async def test_tinfoil_forward_request_redacts_confidential_exception_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.upstream.base.httpx.AsyncClient",
        SecretExceptionEhbpAsyncClient,
    )
    provider = TinfoilUpstreamProvider(api_key="test")
    _bind_local_confidentiality_policy(provider)
    provider.set_confidentiality_status(
        _verified_ehbp_status("tinfoil", model_ids=["tinfoil/secure-model"])
    )
    session = MagicMock()
    session.refresh = AsyncMock()

    with patch("routstr.upstream.base.logger.error") as log_error:
        with pytest.raises(UpstreamError) as exc_info:
            await provider.forward_request(
                request=_request(),
                path="v1/chat/completions",
                headers={"authorization": "Bearer test"},
                request_body=(
                    b'{"model":"tinfoil/secure-model","messages":[{"role":"user",'
                    b'"content":"SECRET_PROMPT"}],"stream":false}'
                ),
                key=_key(),
                max_cost_for_model=10_000,
                session=session,
                model_obj=_model(),
            )

    assert str(exc_info.value) == "An unexpected server error occurred"
    logged = str(log_error.call_args)
    assert "SECRET_PROMPT" not in logged
    assert "<redacted: confidential route>" in logged
    _, kwargs = log_error.call_args
    assert kwargs["extra"]["confidential"] is True
    assert kwargs["extra"]["error"] == "<redacted: confidential route>"
    assert kwargs["extra"]["traceback"] == "<redacted: confidential route>"


@pytest.mark.asyncio
async def test_tinfoil_audio_multipart_request_uses_verified_ehbp_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipient_private = x25519.X25519PrivateKey.from_private_bytes(bytes(range(32)))
    key_config = _key_config(_raw_public_key(recipient_private))
    DummyAudioEhbpAsyncClient.request_body = None
    DummyAudioEhbpAsyncClient.request_headers = {}
    DummyAudioEhbpAsyncClient.decrypted_plaintext = None
    DummyAudioEhbpAsyncClient.recipient_private_key = recipient_private
    monkeypatch.setattr(
        "routstr.upstream.base.httpx.AsyncClient", DummyAudioEhbpAsyncClient
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    _bind_local_confidentiality_policy(
        provider,
        policy=_tinfoil_policy_for_model_ids(["tinfoil/whisper-large-v3-turbo"]),
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
            verified_claims=_tinfoil_verified_ehbp_claims(
                key_config,
                ["tinfoil/whisper-large-v3-turbo"],
            ),
            model_ids=["tinfoil/whisper-large-v3-turbo"],
        )
    )
    boundary = "----routstr-test-audio"
    request_body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="model"\r\n\r\n'
        "tinfoil/whisper-large-v3-turbo\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="prompt"\r\n\r\n'
        "SECRET_STYLE_HINT\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="sample.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
        "SECRET_AUDIO_BYTES\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    response = await provider.forward_request(
        request=_request(),
        path="v1/audio/transcriptions",
        headers={
            "authorization": "Bearer test",
            "content-type": f"multipart/form-data; boundary={boundary}",
            "content-length": str(len(request_body)),
        },
        request_body=request_body,
        key=_key(),
        max_cost_for_model=10_000,
        session=MagicMock(),
        model_obj=_audio_model(),
    )

    assert response.status_code == 200
    assert DummyAudioEhbpAsyncClient.request_body is not None
    assert b"SECRET_AUDIO_BYTES" not in DummyAudioEhbpAsyncClient.request_body
    assert b"SECRET_STYLE_HINT" not in DummyAudioEhbpAsyncClient.request_body
    assert "ehbp-encapsulated-key" in DummyAudioEhbpAsyncClient.request_headers
    assert DummyAudioEhbpAsyncClient.request_headers["transfer-encoding"] == "chunked"
    assert "content-length" not in DummyAudioEhbpAsyncClient.request_headers
    assert DummyAudioEhbpAsyncClient.decrypted_plaintext is not None
    assert (
        b'name="model"\r\n\r\nwhisper-large-v3-turbo'
        in DummyAudioEhbpAsyncClient.decrypted_plaintext
    )
    assert (
        b"tinfoil/whisper-large-v3-turbo"
        not in DummyAudioEhbpAsyncClient.decrypted_plaintext
    )
    assert b"SECRET_AUDIO_BYTES" in DummyAudioEhbpAsyncClient.decrypted_plaintext

    response_body = b"".join([chunk async for chunk in response.body_iterator])
    assert response_body == b'{"text":"ok","usage":{"type":"duration","seconds":"1"}}'


@pytest.mark.asyncio
async def test_tinfoil_audio_multipart_alias_is_rewritten_when_model_id_is_unprefixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipient_private = x25519.X25519PrivateKey.from_private_bytes(bytes(range(32)))
    key_config = _key_config(_raw_public_key(recipient_private))
    DummyAudioEhbpAsyncClient.request_body = None
    DummyAudioEhbpAsyncClient.request_headers = {}
    DummyAudioEhbpAsyncClient.decrypted_plaintext = None
    DummyAudioEhbpAsyncClient.recipient_private_key = recipient_private
    monkeypatch.setattr(
        "routstr.upstream.base.httpx.AsyncClient", DummyAudioEhbpAsyncClient
    )

    provider = TinfoilUpstreamProvider(api_key="test")
    _bind_local_confidentiality_policy(
        provider,
        policy=_tinfoil_policy_for_model_ids(["whisper-large-v3-turbo"]),
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
            verified_claims=_tinfoil_verified_ehbp_claims(
                key_config,
                ["whisper-large-v3-turbo"],
            ),
            model_ids=["whisper-large-v3-turbo"],
        )
    )
    boundary = "----routstr-test-audio-alias"
    request_body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="model"\r\n\r\n'
        "tinfoil/whisper-large-v3-turbo\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="sample.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
        "SECRET_AUDIO_BYTES\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    response = await provider.forward_request(
        request=_request(),
        path="v1/audio/transcriptions",
        headers={
            "authorization": "Bearer test",
            "content-type": f"multipart/form-data; boundary={boundary}",
            "content-length": str(len(request_body)),
        },
        request_body=request_body,
        key=_key(),
        max_cost_for_model=10_000,
        session=MagicMock(),
        model_obj=_base_audio_model(),
    )

    assert response.status_code == 200
    assert DummyAudioEhbpAsyncClient.decrypted_plaintext is not None
    assert (
        b'name="model"\r\n\r\nwhisper-large-v3-turbo'
        in DummyAudioEhbpAsyncClient.decrypted_plaintext
    )
    assert (
        b"tinfoil/whisper-large-v3-turbo"
        not in DummyAudioEhbpAsyncClient.decrypted_plaintext
    )


@pytest.mark.asyncio
async def test_tinfoil_responses_request_uses_verified_ehbp_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipient_private = x25519.X25519PrivateKey.from_private_bytes(bytes(range(32)))
    key_config = _key_config(_raw_public_key(recipient_private))
    DummyEhbpAsyncClient.request_body = None
    DummyEhbpAsyncClient.request_headers = {}
    DummyEhbpAsyncClient.recipient_private_key = recipient_private
    DummyEhbpAsyncClient.response_payload = {
        "id": "resp-secure",
        "object": "response",
        "model": "secure-model",
        "output": [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    monkeypatch.setattr("routstr.upstream.base.httpx.AsyncClient", DummyEhbpAsyncClient)

    provider = TinfoilUpstreamProvider(api_key="test")
    _bind_local_confidentiality_policy(provider)
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
            verified_claims=_tinfoil_verified_ehbp_claims(
                key_config,
                ["tinfoil/secure-model"],
            ),
            model_ids=["tinfoil/secure-model"],
        )
    )
    session = MagicMock()
    session.refresh = AsyncMock()

    with patch(
        "routstr.upstream.base.adjust_payment_for_tokens",
        new=AsyncMock(return_value={"total_msats": 1, "total_usd": 0.0}),
    ):
        response = await provider.forward_responses_request(
            request=_request(),
            path="v1/responses",
            headers={"authorization": "Bearer test"},
            request_body=(
                b'{"model":"tinfoil/secure-model",'
                b'"input":"SECRET_PROMPT","stream":false}'
            ),
            key=_key(),
            max_cost_for_model=10_000,
            session=session,
            model_obj=_model(),
        )

    assert response.status_code == 200
    assert DummyEhbpAsyncClient.request_body is not None
    assert b"SECRET_PROMPT" not in DummyEhbpAsyncClient.request_body
    assert "ehbp-encapsulated-key" in DummyEhbpAsyncClient.request_headers
    assert DummyEhbpAsyncClient.request_headers["transfer-encoding"] == "chunked"
    assert (
        DummyEhbpAsyncClient.request_headers["x-tinfoil-request-usage-metrics"]
        == "true"
    )
    assert "content-length" not in DummyEhbpAsyncClient.request_headers

    body = json.loads(bytes(response.body))
    assert body["model"] == "tinfoil/secure-model"


@pytest.mark.asyncio
async def test_tinfoil_x_cashu_chat_request_uses_verified_ehbp_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipient_private = x25519.X25519PrivateKey.from_private_bytes(bytes(range(32)))
    key_config = _key_config(_raw_public_key(recipient_private))
    DummyEhbpAsyncClient.request_body = None
    DummyEhbpAsyncClient.request_headers = {}
    DummyEhbpAsyncClient.recipient_private_key = recipient_private
    DummyEhbpAsyncClient.response_payload = {
        "id": "chatcmpl-secure",
        "model": "secure-model",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    monkeypatch.setattr("routstr.upstream.base.httpx.AsyncClient", DummyEhbpAsyncClient)

    provider = TinfoilUpstreamProvider(api_key="test")
    _bind_local_confidentiality_policy(provider)
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
            verified_claims=_tinfoil_verified_ehbp_claims(
                key_config,
                ["tinfoil/secure-model"],
            ),
            model_ids=["tinfoil/secure-model"],
        )
    )

    with (
        patch.object(
            provider,
            "get_x_cashu_cost",
            new=AsyncMock(return_value=_cost_data(10_000)),
        ),
        patch.object(
            provider,
            "send_refund",
            new=AsyncMock(side_effect=AssertionError("refund should not be needed")),
        ),
    ):
        response = await provider.forward_x_cashu_request(
            request=_cashu_request(
                b'{"model":"tinfoil/secure-model","messages":[{"role":"user",'
                b'"content":"SECRET_PROMPT"}],"stream":false}'
            ),
            path="v1/chat/completions",
            headers={"authorization": "Bearer test"},
            amount=10_000,
            unit="msat",
            max_cost_for_model=10_000,
            model_obj=_model(),
            mint=None,
        )

    assert response.status_code == 200
    assert DummyEhbpAsyncClient.request_body is not None
    assert b"SECRET_PROMPT" not in DummyEhbpAsyncClient.request_body
    assert "ehbp-encapsulated-key" in DummyEhbpAsyncClient.request_headers
    assert DummyEhbpAsyncClient.request_headers["transfer-encoding"] == "chunked"
    assert "content-length" not in DummyEhbpAsyncClient.request_headers

    body = json.loads(bytes(response.body))
    assert body["choices"][0]["message"]["content"] == "ok"
    assert body["model"] == "tinfoil/secure-model"


@pytest.mark.asyncio
async def test_tinfoil_x_cashu_responses_request_uses_verified_ehbp_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipient_private = x25519.X25519PrivateKey.from_private_bytes(bytes(range(32)))
    key_config = _key_config(_raw_public_key(recipient_private))
    DummyEhbpAsyncClient.request_body = None
    DummyEhbpAsyncClient.request_headers = {}
    DummyEhbpAsyncClient.recipient_private_key = recipient_private
    DummyEhbpAsyncClient.response_payload = {
        "id": "resp-secure",
        "object": "response",
        "model": "secure-model",
        "output": [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    monkeypatch.setattr("routstr.upstream.base.httpx.AsyncClient", DummyEhbpAsyncClient)

    provider = TinfoilUpstreamProvider(api_key="test")
    _bind_local_confidentiality_policy(provider)
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
            verified_claims=_tinfoil_verified_ehbp_claims(
                key_config,
                ["tinfoil/secure-model"],
            ),
            model_ids=["tinfoil/secure-model"],
        )
    )

    with (
        patch.object(
            provider,
            "get_x_cashu_cost",
            new=AsyncMock(return_value=_cost_data(10_000)),
        ),
        patch.object(
            provider,
            "send_refund",
            new=AsyncMock(side_effect=AssertionError("refund should not be needed")),
        ),
    ):
        response = await provider.forward_x_cashu_responses_request(
            request=_cashu_request(
                b'{"model":"tinfoil/secure-model",'
                b'"input":"SECRET_PROMPT","stream":false}'
            ),
            path="v1/responses",
            headers={"authorization": "Bearer test"},
            amount=10_000,
            unit="msat",
            max_cost_for_model=10_000,
            model_obj=_model(),
            mint=None,
        )

    assert response.status_code == 200
    assert DummyEhbpAsyncClient.request_body is not None
    assert b"SECRET_PROMPT" not in DummyEhbpAsyncClient.request_body
    assert "ehbp-encapsulated-key" in DummyEhbpAsyncClient.request_headers
    assert DummyEhbpAsyncClient.request_headers["transfer-encoding"] == "chunked"
    assert "content-length" not in DummyEhbpAsyncClient.request_headers

    body = json.loads(bytes(response.body))
    assert body["model"] == "tinfoil/secure-model"


@pytest.mark.asyncio
async def test_ppq_private_forward_request_encrypts_body_and_strips_private_model_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipient_private = x25519.X25519PrivateKey.from_private_bytes(bytes(range(32)))
    key_config = _key_config(_raw_public_key(recipient_private))
    DummyPPQEhbpAsyncClient.request_body = None
    DummyPPQEhbpAsyncClient.request_headers = {}
    DummyPPQEhbpAsyncClient.decrypted_plaintext = None
    DummyPPQEhbpAsyncClient.recipient_private_key = recipient_private
    DummyPPQEhbpAsyncClient.response_payload = {
        "id": "chatcmpl-secure",
        "model": "secure-model",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    monkeypatch.setattr(
        "routstr.upstream.base.httpx.AsyncClient",
        DummyPPQEhbpAsyncClient,
    )

    provider = PPQPrivateUpstreamProvider(api_key="test")
    _bind_local_confidentiality_policy(provider)
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="ppq-private-tee",
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            verifier="unit-test-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_claims=_verified_ehbp_claims(
                "ppq-private-tee",
                key_config,
                ["private/gpt-oss-120b"],
            ),
            model_ids=["private/gpt-oss-120b"],
        )
    )
    session = MagicMock()
    session.refresh = AsyncMock()

    with patch(
        "routstr.upstream.base.adjust_payment_for_tokens",
        new=AsyncMock(return_value={"total_msats": 1, "total_usd": 0.0}),
    ):
        response = await provider.forward_request(
            request=_request(),
            path="v1/chat/completions",
            headers={"authorization": "Bearer test"},
            request_body=(
                b'{"model":"private/gpt-oss-120b","messages":[{"role":"user",'
                b'"content":"SECRET_PROMPT"}],"stream":false}'
            ),
            key=_key(),
            max_cost_for_model=10_000,
            session=session,
            model_obj=_ppq_private_model(),
        )

    assert response.status_code == 200
    assert DummyPPQEhbpAsyncClient.request_body is not None
    assert b"SECRET_PROMPT" not in DummyPPQEhbpAsyncClient.request_body
    assert (
        DummyPPQEhbpAsyncClient.request_headers["x-private-model"]
        == "private/gpt-oss-120b"
    )
    assert DummyPPQEhbpAsyncClient.request_headers["x-query-source"] == "api"
    assert DummyPPQEhbpAsyncClient.decrypted_plaintext is not None
    assert b'"model": "gpt-oss-120b"' in DummyPPQEhbpAsyncClient.decrypted_plaintext
    assert b"private/gpt-oss-120b" not in DummyPPQEhbpAsyncClient.decrypted_plaintext


@pytest.mark.asyncio
async def test_ppq_private_forward_request_replaces_user_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipient_private = x25519.X25519PrivateKey.from_private_bytes(bytes(range(32)))
    key_config = _key_config(_raw_public_key(recipient_private))
    DummyPPQEhbpAsyncClient.request_body = None
    DummyPPQEhbpAsyncClient.request_headers = {}
    DummyPPQEhbpAsyncClient.decrypted_plaintext = None
    DummyPPQEhbpAsyncClient.recipient_private_key = recipient_private
    DummyPPQEhbpAsyncClient.response_payload = {
        "id": "chatcmpl-secure",
        "model": "secure-model",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    monkeypatch.setattr(
        "routstr.upstream.base.httpx.AsyncClient",
        DummyPPQEhbpAsyncClient,
    )

    provider = PPQPrivateUpstreamProvider(api_key="ppq-upstream-key")
    _bind_local_confidentiality_policy(provider)
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="ppq-private-tee",
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            verifier="unit-test-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_claims=_verified_ehbp_claims(
                "ppq-private-tee",
                key_config,
                ["private/gpt-oss-120b"],
            ),
            model_ids=["private/gpt-oss-120b"],
        )
    )
    session = MagicMock()
    session.refresh = AsyncMock()

    with patch(
        "routstr.upstream.base.adjust_payment_for_tokens",
        new=AsyncMock(return_value={"total_msats": 1, "total_usd": 0.0}),
    ):
        response = await provider.forward_request(
            request=_request(),
            path="v1/chat/completions",
            headers={"authorization": "Bearer routstr-user-key"},
            request_body=(
                b'{"model":"private/gpt-oss-120b","messages":[{"role":"user",'
                b'"content":"SECRET_PROMPT"}],"stream":false}'
            ),
            key=_key(),
            max_cost_for_model=10_000,
            session=session,
            model_obj=_ppq_private_model(),
        )

    assert response.status_code == 200
    assert DummyPPQEhbpAsyncClient.request_headers["authorization"] == (
        "Bearer ppq-upstream-key"
    )
    assert "routstr-user-key" not in str(DummyPPQEhbpAsyncClient.request_headers)


@pytest.mark.asyncio
async def test_tinfoil_forward_request_replaces_user_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipient_private = x25519.X25519PrivateKey.from_private_bytes(bytes(range(32)))
    key_config = _key_config(_raw_public_key(recipient_private))
    DummyEhbpAsyncClient.request_body = None
    DummyEhbpAsyncClient.request_headers = {}
    DummyEhbpAsyncClient.recipient_private_key = recipient_private
    DummyEhbpAsyncClient.response_payload = {
        "id": "chatcmpl-secure",
        "model": "secure-model",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    monkeypatch.setattr("routstr.upstream.base.httpx.AsyncClient", DummyEhbpAsyncClient)

    provider = TinfoilUpstreamProvider(api_key="tinfoil-upstream-key")
    _bind_local_confidentiality_policy(provider)
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
            verified_claims=_tinfoil_verified_ehbp_claims(
                key_config,
                ["tinfoil/secure-model"],
            ),
            model_ids=["tinfoil/secure-model"],
        )
    )
    session = MagicMock()
    session.refresh = AsyncMock()

    with patch(
        "routstr.upstream.base.adjust_payment_for_tokens",
        new=AsyncMock(return_value={"total_msats": 1, "total_usd": 0.0}),
    ):
        response = await provider.forward_request(
            request=_request(),
            path="v1/chat/completions",
            headers={"authorization": "Bearer routstr-user-key"},
            request_body=(
                b'{"model":"tinfoil/secure-model","messages":[{"role":"user",'
                b'"content":"SECRET_PROMPT"}],"stream":false}'
            ),
            key=_key(),
            max_cost_for_model=10_000,
            session=session,
            model_obj=_model(),
        )

    assert response.status_code == 200
    assert DummyEhbpAsyncClient.request_headers["authorization"] == (
        "Bearer tinfoil-upstream-key"
    )
    assert "routstr-user-key" not in str(DummyEhbpAsyncClient.request_headers)


@pytest.mark.asyncio
async def test_privatemode_forward_request_replaces_user_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPlainAsyncClient.request_headers = {}
    DummyPlainAsyncClient.request_body = None
    monkeypatch.setattr("routstr.upstream.base.httpx.AsyncClient", DummyPlainAsyncClient)

    provider = PrivatemodeUpstreamProvider(api_key="privatemode-proxy-key")
    model_ids = ["privatemode/gpt-secure"]
    _bind_local_confidentiality_policy(
        provider,
        policy=_privatemode_policy_for_model_ids(model_ids),
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
            verified_claims=_privatemode_verified_claims(model_ids),
            model_ids=model_ids,
        )
    )
    session = MagicMock()
    session.refresh = AsyncMock()

    with patch(
        "routstr.upstream.base.adjust_payment_for_tokens",
        new=AsyncMock(return_value={"total_msats": 1, "total_usd": 0.0}),
    ):
        response = await provider.forward_request(
            request=_request(),
            path="v1/chat/completions",
            headers={"authorization": "Bearer routstr-user-key"},
            request_body=(
                b'{"model":"privatemode/gpt-secure","messages":[{"role":"user",'
                b'"content":"SECRET_PROMPT"}],"stream":false}'
            ),
            key=_key(),
            max_cost_for_model=10_000,
            session=session,
            model_obj=_privatemode_model(),
        )

    assert response.status_code == 200
    assert DummyPlainAsyncClient.request_headers["authorization"] == (
        "Bearer privatemode-proxy-key"
    )
    assert "routstr-user-key" not in str(DummyPlainAsyncClient.request_headers)


@pytest.mark.asyncio
async def test_ppq_private_forward_request_uses_attested_forwarded_model_for_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipient_private = x25519.X25519PrivateKey.from_private_bytes(bytes(range(32)))
    key_config = _key_config(_raw_public_key(recipient_private))
    DummyPPQEhbpAsyncClient.request_body = None
    DummyPPQEhbpAsyncClient.request_headers = {}
    DummyPPQEhbpAsyncClient.decrypted_plaintext = None
    DummyPPQEhbpAsyncClient.recipient_private_key = recipient_private
    DummyPPQEhbpAsyncClient.response_payload = {
        "id": "chatcmpl-secure",
        "model": "secure-model",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    monkeypatch.setattr(
        "routstr.upstream.base.httpx.AsyncClient",
        DummyPPQEhbpAsyncClient,
    )

    provider = PPQPrivateUpstreamProvider(api_key="test")
    _bind_local_confidentiality_policy(provider)
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="ppq-private-tee",
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            verifier="unit-test-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_claims=_verified_ehbp_claims(
                "ppq-private-tee",
                key_config,
                ["private/gpt-oss-120b"],
            ),
            model_ids=["private/gpt-oss-120b"],
        )
    )
    session = MagicMock()
    session.refresh = AsyncMock()

    with patch(
        "routstr.upstream.base.adjust_payment_for_tokens",
        new=AsyncMock(return_value={"total_msats": 1, "total_usd": 0.0}),
    ):
        response = await provider.forward_request(
            request=_request(),
            path="v1/chat/completions",
            headers={"authorization": "Bearer test"},
            request_body=(
                b'{"model":"ppq-secure-gpt","messages":[{"role":"user",'
                b'"content":"SECRET_PROMPT"}],"stream":false}'
            ),
            key=_key(),
            max_cost_for_model=10_000,
            session=session,
            model_obj=_aliased_ppq_private_model(),
        )

    assert response.status_code == 200
    assert (
        DummyPPQEhbpAsyncClient.request_headers["x-private-model"]
        == "private/gpt-oss-120b"
    )
    assert DummyPPQEhbpAsyncClient.decrypted_plaintext is not None
    assert b'"model": "gpt-oss-120b"' in DummyPPQEhbpAsyncClient.decrypted_plaintext
    assert b"ppq-secure-gpt" not in DummyPPQEhbpAsyncClient.decrypted_plaintext


@pytest.mark.asyncio
async def test_ppq_private_forward_request_rejects_unsupported_embeddings_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = PPQPrivateUpstreamProvider(api_key="test")
    provider.set_confidentiality_status(
        _verified_ehbp_status("ppq-private-tee", model_ids=["private/gpt-oss-120b"])
    )
    monkeypatch.setattr(
        "routstr.upstream.base.httpx.AsyncClient",
        FailIfNetworkAsyncClient,
    )

    with pytest.raises(UpstreamError) as exc_info:
        await provider.forward_request(
            request=_request(),
            path="v1/embeddings",
            headers={"authorization": "Bearer test"},
            request_body=b'{"model":"private/gpt-oss-120b","input":"SECRET_PROMPT"}',
            key=_key(),
            max_cost_for_model=10_000,
            session=MagicMock(),
            model_obj=_ppq_private_model(),
        )

    assert "does not support the Embeddings API" in str(exc_info.value)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_ppq_private_forward_responses_rejects_unsupported_endpoint_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = PPQPrivateUpstreamProvider(api_key="test")
    provider.set_confidentiality_status(
        _verified_ehbp_status("ppq-private-tee", model_ids=["private/gpt-oss-120b"])
    )
    monkeypatch.setattr(
        "routstr.upstream.base.httpx.AsyncClient",
        FailIfNetworkAsyncClient,
    )

    with pytest.raises(UpstreamError) as exc_info:
        await provider.forward_responses_request(
            request=_request(),
            path="v1/responses",
            headers={"authorization": "Bearer test"},
            request_body=(
                b'{"model":"private/gpt-oss-120b",'
                b'"input":"SECRET_PROMPT","stream":false}'
            ),
            key=_key(),
            max_cost_for_model=10_000,
            session=MagicMock(),
            model_obj=_ppq_private_model(),
        )

    assert "does not support the Responses API" in str(exc_info.value)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_ppq_private_x_cashu_request_rejects_unsupported_embeddings_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = PPQPrivateUpstreamProvider(api_key="test")
    provider.set_confidentiality_status(
        _verified_ehbp_status("ppq-private-tee", model_ids=["private/gpt-oss-120b"])
    )
    monkeypatch.setattr(
        "routstr.upstream.base.httpx.AsyncClient",
        FailIfNetworkAsyncClient,
    )

    with pytest.raises(UpstreamError) as exc_info:
        await provider.forward_x_cashu_request(
            request=_cashu_request(
                b'{"model":"private/gpt-oss-120b","input":"SECRET_PROMPT"}'
            ),
            path="v1/embeddings",
            headers={"authorization": "Bearer test"},
            amount=10_000,
            unit="msat",
            max_cost_for_model=10_000,
            model_obj=_ppq_private_model(),
            mint=None,
        )

    assert "does not support the Embeddings API" in str(exc_info.value)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_ppq_private_x_cashu_responses_rejects_unsupported_endpoint_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = PPQPrivateUpstreamProvider(api_key="test")
    provider.set_confidentiality_status(
        _verified_ehbp_status("ppq-private-tee", model_ids=["private/gpt-oss-120b"])
    )
    monkeypatch.setattr(
        "routstr.upstream.base.httpx.AsyncClient",
        FailIfNetworkAsyncClient,
    )

    with pytest.raises(UpstreamError) as exc_info:
        await provider.forward_x_cashu_responses_request(
            request=_cashu_request(
                b'{"model":"private/gpt-oss-120b",'
                b'"input":"SECRET_PROMPT","stream":false}'
            ),
            path="v1/responses",
            headers={"authorization": "Bearer test"},
            amount=10_000,
            unit="msat",
            max_cost_for_model=10_000,
            model_obj=_ppq_private_model(),
            mint=None,
        )

    assert "does not support the Responses API" in str(exc_info.value)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_ppq_private_handle_x_cashu_rejects_unsupported_endpoint_before_token_redemption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = PPQPrivateUpstreamProvider(api_key="test")
    provider.set_confidentiality_status(
        _verified_ehbp_status("ppq-private-tee", model_ids=["private/gpt-oss-120b"])
    )
    mock_receive_token = AsyncMock(return_value=(10_000, "msat", None))
    monkeypatch.setattr("routstr.upstream.base.recieve_token", mock_receive_token)

    with pytest.raises(UpstreamError) as exc_info:
        await provider.handle_x_cashu(
            request=_cashu_request(
                b'{"model":"private/gpt-oss-120b","input":"SECRET_PROMPT"}'
            ),
            x_cashu_token="cashuAtest",
            path="v1/embeddings",
            max_cost_for_model=10_000,
            model_obj=_ppq_private_model(),
        )

    assert "does not support the Embeddings API" in str(exc_info.value)
    assert exc_info.value.status_code == 400
    mock_receive_token.assert_not_awaited()


@pytest.mark.asyncio
async def test_ppq_private_handle_x_cashu_responses_rejects_unsupported_endpoint_before_token_redemption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = PPQPrivateUpstreamProvider(api_key="test")
    provider.set_confidentiality_status(
        _verified_ehbp_status("ppq-private-tee", model_ids=["private/gpt-oss-120b"])
    )
    mock_receive_token = AsyncMock(return_value=(10_000, "msat", None))
    monkeypatch.setattr("routstr.upstream.base.recieve_token", mock_receive_token)

    with pytest.raises(UpstreamError) as exc_info:
        await provider.handle_x_cashu_responses(
            request=_cashu_request(
                b'{"model":"private/gpt-oss-120b",'
                b'"input":"SECRET_PROMPT","stream":false}'
            ),
            x_cashu_token="cashuAtest",
            path="v1/responses",
            max_cost_for_model=10_000,
            model_obj=_ppq_private_model(),
        )

    assert "does not support the Responses API" in str(exc_info.value)
    assert exc_info.value.status_code == 400
    mock_receive_token.assert_not_awaited()


@pytest.mark.asyncio
async def test_tinfoil_handle_x_cashu_requires_ehbp_verification_before_token_redemption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = TinfoilUpstreamProvider(api_key="test")
    mock_receive_token = AsyncMock(return_value=(10_000, "msat", None))
    monkeypatch.setattr("routstr.upstream.base.recieve_token", mock_receive_token)

    with pytest.raises(UpstreamError) as exc_info:
        await provider.handle_x_cashu(
            request=_cashu_request(
                b'{"model":"tinfoil/secure-model","messages":[{"role":"user",'
                b'"content":"SECRET_PROMPT"}],"stream":false}'
            ),
            x_cashu_token="cashuAtest",
            path="v1/chat/completions",
            max_cost_for_model=10_000,
            model_obj=_model(),
        )

    assert "verified EHBP transport is required" in str(exc_info.value)
    assert exc_info.value.status_code == 424
    mock_receive_token.assert_not_awaited()
