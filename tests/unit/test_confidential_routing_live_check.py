from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from importlib import util
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "confidential_routing_live_check.py"
)
SPEC = util.spec_from_file_location("confidential_routing_live_check", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
live_check = util.module_from_spec(SPEC)
sys.modules[SPEC.name] = live_check
SPEC.loader.exec_module(live_check)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _json_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _privatemode_key_release_binding_digest(claims: dict[str, object]) -> str:
    return _json_digest(
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


HPKE_KEY_CONFIG = b"key-config"
HPKE_KEY_CONFIG_DIGEST = "sha256:" + hashlib.sha256(HPKE_KEY_CONFIG).hexdigest()
HPKE_PUBLIC_KEY_DIGEST = _digest("hpke-public-key")
ROUTSTR_PUBLIC_KEY_DIGEST = _digest("routstr-public-key")
TEE_EVIDENCE_DIGEST = _digest("tee-evidence")
TEE_RUNTIME_EVIDENCE_DIGEST = _digest("tee-runtime-evidence")
TEE_CLAIMS_DIGEST = _digest("tee-claims")
TEE_REPORT_DIGEST = _digest("tee-report")
TEE_CERTIFICATE_CHAIN_DIGEST = _digest("tee-chain")
PROVIDER_POLICY_DIGEST = _digest("provider-policy")
PROVIDER_EVIDENCE_DIGEST = _digest("provider-evidence")
PROVIDER_CLAIMS_DIGEST = _digest("provider-claims")
PROVIDER_ATTESTATION_REPORT_DIGEST = _digest("provider-attestation-report")
PROVIDER_MODEL_ATTESTATION_REPORT_DIGEST = _digest("provider-model-attestation-report")
PROVIDER_ENCLAVE_MEASUREMENT = _digest("provider-enclave-measurement")
PROVIDER_CODE_MEASUREMENT = _digest("provider-code-measurement")
PROVIDER_RELEASE_DIGEST = _digest("provider-release")
PROVIDER_TLS_KEY_DIGEST = _digest("provider-tls-key")
PROVIDER_MODEL_ENCLAVE_MEASUREMENT = _digest("provider-model-enclave-measurement")
PROVIDER_MODEL_CODE_MEASUREMENT = _digest("provider-model-code-measurement")
PROVIDER_MODEL_RELEASE_DIGEST = _digest("provider-model-release")
PROVIDER_MODEL_TLS_KEY_DIGEST = _digest("provider-model-tls-key")
PROVIDER_TINFOIL_MODEL_ATTESTATIONS = {
    "tinfoil/gpt-secure": {
        "repo": "tinfoilsh/confidential-gpt-secure",
        "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
        "attestation_report_digest": PROVIDER_MODEL_ATTESTATION_REPORT_DIGEST,
        "attested_hpke_public_key_hex": "e" * 64,
        "enclave_measurement_fingerprint": PROVIDER_MODEL_ENCLAVE_MEASUREMENT,
        "code_measurement_fingerprint": PROVIDER_MODEL_CODE_MEASUREMENT,
        "release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
        "tls_public_key_fingerprint_sha256": PROVIDER_MODEL_TLS_KEY_DIGEST,
        "verification_steps": {
            "hardware_attestation_report": True,
            "hardware_certificate_chain": True,
            "code_transparency": True,
            "measurement_match": True,
            "attested_transport_key_binding": True,
            "freshness": True,
        },
    }
}
PROVIDER_PPQ_PRIVATE_BACKEND_MODEL_ATTESTATIONS = {
    "private/gpt-oss-120b": {
        **PROVIDER_TINFOIL_MODEL_ATTESTATIONS["tinfoil/gpt-secure"],
        "repo": "tinfoilsh/confidential-gpt-oss-120b",
    }
}
PROVIDER_EHBP_PROOF_CLAIMS = {
    "transport": "ehbp",
    "repo": "tinfoilsh/confidential-model-router",
    "payload_policy_digest": PROVIDER_POLICY_DIGEST,
    "payload_evidence_digest": _digest("provider-input-evidence"),
    "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
    "attestation_report_digest": PROVIDER_ATTESTATION_REPORT_DIGEST,
    "attested_hpke_public_key_hex": "b" * 64,
    "enclave_measurement_fingerprint": PROVIDER_ENCLAVE_MEASUREMENT,
    "code_measurement_fingerprint": PROVIDER_CODE_MEASUREMENT,
    "release_digest": PROVIDER_RELEASE_DIGEST,
    "tls_public_key_fingerprint_sha256": PROVIDER_TLS_KEY_DIGEST,
    "verification_steps": {
        "hardware_attestation_report": True,
        "hardware_certificate_chain": True,
        "code_transparency": True,
        "measurement_match": True,
        "attested_transport_key_binding": True,
        "freshness": True,
    },
    "model_attestations": PROVIDER_TINFOIL_MODEL_ATTESTATIONS,
}
PROVIDER_PPQ_PRIVATE_PROOF_CLAIMS = {
    **{
        key: value
        for key, value in PROVIDER_EHBP_PROOF_CLAIMS.items()
        if key != "model_attestations"
    },
    "repo": "ppq-ai/private-tee",
    "attestation_bundle_url_digest": _digest("ppq-private-bundle-url"),
    "client_encryption_boundary": "routstr-tee-ehbp-proxy",
    "selected_model_ids": ["private/gpt-oss-120b"],
    "backend_model_attestations": PROVIDER_PPQ_PRIVATE_BACKEND_MODEL_ATTESTATIONS,
}
PPQ_PRIVATE_BACKEND_VERIFICATION_DETAILS = {
    "backend_tls_matches": True,
    "backend_sigstore_match": True,
    "backend_sigstore_bundle_verified": True,
    "backend_images_verified": True,
}
PPQ_PRIVATE_RESULT_PROVENANCE_DETAILS = {
    "release_digest": PROVIDER_RELEASE_DIGEST,
    "backend_release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
    **PPQ_PRIVATE_BACKEND_VERIFICATION_DETAILS,
}
TINFOIL_CONFIDENTIALITY_POLICY = {
    "repo": "tinfoilsh/confidential-model-router",
    "expected_release_digest": PROVIDER_RELEASE_DIGEST,
    "require_model_attestations": True,
    "model_attestation_targets": {
        "tinfoil/gpt-secure": {
            "repo": "tinfoilsh/confidential-gpt-secure",
            "expected_release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
        },
    },
}
PPQ_PRIVATE_CONFIDENTIALITY_POLICY = {
    "repo": "ppq-ai/private-tee",
    "attestation_bundle_url_digest": _digest("ppq-private-bundle-url"),
    "expected_code_measurement_fingerprint": PROVIDER_CODE_MEASUREMENT,
    "require_model_attestations": True,
    "model_attestation_targets": {
        "private/gpt-oss-120b": {
            "repo": "tinfoilsh/confidential-gpt-oss-120b",
            "expected_release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
        },
    },
}
ROUTING_POLICY = {
    "mode": "required",
    "required": True,
    "client_confidentiality": {
        "mode": "attested-tls-termination",
        "tls_terminates_in_attested_tee": True,
        "inbound_ehbp_ohttp_request_decryption": False,
        "attested_tls_public_key_digest": ROUTSTR_PUBLIC_KEY_DIGEST,
    },
    "routable_with_full_attestation": {
        "tinfoil": ["tinfoil/gpt-secure"],
        "ppq-private": [],
        "privatemode": [],
    },
    "providers": [
        {
            "provider_type": "tinfoil",
            "upstream_name": "tinfoil-prod",
            "base_url": "https://inference.tinfoil.sh/v1",
            "db_id": 7,
            "confidentiality": {
                "enabled": True,
                "verified": True,
                "mode": "tinfoil",
                "verifier": "routstr-tinfoil-go-verifier",
                "policy_digest": PROVIDER_POLICY_DIGEST,
                "evidence_digest": PROVIDER_EVIDENCE_DIGEST,
                "verified_claims_digest": PROVIDER_CLAIMS_DIGEST,
                "proof_claims": dict(PROVIDER_EHBP_PROOF_CLAIMS),
                "verified_at": 1_700_000_000,
                "expires_at": 4_102_444_800,
                "model_ids": ["tinfoil/gpt-secure"],
                "model_id_prefixes": [],
            },
            "confidentiality_policy": dict(TINFOIL_CONFIDENTIALITY_POLICY),
        }
    ],
}
ROUTING_POLICY_DIGEST = _json_digest(ROUTING_POLICY)
OTHER_PROVIDER_EVIDENCE_DIGEST = _digest("other-provider-evidence")
OTHER_HPKE_KEY_CONFIG_DIGEST = _digest("other-hpke-key-config")
OTHER_TEE_RUNTIME_EVIDENCE_DIGEST = _digest("other-tee-runtime-evidence")
PRIVATEMODE_MANIFEST_DIGEST = _digest("privatemode-manifest")
PRIVATEMODE_PROXY_IMAGE_DIGEST = _digest("privatemode-proxy-image")
PRIVATEMODE_PROXY_BINARY_DIGEST = _digest("privatemode-proxy-binary")
PRIVATEMODE_COORDINATOR_MEASUREMENT = _digest("privatemode-coordinator")
PRIVATEMODE_SECRET_SERVICE_MEASUREMENT = _digest("privatemode-secret-service")
PRIVATEMODE_AI_WORKER_MEASUREMENT = _digest("privatemode-ai-worker")
PRIVATEMODE_WORKLOAD_SANS = ["secure-model.default.svc.cluster.local"]
PRIVATEMODE_GPU_ATTESTATION_POLICY = "nvidia-ocsp-good-only"
PRIVATEMODE_CONFIDENTIALITY_POLICY = {
    "expected_trust_tier": "app-e2ee",
    "expected_gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
    "expected_workload_sans": PRIVATEMODE_WORKLOAD_SANS,
    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
    "model_workload_bindings": {
        "privatemode/kimi-latest": {
            "workload_sans": PRIVATEMODE_WORKLOAD_SANS,
        }
    },
}
PRIVATEMODE_EXPECTED_WORKLOAD_IDENTITY_DIGEST = _json_digest(
    {
        "ids": [],
        "sans": PRIVATEMODE_WORKLOAD_SANS,
    }
)
PRIVATEMODE_MODEL_WORKLOAD_BINDING_DIGEST = _json_digest(
    {
        "privatemode/kimi-latest": {
            "workload_ids": [],
            "workload_sans": PRIVATEMODE_WORKLOAD_SANS,
        }
    }
)
TEE_PROOF_CLAIMS = {
    "attestation_document_format": "tdx_quote",
    "hpke_key_config_digest": HPKE_KEY_CONFIG_DIGEST,
    "hpke_public_key_digest": HPKE_PUBLIC_KEY_DIGEST,
    "public_key_digest": ROUTSTR_PUBLIC_KEY_DIGEST,
    "routstr_code_measurement": _digest("routstr-code-measurement"),
    "routstr_config_measurement": ROUTING_POLICY_DIGEST,
    "tee_attestation_report_digest": TEE_REPORT_DIGEST,
    "tee_certificate_chain_digest": TEE_CERTIFICATE_CHAIN_DIGEST,
    "tee_report_nonce": "nonce",
    "tee_report_nonce_digest": _digest("nonce"),
    "verification_steps": {
        "tee_attestation_report": True,
        "tee_certificate_chain": True,
        "measurement_match": True,
        "runtime_policy_binding": True,
        "hpke_key_binding": True,
        "public_key_binding": True,
        "freshness": True,
    },
}
TEE_PROOF_CLAIMS["tee_report_data_digest"] = live_check.routstr_tee_report_data_digest(
    routing_policy_digest=ROUTING_POLICY_DIGEST,
    hpke_key_config_digest=HPKE_KEY_CONFIG_DIGEST,
    hpke_public_key_digest=HPKE_PUBLIC_KEY_DIGEST,
    public_key_digest=ROUTSTR_PUBLIC_KEY_DIGEST,
    verification_nonce="nonce",
)
TEE_PROOF_CLAIMS["tee_report_data_hex"] = live_check.routstr_tee_report_data_hex(
    routing_policy_digest=ROUTING_POLICY_DIGEST,
    hpke_key_config_digest=HPKE_KEY_CONFIG_DIGEST,
    hpke_public_key_digest=HPKE_PUBLIC_KEY_DIGEST,
    public_key_digest=ROUTSTR_PUBLIC_KEY_DIGEST,
    verification_nonce="nonce",
)
PROVIDER_PRIVATEMODE_PROOF_CLAIMS = {
    "transport": "privatemode-proxy",
    "trust_tier": "app-e2ee",
    "payload_policy_digest": PROVIDER_POLICY_DIGEST,
    "payload_evidence_digest": _digest("provider-input-evidence"),
    "manifest_digest": PRIVATEMODE_MANIFEST_DIGEST,
    "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
    "proxy_binary_digest": PRIVATEMODE_PROXY_BINARY_DIGEST,
    "coordinator_measurement": PRIVATEMODE_COORDINATOR_MEASUREMENT,
    "secret_service_measurement": PRIVATEMODE_SECRET_SERVICE_MEASUREMENT,
    "ai_worker_measurement": PRIVATEMODE_AI_WORKER_MEASUREMENT,
    "gpu_attestation_policy": PRIVATEMODE_GPU_ATTESTATION_POLICY,
    "coordinator_attestation_doc_digest": _digest("coordinator-attestation-doc"),
    "mesh_ca_digest": _digest("mesh-ca"),
    "secret_service_certificate_digest": _digest("secret-service-certificate"),
    "ai_worker_manifest_digest": _digest("ai-worker-manifest"),
    "attested_workload_identity_digest": _digest("attested-workload-identity"),
    "attested_workload_policy_digest": PRIVATEMODE_AI_WORKER_MEASUREMENT,
    "expected_workload_identity_digest": PRIVATEMODE_EXPECTED_WORKLOAD_IDENTITY_DIGEST,
    "model_workload_binding_digest": PRIVATEMODE_MODEL_WORKLOAD_BINDING_DIGEST,
    "selected_model_ids": ["privatemode/kimi-latest"],
    "nvidia_ocsp_policy_header_digest": _digest("nvidia-ocsp-policy-header"),
    "nvidia_ocsp_policy_mac_digest": _digest("nvidia-ocsp-policy-mac"),
    "prompt_encryption_ciphertext_digest": _digest("prompt-encryption-ciphertext"),
    "inference_secret_id_digest": _digest("inference-secret-id"),
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
PROVIDER_PRIVATEMODE_PROOF_CLAIMS["key_release_binding"] = (
    _privatemode_key_release_binding_digest(PROVIDER_PRIVATEMODE_PROOF_CLAIMS)
)


class _JsonResponse:
    headers = {"content-type": "application/json"}

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_JsonResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_fetch_json_rejects_non_standard_json_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live validation must not accept NaN/Infinity in proof-bearing JSON."""

    def fake_urlopen(*args: object, **kwargs: object) -> _JsonResponse:
        return _JsonResponse(b'{"mode":"required","non_canonical":NaN}')

    monkeypatch.setattr(live_check, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="returned invalid JSON"):
        live_check.fetch_json(
            "https://routstr.example",
            "/v1/confidentiality/status",
            timeout=1.0,
        )


def test_post_multipart_rejects_non_standard_json_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audio inference probes must not accept NaN/Infinity responses."""

    def fake_urlopen(*args: object, **kwargs: object) -> _JsonResponse:
        return _JsonResponse(b'{"text":NaN}')

    monkeypatch.setattr(live_check, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="returned invalid JSON"):
        live_check.post_multipart(
            "https://routstr.example",
            "/v1/audio/transcriptions",
            fields={"model": "tinfoil/voxtral-small-24b"},
            files=[("file", "sample.wav", b"RIFF....", "audio/wav")],
            headers={},
            timeout=1.0,
        )


def _verified_status() -> dict[str, object]:
    return {
        "mode": "required",
        "required": True,
        "end_to_end_ready": True,
        "routable_with_full_attestation": {
            "tinfoil": ["tinfoil/gpt-secure"],
            "ppq-private": [],
            "privatemode": [],
        },
        "routstr_tee": {
            "required": True,
            "ready": True,
            "client_confidentiality": {
                "mode": "attested-tls-termination",
                "tls_terminates_in_attested_tee": True,
                "inbound_ehbp_ohttp_request_decryption": False,
                "attested_tls_public_key_digest": ROUTSTR_PUBLIC_KEY_DIGEST,
            },
            "attestation_evidence_digest": TEE_EVIDENCE_DIGEST,
            "hpke_key_config_digest": HPKE_KEY_CONFIG_DIGEST,
            "hpke_public_key_digest": HPKE_PUBLIC_KEY_DIGEST,
            "local_verification": {
                "verified": True,
                "verified_at": 1_700_000_000,
                "expires_at": 4_102_444_800,
                "evidence_digest": TEE_RUNTIME_EVIDENCE_DIGEST,
                "verified_claims_digest": TEE_CLAIMS_DIGEST,
                "proof_claims": json.loads(json.dumps(TEE_PROOF_CLAIMS)),
            },
        },
        "providers": [
            {
                "provider_type": "tinfoil",
                "upstream_name": "tinfoil-prod",
                "supported_endpoints": ["/v1/chat/completions"],
                "confidentiality": {
                    "enabled": True,
                    "verified": True,
                    "mode": "tinfoil",
                    "verifier": "routstr-tinfoil-go-verifier",
                    "policy_digest": PROVIDER_POLICY_DIGEST,
                    "evidence_digest": PROVIDER_EVIDENCE_DIGEST,
                    "verified_at": 1_700_000_000,
                    "expires_at": 4_102_444_800,
                    "verified_claims_digest": PROVIDER_CLAIMS_DIGEST,
                    "proof_claims": json.loads(json.dumps(PROVIDER_EHBP_PROOF_CLAIMS)),
                    "model_ids": ["tinfoil/gpt-secure"],
                    "model_id_prefixes": [],
                },
                "confidentiality_policy": dict(TINFOIL_CONFIDENTIALITY_POLICY),
            }
        ],
    }


def _verified_models() -> dict[str, object]:
    return {
        "object": "list",
        "routstr_tee": json.loads(json.dumps(_verified_status()["routstr_tee"])),
        "data": [
            {
                "id": "tinfoil/gpt-secure",
                "confidential": True,
                "attestation_provider": "tinfoil",
                "provider_attestation_status": "verified",
                "attestation_status": "verified",
                "attestation_evidence_digest": PROVIDER_EVIDENCE_DIGEST,
                "confidentiality": {
                    "enabled": True,
                    "verified": True,
                    "attestation_status": "verified",
                    "mode": "tinfoil",
                    "provider_type": "tinfoil",
                    "verifier": "routstr-tinfoil-go-verifier",
                    "policy_digest": PROVIDER_POLICY_DIGEST,
                    "evidence_digest": PROVIDER_EVIDENCE_DIGEST,
                    "verified_at": 1_700_000_000,
                    "expires_at": 4_102_444_800,
                    "verified_claims_digest": PROVIDER_CLAIMS_DIGEST,
                    "model_ids": ["tinfoil/gpt-secure"],
                    "model_id_prefixes": [],
                    "supported_endpoints": ["/v1/chat/completions"],
                    "proof_claims": dict(PROVIDER_EHBP_PROOF_CLAIMS),
                },
            }
        ],
    }


def _verified_privatemode_status() -> dict[str, object]:
    status = _verified_status()
    status["routable_with_full_attestation"] = {
        "tinfoil": [],
        "ppq-private": [],
        "privatemode": ["privatemode/kimi-latest"],
    }
    status["providers"] = [
        {
            "provider_type": "privatemode",
            "upstream_name": "privatemode-prod",
            "supported_endpoints": ["/v1/messages"],
            "confidentiality": {
                "enabled": True,
                "verified": True,
                "mode": "privatemode",
                "verifier": "routstr-privatemode-go-verifier",
                "policy_digest": PROVIDER_POLICY_DIGEST,
                "evidence_digest": PROVIDER_EVIDENCE_DIGEST,
                "verified_at": 1_700_000_000,
                "expires_at": 4_102_444_800,
                "verified_claims_digest": PROVIDER_CLAIMS_DIGEST,
                "proof_claims": dict(PROVIDER_PRIVATEMODE_PROOF_CLAIMS),
                "model_ids": ["privatemode/kimi-latest"],
                "model_id_prefixes": [],
            },
            "confidentiality_policy": dict(PRIVATEMODE_CONFIDENTIALITY_POLICY),
        }
    ]
    _set_routstr_tee_policy_binding(status["routstr_tee"], _privatemode_routing_policy())  # type: ignore[arg-type,index]
    return status


def _verified_ppq_private_status() -> dict[str, object]:
    status = _verified_status()
    status["routable_with_full_attestation"] = {
        "tinfoil": [],
        "ppq-private": ["private/gpt-oss-120b"],
        "privatemode": [],
    }
    status["providers"] = [
        {
            "provider_type": "ppq-private",
            "upstream_name": "ppq-private-prod",
            "supported_endpoints": ["/v1/chat/completions"],
            "confidentiality": {
                "enabled": True,
                "verified": True,
                "mode": "ppq-private-tee",
                "verifier": "routstr-ppq-private-go-verifier",
                "policy_digest": PROVIDER_POLICY_DIGEST,
                "evidence_digest": PROVIDER_EVIDENCE_DIGEST,
                "verified_at": 1_700_000_000,
                "expires_at": 4_102_444_800,
                "verified_claims_digest": PROVIDER_CLAIMS_DIGEST,
                "proof_claims": dict(PROVIDER_PPQ_PRIVATE_PROOF_CLAIMS),
                "model_ids": ["private/gpt-oss-120b"],
                "model_id_prefixes": [],
            },
            "confidentiality_policy": dict(PPQ_PRIVATE_CONFIDENTIALITY_POLICY),
        }
    ]
    _set_routstr_tee_policy_binding(status["routstr_tee"], _ppq_private_routing_policy())  # type: ignore[arg-type,index]
    return status


def _verified_ppq_private_models() -> dict[str, object]:
    models: dict[str, object] = {
        "object": "list",
        "routstr_tee": json.loads(json.dumps(_verified_status()["routstr_tee"])),
        "data": [
            {
                "id": "private/gpt-oss-120b",
                "confidential": True,
                "attestation_provider": "ppq-private",
                "provider_attestation_status": "verified",
                "attestation_status": "verified",
                "attestation_evidence_digest": PROVIDER_EVIDENCE_DIGEST,
                "confidentiality": {
                    "enabled": True,
                    "verified": True,
                    "attestation_status": "verified",
                    "mode": "ppq-private-tee",
                    "provider_type": "ppq-private",
                    "verifier": "routstr-ppq-private-go-verifier",
                    "policy_digest": PROVIDER_POLICY_DIGEST,
                    "evidence_digest": PROVIDER_EVIDENCE_DIGEST,
                    "verified_at": 1_700_000_000,
                    "expires_at": 4_102_444_800,
                    "verified_claims_digest": PROVIDER_CLAIMS_DIGEST,
                    "model_ids": ["private/gpt-oss-120b"],
                    "model_id_prefixes": [],
                    "supported_endpoints": ["/v1/chat/completions"],
                    "proof_claims": dict(PROVIDER_PPQ_PRIVATE_PROOF_CLAIMS),
                },
            }
        ],
    }
    _set_routstr_tee_policy_binding(models["routstr_tee"], _ppq_private_routing_policy())  # type: ignore[arg-type,index]
    return models


def _ppq_private_routing_policy() -> dict[str, object]:
    return {
        "mode": "required",
        "required": True,
        "routable_with_full_attestation": {
            "tinfoil": [],
            "ppq-private": ["private/gpt-oss-120b"],
            "privatemode": [],
        },
        "providers": [
            {
                "provider_type": "ppq-private",
                "upstream_name": "ppq-private-prod",
                "base_url": "https://api.ppq.ai/private/v1",
                "db_id": 8,
                "confidentiality": {
                    "enabled": True,
                    "verified": True,
                    "mode": "ppq-private-tee",
                    "verifier": "routstr-ppq-private-go-verifier",
                    "policy_digest": PROVIDER_POLICY_DIGEST,
                    "evidence_digest": PROVIDER_EVIDENCE_DIGEST,
                    "verified_claims_digest": PROVIDER_CLAIMS_DIGEST,
                    "proof_claims": dict(PROVIDER_PPQ_PRIVATE_PROOF_CLAIMS),
                    "verified_at": 1_700_000_000,
                    "expires_at": 4_102_444_800,
                    "model_ids": ["private/gpt-oss-120b"],
                    "model_id_prefixes": [],
                },
                "confidentiality_policy": dict(PPQ_PRIVATE_CONFIDENTIALITY_POLICY),
            }
        ],
    }


def _verified_privatemode_models() -> dict[str, object]:
    models: dict[str, object] = {
        "object": "list",
        "routstr_tee": json.loads(json.dumps(_verified_status()["routstr_tee"])),
        "data": [
            {
                "id": "privatemode/kimi-latest",
                "confidential": True,
                "attestation_provider": "privatemode",
                "provider_attestation_status": "verified",
                "attestation_status": "verified",
                "attestation_evidence_digest": PROVIDER_EVIDENCE_DIGEST,
                "confidentiality": {
                    "enabled": True,
                    "verified": True,
                    "attestation_status": "verified",
                    "mode": "privatemode",
                    "provider_type": "privatemode",
                    "verifier": "routstr-privatemode-go-verifier",
                    "policy_digest": PROVIDER_POLICY_DIGEST,
                    "evidence_digest": PROVIDER_EVIDENCE_DIGEST,
                    "verified_at": 1_700_000_000,
                    "expires_at": 4_102_444_800,
                    "verified_claims_digest": PROVIDER_CLAIMS_DIGEST,
                    "model_ids": ["privatemode/kimi-latest"],
                    "model_id_prefixes": [],
                    "supported_endpoints": ["/v1/messages"],
                    "proof_claims": dict(PROVIDER_PRIVATEMODE_PROOF_CLAIMS),
                },
            }
        ],
    }
    _set_routstr_tee_policy_binding(models["routstr_tee"], _privatemode_routing_policy())  # type: ignore[arg-type,index]
    return models


def _privatemode_routing_policy() -> dict[str, object]:
    return {
        "mode": "required",
        "required": True,
        "routable_with_full_attestation": {
            "tinfoil": [],
            "ppq-private": [],
            "privatemode": ["privatemode/kimi-latest"],
        },
        "providers": [
            {
                "provider_type": "privatemode",
                "upstream_name": "privatemode-prod",
                "base_url": "http://127.0.0.1:8080/v1",
                "db_id": 8,
                "confidentiality": {
                    "enabled": True,
                    "verified": True,
                    "mode": "privatemode",
                    "verifier": "routstr-privatemode-go-verifier",
                    "policy_digest": PROVIDER_POLICY_DIGEST,
                    "evidence_digest": PROVIDER_EVIDENCE_DIGEST,
                    "verified_claims_digest": PROVIDER_CLAIMS_DIGEST,
                    "proof_claims": dict(PROVIDER_PRIVATEMODE_PROOF_CLAIMS),
                    "verified_at": 1_700_000_000,
                    "expires_at": 4_102_444_800,
                    "model_ids": ["privatemode/kimi-latest"],
                    "model_id_prefixes": [],
                },
                "confidentiality_policy": dict(PRIVATEMODE_CONFIDENTIALITY_POLICY),
            }
        ],
    }


def _attestation_statement(
    routing_policy: dict[str, object] | None = None,
) -> dict[str, object]:
    policy = json.loads(json.dumps(routing_policy or ROUTING_POLICY))
    policy.setdefault(
        "client_confidentiality",
        {
            "mode": "attested-tls-termination",
            "tls_terminates_in_attested_tee": True,
            "inbound_ehbp_ohttp_request_decryption": False,
            "attested_tls_public_key_digest": ROUTSTR_PUBLIC_KEY_DIGEST,
        },
    )
    routing_policy_digest = _json_digest(policy)
    tee_proof_claims = json.loads(json.dumps(TEE_PROOF_CLAIMS))
    tee_proof_claims["routstr_config_measurement"] = routing_policy_digest
    tee_proof_claims[
        "tee_report_data_digest"
    ] = live_check.routstr_tee_report_data_digest(
        routing_policy_digest=routing_policy_digest,
        hpke_key_config_digest=HPKE_KEY_CONFIG_DIGEST,
        hpke_public_key_digest=HPKE_PUBLIC_KEY_DIGEST,
        public_key_digest=ROUTSTR_PUBLIC_KEY_DIGEST,
        verification_nonce=tee_proof_claims["tee_report_nonce"],
    )
    tee_proof_claims["tee_report_data_hex"] = live_check.routstr_tee_report_data_hex(
        routing_policy_digest=routing_policy_digest,
        hpke_key_config_digest=HPKE_KEY_CONFIG_DIGEST,
        hpke_public_key_digest=HPKE_PUBLIC_KEY_DIGEST,
        public_key_digest=ROUTSTR_PUBLIC_KEY_DIGEST,
        verification_nonce=tee_proof_claims["tee_report_nonce"],
    )
    _set_privatemode_proxy_artifact_binding(tee_proof_claims, policy)
    return {
        "schema_version": "routstr-attestation-v1",
        "routing_policy": policy,
        "routing_policy_digest": routing_policy_digest,
        "tee": {
            "hpke_key_config": {
                "key_config_digest": HPKE_KEY_CONFIG_DIGEST,
                "public_key_digest": HPKE_PUBLIC_KEY_DIGEST,
            },
            "local_verification": {
                "verified": True,
                "verified_at": 1_700_000_000,
                "expires_at": 4_102_444_800,
                "evidence_digest": TEE_RUNTIME_EVIDENCE_DIGEST,
                "verified_claims_digest": TEE_CLAIMS_DIGEST,
                "proof_claims": tee_proof_claims,
            },
        },
    }


def _set_routstr_tee_policy_binding(
    routstr_tee: dict[str, object],
    routing_policy: dict[str, object],
) -> None:
    policy = json.loads(json.dumps(routing_policy))
    policy.setdefault(
        "client_confidentiality",
        {
            "mode": "attested-tls-termination",
            "tls_terminates_in_attested_tee": True,
            "inbound_ehbp_ohttp_request_decryption": False,
            "attested_tls_public_key_digest": ROUTSTR_PUBLIC_KEY_DIGEST,
        },
    )
    routing_policy_digest = _json_digest(policy)
    local_verification = routstr_tee["local_verification"]  # type: ignore[index]
    proof_claims = local_verification["proof_claims"]  # type: ignore[index]
    proof_claims["routstr_config_measurement"] = routing_policy_digest  # type: ignore[index]
    proof_claims["tee_report_data_digest"] = live_check.routstr_tee_report_data_digest(  # type: ignore[index]
        routing_policy_digest=routing_policy_digest,
        hpke_key_config_digest=HPKE_KEY_CONFIG_DIGEST,
        hpke_public_key_digest=HPKE_PUBLIC_KEY_DIGEST,
        public_key_digest=ROUTSTR_PUBLIC_KEY_DIGEST,
        verification_nonce=proof_claims["tee_report_nonce"],  # type: ignore[index]
    )
    proof_claims["tee_report_data_hex"] = live_check.routstr_tee_report_data_hex(  # type: ignore[index]
        routing_policy_digest=routing_policy_digest,
        hpke_key_config_digest=HPKE_KEY_CONFIG_DIGEST,
        hpke_public_key_digest=HPKE_PUBLIC_KEY_DIGEST,
        public_key_digest=ROUTSTR_PUBLIC_KEY_DIGEST,
        verification_nonce=proof_claims["tee_report_nonce"],  # type: ignore[index]
    )
    _set_privatemode_proxy_artifact_binding(proof_claims, policy)  # type: ignore[arg-type]


def _set_privatemode_proxy_artifact_binding(
    proof_claims: dict[str, object],
    routing_policy: dict[str, object],
) -> None:
    proxy_artifacts = live_check.required_local_proxy_binary_artifacts(routing_policy)
    if proxy_artifacts:
        proof_claims["attested_local_artifacts"] = {
            artifact_claim: sorted(proxy_digests)[0]
            for artifact_claim, proxy_digests in sorted(proxy_artifacts.items())
        }


def test_live_check_accepts_verified_confidential_surfaces() -> None:
    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert all(result.ok for result in results)


def test_live_check_rejects_status_attestation_routable_model_map_mismatch() -> None:
    attestation = _attestation_statement()
    routing_policy = attestation["routing_policy"]  # type: ignore[index]
    routing_policy["routable_with_full_attestation"] = {  # type: ignore[index]
        "tinfoil": [],
        "ppq-private": [],
        "privatemode": [],
    }
    attestation["routing_policy_digest"] = _json_digest(routing_policy)
    proof_claims = attestation["tee"]["local_verification"]["proof_claims"]  # type: ignore[index]
    proof_claims["routstr_config_measurement"] = attestation["routing_policy_digest"]  # type: ignore[index]
    proof_claims["tee_report_data_digest"] = live_check.routstr_tee_report_data_digest(
        routing_policy_digest=attestation["routing_policy_digest"],  # type: ignore[arg-type]
        hpke_key_config_digest=HPKE_KEY_CONFIG_DIGEST,
        hpke_public_key_digest=HPKE_PUBLIC_KEY_DIGEST,
        public_key_digest=ROUTSTR_PUBLIC_KEY_DIGEST,
        verification_nonce=proof_claims["tee_report_nonce"],  # type: ignore[index]
    )
    proof_claims["tee_report_data_hex"] = live_check.routstr_tee_report_data_hex(
        routing_policy_digest=attestation["routing_policy_digest"],  # type: ignore[arg-type]
        hpke_key_config_digest=HPKE_KEY_CONFIG_DIGEST,
        hpke_public_key_digest=HPKE_PUBLIC_KEY_DIGEST,
        public_key_digest=ROUTSTR_PUBLIC_KEY_DIGEST,
        verification_nonce=proof_claims["tee_report_nonce"],  # type: ignore[index]
    )

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "routable_with_full_attestation matches status and attestation"
        in result.message
        for result in results
    )


def test_live_check_rejects_expected_route_absent_from_exact_routable_map() -> None:
    status = _verified_status()
    status["routable_with_full_attestation"] = {
        "tinfoil": [],
        "ppq-private": [],
        "privatemode": [],
    }
    attestation = _attestation_statement()
    routing_policy = attestation["routing_policy"]  # type: ignore[index]
    routing_policy["routable_with_full_attestation"] = {  # type: ignore[index]
        "tinfoil": [],
        "ppq-private": [],
        "privatemode": [],
    }
    attestation["routing_policy_digest"] = _json_digest(routing_policy)
    proof_claims = attestation["tee"]["local_verification"]["proof_claims"]  # type: ignore[index]
    proof_claims["routstr_config_measurement"] = attestation["routing_policy_digest"]  # type: ignore[index]
    proof_claims["tee_report_data_digest"] = live_check.routstr_tee_report_data_digest(
        routing_policy_digest=attestation["routing_policy_digest"],  # type: ignore[arg-type]
        hpke_key_config_digest=HPKE_KEY_CONFIG_DIGEST,
        hpke_public_key_digest=HPKE_PUBLIC_KEY_DIGEST,
        public_key_digest=ROUTSTR_PUBLIC_KEY_DIGEST,
        verification_nonce=proof_claims["tee_report_nonce"],  # type: ignore[index]
    )
    proof_claims["tee_report_data_hex"] = live_check.routstr_tee_report_data_hex(
        routing_policy_digest=attestation["routing_policy_digest"],  # type: ignore[arg-type]
        hpke_key_config_digest=HPKE_KEY_CONFIG_DIGEST,
        hpke_public_key_digest=HPKE_PUBLIC_KEY_DIGEST,
        public_key_digest=ROUTSTR_PUBLIC_KEY_DIGEST,
        verification_nonce=proof_claims["tee_report_nonce"],  # type: ignore[index]
    )

    results = live_check.collect_results(
        status=status,
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "tinfoil/gpt-secure is listed in routable_with_full_attestation[tinfoil]"
        in result.message
        for result in results
    )


def test_live_check_rejects_public_verified_model_absent_from_exact_routable_map() -> None:
    models = _verified_models()
    extra_model = json.loads(json.dumps(models["data"][0]))  # type: ignore[index]
    extra_model["id"] = "tinfoil/extra"
    models["data"].append(extra_model)  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "tinfoil/extra is listed in routable_with_full_attestation[tinfoil]"
        in result.message
        for result in results
    )


def test_live_check_rejects_exact_routable_map_entry_absent_from_verified_models() -> None:
    status = _verified_status()
    status["routable_with_full_attestation"] = {
        "tinfoil": ["tinfoil/gpt-secure", "tinfoil/ghost"],
        "ppq-private": [],
        "privatemode": [],
    }
    attestation = _attestation_statement()
    routing_policy = attestation["routing_policy"]  # type: ignore[index]
    routing_policy["routable_with_full_attestation"] = {  # type: ignore[index]
        "tinfoil": ["tinfoil/gpt-secure", "tinfoil/ghost"],
        "ppq-private": [],
        "privatemode": [],
    }
    attestation["routing_policy_digest"] = _json_digest(routing_policy)
    proof_claims = attestation["tee"]["local_verification"]["proof_claims"]  # type: ignore[index]
    proof_claims["routstr_config_measurement"] = attestation["routing_policy_digest"]  # type: ignore[index]
    proof_claims["tee_report_data_digest"] = live_check.routstr_tee_report_data_digest(
        routing_policy_digest=attestation["routing_policy_digest"],  # type: ignore[arg-type]
        hpke_key_config_digest=HPKE_KEY_CONFIG_DIGEST,
        hpke_public_key_digest=HPKE_PUBLIC_KEY_DIGEST,
        public_key_digest=ROUTSTR_PUBLIC_KEY_DIGEST,
        verification_nonce=proof_claims["tee_report_nonce"],  # type: ignore[index]
    )
    proof_claims["tee_report_data_hex"] = live_check.routstr_tee_report_data_hex(
        routing_policy_digest=attestation["routing_policy_digest"],  # type: ignore[arg-type]
        hpke_key_config_digest=HPKE_KEY_CONFIG_DIGEST,
        hpke_public_key_digest=HPKE_PUBLIC_KEY_DIGEST,
        public_key_digest=ROUTSTR_PUBLIC_KEY_DIGEST,
        verification_nonce=proof_claims["tee_report_nonce"],  # type: ignore[index]
    )

    results = live_check.collect_results(
        status=status,
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "routable_with_full_attestation[tinfoil] model tinfoil/ghost is fully verified in /v1/models"
        in result.message
        for result in results
    )


def test_live_check_rejects_status_missing_client_confidentiality_boundary() -> None:
    status = _verified_status()
    del status["routstr_tee"]["client_confidentiality"]  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "local Routstr TEE client_confidentiality is present" in result.message
        for result in results
    )


def test_live_check_rejects_status_missing_attested_tls_public_key_digest() -> None:
    status = _verified_status()
    client_confidentiality = status["routstr_tee"]["client_confidentiality"]  # type: ignore[index]
    del client_confidentiality["attested_tls_public_key_digest"]  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and (
            "local Routstr TEE client_confidentiality."
            "attested_tls_public_key_digest"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_status_attested_tls_public_key_digest_mismatch() -> None:
    status = _verified_status()
    client_confidentiality = status["routstr_tee"]["client_confidentiality"]  # type: ignore[index]
    client_confidentiality["attested_tls_public_key_digest"] = _digest("wrong-routstr-public-key")  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and (
            "local Routstr TEE client_confidentiality."
            "attested_tls_public_key_digest matches local TEE proof"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_models_missing_local_routstr_tee_boundary() -> None:
    models = _verified_models()
    del models["routstr_tee"]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "/v1/models routstr_tee status is present" in result.message
        for result in results
    )


def test_live_check_rejects_models_local_tee_evidence_digest_mismatch() -> None:
    models = _verified_models()
    local_verification = models["routstr_tee"]["local_verification"]  # type: ignore[index]
    local_verification["evidence_digest"] = _digest("models-tee-runtime-evidence")  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "/v1/models local TEE evidence digest matches status" in result.message
        for result in results
    )


def test_live_check_rejects_models_local_tee_proof_claim_mismatch() -> None:
    models = _verified_models()
    local_verification = models["routstr_tee"]["local_verification"]  # type: ignore[index]
    proof_claims = local_verification["proof_claims"]  # type: ignore[index]
    proof_claims["tee_report_nonce"] = "models-other-nonce"  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "/v1/models local TEE proof claims match status" in result.message
        for result in results
    )


def test_live_check_rejects_missing_client_confidentiality_boundary() -> None:
    attestation = _attestation_statement()
    del attestation["routing_policy"]["client_confidentiality"]  # type: ignore[index]
    attestation["routing_policy_digest"] = _json_digest(
        attestation["routing_policy"]
    )
    proof_claims = attestation["tee"]["local_verification"]["proof_claims"]  # type: ignore[index]
    proof_claims["routstr_config_measurement"] = attestation["routing_policy_digest"]  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "attestation routing_policy.client_confidentiality is present"
        in result.message
        for result in results
    )


def test_live_check_rejects_attestation_attested_tls_public_key_digest_mismatch() -> None:
    attestation = _attestation_statement()
    attestation["routing_policy"]["client_confidentiality"][  # type: ignore[index]
        "attested_tls_public_key_digest"
    ] = _digest("wrong-routstr-public-key")
    attestation["routing_policy_digest"] = _json_digest(
        attestation["routing_policy"]
    )
    proof_claims = attestation["tee"]["local_verification"]["proof_claims"]  # type: ignore[index]
    proof_claims["routstr_config_measurement"] = attestation["routing_policy_digest"]  # type: ignore[index]
    proof_claims["tee_report_data_digest"] = live_check.routstr_tee_report_data_digest(
        routing_policy_digest=attestation["routing_policy_digest"],
        hpke_key_config_digest=HPKE_KEY_CONFIG_DIGEST,
        hpke_public_key_digest=HPKE_PUBLIC_KEY_DIGEST,
        public_key_digest=ROUTSTR_PUBLIC_KEY_DIGEST,
        verification_nonce=proof_claims["tee_report_nonce"],  # type: ignore[index]
    )
    proof_claims["tee_report_data_hex"] = live_check.routstr_tee_report_data_hex(
        routing_policy_digest=attestation["routing_policy_digest"],
        hpke_key_config_digest=HPKE_KEY_CONFIG_DIGEST,
        hpke_public_key_digest=HPKE_PUBLIC_KEY_DIGEST,
        public_key_digest=ROUTSTR_PUBLIC_KEY_DIGEST,
        verification_nonce=proof_claims["tee_report_nonce"],  # type: ignore[index]
    )

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and (
            "attestation routing_policy.client_confidentiality."
            "attested_tls_public_key_digest matches attestation local TEE proof"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_status_attestation_hpke_public_key_digest_mismatch() -> None:
    attestation = _attestation_statement()
    tee = attestation["tee"]  # type: ignore[index]
    hpke_key_config = tee["hpke_key_config"]  # type: ignore[index]
    hpke_key_config["public_key_digest"] = OTHER_HPKE_KEY_CONFIG_DIGEST  # type: ignore[index]
    proof_claims = tee["local_verification"]["proof_claims"]  # type: ignore[index]
    proof_claims["hpke_public_key_digest"] = OTHER_HPKE_KEY_CONFIG_DIGEST  # type: ignore[index]
    proof_claims["tee_report_data_digest"] = live_check.routstr_tee_report_data_digest(
        routing_policy_digest=attestation["routing_policy_digest"],  # type: ignore[arg-type]
        hpke_key_config_digest=HPKE_KEY_CONFIG_DIGEST,
        hpke_public_key_digest=OTHER_HPKE_KEY_CONFIG_DIGEST,
        public_key_digest=ROUTSTR_PUBLIC_KEY_DIGEST,
        verification_nonce=proof_claims["tee_report_nonce"],  # type: ignore[index]
    )
    proof_claims["tee_report_data_hex"] = live_check.routstr_tee_report_data_hex(
        routing_policy_digest=attestation["routing_policy_digest"],  # type: ignore[arg-type]
        hpke_key_config_digest=HPKE_KEY_CONFIG_DIGEST,
        hpke_public_key_digest=OTHER_HPKE_KEY_CONFIG_DIGEST,
        public_key_digest=ROUTSTR_PUBLIC_KEY_DIGEST,
        verification_nonce=proof_claims["tee_report_nonce"],  # type: ignore[index]
    )

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "local TEE HPKE public key digest matches status and attestation"
        in result.message
        for result in results
    )


def test_live_check_rejects_status_attestation_local_tee_proof_claim_mismatch() -> None:
    attestation = _attestation_statement()
    proof_claims = attestation["tee"]["local_verification"]["proof_claims"]  # type: ignore[index]
    proof_claims["tee_report_nonce"] = "other-nonce"
    proof_claims["tee_report_nonce_digest"] = _digest("other-nonce")
    proof_claims["tee_report_data_digest"] = live_check.routstr_tee_report_data_digest(
        routing_policy_digest=attestation["routing_policy_digest"],  # type: ignore[arg-type]
        hpke_key_config_digest=HPKE_KEY_CONFIG_DIGEST,
        hpke_public_key_digest=HPKE_PUBLIC_KEY_DIGEST,
        public_key_digest=ROUTSTR_PUBLIC_KEY_DIGEST,
        verification_nonce=proof_claims["tee_report_nonce"],  # type: ignore[index]
    )
    proof_claims["tee_report_data_hex"] = live_check.routstr_tee_report_data_hex(
        routing_policy_digest=attestation["routing_policy_digest"],  # type: ignore[arg-type]
        hpke_key_config_digest=HPKE_KEY_CONFIG_DIGEST,
        hpke_public_key_digest=HPKE_PUBLIC_KEY_DIGEST,
        public_key_digest=ROUTSTR_PUBLIC_KEY_DIGEST,
        verification_nonce=proof_claims["tee_report_nonce"],  # type: ignore[index]
    )

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "local TEE proof claims match status and attestation" in result.message
        for result in results
    )


def test_live_check_accepts_verified_privatemode_surfaces() -> None:
    results = live_check.collect_results(
        status=_verified_privatemode_status(),
        attestation=_attestation_statement(_privatemode_routing_policy()),
        models=_verified_privatemode_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("privatemode", "privatemode/kimi-latest")],
    )

    assert all(result.ok for result in results)


def test_live_check_rejects_privatemode_without_local_proxy_artifact_binding() -> None:
    attestation = _attestation_statement(_privatemode_routing_policy())
    proof_claims = attestation["tee"]["local_verification"]["proof_claims"]  # type: ignore[index]
    del proof_claims["attested_local_artifacts"]  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_privatemode_status(),
        attestation=attestation,
        models=_verified_privatemode_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("privatemode", "privatemode/kimi-latest")],
    )

    assert any(
        not result.ok
        and (
            "attestation local TEE proof_claims.attested_local_artifacts."
            "privatemode_proxy_binary matches Privatemode proxy binary digest"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_privatemode_policy_without_manifest_identity() -> None:
    routing_policy = _privatemode_routing_policy()
    policy = routing_policy["providers"][0]["confidentiality_policy"]  # type: ignore[index]
    policy.pop("manifest_digest", None)  # type: ignore[union-attr]
    policy.pop("manifest_log_dir", None)  # type: ignore[union-attr]

    results = live_check.collect_results(
        status=_verified_privatemode_status(),
        attestation=_attestation_statement(routing_policy),
        models=_verified_privatemode_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("privatemode", "privatemode/kimi-latest")],
    )

    assert any(
        not result.ok
        and "confidentiality_policy manifest identity is pinned"
        in result.message
        for result in results
    )


def test_live_check_rejects_privatemode_manifest_log_policy_without_log_digest() -> None:
    policy = dict(PRIVATEMODE_CONFIDENTIALITY_POLICY)
    policy.pop("manifest_digest", None)
    policy["manifest_log_dir"] = "/var/lib/privatemode/manifest-log"
    proof_claims = dict(PROVIDER_PRIVATEMODE_PROOF_CLAIMS)
    proof_claims.pop("manifest_log_digest", None)

    results = live_check.validate_privatemode_policy_proof_binding(
        policy,
        proof_claims,
        model_ids={"privatemode/kimi-latest"},
        label="privatemode-prod",
    )

    assert any(
        not result.ok
        and "privatemode-prod.confidentiality_policy manifest log digest must be a full sha256 digest"
        in result.message
        for result in results
    )


def test_live_check_rejects_expected_provider_missing_from_attested_policy() -> None:
    results = live_check.collect_results(
        status=_verified_privatemode_status(),
        attestation=_attestation_statement(),
        models=_verified_privatemode_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("privatemode", "privatemode/kimi-latest")],
    )

    assert any(
        not result.ok
        and result.message
        == "attestation routing policy includes privatemode for privatemode/kimi-latest"
        for result in results
    )


def test_live_check_rejects_non_expected_model_missing_from_attested_policy() -> None:
    status = _verified_status()
    extra_provider = json.loads(json.dumps(status["providers"][0]))  # type: ignore[index]
    extra_provider["upstream_name"] = "tinfoil-extra"
    extra_provider["confidentiality"]["model_ids"] = ["tinfoil/extra"]
    status["providers"].append(extra_provider)  # type: ignore[union-attr]
    models = _verified_models()
    extra_model = json.loads(json.dumps(models["data"][0]))  # type: ignore[index]
    extra_model["id"] = "tinfoil/extra"
    models["data"].append(extra_model)  # type: ignore[union-attr]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and result.message
        == "attestation routing policy includes tinfoil for tinfoil/extra"
        for result in results
    )


def test_live_check_rejects_unverified_provider() -> None:
    status = _verified_status()
    provider = status["providers"][0]  # type: ignore[index]
    provider["confidentiality"]["verified"] = False  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and "provider verifier reports verified" in result.message
        for result in results
    )


def test_live_check_rejects_missing_status_end_to_end_ready() -> None:
    status = _verified_status()
    del status["end_to_end_ready"]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and result.message == "/v1/confidentiality/status end_to_end_ready"
        for result in results
    )


def test_live_check_rejects_false_status_end_to_end_ready() -> None:
    status = _verified_status()
    status["end_to_end_ready"] = False

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and result.message == "/v1/confidentiality/status end_to_end_ready"
        for result in results
    )


def test_live_check_json_report_exposes_end_to_end_ready() -> None:
    report = live_check.json_report(
        [
            live_check.CheckResult(True, "first proof passed"),
            live_check.CheckResult(True, "second proof passed"),
        ]
    )

    assert report == {
        "confidential_routes_ready": False,
        "end_to_end_ready": True,
        "external_guardrails_ready": False,
        "routable_with_full_attestation": {
            "ppq-private": [],
            "privatemode": [],
            "tinfoil": [],
        },
        "external_guardrails": {},
        "public_surfaces": {},
        "inference_exercised": False,
        "expected_inference_route_count": 0,
        "inference_attempt_count": 0,
        "expected_inference_routes": [],
        "attempted_inference_routes": [],
        "checks": [
            {"ok": True, "message": "first proof passed"},
            {"ok": True, "message": "second proof passed"},
        ],
    }


def test_live_check_json_output_refuses_non_standard_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(live_check, "fetch_json", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        live_check,
        "fetch_bytes",
        lambda *_args, **_kwargs: (b"key-config", "application/ohttp-keys"),
    )
    monkeypatch.setattr(live_check, "public_surface_report", lambda **_kwargs: {})
    monkeypatch.setattr(
        live_check,
        "collect_results",
        lambda **_kwargs: [live_check.CheckResult(True, "proof passed")],
    )
    monkeypatch.setattr(
        live_check,
        "external_guardrail_report",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        live_check,
        "json_report",
        lambda *_args, **_kwargs: {"bad": float("nan")},
    )

    exit_code = live_check.main(
        [
            "https://routstr.example",
            "--expect-provider",
            "tinfoil:tinfoil/gpt-secure",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "failed to serialize live-check JSON report" in captured.err


def test_live_check_json_report_fails_closed_when_any_check_fails() -> None:
    report = live_check.json_report(
        [
            live_check.CheckResult(True, "public proof passed"),
            live_check.CheckResult(False, "provider proof failed"),
        ]
    )

    assert report["end_to_end_ready"] is False
    assert report["checks"][-1] == {
        "ok": False,
        "message": "provider proof failed",
    }


def test_live_check_json_report_exposes_inference_exercised() -> None:
    report = live_check.json_report(
        [live_check.CheckResult(True, "inference passed")],
        inference_exercised=True,
        expected_inference_route_count=2,
        inference_attempt_count=2,
        expected_inference_routes=[
            {
                "provider": "tinfoil",
                "model": "tinfoil/gpt-secure",
                "endpoint": "/v1/chat/completions",
            },
            {
                "provider": "privatemode",
                "model": "privatemode/kimi-latest",
                "endpoint": "/v1/messages",
            },
        ],
        attempted_inference_routes=[
            {
                "provider": "tinfoil",
                "model": "tinfoil/gpt-secure",
                "endpoint": "/v1/chat/completions",
            },
            {
                "provider": "privatemode",
                "model": "privatemode/kimi-latest",
                "endpoint": "/v1/messages",
            },
        ],
    )

    assert report["end_to_end_ready"] is True
    assert report["inference_exercised"] is True
    assert report["confidential_routes_ready"] is False
    assert report["expected_inference_route_count"] == 2
    assert report["inference_attempt_count"] == 2
    assert report["expected_inference_routes"] == report["attempted_inference_routes"]


def test_live_check_json_report_exposes_external_guardrails() -> None:
    report = live_check.json_report(
        [live_check.CheckResult(True, "guardrails passed")],
        external_guardrails={
            "strict_external_guardrails": True,
            "external_guardrails_ready": False,
            "provider_catalog_supplied": True,
            "tinfoil_directory_required": True,
            "tinfoil_results_verified": False,
            "tinfoil_required_hosts": [],
            "tinfoil_verified_hosts": [],
            "ppq_results_required": False,
            "ppq_results_verified": False,
            "ppq_required_hosts": [],
            "ppq_verified_hosts": [],
            "ppq_required_models": [],
            "ppq_verified_models": [],
            "privatemode_results_verified": False,
            "privatemode_required_hosts": [],
            "privatemode_verified_hosts": [],
            "attestation_targets_supplied": True,
            "attestation_results_supplied": True,
        },
    )

    assert report["external_guardrails"] == {
        "strict_external_guardrails": True,
        "external_guardrails_ready": False,
        "provider_catalog_supplied": True,
        "tinfoil_directory_required": True,
        "tinfoil_results_verified": False,
        "tinfoil_required_hosts": [],
        "tinfoil_verified_hosts": [],
        "ppq_results_required": False,
        "ppq_results_verified": False,
        "ppq_required_hosts": [],
        "ppq_verified_hosts": [],
        "ppq_required_models": [],
        "ppq_verified_models": [],
        "privatemode_results_verified": False,
        "privatemode_required_hosts": [],
        "privatemode_verified_hosts": [],
        "attestation_targets_supplied": True,
        "attestation_results_supplied": True,
    }
    assert report["external_guardrails_ready"] is False


def test_live_check_json_report_exposes_top_level_routable_model_sets() -> None:
    routable = {
        "ppq-private": ["private/gpt-oss-120b"],
        "privatemode": [],
        "tinfoil": ["tinfoil/gpt-secure"],
    }

    report = live_check.json_report(
        [live_check.CheckResult(True, "guardrails passed")],
        external_guardrails={
            "strict_external_guardrails": True,
            "external_guardrails_ready": True,
            "routable_with_full_attestation": routable,
        },
    )

    assert report["routable_with_full_attestation"] == routable


def test_live_check_json_report_exposes_top_level_ready_external_guardrails() -> None:
    report = live_check.json_report(
        [live_check.CheckResult(True, "guardrails passed")],
        external_guardrails={
            "strict_external_guardrails": True,
            "external_guardrails_ready": True,
            "provider_catalog_supplied": True,
            "tinfoil_directory_required": False,
            "tinfoil_results_verified": False,
            "tinfoil_required_hosts": [],
            "tinfoil_verified_hosts": [],
            "ppq_results_required": False,
            "ppq_results_verified": False,
            "ppq_required_hosts": [],
            "ppq_verified_hosts": [],
            "ppq_required_models": [],
            "ppq_verified_models": [],
            "privatemode_results_verified": False,
            "privatemode_required_hosts": [],
            "privatemode_verified_hosts": [],
            "attestation_targets_supplied": False,
            "attestation_results_supplied": False,
        },
    )

    assert report["external_guardrails_ready"] is True
    assert report["confidential_routes_ready"] is False


def test_live_check_json_report_marks_confidential_routes_ready() -> None:
    route = {
        "provider": "tinfoil",
        "model": "tinfoil/gpt-secure",
        "endpoint": "/v1/chat/completions",
    }

    report = live_check.json_report(
        [live_check.CheckResult(True, "all proof passed")],
        inference_exercised=True,
        expected_inference_route_count=1,
        inference_attempt_count=1,
        expected_inference_routes=[route],
        attempted_inference_routes=[route],
        external_guardrails={
            "strict_external_guardrails": True,
            "external_guardrails_ready": True,
            "provider_catalog_supplied": True,
            "tinfoil_directory_required": False,
            "tinfoil_results_verified": False,
            "tinfoil_required_hosts": [],
            "tinfoil_verified_hosts": [],
            "ppq_results_required": False,
            "ppq_results_verified": False,
            "ppq_required_hosts": [],
            "ppq_verified_hosts": [],
            "ppq_required_models": [],
            "ppq_verified_models": [],
            "privatemode_results_verified": False,
            "privatemode_required_hosts": [],
            "privatemode_verified_hosts": [],
            "attestation_targets_supplied": False,
            "attestation_results_supplied": False,
        },
    )

    assert report["confidential_routes_ready"] is True


def test_live_check_json_report_requires_strict_external_guardrails_for_readiness() -> None:
    route = {
        "provider": "tinfoil",
        "model": "tinfoil/gpt-secure",
        "endpoint": "/v1/chat/completions",
    }

    report = live_check.json_report(
        [live_check.CheckResult(True, "all proof passed")],
        inference_exercised=True,
        expected_inference_route_count=1,
        inference_attempt_count=1,
        expected_inference_routes=[route],
        attempted_inference_routes=[route],
        external_guardrails={
            "strict_external_guardrails": False,
            "external_guardrails_ready": True,
            "provider_catalog_supplied": True,
        },
    )

    assert report["confidential_routes_ready"] is False


def test_complete_inference_routes_adds_missing_expected_providers() -> None:
    routes = live_check.complete_inference_routes(
        [
            ("tinfoil", "tinfoil/gpt-secure"),
            ("ppq-private", "private/gpt-oss-120b"),
        ],
        [
            live_check.ExpectedInferenceRoute(
                provider="tinfoil",
                model="tinfoil/gpt-secure",
                endpoint="/v1/responses",
            )
        ],
    )

    assert routes == [
        live_check.ExpectedInferenceRoute(
            provider="tinfoil",
            model="tinfoil/gpt-secure",
            endpoint="/v1/responses",
        ),
        live_check.ExpectedInferenceRoute(
            provider="ppq-private",
            model="private/gpt-oss-120b",
            endpoint="/v1/chat/completions",
        ),
    ]


def test_complete_inference_routes_adds_exact_routable_map_entries() -> None:
    models = _verified_models()
    models["data"].append(_verified_ppq_private_models()["data"][0])  # type: ignore[index,union-attr]

    routes = live_check.complete_inference_routes(
        [("tinfoil", "tinfoil/gpt-secure")],
        [
            live_check.ExpectedInferenceRoute(
                provider="tinfoil",
                model="tinfoil/gpt-secure",
                endpoint="/v1/responses",
            )
        ],
        models=models,
        routable_with_full_attestation={
            "tinfoil": ["tinfoil/gpt-secure"],
            "ppq-private": ["private/gpt-oss-120b"],
            "privatemode": [],
        },
    )

    assert routes == [
        live_check.ExpectedInferenceRoute(
            provider="tinfoil",
            model="tinfoil/gpt-secure",
            endpoint="/v1/responses",
        ),
        live_check.ExpectedInferenceRoute(
            provider="ppq-private",
            model="private/gpt-oss-120b",
            endpoint="/v1/chat/completions",
        ),
    ]


def test_live_check_json_report_exposes_public_surfaces() -> None:
    public_surfaces = {
        "status": {
            "path": "/v1/confidentiality/status",
            "sha256": _json_digest(_verified_status()),
        }
    }

    report = live_check.json_report(
        [live_check.CheckResult(True, "surfaces passed")],
        public_surfaces=public_surfaces,
    )

    assert report["public_surfaces"] == public_surfaces


def test_live_check_main_json_output_has_top_level_readiness(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_fetch_json(base_url: str, path: str, *, timeout: float) -> object:
        if path == "/v1/confidentiality/status":
            return _verified_status()
        if path == "/.well-known/routstr-attestation":
            return _attestation_statement()
        if path == "/v1/models":
            return _verified_models()
        raise AssertionError(f"unexpected JSON path: {path}")

    def fake_fetch_bytes(
        base_url: str, path: str, *, timeout: float
    ) -> tuple[bytes, str]:
        assert path == "/.well-known/hpke-keys"
        return HPKE_KEY_CONFIG, "application/ohttp-keys"

    monkeypatch.setattr(live_check, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(live_check, "fetch_bytes", fake_fetch_bytes)

    result = live_check.main(
        [
            "https://routstr.example",
            "--expect-provider",
            "tinfoil:tinfoil/gpt-secure",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert result == 0
    assert report["end_to_end_ready"] is True
    assert report["inference_exercised"] is False
    assert report["expected_inference_route_count"] == 1
    assert report["inference_attempt_count"] == 0
    assert report["expected_inference_routes"] == [
        {
            "provider": "tinfoil",
            "model": "tinfoil/gpt-secure",
            "endpoint": "/v1/chat/completions",
        }
    ]
    assert report["attempted_inference_routes"] == []
    assert report["external_guardrails"] == {
        "strict_external_guardrails": False,
        "external_guardrails_ready": False,
        "provider_catalog_supplied": False,
        "tinfoil_directory_required": True,
        "tinfoil_results_verified": False,
        "tinfoil_required_hosts": ["inference.tinfoil.sh"],
        "tinfoil_verified_hosts": [],
        "tinfoil_required_models": ["tinfoil/gpt-secure"],
        "tinfoil_verified_models": [],
        "tinfoil_routable_with_full_attestation": [],
        "tinfoil_target_models_verified": False,
        "tinfoil_model_release_checks": {},
        "ppq_results_required": False,
        "ppq_attestation_targets_supplied": True,
        "ppq_results_verified": False,
        "ppq_required_hosts": [],
        "ppq_verified_hosts": [],
        "ppq_required_models": [],
        "ppq_verified_models": [],
        "ppq_routable_with_full_attestation": [],
        "ppq_target_models_verified": True,
        "ppq_model_release_checks": {},
        "privatemode_results_verified": False,
        "privatemode_attestation_targets_supplied": True,
        "privatemode_required_hosts": [],
        "privatemode_verified_hosts": [],
        "privatemode_required_models": [],
        "privatemode_verified_models": [],
        "privatemode_routable_with_full_attestation": [],
        "privatemode_target_models_verified": True,
        "privatemode_model_policy_checks": {},
        "attestation_targets_supplied": False,
        "attestation_results_supplied": False,
        "max_attestation_result_age_seconds": None,
        "routable_with_full_attestation": {
            "ppq-private": [],
            "privatemode": [],
            "tinfoil": [],
        },
    }
    assert report["public_surfaces"] == {
        "attestation": {
            "path": "/.well-known/routstr-attestation",
            "sha256": _json_digest(_attestation_statement()),
        },
        "hpke_keys": {
            "content_type": "application/ohttp-keys",
            "path": "/.well-known/hpke-keys",
            "sha256": HPKE_KEY_CONFIG_DIGEST,
        },
        "models": {
            "path": "/v1/models",
            "sha256": _json_digest(_verified_models()),
        },
        "status": {
            "path": "/v1/confidentiality/status",
            "sha256": _json_digest(_verified_status()),
        },
    }
    assert isinstance(report["checks"], list)
    assert all(check["ok"] is True for check in report["checks"])


def test_live_check_main_env_expected_provider_json_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_fetch_json(base_url: str, path: str, *, timeout: float) -> object:
        if path == "/v1/confidentiality/status":
            return _verified_status()
        if path == "/.well-known/routstr-attestation":
            return _attestation_statement()
        if path == "/v1/models":
            return _verified_models()
        raise AssertionError(f"unexpected JSON path: {path}")

    def fake_fetch_bytes(
        base_url: str, path: str, *, timeout: float
    ) -> tuple[bytes, str]:
        assert path == "/.well-known/hpke-keys"
        return HPKE_KEY_CONFIG, "application/ohttp-keys"

    monkeypatch.setenv("EXPECT_PROVIDER", "tinfoil:tinfoil/gpt-secure")
    monkeypatch.setenv("LIVE_CHECK_JSON", "1")
    monkeypatch.setattr(live_check, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(live_check, "fetch_bytes", fake_fetch_bytes)

    result = live_check.main(["https://routstr.example"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert result == 0
    assert report["expected_inference_routes"] == [
        {
            "provider": "tinfoil",
            "model": "tinfoil/gpt-secure",
            "endpoint": "/v1/chat/completions",
        }
    ]


def test_live_check_main_env_strict_confidential_routes_ready_requires_guardrails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_fetch_json(base_url: str, path: str, *, timeout: float) -> object:
        if path == "/v1/confidentiality/status":
            return _verified_status()
        if path == "/.well-known/routstr-attestation":
            return _attestation_statement()
        if path == "/v1/models":
            return _verified_models()
        raise AssertionError(f"unexpected JSON path: {path}")

    def fake_fetch_bytes(
        base_url: str, path: str, *, timeout: float
    ) -> tuple[bytes, str]:
        assert path == "/.well-known/hpke-keys"
        return HPKE_KEY_CONFIG, "application/ohttp-keys"

    monkeypatch.setenv("EXPECT_PROVIDER", "tinfoil:tinfoil/gpt-secure")
    monkeypatch.setenv("LIVE_CHECK_JSON", "1")
    monkeypatch.setenv("STRICT_CONFIDENTIAL_ROUTES_READY", "1")
    monkeypatch.setattr(live_check, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(live_check, "fetch_bytes", fake_fetch_bytes)

    result = live_check.main(["https://routstr.example"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert result == 1
    assert report["confidential_routes_ready"] is False
    assert report["checks"][-1] == {
        "ok": False,
        "message": "strict confidential route readiness requires --strict-external-guardrails",
    }


def test_live_check_cli_max_attestation_age_overrides_invalid_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_ATTESTATION_RESULT_AGE_SECONDS", "not-an-int")

    args = live_check.parse_args(
        [
            "https://routstr.example",
            "--expect-provider",
            "tinfoil:tinfoil/gpt-secure",
            "--max-attestation-result-age-seconds",
            "60",
        ]
    )

    assert args.max_attestation_result_age_seconds == 60


def test_live_check_main_strict_inference_exercised_rejects_surface_only_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch_json(base_url: str, path: str, *, timeout: float) -> object:
        if path == "/v1/confidentiality/status":
            return _verified_status()
        if path == "/.well-known/routstr-attestation":
            return _attestation_statement()
        if path == "/v1/models":
            return _verified_models()
        raise AssertionError(f"unexpected JSON path: {path}")

    def fake_fetch_bytes(
        base_url: str, path: str, *, timeout: float
    ) -> tuple[bytes, str]:
        assert path == "/.well-known/hpke-keys"
        return HPKE_KEY_CONFIG, "application/ohttp-keys"

    monkeypatch.setattr(live_check, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(live_check, "fetch_bytes", fake_fetch_bytes)

    result = live_check.main(
        [
            "https://routstr.example",
            "--expect-provider",
            "tinfoil:tinfoil/gpt-secure",
            "--strict-inference-exercised",
        ]
    )

    assert result == 1


def test_live_check_main_strict_external_guardrails_rejects_missing_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch_json(base_url: str, path: str, *, timeout: float) -> object:
        if path == "/v1/confidentiality/status":
            return _verified_status()
        if path == "/.well-known/routstr-attestation":
            return _attestation_statement()
        if path == "/v1/models":
            return _verified_models()
        raise AssertionError(f"unexpected JSON path: {path}")

    def fake_fetch_bytes(
        base_url: str, path: str, *, timeout: float
    ) -> tuple[bytes, str]:
        assert path == "/.well-known/hpke-keys"
        return HPKE_KEY_CONFIG, "application/ohttp-keys"

    monkeypatch.setattr(live_check, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(live_check, "fetch_bytes", fake_fetch_bytes)

    result = live_check.main(
        [
            "https://routstr.example",
            "--expect-provider",
            "tinfoil:tinfoil/gpt-secure",
            "--strict-external-guardrails",
        ]
    )

    assert result == 1


def test_live_check_strict_external_guardrails_help_names_all_provider_guardrails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        live_check.parse_args(["--help"])
    assert exc_info.value.code == 0

    help_text = capsys.readouterr().out
    assert "provider catalog checks" in help_text
    assert "Tinfoil directory results" in help_text
    assert "PPQ selected model results" in help_text
    assert "Privatemode evidence" in help_text


def test_live_check_main_strict_confidential_routes_ready_rejects_surface_only_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_fetch_json(base_url: str, path: str, *, timeout: float) -> object:
        if path == "/v1/confidentiality/status":
            return _verified_status()
        if path == "/.well-known/routstr-attestation":
            return _attestation_statement()
        if path == "/v1/models":
            return _verified_models()
        raise AssertionError(f"unexpected JSON path: {path}")

    def fake_fetch_bytes(
        base_url: str, path: str, *, timeout: float
    ) -> tuple[bytes, str]:
        assert path == "/.well-known/hpke-keys"
        return HPKE_KEY_CONFIG, "application/ohttp-keys"

    monkeypatch.setattr(live_check, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(live_check, "fetch_bytes", fake_fetch_bytes)

    result = live_check.main(
        [
            "https://routstr.example",
            "--expect-provider",
            "tinfoil:tinfoil/gpt-secure",
            "--strict-confidential-routes-ready",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert result == 1
    assert report["confidential_routes_ready"] is False
    assert report["checks"][-1] == {
        "ok": False,
        "message": "strict confidential route readiness requires --strict-external-guardrails",
    }


def test_live_check_main_strict_confidential_routes_ready_requires_strict_external_guardrails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_fetch_json(base_url: str, path: str, *, timeout: float) -> object:
        if path == "/v1/confidentiality/status":
            return _verified_status()
        if path == "/.well-known/routstr-attestation":
            return _attestation_statement()
        if path == "/v1/models":
            return _verified_models()
        raise AssertionError(f"unexpected JSON path: {path}")

    def fake_fetch_bytes(
        base_url: str, path: str, *, timeout: float
    ) -> tuple[bytes, str]:
        assert path == "/.well-known/hpke-keys"
        return HPKE_KEY_CONFIG, "application/ohttp-keys"

    def fake_post_json(*args: object, **kwargs: object) -> object:
        return {
            "id": "chatcmpl-live-check",
            "object": "chat.completion",
            "model": "tinfoil/gpt-secure",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        }

    def fake_external_guardrail_report(**kwargs: object) -> dict[str, object]:
        return {
            "strict_external_guardrails": True,
            "external_guardrails_ready": True,
            "provider_catalog_supplied": True,
            "tinfoil_directory_required": False,
            "tinfoil_results_verified": True,
            "tinfoil_required_hosts": ["inference.tinfoil.sh"],
            "tinfoil_verified_hosts": ["inference.tinfoil.sh"],
            "ppq_results_required": False,
            "ppq_results_verified": False,
            "ppq_required_hosts": [],
            "ppq_verified_hosts": [],
            "ppq_required_models": [],
            "ppq_verified_models": [],
            "privatemode_results_verified": False,
            "privatemode_required_hosts": [],
            "privatemode_verified_hosts": [],
            "attestation_targets_supplied": True,
            "attestation_results_supplied": True,
            "routable_with_full_attestation": {
                "tinfoil": ["tinfoil/gpt-secure"],
                "ppq-private": [],
                "privatemode": [],
            },
        }

    monkeypatch.setattr(live_check, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(live_check, "fetch_bytes", fake_fetch_bytes)
    monkeypatch.setattr(live_check, "post_json", fake_post_json)
    monkeypatch.setattr(
        live_check, "external_guardrail_report", fake_external_guardrail_report
    )
    monkeypatch.setattr(
        live_check,
        "strict_external_guardrail_checks",
        lambda **kwargs: [],
    )
    monkeypatch.setenv("ROUTSTR_API_KEY", "sk-live-secret")

    result = live_check.main(
        [
            "https://routstr.example",
            "--expect-provider",
            "tinfoil:tinfoil/gpt-secure",
            "--run-inference",
            "--strict-confidential-routes-ready",
            "--bearer-token-env",
            "ROUTSTR_API_KEY",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert result == 1
    assert report["confidential_routes_ready"] is False
    assert report["checks"][-1] == {
        "ok": False,
        "message": "strict confidential route readiness requires --strict-external-guardrails",
    }


def test_live_check_main_json_output_marks_successful_inference_exercised(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_fetch_json(base_url: str, path: str, *, timeout: float) -> object:
        if path == "/v1/confidentiality/status":
            return _verified_status()
        if path == "/.well-known/routstr-attestation":
            return _attestation_statement()
        if path == "/v1/models":
            return _verified_models()
        raise AssertionError(f"unexpected JSON path: {path}")

    def fake_fetch_bytes(
        base_url: str, path: str, *, timeout: float
    ) -> tuple[bytes, str]:
        assert path == "/.well-known/hpke-keys"
        return HPKE_KEY_CONFIG, "application/ohttp-keys"

    def fake_post_json(*args: object, **kwargs: object) -> object:
        return {
            "id": "chatcmpl-live-check",
            "object": "chat.completion",
            "model": "tinfoil/gpt-secure",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        }

    monkeypatch.setattr(live_check, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(live_check, "fetch_bytes", fake_fetch_bytes)
    monkeypatch.setattr(live_check, "post_json", fake_post_json)
    monkeypatch.setenv("ROUTSTR_API_KEY", "sk-live-secret")

    result = live_check.main(
        [
            "https://routstr.example",
            "--expect-provider",
            "tinfoil:tinfoil/gpt-secure",
            "--run-inference",
            "--strict-inference-exercised",
            "--bearer-token-env",
            "ROUTSTR_API_KEY",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert result == 0
    assert report["end_to_end_ready"] is True
    assert report["inference_exercised"] is True
    assert report["expected_inference_route_count"] == 1
    assert report["inference_attempt_count"] == 1
    assert report["expected_inference_routes"] == [
        {
            "provider": "tinfoil",
            "model": "tinfoil/gpt-secure",
            "endpoint": "/v1/chat/completions",
        }
    ]
    assert report["attempted_inference_routes"] == report["expected_inference_routes"]


def test_live_check_main_strict_confidential_routes_ready_accepts_full_gate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_fetch_json(base_url: str, path: str, *, timeout: float) -> object:
        if path == "/v1/confidentiality/status":
            return _verified_status()
        if path == "/.well-known/routstr-attestation":
            return _attestation_statement()
        if path == "/v1/models":
            return _verified_models()
        raise AssertionError(f"unexpected JSON path: {path}")

    def fake_fetch_bytes(
        base_url: str, path: str, *, timeout: float
    ) -> tuple[bytes, str]:
        assert path == "/.well-known/hpke-keys"
        return HPKE_KEY_CONFIG, "application/ohttp-keys"

    def fake_post_json(*args: object, **kwargs: object) -> object:
        return {
            "id": "chatcmpl-live-check",
            "object": "chat.completion",
            "model": "tinfoil/gpt-secure",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        }

    def fake_external_guardrail_report(**kwargs: object) -> dict[str, object]:
        return {
            "strict_external_guardrails": True,
            "external_guardrails_ready": True,
            "provider_catalog_supplied": True,
            "tinfoil_directory_required": False,
            "tinfoil_results_verified": True,
            "tinfoil_required_hosts": ["inference.tinfoil.sh"],
            "tinfoil_verified_hosts": ["inference.tinfoil.sh"],
            "ppq_results_required": False,
            "ppq_results_verified": False,
            "ppq_required_hosts": [],
            "ppq_verified_hosts": [],
            "ppq_required_models": [],
            "ppq_verified_models": [],
            "privatemode_results_verified": False,
            "privatemode_required_hosts": [],
            "privatemode_verified_hosts": [],
            "attestation_targets_supplied": True,
            "attestation_results_supplied": True,
            "routable_with_full_attestation": {
                "tinfoil": ["tinfoil/gpt-secure"],
                "ppq-private": [],
                "privatemode": [],
            },
        }

    monkeypatch.setattr(live_check, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(live_check, "fetch_bytes", fake_fetch_bytes)
    monkeypatch.setattr(live_check, "post_json", fake_post_json)
    monkeypatch.setattr(
        live_check, "external_guardrail_report", fake_external_guardrail_report
    )
    monkeypatch.setattr(
        live_check,
        "strict_external_guardrail_checks",
        lambda **kwargs: [],
    )
    monkeypatch.setenv("ROUTSTR_API_KEY", "sk-live-secret")

    result = live_check.main(
        [
            "https://routstr.example",
            "--expect-provider",
            "tinfoil:tinfoil/gpt-secure",
            "--run-inference",
            "--strict-external-guardrails",
            "--strict-confidential-routes-ready",
            "--bearer-token-env",
            "ROUTSTR_API_KEY",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert result == 0
    assert report["confidential_routes_ready"] is True


def test_live_check_main_strict_confidential_routes_ready_rejects_external_routable_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_fetch_json(base_url: str, path: str, *, timeout: float) -> object:
        if path == "/v1/confidentiality/status":
            return _verified_status()
        if path == "/.well-known/routstr-attestation":
            return _attestation_statement()
        if path == "/v1/models":
            return _verified_models()
        raise AssertionError(f"unexpected JSON path: {path}")

    def fake_fetch_bytes(
        base_url: str, path: str, *, timeout: float
    ) -> tuple[bytes, str]:
        assert path == "/.well-known/hpke-keys"
        return HPKE_KEY_CONFIG, "application/ohttp-keys"

    def fake_post_json(*args: object, **kwargs: object) -> object:
        return {
            "id": "chatcmpl-live-check",
            "object": "chat.completion",
            "model": "tinfoil/gpt-secure",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        }

    monkeypatch.setattr(live_check, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(live_check, "fetch_bytes", fake_fetch_bytes)
    monkeypatch.setattr(live_check, "post_json", fake_post_json)
    monkeypatch.setattr(
        live_check,
        "external_guardrail_report",
        lambda **_kwargs: {
            "strict_external_guardrails": True,
            "external_guardrails_ready": True,
            "routable_with_full_attestation": {
                "tinfoil": [],
                "ppq-private": [],
                "privatemode": [],
            },
        },
    )
    monkeypatch.setattr(
        live_check,
        "strict_external_guardrail_checks",
        lambda **_kwargs: [],
    )
    monkeypatch.setenv("ROUTSTR_API_KEY", "sk-live-secret")

    result = live_check.main(
        [
            "https://routstr.example",
            "--expect-provider",
            "tinfoil:tinfoil/gpt-secure",
            "--run-inference",
            "--strict-external-guardrails",
            "--strict-confidential-routes-ready",
            "--bearer-token-env",
            "ROUTSTR_API_KEY",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert result == 1
    assert report["confidential_routes_ready"] is False
    assert any(
        not check["ok"]
        and check["message"]
        == "external guardrail routable_with_full_attestation matches public status"
        for check in report["checks"]
    )


def test_live_check_strict_confidential_routes_ready_exercises_every_expected_provider(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = _verified_status()
    status["providers"].append(_verified_ppq_private_status()["providers"][0])  # type: ignore[index,union-attr]
    models = _verified_models()
    models["data"].append(_verified_ppq_private_models()["data"][0])  # type: ignore[index,union-attr]
    routing_policy = json.loads(json.dumps(ROUTING_POLICY))
    routing_policy["providers"].append(  # type: ignore[index,union-attr]
        _ppq_private_routing_policy()["providers"][0]
    )
    _set_routstr_tee_policy_binding(status["routstr_tee"], routing_policy)  # type: ignore[arg-type,index]
    _set_routstr_tee_policy_binding(models["routstr_tee"], routing_policy)  # type: ignore[arg-type,index]

    def fake_fetch_json(base_url: str, path: str, *, timeout: float) -> object:
        if path == "/v1/confidentiality/status":
            return status
        if path == "/.well-known/routstr-attestation":
            return _attestation_statement(routing_policy)
        if path == "/v1/models":
            return models
        raise AssertionError(f"unexpected JSON path: {path}")

    def fake_fetch_bytes(
        base_url: str, path: str, *, timeout: float
    ) -> tuple[bytes, str]:
        assert path == "/.well-known/hpke-keys"
        return HPKE_KEY_CONFIG, "application/ohttp-keys"

    def fake_post_json(*args: object, **kwargs: object) -> object:
        return {
            "id": "chatcmpl-live-check",
            "object": "chat.completion",
            "model": "tinfoil/gpt-secure",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        }

    monkeypatch.setattr(live_check, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(live_check, "fetch_bytes", fake_fetch_bytes)
    monkeypatch.setattr(live_check, "post_json", fake_post_json)
    monkeypatch.setattr(
        live_check,
        "external_guardrail_report",
        lambda **_kwargs: {
            "strict_external_guardrails": True,
            "external_guardrails_ready": True,
            "provider_catalog_supplied": True,
        },
    )
    monkeypatch.setattr(
        live_check,
        "strict_external_guardrail_checks",
        lambda **_kwargs: [],
    )
    monkeypatch.setenv("ROUTSTR_API_KEY", "sk-live-secret")

    result = live_check.main(
        [
            "https://routstr.example",
            "--expect-provider",
            "tinfoil:tinfoil/gpt-secure",
            "--expect-provider",
            "ppq-private:private/gpt-oss-120b",
            "--expect-inference",
            "tinfoil:tinfoil/gpt-secure:chat-completions",
            "--run-inference",
            "--strict-external-guardrails",
            "--strict-confidential-routes-ready",
            "--bearer-token-env",
            "ROUTSTR_API_KEY",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert result == 1
    assert report["confidential_routes_ready"] is False
    assert report["expected_inference_routes"] == [
        {
            "provider": "tinfoil",
            "model": "tinfoil/gpt-secure",
            "endpoint": "/v1/chat/completions",
        },
        {
            "provider": "ppq-private",
            "model": "private/gpt-oss-120b",
            "endpoint": "/v1/chat/completions",
        },
    ]
    assert report["attempted_inference_routes"] == []


def test_live_check_rejects_truthy_string_provider_verified() -> None:
    status = _verified_status()
    provider = status["providers"][0]  # type: ignore[index]
    provider["confidentiality"]["verified"] = "false"  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and "provider verifier reports verified" in result.message
        for result in results
    )


def test_live_check_rejects_provider_without_future_expiry() -> None:
    status = _verified_status()
    provider = status["providers"][0]  # type: ignore[index]
    provider["confidentiality"]["expires_at"] = 1  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and "provider verifier expiry is in the future" in result.message
        for result in results
    )


def test_live_check_rejects_string_provider_expiry() -> None:
    status = _verified_status()
    provider = status["providers"][0]  # type: ignore[index]
    provider["confidentiality"]["expires_at"] = "4102444800"  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "provider verifier expiry must be an integer Unix timestamp"
        in result.message
        for result in results
    )


def test_live_check_rejects_string_provider_verified_at() -> None:
    status = _verified_status()
    provider = status["providers"][0]  # type: ignore[index]
    provider["confidentiality"]["verified_at"] = "1700000000"  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "provider verifier verified_at must be an integer Unix timestamp"
        in result.message
        for result in results
    )


def test_live_check_rejects_future_provider_verified_at() -> None:
    status = _verified_status()
    provider = status["providers"][0]  # type: ignore[index]
    provider["confidentiality"]["verified_at"] = int(time.time()) + 60  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "provider verifier verified_at is not in the future" in result.message
        for result in results
    )


def test_live_check_rejects_placeholder_public_digest() -> None:
    status = _verified_status()
    provider = status["providers"][0]  # type: ignore[index]
    provider["confidentiality"]["policy_digest"] = "sha256:not-a-real-digest"  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and "policy digest must be a full sha256 digest" in result.message
        for result in results
    )


def test_live_check_rejects_template_repeated_hex_public_digest() -> None:
    status = _verified_status()
    provider = status["providers"][0]  # type: ignore[index]
    confidentiality = provider["confidentiality"]  # type: ignore[index]
    confidentiality["policy_digest"] = "sha256:" + ("a" * 64)  # type: ignore[index]
    proof_claims = confidentiality["proof_claims"]  # type: ignore[index]
    proof_claims["payload_policy_digest"] = confidentiality["policy_digest"]  # type: ignore[index]

    results = live_check.validate_expected_provider(
        status,
        expected_provider="tinfoil",
        expected_model="tinfoil/gpt-secure",
    )

    assert any(
        not result.ok and "policy digest must not be a template placeholder" in result.message
        for result in results
    )


def test_live_check_rejects_verified_provider_status_without_verifier() -> None:
    status = _verified_status()
    provider = status["providers"][0]  # type: ignore[index]
    del provider["confidentiality"]["verifier"]  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and "tinfoil-prod provider verifier" in result.message
        for result in results
    )


def test_live_check_rejects_expected_provider_without_public_proof_claims() -> None:
    status = _verified_status()
    confidentiality = status["providers"][0]["confidentiality"]  # type: ignore[index]
    del confidentiality["proof_claims"]  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "tinfoil/tinfoil/gpt-secure provider proof_claims are present"
        in result.message
        for result in results
    )


def test_live_check_rejects_tinfoil_without_model_attestation_proof_claims() -> None:
    status = _verified_status()
    proof_claims = status["providers"][0]["confidentiality"]["proof_claims"]  # type: ignore[index]
    proof_claims.pop("model_attestations", None)  # type: ignore[union-attr]
    models = _verified_models()
    model_proof_claims = models["data"][0]["confidentiality"]["proof_claims"]  # type: ignore[index]
    model_proof_claims.pop("model_attestations", None)  # type: ignore[union-attr]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "tinfoil/tinfoil/gpt-secure provider proof_claims.model_attestations contains selected model"
        in result.message
        for result in results
    )


def test_live_check_rejects_provider_status_policy_release_mismatch() -> None:
    status = _verified_status()
    status["providers"][0]["confidentiality_policy"] = {  # type: ignore[index]
        "require_model_attestations": True,
        "model_attestation_targets": {
            "tinfoil/gpt-secure": {
                "repo": "tinfoilsh/confidential-gpt-secure",
                "expected_release_digest": "sha256:" + ("a" * 64),
            }
        },
    }

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and (
            "tinfoil-prod.confidentiality_policy target tinfoil/gpt-secure "
            "release digest matches model attestation"
        )
        in result.message
        for result in results
    )


def test_live_check_requires_tinfoil_provider_status_policy() -> None:
    status = _verified_status()
    status["providers"][0].pop("confidentiality_policy", None)  # type: ignore[index]

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and "tinfoil-prod.confidentiality_policy is required for verified tinfoil"
        in result.message
        for result in results
    )


def test_live_check_rejects_tinfoil_router_policy_release_mismatch() -> None:
    status = _verified_status()
    status["providers"][0]["confidentiality_policy"] = {  # type: ignore[index]
        **TINFOIL_CONFIDENTIALITY_POLICY,
        "expected_release_digest": "sha256:" + ("a" * 64),
    }

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and (
            "tinfoil-prod.confidentiality_policy release digest "
            "matches provider proof"
        )
        in result.message
        for result in results
    )


def test_live_check_requires_tinfoil_router_policy_repo_pin() -> None:
    status = _verified_status()
    policy = dict(TINFOIL_CONFIDENTIALITY_POLICY)
    policy.pop("repo")
    status["providers"][0]["confidentiality_policy"] = policy  # type: ignore[index]

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and "tinfoil-prod.confidentiality_policy repo is pinned" in result.message
        for result in results
    )


def test_live_check_requires_tinfoil_router_policy_release_pin() -> None:
    status = _verified_status()
    policy = dict(TINFOIL_CONFIDENTIALITY_POLICY)
    policy.pop("expected_release_digest")
    status["providers"][0]["confidentiality_policy"] = policy  # type: ignore[index]

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and "tinfoil-prod.confidentiality_policy release digest is pinned"
        in result.message
        for result in results
    )


def test_live_check_requires_tinfoil_model_target_repo_pin() -> None:
    status = _verified_status()
    policy = json.loads(json.dumps(TINFOIL_CONFIDENTIALITY_POLICY))
    policy["model_attestation_targets"]["tinfoil/gpt-secure"].pop("repo")
    status["providers"][0]["confidentiality_policy"] = policy  # type: ignore[index]

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and (
            "tinfoil-prod.confidentiality_policy target tinfoil/gpt-secure "
            "repo is pinned"
        )
        in result.message
        for result in results
    )


def test_live_check_requires_tinfoil_model_target_artifact_pin() -> None:
    status = _verified_status()
    policy = json.loads(json.dumps(TINFOIL_CONFIDENTIALITY_POLICY))
    policy["model_attestation_targets"]["tinfoil/gpt-secure"].pop(
        "expected_release_digest"
    )
    status["providers"][0]["confidentiality_policy"] = policy  # type: ignore[index]

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and (
            "tinfoil-prod.confidentiality_policy target tinfoil/gpt-secure "
            "artifact identity is pinned"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_duplicate_tinfoil_model_attestation_proof_keys() -> None:
    proof_claims = dict(PROVIDER_EHBP_PROOF_CLAIMS)
    proof_claims["model_attestations"] = dict(PROVIDER_TINFOIL_MODEL_ATTESTATIONS)
    proof_claims["model_attestations"][" tinfoil/gpt-secure "] = (  # type: ignore[index]
        PROVIDER_TINFOIL_MODEL_ATTESTATIONS["tinfoil/gpt-secure"]
    )

    results = live_check.validate_provider_proof_claims(
        {
            "verified": True,
            "mode": "tinfoil",
            "model_ids": ["tinfoil/gpt-secure"],
            "proof_claims": proof_claims,
        },
        label="tinfoil-prod",
    )

    assert any(
        not result.ok
        and (
            "tinfoil-prod provider proof_claims.model_attestations keys "
            "are unique after trimming"
        )
        in result.message
        for result in results
    )


@pytest.mark.parametrize("mode", ["tinfoil", "ppq-private-tee", "privatemode"])
def test_live_check_requires_provider_payload_digest_bindings(mode: str) -> None:
    if mode == "privatemode":
        proof_claims = dict(PROVIDER_PRIVATEMODE_PROOF_CLAIMS)
        model_ids = ["privatemode/kimi-latest"]
    elif mode == "ppq-private-tee":
        proof_claims = dict(PROVIDER_PPQ_PRIVATE_PROOF_CLAIMS)
        model_ids = ["private/gpt-oss-120b"]
    else:
        proof_claims = dict(PROVIDER_EHBP_PROOF_CLAIMS)
        model_ids = ["tinfoil/gpt-secure"]
    proof_claims.pop("payload_policy_digest")

    results = live_check.validate_provider_proof_claims(
        {
            "verified": True,
            "mode": mode,
            "policy_digest": PROVIDER_POLICY_DIGEST,
            "model_ids": model_ids,
            "proof_claims": proof_claims,
        },
        label="provider",
    )

    assert any(
        not result.ok
        and "provider proof_claims.payload_policy_digest must be a full sha256 digest"
        in result.message
        for result in results
    )


def test_live_check_rejects_provider_payload_policy_digest_mismatch() -> None:
    proof_claims = dict(PROVIDER_EHBP_PROOF_CLAIMS)
    proof_claims["payload_policy_digest"] = _digest("wrong-provider-policy")

    results = live_check.validate_provider_proof_claims(
        {
            "verified": True,
            "mode": "tinfoil",
            "policy_digest": PROVIDER_POLICY_DIGEST,
            "model_ids": ["tinfoil/gpt-secure"],
            "proof_claims": proof_claims,
        },
        label="provider",
    )

    assert any(
        not result.ok
        and "provider proof_claims.payload_policy_digest matches provider policy digest"
        in result.message
        for result in results
    )


def test_live_check_rejects_ppq_status_policy_bundle_mismatch() -> None:
    status = _verified_ppq_private_status()
    status["providers"][0]["confidentiality_policy"] = {  # type: ignore[index]
        "repo": "ppq-ai/private-tee",
        "attestation_bundle_url_digest": _digest("wrong-ppq-private-bundle-url"),
    }

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and (
            "ppq-private-prod.confidentiality_policy "
            "attestation bundle URL digest matches provider proof"
        )
        in result.message
        for result in results
    )


def test_live_check_requires_ppq_status_policy_bundle_digest_pin() -> None:
    status = _verified_ppq_private_status()
    status["providers"][0]["confidentiality_policy"] = {  # type: ignore[index]
        "repo": "ppq-ai/private-tee",
        "expected_code_measurement_fingerprint": "sha256:" + ("2" * 64),
    }

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and (
            "ppq-private-prod.confidentiality_policy attestation bundle URL "
            "digest is pinned"
        )
        in result.message
        for result in results
    )


def test_live_check_requires_ppq_status_policy_artifact_identity_pin() -> None:
    status = _verified_ppq_private_status()
    status["providers"][0]["confidentiality_policy"] = {  # type: ignore[index]
        "repo": "ppq-ai/private-tee",
        "attestation_bundle_url_digest": _digest("ppq-private-bundle-url"),
    }

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and (
            "ppq-private-prod.confidentiality_policy release digest or code "
            "measurement is pinned"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_ppq_status_backend_model_release_mismatch() -> None:
    status = _verified_ppq_private_status()
    policy = dict(PPQ_PRIVATE_CONFIDENTIALITY_POLICY)
    policy["model_attestation_targets"] = {
        "private/gpt-oss-120b": {
            "repo": "tinfoilsh/confidential-gpt-oss-120b",
            "expected_release_digest": _digest("wrong-ppq-backend-model-release"),
        }
    }
    status["providers"][0]["confidentiality_policy"] = policy  # type: ignore[index]

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and (
            "ppq-private-prod.confidentiality_policy target private/gpt-oss-120b "
            "release digest matches model attestation"
        )
        in result.message
        for result in results
    )


def test_live_check_requires_ppq_provider_status_policy() -> None:
    status = _verified_ppq_private_status()
    status["providers"][0].pop("confidentiality_policy", None)  # type: ignore[index]

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and (
            "ppq-private-prod.confidentiality_policy is required for verified "
            "ppq-private-tee"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_non_boolean_provider_verification_step() -> None:
    status = json.loads(json.dumps(_verified_ppq_private_status()))
    proof_claims = status["providers"][0]["confidentiality"]["proof_claims"]  # type: ignore[index]
    verification_steps = proof_claims["verification_steps"]
    verification_steps["diagnostic"] = "SECRET_PROMPT"

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and "ppq-private-prod provider proof_claims.verification_steps values are JSON booleans"
        in result.message
        for result in results
    )


def test_live_check_rejects_privatemode_status_policy_workload_mismatch() -> None:
    status = _verified_privatemode_status()
    status["providers"][0]["confidentiality_policy"] = {  # type: ignore[index]
        "expected_workload_identity_digest": _digest("wrong-workload"),
    }

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and (
            "privatemode-prod.confidentiality_policy "
            "expected workload identity digest matches provider proof"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_privatemode_policy_without_app_e2ee_trust_tier() -> None:
    status = _verified_privatemode_status()
    status["providers"][0]["confidentiality_policy"].pop(  # type: ignore[index]
        "expected_trust_tier",
        None,
    )

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and (
            "privatemode-prod.confidentiality_policy trust tier matches "
            "provider proof"
        )
        in result.message
        for result in results
    )


def test_live_check_requires_privatemode_provider_status_policy() -> None:
    status = _verified_privatemode_status()
    status["providers"][0].pop("confidentiality_policy", None)  # type: ignore[index]

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and (
            "privatemode-prod.confidentiality_policy is required for verified "
            "privatemode"
        )
        in result.message
        for result in results
    )


def test_live_check_requires_privatemode_policy_proxy_binary_pin() -> None:
    status = _verified_privatemode_status()
    status["providers"][0]["confidentiality_policy"].pop(  # type: ignore[index]
        "proxy_binary_digest",
        None,
    )

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and (
            "privatemode-prod.confidentiality_policy proxy binary digest "
            "is pinned"
        )
        in result.message
        for result in results
    )


def test_live_check_ignores_ppq_policy_legacy_proxy_digest_alias() -> None:
    status = _verified_ppq_private_status()
    policy = dict(PPQ_PRIVATE_CONFIDENTIALITY_POLICY)
    policy["proxyBinaryDigest"] = "sha256:" + ("g" * 64)
    status["providers"][0]["confidentiality_policy"] = policy  # type: ignore[index]

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert all(result.ok for result in results)


def test_live_check_rejects_privatemode_policy_malformed_proxy_digest_alias() -> None:
    status = _verified_privatemode_status()
    status["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "proxyBinaryDigest"
    ] = "sha256:" + ("g" * 64)

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and (
            "privatemode-prod.confidentiality_policy proxy binary digest "
            "aliases are valid and consistent"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_duplicate_privatemode_workload_binding_keys() -> None:
    policy = {
        "expected_workload_sans": [
            "secure-model.default.svc.cluster.local",
            "other.default.svc.cluster.local",
        ],
        "model_workload_bindings": {
            "privatemode/kimi-latest": {
                "workload_sans": ["secure-model.default.svc.cluster.local"],
            },
            " privatemode/kimi-latest ": {
                "workload_sans": ["other.default.svc.cluster.local"],
            },
        },
    }
    overwritten_binding_digest = _json_digest(
        {
            "privatemode/kimi-latest": {
                "workload_ids": [],
                "workload_sans": ["other.default.svc.cluster.local"],
            }
        }
    )
    proof_claims = dict(PROVIDER_PRIVATEMODE_PROOF_CLAIMS)
    proof_claims["model_workload_binding_digest"] = overwritten_binding_digest

    results = live_check.validate_privatemode_policy_proof_binding(
        policy,
        proof_claims,
        model_ids={"privatemode/kimi-latest"},
        label="privatemode-prod",
    )

    assert any(
        not result.ok
        and (
            "privatemode-prod.confidentiality_policy model workload bindings "
            "are valid"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_unbound_privatemode_expected_workload() -> None:
    policy = {
        "expected_workload_sans": [
            "secure-model.default.svc.cluster.local",
            "other.default.svc.cluster.local",
        ],
        "model_workload_bindings": {
            "privatemode/kimi-latest": {
                "workload_sans": ["secure-model.default.svc.cluster.local"],
            },
        },
    }
    proof_claims = dict(PROVIDER_PRIVATEMODE_PROOF_CLAIMS)
    proof_claims["expected_workload_identity_digest"] = _json_digest(
        {
            "ids": [],
            "sans": [
                "other.default.svc.cluster.local",
                "secure-model.default.svc.cluster.local",
            ],
        }
    )
    proof_claims["model_workload_binding_digest"] = _json_digest(
        {
            "privatemode/kimi-latest": {
                "workload_ids": [],
                "workload_sans": ["secure-model.default.svc.cluster.local"],
            },
        }
    )
    proof_claims["key_release_binding"] = _privatemode_key_release_binding_digest(
        proof_claims
    )

    results = live_check.validate_privatemode_policy_proof_binding(
        policy,
        proof_claims,
        model_ids={"privatemode/kimi-latest"},
        label="privatemode-prod",
    )

    assert any(
        not result.ok
        and (
            "privatemode-prod.confidentiality_policy expected workloads are bound "
            "to selected models"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_privatemode_scalar_workload_policy_shape() -> None:
    status = _verified_privatemode_status()
    workload_san = "secure-model.default.svc.cluster.local"
    scalar_policy = {
        "expected_workload_sans": workload_san,
        "model_workload_bindings": {
            "privatemode/kimi-latest": {
                "workload_sans": workload_san,
            }
        },
    }
    proof_claims = status["providers"][0]["confidentiality"]["proof_claims"]  # type: ignore[index]
    proof_claims["expected_workload_identity_digest"] = _json_digest(  # type: ignore[index]
        {"ids": [], "sans": [workload_san]}
    )
    proof_claims["model_workload_binding_digest"] = _json_digest(  # type: ignore[index]
        {
            "privatemode/kimi-latest": {
                "workload_ids": [],
                "workload_sans": [workload_san],
            }
        }
    )
    proof_claims["key_release_binding"] = _privatemode_key_release_binding_digest(  # type: ignore[index]
        proof_claims
    )
    status["providers"][0]["confidentiality_policy"] = scalar_policy  # type: ignore[index]

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and (
            "privatemode-prod.confidentiality_policy "
            "expected workload identity digest matches provider proof"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_verified_known_provider_prefix_selectors() -> None:
    status = _verified_status()
    confidentiality = status["providers"][0]["confidentiality"]  # type: ignore[index]
    confidentiality["model_id_prefixes"] = ["tinfoil/"]  # type: ignore[index]

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and "tinfoil-prod provider verified selectors are exact for tinfoil"
        in result.message
        for result in results
    )


def test_live_check_rejects_unprefixed_privatemode_model_ids() -> None:
    status = _verified_privatemode_status()
    confidentiality = status["providers"][0]["confidentiality"]  # type: ignore[index]
    confidentiality["model_ids"] = ["kimi-latest"]  # type: ignore[index]
    proof_claims = confidentiality["proof_claims"]  # type: ignore[index]
    proof_claims["selected_model_ids"] = ["kimi-latest"]

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and "privatemode-prod provider verified selectors are exact for privatemode"
        in result.message
        for result in results
    )


def test_live_check_rejects_malformed_verified_known_provider_selectors() -> None:
    status = _verified_status()
    confidentiality = status["providers"][0]["confidentiality"]  # type: ignore[index]
    confidentiality["model_ids"] = ["tinfoil/gpt-secure", 123]  # type: ignore[index]

    results = live_check.validate_provider_confidentiality_statuses(status)

    assert any(
        not result.ok
        and "tinfoil-prod provider verified selectors are exact for tinfoil"
        in result.message
        for result in results
    )


def test_live_check_provider_matching_ignores_known_provider_prefix_selectors() -> None:
    provider = _verified_status()["providers"][0]  # type: ignore[index]
    confidentiality = provider["confidentiality"]
    confidentiality["model_ids"] = []  # type: ignore[index]
    confidentiality["model_id_prefixes"] = ["tinfoil/"]  # type: ignore[index]

    assert (
        live_check.provider_matches(
            provider,  # type: ignore[arg-type]
            expected_provider="tinfoil",
            expected_model="tinfoil/gpt-secure",
        )
        is False
    )


def test_live_check_provider_matching_rejects_malformed_known_provider_selectors() -> None:
    provider = _verified_status()["providers"][0]  # type: ignore[index]
    confidentiality = provider["confidentiality"]
    confidentiality["model_ids"] = ["tinfoil/gpt-secure", 123]  # type: ignore[index]

    assert (
        live_check.provider_matches(
            provider,  # type: ignore[arg-type]
            expected_provider="tinfoil",
            expected_model="tinfoil/gpt-secure",
        )
        is False
    )


def test_live_check_rejects_attested_known_provider_prefix_selectors() -> None:
    attestation = _attestation_statement()
    routing_policy = attestation["routing_policy"]
    confidentiality = routing_policy["providers"][0]["confidentiality"]  # type: ignore[index]
    confidentiality["model_id_prefixes"] = ["tinfoil/"]  # type: ignore[index]
    attestation["routing_policy_digest"] = _json_digest(routing_policy)
    local_verification = attestation["tee"]["local_verification"]  # type: ignore[index]
    local_verification["proof_claims"]["routstr_config_measurement"] = attestation[  # type: ignore[index]
        "routing_policy_digest"
    ]

    results = live_check.validate_attestation_routing_policy(attestation)

    assert any(
        not result.ok
        and "attestation routing_policy.providers[0].confidentiality verified selectors are exact for tinfoil"
        in result.message
        for result in results
    )


def test_live_check_rejects_attested_tinfoil_policy_release_mismatch() -> None:
    routing_policy = json.loads(json.dumps(ROUTING_POLICY))
    routing_policy["providers"][0]["confidentiality_policy"] = {  # type: ignore[index]
        "require_model_attestations": True,
        "model_attestation_targets": {
            "tinfoil/gpt-secure": {
                "repo": "tinfoilsh/confidential-gpt-secure",
                "expected_release_digest": "sha256:" + ("a" * 64),
            }
        },
    }
    attestation = _attestation_statement(routing_policy)

    results = live_check.validate_attestation_routing_policy(attestation)

    assert any(
        not result.ok
        and (
            "attestation routing_policy.providers[0].confidentiality_policy "
            "target tinfoil/gpt-secure release digest matches model attestation"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_attested_tinfoil_allowed_model_measurement_mismatch() -> (
    None
):
    routing_policy = json.loads(json.dumps(ROUTING_POLICY))
    routing_policy["providers"][0]["confidentiality_policy"] = {  # type: ignore[index]
        "require_model_attestations": True,
        "model_attestation_targets": {
            "tinfoil/gpt-secure": {
                "repo": "tinfoilsh/confidential-gpt-secure",
                "expected_release_digest": PROVIDER_TINFOIL_MODEL_ATTESTATIONS[
                    "tinfoil/gpt-secure"
                ]["release_digest"],
                "allowed_code_measurement_fingerprints": [
                    "sha256:" + ("a" * 64)
                ],
            }
        },
    }
    attestation = _attestation_statement(routing_policy)

    results = live_check.validate_attestation_routing_policy(attestation)

    assert any(
        not result.ok
        and (
            "attestation routing_policy.providers[0].confidentiality_policy "
            "target tinfoil/gpt-secure allowed code measurements include "
            "model attestation"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_attested_ppq_private_policy_release_mismatch() -> None:
    routing_policy = _ppq_private_routing_policy()
    routing_policy["providers"][0]["confidentiality_policy"] = {  # type: ignore[index]
        "repo": "ppq-ai/private-tee",
        "expected_release_digest": "sha256:" + ("a" * 64),
        "attestation_bundle_url_digest": _digest("ppq-private-bundle-url"),
    }
    attestation = _attestation_statement(routing_policy)

    results = live_check.validate_attestation_routing_policy(attestation)

    assert any(
        not result.ok
        and (
            "attestation routing_policy.providers[0].confidentiality_policy "
            "release digest matches provider proof"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_attested_privatemode_policy_manifest_mismatch() -> None:
    routing_policy = _privatemode_routing_policy()
    routing_policy["providers"][0]["confidentiality_policy"] = {  # type: ignore[index]
        "manifest_digest": _digest("different-privatemode-manifest"),
        "proxy_image_digest": PRIVATEMODE_PROXY_IMAGE_DIGEST,
        "expected_workload_identity_digest": PROVIDER_PRIVATEMODE_PROOF_CLAIMS[
            "expected_workload_identity_digest"
        ],
        "model_workload_binding_digest": PROVIDER_PRIVATEMODE_PROOF_CLAIMS[
            "model_workload_binding_digest"
        ],
    }
    attestation = _attestation_statement(routing_policy)

    results = live_check.validate_attestation_routing_policy(attestation)

    assert any(
        not result.ok
        and (
            "attestation routing_policy.providers[0].confidentiality_policy "
            "manifest digest matches provider proof"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_attested_privatemode_raw_workload_policy_mismatch() -> None:
    routing_policy = _privatemode_routing_policy()
    routing_policy["providers"][0]["confidentiality_policy"] = {  # type: ignore[index]
        "expected_workload_sans": ["wrong.backend.privatemode.example"],
        "model_workload_bindings": {
            "privatemode/kimi-latest": {
                "workload_sans": ["wrong.backend.privatemode.example"],
            }
        },
    }
    attestation = _attestation_statement(routing_policy)

    results = live_check.validate_attestation_routing_policy(attestation)

    assert any(
        not result.ok
        and (
            "attestation routing_policy.providers[0].confidentiality_policy "
            "expected workload identity digest matches provider proof"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_unknown_provider_type_for_verified_mode() -> None:
    status = _verified_status()
    provider = status["providers"][0]  # type: ignore[index]
    provider["provider_type"] = "custom"  # type: ignore[index]
    provider["upstream_name"] = "custom-prod"  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "provider confidentiality mode tinfoil matches provider_type custom"
        in result.message
        for result in results
    )


def test_live_check_rejects_truthy_verified_on_non_expected_provider() -> None:
    status = _verified_status()
    extra_provider = json.loads(json.dumps(status["providers"][0]))  # type: ignore[index]
    extra_provider["upstream_name"] = "tinfoil-extra"
    extra_provider["confidentiality"]["verified"] = "true"
    extra_provider["confidentiality"]["model_ids"] = ["tinfoil/extra"]
    status["providers"].append(extra_provider)  # type: ignore[union-attr]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "tinfoil-extra provider confidentiality.verified is a JSON boolean"
        in result.message
        for result in results
    )


def test_live_check_rejects_malformed_non_expected_verified_provider_proof() -> None:
    status = _verified_status()
    extra_provider = json.loads(json.dumps(status["providers"][0]))  # type: ignore[index]
    extra_provider["upstream_name"] = "tinfoil-extra"
    extra_provider["confidentiality"]["model_ids"] = ["tinfoil/extra"]
    proof_claims = extra_provider["confidentiality"]["proof_claims"]
    del proof_claims["tls_public_key_fingerprint_sha256"]
    proof_claims["tls_public_key"] = "not-a-digest"
    status["providers"].append(extra_provider)  # type: ignore[union-attr]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "tinfoil-extra provider proof_claims.tls_public_key must be a full sha256 digest"
        in result.message
        for result in results
    )


def test_live_check_rejects_expected_tinfoil_provider_without_hpke_key_proof() -> None:
    status = _verified_status()
    proof_claims = status["providers"][0]["confidentiality"]["proof_claims"]  # type: ignore[index]
    del proof_claims["attested_hpke_public_key_hex"]  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "provider proof_claims.attested_hpke_public_key_hex is required"
        in result.message
        for result in results
    )


def test_live_check_rejects_ehbp_placeholder_measurement_claims() -> None:
    status = _verified_status()
    models = _verified_models()
    status_proof_claims = status["providers"][0]["confidentiality"]["proof_claims"]  # type: ignore[index]
    model_proof_claims = models["data"][0]["confidentiality"]["proof_claims"]  # type: ignore[index]
    status_proof_claims["code_measurement_fingerprint"] = "code-fp"  # type: ignore[index]
    model_proof_claims["code_measurement_fingerprint"] = "code-fp"  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "provider proof_claims.code_measurement_fingerprint must be a sha256 digest"
        in result.message
        for result in results
    )


def test_live_check_rejects_expected_tinfoil_provider_without_release_digest() -> None:
    status = _verified_status()
    proof_claims = status["providers"][0]["confidentiality"]["proof_claims"]  # type: ignore[index]
    del proof_claims["release_digest"]  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "provider proof_claims.release_digest must be a full sha256 digest"
        in result.message
        for result in results
    )


def test_live_check_rejects_malformed_tinfoil_tls_public_key_alias() -> None:
    status = _verified_status()
    models = _verified_models()
    status_proof_claims = status["providers"][0]["confidentiality"]["proof_claims"]  # type: ignore[index]
    model_proof_claims = models["data"][0]["confidentiality"]["proof_claims"]  # type: ignore[index]
    del status_proof_claims["tls_public_key_fingerprint_sha256"]  # type: ignore[index]
    del model_proof_claims["tls_public_key_fingerprint_sha256"]  # type: ignore[index]
    status_proof_claims["tls_public_key"] = "not-a-digest"  # type: ignore[index]
    model_proof_claims["tls_public_key"] = "not-a-digest"  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "provider proof_claims.tls_public_key must be a full sha256 digest"
        in result.message
        for result in results
    )


def test_live_check_rejects_present_null_tinfoil_tls_alias() -> None:
    status = _verified_status()
    models = _verified_models()
    valid_tls_public_key = _digest("tls-public-key")
    status_proof_claims = status["providers"][0]["confidentiality"]["proof_claims"]  # type: ignore[index]
    model_proof_claims = models["data"][0]["confidentiality"]["proof_claims"]  # type: ignore[index]
    status_proof_claims["tls_public_key_fingerprint_sha256"] = None  # type: ignore[index]
    model_proof_claims["tls_public_key_fingerprint_sha256"] = None  # type: ignore[index]
    status_proof_claims["tls_public_key"] = valid_tls_public_key  # type: ignore[index]
    model_proof_claims["tls_public_key"] = valid_tls_public_key  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "provider proof_claims.tls_public_key_fingerprint_sha256 must be a full sha256 digest"
        in result.message
        for result in results
    )


def test_live_check_rejects_ppq_private_missing_tls_binding_claim() -> None:
    proof_claims = dict(PROVIDER_EHBP_PROOF_CLAIMS)
    proof_claims.pop("tls_public_key_fingerprint_sha256")
    results = live_check.validate_provider_proof_claims(
        {
            "verified": True,
            "mode": "ppq-private-tee",
            "proof_claims": proof_claims,
        },
        label="ppq-private-prod",
    )

    assert any(
        not result.ok
        and "ppq-private-prod provider proof_claims.tls_public_key must be a full sha256 digest"
        in result.message
        for result in results
    )


def test_live_check_rejects_ppq_private_missing_bundle_digest() -> None:
    proof_claims = dict(PROVIDER_EHBP_PROOF_CLAIMS)
    results = live_check.validate_provider_proof_claims(
        {
            "verified": True,
            "mode": "ppq-private-tee",
            "proof_claims": proof_claims,
        },
        label="ppq-private-prod",
    )

    assert any(
        not result.ok
        and (
            "ppq-private-prod provider "
            "proof_claims.attestation_bundle_url_digest must be a full sha256 digest"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_ppq_private_missing_client_encryption_boundary() -> None:
    proof_claims = dict(PROVIDER_PPQ_PRIVATE_PROOF_CLAIMS)
    proof_claims.pop("client_encryption_boundary", None)
    results = live_check.validate_provider_proof_claims(
        {
            "verified": True,
            "mode": "ppq-private-tee",
            "model_ids": ["private/gpt-oss-120b"],
            "proof_claims": proof_claims,
        },
        label="ppq-private-prod",
    )

    assert any(
        not result.ok
        and (
            "ppq-private-prod provider proof_claims.client_encryption_boundary "
            "is routstr-tee-ehbp-proxy"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_ppq_private_selected_model_claim_mismatch() -> None:
    proof_claims = dict(PROVIDER_PPQ_PRIVATE_PROOF_CLAIMS)
    proof_claims["selected_model_ids"] = ["private/kimi-k2-6"]
    results = live_check.validate_provider_proof_claims(
        {
            "verified": True,
            "mode": "ppq-private-tee",
            "model_ids": ["private/gpt-oss-120b"],
            "proof_claims": proof_claims,
        },
        label="ppq-private-prod",
    )

    assert any(
        not result.ok
        and (
            "ppq-private-prod provider proof_claims.selected_model_ids "
            "exactly match selected models"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_ppq_private_missing_backend_model_attestations() -> None:
    proof_claims = dict(PROVIDER_PPQ_PRIVATE_PROOF_CLAIMS)
    proof_claims.pop("backend_model_attestations", None)
    results = live_check.validate_provider_proof_claims(
        {
            "verified": True,
            "mode": "ppq-private-tee",
            "model_ids": ["private/gpt-oss-120b"],
            "proof_claims": proof_claims,
        },
        label="ppq-private-prod",
    )

    assert any(
        not result.ok
        and (
            "ppq-private-prod provider proof_claims.backend_model_attestations "
            "contains selected model"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_ppq_private_malformed_selected_model_claim() -> None:
    proof_claims = dict(PROVIDER_PPQ_PRIVATE_PROOF_CLAIMS)
    proof_claims["selected_model_ids"] = ["private/gpt-oss-120b", 123]
    results = live_check.validate_provider_proof_claims(
        {
            "verified": True,
            "mode": "ppq-private-tee",
            "model_ids": ["private/gpt-oss-120b"],
            "proof_claims": proof_claims,
        },
        label="ppq-private-prod",
    )

    assert any(
        not result.ok
        and (
            "ppq-private-prod provider proof_claims.selected_model_ids "
            "exactly match selected models"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_ppq_private_duplicate_selected_model_claim() -> None:
    proof_claims = dict(PROVIDER_PPQ_PRIVATE_PROOF_CLAIMS)
    proof_claims["selected_model_ids"] = [
        "private/gpt-oss-120b",
        "private/gpt-oss-120b",
    ]
    results = live_check.validate_provider_proof_claims(
        {
            "verified": True,
            "mode": "ppq-private-tee",
            "model_ids": ["private/gpt-oss-120b"],
            "proof_claims": proof_claims,
        },
        label="ppq-private-prod",
    )

    assert any(
        not result.ok
        and (
            "ppq-private-prod provider proof_claims.selected_model_ids "
            "exactly match selected models"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_ppq_private_malformed_provider_model_ids() -> None:
    proof_claims = dict(PROVIDER_PPQ_PRIVATE_PROOF_CLAIMS)
    results = live_check.validate_provider_proof_claims(
        {
            "verified": True,
            "mode": "ppq-private-tee",
            "model_ids": ["private/gpt-oss-120b", 123],
            "proof_claims": proof_claims,
        },
        label="ppq-private-prod",
    )

    assert any(
        not result.ok
        and (
            "ppq-private-prod provider proof_claims.selected_model_ids "
            "exactly match selected models"
        )
        in result.message
        for result in results
    )


def test_live_check_accepts_ppq_private_without_local_proxy_artifact_binding() -> None:
    attestation = _attestation_statement(_ppq_private_routing_policy())
    proof_claims = attestation["tee"]["local_verification"]["proof_claims"]  # type: ignore[index]

    assert "attested_local_artifacts" not in proof_claims

    results = live_check.collect_results(
        status=_verified_ppq_private_status(),
        attestation=attestation,
        models=_verified_ppq_private_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("ppq-private", "private/gpt-oss-120b")],
    )

    assert all(result.ok for result in results)


def test_live_check_rejects_privatemode_placeholder_manifest_digest() -> None:
    status = _verified_privatemode_status()
    proof_claims = status["providers"][0]["confidentiality"]["proof_claims"]  # type: ignore[index]
    proof_claims["manifest_digest"] = "manifest-placeholder"  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_privatemode_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("privatemode", "privatemode/kimi-latest")],
    )

    assert any(
        not result.ok
        and "provider proof_claims.manifest_digest must be a full sha256 digest"
        in result.message
        for result in results
    )


def test_live_check_rejects_privatemode_selected_model_claim_mismatch() -> None:
    proof_claims = dict(PROVIDER_PRIVATEMODE_PROOF_CLAIMS)
    proof_claims["selected_model_ids"] = ["privatemode/other-model"]
    results = live_check.validate_provider_proof_claims(
        {
            "verified": True,
            "mode": "privatemode",
            "model_ids": ["privatemode/kimi-latest"],
            "proof_claims": proof_claims,
        },
        label="privatemode-prod",
    )

    assert any(
        not result.ok
        and (
            "privatemode-prod provider proof_claims.selected_model_ids "
            "exactly match selected models"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_privatemode_without_app_e2ee_trust_tier() -> None:
    proof_claims = dict(PROVIDER_PRIVATEMODE_PROOF_CLAIMS)
    proof_claims["trust_tier"] = "reference-only"
    results = live_check.validate_provider_proof_claims(
        {
            "verified": True,
            "mode": "privatemode",
            "model_ids": ["privatemode/kimi-latest"],
            "proof_claims": proof_claims,
        },
        label="privatemode-prod",
    )

    assert any(
        not result.ok
        and "privatemode-prod provider proof_claims.trust_tier is app-e2ee"
        in result.message
        for result in results
    )


def test_live_check_rejects_privatemode_unknown_gpu_attestation_policy() -> None:
    proof_claims = dict(PROVIDER_PRIVATEMODE_PROOF_CLAIMS)
    proof_claims["gpu_attestation_policy"] = "gpu-attestation-policy"
    results = live_check.validate_provider_proof_claims(
        {
            "verified": True,
            "mode": "privatemode",
            "model_ids": ["privatemode/kimi-latest"],
            "proof_claims": proof_claims,
        },
        label="privatemode-prod",
    )

    assert any(
        not result.ok
        and (
            "privatemode-prod provider proof_claims.gpu_attestation_policy "
            "is nvidia-ocsp-good-only"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_privatemode_image_only_proxy_proof() -> None:
    proof_claims = dict(PROVIDER_PRIVATEMODE_PROOF_CLAIMS)
    proof_claims.pop("proxy_binary_digest", None)
    results = live_check.validate_provider_proof_claims(
        {
            "verified": True,
            "mode": "privatemode",
            "model_ids": ["privatemode/kimi-latest"],
            "proof_claims": proof_claims,
        },
        label="privatemode-prod",
    )

    assert any(
        not result.ok
        and (
            "privatemode-prod provider proof_claims.proxy_binary_digest "
            "must be a full sha256 digest"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_present_null_privatemode_manifest_alias() -> None:
    status = _verified_privatemode_status()
    models = _verified_privatemode_models()
    status_proof_claims = status["providers"][0]["confidentiality"]["proof_claims"]  # type: ignore[index]
    model_proof_claims = models["data"][0]["confidentiality"]["proof_claims"]  # type: ignore[index]
    status_proof_claims["manifest_digest"] = None  # type: ignore[index]
    model_proof_claims["manifest_digest"] = None  # type: ignore[index]
    status_proof_claims["manifest_log_manifest_digest"] = PRIVATEMODE_MANIFEST_DIGEST  # type: ignore[index]
    model_proof_claims["manifest_log_manifest_digest"] = PRIVATEMODE_MANIFEST_DIGEST  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("privatemode", "privatemode/kimi-latest")],
    )

    assert any(
        not result.ok
        and "provider proof_claims.manifest_digest must be a full sha256 digest"
        in result.message
        for result in results
    )


def test_live_check_rejects_conflicting_privatemode_manifest_aliases() -> None:
    status = _verified_privatemode_status()
    models = _verified_privatemode_models()
    status_proof_claims = status["providers"][0]["confidentiality"]["proof_claims"]  # type: ignore[index]
    model_proof_claims = models["data"][0]["confidentiality"]["proof_claims"]  # type: ignore[index]
    status_proof_claims["manifest_log_manifest_digest"] = _digest("other-manifest")  # type: ignore[index]
    model_proof_claims["manifest_log_manifest_digest"] = _digest("other-manifest")  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("privatemode", "privatemode/kimi-latest")],
    )

    assert any(
        not result.ok
        and "provider proof_claims.manifest_digest aliases match" in result.message
        for result in results
    )


def test_live_check_rejects_present_object_privatemode_proxy_alias() -> None:
    status = _verified_privatemode_status()
    models = _verified_privatemode_models()
    status_proof_claims = status["providers"][0]["confidentiality"]["proof_claims"]  # type: ignore[index]
    model_proof_claims = models["data"][0]["confidentiality"]["proof_claims"]  # type: ignore[index]
    status_proof_claims["proxy_image_digest"] = {
        "digest": PRIVATEMODE_PROXY_IMAGE_DIGEST
    }  # type: ignore[index]
    model_proof_claims["proxy_image_digest"] = {
        "digest": PRIVATEMODE_PROXY_IMAGE_DIGEST
    }  # type: ignore[index]
    status_proof_claims["proxy_binary_digest"] = PRIVATEMODE_PROXY_IMAGE_DIGEST  # type: ignore[index]
    model_proof_claims["proxy_binary_digest"] = PRIVATEMODE_PROXY_IMAGE_DIGEST  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("privatemode", "privatemode/kimi-latest")],
    )

    assert any(
        not result.ok
        and "provider proof_claims.proxy_image_digest must be a full sha256 digest"
        in result.message
        for result in results
    )


def test_live_check_rejects_privatemode_placeholder_component_measurement() -> None:
    status = _verified_privatemode_status()
    proof_claims = status["providers"][0]["confidentiality"]["proof_claims"]  # type: ignore[index]
    proof_claims["coordinator_measurement"] = "coordinator-placeholder"  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_privatemode_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("privatemode", "privatemode/kimi-latest")],
    )

    assert any(
        not result.ok
        and "provider proof_claims.coordinator_measurement must be a full sha256 digest"
        in result.message
        for result in results
    )


def test_live_check_rejects_model_metadata_without_provider_proof_claims() -> None:
    models = _verified_models()
    confidentiality = models["data"][0]["confidentiality"]  # type: ignore[index]
    del confidentiality["proof_claims"]  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "tinfoil/gpt-secure nested model proof claims match provider status"
        in result.message
        for result in results
    )


def test_live_check_rejects_local_tee_without_future_expiry() -> None:
    status = _verified_status()
    status["routstr_tee"]["local_verification"]["expires_at"] = 1  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and "local TEE verifier expiry is in the future" in result.message
        for result in results
    )


def test_live_check_rejects_attestation_statement_without_future_expiry() -> None:
    attestation = _attestation_statement()
    attestation["tee"]["local_verification"]["expires_at"] = 1  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "attestation local TEE verifier expiry is in the future" in result.message
        for result in results
    )


def test_live_check_rejects_string_local_tee_expiry() -> None:
    status = _verified_status()
    status["routstr_tee"]["local_verification"]["expires_at"] = "4102444800"  # type: ignore[index]
    attestation = _attestation_statement()
    attestation["tee"]["local_verification"]["expires_at"] = "4102444800"  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "local TEE verifier expiry must be an integer Unix timestamp"
        in result.message
        for result in results
    )
    assert any(
        not result.ok
        and "attestation local TEE verifier expiry must be an integer Unix timestamp"
        in result.message
        for result in results
    )


def test_live_check_rejects_string_local_tee_verified_at() -> None:
    status = _verified_status()
    status["routstr_tee"]["local_verification"]["verified_at"] = "1700000000"  # type: ignore[index]
    attestation = _attestation_statement()
    attestation["tee"]["local_verification"]["verified_at"] = "1700000000"  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "local TEE verifier verified_at must be an integer Unix timestamp"
        in result.message
        for result in results
    )
    assert any(
        not result.ok
        and "attestation local TEE verifier verified_at must be an integer Unix timestamp"
        in result.message
        for result in results
    )


def test_live_check_rejects_future_local_tee_verified_at() -> None:
    status = _verified_status()
    status["routstr_tee"]["local_verification"]["verified_at"] = (
        int(  # type: ignore[index]
            time.time()
        )
        + 60
    )
    attestation = _attestation_statement()
    attestation["tee"]["local_verification"]["verified_at"] = (
        int(  # type: ignore[index]
            time.time()
        )
        + 60
    )

    results = live_check.collect_results(
        status=status,
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "local TEE verifier verified_at is not in the future" in result.message
        for result in results
    )
    assert any(
        not result.ok
        and "attestation local TEE verifier verified_at is not in the future"
        in result.message
        for result in results
    )


def test_live_check_rejects_status_without_local_tee_report_data_proof() -> None:
    status = _verified_status()
    proof_claims = status["routstr_tee"]["local_verification"]["proof_claims"]  # type: ignore[index]
    del proof_claims["tee_report_data_hex"]  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "local TEE status proof_claims.tee_report_data_hex is required"
        in result.message
        for result in results
    )


def test_live_check_rejects_status_without_local_tee_hpke_key_config_digest_claim() -> None:
    status = _verified_status()
    proof_claims = status["routstr_tee"]["local_verification"]["proof_claims"]  # type: ignore[index]
    proof_claims.pop("hpke_key_config_digest", None)  # type: ignore[union-attr]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "local TEE status proof_claims.hpke_key_config_digest must be a full sha256 digest"
        in result.message
        for result in results
    )


def test_live_check_rejects_status_local_tee_hpke_key_config_digest_mismatch() -> None:
    status = _verified_status()
    proof_claims = status["routstr_tee"]["local_verification"]["proof_claims"]  # type: ignore[index]
    proof_claims["hpke_key_config_digest"] = OTHER_HPKE_KEY_CONFIG_DIGEST  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "local TEE status proof_claims.hpke_key_config_digest matches HPKE key config"
        in result.message
        for result in results
    )


def test_live_check_rejects_status_without_local_tee_report_nonce_proof() -> None:
    status = _verified_status()
    proof_claims = status["routstr_tee"]["local_verification"]["proof_claims"]  # type: ignore[index]
    del proof_claims["tee_report_nonce"]  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "local TEE status proof_claims.tee_report_nonce is required"
        in result.message
        for result in results
    )


def test_live_check_rejects_status_without_local_tee_report_nonce_digest() -> None:
    status = _verified_status()
    proof_claims = status["routstr_tee"]["local_verification"]["proof_claims"]  # type: ignore[index]
    del proof_claims["tee_report_nonce_digest"]  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "local TEE status proof_claims.tee_report_nonce_digest must be a full sha256 digest"
        in result.message
        for result in results
    )


def test_live_check_rejects_status_with_unbound_local_tee_report_data() -> None:
    status = _verified_status()
    proof_claims = status["routstr_tee"]["local_verification"]["proof_claims"]  # type: ignore[index]
    proof_claims["tee_report_data_digest"] = _digest("wrong-report-data")

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "local TEE status proof_claims.tee_report_data_digest binds policy, keys, and nonce"
        in result.message
        for result in results
    )


def test_live_check_rejects_non_boolean_local_tee_verification_step() -> None:
    status = _verified_status()
    proof_claims = status["routstr_tee"]["local_verification"]["proof_claims"]  # type: ignore[index]
    verification_steps = proof_claims["verification_steps"]  # type: ignore[index]
    verification_steps["diagnostic"] = "SECRET_PROMPT"  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "local TEE status proof_claims.verification_steps values are JSON booleans"
        in result.message
        for result in results
    )
    assert "SECRET_PROMPT" not in str(results)


def test_live_check_rejects_attestation_without_local_tee_report_data_proof() -> None:
    attestation = _attestation_statement()
    proof_claims = attestation["tee"]["local_verification"]["proof_claims"]  # type: ignore[index]
    del proof_claims["tee_report_data_digest"]  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "attestation local TEE proof_claims.tee_report_data_digest must be a full sha256 digest"
        in result.message
        for result in results
    )


def test_live_check_rejects_expected_provider_without_model_selectors() -> None:
    status = _verified_status()
    confidentiality = status["providers"][0]["confidentiality"]  # type: ignore[index]
    confidentiality["model_ids"] = []  # type: ignore[index]
    confidentiality["model_id_prefixes"] = []  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and "no status covering" in result.message for result in results
    )


def test_live_check_rejects_malformed_provider_model_selectors() -> None:
    status = _verified_status()
    confidentiality = status["providers"][0]["confidentiality"]  # type: ignore[index]
    confidentiality["model_ids"] = [  # type: ignore[index]
        "tinfoil/gpt-secure",
        {"unexpected": "object"},
    ]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "provider model_ids must contain only non-empty strings" in result.message
        for result in results
    )


def test_live_check_rejects_non_string_model_id() -> None:
    status = _verified_status()
    confidentiality = status["providers"][0]["confidentiality"]  # type: ignore[index]
    confidentiality["model_ids"] = ["123"]  # type: ignore[index]
    models = _verified_models()
    models["data"][0]["id"] = 123  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "123")],
    )

    assert any(
        not result.ok
        and "/v1/models data[0].id must be a non-empty string" in result.message
        for result in results
    )


def test_live_check_rejects_non_string_status_provider_identity() -> None:
    status = _verified_status()
    status["providers"][0]["provider_type"] = 123  # type: ignore[index]
    models = _verified_models()
    models["data"][0]["attestation_provider"] = "123"  # type: ignore[index]
    models["data"][0]["confidentiality"]["provider_type"] = "123"  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("123", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "provider provider_type must be a non-empty string" in result.message
        for result in results
    )


def test_live_check_rejects_missing_local_tee_readiness() -> None:
    status = _verified_status()
    status["routstr_tee"]["ready"] = False  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and result.message == "local Routstr TEE is ready"
        for result in results
    )


def test_live_check_rejects_truthy_string_local_tee_readiness() -> None:
    status = _verified_status()
    status["routstr_tee"]["ready"] = "false"  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and result.message == "local Routstr TEE is ready"
        for result in results
    )


def test_live_check_rejects_truthy_string_attestation_local_verifier() -> None:
    attestation = _attestation_statement()
    attestation["tee"]["local_verification"]["verified"] = "false"  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and result.message == "attestation local TEE verifier reports verified"
        for result in results
    )


def test_live_check_rejects_non_required_public_modes() -> None:
    status = _verified_status()
    status["mode"] = "permissive"
    attestation = _attestation_statement()
    routing_policy = attestation["routing_policy"]
    routing_policy["mode"] = "permissive"  # type: ignore[index]
    attestation["routing_policy_digest"] = _json_digest(routing_policy)
    attestation["tee"]["local_verification"]["proof_claims"][  # type: ignore[index]
        "routstr_config_measurement"
    ] = attestation["routing_policy_digest"]

    results = live_check.collect_results(
        status=status,
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and result.message == "confidential routing mode is required"
        for result in results
    )
    assert any(
        not result.ok
        and result.message == "attestation routing_policy.mode is required"
        for result in results
    )


def test_live_check_rejects_malformed_attestation_routing_policy_identity() -> None:
    attestation = _attestation_statement()
    routing_policy = {
        "mode": {"value": "required"},
        "required": "false",
        "providers": [
            {
                "provider_type": {"value": "tinfoil"},
                "upstream_name": {"value": "tinfoil-prod"},
                "base_url": "https://inference.tinfoil.sh/v1",
                "db_id": {"value": 7},
            }
        ],
    }
    attestation["routing_policy"] = routing_policy
    attestation["routing_policy_digest"] = _json_digest(routing_policy)
    attestation["tee"]["local_verification"]["proof_claims"][  # type: ignore[index]
        "routstr_config_measurement"
    ] = attestation["routing_policy_digest"]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and "attestation routing_policy.mode" in result.message
        for result in results
    )
    assert any(
        not result.ok and "attestation routing_policy.required" in result.message
        for result in results
    )
    assert any(
        not result.ok
        and "attestation routing_policy.providers[0].provider_type" in result.message
        for result in results
    )


def test_live_check_rejects_non_canonical_attestation_routing_policy() -> None:
    attestation = _attestation_statement()
    routing_policy = attestation["routing_policy"]
    routing_policy["non_canonical"] = float("nan")  # type: ignore[index]
    attestation["routing_policy_digest"] = _json_digest(routing_policy)

    results = live_check.validate_attestation_routing_policy(attestation)

    assert any(
        not result.ok
        and result.message == "attestation routing_policy is canonical JSON"
        for result in results
    )


def test_live_check_rejects_attestation_routing_policy_provider_mode_mismatch() -> None:
    attestation = _attestation_statement()
    routing_policy = attestation["routing_policy"]
    routing_policy["providers"][0]["provider_type"] = "custom"  # type: ignore[index]
    attestation["routing_policy_digest"] = _json_digest(routing_policy)
    attestation["tee"]["local_verification"]["proof_claims"][  # type: ignore[index]
        "routstr_config_measurement"
    ] = attestation["routing_policy_digest"]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "attestation routing_policy.providers[0] confidentiality mode tinfoil matches provider_type custom"
        in result.message
        for result in results
    )


def test_live_check_rejects_malformed_attestation_routing_policy_confidentiality() -> (
    None
):
    attestation = _attestation_statement()
    routing_policy = attestation["routing_policy"]
    confidentiality = routing_policy["providers"][0]["confidentiality"]  # type: ignore[index]
    confidentiality["enabled"] = "true"  # type: ignore[index]
    confidentiality["verified"] = "true"  # type: ignore[index]
    confidentiality["mode"] = {"value": "tinfoil"}  # type: ignore[index]
    confidentiality["verifier"] = 123  # type: ignore[index]
    confidentiality["policy_digest"] = "sha256:policy"  # type: ignore[index]
    confidentiality["evidence_digest"] = "sha256:evidence"  # type: ignore[index]
    confidentiality["verified_at"] = "1700000000"  # type: ignore[index]
    confidentiality["expires_at"] = "4102444800"  # type: ignore[index]
    confidentiality["model_ids"] = ["tinfoil/gpt-secure", {"unexpected": "object"}]  # type: ignore[index]
    attestation["routing_policy_digest"] = _json_digest(routing_policy)
    attestation["tee"]["local_verification"]["proof_claims"][  # type: ignore[index]
        "routstr_config_measurement"
    ] = attestation["routing_policy_digest"]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "attestation routing_policy.providers[0].confidentiality.enabled"
        in result.message
        for result in results
    )
    assert any(
        not result.ok
        and "attestation routing_policy.providers[0].confidentiality.mode"
        in result.message
        for result in results
    )
    assert any(
        not result.ok
        and "attestation routing_policy.providers[0].confidentiality.policy_digest"
        in result.message
        for result in results
    )
    assert any(
        not result.ok
        and "attestation routing_policy.providers[0].confidentiality.verified_at"
        in result.message
        for result in results
    )
    assert any(
        not result.ok
        and "attestation routing_policy.providers[0].confidentiality.model_ids"
        in result.message
        for result in results
    )


def test_live_check_rejects_attestation_routing_policy_without_provider_proof() -> None:
    attestation = _attestation_statement()
    routing_policy = attestation["routing_policy"]
    confidentiality = routing_policy["providers"][0]["confidentiality"]  # type: ignore[index]
    confidentiality.pop("verified_claims_digest")  # type: ignore[union-attr]
    confidentiality.pop("proof_claims")  # type: ignore[union-attr]
    attestation["routing_policy_digest"] = _json_digest(routing_policy)
    attestation["tee"]["local_verification"]["proof_claims"][  # type: ignore[index]
        "routstr_config_measurement"
    ] = attestation["routing_policy_digest"]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "attestation routing_policy.providers[0].confidentiality.verified_claims_digest"
        in result.message
        for result in results
    )
    assert any(
        not result.ok
        and "attestation routing_policy.providers[0].confidentiality provider proof_claims are present"
        in result.message
        for result in results
    )


def test_live_check_rejects_relative_attestation_routing_policy_base_url() -> None:
    attestation = _attestation_statement()
    routing_policy = attestation["routing_policy"]
    routing_policy["providers"][0]["base_url"] = "relative/private/path"  # type: ignore[index]
    attestation["routing_policy_digest"] = _json_digest(routing_policy)
    attestation["tee"]["local_verification"]["proof_claims"][  # type: ignore[index]
        "routstr_config_measurement"
    ] = attestation["routing_policy_digest"]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "attestation routing_policy.providers[0].base_url is an absolute URL"
        in result.message
        for result in results
    )


def test_live_check_rejects_non_https_attestation_routing_policy_base_url() -> None:
    attestation = _attestation_statement()
    routing_policy = attestation["routing_policy"]
    routing_policy["providers"][0]["base_url"] = "http://inference.tinfoil.sh/v1"  # type: ignore[index]
    attestation["routing_policy_digest"] = _json_digest(routing_policy)
    attestation["tee"]["local_verification"]["proof_claims"][  # type: ignore[index]
        "routstr_config_measurement"
    ] = attestation["routing_policy_digest"]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "attestation routing_policy.providers[0].base_url uses an allowed scheme"
        in result.message
        for result in results
    )


def test_live_check_rejects_tinfoil_non_default_routing_policy_base_url() -> None:
    attestation = _attestation_statement()
    routing_policy = attestation["routing_policy"]
    routing_policy["providers"][0]["base_url"] = "https://attacker.example/v1"  # type: ignore[index]
    attestation["routing_policy_digest"] = _json_digest(routing_policy)
    attestation["tee"]["local_verification"]["proof_claims"][  # type: ignore[index]
        "routstr_config_measurement"
    ] = attestation["routing_policy_digest"]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "attestation routing_policy.providers[0].base_url is Tinfoil default router URL"
        in result.message
        for result in results
    )


def test_live_check_rejects_attestation_routing_policy_base_url_without_host() -> None:
    attestation = _attestation_statement()
    routing_policy = attestation["routing_policy"]
    routing_policy["providers"][0]["base_url"] = "https://:443/v1"  # type: ignore[index]
    attestation["routing_policy_digest"] = _json_digest(routing_policy)
    attestation["tee"]["local_verification"]["proof_claims"][  # type: ignore[index]
        "routstr_config_measurement"
    ] = attestation["routing_policy_digest"]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "attestation routing_policy.providers[0].base_url includes a host"
        in result.message
        for result in results
    )


def test_live_check_rejects_attestation_routing_policy_base_url_invalid_port() -> None:
    attestation = _attestation_statement()
    routing_policy = attestation["routing_policy"]
    routing_policy["providers"][0]["base_url"] = (  # type: ignore[index]
        "https://inference.tinfoil.sh:bad/v1"
    )
    attestation["routing_policy_digest"] = _json_digest(routing_policy)
    attestation["tee"]["local_verification"]["proof_claims"][  # type: ignore[index]
        "routstr_config_measurement"
    ] = attestation["routing_policy_digest"]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "attestation routing_policy.providers[0].base_url includes a valid port"
        in result.message
        for result in results
    )


def test_live_check_rejects_ppq_private_non_private_routing_policy_base_url() -> None:
    attestation = _attestation_statement()
    routing_policy = {
        "mode": "required",
        "required": True,
        "providers": [
            {
                "provider_type": "ppq-private",
                "upstream_name": "ppq-private-prod",
                "base_url": "https://api.ppq.ai/v1",
                "db_id": 8,
                "confidentiality": {
                    "enabled": True,
                    "verified": True,
                    "mode": "ppq-private-tee",
                    "verifier": "routstr-ppq-private-go-verifier",
                    "policy_digest": PROVIDER_POLICY_DIGEST,
                    "evidence_digest": PROVIDER_EVIDENCE_DIGEST,
                    "verified_at": 1_700_000_000,
                    "expires_at": 4_102_444_800,
                    "model_ids": ["private/gpt-oss-120b"],
                    "model_id_prefixes": [],
                },
            }
        ],
    }
    attestation["routing_policy"] = routing_policy
    attestation["routing_policy_digest"] = _json_digest(routing_policy)
    attestation["tee"]["local_verification"]["proof_claims"][  # type: ignore[index]
        "routstr_config_measurement"
    ] = attestation["routing_policy_digest"]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "attestation routing_policy.providers[0].base_url uses PPQ private path"
        in result.message
        for result in results
    )


def test_live_check_rejects_ppq_private_unowned_routing_policy_base_url() -> None:
    attestation = _attestation_statement()
    routing_policy = {
        "mode": "required",
        "required": True,
        "providers": [
            {
                "provider_type": "ppq-private",
                "upstream_name": "ppq-private-prod",
                "base_url": "https://attacker.example/private/v1",
                "db_id": 8,
                "confidentiality": {
                    "enabled": True,
                    "verified": True,
                    "mode": "ppq-private-tee",
                    "verifier": "routstr-ppq-private-go-verifier",
                    "policy_digest": PROVIDER_POLICY_DIGEST,
                    "evidence_digest": PROVIDER_EVIDENCE_DIGEST,
                    "verified_at": 1_700_000_000,
                    "expires_at": 4_102_444_800,
                    "model_ids": ["private/gpt-oss-120b"],
                    "model_id_prefixes": [],
                },
            }
        ],
    }
    attestation["routing_policy"] = routing_policy
    attestation["routing_policy_digest"] = _json_digest(routing_policy)
    attestation["tee"]["local_verification"]["proof_claims"][  # type: ignore[index]
        "routstr_config_measurement"
    ] = attestation["routing_policy_digest"]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and (
            "attestation routing_policy.providers[0].base_url host is ppq.ai "
            "or a ppq.ai subdomain"
        )
        in result.message
        for result in results
    )


def test_live_check_rejects_privatemode_non_loopback_routing_policy_base_url() -> None:
    routing_policy = _privatemode_routing_policy()
    routing_policy["providers"][0]["base_url"] = "https://api.privatemode.ai/v1"  # type: ignore[index]
    attestation = _attestation_statement(routing_policy)

    results = live_check.collect_results(
        status=_verified_privatemode_status(),
        attestation=attestation,
        models=_verified_privatemode_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("privatemode", "privatemode/kimi-latest")],
    )

    assert any(
        not result.ok
        and "attestation routing_policy.providers[0].base_url is Privatemode loopback proxy URL"
        in result.message
        for result in results
    )


def test_live_check_rejects_public_raw_claims() -> None:
    status = _verified_status()
    provider = status["providers"][0]  # type: ignore[index]
    provider["confidentiality"]["verified_claims"] = {  # type: ignore[index]
        "api_key": "SECRET"
    }

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and "forbidden fields" in result.message for result in results
    )


def test_live_check_rejects_public_private_key_leakage() -> None:
    status = _verified_status()
    provider = status["providers"][0]  # type: ignore[index]
    provider["confidentiality"]["proof_claims"]["private_key"] = "SECRET"  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and "forbidden fields" in result.message for result in results
    )


def test_live_check_rejects_public_pem_private_key_value_leakage() -> None:
    status = _verified_status()
    provider = status["providers"][0]  # type: ignore[index]
    provider["confidentiality"]["proof_claims"]["diagnostic"] = (  # type: ignore[index]
        "-----BEGIN PRIVATE KEY-----\nSECRET\n-----END PRIVATE KEY-----"
    )

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and "forbidden fields" in result.message for result in results
    )


def test_live_check_rejects_public_cashu_token_value_leakage() -> None:
    status = _verified_status()
    provider = status["providers"][0]  # type: ignore[index]
    provider["confidentiality"]["proof_claims"]["diagnostic"] = "cashuAsecret"  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and "forbidden fields" in result.message for result in results
    )


def test_live_check_rejects_runtime_provider_url_leakage() -> None:
    status = _verified_status()
    status["providers"][0]["base_url"] = "https://inference.tinfoil.sh/v1"  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "runtime public confidentiality surfaces expose forbidden fields"
        in result.message
        for result in results
    )


def test_live_check_allows_attestation_policy_url_binding() -> None:
    attestation = _attestation_statement()

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert all(result.ok for result in results)


def test_live_check_rejects_wrong_hpke_content_type() -> None:
    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/octet-stream",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and "application/ohttp-keys" in result.message
        for result in results
    )


def test_live_check_rejects_status_hpke_digest_mismatch() -> None:
    status = _verified_status()
    status["routstr_tee"]["hpke_key_config_digest"] = OTHER_HPKE_KEY_CONFIG_DIGEST  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and "status local HPKE" in result.message for result in results
    )


def test_live_check_rejects_attestation_hpke_digest_mismatch() -> None:
    attestation = _attestation_statement()
    attestation["tee"]["hpke_key_config"]["key_config_digest"] = (
        OTHER_HPKE_KEY_CONFIG_DIGEST  # type: ignore[index]
    )

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and "attestation HPKE" in result.message for result in results
    )


def test_live_check_rejects_model_provider_evidence_mismatch() -> None:
    models = _verified_models()
    models["data"][0]["attestation_evidence_digest"] = OTHER_PROVIDER_EVIDENCE_DIGEST  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "model evidence digest matches provider status" in result.message
        for result in results
    )


def test_live_check_rejects_model_policy_digest_mismatch() -> None:
    models = _verified_models()
    models["data"][0]["confidentiality"]["policy_digest"] = _digest("other-policy")  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "nested model policy digest matches provider status" in result.message
        for result in results
    )


def test_live_check_rejects_model_verified_claims_digest_mismatch() -> None:
    models = _verified_models()
    models["data"][0]["confidentiality"]["verified_claims_digest"] = _digest(  # type: ignore[index]
        "other-provider-claims"
    )

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "nested model verified claims digest matches provider status"
        in result.message
        for result in results
    )


def test_live_check_rejects_model_verifier_mismatch() -> None:
    models = _verified_models()
    models["data"][0]["confidentiality"]["verifier"] = "other-verifier"  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "nested model confidentiality snapshot matches a single provider status"
        in result.message
        for result in results
    )


def test_live_check_rejects_model_freshness_mismatch() -> None:
    models = _verified_models()
    models["data"][0]["confidentiality"]["verified_at"] = 1_800_000_000  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "nested model confidentiality snapshot matches a single provider status"
        in result.message
        for result in results
    )


def test_live_check_rejects_model_selector_mismatch() -> None:
    models = _verified_models()
    models["data"][0]["confidentiality"]["model_ids"] = [  # type: ignore[index]
        "tinfoil/gpt-secure",
        "tinfoil/plain-extra",
    ]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "nested model confidentiality snapshot matches a single provider status"
        in result.message
        for result in results
    )


def test_live_check_rejects_model_malformed_selector_snapshot_match() -> None:
    models = _verified_models()
    models["data"][0]["confidentiality"]["model_ids"] = [  # type: ignore[index]
        "tinfoil/gpt-secure",
        123,
    ]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "nested model confidentiality snapshot matches a single provider status"
        in result.message
        for result in results
    )


def test_live_check_rejects_attested_provider_verifier_mismatch() -> None:
    routing_policy = json.loads(json.dumps(ROUTING_POLICY))
    routing_policy["providers"][0]["confidentiality"]["verifier"] = "other-verifier"
    attestation = _attestation_statement(routing_policy)

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "attested provider confidentiality snapshot matches a single provider status"
        in result.message
        for result in results
    )


def test_live_check_rejects_attested_provider_selector_mismatch() -> None:
    routing_policy = json.loads(json.dumps(ROUTING_POLICY))
    routing_policy["providers"][0]["confidentiality"]["model_ids"] = [
        "tinfoil/gpt-secure",
        "tinfoil/plain-extra",
    ]
    attestation = _attestation_statement(routing_policy)

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "attested provider confidentiality snapshot matches a single provider status"
        in result.message
        for result in results
    )


def test_live_check_rejects_attested_provider_policy_snapshot_mismatch() -> None:
    status = _verified_status()
    status["providers"][0]["confidentiality_policy"] = {  # type: ignore[index]
        "repo": "tinfoilsh/confidential-model-router",
        "expected_release_digest": PROVIDER_EHBP_PROOF_CLAIMS["release_digest"],
    }
    routing_policy = json.loads(json.dumps(ROUTING_POLICY))
    routing_policy["providers"][0]["confidentiality_policy"] = {  # type: ignore[index]
        "repo": "tinfoilsh/confidential-model-router",
        "expected_repo": "tinfoilsh/other-router",
        "expected_release_digest": PROVIDER_EHBP_PROOF_CLAIMS["release_digest"],
    }
    attestation = _attestation_statement(routing_policy)

    results = live_check.collect_results(
        status=status,
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "tinfoil/gpt-secure attested provider policy snapshot matches provider status"
        in result.message
        for result in results
    )


def test_live_check_rejects_model_mixed_provider_confidentiality_snapshot() -> None:
    status = _verified_status()
    first_provider = status["providers"][0]  # type: ignore[index]
    second_provider = json.loads(json.dumps(first_provider))
    second_confidentiality = second_provider["confidentiality"]
    second_confidentiality["policy_digest"] = _digest("second-provider-policy")
    second_confidentiality["evidence_digest"] = _digest("second-provider-evidence")
    second_confidentiality["verified_claims_digest"] = _digest("second-provider-claims")
    second_confidentiality["proof_claims"]["release_digest"] = _digest(
        "second-provider-release"
    )
    status["providers"].append(second_provider)  # type: ignore[union-attr]

    models = _verified_models()
    model = models["data"][0]  # type: ignore[index]
    model_confidentiality = model["confidentiality"]
    model["attestation_evidence_digest"] = second_confidentiality["evidence_digest"]
    model_confidentiality["evidence_digest"] = second_confidentiality["evidence_digest"]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "nested model confidentiality snapshot matches a single provider status"
        in result.message
        for result in results
    )


def test_live_check_rejects_attested_mixed_provider_confidentiality_snapshot() -> None:
    status = _verified_status()
    first_provider = status["providers"][0]  # type: ignore[index]
    second_provider = json.loads(json.dumps(first_provider))
    second_confidentiality = second_provider["confidentiality"]
    second_confidentiality["policy_digest"] = _digest("second-provider-policy")
    second_confidentiality["evidence_digest"] = _digest("second-provider-evidence")
    second_confidentiality["verified_claims_digest"] = _digest("second-provider-claims")
    second_confidentiality["proof_claims"]["release_digest"] = _digest(
        "second-provider-release"
    )
    status["providers"].append(second_provider)  # type: ignore[union-attr]

    routing_policy = json.loads(json.dumps(ROUTING_POLICY))
    attested_confidentiality = routing_policy["providers"][0]["confidentiality"]
    attested_confidentiality["evidence_digest"] = second_confidentiality[
        "evidence_digest"
    ]
    attestation = _attestation_statement(routing_policy)

    results = live_check.collect_results(
        status=status,
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "attested provider confidentiality snapshot matches a single provider status"
        in result.message
        for result in results
    )


def test_live_check_accepts_model_supported_endpoint_subset() -> None:
    """Provider-wide endpoint metadata may be broader than a specific model."""
    status = _verified_status()
    status["providers"][0]["supported_endpoints"] = [  # type: ignore[index]
        "/v1/chat/completions",
        "/v1/audio/transcriptions",
        "/v1/embeddings",
        "/v1/responses",
    ]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert all(
        result.ok
        for result in results
        if "nested model supported endpoints" in result.message
    )


def test_live_check_accepts_audio_family_covering_exact_model_endpoint() -> None:
    """Provider `/v1/audio` coverage should include exact audio sub-endpoints."""
    status = _verified_status()
    status["providers"][0]["supported_endpoints"] = ["/v1/audio"]  # type: ignore[index]
    models = _verified_models()
    models["data"][0]["confidentiality"]["supported_endpoints"] = [  # type: ignore[index]
        "/v1/audio/transcriptions"
    ]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert all(
        result.ok
        for result in results
        if (
            "nested model supported endpoints" in result.message
            or "model confidentiality metadata matches one provider status"
            in result.message
        )
    )


def test_live_check_rejects_audio_family_model_without_provider_family() -> None:
    """A model claiming all audio endpoints must not hide behind one sub-endpoint."""
    status = _verified_status()
    status["providers"][0]["supported_endpoints"] = [  # type: ignore[index]
        "/v1/audio/transcriptions"
    ]
    models = _verified_models()
    models["data"][0]["confidentiality"]["supported_endpoints"] = ["/v1/audio"]  # type: ignore[index]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "nested model supported endpoints are covered by provider status"
        in result.message
        for result in results
    )


def test_live_check_rejects_model_supported_endpoint_mismatch() -> None:
    models = _verified_models()
    models["data"][0]["confidentiality"]["supported_endpoints"] = [  # type: ignore[index]
        "/v1/chat/completions",
        "/v1/responses",
    ]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "nested model supported endpoints are covered by provider status"
        in result.message
        for result in results
    )


def test_live_check_rejects_malformed_provider_supported_endpoints() -> None:
    status = _verified_status()
    status["providers"][0]["supported_endpoints"] = [  # type: ignore[index]
        "/v1/chat/completions",
        {"unexpected": "object"},
    ]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "provider supported_endpoints must contain only non-empty strings"
        in result.message
        for result in results
    )


def test_live_check_rejects_duplicate_provider_model_ids() -> None:
    status = _verified_status()
    confidentiality = status["providers"][0]["confidentiality"]  # type: ignore[index]
    confidentiality["model_ids"] = [  # type: ignore[index]
        "tinfoil/gpt-secure",
        "tinfoil/gpt-secure",
    ]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "provider model_ids must not contain duplicates" in result.message
        for result in results
    )


def test_live_check_rejects_case_variant_duplicate_provider_model_ids() -> None:
    status = _verified_status()
    confidentiality = status["providers"][0]["confidentiality"]  # type: ignore[index]
    confidentiality["model_ids"] = [  # type: ignore[index]
        "TINFOIL/gpt-secure",
        "tinfoil/gpt-secure",
    ]

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "provider model_ids must not contain duplicates" in result.message
        for result in results
    )


def test_live_check_rejects_duplicate_model_supported_endpoints() -> None:
    models = _verified_models()
    models["data"][0]["confidentiality"]["supported_endpoints"] = [  # type: ignore[index]
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "model supported_endpoints must not contain duplicates" in result.message
        for result in results
    )


def test_live_check_rejects_malformed_model_supported_endpoints() -> None:
    models = _verified_models()
    models["data"][0]["confidentiality"]["supported_endpoints"] = [  # type: ignore[index]
        "/v1/chat/completions",
        {"unexpected": "object"},
    ]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "model supported_endpoints must contain only non-empty strings"
        in result.message
        for result in results
    )


def test_live_check_rejects_model_attestation_provider_mismatch() -> None:
    models = _verified_models()
    models["data"][0]["attestation_provider"] = "plain"  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and "model attestation provider matches" in result.message
        for result in results
    )


def test_live_check_rejects_model_attestation_provider_mode_alias_mismatch() -> None:
    models = _verified_ppq_private_models()
    models["data"][0]["attestation_provider"] = "ppq-private-tee"  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_ppq_private_status(),
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("ppq-private", "private/gpt-oss-120b")],
    )

    assert any(
        not result.ok
        and "model attestation provider matches nested confidentiality provider_type"
        in result.message
        for result in results
    )


def test_live_check_rejects_expected_provider_mode_alias() -> None:
    results = live_check.collect_results(
        status=_verified_ppq_private_status(),
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        models=_verified_ppq_private_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("ppq-private-tee", "private/gpt-oss-120b")],
    )

    assert any(
        not result.ok
        and "ppq-private-tee has no status covering private/gpt-oss-120b"
        in result.message
        for result in results
    )
    assert any(
        not result.ok
        and "model attestation provider matches ppq-private-tee" in result.message
        for result in results
    )


def test_live_check_rejects_model_nested_provider_type_mode_mismatch() -> None:
    models = _verified_models()
    models["data"][0]["confidentiality"]["provider_type"] = "privatemode"  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "model confidentiality mode tinfoil matches provider_type privatemode"
        in result.message
        for result in results
    )


def test_live_check_rejects_model_missing_nested_confidentiality() -> None:
    models = _verified_models()
    del models["data"][0]["confidentiality"]  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and "nested confidentiality metadata is present" in result.message
        for result in results
    )


def test_live_check_rejects_model_missing_provider_attestation_status() -> None:
    models = _verified_models()
    del models["data"][0]["provider_attestation_status"]  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "provider attestation status is verified" in result.message
        for result in results
    )


def test_live_check_rejects_plain_model_published_in_required_mode() -> None:
    models = _verified_models()
    models["data"].append(  # type: ignore[union-attr]
        {
            "id": "plain/gpt",
            "confidential": False,
            "attestation_status": "unavailable",
        }
    )

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "plain/gpt is not a verified confidential model" in result.message
        for result in results
    )


def test_live_check_rejects_malformed_non_expected_model_proof() -> None:
    models = _verified_models()
    extra_model = json.loads(json.dumps(models["data"][0]))  # type: ignore[index]
    extra_model["id"] = "tinfoil/extra"
    extra_model["confidentiality"]["model_ids"] = ["tinfoil/extra"]
    proof_claims = extra_model["confidentiality"]["proof_claims"]
    del proof_claims["tls_public_key_fingerprint_sha256"]
    proof_claims["tls_public_key"] = "not-a-digest"
    models["data"].append(extra_model)  # type: ignore[union-attr]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "tinfoil/extra nested provider proof_claims.tls_public_key must be a full sha256 digest"
        in result.message
        for result in results
    )


def test_live_check_rejects_truthy_verified_on_non_expected_model() -> None:
    models = _verified_models()
    extra_model = json.loads(json.dumps(models["data"][0]))  # type: ignore[index]
    extra_model["id"] = "tinfoil/extra"
    extra_model["confidentiality"]["model_ids"] = ["tinfoil/extra"]
    extra_model["confidentiality"]["verified"] = "true"
    models["data"].append(extra_model)  # type: ignore[union-attr]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "tinfoil/extra nested confidentiality.verified is a JSON boolean"
        in result.message
        for result in results
    )


def test_live_check_rejects_non_expected_model_without_matching_provider_status() -> (
    None
):
    models = _verified_models()
    extra_model = json.loads(json.dumps(models["data"][0]))  # type: ignore[index]
    extra_model["id"] = "tinfoil/extra"
    extra_model["confidentiality"]["model_ids"] = ["tinfoil/extra"]
    models["data"].append(extra_model)  # type: ignore[union-attr]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "tinfoil/extra model evidence digest matches provider status"
        in result.message
        for result in results
    )


def test_live_check_rejects_model_nested_confidentiality_without_future_expiry() -> (
    None
):
    models = _verified_models()
    models["data"][0]["confidentiality"]["expires_at"] = 1  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and "nested verifier expiry is in the future" in result.message
        for result in results
    )


def test_live_check_rejects_string_model_nested_confidentiality_expiry() -> None:
    models = _verified_models()
    models["data"][0]["confidentiality"]["expires_at"] = "4102444800"  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "nested verifier expiry must be an integer Unix timestamp" in result.message
        for result in results
    )


def test_live_check_rejects_string_model_nested_confidentiality_verified_at() -> None:
    models = _verified_models()
    models["data"][0]["confidentiality"]["verified_at"] = "1700000000"  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "nested verifier verified_at must be an integer Unix timestamp"
        in result.message
        for result in results
    )


def test_live_check_rejects_future_model_nested_confidentiality_verified_at() -> None:
    models = _verified_models()
    models["data"][0]["confidentiality"]["verified_at"] = int(time.time()) + 60  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok
        and "nested verifier verified_at is not in the future" in result.message
        for result in results
    )


def test_live_check_rejects_local_verification_evidence_digest_mismatch() -> None:
    attestation = _attestation_statement()
    attestation["tee"]["local_verification"]["evidence_digest"] = (
        OTHER_TEE_RUNTIME_EVIDENCE_DIGEST  # type: ignore[index]
    )

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and "local TEE evidence digest matches" in result.message
        for result in results
    )


def test_live_check_rejects_local_verification_claims_digest_mismatch() -> None:
    attestation = _attestation_statement()
    attestation["tee"]["local_verification"]["verified_claims_digest"] = _digest(  # type: ignore[index]
        "other-tee-claims"
    )

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=attestation,
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
    )

    assert any(
        not result.ok and "local TEE claims digest matches" in result.message
        for result in results
    )


def test_live_check_builds_bearer_auth_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROUTSTR_API_KEY", "sk-live-secret")

    headers = live_check.inference_auth_headers(
        bearer_token_env="ROUTSTR_API_KEY",
        cashu_token_env=None,
    )

    assert headers == {"Authorization": "Bearer sk-live-secret"}


def test_live_check_rejects_ambiguous_inference_auth_envs() -> None:
    with pytest.raises(ValueError, match="choose only one"):
        live_check.inference_auth_headers(
            bearer_token_env="ROUTSTR_API_KEY",
            cashu_token_env="ROUTSTR_CASHU_TOKEN",
        )


def test_live_check_accepts_openai_shaped_chat_completion() -> None:
    results = live_check.validate_chat_completion_response(
        "tinfoil/gpt-secure",
        {
            "id": "chatcmpl-live-check",
            "object": "chat.completion",
            "model": "tinfoil/gpt-secure",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        },
    )

    assert all(result.ok for result in results)


def test_live_check_rejects_chat_completion_wrong_model_echo() -> None:
    results = live_check.validate_chat_completion_response(
        "tinfoil/gpt-secure",
        {
            "id": "chatcmpl-live-check",
            "object": "chat.completion",
            "model": "plain/gpt",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        },
    )

    assert any(
        not result.ok and "response model matches requested model" in result.message
        for result in results
    )


def test_live_check_rejects_non_chat_completion_response() -> None:
    results = live_check.validate_chat_completion_response(
        "tinfoil/gpt-secure",
        {"ok": True},
    )

    assert any(not result.ok and "choices" in result.message for result in results)


def test_live_check_skips_inference_when_surface_checks_failed() -> None:
    results = live_check.gate_inference_checks(
        [live_check.CheckResult(False, "public check failed")]
    )

    assert len(results) == 1
    assert results[0].ok is False
    assert "skipped" in results[0].message


def test_live_check_parses_endpoint_specific_inference_route() -> None:
    route = live_check.parse_expected_inference_route(
        "privatemode:privatemode/qwen3-embedding-4b:embeddings"
    )

    assert route.provider == "privatemode"
    assert route.model == "privatemode/qwen3-embedding-4b"
    assert route.endpoint == "/v1/embeddings"


def test_live_check_parses_responses_inference_route() -> None:
    route = live_check.parse_expected_inference_route(
        "tinfoil:tinfoil/gpt-secure:responses"
    )

    assert route.provider == "tinfoil"
    assert route.model == "tinfoil/gpt-secure"
    assert route.endpoint == "/v1/responses"


def test_live_check_rejects_inference_endpoint_not_advertised() -> None:
    route = live_check.ExpectedInferenceRoute(
        provider="tinfoil",
        model="tinfoil/gpt-secure",
        endpoint="/v1/embeddings",
    )

    results = live_check.validate_expected_inference_support(
        models=_verified_models(),
        routes=[route],
    )

    assert any(
        not result.ok and "does not advertise /v1/embeddings" in result.message
        for result in results
    )


def test_live_check_collects_inference_endpoint_support_without_running_inference() -> (
    None
):
    route = live_check.ExpectedInferenceRoute(
        provider="tinfoil",
        model="tinfoil/gpt-secure",
        endpoint="/v1/embeddings",
    )

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
        inference_routes=[route],
    )

    assert any(
        not result.ok and "does not advertise /v1/embeddings" in result.message
        for result in results
    )


def test_live_check_rejects_expected_model_absent_from_provider_catalog() -> None:
    catalog = live_check.ProviderModelCatalog(
        provider_models={"tinfoil": {"other-model"}}
    )

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
        provider_model_catalog=catalog,
    )

    assert any(
        not result.ok
        and "selected model tinfoil/gpt-secure is not present in "
        "confidential-inference provider catalog for tinfoil"
        in result.message
        for result in results
    )


def test_live_check_rejects_public_verified_model_absent_from_provider_catalog() -> None:
    models = _verified_models()
    extra_model = json.loads(json.dumps(models["data"][0]))  # type: ignore[index]
    extra_model["id"] = "tinfoil/not-in-catalog"
    models["data"].append(extra_model)  # type: ignore[index,union-attr]
    catalog = live_check.ProviderModelCatalog(
        provider_models={"tinfoil": {"gpt-secure"}}
    )

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
        provider_model_catalog=catalog,
    )

    assert any(
        not result.ok
        and "public provider-verified model tinfoil/not-in-catalog is not present in "
        "confidential-inference provider catalog for tinfoil"
        in result.message
        for result in results
    )


def test_live_check_rejects_provider_verified_model_absent_from_provider_catalog_when_local_tee_unavailable() -> None:
    models = _verified_models()
    models["data"][0]["id"] = "tinfoil/not-in-catalog"  # type: ignore[index]
    models["data"][0]["confidential"] = False  # type: ignore[index]
    models["data"][0]["attestation_status"] = "unavailable"  # type: ignore[index]
    catalog = live_check.ProviderModelCatalog(
        provider_models={"tinfoil": {"gpt-secure"}}
    )

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
        provider_model_catalog=catalog,
    )

    assert any(
        not result.ok
        and "public provider-verified model tinfoil/not-in-catalog is not present in "
        "confidential-inference provider catalog for tinfoil"
        in result.message
        for result in results
    )


def test_live_check_accepts_ppq_private_dot_dash_provider_catalog_alias() -> None:
    catalog = live_check.ProviderModelCatalog(
        provider_models={"ppq": {"gpt-oss-120b", "kimi-k2-6"}}
    )

    results = live_check.collect_results(
        status=_verified_ppq_private_status(),
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        models=_verified_ppq_private_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("ppq-private", "private/kimi-k2.6")],
        provider_model_catalog=catalog,
    )

    assert not any(
        not result.ok and "confidential-inference provider catalog" in result.message
        for result in results
    )


def test_live_check_accepts_ppq_private_catalog_descriptive_aliases() -> None:
    catalog = live_check.ProviderModelCatalog(
        provider_models={
            "ppq": {
                "gemma-4-31b",
                "llama-3-3-70b",
                "qwen3-vl-30b-a3b",
            }
        }
    )

    expected_results = live_check.validate_expected_provider_catalog_models(
        expected=[
            ("ppq-private", "private/gemma4-31b"),
            ("ppq-private", "private/llama3-3-70b"),
            ("ppq-private", "private/qwen3-vl-30b"),
        ],
        provider_model_catalog=catalog,
    )

    assert expected_results
    assert all(result.ok for result in expected_results)


def test_live_check_does_not_apply_ppq_descriptive_aliases_to_privatemode_catalog() -> None:
    catalog = live_check.ProviderModelCatalog(
        provider_models={"privatemode": {"qwen3-coder-30b-a3b"}}
    )

    results = live_check.validate_expected_provider_catalog_models(
        expected=[("privatemode", "privatemode/qwen3-coder-30b")],
        provider_model_catalog=catalog,
    )

    assert results
    assert not results[0].ok
    assert (
        "selected model privatemode/qwen3-coder-30b is not present in "
        "confidential-inference provider catalog for privatemode"
    ) == results[0].message


def test_live_check_accepts_public_ppq_private_catalog_descriptive_alias() -> None:
    models = _verified_ppq_private_models()
    public_model = models["data"][0]  # type: ignore[index]
    public_model["id"] = "private/qwen3-vl-30b"  # type: ignore[index]
    public_model["confidentiality"]["model_ids"] = ["private/qwen3-vl-30b"]  # type: ignore[index]
    catalog = live_check.ProviderModelCatalog(
        provider_models={"ppq": {"qwen3-vl-30b-a3b"}}
    )

    results = live_check.validate_public_verified_models_in_provider_catalog(
        models=models,
        provider_model_catalog=catalog,
    )

    assert results
    assert all(result.ok for result in results)


def test_live_check_rejects_attested_tinfoil_router_failed_in_attestation_results() -> (
    None
):
    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={
                "tinfoil": {
                    "inference.tinfoil.sh": "failed",
                }
            },
            provider_host_errors={
                "tinfoil": {
                    "inference.tinfoil.sh": (
                        "Trust decision failed: TLS certificate binding failed"
                    ),
                }
            },
        ),
    )

    assert any(
        not result.ok
        and "attestation routing_policy.providers[0] Tinfoil host "
        "inference.tinfoil.sh is not verified in confidential-inference "
        "attestation results"
        in result.message
        for result in results
    )


def test_live_check_strict_external_guardrails_require_provider_catalog() -> None:
    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(_privatemode_routing_policy()),
        expected=[("privatemode", "privatemode/kimi-latest")],
        provider_model_catalog=None,
        attestation_targets_directory=None,
        attestation_results_directory=None,
    )

    assert any(
        not result.ok
        and result.message
        == "confidential-inference provider catalog guardrail is supplied"
        for result in results
    )
    assert not any("Tinfoil" in result.message for result in results)


def test_live_check_strict_external_guardrails_reject_tinfoil_model_release_mismatch() -> (
    None
):
    routing_policy = json.loads(json.dumps(ROUTING_POLICY))
    routing_policy["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "model_attestation_targets"
    ]["tinfoil/gpt-secure"][  # type: ignore[index]
        "attestation_url"
    ] = "https://gpt-secure.tinfoil.example/.well-known/tinfoil-attestation"

    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(routing_policy),
        expected=[("tinfoil", "tinfoil/gpt-secure")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"tinfoil": {"gpt-secure"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={"tinfoil": {"tinfoil-router": "inference.tinfoil.sh"}},
            provider_target_models={"tinfoil": {"tinfoil/gpt-secure"}},
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={
                "tinfoil": {
                    "inference.tinfoil.sh": "verified",
                    "gpt-secure.tinfoil.example": "verified",
                }
            },
            provider_host_errors={},
            provider_model_statuses={
                "tinfoil": {"tinfoil/gpt-secure": "verified"}
            },
            provider_model_details={
                "tinfoil": {
                    "tinfoil/gpt-secure": {
                        "release_digest": _digest("wrong-tinfoil-model-release"),
                    }
                }
            },
        ),
    )

    assert any(
        not result.ok
        and "bind model tinfoil/gpt-secure release digest to deployed policy"
        in result.message
        for result in results
    )


def test_live_check_external_guardrail_report_marks_non_tinfoil_directory_optional(
    tmp_path: Path,
) -> None:
    providers_path = tmp_path / "providers.json"
    providers_path.write_text('{"providers":[]}', encoding="utf-8")

    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=False,
        attestation=_attestation_statement(_privatemode_routing_policy()),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"privatemode": {"kimi-latest"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=None,
        provider_catalog_json=providers_path,
    )

    assert report["strict_external_guardrails"] is True
    assert report["provider_catalog_supplied"] is True
    assert report["tinfoil_directory_required"] is False
    assert report["tinfoil_required_hosts"] == []
    assert report["tinfoil_verified_hosts"] == []
    assert report["ppq_results_required"] is False
    assert report["ppq_required_hosts"] == []
    assert report["ppq_verified_hosts"] == []
    assert report["privatemode_required_hosts"] == []
    assert report["privatemode_verified_hosts"] == []
    assert report["privatemode_results_verified"] is False
    assert report["attestation_targets_supplied"] is False
    assert report["attestation_results_supplied"] is False
    assert report["provider_catalog_path"] == str(providers_path)
    assert report["provider_catalog_sha256"] == (
        "sha256:" + hashlib.sha256(b'{"providers":[]}').hexdigest()
    )


def test_live_check_external_guardrail_report_hashes_tinfoil_directories(
    tmp_path: Path,
) -> None:
    targets_path = tmp_path / "attestation-targets.json"
    results_path = tmp_path / "attestation-results.json"
    targets_path.write_text('{"providers":[]}', encoding="utf-8")
    results_path.write_text('{"checks":[]}', encoding="utf-8")

    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=True,
        needs_ppq_results=False,
        attestation=_attestation_statement(),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"tinfoil": {"gpt-secure"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={},
            provider_target_models={"tinfoil": {"tinfoil/gpt-secure"}},
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={},
            provider_host_errors={},
        ),
        attestation_targets_json=targets_path,
        attestation_results_json=results_path,
        max_attestation_result_age_seconds=86_400,
    )

    assert report["max_attestation_result_age_seconds"] == 86_400
    assert report["attestation_targets_path"] == str(targets_path)
    assert report["attestation_targets_sha256"] == (
        "sha256:" + hashlib.sha256(b'{"providers":[]}').hexdigest()
    )
    assert report["attestation_results_path"] == str(results_path)
    assert report["attestation_results_sha256"] == (
        "sha256:" + hashlib.sha256(b'{"checks":[]}').hexdigest()
    )


def test_live_check_external_guardrail_report_exposes_tinfoil_hosts() -> None:
    routing_policy = json.loads(json.dumps(ROUTING_POLICY))
    routing_policy["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "model_attestation_targets"
    ]["tinfoil/gpt-secure"][  # type: ignore[index]
        "attestation_url"
    ] = "https://gpt-secure.tinfoil.example/.well-known/tinfoil-attestation"
    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=True,
        needs_ppq_results=False,
        attestation=_attestation_statement(routing_policy),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"tinfoil": {"gpt-secure"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={},
            provider_target_models={"tinfoil": {"tinfoil/gpt-secure"}},
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={
                "tinfoil": {
                    "inference.tinfoil.sh": "verified",
                    "gpt-secure.tinfoil.example": "failed",
                }
            },
            provider_host_errors={},
        ),
    )

    assert report["tinfoil_required_hosts"] == [
        "gpt-secure.tinfoil.example",
        "inference.tinfoil.sh",
    ]
    assert report["tinfoil_verified_hosts"] == ["inference.tinfoil.sh"]
    assert report["tinfoil_results_verified"] is False
    assert report["external_guardrails_ready"] is False


def test_live_check_external_guardrail_report_accepts_all_tinfoil_hosts() -> None:
    routing_policy = json.loads(json.dumps(ROUTING_POLICY))
    routing_policy["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "model_attestation_targets"
    ]["tinfoil/gpt-secure"][  # type: ignore[index]
        "attestation_url"
    ] = "https://gpt-secure.tinfoil.example/.well-known/tinfoil-attestation"
    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=True,
        needs_ppq_results=False,
        attestation=_attestation_statement(routing_policy),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"tinfoil": {"gpt-secure"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={},
            provider_target_models={"tinfoil": {"tinfoil/gpt-secure"}},
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={
                "tinfoil": {
                    "inference.tinfoil.sh": "verified",
                    "gpt-secure.tinfoil.example": "verified",
                }
            },
            provider_host_errors={},
            provider_model_statuses={
                "tinfoil": {"tinfoil/gpt-secure": "verified"}
            },
            provider_model_details={
                "tinfoil": {
                    "tinfoil/gpt-secure": {
                        "release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                    }
                }
            },
        ),
    )

    assert report["tinfoil_required_hosts"] == [
        "gpt-secure.tinfoil.example",
        "inference.tinfoil.sh",
    ]
    assert report["tinfoil_verified_hosts"] == [
        "gpt-secure.tinfoil.example",
        "inference.tinfoil.sh",
    ]
    assert report["tinfoil_results_verified"] is True
    assert report["tinfoil_routable_with_full_attestation"] == [
        "tinfoil/gpt-secure"
    ]
    assert report["external_guardrails_ready"] is True


def test_live_check_external_guardrail_report_rejects_tinfoil_model_release_mismatch() -> (
    None
):
    routing_policy = json.loads(json.dumps(ROUTING_POLICY))
    routing_policy["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "model_attestation_targets"
    ]["tinfoil/gpt-secure"][  # type: ignore[index]
        "attestation_url"
    ] = "https://gpt-secure.tinfoil.example/.well-known/tinfoil-attestation"

    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=True,
        needs_ppq_results=False,
        attestation=_attestation_statement(routing_policy),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"tinfoil": {"gpt-secure"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={},
            provider_target_models={"tinfoil": {"tinfoil/gpt-secure"}},
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={
                "tinfoil": {
                    "inference.tinfoil.sh": "verified",
                    "gpt-secure.tinfoil.example": "verified",
                }
            },
            provider_host_errors={},
            provider_model_statuses={
                "tinfoil": {"tinfoil/gpt-secure": "verified"}
            },
            provider_model_details={
                "tinfoil": {
                    "tinfoil/gpt-secure": {
                        "release_digest": _digest("wrong-tinfoil-model-release"),
                    }
                }
            },
        ),
    )

    assert report["tinfoil_verified_models"] == []
    assert report["tinfoil_routable_with_full_attestation"] == []
    assert report["external_guardrails_ready"] is False
    assert any(
        not check["ok"]
        and "bind model tinfoil/gpt-secure release digest to deployed policy"
        in check["message"]
        for check in report["tinfoil_model_release_checks"]["tinfoil/gpt-secure"]
    )


def test_live_check_external_guardrail_report_requires_tinfoil_attestation_target_for_selected_model() -> (
    None
):
    routing_policy = json.loads(json.dumps(ROUTING_POLICY))
    routing_policy["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "model_attestation_targets"
    ]["tinfoil/gpt-secure"][  # type: ignore[index]
        "attestation_url"
    ] = "https://gpt-secure.tinfoil.example/.well-known/tinfoil-attestation"

    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=True,
        needs_ppq_results=False,
        attestation=_attestation_statement(routing_policy),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"tinfoil": {"gpt-secure"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={"tinfoil": {"tinfoil-router": "inference.tinfoil.sh"}},
            provider_target_models={"tinfoil": {"tinfoil/other-model"}},
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={
                "tinfoil": {
                    "inference.tinfoil.sh": "verified",
                    "gpt-secure.tinfoil.example": "verified",
                }
            },
            provider_host_errors={},
            provider_model_statuses={
                "tinfoil": {"tinfoil/gpt-secure": "verified"}
            },
            provider_model_details={
                "tinfoil": {
                    "tinfoil/gpt-secure": {
                        "release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                    }
                }
            },
        ),
        expected_tinfoil_models={"tinfoil/gpt-secure"},
    )

    assert report["tinfoil_required_models"] == ["tinfoil/gpt-secure"]
    assert report["tinfoil_verified_models"] == ["tinfoil/gpt-secure"]
    assert report["tinfoil_target_models_verified"] is False
    assert report["tinfoil_results_verified"] is False
    assert report["external_guardrails_ready"] is False


def test_live_check_external_guardrail_report_requires_expected_tinfoil_model_results() -> (
    None
):
    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=True,
        needs_ppq_results=False,
        attestation=_attestation_statement(),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"tinfoil": {"gpt-secure"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={}
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"tinfoil": {"inference.tinfoil.sh": "verified"}},
            provider_host_errors={},
        ),
        expected_tinfoil_models={"tinfoil/gpt-secure"},
    )

    assert report["tinfoil_required_models"] == ["tinfoil/gpt-secure"]
    assert report["tinfoil_verified_models"] == []
    assert report["tinfoil_results_verified"] is False
    assert report["external_guardrails_ready"] is False


def test_live_check_external_guardrail_report_exposes_ppq_hosts() -> None:
    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=True,
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={
                "ppq": {"api.ppq.ai": "verified", "other.ppq.ai": "verified"}
            },
            provider_host_errors={},
        ),
    )

    assert report["ppq_required_hosts"] == ["api.ppq.ai"]
    assert report["ppq_verified_hosts"] == ["api.ppq.ai"]
    assert report["ppq_required_models"] == ["private/gpt-oss-120b"]
    assert report["ppq_verified_models"] == []
    assert report["ppq_results_verified"] is False
    assert report["external_guardrails_ready"] is False


def test_live_check_external_guardrail_report_exposes_ppq_models() -> None:
    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=True,
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={},
            provider_target_models={"ppq": {"private/gpt-oss-120b"}},
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "verified"}},
            provider_host_errors={},
            provider_model_statuses={
                "ppq": {
                    "private/gpt-oss-120b": "verified",
                    "private/llama3-3-70b": "verified",
                }
            },
            provider_model_errors={},
            provider_model_details={
                "ppq": {
                    "private/gpt-oss-120b": {
                        **PPQ_PRIVATE_BACKEND_VERIFICATION_DETAILS,
                        "backend_release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                    },
                    "private/llama3-3-70b": {
                        **PPQ_PRIVATE_BACKEND_VERIFICATION_DETAILS,
                        "backend_release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                    },
                }
            },
        ),
        expected_ppq_models={"private/gpt-oss-120b"},
    )

    assert report["ppq_required_models"] == ["private/gpt-oss-120b"]
    assert report["ppq_verified_models"] == ["private/gpt-oss-120b"]
    assert report["ppq_routable_with_full_attestation"] == ["private/gpt-oss-120b"]
    assert report["routable_with_full_attestation"] == {
        "ppq-private": ["private/gpt-oss-120b"],
        "privatemode": [],
        "tinfoil": [],
    }
    assert report["ppq_results_verified"] is True
    assert report["external_guardrails_ready"] is True


def test_live_check_external_guardrail_report_requires_ppq_selected_model_rows() -> (
    None
):
    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=True,
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "verified"}},
            provider_host_errors={},
        ),
        expected_ppq_models={"private/gpt-oss-120b"},
    )

    assert report["ppq_required_hosts"] == ["api.ppq.ai"]
    assert report["ppq_verified_hosts"] == ["api.ppq.ai"]
    assert report["ppq_required_models"] == ["private/gpt-oss-120b"]
    assert report["ppq_verified_models"] == []
    assert report["ppq_routable_with_full_attestation"] == []
    assert report["ppq_results_verified"] is False
    assert report["external_guardrails_ready"] is False


def test_live_check_external_guardrail_report_requires_ppq_attestation_targets_for_selected_models() -> (
    None
):
    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=True,
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "verified"}},
            provider_host_errors={},
            provider_model_statuses={"ppq": {"private/gpt-oss-120b": "verified"}},
            provider_model_errors={},
            provider_model_details={
                "ppq": {
                    "private/gpt-oss-120b": {
                        **PPQ_PRIVATE_BACKEND_VERIFICATION_DETAILS,
                        "backend_release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                    }
                }
            },
        ),
        expected_ppq_models={"private/gpt-oss-120b"},
    )

    assert report["ppq_required_models"] == ["private/gpt-oss-120b"]
    assert report["ppq_verified_models"] == []
    assert report["ppq_results_verified"] is False
    assert report["external_guardrails_ready"] is False


def test_live_check_external_guardrail_report_rejects_ppq_wrong_attestation_target_model() -> (
    None
):
    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=True,
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={},
            provider_target_models={"ppq": {"private/llama3-3-70b"}},
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "verified"}},
            provider_host_errors={},
            provider_model_statuses={"ppq": {"private/gpt-oss-120b": "verified"}},
            provider_model_errors={},
            provider_model_details={
                "ppq": {
                    "private/gpt-oss-120b": dict(PPQ_PRIVATE_RESULT_PROVENANCE_DETAILS)
                }
            },
        ),
        expected_ppq_models={"private/gpt-oss-120b"},
    )

    assert report["ppq_required_models"] == ["private/gpt-oss-120b"]
    assert report["ppq_verified_models"] == []
    assert report["ppq_routable_with_full_attestation"] == []
    assert report["ppq_results_verified"] is False
    assert report["external_guardrails_ready"] is False


def test_live_check_external_guardrail_report_uses_attested_ppq_models_without_expectation() -> (
    None
):
    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=True,
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "verified"}},
            provider_host_errors={},
        ),
    )

    assert report["ppq_required_hosts"] == ["api.ppq.ai"]
    assert report["ppq_verified_hosts"] == ["api.ppq.ai"]
    assert report["ppq_required_models"] == ["private/gpt-oss-120b"]
    assert report["ppq_verified_models"] == []
    assert report["ppq_results_verified"] is False
    assert report["external_guardrails_ready"] is False


def test_live_check_strict_external_guardrails_require_ppq_selected_model_rows() -> (
    None
):
    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        expected=[("ppq-private", "private/gpt-oss-120b")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "verified"}},
            provider_host_errors={},
        ),
    )

    assert any(
        not result.ok
        and result.message
        == (
            "confidential-inference PPQ private attestation results include "
            "verified model private/gpt-oss-120b"
        )
        for result in results
    )


def test_live_check_strict_external_guardrails_require_ppq_attestation_targets() -> (
    None
):
    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        expected=[("ppq-private", "private/gpt-oss-120b")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "verified"}},
            provider_host_errors={},
            provider_model_statuses={"ppq": {"private/gpt-oss-120b": "verified"}},
            provider_model_errors={},
            provider_model_details={
                "ppq": {
                    "private/gpt-oss-120b": {
                        **PPQ_PRIVATE_BACKEND_VERIFICATION_DETAILS,
                        "backend_release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                    }
                }
            },
        ),
    )

    assert any(
        not result.ok
        and result.message
        == "confidential-inference PPQ private attestation targets guardrail is supplied"
        for result in results
    )


def test_live_check_strict_external_guardrails_reject_ppq_wrong_attestation_target_model() -> (
    None
):
    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        expected=[("ppq-private", "private/gpt-oss-120b")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={},
            provider_target_models={"ppq": {"private/llama3-3-70b"}},
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "verified"}},
            provider_host_errors={},
            provider_model_statuses={"ppq": {"private/gpt-oss-120b": "verified"}},
            provider_model_errors={},
            provider_model_details={
                "ppq": {
                    "private/gpt-oss-120b": dict(
                        PPQ_PRIVATE_RESULT_PROVENANCE_DETAILS
                    )
                }
            },
        ),
    )

    assert any(
        not result.ok
        and result.message
        == (
            "confidential-inference PPQ private attestation targets include "
            "model private/gpt-oss-120b"
        )
        for result in results
    )


def test_live_check_strict_external_guardrails_require_attested_ppq_model_rows_without_expectation() -> (
    None
):
    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        expected=[],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "verified"}},
            provider_host_errors={},
        ),
    )

    assert any(
        not result.ok
        and result.message
        == (
            "confidential-inference PPQ private attestation results include "
            "verified model private/gpt-oss-120b"
        )
        for result in results
    )


def test_live_check_external_guardrail_report_requires_ppq_release_digest_match() -> (
    None
):
    routing_policy = _ppq_private_routing_policy()
    provider = routing_policy["providers"][0]  # type: ignore[index]
    provider["confidentiality_policy"] = {  # type: ignore[index]
        **PPQ_PRIVATE_CONFIDENTIALITY_POLICY,
        "expected_release_digest": PROVIDER_RELEASE_DIGEST,
    }

    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=True,
        attestation=_attestation_statement(routing_policy),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "verified"}},
            provider_host_errors={},
            provider_model_statuses={"ppq": {"private/gpt-oss-120b": "verified"}},
            provider_model_errors={},
            provider_model_details={
                "ppq": {
                    "private/gpt-oss-120b": {
                        **PPQ_PRIVATE_BACKEND_VERIFICATION_DETAILS,
                        "release_digest": PROVIDER_RELEASE_DIGEST,
                        "backend_release_digest": _digest("wrong-backend-release"),
                    }
                }
            },
        ),
        expected_ppq_models={"private/gpt-oss-120b"},
    )

    assert report["ppq_required_models"] == ["private/gpt-oss-120b"]
    assert report["ppq_verified_models"] == []
    assert report["ppq_results_verified"] is False
    assert report["external_guardrails_ready"] is False
    assert report["ppq_model_release_checks"]["private/gpt-oss-120b"] == [
        {
            "ok": True,
            "message": (
                "confidential-inference PPQ private attestation results bind model "
                "private/gpt-oss-120b router release digest to deployed policy"
            ),
        },
        {
            "ok": False,
            "message": (
                "confidential-inference PPQ private attestation results bind model "
                "private/gpt-oss-120b backend release digest to deployed policy"
            ),
        },
        {
            "ok": True,
            "message": (
                "confidential-inference PPQ private attestation results prove model "
                "private/gpt-oss-120b backend_tls_matches"
            ),
        },
        {
            "ok": True,
            "message": (
                "confidential-inference PPQ private attestation results prove model "
                "private/gpt-oss-120b backend_sigstore_match"
            ),
        },
        {
            "ok": True,
            "message": (
                "confidential-inference PPQ private attestation results prove model "
                "private/gpt-oss-120b backend_sigstore_bundle_verified"
            ),
        },
        {
            "ok": True,
            "message": (
                "confidential-inference PPQ private attestation results prove model "
                "private/gpt-oss-120b backend_images_verified"
            ),
        },
    ]


def test_live_check_external_guardrail_report_requires_ppq_backend_evidence() -> None:
    routing_policy = _ppq_private_routing_policy()
    provider = routing_policy["providers"][0]  # type: ignore[index]
    provider["confidentiality_policy"] = {  # type: ignore[index]
        **PPQ_PRIVATE_CONFIDENTIALITY_POLICY,
        "expected_release_digest": PROVIDER_RELEASE_DIGEST,
    }

    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=True,
        attestation=_attestation_statement(routing_policy),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "verified"}},
            provider_host_errors={},
            provider_model_statuses={"ppq": {"private/gpt-oss-120b": "verified"}},
            provider_model_errors={},
            provider_model_details={
                "ppq": {
                    "private/gpt-oss-120b": {
                        "release_digest": PROVIDER_RELEASE_DIGEST,
                        "backend_release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                        "backend_tls_matches": True,
                        "backend_sigstore_match": False,
                        "backend_sigstore_bundle_verified": True,
                    }
                }
            },
        ),
        expected_ppq_models={"private/gpt-oss-120b"},
    )

    assert report["ppq_verified_models"] == []
    assert report["ppq_results_verified"] is False
    assert report["external_guardrails_ready"] is False
    assert {
        "ok": False,
        "message": (
            "confidential-inference PPQ private attestation results prove model "
            "private/gpt-oss-120b backend_sigstore_match"
        ),
    } in report["ppq_model_release_checks"]["private/gpt-oss-120b"]
    assert {
        "ok": False,
        "message": (
            "confidential-inference PPQ private attestation results prove model "
            "private/gpt-oss-120b backend_images_verified"
        ),
    } in report["ppq_model_release_checks"]["private/gpt-oss-120b"]


def test_live_check_external_guardrail_report_requires_ppq_host_and_model_evidence() -> (
    None
):
    routing_policy = _ppq_private_routing_policy()
    provider = routing_policy["providers"][0]  # type: ignore[index]
    provider["confidentiality_policy"] = {  # type: ignore[index]
        **PPQ_PRIVATE_CONFIDENTIALITY_POLICY,
        "expected_release_digest": PROVIDER_RELEASE_DIGEST,
    }

    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=True,
        attestation=_attestation_statement(routing_policy),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "unreachable"}},
            provider_host_errors={"ppq": {"api.ppq.ai": "GitHub rate limit"}},
            provider_model_statuses={
                "ppq": {
                    "private/gpt-oss-120b": "verified",
                    "private/kimi-k2-6": "unreachable",
                }
            },
            provider_model_errors={"ppq": {"private/kimi-k2-6": "GitHub rate limit"}},
            provider_model_details={
                "ppq": {
                    "private/gpt-oss-120b": {
                        **PPQ_PRIVATE_BACKEND_VERIFICATION_DETAILS,
                        "release_digest": PROVIDER_RELEASE_DIGEST,
                        "backend_release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                    }
                }
            },
        ),
        expected_ppq_models={"private/gpt-oss-120b"},
    )

    assert report["ppq_verified_hosts"] == []
    assert report["ppq_verified_models"] == []
    assert report["ppq_results_verified"] is False
    assert report["external_guardrails_ready"] is False


def test_live_check_external_guardrail_report_rejects_wrong_ppq_model() -> None:
    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=True,
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "verified"}},
            provider_host_errors={},
            provider_model_statuses={"ppq": {"private/llama3-3-70b": "verified"}},
            provider_model_errors={},
        ),
        expected_ppq_models={"private/gpt-oss-120b"},
    )

    assert report["ppq_required_models"] == ["private/gpt-oss-120b"]
    assert report["ppq_verified_models"] == []
    assert report["ppq_results_verified"] is False
    assert report["external_guardrails_ready"] is False


def test_live_check_external_guardrail_report_rejects_wrong_ppq_verified_host() -> None:
    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=True,
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"other.ppq.ai": "verified"}},
            provider_host_errors={},
        ),
    )

    assert report["ppq_required_hosts"] == ["api.ppq.ai"]
    assert report["ppq_verified_hosts"] == []
    assert report["ppq_results_verified"] is False
    assert report["external_guardrails_ready"] is False


def test_live_check_external_guardrail_report_requires_privatemode_policy_bound_model_evidence() -> (
    None
):
    routing_policy = _privatemode_routing_policy()
    routing_policy["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "manifest_url"
    ] = "https://cdn.confidential.cloud/privatemode/v2/manifest.json"

    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=False,
        attestation=_attestation_statement(routing_policy),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"privatemode": {"kimi-latest"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={
                "privatemode": {
                    "cdn.confidential.cloud": "verified",
                    "docs.privatemode.ai": "failed",
                }
            },
            provider_host_errors={},
            provider_model_statuses={
                "privatemode": {"privatemode/kimi-latest": "verified"}
            },
            provider_model_errors={},
        ),
    )

    assert report["privatemode_required_hosts"] == ["cdn.confidential.cloud"]
    assert report["privatemode_verified_hosts"] == ["cdn.confidential.cloud"]
    assert report["privatemode_required_models"] == ["privatemode/kimi-latest"]
    assert report["privatemode_verified_models"] == []
    assert report["privatemode_results_verified"] is False
    assert report["external_guardrails_ready"] is False


def test_live_check_external_guardrail_report_requires_privatemode_attestation_targets_for_selected_models() -> (
    None
):
    routing_policy = _privatemode_routing_policy()
    routing_policy["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "manifest_url"
    ] = "https://cdn.confidential.cloud/privatemode/v2/manifest.json"

    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=False,
        attestation=_attestation_statement(routing_policy),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"privatemode": {"kimi-latest"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={
                "privatemode": {"cdn.confidential.cloud": "verified"}
            },
            provider_host_errors={},
            provider_model_statuses={
                "privatemode": {"privatemode/kimi-latest": "verified"}
            },
            provider_model_errors={},
            provider_model_details={
                "privatemode": {
                    "privatemode/kimi-latest": dict(PROVIDER_PRIVATEMODE_PROOF_CLAIMS)
                }
            },
        ),
    )

    assert report["privatemode_required_models"] == ["privatemode/kimi-latest"]
    assert report["privatemode_verified_models"] == []
    assert report["privatemode_results_verified"] is False
    assert report["external_guardrails_ready"] is False


def test_live_check_external_guardrail_report_rejects_privatemode_wrong_attestation_target_model() -> (
    None
):
    routing_policy = _privatemode_routing_policy()
    routing_policy["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "manifest_url"
    ] = "https://cdn.confidential.cloud/privatemode/v2/manifest.json"

    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=False,
        attestation=_attestation_statement(routing_policy),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"privatemode": {"kimi-latest"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={},
            provider_target_models={"privatemode": {"privatemode/other-model"}},
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={
                "privatemode": {"cdn.confidential.cloud": "verified"}
            },
            provider_host_errors={},
            provider_model_statuses={
                "privatemode": {"privatemode/kimi-latest": "verified"}
            },
            provider_model_errors={},
            provider_model_details={
                "privatemode": {
                    "privatemode/kimi-latest": dict(PROVIDER_PRIVATEMODE_PROOF_CLAIMS)
                }
            },
        ),
    )

    assert report["privatemode_required_models"] == ["privatemode/kimi-latest"]
    assert report["privatemode_verified_models"] == []
    assert report["privatemode_routable_with_full_attestation"] == []
    assert report["privatemode_results_verified"] is False
    assert report["external_guardrails_ready"] is False


def test_live_check_external_guardrail_report_accepts_privatemode_policy_bound_model_evidence() -> (
    None
):
    routing_policy = _privatemode_routing_policy()
    routing_policy["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "manifest_url"
    ] = "https://cdn.confidential.cloud/privatemode/v2/manifest.json"

    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=False,
        attestation=_attestation_statement(routing_policy),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"privatemode": {"kimi-latest"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={},
            provider_target_models={"privatemode": {"privatemode/kimi-latest"}},
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={
                "privatemode": {"cdn.confidential.cloud": "verified"}
            },
            provider_host_errors={},
            provider_model_statuses={
                "privatemode": {"privatemode/kimi-latest": "verified"}
            },
            provider_model_errors={},
            provider_model_details={
                "privatemode": {
                    "privatemode/kimi-latest": dict(PROVIDER_PRIVATEMODE_PROOF_CLAIMS)
                }
            },
        ),
    )

    assert report["privatemode_required_hosts"] == ["cdn.confidential.cloud"]
    assert report["privatemode_verified_hosts"] == ["cdn.confidential.cloud"]
    assert report["privatemode_required_models"] == ["privatemode/kimi-latest"]
    assert report["privatemode_verified_models"] == ["privatemode/kimi-latest"]
    assert report["privatemode_routable_with_full_attestation"] == [
        "privatemode/kimi-latest"
    ]
    assert report["routable_with_full_attestation"] == {
        "ppq-private": [],
        "privatemode": ["privatemode/kimi-latest"],
        "tinfoil": [],
    }
    assert all(
        check["ok"]
        for check in report["privatemode_model_policy_checks"][
            "privatemode/kimi-latest"
        ]
    )
    assert report["privatemode_results_verified"] is True
    assert report["external_guardrails_ready"] is True


def test_live_check_external_guardrail_report_requires_privatemode_model_rows() -> None:
    routing_policy = _privatemode_routing_policy()
    routing_policy["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "manifest_url"
    ] = "https://cdn.confidential.cloud/privatemode/v2/manifest.json"

    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=False,
        attestation=_attestation_statement(routing_policy),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"privatemode": {"kimi-latest"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={
                "privatemode": {"cdn.confidential.cloud": "verified"}
            },
            provider_host_errors={},
            provider_model_statuses={},
            provider_model_errors={},
        ),
    )

    assert report["privatemode_required_hosts"] == ["cdn.confidential.cloud"]
    assert report["privatemode_verified_hosts"] == ["cdn.confidential.cloud"]
    assert report["privatemode_required_models"] == ["privatemode/kimi-latest"]
    assert report["privatemode_verified_models"] == []
    assert report["privatemode_results_verified"] is False
    assert report["external_guardrails_ready"] is False


def test_live_check_external_guardrail_report_includes_expected_privatemode_models() -> (
    None
):
    routing_policy = _privatemode_routing_policy()
    routing_policy["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "manifest_url"
    ] = "https://cdn.confidential.cloud/privatemode/v2/manifest.json"
    routing_policy["providers"][0]["confidentiality"][  # type: ignore[index]
        "model_ids"
    ] = []

    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=False,
        attestation=_attestation_statement(routing_policy),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"privatemode": {"kimi-latest"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={
                "privatemode": {"cdn.confidential.cloud": "verified"}
            },
            provider_host_errors={},
            provider_model_statuses={},
            provider_model_errors={},
        ),
        expected_privatemode_models={"privatemode/kimi-latest"},
    )

    assert report["privatemode_required_models"] == ["privatemode/kimi-latest"]
    assert report["privatemode_verified_models"] == []
    assert report["privatemode_results_verified"] is False
    assert report["external_guardrails_ready"] is False


def test_live_check_external_guardrail_report_requires_expected_privatemode_evidence_without_hosts() -> (
    None
):
    routing_policy = _privatemode_routing_policy()
    routing_policy["providers"][0]["confidentiality_policy"] = dict(  # type: ignore[index]
        PRIVATEMODE_CONFIDENTIALITY_POLICY
    )

    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=False,
        attestation=_attestation_statement(routing_policy),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"privatemode": {"kimi-latest"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={},
            provider_host_errors={},
            provider_model_statuses={
                "privatemode": {"privatemode/kimi-latest": "verified"}
            },
            provider_model_errors={},
            provider_model_details={
                "privatemode": {
                    "privatemode/kimi-latest": dict(PROVIDER_PRIVATEMODE_PROOF_CLAIMS)
                }
            },
        ),
        expected_privatemode_models={"privatemode/kimi-latest"},
    )

    assert report["privatemode_required_hosts"] == []
    assert report["privatemode_required_models"] == ["privatemode/kimi-latest"]
    assert report["privatemode_verified_models"] == []
    assert report["privatemode_results_verified"] is False
    assert report["external_guardrails_ready"] is False


def test_live_check_external_guardrail_report_accepts_privatemode_model_evidence_without_public_host() -> (
    None
):
    routing_policy = _privatemode_routing_policy()
    routing_policy["providers"][0]["confidentiality_policy"] = dict(  # type: ignore[index]
        PRIVATEMODE_CONFIDENTIALITY_POLICY
    )

    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=False,
        attestation=_attestation_statement(routing_policy),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"privatemode": {"kimi-latest"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={},
            provider_target_models={"privatemode": {"privatemode/kimi-latest"}},
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={},
            provider_host_errors={},
            provider_model_statuses={
                "privatemode": {"privatemode/kimi-latest": "verified"}
            },
            provider_model_errors={},
            provider_model_details={
                "privatemode": {
                    "privatemode/kimi-latest": dict(PROVIDER_PRIVATEMODE_PROOF_CLAIMS)
                }
            },
        ),
        expected_privatemode_models={"privatemode/kimi-latest"},
    )

    assert report["privatemode_required_hosts"] == []
    assert report["privatemode_verified_hosts"] == []
    assert report["privatemode_required_models"] == ["privatemode/kimi-latest"]
    assert report["privatemode_verified_models"] == ["privatemode/kimi-latest"]
    assert report["privatemode_results_verified"] is True
    assert report["privatemode_routable_with_full_attestation"] == [
        "privatemode/kimi-latest"
    ]
    assert report["external_guardrails_ready"] is True


def test_live_check_external_guardrail_report_rejects_missing_privatemode_host() -> None:
    routing_policy = _privatemode_routing_policy()
    routing_policy["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "manifest_url"
    ] = "https://cdn.confidential.cloud/privatemode/v2/manifest.json"

    report = live_check.external_guardrail_report(
        strict_external_guardrails=True,
        needs_tinfoil_directory=False,
        needs_ppq_results=False,
        attestation=_attestation_statement(routing_policy),
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"privatemode": {"kimi-latest"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"privatemode": {"docs.privatemode.ai": "verified"}},
            provider_host_errors={},
        ),
    )

    assert report["privatemode_required_hosts"] == ["cdn.confidential.cloud"]
    assert report["privatemode_verified_hosts"] == []
    assert report["privatemode_results_verified"] is False
    assert report["external_guardrails_ready"] is False


def test_live_check_strict_external_guardrails_require_tinfoil_directories() -> None:
    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(),
        expected=[("tinfoil", "tinfoil/gpt-secure")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"tinfoil": {"gpt-secure"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=None,
    )

    assert any(
        not result.ok
        and result.message
        == "confidential-inference Tinfoil attestation targets guardrail is supplied"
        for result in results
    )
    assert any(
        not result.ok
        and result.message
        == "confidential-inference Tinfoil attestation results guardrail is supplied"
        for result in results
    )


def test_live_check_strict_external_guardrails_require_expected_tinfoil_model_results() -> (
    None
):
    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(),
        expected=[("tinfoil", "tinfoil/gpt-secure")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"tinfoil": {"gpt-secure"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={"tinfoil": {"tinfoil-router": "inference.tinfoil.sh"}},
            provider_target_models={"tinfoil": {"tinfoil/gpt-secure"}},
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"tinfoil": {"inference.tinfoil.sh": "verified"}},
            provider_host_errors={},
        ),
    )

    assert any(
        not result.ok
        and result.message
        == (
            "confidential-inference Tinfoil attestation results include "
            "verified model tinfoil/gpt-secure"
        )
        for result in results
    )


def test_live_check_strict_external_guardrails_require_tinfoil_router_target() -> (
    None
):
    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(),
        expected=[("tinfoil", "tinfoil/gpt-secure")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"tinfoil": {"gpt-secure"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={
                "tinfoil": {"tinfoil-gpt-secure": "gpt-secure.tinfoil.example"}
            },
            provider_target_models={"tinfoil": {"tinfoil/gpt-secure"}},
            provider_check_models={
                "tinfoil": {"tinfoil-gpt-secure": "tinfoil/gpt-secure"}
            },
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={
                "tinfoil": {"gpt-secure.tinfoil.example": "verified"}
            },
            provider_host_errors={},
            provider_model_statuses={
                "tinfoil": {"tinfoil/gpt-secure": "verified"}
            },
        ),
    )

    assert any(
        not result.ok
        and result.message
        == "confidential-inference Tinfoil attestation targets include provider-level router"
        for result in results
    )
    assert any(
        not result.ok
        and result.message
        == "confidential-inference Tinfoil attestation results include verified provider-level router"
        for result in results
    )


def test_live_check_tinfoil_targets_infer_model_from_known_host_without_model_field(
    tmp_path: Path,
) -> None:
    targets_path = tmp_path / "attestation-targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "slug": "tinfoil",
                        "checks": [
                            {
                                "id": "tinfoil-gpt-oss-120b",
                                "label": "GPT-OSS 120B enclave",
                                "type": "tinfoil",
                                "host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                                "repo": "tinfoilsh/confidential-gpt-oss-120b",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    targets = live_check.load_attestation_targets_directory(targets_path)

    assert "tinfoil/gpt-oss-120b" in live_check.target_directory_tinfoil_models(
        targets
    )
    assert (
        targets.provider_check_models["tinfoil"]["tinfoil-gpt-oss-120b"]
        == "tinfoil/gpt-oss-120b"
    )


def test_live_check_strict_external_guardrails_accept_known_tinfoil_model_alias() -> (
    None
):
    routing_policy = json.loads(json.dumps(ROUTING_POLICY))
    provider = routing_policy["providers"][0]  # type: ignore[index]
    provider["confidentiality"]["model_ids"] = ["tinfoil/llama-3.3-70b"]
    provider["confidentiality_policy"]["model_attestation_targets"] = {
        "tinfoil/llama-3.3-70b": {
            "attestation_url": "https://llama3-3-70b.tinfoil.containers.tinfoil.dev/.well-known/tinfoil-attestation",
            "expected_release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
        }
    }

    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(routing_policy),
        expected=[("tinfoil", "tinfoil/llama-3.3-70b")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"tinfoil": {"llama-3-3-70b"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={"tinfoil": {"tinfoil-llama": "inference.tinfoil.sh"}},
            provider_target_models={"tinfoil": {"tinfoil/llama3-3-70b"}},
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"tinfoil": {"inference.tinfoil.sh": "verified"}},
            provider_host_errors={},
            provider_model_statuses={
                "tinfoil": {"tinfoil/llama3-3-70b": "verified"}
            },
            provider_model_details={
                "tinfoil": {
                    "tinfoil/llama-3.3-70b": {
                        "release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                    }
                }
            },
        ),
    )

    tinfoil_model_results = [
        result
        for result in results
        if "Tinfoil attestation" in result.message
        and "model tinfoil/llama-3.3-70b" in result.message
    ]
    assert tinfoil_model_results
    assert all(result.ok for result in tinfoil_model_results)


def test_live_check_strict_external_guardrails_accept_tinfoil_alias_keyed_release_details() -> (
    None
):
    routing_policy = json.loads(json.dumps(ROUTING_POLICY))
    provider = routing_policy["providers"][0]  # type: ignore[index]
    provider["confidentiality"]["model_ids"] = ["tinfoil/llama-3.3-70b"]
    provider["confidentiality_policy"]["model_attestation_targets"] = {
        "tinfoil/llama-3.3-70b": {
            "attestation_url": "https://llama3-3-70b.tinfoil.containers.tinfoil.dev/.well-known/tinfoil-attestation",
            "expected_release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
        }
    }

    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(routing_policy),
        expected=[("tinfoil", "tinfoil/llama-3.3-70b")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"tinfoil": {"llama-3-3-70b"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={"tinfoil": {"tinfoil-router": "inference.tinfoil.sh"}},
            provider_target_models={"tinfoil": {"tinfoil/llama3-3-70b"}},
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"tinfoil": {"inference.tinfoil.sh": "verified"}},
            provider_host_errors={},
            provider_model_statuses={
                "tinfoil": {"tinfoil/llama3-3-70b": "verified"}
            },
            provider_model_details={
                "tinfoil": {
                    "tinfoil/llama3-3-70b": {
                        "release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                    }
                }
            },
        ),
    )

    release_results = [
        result
        for result in results
        if "bind model tinfoil/llama-3.3-70b release digest" in result.message
    ]
    assert release_results
    assert all(result.ok for result in release_results)


def test_live_check_strict_external_guardrails_reject_ambiguous_tinfoil_alias_release_details() -> (
    None
):
    routing_policy = json.loads(json.dumps(ROUTING_POLICY))
    provider = routing_policy["providers"][0]  # type: ignore[index]
    provider["confidentiality"]["model_ids"] = ["tinfoil/llama-3.3-70b"]
    provider["confidentiality_policy"]["model_attestation_targets"] = {
        "tinfoil/llama-3.3-70b": {
            "attestation_url": "https://llama3-3-70b.tinfoil.containers.tinfoil.dev/.well-known/tinfoil-attestation",
            "expected_release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
        }
    }

    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(routing_policy),
        expected=[("tinfoil", "tinfoil/llama-3.3-70b")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"tinfoil": {"llama-3-3-70b"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={"tinfoil": {"tinfoil-router": "inference.tinfoil.sh"}},
            provider_target_models={"tinfoil": {"tinfoil/llama3-3-70b"}},
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"tinfoil": {"inference.tinfoil.sh": "verified"}},
            provider_host_errors={},
            provider_model_statuses={
                "tinfoil": {
                    "tinfoil/llama3-3-70b": "verified",
                    "tinfoil/llama-3-3-70b": "verified",
                }
            },
            provider_model_details={
                "tinfoil": {
                    "tinfoil/llama3-3-70b": {
                        "release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                    },
                    "tinfoil/llama-3-3-70b": {
                        "release_digest": _digest("wrong-llama-alias-release"),
                    },
                }
            },
        ),
    )

    assert any(
        not result.ok
        and "bind model tinfoil/llama-3.3-70b release digest" in result.message
        for result in results
    )


def test_live_check_ignores_tinfoil_router_model_in_target_directory(
    tmp_path: Path,
) -> None:
    targets_path = tmp_path / "attestation-targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "slug": "tinfoil",
                        "checks": [
                            {
                                "id": "tinfoil-router",
                                "label": "Router (inference.tinfoil.sh)",
                                "type": "tinfoil",
                                "host": "inference.tinfoil.sh",
                                "model": "tinfoil/gpt-secure",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    targets = live_check.load_attestation_targets_directory(targets_path)

    assert targets.provider_check_hosts["tinfoil"]["tinfoil-router"] == (
        "inference.tinfoil.sh"
    )
    assert targets.provider_check_models.get("tinfoil", {}) == {}
    assert targets.provider_target_models.get("tinfoil", set()) == set()


def test_live_check_ignores_tinfoil_router_model_in_result_directory(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "tinfoil",
                        "id": "tinfoil-router",
                        "label": "Router (inference.tinfoil.sh)",
                        "host": "inference.tinfoil.sh",
                        "model": "tinfoil/gpt-secure",
                        "status": "verified",
                        "release_digest": PROVIDER_RELEASE_DIGEST,
                        "tls_matches": True,
                        "sigstore_match": True,
                        "sigstore_bundle_verified": True,
                        "images_verified": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    results = live_check.load_attestation_results_directory(results_path)

    assert results.provider_host_statuses["tinfoil"]["inference.tinfoil.sh"] == (
        "verified"
    )
    assert results.provider_model_statuses.get("tinfoil", {}) == {}


def test_live_check_maps_ppq_private_attestation_result_id_to_target_host(
    tmp_path: Path,
) -> None:
    targets_path = tmp_path / "attestation-targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "slug": "ppq",
                        "checks": [
                            {
                                "id": "ppq-private-router",
                                "type": "ppq",
                                "host": "api.ppq.ai",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "ppq",
                        "id": "ppq-private-router",
                        "status": "verified",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    targets_directory = live_check.load_attestation_targets_directory(targets_path)
    results_directory = live_check.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets_directory,
    )

    assert results_directory.provider_host_statuses["ppq"]["api.ppq.ai"] == "verified"


def test_live_check_rejects_tinfoil_attestation_result_without_release_digest(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "tinfoil",
                        "id": "tinfoil-gpt-secure",
                        "host": "secure-model.tinfoil.sh",
                        "model": "tinfoil/gpt-secure",
                        "status": "verified",
                        "tls_matches": True,
                        "sigstore_match": True,
                        "sigstore_bundle_verified": True,
                        "images_verified": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    results = live_check.load_attestation_results_directory(results_path)

    assert results.provider_host_statuses["tinfoil"]["secure-model.tinfoil.sh"] == (
        "failed"
    )
    assert results.provider_model_statuses["tinfoil"]["tinfoil/gpt-secure"] == (
        "failed"
    )
    assert (
        results.provider_model_errors["tinfoil"]["tinfoil/gpt-secure"]
        == "release_digest is missing or invalid in "
        "confidential-inference attestation results"
    )


def test_live_check_rejects_ppq_private_attestation_result_without_release_digests(
    tmp_path: Path,
) -> None:
    targets_path = tmp_path / "attestation-targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "slug": "ppq",
                        "checks": [
                            {
                                "id": "ppq-gpt-oss-120b",
                                "type": "ppq",
                                "host": "api.ppq.ai",
                                "model": "private/gpt-oss-120b",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "ppq",
                        "id": "ppq-gpt-oss-120b",
                        "host": "api.ppq.ai",
                        "model": "private/gpt-oss-120b",
                        "status": "verified",
                        "backend_tls_matches": True,
                        "backend_sigstore_match": True,
                        "backend_sigstore_bundle_verified": True,
                        "backend_images_verified": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    targets = live_check.load_attestation_targets_directory(targets_path)
    results = live_check.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets,
    )

    assert results.provider_model_statuses["ppq"]["private/gpt-oss-120b"] == "failed"
    assert (
        results.provider_model_errors["ppq"]["private/gpt-oss-120b"]
        == "release_digest is missing or invalid in "
        "confidential-inference attestation results"
    )


def test_live_check_ignores_duplicate_attestation_target_check_ids(
    tmp_path: Path,
) -> None:
    targets_path = tmp_path / "attestation-targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "slug": "ppq",
                        "checks": [
                            {
                                "id": "ppq-private-model",
                                "type": "ppq",
                                "host": "api.ppq.ai",
                                "model": "private/gpt-oss-120b",
                            },
                            {
                                "id": "ppq-private-model",
                                "type": "ppq",
                                "host": "api.ppq.ai",
                                "model": "private/llama3-3-70b",
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "ppq",
                        "id": "ppq-private-model",
                        "status": "verified",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    targets_directory = live_check.load_attestation_targets_directory(targets_path)
    results_directory = live_check.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets_directory,
    )

    assert targets_directory.provider_check_hosts.get("ppq", {}) == {}
    assert targets_directory.provider_check_models.get("ppq", {}) == {}
    assert targets_directory.provider_target_models.get("ppq", set()) == set()
    assert results_directory.provider_host_statuses == {}
    assert results_directory.provider_model_statuses == {}


def test_live_check_downgrades_stale_attestation_result_rows(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "ppq",
                        "host": "api.ppq.ai",
                        "status": "verified",
                        "ts": "2026-01-01T00:00:00+00:00",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    results = live_check.load_attestation_results_directory(
        results_path,
        max_result_age_seconds=60,
        now=datetime(2026, 1, 1, 0, 2, 1, tzinfo=timezone.utc),
    )

    assert results.provider_host_statuses["ppq"]["api.ppq.ai"] == "failed"
    assert (
        results.provider_host_errors["ppq"]["api.ppq.ai"]
        == "attestation result is stale: age_seconds=121 max_age_seconds=60"
    )


def test_live_check_accepts_fresh_attestation_result_last_run_fallback(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "last_run": "2026-01-01T00:00:00.123456789+00:00",
                "checks": [
                    {
                        "provider": "ppq",
                        "host": "api.ppq.ai",
                        "status": "verified",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    results = live_check.load_attestation_results_directory(
        results_path,
        max_result_age_seconds=60,
        now=datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc),
    )

    assert results.provider_host_statuses["ppq"]["api.ppq.ai"] == "verified"
    assert "ppq" not in results.provider_host_errors


def test_live_check_downgrades_malformed_attestation_result_row_timestamp(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "last_run": "2026-01-01T00:00:00+00:00",
                "checks": [
                    {
                        "provider": "ppq",
                        "host": "api.ppq.ai",
                        "status": "verified",
                        "ts": "not-a-timestamp",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    results = live_check.load_attestation_results_directory(
        results_path,
        max_result_age_seconds=60,
        now=datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc),
    )

    assert results.provider_host_statuses["ppq"]["api.ppq.ai"] == "failed"
    assert (
        results.provider_host_errors["ppq"]["api.ppq.ai"]
        == "attestation result timestamp is missing or invalid"
    )


def test_live_check_downgrades_duplicate_attestation_result_host_rows(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {"provider": "ppq", "host": "api.ppq.ai", "status": "failed"},
                    {"provider": "ppq", "host": "api.ppq.ai", "status": "verified"},
                ],
            }
        ),
        encoding="utf-8",
    )

    results = live_check.load_attestation_results_directory(results_path)

    assert results.provider_host_statuses["ppq"]["api.ppq.ai"] == "failed"
    assert (
        results.provider_host_errors["ppq"]["api.ppq.ai"]
        == "duplicate attestation result rows for provider ppq host api.ppq.ai"
    )


def test_live_check_allows_ppq_model_rows_to_share_private_api_host(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "ppq",
                        "host": "api.ppq.ai",
                        "model": "private/gpt-oss-120b",
                        "status": "verified",
                        **PPQ_PRIVATE_RESULT_PROVENANCE_DETAILS,
                    },
                    {
                        "provider": "ppq",
                        "host": "api.ppq.ai",
                        "model": "private/kimi-k2-6",
                        "status": "verified",
                        **PPQ_PRIVATE_RESULT_PROVENANCE_DETAILS,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    results = live_check.load_attestation_results_directory(results_path)

    assert results.provider_host_statuses["ppq"]["api.ppq.ai"] == "verified"
    assert "api.ppq.ai" not in results.provider_host_errors.get("ppq", {})
    assert results.provider_model_statuses["ppq"] == {
        "private/gpt-oss-120b": "verified",
        "private/kimi-k2-6": "verified",
    }


def test_live_check_ppq_shared_host_uses_any_verified_model_row(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "ppq",
                        "host": "api.ppq.ai",
                        "model": "private/kimi-k2-6",
                        "status": "failed",
                        "error": "backend unavailable",
                    },
                    {
                        "provider": "ppq",
                        "host": "api.ppq.ai",
                        "model": "private/gpt-oss-120b",
                        "status": "verified",
                        **PPQ_PRIVATE_RESULT_PROVENANCE_DETAILS,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    results = live_check.load_attestation_results_directory(results_path)

    assert results.provider_host_statuses["ppq"]["api.ppq.ai"] == "verified"
    assert "api.ppq.ai" not in results.provider_host_errors.get("ppq", {})
    assert results.provider_model_statuses["ppq"] == {
        "private/kimi-k2-6": "failed",
        "private/gpt-oss-120b": "verified",
    }
    assert (
        results.provider_model_errors["ppq"]["private/kimi-k2-6"]
        == "backend unavailable"
    )


def test_live_check_downgrades_duplicate_attestation_result_model_rows(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "ppq",
                        "host": "api.ppq.ai",
                        "model": "private/gpt-oss-120b",
                        "status": "failed",
                    },
                    {
                        "provider": "ppq",
                        "host": "api2.ppq.ai",
                        "model": "private/gpt-oss-120b",
                        "status": "verified",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    results = live_check.load_attestation_results_directory(results_path)

    assert (
        results.provider_model_statuses["ppq"]["private/gpt-oss-120b"] == "failed"
    )
    assert (
        results.provider_model_errors["ppq"]["private/gpt-oss-120b"]
        == "duplicate attestation result rows for provider ppq model private/gpt-oss-120b"
    )


def test_live_check_downgrades_duplicate_attestation_result_check_ids(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "ppq",
                        "id": "ppq-private-model",
                        "host": "api.ppq.ai",
                        "model": "private/gpt-oss-120b",
                        "status": "verified",
                    },
                    {
                        "provider": "ppq",
                        "id": "ppq-private-model",
                        "host": "api.ppq.ai",
                        "model": "private/llama3-3-70b",
                        "status": "verified",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    results = live_check.load_attestation_results_directory(results_path)

    assert results.provider_host_statuses["ppq"]["api.ppq.ai"] == "failed"
    assert (
        results.provider_host_errors["ppq"]["api.ppq.ai"]
        == "duplicate attestation result rows for provider ppq check ppq-private-model"
    )
    assert results.provider_model_statuses["ppq"] == {
        "private/gpt-oss-120b": "failed",
        "private/llama3-3-70b": "failed",
    }
    assert results.provider_model_errors["ppq"] == {
        "private/gpt-oss-120b": (
            "duplicate attestation result rows for provider ppq check "
            "ppq-private-model"
        ),
        "private/llama3-3-70b": (
            "duplicate attestation result rows for provider ppq check "
            "ppq-private-model"
        ),
    }


def test_live_check_maps_ppq_private_attestation_result_id_to_model(
    tmp_path: Path,
) -> None:
    targets_path = tmp_path / "attestation-targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "slug": "ppq",
                        "checks": [
                            {
                                "id": "ppq-gpt-oss-120b",
                                "type": "ppq",
                                "host": "api.ppq.ai",
                                "model": "private/gpt-oss-120b",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "ppq",
                        "id": "ppq-gpt-oss-120b",
                        "status": "verified",
                        **PPQ_PRIVATE_RESULT_PROVENANCE_DETAILS,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    targets_directory = live_check.load_attestation_targets_directory(targets_path)
    results_directory = live_check.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets_directory,
    )

    assert (
        results_directory.provider_model_statuses["ppq"]["private/gpt-oss-120b"]
        == "verified"
    )


def test_live_check_rejects_attestation_result_model_mismatch_with_target_id(
    tmp_path: Path,
) -> None:
    targets_path = tmp_path / "attestation-targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "slug": "ppq",
                        "checks": [
                            {
                                "id": "ppq-gpt-oss-120b",
                                "type": "ppq",
                                "host": "api.ppq.ai",
                                "model": "private/gpt-oss-120b",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "ppq",
                        "id": "ppq-gpt-oss-120b",
                        "host": "api.ppq.ai",
                        "model": "private/llama3-3-70b",
                        "status": "verified",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    targets_directory = live_check.load_attestation_targets_directory(targets_path)
    results_directory = live_check.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets_directory,
    )

    assert results_directory.provider_host_statuses["ppq"]["api.ppq.ai"] == "failed"
    assert (
        results_directory.provider_host_errors["ppq"]["api.ppq.ai"]
        == "model does not match confidential-inference attestation target"
    )
    assert (
        results_directory.provider_model_statuses["ppq"]["private/llama3-3-70b"]
        == "failed"
    )
    assert (
        results_directory.provider_model_errors["ppq"]["private/llama3-3-70b"]
        == "model does not match confidential-inference attestation target"
    )


def test_live_check_rejects_attestation_result_host_mismatch_with_target_id(
    tmp_path: Path,
) -> None:
    targets_path = tmp_path / "attestation-targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "slug": "ppq",
                        "checks": [
                            {
                                "id": "ppq-gpt-oss-120b",
                                "type": "ppq",
                                "host": "api.ppq.ai",
                                "model": "private/gpt-oss-120b",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "ppq",
                        "id": "ppq-gpt-oss-120b",
                        "host": "api2.ppq.ai",
                        "model": "private/gpt-oss-120b",
                        "status": "verified",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    targets_directory = live_check.load_attestation_targets_directory(targets_path)
    results_directory = live_check.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=targets_directory,
    )

    assert results_directory.provider_host_statuses["ppq"]["api2.ppq.ai"] == "failed"
    assert (
        results_directory.provider_host_errors["ppq"]["api2.ppq.ai"]
        == "host does not match confidential-inference attestation target"
    )
    assert (
        results_directory.provider_model_statuses["ppq"]["private/gpt-oss-120b"]
        == "failed"
    )
    assert (
        results_directory.provider_model_errors["ppq"]["private/gpt-oss-120b"]
        == "host does not match confidential-inference attestation target"
    )


def test_live_check_downgrades_privatemode_verified_result_without_app_e2ee_tier(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "privatemode",
                        "host": "cdn.confidential.cloud",
                        "status": "verified",
                        "trust_tier": "none",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    results = live_check.load_attestation_results_directory(results_path)

    assert (
        results.provider_host_statuses["privatemode"]["cdn.confidential.cloud"]
        == "failed"
    )
    assert (
        results.provider_host_errors["privatemode"]["cdn.confidential.cloud"]
        == "Privatemode verified attestation results must have trust_tier=app-e2ee"
    )


def test_live_check_downgrades_bare_privatemode_verified_model_result(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "privatemode",
                        "host": "cdn.confidential.cloud",
                        "model": "privatemode/kimi-latest",
                        "status": "verified",
                        "trust_tier": "app-e2ee",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    results = live_check.load_attestation_results_directory(results_path)

    assert (
        results.provider_model_statuses["privatemode"]["privatemode/kimi-latest"]
        == "failed"
    )
    assert (
        results.provider_model_errors["privatemode"]["privatemode/kimi-latest"]
        == "Privatemode selected model evidence is missing policy-bound "
        "proof details"
    )


def test_live_check_strict_external_guardrails_reject_wrong_ppq_model_result() -> None:
    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        expected=[("ppq-private", "private/gpt-oss-120b")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "verified"}},
            provider_host_errors={},
            provider_model_statuses={"ppq": {"private/llama3-3-70b": "verified"}},
            provider_model_errors={},
        ),
    )

    assert any(
        not result.ok
        and result.message
        == "confidential-inference PPQ private attestation results include verified model private/gpt-oss-120b"
        for result in results
    )


def test_live_check_strict_external_guardrails_reject_ppq_release_digest_mismatch() -> (
    None
):
    routing_policy = _ppq_private_routing_policy()
    provider = routing_policy["providers"][0]  # type: ignore[index]
    provider["confidentiality_policy"] = {  # type: ignore[index]
        **PPQ_PRIVATE_CONFIDENTIALITY_POLICY,
        "expected_release_digest": PROVIDER_RELEASE_DIGEST,
    }

    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(routing_policy),
        expected=[("ppq-private", "private/gpt-oss-120b")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "verified"}},
            provider_host_errors={},
            provider_model_statuses={"ppq": {"private/gpt-oss-120b": "verified"}},
            provider_model_errors={},
            provider_model_details={
                "ppq": {
                    "private/gpt-oss-120b": {
                        **PPQ_PRIVATE_BACKEND_VERIFICATION_DETAILS,
                        "release_digest": _digest("wrong-ppq-router-release"),
                        "backend_release_digest": _digest("wrong-ppq-backend-release"),
                    }
                }
            },
        ),
    )

    assert any(
        not result.ok
        and result.message
        == (
            "confidential-inference PPQ private attestation results bind model "
            "private/gpt-oss-120b router release digest to deployed policy"
        )
        for result in results
    )
    assert any(
        not result.ok
        and result.message
        == (
            "confidential-inference PPQ private attestation results bind model "
            "private/gpt-oss-120b backend release digest to deployed policy"
        )
        for result in results
    )


def test_live_check_strict_external_guardrails_require_ppq_model_result_when_targets_are_model_aware() -> (
    None
):
    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        expected=[("ppq-private", "private/gpt-oss-120b")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={
                "ppq": {
                    "ppq-gpt-oss-120b": "api.ppq.ai",
                    "ppq-llama3-70b": "api.ppq.ai",
                }
            },
            provider_check_models={
                "ppq": {
                    "ppq-gpt-oss-120b": "private/gpt-oss-120b",
                    "ppq-llama3-70b": "private/llama3-3-70b",
                }
            },
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "verified"}},
            provider_host_errors={},
            provider_model_statuses={},
            provider_model_errors={},
        ),
    )

    assert any(
        not result.ok
        and result.message
        == "confidential-inference PPQ private attestation results include verified model private/gpt-oss-120b"
        for result in results
    )


def test_live_check_strict_external_guardrails_require_ppq_results() -> None:
    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        expected=[("ppq-private", "private/gpt-oss-120b")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=None,
    )

    assert any(
        not result.ok
        and result.message
        == "confidential-inference PPQ private attestation results guardrail is supplied"
        for result in results
    )


def test_live_check_strict_external_guardrails_reject_ppq_router_only_results() -> None:
    routing_policy = _ppq_private_routing_policy()
    provider = routing_policy["providers"][0]  # type: ignore[index]
    confidentiality = provider["confidentiality"]  # type: ignore[index]
    confidentiality["model_ids"] = []  # type: ignore[index]

    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(routing_policy),
        expected=[],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "verified"}},
            provider_host_errors={},
        ),
    )

    assert any(
        not result.ok
        and result.message
        == (
            "confidential-inference PPQ private attestation results include "
            "selected model evidence"
        )
        for result in results
    )


def test_live_check_strict_external_guardrails_reject_unverified_ppq_results() -> None:
    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        expected=[("ppq-private", "private/gpt-oss-120b")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "unreachable"}},
            provider_host_errors={"ppq": {"api.ppq.ai": "timeout"}},
        ),
    )

    assert any(
        not result.ok
        and result.message
        == (
            "confidential-inference PPQ private attestation results include "
            "verified model private/gpt-oss-120b"
        )
        for result in results
    )


def test_live_check_strict_external_guardrails_require_verified_ppq_provider_host() -> (
    None
):
    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        expected=[("ppq-private", "private/gpt-oss-120b")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "unreachable"}},
            provider_host_errors={"ppq": {"api.ppq.ai": "timeout"}},
            provider_model_statuses={"ppq": {"private/gpt-oss-120b": "verified"}},
            provider_model_errors={},
            provider_model_details={
                "ppq": {
                    "private/gpt-oss-120b": {
                        **PPQ_PRIVATE_BACKEND_VERIFICATION_DETAILS,
                        "release_digest": PROVIDER_RELEASE_DIGEST,
                        "backend_release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                    }
                }
            },
        ),
    )

    assert any(
        not result.ok
        and result.message
        == (
            "confidential-inference PPQ private attestation results include "
            "verified provider host api.ppq.ai"
        )
        for result in results
    )


def test_live_check_strict_external_guardrails_reject_wrong_ppq_result_host() -> None:
    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        expected=[("ppq-private", "private/gpt-oss-120b")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"other.ppq.ai": "verified"}},
            provider_host_errors={},
        ),
    )

    assert any(
        not result.ok
        and result.message
        == (
            "confidential-inference PPQ private attestation results include "
            "verified model private/gpt-oss-120b"
        )
        for result in results
    )


def test_live_check_strict_external_guardrails_accept_ppq_selected_model_result() -> (
    None
):
    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        expected=[("ppq-private", "private/gpt-oss-120b")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"ppq": {"gpt-oss-120b"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={},
            provider_target_models={"ppq": {"private/gpt-oss-120b"}},
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "verified"}},
            provider_host_errors={},
            provider_model_statuses={"ppq": {"private/gpt-oss-120b": "verified"}},
            provider_model_errors={},
            provider_model_details={
                "ppq": {
                    "private/gpt-oss-120b": {
                        **PPQ_PRIVATE_BACKEND_VERIFICATION_DETAILS,
                        "release_digest": PROVIDER_RELEASE_DIGEST,
                        "backend_release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                    }
                }
            },
        ),
    )

    assert any(
        result.ok
        and result.message
        == (
            "confidential-inference PPQ private attestation results include "
            "verified model private/gpt-oss-120b"
        )
        for result in results
    )
    assert all(result.ok for result in results)


def test_live_check_rejects_ppq_result_backend_host_mismatch_with_targets(
    tmp_path: Path,
) -> None:
    targets_path = tmp_path / "attestation-targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "slug": "ppq",
                        "checks": [
                            {
                                "id": "ppq-gpt-oss-120b",
                                "type": "ppq",
                                "host": "api.ppq.ai",
                                "model": "private/gpt-oss-120b",
                                "backend_host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                                "backend_repo": "tinfoilsh/confidential-gpt-oss-120b",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "ppq",
                        "id": "ppq-gpt-oss-120b",
                        "status": "verified",
                        "release_digest": PROVIDER_RELEASE_DIGEST,
                        "backend_release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                        "backend_host": "wrong-backend.tinfoil.example",
                        **PPQ_PRIVATE_BACKEND_VERIFICATION_DETAILS,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    results_directory = live_check.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=live_check.load_attestation_targets_directory(
            targets_path
        ),
    )

    assert results_directory.provider_model_statuses["ppq"][
        "private/gpt-oss-120b"
    ] == "failed"
    assert (
        results_directory.provider_model_errors["ppq"]["private/gpt-oss-120b"]
        == "backend_host does not match confidential-inference attestation target"
    )


def test_live_check_rejects_ppq_result_backend_host_mismatch_without_check_id(
    tmp_path: Path,
) -> None:
    targets_path = tmp_path / "attestation-targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "slug": "ppq",
                        "checks": [
                            {
                                "id": "ppq-gpt-oss-120b",
                                "type": "ppq",
                                "host": "api.ppq.ai",
                                "model": "private/gpt-oss-120b",
                                "backend_host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                                "backend_repo": "tinfoilsh/confidential-gpt-oss-120b",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "ppq",
                        "host": "api.ppq.ai",
                        "model": "private/gpt-oss-120b",
                        "status": "verified",
                        "release_digest": PROVIDER_RELEASE_DIGEST,
                        "backend_release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                        "backend_host": "wrong-backend.tinfoil.example",
                        **PPQ_PRIVATE_BACKEND_VERIFICATION_DETAILS,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    results_directory = live_check.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=live_check.load_attestation_targets_directory(
            targets_path
        ),
    )

    assert results_directory.provider_model_statuses["ppq"][
        "private/gpt-oss-120b"
    ] == "failed"
    assert (
        results_directory.provider_model_errors["ppq"]["private/gpt-oss-120b"]
        == "backend_host does not match confidential-inference attestation target"
    )


def test_live_check_rejects_ppq_result_backend_repo_mismatch_without_check_id(
    tmp_path: Path,
) -> None:
    targets_path = tmp_path / "attestation-targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "slug": "ppq",
                        "checks": [
                            {
                                "id": "ppq-gpt-oss-120b",
                                "type": "ppq",
                                "host": "api.ppq.ai",
                                "model": "private/gpt-oss-120b",
                                "backend_host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                                "backend_repo": "tinfoilsh/confidential-gpt-oss-120b",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "ppq",
                        "host": "api.ppq.ai",
                        "model": "private/gpt-oss-120b",
                        "status": "verified",
                        "release_digest": PROVIDER_RELEASE_DIGEST,
                        "backend_release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                        "backend_host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                        "backend_repo": "tinfoilsh/confidential-other",
                        **PPQ_PRIVATE_BACKEND_VERIFICATION_DETAILS,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    results_directory = live_check.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=live_check.load_attestation_targets_directory(
            targets_path
        ),
    )

    assert results_directory.provider_model_statuses["ppq"][
        "private/gpt-oss-120b"
    ] == "failed"
    assert (
        results_directory.provider_model_errors["ppq"]["private/gpt-oss-120b"]
        == "backend_repo does not match confidential-inference attestation target"
    )


def test_live_check_rejects_ppq_result_host_mismatch_without_check_id(
    tmp_path: Path,
) -> None:
    targets_path = tmp_path / "attestation-targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "slug": "ppq",
                        "checks": [
                            {
                                "id": "ppq-gpt-oss-120b",
                                "type": "ppq",
                                "host": "api.ppq.ai",
                                "model": "private/gpt-oss-120b",
                                "backend_host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                                "backend_repo": "tinfoilsh/confidential-gpt-oss-120b",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "ppq",
                        "host": "api2.ppq.ai",
                        "model": "private/gpt-oss-120b",
                        "status": "verified",
                        "release_digest": PROVIDER_RELEASE_DIGEST,
                        "backend_release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                        "backend_host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                        "backend_repo": "tinfoilsh/confidential-gpt-oss-120b",
                        **PPQ_PRIVATE_BACKEND_VERIFICATION_DETAILS,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    results_directory = live_check.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=live_check.load_attestation_targets_directory(
            targets_path
        ),
    )

    assert results_directory.provider_host_statuses["ppq"]["api2.ppq.ai"] == "failed"
    assert (
        results_directory.provider_host_errors["ppq"]["api2.ppq.ai"]
        == "host does not match confidential-inference attestation target"
    )
    assert results_directory.provider_model_statuses["ppq"][
        "private/gpt-oss-120b"
    ] == "failed"
    assert (
        results_directory.provider_model_errors["ppq"]["private/gpt-oss-120b"]
        == "host does not match confidential-inference attestation target"
    )


def test_live_check_rejects_ppq_result_without_check_id_for_duplicate_model_targets(
    tmp_path: Path,
) -> None:
    targets_path = tmp_path / "attestation-targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "slug": "ppq",
                        "checks": [
                            {
                                "id": "ppq-gpt-oss-120b-a",
                                "type": "ppq",
                                "host": "api.ppq.ai",
                                "model": "private/gpt-oss-120b",
                                "backend_host": "gpt-oss-120b-a.inf10.tinfoil.sh",
                            },
                            {
                                "id": "ppq-gpt-oss-120b-b",
                                "type": "ppq",
                                "host": "api.ppq.ai",
                                "model": "private/gpt-oss-120b",
                                "backend_host": "gpt-oss-120b-b.inf10.tinfoil.sh",
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "ppq",
                        "host": "api.ppq.ai",
                        "model": "private/gpt-oss-120b",
                        "status": "verified",
                        "release_digest": PROVIDER_RELEASE_DIGEST,
                        "backend_release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                        "backend_host": "gpt-oss-120b-a.inf10.tinfoil.sh",
                        **PPQ_PRIVATE_BACKEND_VERIFICATION_DETAILS,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    results_directory = live_check.load_attestation_results_directory(
        results_path,
        attestation_targets_directory=live_check.load_attestation_targets_directory(
            targets_path
        ),
    )

    assert results_directory.provider_model_statuses["ppq"][
        "private/gpt-oss-120b"
    ] == "failed"
    assert (
        results_directory.provider_model_errors["ppq"]["private/gpt-oss-120b"]
        == "model has multiple confidential-inference attestation targets; "
        "result row must include a check id"
    )


def test_live_check_strict_external_guardrails_require_privatemode_results_when_policy_has_public_host() -> (
    None
):
    routing_policy = _privatemode_routing_policy()
    routing_policy["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "manifest_url"
    ] = "https://cdn.confidential.cloud/privatemode/v2/manifest.json"

    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(routing_policy),
        expected=[("privatemode", "privatemode/kimi-latest")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"privatemode": {"kimi-latest"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=None,
    )

    assert any(
        not result.ok
        and result.message
        == "confidential-inference Privatemode attestation results guardrail is supplied"
        for result in results
    )


def test_live_check_strict_external_guardrails_require_expected_privatemode_results_without_policy_host() -> (
    None
):
    routing_policy = _privatemode_routing_policy()
    routing_policy["providers"][0]["confidentiality_policy"] = {}  # type: ignore[index]

    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(routing_policy),
        expected=[("privatemode", "privatemode/kimi-latest")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"privatemode": {"kimi-latest"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=None,
    )

    assert any(
        not result.ok
        and result.message
        == "confidential-inference Privatemode attestation results guardrail is supplied"
        for result in results
    )


def test_live_check_strict_external_guardrails_reject_wrong_privatemode_result_host() -> (
    None
):
    routing_policy = _privatemode_routing_policy()
    routing_policy["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "manifest_url"
    ] = "https://cdn.confidential.cloud/privatemode/v2/manifest.json"

    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(routing_policy),
        expected=[("privatemode", "privatemode/kimi-latest")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"privatemode": {"kimi-latest"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"privatemode": {"docs.privatemode.ai": "verified"}},
            provider_host_errors={},
        ),
    )

    assert any(
        not result.ok
        and result.message
        == "confidential-inference Privatemode attestation results include verified host cdn.confidential.cloud"
        for result in results
    )


def test_live_check_strict_external_guardrails_require_privatemode_model_result() -> (
    None
):
    routing_policy = _privatemode_routing_policy()
    routing_policy["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "manifest_url"
    ] = "https://cdn.confidential.cloud/privatemode/v2/manifest.json"

    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(routing_policy),
        expected=[("privatemode", "privatemode/kimi-latest")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"privatemode": {"kimi-latest"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={
                "privatemode": {"cdn.confidential.cloud": "verified"}
            },
            provider_host_errors={},
            provider_model_statuses={},
            provider_model_errors={},
        ),
    )

    assert any(
        not result.ok
        and result.message
        == (
            "confidential-inference Privatemode attestation results include "
            "verified model privatemode/kimi-latest"
        )
        for result in results
    )


def test_live_check_strict_external_guardrails_require_privatemode_attestation_targets() -> (
    None
):
    routing_policy = _privatemode_routing_policy()
    routing_policy["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "manifest_url"
    ] = "https://cdn.confidential.cloud/privatemode/v2/manifest.json"

    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(routing_policy),
        expected=[("privatemode", "privatemode/kimi-latest")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"privatemode": {"kimi-latest"}}
        ),
        attestation_targets_directory=None,
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={
                "privatemode": {"cdn.confidential.cloud": "verified"}
            },
            provider_host_errors={},
            provider_model_statuses={
                "privatemode": {"privatemode/kimi-latest": "verified"}
            },
            provider_model_errors={},
            provider_model_details={
                "privatemode": {
                    "privatemode/kimi-latest": dict(PROVIDER_PRIVATEMODE_PROOF_CLAIMS)
                }
            },
        ),
    )

    assert any(
        not result.ok
        and result.message
        == "confidential-inference Privatemode attestation targets guardrail is supplied"
        for result in results
    )


def test_live_check_strict_external_guardrails_reject_privatemode_wrong_attestation_target_model() -> (
    None
):
    routing_policy = _privatemode_routing_policy()

    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(routing_policy),
        expected=[("privatemode", "privatemode/kimi-latest")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"privatemode": {"kimi-latest"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={},
            provider_target_models={"privatemode": {"privatemode/other-model"}},
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={},
            provider_host_errors={},
            provider_model_statuses={
                "privatemode": {"privatemode/kimi-latest": "verified"}
            },
            provider_model_errors={},
            provider_model_details={
                "privatemode": {
                    "privatemode/kimi-latest": dict(PROVIDER_PRIVATEMODE_PROOF_CLAIMS)
                }
            },
        ),
    )

    assert any(
        not result.ok
        and result.message
        == (
            "confidential-inference Privatemode attestation targets include "
            "model privatemode/kimi-latest"
        )
        for result in results
    )


def test_live_check_strict_external_guardrails_accept_privatemode_result_host() -> None:
    routing_policy = _privatemode_routing_policy()
    routing_policy["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "manifest_url"
    ] = "https://cdn.confidential.cloud/privatemode/v2/manifest.json"

    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(routing_policy),
        expected=[("privatemode", "privatemode/kimi-latest")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"privatemode": {"kimi-latest"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={},
            provider_target_models={"privatemode": {"privatemode/kimi-latest"}},
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={
                "privatemode": {"cdn.confidential.cloud": "verified"}
            },
            provider_host_errors={},
            provider_model_statuses={
                "privatemode": {"privatemode/kimi-latest": "verified"}
            },
            provider_model_errors={},
            provider_model_details={
                "privatemode": {
                    "privatemode/kimi-latest": dict(PROVIDER_PRIVATEMODE_PROOF_CLAIMS)
                }
            },
        ),
    )

    assert any(
        result.ok
        and result.message
        == "confidential-inference Privatemode attestation results include verified host cdn.confidential.cloud"
        for result in results
    )
    assert any(
        result.ok
        and result.message
        == (
            "confidential-inference Privatemode attestation results include "
            "verified model privatemode/kimi-latest"
        )
        for result in results
    )
    assert all(result.ok for result in results)


def test_live_check_strict_external_guardrails_pass_when_required_inputs_present() -> (
    None
):
    results = live_check.strict_external_guardrail_checks(
        attestation=_attestation_statement(),
        expected=[("tinfoil", "tinfoil/gpt-secure")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"tinfoil": {"gpt-secure"}}
        ),
        attestation_targets_directory=live_check.AttestationTargetsDirectory(
            provider_check_hosts={"tinfoil": {"tinfoil-router": "inference.tinfoil.sh"}},
            provider_target_models={"tinfoil": {"tinfoil/gpt-secure"}},
        ),
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"tinfoil": {"inference.tinfoil.sh": "verified"}},
            provider_host_errors={},
            provider_model_statuses={
                "tinfoil": {"tinfoil/gpt-secure": "verified"}
            },
            provider_model_details={
                "tinfoil": {
                    "tinfoil/gpt-secure": {
                        "release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                    }
                }
            },
        ),
    )

    assert all(result.ok for result in results)


def test_live_check_rejects_attested_tinfoil_router_without_tls_binding_in_results(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "tinfoil",
                        "id": "tinfoil-router",
                        "status": "verified",
                        "source_url": "https://inference.tinfoil.sh/.well-known/tinfoil-attestation",
                        "tls_matches": False,
                    },
                    {
                        "provider": "tinfoil",
                        "id": "tinfoil-gpt-secure",
                        "status": "verified",
                        "source_url": "https://gpt-secure.tinfoil.example/.well-known/tinfoil-attestation",
                        "tls_matches": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
        attestation_results_directory=live_check.load_attestation_results_directory(
            results_path
        ),
    )

    assert any(
        not result.ok
        and "attestation routing_policy.providers[0] Tinfoil host "
        "inference.tinfoil.sh is not verified in confidential-inference "
        "attestation results"
        in result.message
        and "TLS certificate binding was not verified" in result.message
        for result in results
    )


def test_live_check_allows_tinfoil_ehbp_policy_without_tls_binding(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "tinfoil",
                        "id": "tinfoil-router",
                        "status": "verified",
                        "source_url": "https://inference.tinfoil.sh/.well-known/tinfoil-attestation",
                        "tls_matches": False,
                    },
                    {
                        "provider": "tinfoil",
                        "id": "tinfoil-gpt-secure",
                        "status": "verified",
                        "source_url": "https://gpt-secure.tinfoil.example/.well-known/tinfoil-attestation",
                        "tls_matches": False,
                        "sigstore_match": True,
                        "sigstore_bundle_verified": True,
                        "images_verified": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    status = _verified_status()
    attestation = _attestation_statement()
    models = _verified_models()
    for surface in (
        status["providers"][0],
        attestation["routing_policy"]["providers"][0],
    ):
        surface["confidentiality_policy"] = {
            **surface["confidentiality_policy"],
            "transport_security": "ehbp",
            "model_attestation_targets": {
                "tinfoil/gpt-secure": {
                    **surface["confidentiality_policy"]["model_attestation_targets"][
                        "tinfoil/gpt-secure"
                    ],
                    "host": "gpt-secure.tinfoil.example",
                }
            },
        }
        proof_claims = surface["confidentiality"]["proof_claims"]
        proof_claims.pop("tls_public_key_fingerprint_sha256", None)
        proof_claims["model_attestations"]["tinfoil/gpt-secure"].pop(
            "tls_public_key_fingerprint_sha256",
            None,
        )
    model_confidentiality = models["data"][0]["confidentiality"]
    model_confidentiality["proof_claims"].pop("tls_public_key_fingerprint_sha256", None)
    model_confidentiality["proof_claims"]["model_attestations"][
        "tinfoil/gpt-secure"
    ].pop("tls_public_key_fingerprint_sha256", None)

    results = live_check.collect_results(
        status=status,
        attestation=attestation,
        models=models,
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
        provider_model_catalog=live_check.ProviderModelCatalog(
            provider_models={"tinfoil": {"gpt-secure"}}
        ),
        attestation_results_directory=live_check.load_attestation_results_directory(
            results_path
        ),
    )

    failures = [result.message for result in results if not result.ok]
    assert not [
        message
        for message in failures
        if "tls_public_key" in message
        or "TLS certificate binding was not verified" in message
    ]


def test_live_check_maps_attestation_result_ids_to_tinfoil_model_hosts(
    tmp_path: Path,
) -> None:
    targets_path = tmp_path / "attestation-targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "slug": "tinfoil",
                        "checks": [
                            {
                                "id": "tinfoil-router",
                                "type": "tinfoil",
                                "host": "inference.tinfoil.sh",
                            },
                            {
                                "id": "tinfoil-gpt-secure",
                                "type": "tinfoil",
                                "host": "gpt-secure.tinfoil.example",
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "tinfoil",
                        "id": "tinfoil-router",
                        "status": "verified",
                        "release_digest": PROVIDER_RELEASE_DIGEST,
                        "tls_matches": True,
                        "sigstore_match": True,
                        "sigstore_bundle_verified": True,
                        "images_verified": True,
                    },
                    {
                        "provider": "tinfoil",
                        "id": "tinfoil-gpt-secure",
                        "status": "verified",
                        "release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                        "tls_matches": True,
                        "sigstore_match": False,
                        "sigstore_bundle_verified": True,
                        "images_verified": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    routing_policy = json.loads(json.dumps(ROUTING_POLICY))
    routing_policy["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "model_attestation_targets"
    ]["tinfoil/gpt-secure"][  # type: ignore[index]
        "attestation_url"
    ] = "https://gpt-secure.tinfoil.example/.well-known/tinfoil-attestation"

    targets_directory = live_check.load_attestation_targets_directory(targets_path)
    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(routing_policy),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
        attestation_results_directory=live_check.load_attestation_results_directory(
            results_path,
            attestation_targets_directory=targets_directory,
        ),
    )

    assert any(
        not result.ok
        and "attestation routing_policy.providers[0] Tinfoil host "
        "gpt-secure.tinfoil.example is not verified in "
        "confidential-inference attestation results"
        in result.message
        and "sigstore_match was not verified" in result.message
        for result in results
    )


def test_live_check_maps_attestation_result_ids_to_tinfoil_router_host(
    tmp_path: Path,
) -> None:
    targets_path = tmp_path / "attestation-targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "slug": "tinfoil",
                        "checks": [
                            {
                                "id": "tinfoil-router",
                                "type": "tinfoil",
                                "host": "inference.tinfoil.sh",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "tinfoil",
                        "id": "tinfoil-router",
                        "status": "verified",
                        "tls_matches": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    targets_directory = live_check.load_attestation_targets_directory(targets_path)
    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
        attestation_results_directory=live_check.load_attestation_results_directory(
            results_path,
            attestation_targets_directory=targets_directory,
        ),
    )

    assert any(
        not result.ok
        and "attestation routing_policy.providers[0] Tinfoil host "
        "inference.tinfoil.sh is not verified in confidential-inference "
        "attestation results"
        in result.message
        and "TLS certificate binding was not verified" in result.message
        for result in results
    )


def test_live_check_rejects_attested_tinfoil_model_without_code_transparency_in_results(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "tinfoil",
                        "id": "tinfoil-router",
                        "status": "verified",
                        "source_url": "https://inference.tinfoil.sh/.well-known/tinfoil-attestation",
                        "release_digest": PROVIDER_RELEASE_DIGEST,
                        "tls_matches": True,
                    },
                    {
                        "provider": "tinfoil",
                        "id": "tinfoil-gpt-secure",
                        "status": "verified",
                        "source_url": "https://gpt-secure.tinfoil.example/.well-known/tinfoil-attestation",
                        "release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                        "tls_matches": True,
                        "sigstore_match": False,
                        "sigstore_bundle_verified": True,
                        "images_verified": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    routing_policy = json.loads(json.dumps(ROUTING_POLICY))
    routing_policy["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "model_attestation_targets"
    ]["tinfoil/gpt-secure"][  # type: ignore[index]
        "attestation_url"
    ] = "https://gpt-secure.tinfoil.example/.well-known/tinfoil-attestation"

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(routing_policy),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
        attestation_results_directory=live_check.load_attestation_results_directory(
            results_path
        ),
    )

    assert any(
        not result.ok
        and "attestation routing_policy.providers[0] Tinfoil host "
        "gpt-secure.tinfoil.example is not verified in "
        "confidential-inference attestation results"
        in result.message
        and "sigstore_match was not verified" in result.message
        for result in results
    )


def test_live_check_rejects_attested_tinfoil_model_missing_transparency_fields_in_results(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "attestation-results.json"
    results_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "provider": "tinfoil",
                        "id": "tinfoil-router",
                        "status": "verified",
                        "source_url": "https://inference.tinfoil.sh/.well-known/tinfoil-attestation",
                        "release_digest": PROVIDER_RELEASE_DIGEST,
                        "tls_matches": True,
                        "sigstore_match": True,
                        "sigstore_bundle_verified": True,
                        "images_verified": True,
                    },
                    {
                        "provider": "tinfoil",
                        "id": "tinfoil-gpt-secure",
                        "status": "verified",
                        "source_url": "https://gpt-secure.tinfoil.example/.well-known/tinfoil-attestation",
                        "release_digest": PROVIDER_MODEL_RELEASE_DIGEST,
                        "tls_matches": True,
                        "sigstore_bundle_verified": True,
                        "images_verified": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    routing_policy = json.loads(json.dumps(ROUTING_POLICY))
    routing_policy["providers"][0]["confidentiality_policy"][  # type: ignore[index]
        "model_attestation_targets"
    ]["tinfoil/gpt-secure"][  # type: ignore[index]
        "attestation_url"
    ] = "https://gpt-secure.tinfoil.example/.well-known/tinfoil-attestation"

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(routing_policy),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
        attestation_results_directory=live_check.load_attestation_results_directory(
            results_path
        ),
    )

    assert any(
        not result.ok
        and "attestation routing_policy.providers[0] Tinfoil host "
        "gpt-secure.tinfoil.example is not verified in "
        "confidential-inference attestation results"
        in result.message
        and "sigstore_match is missing" in result.message
        for result in results
    )


def test_live_check_rejects_attested_tinfoil_without_result_host_when_results_required() -> (
    None
):
    routing_policy = json.loads(json.dumps(ROUTING_POLICY))
    routing_policy["providers"][0].pop("base_url", None)  # type: ignore[index]

    results = live_check.collect_results(
        status=_verified_status(),
        attestation=_attestation_statement(routing_policy),
        models=_verified_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("tinfoil", "tinfoil/gpt-secure")],
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"tinfoil": {"inference.tinfoil.sh": "verified"}},
            provider_host_errors={},
        ),
    )

    assert any(
        not result.ok
        and "attestation routing_policy.providers[0] Tinfoil host is present for confidential-inference attestation results"
        == result.message
        for result in results
    )


def test_live_check_does_not_treat_ppq_attestation_results_as_provider_proof() -> (
    None
):
    status = _verified_ppq_private_status()
    status["providers"][0]["confidentiality_policy"] = dict(  # type: ignore[index]
        PPQ_PRIVATE_CONFIDENTIALITY_POLICY
    )

    results = live_check.collect_results(
        status=status,
        attestation=_attestation_statement(_ppq_private_routing_policy()),
        models=_verified_ppq_private_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("ppq-private", "private/gpt-oss-120b")],
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"ppq": {"api.ppq.ai": "unreachable"}},
            provider_host_errors={"ppq": {"api.ppq.ai": "health endpoint timeout"}},
        ),
    )

    assert not any(
        "confidential-inference attestation results" in result.message
        for result in results
    )
    assert all(result.ok for result in results)


def test_live_check_does_not_treat_privatemode_attestation_results_as_provider_proof() -> (
    None
):
    results = live_check.collect_results(
        status=_verified_privatemode_status(),
        attestation=_attestation_statement(_privatemode_routing_policy()),
        models=_verified_privatemode_models(),
        hpke_key_config=HPKE_KEY_CONFIG,
        hpke_content_type="application/ohttp-keys",
        expected=[("privatemode", "privatemode/kimi-latest")],
        attestation_results_directory=live_check.AttestationResultsDirectory(
            provider_host_statuses={"privatemode": {"api.privatemode.ai": "failed"}},
            provider_host_errors={
                "privatemode": {
                    "api.privatemode.ai": "public manifest reference only"
                }
            },
        ),
    )

    assert not any(
        "confidential-inference attestation results" in result.message
        for result in results
    )
    assert all(result.ok for result in results)


def test_live_check_main_validates_default_chat_support_before_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = _verified_models()
    models["data"][0]["confidentiality"]["supported_endpoints"] = []  # type: ignore[index]
    post_calls: list[object] = []

    def fake_fetch_json(base_url: str, path: str, *, timeout: float) -> object:
        if path == "/v1/confidentiality/status":
            status = _verified_status()
            status["providers"][0]["supported_endpoints"] = []  # type: ignore[index]
            return status
        if path == "/.well-known/routstr-attestation":
            return _attestation_statement()
        if path == "/v1/models":
            return models
        raise AssertionError(f"unexpected JSON path: {path}")

    def fake_fetch_bytes(
        base_url: str, path: str, *, timeout: float
    ) -> tuple[bytes, str]:
        assert path == "/.well-known/hpke-keys"
        return HPKE_KEY_CONFIG, "application/ohttp-keys"

    def fake_post_json(*args: object, **kwargs: object) -> object:
        post_calls.append((args, kwargs))
        return {
            "id": "chatcmpl-live-check",
            "object": "chat.completion",
            "model": "tinfoil/gpt-secure",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        }

    monkeypatch.setattr(live_check, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(live_check, "fetch_bytes", fake_fetch_bytes)
    monkeypatch.setattr(live_check, "post_json", fake_post_json)
    monkeypatch.setenv("ROUTSTR_API_KEY", "sk-live-secret")

    result = live_check.main(
        [
            "https://routstr.example",
            "--expect-provider",
            "tinfoil:tinfoil/gpt-secure",
            "--run-inference",
            "--bearer-token-env",
            "ROUTSTR_API_KEY",
        ]
    )

    assert result == 1
    assert post_calls == []


def test_live_check_main_defaults_inference_to_advertised_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_fetch_json(base_url: str, path: str, *, timeout: float) -> object:
        if path == "/v1/confidentiality/status":
            return _verified_privatemode_status()
        if path == "/.well-known/routstr-attestation":
            return _attestation_statement(_privatemode_routing_policy())
        if path == "/v1/models":
            return _verified_privatemode_models()
        raise AssertionError(f"unexpected JSON path: {path}")

    def fake_fetch_bytes(
        base_url: str, path: str, *, timeout: float
    ) -> tuple[bytes, str]:
        assert path == "/.well-known/hpke-keys"
        return HPKE_KEY_CONFIG, "application/ohttp-keys"

    def fake_post_json(
        base_url: str,
        path: str,
        *,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, object]:
        captured.update({"path": path, "payload": payload, "headers": headers})
        return {
            "id": "msg-live-check",
            "type": "message",
            "role": "assistant",
            "model": "privatemode/kimi-latest",
            "content": [{"type": "text", "text": "ok"}],
        }

    monkeypatch.setattr(live_check, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(live_check, "fetch_bytes", fake_fetch_bytes)
    monkeypatch.setattr(live_check, "post_json", fake_post_json)
    monkeypatch.setenv("ROUTSTR_API_KEY", "sk-live-secret")

    result = live_check.main(
        [
            "https://routstr.example",
            "--expect-provider",
            "privatemode:privatemode/kimi-latest",
            "--run-inference",
            "--bearer-token-env",
            "ROUTSTR_API_KEY",
        ]
    )

    assert result == 0
    assert captured["path"] == "/v1/messages"
    assert captured["payload"] == {
        "model": "privatemode/kimi-latest",
        "messages": [{"role": "user", "content": "Return the word ok."}],
        "max_tokens": 8,
        "stream": False,
    }


def test_live_check_runs_embedding_inference_with_endpoint_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_post_json(
        base_url: str,
        path: str,
        *,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, object]:
        captured.update(
            {
                "base_url": base_url,
                "path": path,
                "payload": payload,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return {
            "object": "list",
            "model": "privatemode/qwen3-embedding-4b",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.0, 1.0],
                }
            ],
        }

    monkeypatch.setattr(live_check, "post_json", fake_post_json)
    route = live_check.ExpectedInferenceRoute(
        provider="privatemode",
        model="privatemode/qwen3-embedding-4b",
        endpoint="/v1/embeddings",
    )

    results = live_check.run_inference_checks(
        base_url="https://routstr.example",
        routes=[route],
        auth_headers={"Authorization": "Bearer sk-live-secret"},
        prompt="Return the word ok.",
        timeout=3.0,
    )

    assert all(result.ok for result in results)
    assert captured["path"] == "/v1/embeddings"
    assert captured["payload"] == {
        "model": "privatemode/qwen3-embedding-4b",
        "input": "Return the word ok.",
    }


def test_live_check_runs_messages_inference_with_endpoint_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_post_json(
        base_url: str,
        path: str,
        *,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, object]:
        captured.update({"path": path, "payload": payload})
        return {
            "id": "msg-live-check",
            "type": "message",
            "role": "assistant",
            "model": "privatemode/kimi-latest",
            "content": [{"type": "text", "text": "ok"}],
        }

    monkeypatch.setattr(live_check, "post_json", fake_post_json)
    route = live_check.ExpectedInferenceRoute(
        provider="privatemode",
        model="privatemode/kimi-latest",
        endpoint="/v1/messages",
    )

    results = live_check.run_inference_checks(
        base_url="https://routstr.example",
        routes=[route],
        auth_headers={"Authorization": "Bearer sk-live-secret"},
        prompt="Return the word ok.",
        timeout=3.0,
    )

    assert all(result.ok for result in results)
    assert captured["path"] == "/v1/messages"
    assert captured["payload"] == {
        "model": "privatemode/kimi-latest",
        "messages": [{"role": "user", "content": "Return the word ok."}],
        "max_tokens": 8,
        "stream": False,
    }


def test_live_check_runs_legacy_completions_inference_with_endpoint_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_post_json(
        base_url: str,
        path: str,
        *,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, object]:
        captured.update({"path": path, "payload": payload})
        return {
            "id": "cmpl-live-check",
            "object": "text_completion",
            "model": "privatemode/kimi-latest",
            "choices": [{"text": "ok", "index": 0, "finish_reason": "stop"}],
        }

    monkeypatch.setattr(live_check, "post_json", fake_post_json)
    route = live_check.parse_expected_inference_route(
        "privatemode:privatemode/kimi-latest:completions"
    )

    results = live_check.run_inference_checks(
        base_url="https://routstr.example",
        routes=[route],
        auth_headers={"Authorization": "Bearer sk-live-secret"},
        prompt="Return the word ok.",
        timeout=3.0,
    )

    assert all(result.ok for result in results)
    assert captured["path"] == "/v1/completions"
    assert captured["payload"] == {
        "model": "privatemode/kimi-latest",
        "prompt": "Return the word ok.",
        "temperature": 0,
        "max_tokens": 8,
        "stream": False,
    }


def test_live_check_runs_responses_inference_with_endpoint_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_post_json(
        base_url: str,
        path: str,
        *,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, object]:
        captured.update({"path": path, "payload": payload})
        return {
            "id": "resp-live-check",
            "object": "response",
            "status": "completed",
            "model": "tinfoil/gpt-secure",
            "output_text": "ok",
            "output": [],
        }

    monkeypatch.setattr(live_check, "post_json", fake_post_json)
    route = live_check.ExpectedInferenceRoute(
        provider="tinfoil",
        model="tinfoil/gpt-secure",
        endpoint="/v1/responses",
    )

    results = live_check.run_inference_checks(
        base_url="https://routstr.example",
        routes=[route],
        auth_headers={"Authorization": "Bearer sk-live-secret"},
        prompt="Return the word ok.",
        timeout=3.0,
    )

    assert all(result.ok for result in results)
    assert captured["path"] == "/v1/responses"
    assert captured["payload"] == {
        "model": "tinfoil/gpt-secure",
        "input": "Return the word ok.",
        "stream": False,
        "store": False,
    }


def test_live_check_rejects_responses_wrong_model_echo() -> None:
    route = live_check.ExpectedInferenceRoute(
        provider="tinfoil",
        model="tinfoil/gpt-secure",
        endpoint="/v1/responses",
    )

    results = live_check.validate_inference_response(
        route,
        {
            "id": "resp-live-check",
            "object": "response",
            "status": "completed",
            "model": "plain/gpt",
            "output_text": "ok",
            "output": [],
        },
    )

    assert any(
        not result.ok and "response model matches requested model" in result.message
        for result in results
    )


def test_live_check_rejects_responses_without_completed_status() -> None:
    route = live_check.ExpectedInferenceRoute(
        provider="tinfoil",
        model="tinfoil/gpt-secure",
        endpoint="/v1/responses",
    )

    results = live_check.validate_inference_response(
        route,
        {
            "id": "resp-live-check",
            "object": "response",
            "model": "tinfoil/gpt-secure",
            "output_text": "ok",
            "output": [],
        },
    )

    assert any(
        not result.ok and "response status is completed" in result.message
        for result in results
    )


def test_live_check_rejects_responses_without_text_or_output_items() -> None:
    route = live_check.ExpectedInferenceRoute(
        provider="tinfoil",
        model="tinfoil/gpt-secure",
        endpoint="/v1/responses",
    )

    results = live_check.validate_inference_response(
        route,
        {
            "id": "resp-live-check",
            "object": "response",
            "status": "completed",
            "model": "tinfoil/gpt-secure",
            "output": [],
        },
    )

    assert any(
        not result.ok and "response is missing output" in result.message
        for result in results
    )


def test_live_check_accepts_audio_inference_when_audio_family_is_advertised() -> None:
    models = _verified_models()
    models["data"][0]["id"] = "tinfoil/voxtral-small-24b"  # type: ignore[index]
    models["data"][0]["confidentiality"]["supported_endpoints"] = ["/v1/audio"]  # type: ignore[index]
    route = live_check.parse_expected_inference_route(
        "tinfoil:tinfoil/voxtral-small-24b:audio"
    )

    results = live_check.validate_expected_inference_support(
        models=models,
        routes=[route],
    )

    assert all(result.ok for result in results)
    assert route.endpoint == "/v1/audio/transcriptions"


def test_live_check_accepts_audio_translations_inference_alias() -> None:
    models = _verified_models()
    models["data"][0]["id"] = "tinfoil/voxtral-small-24b"  # type: ignore[index]
    models["data"][0]["confidentiality"]["supported_endpoints"] = [  # type: ignore[index]
        "/v1/audio/translations"
    ]
    route = live_check.parse_expected_inference_route(
        "tinfoil:tinfoil/voxtral-small-24b:audio-translations"
    )

    results = live_check.validate_expected_inference_support(
        models=models,
        routes=[route],
    )

    assert all(result.ok for result in results)
    assert route.endpoint == "/v1/audio/translations"


def test_live_check_runs_audio_inference_with_multipart_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_post_multipart(
        base_url: str,
        path: str,
        *,
        fields: dict[str, str],
        files: list[tuple[str, str, bytes, str]],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, object]:
        captured.update(
            {
                "base_url": base_url,
                "path": path,
                "fields": fields,
                "files": files,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return {"text": "ok"}

    monkeypatch.setattr(live_check, "post_multipart", fake_post_multipart)
    route = live_check.ExpectedInferenceRoute(
        provider="tinfoil",
        model="tinfoil/voxtral-small-24b",
        endpoint="/v1/audio/transcriptions",
    )

    results = live_check.run_inference_checks(
        base_url="https://routstr.example",
        routes=[route],
        auth_headers={"Authorization": "Bearer sk-live-secret"},
        prompt="Return the word ok.",
        timeout=3.0,
    )

    assert all(result.ok for result in results)
    assert captured["path"] == "/v1/audio/transcriptions"
    assert captured["fields"] == {"model": "tinfoil/voxtral-small-24b"}
    files = captured["files"]
    assert isinstance(files, list)
    assert files[0][0] == "file"
    assert files[0][1] == "routstr-live-check.wav"
    assert files[0][2].startswith(b"RIFF")
    assert files[0][3] == "audio/wav"


def test_live_check_runs_audio_translation_inference_with_multipart_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_post_multipart(
        base_url: str,
        path: str,
        *,
        fields: dict[str, str],
        files: list[tuple[str, str, bytes, str]],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, object]:
        captured.update(
            {
                "base_url": base_url,
                "path": path,
                "fields": fields,
                "files": files,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return {"text": "ok"}

    monkeypatch.setattr(live_check, "post_multipart", fake_post_multipart)
    route = live_check.ExpectedInferenceRoute(
        provider="tinfoil",
        model="tinfoil/voxtral-small-24b",
        endpoint="/v1/audio/translations",
    )

    results = live_check.run_inference_checks(
        base_url="https://routstr.example",
        routes=[route],
        auth_headers={"Authorization": "Bearer sk-live-secret"},
        prompt="Return the word ok.",
        timeout=3.0,
    )

    assert all(result.ok for result in results)
    assert captured["path"] == "/v1/audio/translations"
    assert captured["fields"] == {"model": "tinfoil/voxtral-small-24b"}
    files = captured["files"]
    assert isinstance(files, list)
    assert files[0][0] == "file"
    assert files[0][1] == "routstr-live-check.wav"
    assert files[0][2].startswith(b"RIFF")
    assert files[0][3] == "audio/wav"
