import hashlib
import json
import time
from types import SimpleNamespace
from unittest.mock import patch

from routstr.core.confidentiality_public import (
    is_full_sha256_digest,
    public_confidentiality_policy_binds_provider_proof,
    public_provider_proof_claims,
    public_provider_proof_claims_cover_model_selectors,
    public_verified_model_selectors_satisfy_provider,
    verified_public_provider_proof_claims,
)
from routstr.core.settings import settings
from routstr.payment.models import (
    Architecture,
    Model,
    Pricing,
    _public_model_confidentiality,
)
from routstr.proxy import _public_confidentiality_status, get_confidentiality_status
from routstr.upstream.base import BaseUpstreamProvider, ConfidentialityStatus


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


VALID_POLICY_DIGEST = _digest("policy")
VALID_EVIDENCE_DIGEST = _digest("evidence")
VALID_PRIVATEMODE_GPU_POLICY = "nvidia-ocsp-good-only"
VALID_CLAIMS_DIGEST = _digest("claims")
VALID_RELEASE_DIGEST = _digest("release")


def test_is_full_sha256_digest_rejects_repeated_hex_template_placeholders() -> None:
    assert is_full_sha256_digest("sha256:" + ("a" * 64)) is False


def _sha256_json_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


class _NonSerializableClaim:
    pass


def _tinfoil_public_proof_claims() -> dict[str, object]:
    return {
        "transport": "ehbp",
        "repo": "tinfoilsh/confidential-model-router",
        "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
        "attestation_report_digest": _digest("provider-attestation-report"),
        "attested_hpke_public_key_hex": "b" * 64,
        "code_measurement_fingerprint": _digest("provider-code-measurement"),
        "enclave_measurement_fingerprint": _digest("provider-enclave-measurement"),
        "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
        "payload_policy_digest": VALID_POLICY_DIGEST,
        "release_digest": VALID_RELEASE_DIGEST,
        "tls_public_key_fingerprint_sha256": _digest("provider-tls-key"),
        "verification_steps": {
            "attested_transport_key_binding": True,
            "code_transparency": True,
            "freshness": True,
            "hardware_attestation_report": True,
            "hardware_certificate_chain": True,
            "measurement_match": True,
        },
        "model_attestations": {
            "tinfoil/gpt-secure": {
                "repo": "tinfoilsh/confidential-gpt-secure",
                "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
                "attestation_report_digest": _digest("model-attestation-report"),
                "attested_hpke_public_key_hex": "c" * 64,
                "code_measurement_fingerprint": _digest("model-code-measurement"),
                "enclave_measurement_fingerprint": _digest(
                    "model-enclave-measurement"
                ),
                "release_digest": VALID_RELEASE_DIGEST,
                "tls_public_key_fingerprint_sha256": _digest("model-tls-key"),
                "verification_steps": {
                    "attested_transport_key_binding": True,
                    "code_transparency": True,
                    "freshness": True,
                    "hardware_attestation_report": True,
                    "hardware_certificate_chain": True,
                    "measurement_match": True,
                },
            }
        },
    }


def _ppq_private_public_proof_claims() -> dict[str, object]:
    return {
        "transport": "ehbp",
        "client_encryption_boundary": "routstr-tee-ehbp-proxy",
        "repo": "ppq-ai/private-tee",
        "attestation_bundle_url_digest": _digest("ppq-attestation-bundle-url"),
        "release_digest": VALID_RELEASE_DIGEST,
        "attested_hpke_public_key_hex": "b" * 64,
        "code_measurement_fingerprint": _digest("provider-code-measurement"),
        "enclave_measurement_fingerprint": _digest("provider-enclave-measurement"),
        "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
        "payload_policy_digest": VALID_POLICY_DIGEST,
        "selected_model_ids": ["private/gpt-oss-120b"],
        "backend_model_attestations": {
            "private/gpt-oss-120b": {
                "repo": "tinfoilsh/confidential-gpt-oss-120b",
                "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
                "attestation_report_digest": _digest("attestation-report"),
                "attested_hpke_public_key_hex": "b" * 64,
                "code_measurement_fingerprint": _digest("model-code-measurement"),
                "enclave_measurement_fingerprint": _digest(
                    "model-enclave-measurement"
                ),
                "release_digest": VALID_RELEASE_DIGEST,
                "tls_public_key_fingerprint_sha256": _digest("model-tls-key"),
                "verification_steps": {
                    "attested_transport_key_binding": True,
                    "code_transparency": True,
                    "freshness": True,
                    "hardware_attestation_report": True,
                    "hardware_certificate_chain": True,
                    "measurement_match": True,
                },
            }
        },
        "tls_public_key_fingerprint_sha256": _digest("provider-tls-key"),
        "verification_steps": {
            "attested_transport_key_binding": True,
            "code_transparency": True,
            "freshness": True,
            "hardware_attestation_report": True,
            "hardware_certificate_chain": True,
            "measurement_match": True,
        },
    }


def _privatemode_public_proof_claims() -> dict[str, object]:
    claims = {
        "transport": "privatemode-proxy",
        "trust_tier": "app-e2ee",
        "manifest_digest": _digest("privatemode-manifest"),
        "proxy_image_digest": _digest("privatemode-proxy-image"),
        "proxy_binary_digest": _digest("privatemode-proxy-binary"),
        "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
        "payload_policy_digest": VALID_POLICY_DIGEST,
        "coordinator_measurement": _digest("privatemode-coordinator"),
        "secret_service_measurement": _digest("privatemode-secret-service"),
        "ai_worker_measurement": _digest("privatemode-ai-worker"),
        "attested_workload_identity_digest": _digest("attested-workload-identity"),
        "attested_workload_policy_digest": _digest("privatemode-ai-worker"),
        "expected_workload_identity_digest": _digest("expected-workload-identity"),
        "model_workload_binding_digest": _digest("model-workload-binding"),
        "selected_model_ids": ["privatemode/gpt-oss-120b"],
        "gpu_attestation_policy": VALID_PRIVATEMODE_GPU_POLICY,
        "coordinator_attestation_doc_digest": _digest("coordinator-attestation-doc"),
        "mesh_ca_digest": _digest("mesh-ca"),
        "secret_service_certificate_digest": _digest("secret-service-certificate"),
        "ai_worker_manifest_digest": _digest("ai-worker-manifest"),
        "nvidia_ocsp_policy_header_digest": _digest("nvidia-ocsp-policy-header"),
        "nvidia_ocsp_policy_mac_digest": _digest("nvidia-ocsp-policy-mac"),
        "prompt_encryption_ciphertext_digest": _digest("prompt-encryption"),
        "inference_secret_id_digest": _digest("inference-secret"),
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
    claims["key_release_binding"] = _sha256_json_digest(
        {
            "attested_workload_policy_digest": claims[
                "attested_workload_policy_digest"
            ],
            "expected_workload_identity_digest": claims[
                "expected_workload_identity_digest"
            ],
            "model_workload_binding_digest": claims["model_workload_binding_digest"],
            "manifest_digest": claims["manifest_digest"],
            "mesh_ca_digest": claims["mesh_ca_digest"],
            "secret_service_certificate_digest": claims[
                "secret_service_certificate_digest"
            ],
            "inference_secret_id_digest": claims["inference_secret_id_digest"],
            "nvidia_ocsp_policy_mac_digest": claims["nvidia_ocsp_policy_mac_digest"],
        }
    )
    return claims


def _recompute_privatemode_key_release_binding(
    claims: dict[str, object],
) -> dict[str, object]:
    claims["key_release_binding"] = _sha256_json_digest(
        {
            "attested_workload_policy_digest": claims[
                "attested_workload_policy_digest"
            ],
            "expected_workload_identity_digest": claims[
                "expected_workload_identity_digest"
            ],
            "model_workload_binding_digest": claims["model_workload_binding_digest"],
            "manifest_digest": claims["manifest_digest"],
            "mesh_ca_digest": claims["mesh_ca_digest"],
            "secret_service_certificate_digest": claims[
                "secret_service_certificate_digest"
            ],
            "inference_secret_id_digest": claims["inference_secret_id_digest"],
            "nvidia_ocsp_policy_mac_digest": claims["nvidia_ocsp_policy_mac_digest"],
        }
    )
    return claims


def test_public_provider_proof_claims_normalizes_and_filters_proof_shapes() -> None:
    code_measurement = _digest("normalized-code-measurement")
    release_digest = _digest("normalized-release")
    tls_digest = _digest("normalized-tls-key")
    proof_claims = public_provider_proof_claims(
        {
            "transport": "plaintext",
            "repo": "tinfoilsh/confidential-model-router",
            "attestation_report_digest": "sha256:attestation-report",
            "attested_hpke_public_key_hex": "not-hex",
            "code_measurement_fingerprint": code_measurement.removeprefix("sha256:"),
            "enclave_measurement_fingerprint": "enclave-placeholder",
            "release_digest": release_digest.upper(),
            "tls_public_key_fingerprint_sha256": tls_digest.removeprefix("sha256:"),
            "verification_steps": {
                "hardware_attestation_report": True,
                "measurement_match": True,
                "made_up_step": True,
            },
        }
    )

    assert proof_claims == {
        "code_measurement_fingerprint": code_measurement,
        "release_digest": release_digest,
        "repo": "tinfoilsh/confidential-model-router",
        "tls_public_key_fingerprint_sha256": tls_digest,
        "verification_steps": {
            "hardware_attestation_report": True,
            "measurement_match": True,
        },
    }


def test_public_provider_proof_claims_keeps_payload_digest_bindings_only() -> None:
    proof_claims = public_provider_proof_claims(
        {
            "transport": "ehbp",
            "payload_policy_digest": VALID_POLICY_DIGEST,
            "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
            "payload_verification_nonce": "nonce-kept-internal",
        }
    )

    assert proof_claims == {
        "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
        "payload_policy_digest": VALID_POLICY_DIGEST,
        "transport": "ehbp",
    }


def test_verified_public_provider_proof_claims_requires_payload_digest_bindings() -> (
    None
):
    for mode, claims in (
        ("tinfoil", _tinfoil_public_proof_claims()),
        ("ppq-private-tee", _ppq_private_public_proof_claims()),
        ("privatemode", _privatemode_public_proof_claims()),
    ):
        claims.pop("payload_policy_digest")
        assert verified_public_provider_proof_claims(mode, claims) is None


def test_verified_public_provider_proof_claims_rejects_duplicate_ppq_private_selected_models() -> (
    None
):
    claims = _ppq_private_public_proof_claims()
    claims["selected_model_ids"] = [
        "private/gpt-oss-120b",
        "private/gpt-oss-120b",
    ]

    assert verified_public_provider_proof_claims("ppq-private-tee", claims) is None


def test_verified_public_provider_proof_claims_rejects_case_variant_ppq_private_selected_models() -> (
    None
):
    claims = _ppq_private_public_proof_claims()
    claims["selected_model_ids"] = [
        "PRIVATE/gpt-oss-120b",
        "private/gpt-oss-120b",
    ]

    assert verified_public_provider_proof_claims("ppq-private-tee", claims) is None


def test_verified_public_provider_proof_claims_rejects_ppq_private_missing_client_encryption_boundary() -> (
    None
):
    claims = _ppq_private_public_proof_claims()
    claims.pop("client_encryption_boundary", None)

    assert verified_public_provider_proof_claims("ppq-private-tee", claims) is None


def test_verified_public_provider_proof_claims_allows_ppq_private_missing_router_release_digest() -> (
    None
):
    claims = _ppq_private_public_proof_claims()
    claims.pop("release_digest", None)

    proof_claims = verified_public_provider_proof_claims("ppq-private-tee", claims)

    assert proof_claims is not None
    assert "release_digest" not in proof_claims


def test_verified_public_provider_proof_claims_allows_ppq_private_without_proxy_binary_claim() -> (
    None
):
    claims = _ppq_private_public_proof_claims()
    claims.pop("proxy_binary_digest", None)

    proof_claims = verified_public_provider_proof_claims("ppq-private-tee", claims)

    assert proof_claims is not None
    assert "proxy_binary_digest" not in proof_claims


def test_verified_public_provider_proof_claims_rejects_incomplete_ppq_backend_attestation() -> (
    None
):
    claims = _ppq_private_public_proof_claims()
    backend_attestations = claims["backend_model_attestations"]
    assert isinstance(backend_attestations, dict)
    backend_claims = backend_attestations["private/gpt-oss-120b"]
    assert isinstance(backend_claims, dict)
    backend_claims.pop("release_digest")

    assert verified_public_provider_proof_claims("ppq-private-tee", claims) is None


def test_verified_public_provider_proof_claims_rejects_duplicate_privatemode_selected_models() -> (
    None
):
    claims = _privatemode_public_proof_claims()
    claims["selected_model_ids"] = [
        "privatemode/gpt-oss-120b",
        "privatemode/gpt-oss-120b",
    ]

    assert verified_public_provider_proof_claims("privatemode", claims) is None


def test_verified_public_provider_proof_claims_rejects_case_variant_privatemode_selected_models() -> (
    None
):
    claims = _privatemode_public_proof_claims()
    claims["selected_model_ids"] = [
        "PRIVATEMODE/gpt-oss-120b",
        "privatemode/gpt-oss-120b",
    ]

    assert verified_public_provider_proof_claims("privatemode", claims) is None


def test_verified_public_provider_proof_claims_rejects_unprefixed_privatemode_selected_model() -> (
    None
):
    claims = _privatemode_public_proof_claims()
    claims["selected_model_ids"] = ["gpt-oss-120b"]

    assert verified_public_provider_proof_claims("privatemode", claims) is None


def test_verified_public_provider_proof_claims_rejects_non_app_e2ee_privatemode_tier() -> (
    None
):
    claims = _privatemode_public_proof_claims()
    claims["trust_tier"] = "reference-only"

    assert verified_public_provider_proof_claims("privatemode", claims) is None


def test_verified_public_provider_proof_claims_rejects_missing_privatemode_tier() -> (
    None
):
    claims = _privatemode_public_proof_claims()
    claims.pop("trust_tier", None)

    assert verified_public_provider_proof_claims("privatemode", claims) is None


def test_verified_public_provider_proof_claims_rejects_unknown_privatemode_gpu_policy() -> (
    None
):
    claims = _privatemode_public_proof_claims()
    claims["gpu_attestation_policy"] = "strict-ocsp"

    assert verified_public_provider_proof_claims("privatemode", claims) is None


def test_verified_public_provider_proof_claims_rejects_non_boolean_verification_step() -> (
    None
):
    claims = _ppq_private_public_proof_claims()
    claims["verification_steps"]["diagnostic"] = "SECRET_PROMPT"  # type: ignore[index]

    assert verified_public_provider_proof_claims("ppq-private-tee", claims) is None


def test_verified_public_provider_proof_claims_rejects_non_string_verification_step_key() -> (
    None
):
    claims = _ppq_private_public_proof_claims()
    claims["verification_steps"][123] = True  # type: ignore[index]

    assert verified_public_provider_proof_claims("ppq-private-tee", claims) is None


def test_public_provider_proof_claims_exposes_model_attestation_proof() -> None:
    report_digest = _digest("normalized-model-report")
    enclave_measurement = _digest("normalized-model-enclave")
    code_measurement = _digest("normalized-model-code")
    release_digest = _digest("normalized-model-release")
    tls_digest = _digest("normalized-model-tls")
    proof_claims = public_provider_proof_claims(
        {
            "transport": "ehbp",
            "repo": "tinfoilsh/confidential-model-router",
            "model_attestations": {
                "tinfoil/gpt-secure": {
                    "repo": "tinfoilsh/confidential-gpt-secure",
                    "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
                    "attestation_report_digest": report_digest.removeprefix("sha256:"),
                    "attested_hpke_public_key_hex": "b" * 64,
                    "enclave_measurement_fingerprint": (
                        enclave_measurement.removeprefix("sha256:")
                    ),
                    "code_measurement_fingerprint": code_measurement.removeprefix(
                        "sha256:"
                    ),
                    "release_digest": release_digest.removeprefix("sha256:"),
                    "tls_public_key_fingerprint_sha256": tls_digest.removeprefix(
                        "sha256:"
                    ),
                    "api_key": "SECRET_MODEL_KEY",
                    "verification_steps": {
                        "hardware_attestation_report": True,
                        "hardware_certificate_chain": True,
                        "code_transparency": True,
                        "measurement_match": True,
                        "attested_transport_key_binding": True,
                        "freshness": True,
                        "Authorization": True,
                    },
                }
            },
        }
    )

    assert proof_claims is not None
    assert proof_claims["model_attestations"] == {
        "tinfoil/gpt-secure": {
            "repo": "tinfoilsh/confidential-gpt-secure",
            "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
            "attestation_report_digest": report_digest,
            "attested_hpke_public_key_hex": "b" * 64,
            "enclave_measurement_fingerprint": enclave_measurement,
            "code_measurement_fingerprint": code_measurement,
            "release_digest": release_digest,
            "tls_public_key_fingerprint_sha256": tls_digest,
            "verification_steps": {
                "attested_transport_key_binding": True,
                "code_transparency": True,
                "freshness": True,
                "hardware_attestation_report": True,
                "hardware_certificate_chain": True,
                "measurement_match": True,
            },
        }
    }
    assert "SECRET_MODEL_KEY" not in json.dumps(proof_claims, sort_keys=True)


def test_public_provider_proof_claims_rejects_tinfoil_model_attestation_without_steps() -> (
    None
):
    claims = _tinfoil_public_proof_claims()
    model_claims = claims["model_attestations"]["tinfoil/gpt-secure"]
    assert isinstance(model_claims, dict)
    model_claims.pop("verification_steps")

    assert public_provider_proof_claims(claims) is None


def test_public_model_confidentiality_requires_strict_boolean_status() -> None:
    confidentiality = _public_model_confidentiality(
        {
            "enabled": "false",
            "verified": "false",
            "mode": "tinfoil",
            "provider_type": "tinfoil",
        }
    )

    assert confidentiality is not None
    assert confidentiality["enabled"] is False
    assert confidentiality["verified"] is False


def test_public_model_confidentiality_rejects_non_string_identity_fields() -> None:
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": {"secret": "not-a-mode"},
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "expires_at": 4_102_444_800,
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_rejects_verified_placeholder_digests() -> None:
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": "sha256:policy",
            "evidence_digest": "sha256:evidence",
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "expires_at": 4_102_444_800,
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_requires_verified_at_for_verified_proof() -> None:
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "expires_at": 4_102_444_800,
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_rejects_expired_verified_proof() -> None:
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 1,
            "proof_claims": _tinfoil_public_proof_claims(),
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_rejects_future_verified_at() -> None:
    now = int(time.time())
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": now + 60,
            "expires_at": now + 300,
            "proof_claims": _tinfoil_public_proof_claims(),
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_requires_public_proof_claims() -> None:
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_rejects_claimed_digest_without_raw_claims() -> (
    None
):
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "model_ids": ["tinfoil/gpt-secure"],
            "model_id_prefixes": [],
            "proof_claims": _tinfoil_public_proof_claims(),
            "confidentiality_policy": {
                "repo": "tinfoilsh/confidential-model-router",
                "expected_release_digest": VALID_RELEASE_DIGEST,
                "require_model_attestations": True,
                "model_attestation_targets": {
                    "tinfoil/gpt-secure": {
                        "repo": "tinfoilsh/confidential-gpt-secure",
                        "expected_release_digest": VALID_RELEASE_DIGEST,
                    }
                },
            },
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_requires_tinfoil_model_attestation_coverage() -> (
    None
):
    proof_claims = _tinfoil_public_proof_claims()
    proof_claims.pop("model_attestations")

    confidentiality = _public_model_confidentiality(
        {
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
            "expires_at": 4_102_444_800,
            "model_ids": ["tinfoil/gpt-secure"],
            "model_id_prefixes": [],
            "proof_claims": proof_claims,
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_requires_public_policy_binding() -> None:
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": int(time.time()) + 300,
            "model_ids": ["tinfoil/gpt-secure"],
            "model_id_prefixes": [],
            "verified_claims": _tinfoil_public_proof_claims(),
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_rejects_policy_proof_mismatch() -> None:
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": int(time.time()) + 300,
            "model_ids": ["tinfoil/gpt-secure"],
            "model_id_prefixes": [],
            "verified_claims": _tinfoil_public_proof_claims(),
            "confidentiality_policy": {
                "repo": "tinfoilsh/confidential-model-router",
                "expected_release_digest": "sha256:" + ("9" * 64),
                "require_model_attestations": True,
                "model_attestation_targets": {
                    "tinfoil/gpt-secure": {
                        "host": "gpt-secure.tinfoil.example",
                        "repo": "tinfoilsh/confidential-gpt-secure",
                        "expected_release_digest": "sha256:" + ("1" * 64),
                    },
                },
            },
            "proof_claims": _tinfoil_public_proof_claims(),
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_rejects_incomplete_tinfoil_model_attestation_proof() -> (
    None
):
    proof_claims = _tinfoil_public_proof_claims()
    model_claims = proof_claims["model_attestations"]["tinfoil/gpt-secure"]
    assert isinstance(model_claims, dict)
    model_claims.pop("attestation_report_digest")

    confidentiality = _public_model_confidentiality(
        {
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
            "expires_at": 4_102_444_800,
            "model_ids": ["tinfoil/gpt-secure"],
            "model_id_prefixes": [],
            "proof_claims": proof_claims,
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_requires_complete_public_proof_claims() -> None:
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "proof_claims": {
                "transport": "ehbp",
                "repo": "tinfoilsh/confidential-model-router",
            },
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_allows_tinfoil_ehbp_policy_without_tls_binding() -> (
    None
):
    proof_claims = _tinfoil_public_proof_claims()
    proof_claims.pop("tls_public_key_fingerprint_sha256")
    model_claims = proof_claims["model_attestations"]["tinfoil/gpt-secure"]
    model_claims.pop("tls_public_key_fingerprint_sha256")

    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "model_ids": ["tinfoil/gpt-secure"],
            "model_id_prefixes": [],
            "verified_claims": proof_claims,
            "proof_claims": proof_claims,
            "confidentiality_policy": {
                "transport_security": "ehbp",
                "repo": "tinfoilsh/confidential-model-router",
                "expected_release_digest": VALID_RELEASE_DIGEST,
                "require_model_attestations": True,
                "model_attestation_targets": {
                    "tinfoil/gpt-secure": {
                        "repo": "tinfoilsh/confidential-gpt-secure",
                        "expected_release_digest": VALID_RELEASE_DIGEST,
                    }
                },
            },
        }
    )

    assert confidentiality is not None
    assert confidentiality["proof_claims"] == proof_claims


def test_public_model_confidentiality_requires_ppq_private_tls_binding() -> None:
    proof_claims = _ppq_private_public_proof_claims()
    proof_claims.pop("tls_public_key_fingerprint_sha256")
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "ppq-private-tee",
            "provider_type": "ppq-private",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "proof_claims": proof_claims,
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_requires_ppq_private_bundle_digest() -> None:
    proof_claims = _ppq_private_public_proof_claims()
    proof_claims.pop("attestation_bundle_url_digest", None)
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "ppq-private-tee",
            "provider_type": "ppq-private",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "proof_claims": proof_claims,
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_derives_ppq_private_bundle_digest() -> None:
    attestation_bundle_url = "https://api.ppq.ai/private"
    proof_claims = _ppq_private_public_proof_claims()
    proof_claims["attestation_bundle_url_digest"] = _sha256_json_digest(
        attestation_bundle_url
    )
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "ppq-private-tee",
            "provider_type": "ppq-private",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": int(time.time()) + 300,
            "model_ids": ["private/gpt-oss-120b"],
            "model_id_prefixes": [],
            "verified_claims": proof_claims,
            "proof_claims": proof_claims,
            "confidentiality_policy": {
                "repo": "ppq-ai/private-tee",
                "attestation_bundle_url": attestation_bundle_url,
                "proxy_binary_digest": _digest("ppq-proxy-binary"),
                "expected_release_digest": VALID_RELEASE_DIGEST,
                "expected_code_measurement_fingerprint": _digest(
                    "provider-code-measurement"
                ),
                "require_model_attestations": True,
                "model_attestation_targets": {
                    "private/gpt-oss-120b": {
                        "repo": "tinfoilsh/confidential-gpt-oss-120b",
                        "expected_release_digest": VALID_RELEASE_DIGEST,
                    }
                },
            },
        }
    )

    assert confidentiality is not None
    assert confidentiality["proof_claims"] == proof_claims


def test_public_model_confidentiality_rejects_payload_policy_digest_mismatch() -> (
    None
):
    proof_claims = _ppq_private_public_proof_claims()
    proof_claims["payload_policy_digest"] = _digest("other-policy")
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "ppq-private-tee",
            "provider_type": "ppq-private",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "model_ids": ["private/gpt-oss-120b"],
            "model_id_prefixes": [],
            "proof_claims": proof_claims,
            "confidentiality_policy": {
                "repo": "ppq-ai/private-tee",
                "attestation_bundle_url_digest": _digest("ppq-attestation-bundle-url"),
                "proxy_binary_digest": _digest("ppq-proxy-binary"),
                "expected_code_measurement_fingerprint": _digest(
                    "provider-code-measurement"
                ),
                "require_model_attestations": True,
                "model_attestation_targets": {
                    "private/gpt-oss-120b": {
                        "repo": "tinfoilsh/confidential-gpt-oss-120b",
                        "expected_release_digest": VALID_RELEASE_DIGEST,
                    }
                },
            },
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_rejects_payload_evidence_digest_mismatch() -> (
    None
):
    proof_claims = _ppq_private_public_proof_claims()
    proof_claims["payload_evidence_digest"] = _digest("other-evidence")
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "ppq-private-tee",
            "provider_type": "ppq-private",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "model_ids": ["private/gpt-oss-120b"],
            "model_id_prefixes": [],
            "proof_claims": proof_claims,
            "confidentiality_policy": {
                "repo": "ppq-ai/private-tee",
                "attestation_bundle_url_digest": _digest("ppq-attestation-bundle-url"),
                "proxy_binary_digest": _digest("ppq-proxy-binary"),
                "expected_code_measurement_fingerprint": _digest(
                    "provider-code-measurement"
                ),
                "require_model_attestations": True,
                "model_attestation_targets": {
                    "private/gpt-oss-120b": {
                        "repo": "tinfoilsh/confidential-gpt-oss-120b",
                        "expected_release_digest": VALID_RELEASE_DIGEST,
                    }
                },
            },
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_accepts_tinfoil_runtime_evidence_digest() -> None:
    verified_claims = _tinfoil_public_proof_claims()
    runtime_evidence_digest = _sha256_json_digest(
        {
            key: value
            for key, value in verified_claims.items()
            if key
            not in {
                "payload_policy_digest",
                "payload_evidence_digest",
                "payload_verification_nonce",
                "runtime_evidence_digest",
            }
        }
    )
    assert runtime_evidence_digest != VALID_EVIDENCE_DIGEST
    verified_claims["runtime_evidence_digest"] = runtime_evidence_digest
    proof_claims = {
        key: value
        for key, value in verified_claims.items()
        if key != "runtime_evidence_digest"
    }

    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "attestation_status": "verified",
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": runtime_evidence_digest,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "model_ids": ["tinfoil/gpt-secure"],
            "model_id_prefixes": [],
            "supported_endpoints": ["/v1/chat/completions"],
            "metadata_leakage": ["model"],
            "_verified_claims": verified_claims,
            "proof_claims": proof_claims,
            "confidentiality_policy": {
                "repo": "tinfoilsh/confidential-model-router",
                "expected_release_digest": VALID_RELEASE_DIGEST,
                "require_model_attestations": True,
                "model_attestation_targets": {
                    "tinfoil/gpt-secure": {
                        "repo": "tinfoilsh/confidential-gpt-secure",
                        "expected_release_digest": VALID_RELEASE_DIGEST,
                    }
                },
            },
        },
        model_id="gpt-secure",
    )

    assert confidentiality is not None
    assert confidentiality["evidence_digest"] == runtime_evidence_digest
    assert confidentiality["proof_claims"]["payload_evidence_digest"] == (
        VALID_EVIDENCE_DIGEST
    )


def test_public_model_confidentiality_rejects_conflicting_tls_aliases() -> None:
    proof_claims = _tinfoil_public_proof_claims()
    proof_claims["tls_public_key"] = "sha256:" + ("5" * 64)
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "proof_claims": proof_claims,
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_rejects_malformed_present_tls_alias() -> None:
    proof_claims = _tinfoil_public_proof_claims()
    proof_claims["tls_public_key_fingerprint_sha256"] = "not-a-digest"
    proof_claims["tls_public_key"] = "sha256:" + ("4" * 64)

    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "proof_claims": proof_claims,
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_rejects_conflicting_privatemode_manifest_aliases() -> (
    None
):
    proof_claims = _privatemode_public_proof_claims()
    proof_claims["manifest_log_manifest_digest"] = "sha256:" + ("f" * 64)

    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "privatemode",
            "provider_type": "privatemode",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "proof_claims": proof_claims,
        }
    )

    assert confidentiality is None


def test_public_privatemode_policy_binding_requires_manifest_identity() -> None:
    policy = {
        "proxy_binary_digest": "sha256:" + ("6" * 64),
        "expected_workload_sans": ["backend.privatemode.example"],
        "model_workload_bindings": {
            "privatemode/gpt-oss-120b": {
                "workload_sans": ["backend.privatemode.example"],
            }
        },
    }
    proof_claims = _privatemode_public_proof_claims()
    proof_claims["expected_workload_identity_digest"] = _sha256_json_digest(
        {
            "ids": [],
            "sans": ["backend.privatemode.example"],
        }
    )
    proof_claims["model_workload_binding_digest"] = _sha256_json_digest(
        {
            "privatemode/gpt-oss-120b": {
                "workload_ids": [],
                "workload_sans": ["backend.privatemode.example"],
            }
        }
    )

    assert (
        public_confidentiality_policy_binds_provider_proof(
            "privatemode",
            "privatemode",
            policy,
            proof_claims,
            ["privatemode/gpt-oss-120b"],
        )
        is False
    )


def test_public_privatemode_policy_binding_rejects_unbound_expected_workload() -> None:
    policy = {
        "manifest_digest": "sha256:" + ("1" * 64),
        "proxy_binary_digest": "sha256:" + ("6" * 64),
        "expected_workload_sans": ["backend.privatemode.example"],
        "model_workload_bindings": {
            "privatemode/gpt-oss-120b": {
                "workload_sans": [],
                "workload_ids": [],
            }
        },
    }
    proof_claims = _privatemode_public_proof_claims()
    proof_claims["expected_workload_identity_digest"] = _sha256_json_digest(
        {
            "ids": [],
            "sans": ["backend.privatemode.example"],
        }
    )
    proof_claims["model_workload_binding_digest"] = _sha256_json_digest(
        {
            "privatemode/gpt-oss-120b": {
                "workload_ids": [],
                "workload_sans": [],
            }
        }
    )
    _recompute_privatemode_key_release_binding(proof_claims)

    assert (
        public_confidentiality_policy_binds_provider_proof(
            "privatemode",
            "privatemode",
            policy,
            proof_claims,
            ["privatemode/gpt-oss-120b"],
        )
        is False
    )


def test_public_privatemode_policy_binding_rejects_conflicting_proxy_digest_aliases() -> (
    None
):
    policy = {
        "manifest_digest": "sha256:" + ("1" * 64),
        "proxy_binary_digest": "sha256:" + ("6" * 64),
        "proxyBinaryDigest": "sha256:" + ("7" * 64),
        "expected_workload_sans": ["backend.privatemode.example"],
        "model_workload_bindings": {
            "privatemode/gpt-oss-120b": {
                "workload_sans": ["backend.privatemode.example"],
            }
        },
    }
    proof_claims = _privatemode_public_proof_claims()
    proof_claims["expected_workload_identity_digest"] = _sha256_json_digest(
        {
            "ids": [],
            "sans": ["backend.privatemode.example"],
        }
    )
    proof_claims["model_workload_binding_digest"] = _sha256_json_digest(
        {
            "privatemode/gpt-oss-120b": {
                "workload_ids": [],
                "workload_sans": ["backend.privatemode.example"],
            }
        }
    )
    _recompute_privatemode_key_release_binding(proof_claims)

    assert (
        public_confidentiality_policy_binds_provider_proof(
            "privatemode",
            "privatemode",
            policy,
            proof_claims,
            ["privatemode/gpt-oss-120b"],
        )
        is False
    )


def test_public_ppq_policy_binding_ignores_legacy_proxy_digest_aliases() -> None:
    policy = {
        "repo": "ppq-ai/private-tee",
        "attestation_bundle_url_digest": _digest("ppq-attestation-bundle-url"),
        "proxy_binary_digest": "sha256:" + ("6" * 64),
        "proxyBinaryDigest": "sha256:" + ("7" * 64),
        "expected_code_measurement_fingerprint": _digest("provider-code-measurement"),
        "require_model_attestations": True,
        "model_attestation_targets": {
            "private/gpt-oss-120b": {
                "repo": "tinfoilsh/confidential-gpt-oss-120b",
                "expected_release_digest": VALID_RELEASE_DIGEST,
            }
        },
    }
    proof_claims = _ppq_private_public_proof_claims()
    proof_claims.pop("proxy_binary_digest", None)

    assert (
        public_confidentiality_policy_binds_provider_proof(
            "ppq-private",
            "ppq-private-tee",
            policy,
            proof_claims,
            ["private/gpt-oss-120b"],
        )
        is True
    )


def test_public_ppq_policy_binding_requires_backend_model_attestation_proof() -> None:
    proof_claims = _ppq_private_public_proof_claims()
    proof_claims.pop("backend_model_attestations", None)
    policy = {
        "repo": "ppq-ai/private-tee",
        "attestation_bundle_url_digest": _digest("ppq-attestation-bundle-url"),
        "proxy_binary_digest": _digest("ppq-proxy-binary"),
        "expected_release_digest": VALID_RELEASE_DIGEST,
        "expected_code_measurement_fingerprint": _digest("provider-code-measurement"),
        "require_model_attestations": True,
        "model_attestation_targets": {
            "private/gpt-oss-120b": {
                "repo": "tinfoilsh/confidential-gpt-oss-120b",
                "expected_release_digest": VALID_RELEASE_DIGEST,
            }
        },
    }

    assert (
        public_confidentiality_policy_binds_provider_proof(
            "ppq-private",
            "ppq-private-tee",
            policy,
            proof_claims,
            ["private/gpt-oss-120b"],
        )
        is False
    )


def test_public_ppq_policy_binding_accepts_router_code_measurement_pin_without_release_digest() -> None:
    policy = {
        "repo": "ppq-ai/private-tee",
        "attestation_bundle_url_digest": _digest("ppq-attestation-bundle-url"),
        "expected_code_measurement_fingerprint": _digest("provider-code-measurement"),
        "require_model_attestations": True,
        "model_attestation_targets": {
            "private/gpt-oss-120b": {
                "repo": "tinfoilsh/confidential-gpt-oss-120b",
                "expected_release_digest": VALID_RELEASE_DIGEST,
            }
        },
    }
    proof_claims = _ppq_private_public_proof_claims()
    proof_claims.pop("proxy_binary_digest", None)

    assert (
        public_confidentiality_policy_binds_provider_proof(
            "ppq-private",
            "ppq-private-tee",
            policy,
            proof_claims,
            ["private/gpt-oss-120b"],
        )
        is True
    )


def test_public_ppq_policy_binding_requires_router_release_or_code_pin() -> None:
    policy = {
        "repo": "ppq-ai/private-tee",
        "attestation_bundle_url_digest": _digest("ppq-attestation-bundle-url"),
        "proxy_binary_digest": _digest("ppq-proxy-binary"),
        "require_model_attestations": True,
        "model_attestation_targets": {
            "private/gpt-oss-120b": {
                "repo": "tinfoilsh/confidential-gpt-oss-120b",
                "expected_release_digest": VALID_RELEASE_DIGEST,
            }
        },
    }

    assert (
        public_confidentiality_policy_binds_provider_proof(
            "ppq-private",
            "ppq-private-tee",
            policy,
            _ppq_private_public_proof_claims(),
            ["private/gpt-oss-120b"],
        )
        is False
    )


def test_public_model_confidentiality_rejects_privatemode_image_only_proxy_proof() -> (
    None
):
    proof_claims = _privatemode_public_proof_claims()
    proof_claims.pop("proxy_binary_digest", None)

    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "privatemode",
            "provider_type": "privatemode",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "proof_claims": proof_claims,
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_rejects_unbound_privatemode_key_release() -> None:
    proof_claims = _privatemode_public_proof_claims()
    proof_claims["key_release_binding"] = "sha256:" + ("f" * 64)

    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "privatemode",
            "provider_type": "privatemode",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "proof_claims": proof_claims,
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_allows_privatemode_manifest_log_policy() -> None:
    proof_claims = _privatemode_public_proof_claims()
    proof_claims["expected_workload_identity_digest"] = _sha256_json_digest(
        {
            "ids": [],
            "sans": ["backend.privatemode.example"],
        }
    )
    proof_claims["model_workload_binding_digest"] = _sha256_json_digest(
        {
            "privatemode/gpt-oss-120b": {
                "workload_ids": [],
                "workload_sans": ["backend.privatemode.example"],
            }
        }
    )
    _recompute_privatemode_key_release_binding(proof_claims)
    proof_claims["manifest_log_digest"] = _digest("privatemode-manifest-log")
    proof_claims["manifest_log_manifest_digest"] = proof_claims["manifest_digest"]

    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "privatemode",
            "provider_type": "privatemode",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": int(time.time()) + 300,
            "model_ids": ["privatemode/gpt-oss-120b"],
            "model_id_prefixes": [],
            "verified_claims": proof_claims,
            "proof_claims": proof_claims,
            "confidentiality_policy": {
                "manifest_log_dir": "/state/privatemode-manifest-log",
                "proxy_binary_digest": _digest("privatemode-proxy-binary"),
                "expected_trust_tier": "app-e2ee",
                "expected_gpu_attestation_policy": "nvidia-ocsp-good-only",
                "expected_workload_sans": ["backend.privatemode.example"],
                "model_workload_bindings": {
                    "privatemode/gpt-oss-120b": {
                        "workload_sans": ["backend.privatemode.example"],
                    }
                },
            },
        }
    )

    assert confidentiality is not None
    assert confidentiality["proof_claims"]["manifest_log_digest"] == proof_claims[
        "manifest_log_digest"
    ]


def test_public_model_confidentiality_rejects_privatemode_selected_model_mismatch() -> (
    None
):
    proof_claims = _privatemode_public_proof_claims()
    proof_claims["selected_model_ids"] = ["privatemode/kimi-k2-6"]
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "privatemode",
            "provider_type": "privatemode",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "model_ids": ["privatemode/gpt-oss-120b"],
            "model_id_prefixes": [],
            "proof_claims": proof_claims,
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_rejects_unknown_verified_modes() -> None:
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "custom-confidential",
            "provider_type": "custom",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "proof_claims": {
                "repo": "custom/confidential-model-router",
            },
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_rejects_provider_mode_mismatch() -> None:
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "provider_type": "custom",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "proof_claims": _tinfoil_public_proof_claims(),
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_rejects_malformed_verified_public_lists() -> None:
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "model_ids": ["tinfoil/gpt-secure", {"api_key": "SECRET_SELECTOR"}],
            "model_id_prefixes": [],
            "supported_endpoints": ["/v1/chat/completions"],
            "proof_claims": _tinfoil_public_proof_claims(),
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_rejects_known_provider_prefix_selectors() -> None:
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "model_ids": ["tinfoil/gpt-secure"],
            "model_id_prefixes": ["tinfoil/"],
            "supported_endpoints": ["/v1/chat/completions"],
            "proof_claims": _tinfoil_public_proof_claims(),
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_requires_ppq_private_exact_model_ids() -> None:
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "ppq-private-tee",
            "provider_type": "ppq-private",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "model_ids": ["gpt-oss-120b"],
            "model_id_prefixes": [],
            "proof_claims": _ppq_private_public_proof_claims(),
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_requires_privatemode_exact_model_ids() -> None:
    proof_claims = _privatemode_public_proof_claims()
    proof_claims["selected_model_ids"] = ["gpt-oss-120b"]
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "privatemode",
            "provider_type": "privatemode",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "model_ids": ["gpt-oss-120b"],
            "model_id_prefixes": [],
            "proof_claims": proof_claims,
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_rejects_ppq_private_selected_model_mismatch() -> (
    None
):
    proof_claims = _ppq_private_public_proof_claims()
    proof_claims["selected_model_ids"] = ["private/kimi-k2-6"]
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "ppq-private-tee",
            "provider_type": "ppq-private",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_CLAIMS_DIGEST,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "model_ids": ["private/gpt-oss-120b"],
            "model_id_prefixes": [],
            "proof_claims": proof_claims,
        }
    )

    assert confidentiality is None


def test_public_provider_proof_claims_cover_model_selectors_rejects_malformed_selected_model_ids() -> (
    None
):
    proof_claims = _ppq_private_public_proof_claims()
    proof_claims["selected_model_ids"] = ["private/gpt-oss-120b", 123]

    assert (
        public_provider_proof_claims_cover_model_selectors(
            "ppq-private",
            "ppq-private-tee",
            proof_claims,
            ["private/gpt-oss-120b"],
        )
        is False
    )


def test_public_provider_proof_claims_cover_model_selectors_rejects_duplicate_selected_model_ids() -> (
    None
):
    proof_claims = _ppq_private_public_proof_claims()
    proof_claims["selected_model_ids"] = [
        "private/gpt-oss-120b",
        "private/gpt-oss-120b",
    ]

    assert (
        public_provider_proof_claims_cover_model_selectors(
            "ppq-private",
            "ppq-private-tee",
            proof_claims,
            ["private/gpt-oss-120b"],
        )
        is False
    )


def test_public_provider_proof_claims_cover_model_selectors_rejects_duplicate_provider_model_ids() -> (
    None
):
    proof_claims = _ppq_private_public_proof_claims()

    assert (
        public_provider_proof_claims_cover_model_selectors(
            "ppq-private",
            "ppq-private-tee",
            proof_claims,
            ["private/gpt-oss-120b", "private/gpt-oss-120b"],
        )
        is False
    )


def test_public_verified_model_selectors_reject_duplicate_known_provider_model_ids() -> (
    None
):
    assert (
        public_verified_model_selectors_satisfy_provider(
            "tinfoil",
            ["tinfoil/gpt-secure", "tinfoil/gpt-secure"],
            [],
        )
        is False
    )


def test_public_verified_model_selectors_reject_case_variant_known_provider_model_ids() -> (
    None
):
    assert (
        public_verified_model_selectors_satisfy_provider(
            "tinfoil",
            ["TINFOIL/gpt-secure", "tinfoil/gpt-secure"],
            [],
        )
        is False
    )


def test_public_provider_proof_claims_cover_model_selectors_rejects_malformed_tinfoil_attestation_keys() -> (
    None
):
    proof_claims = _tinfoil_public_proof_claims()
    proof_claims["model_attestations"][123] = proof_claims["model_attestations"][  # type: ignore[index]
        "tinfoil/gpt-secure"
    ]

    assert (
        public_provider_proof_claims_cover_model_selectors(
            "tinfoil",
            "tinfoil",
            proof_claims,
            ["tinfoil/gpt-secure"],
        )
        is False
    )


def test_public_verified_provider_proof_claims_rejects_duplicate_tinfoil_model_attestation_keys() -> (
    None
):
    proof_claims = _tinfoil_public_proof_claims()
    proof_claims["model_attestations"][" tinfoil/gpt-secure "] = proof_claims[  # type: ignore[index]
        "model_attestations"
    ]["tinfoil/gpt-secure"]

    assert verified_public_provider_proof_claims("tinfoil", proof_claims) is None


def test_public_verified_provider_proof_claims_rejects_case_variant_tinfoil_model_attestation_keys() -> (
    None
):
    proof_claims = _tinfoil_public_proof_claims()
    proof_claims["model_attestations"]["TINFOIL/gpt-secure"] = proof_claims[  # type: ignore[index]
        "model_attestations"
    ]["tinfoil/gpt-secure"]

    assert verified_public_provider_proof_claims("tinfoil", proof_claims) is None


def test_public_model_confidentiality_drops_unverified_claims() -> None:
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": False,
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "failure_reason": "verification failed",
            "verified_claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "debug": _NonSerializableClaim(),
            },
        }
    )

    assert confidentiality is not None
    assert confidentiality["verified"] is False
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_public_model_confidentiality_rejects_verified_non_json_claims() -> None:
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "expires_at": 4_102_444_800,
            "verified_at": 1_700_000_000,
            "verified_claims": {
                "repo": "tinfoilsh/confidential-model-router",
                "debug": _NonSerializableClaim(),
            },
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_rejects_non_canonical_verified_claims() -> None:
    verified_claims = _tinfoil_public_proof_claims()
    verified_claims["non_canonical"] = float("nan")

    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_EVIDENCE_DIGEST,
            "expires_at": 4_102_444_800,
            "verified_at": 1_700_000_000,
            "verified_claims": verified_claims,
        }
    )

    assert confidentiality is None


def test_public_confidentiality_status_rejects_non_string_provider_identity() -> None:
    """Public provider identity fields are proof selectors, not display text."""

    class StringyProviderType:
        def __str__(self) -> str:
            return "tinfoil"

    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = StringyProviderType()  # type: ignore[assignment]
    provider.upstream_name = {"api_key": "SECRET_UPSTREAM"}  # type: ignore[assignment]
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=False,
            mode="none",
            model_ids=["gpt-secure"],
            policy_digest=VALID_POLICY_DIGEST,
        )
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(include_routstr_tee=False)

    assert status["providers"] == []
    assert "SECRET_UPSTREAM" not in json.dumps(status)


def test_public_confidentiality_status_does_not_stringify_malformed_mode(
    monkeypatch,
) -> None:
    """Top-level mode is public proof metadata, not arbitrary display text."""

    class SecretMode:
        def __str__(self) -> str:
            return "SECRET_MODE"

    monkeypatch.setattr(settings, "confidential_routing_mode", SecretMode())

    with patch("routstr.proxy._upstreams", []):
        status = get_confidentiality_status(include_routstr_tee=False)

    assert status["mode"] == "disabled"
    assert status["required"] is False
    assert "SECRET_MODE" not in json.dumps(status)


def test_public_confidentiality_status_exposes_exact_routable_full_attestation_models(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    public_key_digest = _digest("routstr-public-key")
    hpke_key_config_digest = _digest("hpke-key-config")
    hpke_public_key_digest = _digest("hpke-public-key")
    evidence_digest = _digest("routstr-tee-evidence")
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil"
    provider.confidentiality_policy = lambda: SimpleNamespace(
        provider_type="tinfoil",
        mode="tinfoil",
        digest=VALID_POLICY_DIGEST,
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": VALID_RELEASE_DIGEST,
            "require_model_attestations": True,
            "model_attestation_targets": {
                "tinfoil/gpt-secure": {
                    "repo": "tinfoilsh/confidential-gpt-secure",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                }
            },
        },
    )
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="tinfoil",
            verifier="unit-test-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_at=1_700_000_000,
            expires_at=int(time.time()) + 300,
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=_tinfoil_public_proof_claims(),
        )
    )
    model = Model(
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
        patch("routstr.proxy._model_instances", {"tinfoil/gpt-secure": model}),
        patch("routstr.proxy._provider_map", {"tinfoil/gpt-secure": [provider]}),
        patch("routstr.proxy.is_routable_confidential_model", return_value=True),
        patch("routstr.proxy.is_confidential_provider_for_model", return_value=True),
        patch(
            "routstr.core.attestation.get_public_routstr_tee_status",
            lambda: {
                "required": True,
                "ready": True,
                "client_confidentiality": {
                    "mode": "attested-tls-termination",
                    "tls_terminates_in_attested_tee": True,
                    "inbound_ehbp_ohttp_request_decryption": False,
                    "attested_tls_public_key_digest": public_key_digest,
                },
                "attestation_evidence_digest": evidence_digest,
                "hpke_key_config_digest": hpke_key_config_digest,
                "hpke_public_key_digest": hpke_public_key_digest,
                "local_verification": {
                    "verified": True,
                    "verified_at": 1_700_000_000,
                    "expires_at": int(time.time()) + 300,
                    "evidence_digest": evidence_digest,
                    "verified_claims_digest": _digest("routstr-tee-claims"),
                    "proof_claims": {
                        "hpke_key_config_digest": hpke_key_config_digest,
                        "hpke_public_key_digest": hpke_public_key_digest,
                        "public_key_digest": public_key_digest,
                    },
                },
            },
        ),
    ):
        status = get_confidentiality_status()

    assert status["end_to_end_ready"] is True
    assert status["routable_with_full_attestation"] == {
        "ppq-private": [],
        "privatemode": [],
        "tinfoil": ["tinfoil/gpt-secure"],
    }


def test_public_confidentiality_status_requires_public_model_metadata_for_exact_map(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil"
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="tinfoil",
            verifier="unit-test-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_at=1_700_000_000,
            expires_at=int(time.time()) + 300,
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=_tinfoil_public_proof_claims(),
        )
    )
    model = Model(
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
        patch("routstr.proxy._model_instances", {"tinfoil/gpt-secure": model}),
        patch("routstr.proxy._provider_map", {"tinfoil/gpt-secure": [provider]}),
        patch("routstr.proxy.is_confidential_provider_for_model", return_value=True),
        patch(
            "routstr.proxy.public_confidentiality_metadata",
            return_value=None,
        ),
        patch(
            "routstr.core.attestation.get_public_routstr_tee_status",
            lambda: {
                "required": True,
                "ready": True,
                "client_confidentiality": {
                    "mode": "attested-tls-termination",
                    "tls_terminates_in_attested_tee": True,
                    "inbound_ehbp_ohttp_request_decryption": False,
                    "attested_tls_public_key_digest": _digest("public-key"),
                },
                "attestation_evidence_digest": _digest("tee-evidence"),
                "hpke_key_config_digest": _digest("hpke-key-config"),
                "hpke_public_key_digest": _digest("hpke-public-key"),
                "local_verification": {
                    "verified": True,
                    "verified_at": 1_700_000_000,
                    "expires_at": int(time.time()) + 300,
                    "evidence_digest": _digest("tee-evidence"),
                    "verified_claims_digest": _digest("routstr-tee-claims"),
                    "proof_claims": {
                        "hpke_key_config_digest": _digest("hpke-key-config"),
                        "hpke_public_key_digest": _digest("hpke-public-key"),
                        "public_key_digest": _digest("public-key"),
                    },
                },
            },
        ),
    ):
        status = get_confidentiality_status()

    assert status["end_to_end_ready"] is False
    assert status["routable_with_full_attestation"] == {
        "ppq-private": [],
        "privatemode": [],
        "tinfoil": [],
    }


def test_public_confidentiality_status_keeps_routable_full_attestation_empty_without_tee(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil"
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="tinfoil",
            verifier="unit-test-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_at=1_700_000_000,
            expires_at=int(time.time()) + 300,
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=_tinfoil_public_proof_claims(),
        )
    )
    model = Model(
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
        patch("routstr.proxy._model_instances", {"tinfoil/gpt-secure": model}),
        patch("routstr.proxy._provider_map", {"tinfoil/gpt-secure": [provider]}),
        patch("routstr.proxy.is_confidential_provider_for_model", return_value=True),
        patch(
            "routstr.core.attestation.get_public_routstr_tee_status",
            lambda: {"required": True, "ready": False},
        ),
    ):
        status = get_confidentiality_status()

    assert status["end_to_end_ready"] is False
    assert status["routable_with_full_attestation"] == {
        "ppq-private": [],
        "privatemode": [],
        "tinfoil": [],
    }


def test_public_confidentiality_status_redacts_provider_url_userinfo() -> None:
    """Optional provider URLs must still be safe to publish."""
    provider = BaseUpstreamProvider(
        base_url="https://operator:secret-token@verified.example/v1?tenant=one",
        api_key="test",
    )
    provider.provider_type = "openai-compatible"
    provider.upstream_name = "plain-provider"
    provider.confidentiality_policy = lambda: SimpleNamespace(digest=VALID_POLICY_DIGEST)
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=False,
            mode="none",
            model_ids=["gpt-secure"],
            policy_digest=VALID_POLICY_DIGEST,
        )
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(
            include_provider_urls=True,
            include_routstr_tee=False,
        )

    provider_status = status["providers"][0]
    assert provider_status["base_url"] == "https://verified.example/v1?tenant=one"
    serialized = json.dumps(status, sort_keys=True)
    assert "secret-token" not in serialized
    assert "operator:" not in serialized


def test_public_confidentiality_status_omits_tinfoil_non_default_provider_url() -> None:
    """Optional public URLs must not advertise invalid fixed-provider identity."""
    provider = BaseUpstreamProvider(
        base_url="https://attacker.example/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil"
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=False,
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
            policy_digest=VALID_POLICY_DIGEST,
        )
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(
            include_provider_urls=True,
            include_routstr_tee=False,
        )

    provider_status = status["providers"][0]
    assert "base_url" not in provider_status
    assert "attacker.example" not in json.dumps(status)


def test_public_confidentiality_status_includes_sanitized_policy_for_attestation() -> None:
    model_release_digest = _digest("tinfoil-gpt-secure-release")
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
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/gpt-secure": {
                            "host": "gpt-secure.tinfoil.example",
                            "repo": "tinfoilsh/confidential-gpt-secure",
                            "expected_release_digest": model_release_digest,
                        }
                    },
                    "verifier_command": "/opt/routstr/bin/tinfoil-verifier",
                    "api_key_env": "SECRET_ENV_NAME",
                },
            }
        }
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(
            include_provider_urls=True,
            include_routstr_tee=False,
        )

    provider_status = status["providers"][0]
    assert provider_status["confidentiality_policy"] == {
        "expected_release_digest": VALID_RELEASE_DIGEST,
        "model_attestation_targets": {
            "tinfoil/gpt-secure": {
                "expected_release_digest": model_release_digest,
                "host": "gpt-secure.tinfoil.example",
                "repo": "tinfoilsh/confidential-gpt-secure",
            }
        },
        "repo": "tinfoilsh/confidential-model-router",
        "require_model_attestations": True,
    }
    serialized = json.dumps(provider_status, sort_keys=True)
    assert "SECRET_ENV_NAME" not in serialized
    assert "verifier_command" not in serialized


def test_public_confidentiality_status_omits_tinfoil_target_urls() -> None:
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
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/gpt-secure": {
                            "host": "gpt-secure.tinfoil.example",
                            "repo": "tinfoilsh/confidential-gpt-secure",
                            "expected_release_digest": VALID_RELEASE_DIGEST,
                            "attestation_url": (
                                "https://gpt-secure.tinfoil.example/.well-known/"
                                "tinfoil-attestation"
                            ),
                            "hpke_keys_url": (
                                "https://gpt-secure.tinfoil.example/.well-known/"
                                "hpke-keys"
                            ),
                        }
                    },
                },
            }
        }
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(
            include_provider_urls=True,
            include_routstr_tee=False,
        )

    provider_status = status["providers"][0]
    serialized = json.dumps(provider_status, sort_keys=True)
    target = provider_status["confidentiality_policy"]["model_attestation_targets"][
        "tinfoil/gpt-secure"
    ]
    assert "attestation_url" not in target
    assert "hpke_keys_url" not in target
    assert "tinfoil-attestation" not in serialized
    assert "hpke-keys" not in serialized


def test_public_confidentiality_status_rejects_blank_policy_identity_pins() -> None:
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
                "policy": {
                    "repo": "   ",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/gpt-secure": {
                            "host": "   ",
                            "repo": "tinfoilsh/confidential-gpt-secure",
                            "expected_release_digest": VALID_RELEASE_DIGEST,
                        }
                    },
                },
            }
        }
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(
            include_provider_urls=True,
            include_routstr_tee=False,
        )

    provider_status = status["providers"][0]
    assert "confidentiality_policy" not in provider_status


def test_public_confidentiality_status_includes_sanitized_policy_without_urls() -> None:
    model_release_digest = _digest("tinfoil-gpt-secure-release")
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
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/gpt-secure": {
                            "host": "gpt-secure.tinfoil.example",
                            "repo": "tinfoilsh/confidential-gpt-secure",
                            "expected_release_digest": model_release_digest,
                        }
                    },
                },
            }
        }
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(include_routstr_tee=False)

    provider_status = status["providers"][0]
    assert "base_url" not in provider_status
    assert "db_id" not in provider_status
    assert provider_status["confidentiality_policy"]["repo"] == (
        "tinfoilsh/confidential-model-router"
    )


def test_public_confidentiality_status_rejects_tinfoil_model_target_without_release_digest() -> (
    None
):
    public_policy = {
        "repo": "tinfoilsh/confidential-model-router",
        "expected_release_digest": VALID_RELEASE_DIGEST,
        "require_model_attestations": True,
        "model_attestation_targets": {
            "tinfoil/gpt-secure": {
                "host": "gpt-secure.tinfoil.example",
                "repo": "tinfoilsh/confidential-gpt-secure",
                "expected_code_measurement_fingerprint": "sha256:" + ("6" * 64),
            }
        },
    }
    assert (
        public_confidentiality_policy_binds_provider_proof(
            "tinfoil",
            "tinfoil",
            public_policy,
            _tinfoil_public_proof_claims(),
            ["tinfoil/gpt-secure"],
        )
        is False
    )

    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil"
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="tinfoil",
            verifier="unit-test-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_at=1_700_000_000,
            expires_at=int(time.time()) + 300,
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=_tinfoil_public_proof_claims(),
        )
    )
    provider.confidentiality_policy = lambda: SimpleNamespace(
        policy=public_policy
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(include_routstr_tee=False)

    confidentiality = status["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is False
    assert "proof_claims" not in confidentiality


def test_public_confidentiality_status_includes_sanitized_ppq_policy_pins() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.provider_type = "ppq-private"
    provider.upstream_name = "ppq-private"
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "repo": "ppq-ai/private-tee",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                    "attestation_bundle_url": "https://api.ppq.ai/private/attestation",
                    "verifier_command": "/opt/routstr/bin/ppq-verifier",
                },
            }
        }
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(
            include_provider_urls=True,
            include_routstr_tee=False,
        )

    provider_status = status["providers"][0]
    assert provider_status["confidentiality_policy"] == {
        "attestation_bundle_url_digest": (
            "sha256:c9ffb67a911a82f7a70ec60b658aa9fe2d42a123f49c1cf157b9f4bf3e1138b4"
        ),
        "expected_release_digest": VALID_RELEASE_DIGEST,
        "repo": "ppq-ai/private-tee",
    }
    serialized = json.dumps(provider_status, sort_keys=True)
    assert "https://api.ppq.ai/private/attestation" not in serialized
    assert "ppq-verifier" not in serialized


def test_public_confidentiality_status_rejects_nested_policy_lists_with_objects() -> None:
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
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/gpt-secure": {
                            "host": "gpt-secure.tinfoil.example",
                            "repo": "tinfoilsh/confidential-gpt-secure",
                            "allowed_release_digests": [
                                "sha256:" + ("1" * 64),
                                {"unexpected": "object"},
                            ],
                        }
                    },
                },
            }
        }
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(
            include_provider_urls=True,
            include_routstr_tee=False,
        )

    provider_status = status["providers"][0]
    serialized = json.dumps(provider_status, sort_keys=True)
    assert "confidentiality_policy" not in provider_status
    assert "unexpected" not in serialized


def test_public_confidentiality_status_rejects_nested_workload_lists_with_objects() -> None:
    provider = BaseUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.provider_type = "privatemode"
    provider.upstream_name = "privatemode"
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/gpt-oss-120b"],
                "policy": {
                    "manifest_digest": VALID_RELEASE_DIGEST,
                    "proxy_image_digest": "sha256:" + ("2" * 64),
                    "expected_workload_sans": ["backend.privatemode.example"],
                    "model_workload_bindings": {
                        "privatemode/gpt-oss-120b": {
                            "workload_sans": [
                                "backend.privatemode.example",
                                {"unexpected": "object"},
                            ],
                        }
                    },
                },
            }
        }
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(
            include_provider_urls=True,
            include_routstr_tee=False,
        )

    provider_status = status["providers"][0]
    serialized = json.dumps(provider_status, sort_keys=True)
    assert "confidentiality_policy" not in provider_status
    assert "unexpected" not in serialized


def test_public_confidentiality_status_rejects_duplicate_tinfoil_policy_targets() -> (
    None
):
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
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/gpt-secure": {
                            "host": "gpt-secure.tinfoil.example",
                            "repo": "tinfoilsh/confidential-gpt-secure",
                            "expected_release_digest": VALID_RELEASE_DIGEST,
                        },
                        " tinfoil/gpt-secure ": {
                            "host": "other.tinfoil.example",
                            "repo": "tinfoilsh/confidential-other",
                            "expected_release_digest": "sha256:" + ("2" * 64),
                        },
                    },
                },
            }
        }
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(
            include_provider_urls=True,
            include_routstr_tee=False,
        )

    provider_status = status["providers"][0]
    assert "confidentiality_policy" not in provider_status


def test_public_confidentiality_status_rejects_case_variant_tinfoil_policy_targets() -> (
    None
):
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
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/gpt-secure": {
                            "host": "gpt-secure.tinfoil.example",
                            "repo": "tinfoilsh/confidential-gpt-secure",
                            "expected_release_digest": VALID_RELEASE_DIGEST,
                        },
                        "TINFOIL/gpt-secure": {
                            "host": "other.tinfoil.example",
                            "repo": "tinfoilsh/confidential-other",
                            "expected_release_digest": "sha256:" + ("2" * 64),
                        },
                    },
                },
            }
        }
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(
            include_provider_urls=True,
            include_routstr_tee=False,
        )

    provider_status = status["providers"][0]
    assert "confidentiality_policy" not in provider_status


def test_public_confidentiality_status_rejects_duplicate_privatemode_policy_bindings() -> (
    None
):
    provider = BaseUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.provider_type = "privatemode"
    provider.upstream_name = "privatemode"
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/gpt-oss-120b"],
                "policy": {
                    "manifest_digest": VALID_RELEASE_DIGEST,
                    "proxy_image_digest": "sha256:" + ("2" * 64),
                    "expected_workload_sans": ["backend.privatemode.example"],
                    "model_workload_bindings": {
                        "privatemode/gpt-oss-120b": {
                            "workload_sans": ["backend.privatemode.example"],
                        },
                        " privatemode/gpt-oss-120b ": {
                            "workload_sans": ["other.privatemode.example"],
                        },
                    },
                    "expected_workload_identity_digest": "sha256:" + ("3" * 64),
                    "model_workload_binding_digest": "sha256:" + ("4" * 64),
                },
            }
        }
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(
            include_provider_urls=True,
            include_routstr_tee=False,
        )

    provider_status = status["providers"][0]
    assert "confidentiality_policy" not in provider_status


def test_public_confidentiality_status_rejects_case_variant_privatemode_policy_bindings() -> (
    None
):
    provider = BaseUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.provider_type = "privatemode"
    provider.upstream_name = "privatemode"
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/gpt-oss-120b"],
                "policy": {
                    "manifest_digest": VALID_RELEASE_DIGEST,
                    "proxy_image_digest": "sha256:" + ("2" * 64),
                    "expected_workload_sans": ["backend.privatemode.example"],
                    "model_workload_bindings": {
                        "privatemode/gpt-oss-120b": {
                            "workload_sans": ["backend.privatemode.example"],
                        },
                        "PRIVATEMODE/gpt-oss-120b": {
                            "workload_sans": ["other.privatemode.example"],
                        },
                    },
                    "expected_workload_identity_digest": "sha256:" + ("3" * 64),
                    "model_workload_binding_digest": "sha256:" + ("4" * 64),
                },
            }
        }
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(
            include_provider_urls=True,
            include_routstr_tee=False,
        )

    provider_status = status["providers"][0]
    assert "confidentiality_policy" not in provider_status


def test_public_confidentiality_status_rejects_boolean_policy_digest_pins() -> None:
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
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": True,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/gpt-secure": {
                            "host": "gpt-secure.tinfoil.example",
                            "repo": "tinfoilsh/confidential-gpt-secure",
                            "expected_release_digest": VALID_RELEASE_DIGEST,
                        }
                    },
                },
            }
        }
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(
            include_provider_urls=True,
            include_routstr_tee=False,
        )

    provider_status = status["providers"][0]
    assert "confidentiality_policy" not in provider_status


def test_public_confidentiality_status_rejects_malformed_policy_digest_pins() -> None:
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
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": "not-a-digest",
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/gpt-secure": {
                            "host": "gpt-secure.tinfoil.example",
                            "repo": "tinfoilsh/confidential-gpt-secure",
                            "expected_release_digest": VALID_RELEASE_DIGEST,
                        }
                    },
                },
            }
        }
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(
            include_provider_urls=True,
            include_routstr_tee=False,
        )

    provider_status = status["providers"][0]
    assert "confidentiality_policy" not in provider_status


def test_public_confidentiality_status_rejects_malformed_measurement_policy_pins() -> None:
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
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                    "expected_code_measurement_fingerprint": "not-a-digest",
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/gpt-secure": {
                            "host": "gpt-secure.tinfoil.example",
                            "repo": "tinfoilsh/confidential-gpt-secure",
                            "expected_release_digest": VALID_RELEASE_DIGEST,
                        }
                    },
                },
            }
        }
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(
            include_provider_urls=True,
            include_routstr_tee=False,
        )

    provider_status = status["providers"][0]
    assert "confidentiality_policy" not in provider_status


def test_public_confidentiality_status_rejects_malformed_privatemode_component_pins() -> None:
    provider = BaseUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.provider_type = "privatemode"
    provider.upstream_name = "privatemode"
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/gpt-oss-120b"],
                "policy": {
                    "manifest_digest": VALID_RELEASE_DIGEST,
                    "proxy_image_digest": "sha256:" + ("2" * 64),
                    "expected_coordinator_measurement": "not-a-digest",
                    "expected_workload_sans": ["backend.privatemode.example"],
                    "model_workload_bindings": {
                        "privatemode/gpt-oss-120b": {
                            "workload_sans": ["backend.privatemode.example"],
                        }
                    },
                },
            }
        }
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(
            include_provider_urls=True,
            include_routstr_tee=False,
        )

    provider_status = status["providers"][0]
    assert "confidentiality_policy" not in provider_status


def test_public_confidentiality_status_includes_sanitized_privatemode_policy_pins() -> None:
    proxy_image_digest = _digest("privatemode-policy-proxy-image")
    expected_workload_identity_digest = _digest("privatemode-policy-workload")
    model_workload_binding_digest = _digest("privatemode-policy-model-binding")
    provider = BaseUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.provider_type = "privatemode"
    provider.upstream_name = "privatemode"
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/gpt-oss-120b"],
                "policy": {
                    "manifest_digest": VALID_RELEASE_DIGEST,
                    "proxy_image_digest": proxy_image_digest,
                    "expected_workload_sans": ["backend.privatemode.example"],
                    "model_workload_bindings": {
                        "privatemode/gpt-oss-120b": {
                            "workload_sans": ["backend.privatemode.example"],
                        }
                    },
                    "expected_workload_identity_digest": expected_workload_identity_digest,
                    "model_workload_binding_digest": model_workload_binding_digest,
                    "api_key_env": "PRIVATEMODE_API_KEY",
                    "manifest_url": (
                        "https://cdn.confidential.cloud/privatemode/v2/manifest.json"
                    ),
                },
            }
        }
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(
            include_provider_urls=True,
            include_routstr_tee=False,
        )

    provider_status = status["providers"][0]
    assert provider_status["confidentiality_policy"] == {
        "expected_workload_sans": ["backend.privatemode.example"],
        "expected_workload_identity_digest": expected_workload_identity_digest,
        "manifest_digest": VALID_RELEASE_DIGEST,
        "model_workload_bindings": {
            "privatemode/gpt-oss-120b": {
                "workload_sans": ["backend.privatemode.example"],
            }
        },
        "model_workload_binding_digest": model_workload_binding_digest,
        "proxy_image_digest": proxy_image_digest,
    }
    serialized = json.dumps(provider_status, sort_keys=True)
    assert "PRIVATEMODE_API_KEY" not in serialized
    assert "manifest_url" not in serialized


def test_public_confidentiality_status_downgrades_unroutable_verified_proof() -> None:
    """Public status must not claim verified when routing proof checks fail."""
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
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/gpt-secure": {
                            "host": "gpt-secure.tinfoil.example",
                            "repo": "tinfoilsh/confidential-gpt-secure",
                            "expected_release_digest": VALID_RELEASE_DIGEST,
                        }
                    },
                },
            }
        }
    )
    policy = provider.confidentiality_policy()
    assert policy is not None
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="tinfoil",
            verifier="unit-test-verifier",
            policy_digest=policy.digest,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            model_ids=["tinfoil/gpt-secure"],
            verified_claims={
                "transport": "ehbp",
                "repo": "tinfoilsh/confidential-model-router",
                "attested_hpke_public_key_hex": "b" * 64,
                "enclave_measurement_fingerprint": "sha256:" + ("1" * 64),
                "code_measurement_fingerprint": "sha256:" + ("2" * 64),
                "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
                "attestation_report_digest": "sha256:" + ("3" * 64),
                "release_digest": VALID_RELEASE_DIGEST,
                "tls_public_key_fingerprint_sha256": "sha256:not-a-real-digest",
                "verification_steps": {
                    "hardware_attestation_report": True,
                    "hardware_certificate_chain": True,
                    "code_transparency": True,
                    "measurement_match": True,
                    "attested_transport_key_binding": True,
                    "freshness": True,
                },
            },
        )
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(include_routstr_tee=False)

    confidentiality = status["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert confidentiality["verified_at"] is None
    assert confidentiality["expires_at"] is None
    assert "verified_claims" not in confidentiality
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_public_confidentiality_status_requires_verified_claims_to_claim_verified() -> (
    None
):
    """Public status must not claim verification without publishable proof claims."""
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="tinfoil",
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        model_ids=["tinfoil/gpt-secure"],
        verified_claims={},
    )

    confidentiality = _public_confidentiality_status(
        status,
        provider_type="tinfoil",
        currently_verified=True,
    )

    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert confidentiality["verified_at"] is None
    assert confidentiality["expires_at"] is None
    assert "proof_claims" not in confidentiality
    assert "verified_claims_digest" not in confidentiality


def test_public_confidentiality_status_requires_tinfoil_model_attestation_coverage() -> (
    None
):
    verified_claims = _tinfoil_public_proof_claims()
    verified_claims.pop("model_attestations")
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="tinfoil",
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        model_ids=["tinfoil/gpt-secure"],
        verified_claims=verified_claims,
    )

    confidentiality = _public_confidentiality_status(
        status,
        provider_type="tinfoil",
        currently_verified=True,
    )

    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_public_confidentiality_status_requires_public_policy_binding() -> None:
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="tinfoil",
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        model_ids=["tinfoil/gpt-secure"],
        verified_claims=_tinfoil_public_proof_claims(),
    )

    confidentiality = _public_confidentiality_status(
        status,
        provider_type="tinfoil",
        currently_verified=True,
        public_policy=None,
    )

    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_public_confidentiality_status_rejects_weak_public_policy_binding() -> None:
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="tinfoil",
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        model_ids=["tinfoil/gpt-secure"],
        verified_claims=_tinfoil_public_proof_claims(),
    )

    confidentiality = _public_confidentiality_status(
        status,
        provider_type="tinfoil",
        currently_verified=True,
        public_policy={
            "repo": "tinfoilsh/confidential-model-router",
            "require_model_attestations": True,
            "model_attestation_targets": {
                "tinfoil/gpt-secure": {
                    "repo": "tinfoilsh/confidential-gpt-secure",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                }
            },
        },
    )

    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_public_confidentiality_status_rejects_payload_policy_digest_mismatch() -> (
    None
):
    verified_claims = _tinfoil_public_proof_claims()
    verified_claims["payload_policy_digest"] = "sha256:" + ("9" * 64)
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="tinfoil",
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        model_ids=["tinfoil/gpt-secure"],
        verified_claims=verified_claims,
    )

    confidentiality = _public_confidentiality_status(
        status,
        provider_type="tinfoil",
        currently_verified=True,
        public_policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": VALID_RELEASE_DIGEST,
            "require_model_attestations": True,
            "model_attestation_targets": {
                "tinfoil/gpt-secure": {
                    "repo": "tinfoilsh/confidential-gpt-secure",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                }
            },
        },
    )

    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_public_confidentiality_status_rejects_payload_evidence_digest_mismatch() -> (
    None
):
    verified_claims = _tinfoil_public_proof_claims()
    verified_claims["payload_evidence_digest"] = _digest("other-evidence")
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="tinfoil",
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        model_ids=["tinfoil/gpt-secure"],
        verified_claims=verified_claims,
    )

    confidentiality = _public_confidentiality_status(
        status,
        provider_type="tinfoil",
        currently_verified=True,
        public_policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": VALID_RELEASE_DIGEST,
            "require_model_attestations": True,
            "model_attestation_targets": {
                "tinfoil/gpt-secure": {
                    "repo": "tinfoilsh/confidential-gpt-secure",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                }
            },
        },
    )

    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_public_confidentiality_status_rejects_ppq_policy_without_bundle_pin() -> None:
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="ppq-private-tee",
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        model_ids=["private/gpt-oss-120b"],
        verified_claims=_ppq_private_public_proof_claims(),
    )

    confidentiality = _public_confidentiality_status(
        status,
        provider_type="ppq-private",
        currently_verified=True,
        public_policy={
            "repo": "ppq-ai/private-tee",
            "expected_code_measurement_fingerprint": "sha256:" + ("2" * 64),
        },
    )

    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_public_confidentiality_status_accepts_ppq_policy_without_proxy_pin() -> None:
    proof_claims = _ppq_private_public_proof_claims()
    proof_claims.pop("proxy_binary_digest", None)
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="ppq-private-tee",
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        model_ids=["private/gpt-oss-120b"],
        verified_claims=proof_claims,
    )

    confidentiality = _public_confidentiality_status(
        status,
        provider_type="ppq-private",
        currently_verified=True,
        public_policy={
            "repo": "ppq-ai/private-tee",
            "attestation_bundle_url_digest": _digest("ppq-attestation-bundle-url"),
            "expected_code_measurement_fingerprint": _digest(
                "provider-code-measurement"
            ),
            "require_model_attestations": True,
            "model_attestation_targets": {
                "private/gpt-oss-120b": {
                    "repo": "tinfoilsh/confidential-gpt-oss-120b",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                }
            },
        },
    )

    assert confidentiality["verified"] is True
    assert confidentiality["verifier"] == "unit-test-verifier"
    assert confidentiality["evidence_digest"] == VALID_EVIDENCE_DIGEST
    assert "proxy_binary_digest" not in confidentiality["proof_claims"]


def test_public_confidentiality_status_allows_privatemode_manifest_log_policy() -> (
    None
):
    claims = _privatemode_public_proof_claims()
    claims["proxy_base_url"] = "http://127.0.0.1:8080/v1"
    claims["expected_workload_identity_digest"] = _sha256_json_digest(
        {
            "ids": [],
            "sans": ["backend.privatemode.example"],
        }
    )
    claims["model_workload_binding_digest"] = _sha256_json_digest(
        {
            "privatemode/gpt-oss-120b": {
                "workload_ids": [],
                "workload_sans": ["backend.privatemode.example"],
            }
        }
    )
    _recompute_privatemode_key_release_binding(claims)
    claims["manifest_log_digest"] = _digest("privatemode-manifest-log")
    claims["manifest_log_manifest_digest"] = claims["manifest_digest"]
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="privatemode",
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_at=1_700_000_000,
        expires_at=int(time.time()) + 300,
        model_ids=["privatemode/gpt-oss-120b"],
        verified_claims=claims,
    )

    confidentiality = _public_confidentiality_status(
        status,
        provider_type="privatemode",
        currently_verified=True,
        public_policy={
            "manifest_log_dir": "/state/privatemode-manifest-log",
            "proxy_binary_digest": _digest("privatemode-proxy-binary"),
            "expected_trust_tier": "app-e2ee",
            "expected_gpu_attestation_policy": "nvidia-ocsp-good-only",
            "expected_workload_sans": ["backend.privatemode.example"],
            "model_workload_bindings": {
                "privatemode/gpt-oss-120b": {
                    "workload_sans": ["backend.privatemode.example"],
                }
            },
        },
    )

    assert confidentiality["verified"] is True
    assert confidentiality["proof_claims"]["manifest_log_digest"] == claims[
        "manifest_log_digest"
    ]


def test_public_confidentiality_status_sanitizes_privatemode_manifest_log_policy() -> (
    None
):
    claims = _privatemode_public_proof_claims()
    claims["expected_workload_identity_digest"] = _sha256_json_digest(
        {
            "ids": [],
            "sans": ["backend.privatemode.example"],
        }
    )
    claims["model_workload_binding_digest"] = _sha256_json_digest(
        {
            "privatemode/gpt-oss-120b": {
                "workload_ids": [],
                "workload_sans": ["backend.privatemode.example"],
            }
        }
    )
    _recompute_privatemode_key_release_binding(claims)
    claims["manifest_log_digest"] = _digest("privatemode-manifest-log")
    claims["manifest_log_manifest_digest"] = claims["manifest_digest"]
    provider = BaseUpstreamProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="test",
    )
    provider.provider_type = "privatemode"
    provider.upstream_name = "privatemode"
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "mode": "privatemode",
                "model_ids": ["privatemode/gpt-oss-120b"],
                "policy": {
                    "manifest_log_dir": "/state/privatemode-manifest-log",
                    "proxy_binary_digest": _digest("privatemode-proxy-binary"),
                    "expected_trust_tier": "app-e2ee",
                    "expected_gpu_attestation_policy": "nvidia-ocsp-good-only",
                    "expected_workload_sans": ["backend.privatemode.example"],
                    "model_workload_bindings": {
                        "privatemode/gpt-oss-120b": {
                            "workload_sans": ["backend.privatemode.example"],
                        }
                    },
                },
            }
        }
    )
    claims["payload_policy_digest"] = provider.confidentiality_policy().digest
    claims["proxy_base_url"] = "http://127.0.0.1:8080/v1"
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="privatemode",
            verifier="unit-test-verifier",
            policy_digest=provider.confidentiality_policy().digest,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_at=1_700_000_000,
            expires_at=int(time.time()) + 300,
            model_ids=["privatemode/gpt-oss-120b"],
            verified_claims=claims,
        )
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(include_routstr_tee=False)

    provider_status = status["providers"][0]
    assert provider_status["confidentiality_policy"]["manifest_log_dir"] == (
        "/state/privatemode-manifest-log"
    )
    confidentiality = provider_status["confidentiality"]
    assert confidentiality["verified"] is True
    assert confidentiality["proof_claims"]["manifest_log_digest"] == claims[
        "manifest_log_digest"
    ]


def test_public_confidentiality_status_rejects_privatemode_workload_identity_mismatch() -> (
    None
):
    claims = _privatemode_public_proof_claims()
    claims["expected_workload_identity_digest"] = _sha256_json_digest(
        {
            "ids": [],
            "sans": ["backend.privatemode.example"],
        }
    )
    claims["model_workload_binding_digest"] = _sha256_json_digest(
        {
            "privatemode/gpt-oss-120b": {
                "workload_ids": [],
                "workload_sans": ["backend.privatemode.example"],
            }
        }
    )
    _recompute_privatemode_key_release_binding(claims)
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="privatemode",
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        model_ids=["privatemode/gpt-oss-120b"],
        verified_claims=claims,
    )

    confidentiality = _public_confidentiality_status(
        status,
        provider_type="privatemode",
        currently_verified=True,
        public_policy={
            "proxy_binary_digest": "sha256:" + ("6" * 64),
            "expected_workload_sans": ["wrong.privatemode.example"],
            "model_workload_bindings": {
                "privatemode/gpt-oss-120b": {
                    "workload_sans": ["backend.privatemode.example"],
                }
            },
        },
    )

    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_public_confidentiality_status_rejects_privatemode_gpu_policy_mismatch() -> (
    None
):
    claims = _privatemode_public_proof_claims()
    claims["gpu_attestation_policy"] = "reference-only"
    claims["expected_workload_identity_digest"] = _sha256_json_digest(
        {
            "ids": [],
            "sans": ["backend.privatemode.example"],
        }
    )
    claims["model_workload_binding_digest"] = _sha256_json_digest(
        {
            "privatemode/gpt-oss-120b": {
                "workload_ids": [],
                "workload_sans": ["backend.privatemode.example"],
            }
        }
    )
    _recompute_privatemode_key_release_binding(claims)
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="privatemode",
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        model_ids=["privatemode/gpt-oss-120b"],
        verified_claims=claims,
    )

    confidentiality = _public_confidentiality_status(
        status,
        provider_type="privatemode",
        currently_verified=True,
        public_policy={
            "manifest_digest": _digest("privatemode-manifest"),
            "proxy_binary_digest": _digest("privatemode-proxy-binary"),
            "expected_trust_tier": "app-e2ee",
            "expected_gpu_attestation_policy": "nvidia-ocsp-good-only",
            "expected_workload_sans": ["backend.privatemode.example"],
            "model_workload_bindings": {
                "privatemode/gpt-oss-120b": {
                    "workload_sans": ["backend.privatemode.example"],
                }
            },
        },
    )

    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_public_confidentiality_status_rejects_privatemode_policy_without_app_e2ee_trust_tier() -> (
    None
):
    claims = _privatemode_public_proof_claims()
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="privatemode",
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        model_ids=["privatemode/gpt-oss-120b"],
        verified_claims=claims,
    )

    confidentiality = _public_confidentiality_status(
        status,
        provider_type="privatemode",
        currently_verified=True,
        public_policy={
            "manifest_digest": _digest("privatemode-manifest"),
            "proxy_binary_digest": _digest("privatemode-proxy-binary"),
            "expected_workload_sans": ["gpt-oss-120b.default.svc.cluster.local"],
            "model_workload_bindings": {
                "privatemode/gpt-oss-120b": {
                    "workload_sans": ["gpt-oss-120b.default.svc.cluster.local"],
                }
            },
        },
    )

    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_public_confidentiality_status_rejects_duplicate_verified_model_ids() -> None:
    verified_claims = _tinfoil_public_proof_claims()
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="tinfoil",
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        model_ids=["tinfoil/gpt-secure", "tinfoil/gpt-secure"],
        verified_claims=verified_claims,
    )

    confidentiality = _public_confidentiality_status(
        status,
        provider_type="tinfoil",
        currently_verified=True,
    )

    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_public_confidentiality_status_downgrades_malformed_public_claims() -> None:
    """Verified public status must not hide malformed proof-bearing claims."""
    verified_claims = _tinfoil_public_proof_claims()
    verified_claims["manifest_digest"] = "not-a-digest"
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
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/gpt-secure": {
                            "host": "gpt-secure.tinfoil.example",
                            "repo": "tinfoilsh/confidential-gpt-secure",
                            "expected_release_digest": VALID_RELEASE_DIGEST,
                        }
                    },
                },
            }
        }
    )
    policy = provider.confidentiality_policy()
    assert policy is not None
    verified_claims["payload_policy_digest"] = policy.digest
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="tinfoil",
            verifier="unit-test-verifier",
            policy_digest=policy.digest,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=verified_claims,
        )
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(include_routstr_tee=False)

    confidentiality = status["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert confidentiality["verified_at"] is None
    assert confidentiality["expires_at"] is None
    assert "verified_claims" not in confidentiality
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_public_confidentiality_status_redacts_malformed_copied_selectors() -> None:
    """Malformed copied selector objects must not leak through public status."""
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil"
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="tinfoil",
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        model_ids=["tinfoil/gpt-secure"],
        verified_claims=_tinfoil_public_proof_claims(),
    ).copy(
        update={
            "model_ids": [
                "tinfoil/gpt-secure",
                {"api_key": "SECRET_SELECTOR"},
            ],
        }
    )
    provider.set_confidentiality_status(status)

    with patch("routstr.proxy._upstreams", [provider]):
        public_status = get_confidentiality_status(include_routstr_tee=False)

    confidentiality = public_status["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is False
    assert confidentiality["model_ids"] == ["tinfoil/gpt-secure"]
    serialized = json.dumps(public_status)
    assert "SECRET_SELECTOR" not in serialized


def test_public_confidentiality_status_rejects_non_list_verified_selectors() -> None:
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="tinfoil",
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        model_ids=["tinfoil/gpt-secure"],
        verified_claims=_tinfoil_public_proof_claims(),
    ).copy(update={"model_ids": ("tinfoil/gpt-secure",)})

    public_status = _public_confidentiality_status(
        status,
        provider_type="tinfoil",
        currently_verified=True,
    )

    assert public_status["verified"] is False
    assert public_status["verifier"] is None
    assert public_status["evidence_digest"] is None
    assert public_status["model_ids"] == ["tinfoil/gpt-secure"]
    assert "verified_claims_digest" not in public_status
    assert "proof_claims" not in public_status


def test_public_confidentiality_status_downgrades_raw_claims_provider_mode_mismatch() -> (
    None
):
    """Raw-claims snapshots must not bypass provider verifier selection."""
    provider = BaseUpstreamProvider(
        base_url="https://custom.example/v1",
        api_key="test",
    )
    provider.provider_type = "custom"
    provider.upstream_name = "custom"
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="tinfoil",
            verifier="unit-test-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=_tinfoil_public_proof_claims(),
        )
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(
            include_routstr_tee=False,
            include_verified_claims=True,
        )

    confidentiality = status["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert confidentiality["verified_at"] is None
    assert confidentiality["expires_at"] is None
    assert "verified_claims" not in confidentiality


def test_public_confidentiality_status_rejects_provider_type_mode_mismatch_directly() -> (
    None
):
    """The public sanitizer must fail closed even if caller state is stale."""
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="tinfoil",
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        model_ids=["tinfoil/gpt-secure"],
        verified_claims=_tinfoil_public_proof_claims(),
    )

    confidentiality = _public_confidentiality_status(
        status,
        provider_type="custom",
        currently_verified=True,
        public_policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": VALID_RELEASE_DIGEST,
            "require_model_attestations": True,
            "model_attestation_targets": {
                "tinfoil/gpt-secure": {
                    "repo": "tinfoilsh/confidential-gpt-secure",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                }
            },
        },
    )

    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert confidentiality["verified_at"] is None
    assert confidentiality["expires_at"] is None
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_public_confidentiality_status_never_exposes_raw_verified_claims() -> None:
    """Public status must only expose digests and audited proof-claim subsets."""
    verified_claims = _tinfoil_public_proof_claims()
    verified_claims["model_attestations"] = {
        "tinfoil/gpt-secure": {
            "repo": "tinfoilsh/confidential-gpt-secure",
            "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
            "attestation_report_digest": _digest("gpt-secure-attestation-report"),
            "attested_hpke_public_key_hex": "b" * 64,
            "enclave_measurement_fingerprint": _digest("gpt-secure-enclave"),
            "code_measurement_fingerprint": _digest("gpt-secure-code"),
            "tls_public_key_fingerprint_sha256": _digest("gpt-secure-tls"),
            "release_digest": VALID_RELEASE_DIGEST,
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
    verified_claims["debug_trace"] = "SECRET_VERIFIER_TRACE"
    verified_claims["raw_attestation_document"] = {
        "certificate_chain": "SECRET_CERTIFICATE_CHAIN"
    }
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
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                    "require_model_attestations": True,
                    "model_attestation_targets": {
                        "tinfoil/gpt-secure": {
                            "host": "gpt-secure.tinfoil.example",
                            "repo": "tinfoilsh/confidential-gpt-secure",
                            "expected_release_digest": VALID_RELEASE_DIGEST,
                        }
                    },
                },
            }
        }
    )
    policy = provider.confidentiality_policy()
    assert policy is not None
    verified_claims["payload_policy_digest"] = policy.digest
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="tinfoil",
            verifier="unit-test-verifier",
            policy_digest=policy.digest,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            model_ids=["tinfoil/gpt-secure"],
            verified_claims=verified_claims,
        )
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(
            include_routstr_tee=False,
            include_verified_claims=True,
        )

    confidentiality = status["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is True
    assert "verified_claims" not in confidentiality
    assert "verified_claims_digest" in confidentiality
    assert confidentiality["proof_claims"]["repo"] == (
        "tinfoilsh/confidential-model-router"
    )
    serialized = json.dumps(status, sort_keys=True)
    assert "SECRET_VERIFIER_TRACE" not in serialized
    assert "SECRET_CERTIFICATE_CHAIN" not in serialized


def test_public_confidentiality_status_rejects_non_canonical_verified_claims() -> None:
    """Digest-bound public status must reject non-standard JSON verifier claims."""
    verified_claims = _tinfoil_public_proof_claims()
    verified_claims["debug_score"] = float("nan")
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="tinfoil",
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        model_ids=["tinfoil/gpt-secure"],
        verified_claims=verified_claims,
    )

    confidentiality = _public_confidentiality_status(
        status.copy(update={"verified_claims": verified_claims}),
        currently_verified=True,
    )
    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert "verified_claims" not in confidentiality
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_public_confidentiality_status_drops_unverified_claims() -> None:
    """Unverified copied claims are not public proof and may be malformed."""
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.upstream_name = "tinfoil"
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=False,
            mode="tinfoil",
            model_ids=["tinfoil/gpt-secure"],
            verifier="failed-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            failure_reason="verification failed",
            verified_claims={
                "repo": "tinfoilsh/confidential-model-router",
                "debug": _NonSerializableClaim(),
            },
        )
    )

    with patch("routstr.proxy._upstreams", [provider]):
        status = get_confidentiality_status(include_routstr_tee=False)

    confidentiality = status["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is False
    assert "failure_reason_digest" in confidentiality
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality
