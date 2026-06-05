import hashlib
import json
import time
from types import SimpleNamespace

import pytest
from pydantic.v1 import ValidationError

from routstr.upstream.base import (
    BaseUpstreamProvider,
    ConfidentialityStatus,
    ConfidentialVerifierPolicy,
)
from routstr.upstream.confidential_verifiers import (
    _privatemode_component_policy_violations,
    _privatemode_model_workload_policy_violations,
    _require_tinfoil_model_attestation_claims,
    _tinfoil_model_attestation_policy_violations,
    _validate_privatemode_verifier_result,
    _validate_tinfoil_verifier_result,
)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


VALID_POLICY_DIGEST = _digest("policy")
VALID_EVIDENCE_DIGEST = _digest("evidence")
VALID_RELEASE_DIGEST = _digest("release")
VALID_ENCLAVE_MEASUREMENT = _digest("enclave-measurement")
VALID_CODE_MEASUREMENT = _digest("code-measurement")
VALID_ATTESTATION_REPORT_DIGEST = _digest("attestation-report")
VALID_TLS_DIGEST = _digest("tls-key")
VALID_PRIVATEMODE_GPU_POLICY = "nvidia-ocsp-good-only"
VALID_PRIVATEMODE_TRUST_TIER = "app-e2ee"
VALID_EHBP_PUBLIC_KEY_HEX = "22" * 32
VALID_EHBP_KEY_CONFIG_B64 = (
    "AAAgIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIABAEAAAI="
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


def _tinfoil_runtime_policy(
    *,
    transport_security: str | None = None,
) -> ConfidentialVerifierPolicy:
    policy: dict[str, object] = {
        "repo": "tinfoilsh/confidential-model-router",
        "expected_release_digest": VALID_RELEASE_DIGEST,
        "require_model_attestations": True,
        "model_attestation_targets": {
            "tinfoil/gpt-secure": {
                "host": "gpt-secure.tinfoil.example",
                "repo": "tinfoilsh/confidential-gpt-secure",
                "expected_release_digest": VALID_RELEASE_DIGEST,
                "expected_code_measurement_fingerprint": VALID_CODE_MEASUREMENT,
            }
        },
    }
    if transport_security is not None:
        policy["transport_security"] = transport_security
    return ConfidentialVerifierPolicy(
        provider_type="tinfoil",
        mode="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        model_ids=["tinfoil/gpt-secure"],
        policy=policy,
    )


def _privatemode_runtime_policy() -> ConfidentialVerifierPolicy:
    return ConfidentialVerifierPolicy(
        provider_type="privatemode",
        mode="privatemode",
        base_url="http://127.0.0.1:8080/v1",
        model_ids=["privatemode/kimi-k2-6"],
        policy={
            "manifest_digest": _digest("privatemode-manifest"),
            "proxy_binary_digest": _digest("privatemode-proxy-binary"),
            "expected_trust_tier": VALID_PRIVATEMODE_TRUST_TIER,
            "expected_gpu_attestation_policy": VALID_PRIVATEMODE_GPU_POLICY,
            "expected_workload_sans": ["workload-kimi-k2-6"],
            "model_workload_bindings": {
                "privatemode/kimi-k2-6": {
                    "workload_sans": ["workload-kimi-k2-6"],
                }
            },
        },
    )


def _privatemode_verifier_result(policy: ConfidentialVerifierPolicy) -> dict[str, object]:
    expected_workload_identity_digest = _sha256_json_digest(
        {
            "ids": [],
            "sans": ["workload-kimi-k2-6"],
        }
    )
    model_workload_binding_digest = _sha256_json_digest(
        {
            "privatemode/kimi-k2-6": {
                "workload_ids": [],
                "workload_sans": ["workload-kimi-k2-6"],
            }
        }
    )
    claims: dict[str, object] = {
        "payload_policy_digest": policy.digest,
        "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
        "payload_verification_nonce": "nonce",
        "transport": "privatemode-proxy",
        "proxy_base_url": "http://127.0.0.1:8080/v1",
        "manifest_digest": _digest("privatemode-manifest"),
        "proxy_binary_digest": _digest("privatemode-proxy-binary"),
        "coordinator_measurement": _digest("privatemode-coordinator"),
        "secret_service_measurement": _digest("privatemode-secret-service"),
        "ai_worker_measurement": _digest("privatemode-ai-worker"),
        "trust_tier": VALID_PRIVATEMODE_TRUST_TIER,
        "gpu_attestation_policy": VALID_PRIVATEMODE_GPU_POLICY,
        "attested_workload_identity_digest": _digest("attested-workload-identity"),
        "attested_workload_policy_digest": _digest("privatemode-ai-worker"),
        "expected_workload_identity_digest": expected_workload_identity_digest,
        "model_workload_binding_digest": model_workload_binding_digest,
        "coordinator_attestation_doc_digest": _digest("coordinator-attestation-doc"),
        "mesh_ca_digest": _digest("mesh-ca"),
        "secret_service_certificate_digest": _digest("secret-service-certificate"),
        "ai_worker_manifest_digest": _digest("ai-worker-manifest"),
        "nvidia_ocsp_policy_header_digest": _digest("nvidia-ocsp-policy-header"),
        "nvidia_ocsp_policy_mac_digest": _digest("nvidia-ocsp-policy-mac"),
        "prompt_encryption_ciphertext_digest": _digest("prompt-encryption"),
        "inference_secret_id_digest": _digest("inference-secret"),
        "selected_model_ids": ["privatemode/kimi-k2-6"],
        "verification_steps": _privatemode_verification_steps(),
    }
    claims["key_release_binding"] = _privatemode_key_release_binding_digest(claims)
    return {
        "verified": True,
        "verifier": "routstr-privatemode-go-verifier",
        "policy_digest": policy.digest,
        "evidence_digest": VALID_EVIDENCE_DIGEST,
        "verification_nonce": "nonce",
        "claims": claims,
    }


def _tinfoil_model_evidence() -> list[dict[str, object]]:
    return [
        {
            "model_id": "tinfoil/gpt-secure",
            "hpke_public_key_hex": VALID_EHBP_PUBLIC_KEY_HEX,
            "attestation": {
                "format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
                "report_digest": VALID_ATTESTATION_REPORT_DIGEST,
            },
        }
    ]


def _tinfoil_verifier_result(
    *,
    policy_digest: str,
    include_tls: bool = True,
    model_include_tls: bool = True,
    tls_value: object = VALID_TLS_DIGEST,
) -> dict[str, object]:
    model_claims: dict[str, object] = {
        "repo": "tinfoilsh/confidential-gpt-secure",
        "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
        "attestation_report_digest": VALID_ATTESTATION_REPORT_DIGEST,
        "attested_hpke_public_key_hex": VALID_EHBP_PUBLIC_KEY_HEX,
        "enclave_measurement_fingerprint": VALID_ENCLAVE_MEASUREMENT,
        "code_measurement_fingerprint": VALID_CODE_MEASUREMENT,
        "release_digest": VALID_RELEASE_DIGEST,
        "verification_steps": _ehbp_verification_steps(),
    }
    if model_include_tls:
        model_claims["tls_public_key_fingerprint_sha256"] = tls_value

    claims: dict[str, object] = {
        "repo": "tinfoilsh/confidential-model-router",
        "payload_policy_digest": policy_digest,
        "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
        "payload_verification_nonce": "nonce",
        "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
        "attestation_report_digest": VALID_ATTESTATION_REPORT_DIGEST,
        "attested_hpke_public_key_hex": VALID_EHBP_PUBLIC_KEY_HEX,
        "enclave_measurement_fingerprint": VALID_ENCLAVE_MEASUREMENT,
        "code_measurement_fingerprint": VALID_CODE_MEASUREMENT,
        "release_digest": VALID_RELEASE_DIGEST,
        "verification_steps": _ehbp_verification_steps(),
        "model_attestations": {"tinfoil/gpt-secure": model_claims},
    }
    if include_tls:
        claims["tls_public_key_fingerprint_sha256"] = tls_value
    return {
        "verified": True,
        "verifier": "routstr-tinfoil-go-verifier",
        "policy_digest": policy_digest,
        "evidence_digest": VALID_EVIDENCE_DIGEST,
        "verification_nonce": "nonce",
        "claims": claims,
    }


def test_static_provider_settings_cannot_mark_confidentiality_verified() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"

    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "verified": True,
                "mode": "tinfoil",
                "expires_at": 4_102_444_800,
                "model_ids": ["tinfoil/gpt-secure"],
                "verifier": "static-config",
                "policy_digest": VALID_POLICY_DIGEST,
                "evidence_digest": VALID_EVIDENCE_DIGEST,
                "verified_claims": {"tee": "static"},
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                },
            }
        }
    )

    policy = provider.confidentiality_policy()
    assert policy is not None
    status = provider.confidentiality_status()
    assert status.enabled is True
    assert status.verified is False
    assert status.verified_at is None
    assert status.expires_at is None
    assert status.mode == "tinfoil"
    assert status.model_ids == ["tinfoil/gpt-secure"]
    assert status.model_id_prefixes == []
    assert status.policy_digest == policy.digest
    assert status.verifier is None
    assert status.evidence_digest is None
    assert status.verified_claims == {}
    assert status.failure_reason == "static configuration is not attestation evidence"


def test_confidentiality_policy_digest_is_derived_from_provider_settings() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"

    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": ["glm-5-1"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "allowed_predicate_types": [
                        "https://tinfoil.sh/predicate/snp-tdx-multiplatform/v1"
                    ],
                },
            }
        }
    )

    policy = provider.confidentiality_policy()
    status = provider.confidentiality_status()

    assert policy is not None
    assert policy.provider_type == "tinfoil"
    assert policy.mode == "tinfoil"
    assert policy.model_ids == ["glm-5-1"]
    assert policy.digest.startswith("sha256:")
    assert status.policy_digest == policy.digest
    assert status.verified is False


def test_confidential_provider_settings_require_verifier_policy_for_model_selectors() -> (
    None
):
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"

    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
            }
        }
    )

    assert provider.confidentiality_policy() is None
    status = provider.confidentiality_status()
    assert status.enabled is False
    assert status.verified is False
    assert status.mode == "none"
    assert status.policy_digest is None
    assert status.failure_reason == (
        "confidential verifier policy is required when confidentiality mode "
        "or model selectors are configured"
    )


def test_confidentiality_policy_rejects_unsupported_mode_from_provider_settings() -> (
    None
):
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"

    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "custom-confidential-mode",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                },
            }
        }
    )

    status = provider.confidentiality_status()
    assert provider.confidentiality_policy() is None
    assert status.enabled is False
    assert status.verified is False
    assert status.mode == "none"
    assert status.policy_digest is None
    assert status.failure_reason == (
        "confidentiality mode must be tinfoil, ppq-private-tee, or privatemode"
    )


def test_confidentiality_policy_rejects_duplicate_model_ids() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"

    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": ["glm-5-1", "glm-5-1"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "allowed_predicate_types": [
                        "https://tinfoil.sh/predicate/snp-tdx-multiplatform/v1"
                    ],
                },
            }
        }
    )

    assert provider.confidentiality_policy() is None
    status = provider.confidentiality_status()
    assert status.enabled is False
    assert status.mode == "none"
    assert "model_ids must not contain duplicates" in (status.failure_reason or "")


def test_confidentiality_policy_rejects_case_variant_duplicate_model_ids() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"

    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": ["GLM-5-1", "glm-5-1"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "allowed_predicate_types": [
                        "https://tinfoil.sh/predicate/snp-tdx-multiplatform/v1"
                    ],
                },
            }
        }
    )

    assert provider.confidentiality_policy() is None
    status = provider.confidentiality_status()
    assert status.enabled is False
    assert status.mode == "none"
    assert "model_ids must not contain duplicates" in (status.failure_reason or "")


def test_confidentiality_policy_rejects_scalar_model_ids() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"

    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": "glm-5-1",
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "allowed_predicate_types": [
                        "https://tinfoil.sh/predicate/snp-tdx-multiplatform/v1"
                    ],
                },
            }
        }
    )

    assert provider.confidentiality_policy() is None
    status = provider.confidentiality_status()
    assert status.enabled is False
    assert status.mode == "none"
    assert "model_ids must be a list of non-empty strings" in (
        status.failure_reason or ""
    )


def test_tinfoil_runtime_policy_rejects_model_target_on_router_origin() -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="tinfoil",
        mode="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        model_ids=["tinfoil/kimi-k2-6"],
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": _digest("router-release"),
            "require_model_attestations": True,
            "model_attestation_targets": {
                "tinfoil/kimi-k2-6": {
                    "attestation_url": (
                        "https://inference.tinfoil.sh/.well-known/"
                        "tinfoil-attestation"
                    ),
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": _digest("kimi-release"),
                }
            },
        },
    )

    violations = _tinfoil_model_attestation_policy_violations(policy)

    assert (
        "model_attestation_targets for tinfoil/kimi-k2-6 must use a distinct "
        "model enclave origin, not the Tinfoil router origin"
    ) in violations


def test_tinfoil_runtime_validation_requires_tls_binding_by_default() -> None:
    policy = _tinfoil_runtime_policy()
    with pytest.raises(ValueError, match="TLS public key binding claim is required"):
        _validate_tinfoil_verifier_result(
            verifier_result=_tinfoil_verifier_result(
                policy_digest=policy.digest,
                include_tls=False,
            ),
            policy=policy,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verification_nonce="nonce",
            attestation_format="https://tinfoil.sh/predicate/sev-snp-guest/v2",
            attestation_report_digest=VALID_ATTESTATION_REPORT_DIGEST,
            ehbp_public_key_hex=VALID_EHBP_PUBLIC_KEY_HEX,
            ehbp_key_config_b64=VALID_EHBP_KEY_CONFIG_B64,
            ehbp_claims={},
            model_attestations=_tinfoil_model_evidence(),
        )


def test_tinfoil_runtime_validation_rejects_selected_model_without_model_attestation_policy() -> (
    None
):
    policy = _tinfoil_runtime_policy()
    policy.policy.pop("require_model_attestations")
    policy.policy.pop("model_attestation_targets")
    verifier_result = _tinfoil_verifier_result(policy_digest=policy.digest)
    claims = verifier_result["claims"]
    assert isinstance(claims, dict)
    claims.pop("model_attestations")

    with pytest.raises(
        ValueError,
        match="Tinfoil require_model_attestations must be true for selected model routing",
    ):
        _validate_tinfoil_verifier_result(
            verifier_result=verifier_result,
            policy=policy,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verification_nonce="nonce",
            attestation_format="https://tinfoil.sh/predicate/sev-snp-guest/v2",
            attestation_report_digest=VALID_ATTESTATION_REPORT_DIGEST,
            ehbp_public_key_hex=VALID_EHBP_PUBLIC_KEY_HEX,
            ehbp_key_config_b64=VALID_EHBP_KEY_CONFIG_B64,
            ehbp_claims={},
            model_attestations=[],
        )


def test_shared_model_attestation_claim_helper_rejects_selected_model_without_policy() -> (
    None
):
    policy = _tinfoil_runtime_policy()
    policy.policy.pop("require_model_attestations")
    policy.policy.pop("model_attestation_targets")

    with pytest.raises(
        ValueError,
        match="Tinfoil require_model_attestations must be true for selected model routing",
    ):
        _require_tinfoil_model_attestation_claims(
            claims={},
            policy=policy,
            model_attestations=[],
            claim_key="backend_model_attestations",
        )


def test_tinfoil_runtime_validation_allows_missing_tls_only_for_explicit_ehbp_policy() -> (
    None
):
    policy = _tinfoil_runtime_policy(transport_security="ehbp")
    claims = _validate_tinfoil_verifier_result(
        verifier_result=_tinfoil_verifier_result(
            policy_digest=policy.digest,
            include_tls=False,
            model_include_tls=False,
        ),
        policy=policy,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verification_nonce="nonce",
        attestation_format="https://tinfoil.sh/predicate/sev-snp-guest/v2",
        attestation_report_digest=VALID_ATTESTATION_REPORT_DIGEST,
        ehbp_public_key_hex=VALID_EHBP_PUBLIC_KEY_HEX,
        ehbp_key_config_b64=VALID_EHBP_KEY_CONFIG_B64,
        ehbp_claims={},
        model_attestations=_tinfoil_model_evidence(),
    )

    assert claims["transport"] == "ehbp"
    assert "tls_public_key_fingerprint_sha256" not in claims
    model_claims = claims["model_attestations"]["tinfoil/gpt-secure"]
    assert "tls_public_key_fingerprint_sha256" not in model_claims


def test_tinfoil_runtime_validation_rejects_malformed_tls_even_for_ehbp_policy() -> (
    None
):
    policy = _tinfoil_runtime_policy(transport_security="ehbp")
    with pytest.raises(
        ValueError,
        match="TLS public key fingerprint claim must be a sha256 digest",
    ):
        _validate_tinfoil_verifier_result(
            verifier_result=_tinfoil_verifier_result(
                policy_digest=policy.digest,
                tls_value="not-a-digest",
            ),
            policy=policy,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verification_nonce="nonce",
            attestation_format="https://tinfoil.sh/predicate/sev-snp-guest/v2",
            attestation_report_digest=VALID_ATTESTATION_REPORT_DIGEST,
            ehbp_public_key_hex=VALID_EHBP_PUBLIC_KEY_HEX,
            ehbp_key_config_b64=VALID_EHBP_KEY_CONFIG_B64,
            ehbp_claims={},
            model_attestations=_tinfoil_model_evidence(),
        )


@pytest.mark.parametrize(
    ("target", "expected_issue"),
    [
        (
            {"host": "127.0.0.1"},
            "model_attestation_targets for tinfoil/kimi-k2-6: "
            "host must not use localhost or a non-global IP address",
        ),
        (
            {"attestation_url": "https://10.0.0.5/.well-known/tinfoil-attestation"},
            "model_attestation_targets for tinfoil/kimi-k2-6: "
            "attestation_url must not use localhost or a non-global IP address",
        ),
    ],
)
def test_tinfoil_runtime_policy_rejects_private_model_target_hosts(
    target: dict[str, object],
    expected_issue: str,
) -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="tinfoil",
        mode="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        model_ids=["tinfoil/kimi-k2-6"],
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": "sha256:" + ("a" * 64),
            "require_model_attestations": True,
            "model_attestation_targets": {
                "tinfoil/kimi-k2-6": {
                    **target,
                    "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                    "expected_release_digest": "sha256:" + ("b" * 64),
                }
            },
        },
    )

    violations = _tinfoil_model_attestation_policy_violations(policy)

    assert expected_issue in violations


def test_tinfoil_runtime_policy_requires_model_target_release_digest() -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="tinfoil",
        mode="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        model_ids=["tinfoil/kimi-k2-6"],
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": "sha256:" + ("a" * 64),
            "require_model_attestations": True,
            "model_attestation_targets": {
                "tinfoil/kimi-k2-6": {
                    "host": "kimi-k2-6.inf13.tinfoil.sh",
                    "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                    "expected_code_measurement_fingerprint": (
                        "sha256:" + ("b" * 64)
                    ),
                }
            },
        },
    )

    violations = _tinfoil_model_attestation_policy_violations(policy)

    assert (
        "model_attestation_targets for tinfoil/kimi-k2-6: "
        "expected_release_digest or allowed_release_digests is required"
    ) in violations


def test_tinfoil_runtime_policy_rejects_known_model_target_for_different_model() -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="tinfoil",
        mode="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        model_ids=["tinfoil/llama-3.3-70b"],
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": _digest("router-release"),
            "require_model_attestations": True,
            "model_attestation_targets": {
                "tinfoil/llama-3.3-70b": {
                    "host": "kimi-k2-6.inf13.tinfoil.sh",
                    "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                    "expected_release_digest": _digest("kimi-release"),
                }
            },
        },
    )

    violations = _tinfoil_model_attestation_policy_violations(policy)

    assert (
        "model_attestation_targets for tinfoil/llama-3.3-70b host "
        "kimi-k2-6.inf13.tinfoil.sh is a known Tinfoil target for a different model"
    ) in violations


def test_tinfoil_runtime_policy_rejects_known_model_target_repo_mismatch() -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="tinfoil",
        mode="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        model_ids=["tinfoil/kimi-k2-6"],
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": _digest("router-release"),
            "require_model_attestations": True,
            "model_attestation_targets": {
                "tinfoil/kimi-k2-6": {
                    "host": "kimi-k2-6.inf13.tinfoil.sh",
                    "repo": "tinfoilsh/confidential-kimi-k2-5",
                    "expected_release_digest": _digest("kimi-release"),
                }
            },
        },
    )

    violations = _tinfoil_model_attestation_policy_violations(policy)

    assert (
        "model_attestation_targets for tinfoil/kimi-k2-6 host "
        "kimi-k2-6.inf13.tinfoil.sh repo tinfoilsh/confidential-kimi-k2-5 "
        "does not match known Tinfoil target repos"
    ) in violations


def test_tinfoil_runtime_policy_rejects_case_variant_duplicate_model_targets() -> (
    None
):
    policy = ConfidentialVerifierPolicy(
        provider_type="tinfoil",
        mode="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        model_ids=["tinfoil/kimi-k2-6"],
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": "sha256:" + ("a" * 64),
            "require_model_attestations": True,
            "model_attestation_targets": {
                "tinfoil/kimi-k2-6": {
                    "host": "kimi-k2-6.inf13.tinfoil.sh",
                    "repo": "tinfoilsh/confidential-kimi-k2-6-b200",
                    "expected_release_digest": "sha256:" + ("b" * 64),
                },
                "TINFOIL/kimi-k2-6": {
                    "host": "other.inf13.tinfoil.sh",
                    "repo": "tinfoilsh/confidential-other",
                    "expected_release_digest": "sha256:" + ("c" * 64),
                },
            },
        },
    )

    violations = _tinfoil_model_attestation_policy_violations(policy)

    assert (
        "model_attestation_targets contains duplicate model target TINFOIL/kimi-k2-6"
    ) in violations


def test_privatemode_runtime_policy_rejects_scalar_expected_workload_selectors() -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="privatemode",
        mode="privatemode",
        base_url="http://127.0.0.1:8080/v1",
        model_ids=["privatemode/kimi-k2-6"],
        policy={
            "manifest_digest": "sha256:" + ("a" * 64),
            "proxy_image_digest": "sha256:" + ("b" * 64),
            "expected_workload_sans": "workload-kimi-k2-6",
            "model_workload_bindings": {
                "privatemode/kimi-k2-6": {
                    "workload_sans": ["workload-kimi-k2-6"],
                }
            },
        },
    )

    violations = _privatemode_model_workload_policy_violations(policy)

    assert "expected_workload_sans must be a list of non-empty strings" in violations


def test_privatemode_runtime_policy_rejects_case_variant_duplicate_workload_bindings() -> (
    None
):
    policy = ConfidentialVerifierPolicy(
        provider_type="privatemode",
        mode="privatemode",
        base_url="http://127.0.0.1:8080/v1",
        model_ids=["privatemode/kimi-k2-6"],
        policy={
            "manifest_digest": "sha256:" + ("a" * 64),
            "proxy_image_digest": "sha256:" + ("b" * 64),
            "expected_workload_sans": ["workload-kimi-k2-6"],
            "model_workload_bindings": {
                "privatemode/kimi-k2-6": {
                    "workload_sans": ["workload-kimi-k2-6"],
                },
                "PRIVATEMODE/kimi-k2-6": {
                    "workload_sans": ["other-workload"],
                },
            },
        },
    )

    violations = _privatemode_model_workload_policy_violations(policy)

    assert (
        "model_workload_bindings contains duplicate model binding "
        "PRIVATEMODE/kimi-k2-6"
    ) in violations


def test_privatemode_runtime_policy_requires_exact_model_ids() -> None:
    policy = ConfidentialVerifierPolicy(
        provider_type="privatemode",
        mode="privatemode",
        base_url="http://127.0.0.1:8080/v1",
        model_ids=[],
        policy={
            "manifest_digest": "sha256:" + ("a" * 64),
            "proxy_image_digest": "sha256:" + ("b" * 64),
            "expected_workload_sans": ["workload-kimi-k2-6"],
            "model_workload_bindings": {
                "privatemode/kimi-k2-6": {
                    "workload_sans": ["workload-kimi-k2-6"],
                },
            },
        },
    )

    violations = _privatemode_model_workload_policy_violations(policy)

    assert (
        "Privatemode model_ids are required with model_workload_bindings"
        in violations
    )


def test_privatemode_runtime_policy_rejects_placeholder_component_measurement() -> None:
    violations = _privatemode_component_policy_violations(
        {
            "expected_coordinator_measurement": "sha256:" + ("c" * 64),
        }
    )

    assert "coordinator_measurement policy values must not be placeholder digests" in (
        violations
    )


def test_privatemode_runtime_policy_rejects_gpu_policy_not_emitted_by_verifier() -> (
    None
):
    violations = _privatemode_component_policy_violations(
        {
            "expected_gpu_attestation_policy": "strict-ocsp",
        }
    )

    assert "expected_gpu_attestation_policy must be nvidia-ocsp-good-only" in (
        violations
    )


def test_privatemode_runtime_policy_requires_app_e2ee_trust_tier() -> None:
    violations = _privatemode_component_policy_violations(
        {
            "expected_gpu_attestation_policy": VALID_PRIVATEMODE_GPU_POLICY,
        }
    )

    assert "expected_trust_tier must be app-e2ee" in violations


def test_privatemode_verifier_result_rejects_reference_only_trust_tier() -> None:
    policy = _privatemode_runtime_policy()
    verifier_result = _privatemode_verifier_result(policy)
    claims = verifier_result["claims"]
    assert isinstance(claims, dict)
    claims["trust_tier"] = "reference-only"

    with pytest.raises(ValueError, match="trust_tier claim does not match policy"):
        _validate_privatemode_verifier_result(
            verifier_result=verifier_result,
            policy=policy,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verification_nonce="nonce",
            proxy_base_url="http://127.0.0.1:8080/v1",
        )


def test_privatemode_verifier_result_rejects_policy_without_trust_tier() -> None:
    policy = _privatemode_runtime_policy()
    policy.policy.pop("expected_trust_tier")
    verifier_result = _privatemode_verifier_result(policy)

    with pytest.raises(ValueError, match="expected_trust_tier must be app-e2ee"):
        _validate_privatemode_verifier_result(
            verifier_result=verifier_result,
            policy=policy,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verification_nonce="nonce",
            proxy_base_url="http://127.0.0.1:8080/v1",
        )


def test_privatemode_verifier_result_rejects_placeholder_required_proof_digest() -> (
    None
):
    policy = _privatemode_runtime_policy()
    verifier_result = _privatemode_verifier_result(policy)
    claims = verifier_result["claims"]
    assert isinstance(claims, dict)
    claims["coordinator_attestation_doc_digest"] = "sha256:" + ("6" * 64)

    with pytest.raises(
        ValueError,
        match="coordinator_attestation_doc_digest claim must not be a placeholder digest",
    ):
        _validate_privatemode_verifier_result(
            verifier_result=verifier_result,
            policy=policy,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verification_nonce="nonce",
            proxy_base_url="http://127.0.0.1:8080/v1",
        )


def test_confidentiality_policy_rejects_inline_secret_before_digesting() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"

    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": ["glm-5-1"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "headers": {"X-Provider-Credential": "sk-inline-secret"},
                },
            }
        }
    )

    assert provider.confidentiality_policy() is None
    status = provider.confidentiality_status()
    assert status.policy_digest is None


def test_confidentiality_policy_rejects_embedded_secret_argument_before_digesting() -> (
    None
):
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"

    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": ["glm-5-1"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "verifier_command": [
                        "/opt/routstr/bin/tinfoil-verifier",
                        "--token=sk-inline-secret",
                    ],
                },
            }
        }
    )

    assert provider.confidentiality_policy() is None
    policy_error = provider.confidentiality_policy_error() or ""
    assert "secret-like value must not be embedded" in policy_error
    assert "sk-inline-secret" not in policy_error
    status = provider.confidentiality_status()
    assert status.policy_digest is None


def test_confidentiality_policy_rejects_url_credentials_before_digesting() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"

    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": ["glm-5-1"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "attestation_url": (
                        "https://operator:secret-token@example.test/"
                        ".well-known/tinfoil-attestation"
                    ),
                },
            }
        }
    )

    assert provider.confidentiality_policy() is None
    assert "secret-token" not in (provider.confidentiality_policy_error() or "")
    status = provider.confidentiality_status()
    assert status.policy_digest is None


def test_confidentiality_policy_rejects_url_query_credentials_before_digesting() -> (
    None
):
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"

    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": ["glm-5-1"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "attestation_url": (
                        "https://example.test/.well-known/tinfoil-attestation"
                        "?api_key=secret-token"
                    ),
                },
            }
        }
    )

    assert provider.confidentiality_policy() is None
    assert "secret-token" not in (provider.confidentiality_policy_error() or "")
    assert "api_key must not be embedded" in (
        provider.confidentiality_policy_error() or ""
    )
    status = provider.confidentiality_status()
    assert status.policy_digest is None


def test_confidentiality_policy_rejects_non_json_policy_before_digesting() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"

    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": ["glm-5-1"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "non_json": {1, 2},
                },
            }
        }
    )

    assert provider.confidentiality_policy() is None
    assert "canonical JSON" in (provider.confidentiality_policy_error() or "")
    status = provider.confidentiality_status()
    assert status.policy_digest is None


def test_confidentiality_policy_digest_strips_base_url_userinfo() -> None:
    provider_with_credentials = BaseUpstreamProvider(
        base_url="https://operator:secret-token@example.test/v1",
        api_key="test",
    )
    provider_with_credentials.provider_type = "tinfoil"
    provider_without_credentials = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider_without_credentials.provider_type = "tinfoil"
    provider_settings = {
        "confidentiality": {
            "enabled": True,
            "mode": "tinfoil",
            "model_ids": ["glm-5-1"],
            "policy": {"repo": "tinfoilsh/confidential-model-router"},
        }
    }

    provider_with_credentials.configure_confidentiality_from_settings(provider_settings)
    provider_without_credentials.configure_confidentiality_from_settings(
        provider_settings
    )

    policy_with_credentials = provider_with_credentials.confidentiality_policy()
    policy_without_credentials = provider_without_credentials.confidentiality_policy()
    assert policy_with_credentials is not None
    assert policy_without_credentials is not None
    assert policy_with_credentials.digest == policy_without_credentials.digest
    assert "secret-token" not in policy_with_credentials.digest


def test_confidentiality_policy_rejects_malformed_model_selectors() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"

    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": [123, "glm-5-1"],
                "model_id_prefixes": [{"unexpected": "object"}, "private/"],
                "policy": {"repo": "tinfoilsh/confidential-model-router"},
            }
        }
    )

    policy = provider.confidentiality_policy()

    assert policy is None
    status = provider.confidentiality_status()
    assert status.enabled is False
    assert status.mode == "none"


def test_known_confidential_policy_rejects_prefix_selectors_from_settings() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"

    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_id_prefixes": ["tinfoil/"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                },
            }
        }
    )

    assert provider.confidentiality_policy() is None
    status = provider.confidentiality_status()
    assert status.enabled is False
    assert status.verified is False
    assert status.mode == "none"
    assert status.failure_reason == (
        "confidential verifier policy for tinfoil requires exact model_ids; "
        "model_id_prefixes are not supported"
    )


def test_static_confidentiality_status_rejects_malformed_model_selectors() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"

    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": [123, "glm-5-1"],
                "model_id_prefixes": [{"unexpected": "object"}, "private/"],
            }
        }
    )

    status = provider.confidentiality_status()

    assert status.enabled is False
    assert status.mode == "none"
    assert status.model_ids == []
    assert status.model_id_prefixes == []


def test_provider_settings_json_rejects_non_standard_constants_before_configuration() -> (
    None
):
    from routstr.upstream.helpers import _configure_provider_confidentiality

    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider_row = SimpleNamespace(
        provider_type="tinfoil",
        base_url="https://example.test/v1",
        provider_settings=(
            '{"confidentiality":{"enabled":true,"mode":"tinfoil",'
            '"model_ids":["glm-5-1"],'
            '"policy":{"repo":"tinfoilsh/confidential-model-router"},'
            '"ignored_non_standard":NaN}}'
        ),
    )

    _configure_provider_confidentiality(provider, provider_row)

    assert provider.confidentiality_policy() is None
    status = provider.confidentiality_status()
    assert status.enabled is False
    assert status.mode == "none"


def test_provider_instantiation_rejects_non_finite_provider_fee() -> None:
    from routstr.upstream.helpers import _instantiate_provider

    provider_row = SimpleNamespace(
        provider_type="custom",
        base_url="https://example.test/v1",
        api_key="test",
        provider_fee=float("nan"),
    )

    assert _instantiate_provider(provider_row) is None


@pytest.mark.parametrize(
    ("provider_type", "base_url"),
    [
        ("tinfoil", "https://attacker.example/v1"),
        ("ppq-private", "https://api.ppq.ai/v1"),
        ("privatemode", "https://api.privatemode.ai/v1"),
    ],
)
def test_provider_instantiation_rejects_invalid_confidential_provider_identity(
    provider_type: str,
    base_url: str,
) -> None:
    from routstr.upstream.helpers import _instantiate_provider

    provider_row = SimpleNamespace(
        provider_type=provider_type,
        base_url=base_url,
        api_key="test",
        provider_fee=1.0,
    )

    assert _instantiate_provider(provider_row) is None


def test_confidentiality_policy_rejects_non_string_mode() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"

    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": 123,
                "model_ids": ["glm-5-1"],
                "policy": {"repo": "tinfoilsh/confidential-model-router"},
            }
        }
    )

    assert provider.confidentiality_policy() is None
    status = provider.confidentiality_status()
    assert status.enabled is False
    assert status.mode == "none"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider_type", 123),
        ("mode", 123),
        ("base_url", 123),
        ("model_ids", [123]),
        ("model_id_prefixes", [123]),
    ],
)
def test_confidential_verifier_policy_rejects_coerced_selector_fields(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, object] = {
        "provider_type": "tinfoil",
        "mode": "tinfoil",
        "base_url": "https://example.test/v1",
        "policy": {"repo": "tinfoilsh/confidential-model-router"},
        "model_ids": ["glm-5-1"],
        "model_id_prefixes": ["private/"],
    }
    kwargs[field] = value

    with pytest.raises(ValidationError):
        ConfidentialVerifierPolicy(**kwargs)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("model_ids", ["glm-5-1", "GLM-5-1"], "model_ids must not contain duplicates"),
        ("model_ids", [" "], "model_ids must be a list of non-empty strings"),
        (
            "model_id_prefixes",
            ["private/", "PRIVATE/"],
            "model_id_prefixes must not contain duplicates",
        ),
        (
            "model_id_prefixes",
            [""],
            "model_id_prefixes must be a list of non-empty strings",
        ),
    ],
)
def test_confidential_verifier_policy_rejects_invalid_direct_selectors(
    field: str,
    value: object,
    expected: str,
) -> None:
    kwargs: dict[str, object] = {
        "provider_type": "tinfoil",
        "mode": "tinfoil",
        "base_url": "https://example.test/v1",
        "policy": {"repo": "tinfoilsh/confidential-model-router"},
        "model_ids": ["glm-5-1"],
        "model_id_prefixes": [],
    }
    kwargs[field] = value

    with pytest.raises(ValidationError) as exc_info:
        ConfidentialVerifierPolicy(**kwargs)

    assert expected in str(exc_info.value)


def test_confidential_verifier_policy_rejects_unsupported_direct_mode() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ConfidentialVerifierPolicy(
            provider_type="tinfoil",
            mode="custom-confidential-mode",
            base_url="https://example.test/v1",
            policy={"repo": "tinfoilsh/confidential-model-router"},
            model_ids=["tinfoil/gpt-secure"],
            model_id_prefixes=[],
        )

    assert (
        "confidentiality mode must be tinfoil, ppq-private-tee, or privatemode"
        in str(exc_info.value)
    )


def test_confidential_verifier_policy_rejects_secret_policy_on_direct_construction() -> (
    None
):
    with pytest.raises(ValidationError) as exc_info:
        ConfidentialVerifierPolicy(
            provider_type="tinfoil",
            mode="tinfoil",
            base_url="https://example.test/v1",
            policy={
                "repo": "tinfoilsh/confidential-model-router",
                "attestation_url": (
                    "https://operator:secret-token@example.test/"
                    ".well-known/tinfoil-attestation"
                ),
            },
            model_ids=["glm-5-1"],
            model_id_prefixes=[],
        )

    error_text = str(exc_info.value)
    assert "attestation_url must not include credentials" in error_text
    assert "secret-token" not in error_text


def test_runtime_status_requires_evidence_to_mark_verified() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )

    provider.set_confidentiality_status(
        provider.confidentiality_status().copy(
            update={
                "enabled": True,
                "verified": True,
                "mode": "tinfoil",
                "verified_at": 1_700_000_000,
                "expires_at": 4_102_444_800,
            }
        )
    )

    status = provider.confidentiality_status()
    assert status.enabled is True
    assert status.verified is False
    assert status.verified_at is None
    assert status.expires_at is None
    assert (
        status.failure_reason
        == "confidentiality verifier did not return required evidence"
    )


def test_known_confidential_runtime_status_rejects_prefix_selectors() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://inference.tinfoil.sh/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider._confidentiality_policy = ConfidentialVerifierPolicy(
        provider_type="tinfoil",
        mode="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        policy={
            "repo": "tinfoilsh/confidential-model-router",
            "expected_release_digest": VALID_RELEASE_DIGEST,
        },
        model_id_prefixes=["tinfoil/"],
    )

    policy = provider.confidentiality_policy()
    assert policy is not None
    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="tinfoil",
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            verifier="unit-test-verifier",
            policy_digest=policy.digest,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_claims={"repo": "tinfoilsh/confidential-model-router"},
            model_id_prefixes=["tinfoil/"],
        )
    )

    status = provider.confidentiality_status()

    assert status.enabled is True
    assert status.verified is False
    assert status.failure_reason == (
        "confidentiality verifier prefix selectors are not valid provider evidence"
    )
    assert status.verifier is None
    assert status.evidence_digest is None
    assert status.verified_claims == {}


def test_runtime_status_accepts_verified_evidence_metadata() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )

    provider.set_confidentiality_status(
        provider.confidentiality_status().copy(
            update={
                "enabled": True,
                "verified": True,
                "mode": "tinfoil",
                "verified_at": 1_700_000_000,
                "expires_at": 4_102_444_800,
                "verifier": "unit-test-verifier",
                "policy_digest": VALID_POLICY_DIGEST,
                "evidence_digest": VALID_EVIDENCE_DIGEST,
                "verified_claims": {"tee": "unit-test"},
            }
        )
    )

    status = provider.confidentiality_status()
    assert status.verified is True
    assert status.verifier == "unit-test-verifier"
    assert status.policy_digest == VALID_POLICY_DIGEST
    assert status.evidence_digest == VALID_EVIDENCE_DIGEST
    assert status.verified_claims == {"tee": "unit-test"}


def test_runtime_status_rejects_known_confidential_provider_without_policy() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"

    provider.set_confidentiality_status(
        provider.confidentiality_status().copy(
            update={
                "enabled": True,
                "verified": True,
                "mode": "tinfoil",
                "verified_at": 1_700_000_000,
                "expires_at": 4_102_444_800,
                "verifier": "unit-test-verifier",
                "policy_digest": VALID_POLICY_DIGEST,
                "evidence_digest": VALID_EVIDENCE_DIGEST,
                "verified_claims": {"tee": "unit-test"},
                "model_ids": ["tinfoil/gpt-secure"],
            }
        )
    )

    status = provider.confidentiality_status()
    assert status.verified is False
    assert status.verified_at is None
    assert status.expires_at is None
    assert status.verified_claims == {}
    assert status.failure_reason == (
        "confidentiality verifier policy is required for provider"
    )


def test_runtime_status_rejects_ppq_private_without_backend_model_attestations() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://api.ppq.ai/v1",
        api_key="test",
    )
    provider.provider_type = "ppq-private"
    provider._confidentiality_policy = ConfidentialVerifierPolicy(
        provider_type="ppq-private",
        mode="ppq-private-tee",
        base_url="https://api.ppq.ai/v1",
        model_ids=["private/gpt-oss-120b"],
        policy={
            "attestation_bundle_url": "https://api.ppq.ai/private/v1/attestation-bundle",
            "repo": "ppq-ai/private-tee",
            "expected_release_digest": VALID_RELEASE_DIGEST,
            "expected_code_measurement_fingerprint": VALID_CODE_MEASUREMENT,
            "proxy_binary_digest": _digest("ppq-proxy"),
            "proxy_binary_path": "/opt/ppq/ppq-private-mode-proxy",
            "require_model_attestations": True,
            "model_attestation_targets": {
                "private/gpt-oss-120b": {
                    "host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                    "repo": "tinfoilsh/confidential-gpt-oss-120b",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                    "expected_code_measurement_fingerprint": VALID_CODE_MEASUREMENT,
                }
            },
        },
    )
    policy = provider.confidentiality_policy()
    assert policy is not None

    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="ppq-private-tee",
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            verifier="unit-test-verifier",
            policy_digest=policy.digest,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            model_ids=["private/gpt-oss-120b"],
            verified_claims={
                "payload_policy_digest": policy.digest,
                "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
                "transport": "ehbp",
                "repo": "ppq-ai/private-tee",
                "attestation_bundle_url": "https://api.ppq.ai/private/v1/attestation-bundle",
                "selected_model_ids": ["private/gpt-oss-120b"],
            },
        )
    )

    status = provider.confidentiality_status()

    assert status.verified is False
    assert status.verified_at is None
    assert status.expires_at is None
    assert status.verified_claims == {}
    assert status.failure_reason == (
        "confidentiality verifier proof claims do not satisfy provider policy"
    )


def test_runtime_status_rejects_ppq_private_backend_attestation_policy_mismatch() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://api.ppq.ai/v1",
        api_key="test",
    )
    provider.provider_type = "ppq-private"
    attestation_bundle_url = "https://api.ppq.ai/private/v1/attestation-bundle"
    provider._confidentiality_policy = ConfidentialVerifierPolicy(
        provider_type="ppq-private",
        mode="ppq-private-tee",
        base_url="https://api.ppq.ai/v1",
        model_ids=["private/gpt-oss-120b"],
        policy={
            "attestation_bundle_url": attestation_bundle_url,
            "attestation_bundle_url_digest": _sha256_json_digest(
                attestation_bundle_url
            ),
            "repo": "ppq-ai/private-tee",
            "expected_release_digest": VALID_RELEASE_DIGEST,
            "expected_code_measurement_fingerprint": VALID_CODE_MEASUREMENT,
            "expected_enclave_measurement_fingerprint": VALID_ENCLAVE_MEASUREMENT,
            "proxy_binary_digest": _digest("ppq-proxy"),
            "proxy_binary_path": "/opt/ppq/ppq-private-mode-proxy",
            "require_model_attestations": True,
            "model_attestation_targets": {
                "private/gpt-oss-120b": {
                    "host": "gpt-oss-120b-1.inf10.tinfoil.sh",
                    "repo": "tinfoilsh/confidential-gpt-oss-120b",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                    "expected_code_measurement_fingerprint": VALID_CODE_MEASUREMENT,
                    "expected_enclave_measurement_fingerprint": (
                        VALID_ENCLAVE_MEASUREMENT
                    ),
                }
            },
        },
    )
    policy = provider.confidentiality_policy()
    assert policy is not None

    provider.set_confidentiality_status(
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="ppq-private-tee",
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            verifier="unit-test-verifier",
            policy_digest=policy.digest,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            model_ids=["private/gpt-oss-120b"],
            verified_claims={
                "payload_policy_digest": policy.digest,
                "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
                "transport": "ehbp",
                "repo": "ppq-ai/private-tee",
                "attestation_bundle_url": attestation_bundle_url,
                "attestation_bundle_url_digest": _sha256_json_digest(
                    attestation_bundle_url
                ),
                "client_encryption_boundary": "routstr-tee-ehbp-proxy",
                "attested_hpke_public_key_hex": VALID_EHBP_PUBLIC_KEY_HEX,
                "enclave_measurement_fingerprint": VALID_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": VALID_CODE_MEASUREMENT,
                "release_digest": VALID_RELEASE_DIGEST,
                "tls_public_key_fingerprint_sha256": VALID_TLS_DIGEST,
                "proxy_binary_digest": _digest("ppq-proxy"),
                "selected_model_ids": ["private/gpt-oss-120b"],
                "verification_steps": _ehbp_verification_steps(),
                "backend_model_attestations": {
                    "private/gpt-oss-120b": {
                        "repo": "tinfoilsh/confidential-wrong-model",
                        "attestation_format": (
                            "https://tinfoil.sh/predicate/sev-snp-guest/v2"
                        ),
                        "attestation_report_digest": (
                            VALID_ATTESTATION_REPORT_DIGEST
                        ),
                        "attested_hpke_public_key_hex": VALID_EHBP_PUBLIC_KEY_HEX,
                        "enclave_measurement_fingerprint": (
                            VALID_ENCLAVE_MEASUREMENT
                        ),
                        "code_measurement_fingerprint": VALID_CODE_MEASUREMENT,
                        "release_digest": VALID_RELEASE_DIGEST,
                        "tls_public_key_fingerprint_sha256": VALID_TLS_DIGEST,
                        "verification_steps": _ehbp_verification_steps(),
                    }
                },
            },
        )
    )

    status = provider.confidentiality_status()

    assert status.verified is False
    assert status.verified_at is None
    assert status.expires_at is None
    assert status.verified_claims == {}
    assert status.failure_reason == (
        "confidentiality verifier proof claims do not satisfy provider policy"
    )


def test_runtime_status_rejects_incomplete_confidential_policy_object() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.confidentiality_policy = lambda: SimpleNamespace(
        digest=VALID_POLICY_DIGEST
    )

    provider.set_confidentiality_status(
        provider.confidentiality_status().copy(
            update={
                "enabled": True,
                "verified": True,
                "mode": "tinfoil",
                "verified_at": 1_700_000_000,
                "expires_at": 4_102_444_800,
                "verifier": "unit-test-verifier",
                "policy_digest": VALID_POLICY_DIGEST,
                "evidence_digest": VALID_EVIDENCE_DIGEST,
                "verified_claims": {"tee": "unit-test"},
                "model_ids": ["tinfoil/gpt-secure"],
            }
        )
    )

    status = provider.confidentiality_status()
    assert status.verified is False
    assert status.verified_at is None
    assert status.expires_at is None
    assert status.verified_claims == {}
    assert status.failure_reason == (
        "confidentiality verifier policy is incomplete"
    )


def test_runtime_status_rejects_known_provider_without_exact_model_ids() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.confidentiality_policy = lambda: SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="tinfoil",
        model_ids=[],
        model_id_prefixes=[],
    )

    provider.set_confidentiality_status(
        provider.confidentiality_status().copy(
            update={
                "enabled": True,
                "verified": True,
                "mode": "tinfoil",
                "verified_at": 1_700_000_000,
                "expires_at": 4_102_444_800,
                "verifier": "unit-test-verifier",
                "policy_digest": VALID_POLICY_DIGEST,
                "evidence_digest": VALID_EVIDENCE_DIGEST,
                "verified_claims": {"tee": "unit-test"},
                "model_ids": [],
                "model_id_prefixes": [],
            }
        )
    )

    status = provider.confidentiality_status()
    assert status.verified is False
    assert status.verified_at is None
    assert status.expires_at is None
    assert status.verified_claims == {}
    assert status.failure_reason == (
        "confidentiality verifier exact model selectors are required for provider"
    )


def test_runtime_status_rejects_policy_mode_mismatched_to_provider_type() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.confidentiality_policy = lambda: SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        mode="privatemode",
        model_ids=["privatemode/gpt-secure"],
        model_id_prefixes=[],
    )

    provider.set_confidentiality_status(
        provider.confidentiality_status().copy(
            update={
                "enabled": True,
                "verified": True,
                "mode": "privatemode",
                "verified_at": 1_700_000_000,
                "expires_at": 4_102_444_800,
                "verifier": "unit-test-verifier",
                "policy_digest": VALID_POLICY_DIGEST,
                "evidence_digest": VALID_EVIDENCE_DIGEST,
                "verified_claims": {"tee": "unit-test"},
                "model_ids": ["privatemode/gpt-secure"],
            }
        )
    )

    status = provider.confidentiality_status()
    assert status.verified is False
    assert status.failure_reason == (
        "confidentiality verifier mode does not match provider_type"
    )


def test_runtime_status_rejects_policy_provider_type_mismatched_to_provider() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.confidentiality_policy = lambda: SimpleNamespace(
        digest=VALID_POLICY_DIGEST,
        provider_type="ppq-private",
        mode="tinfoil",
        model_ids=["tinfoil/gpt-secure"],
        model_id_prefixes=[],
    )

    provider.set_confidentiality_status(
        provider.confidentiality_status().copy(
            update={
                "enabled": True,
                "verified": True,
                "mode": "tinfoil",
                "verified_at": 1_700_000_000,
                "expires_at": 4_102_444_800,
                "verifier": "unit-test-verifier",
                "policy_digest": VALID_POLICY_DIGEST,
                "evidence_digest": VALID_EVIDENCE_DIGEST,
                "verified_claims": {"tee": "unit-test"},
                "model_ids": ["tinfoil/gpt-secure"],
            }
        )
    )

    status = provider.confidentiality_status()
    assert status.verified is False
    assert status.failure_reason == (
        "confidentiality verifier policy provider_type does not match provider"
    )


def test_runtime_status_rejects_verified_selectors_broader_than_policy() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.provider_type = "tinfoil"
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": ["tinfoil/gpt-secure"],
                "policy": {
                    "repo": "tinfoilsh/confidential-model-router",
                    "expected_release_digest": VALID_RELEASE_DIGEST,
                },
            }
        }
    )
    policy = provider.confidentiality_policy()
    assert policy is not None

    provider.set_confidentiality_status(
        provider.confidentiality_status().copy(
            update={
                "enabled": True,
                "verified": True,
                "mode": "tinfoil",
                "verified_at": 1_700_000_000,
                "expires_at": 4_102_444_800,
                "verifier": "unit-test-verifier",
                "policy_digest": policy.digest,
                "evidence_digest": VALID_EVIDENCE_DIGEST,
                "verified_claims": {"tee": "unit-test"},
                "model_ids": ["tinfoil/gpt-secure", "tinfoil/extra-model"],
            }
        )
    )

    status = provider.confidentiality_status()
    assert status.verified is False
    assert status.verified_at is None
    assert status.expires_at is None
    assert status.verified_claims == {}
    assert status.failure_reason == (
        "confidentiality verifier model selectors do not match policy"
    )


def test_runtime_status_rejects_truthy_string_verified() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )

    provider.set_confidentiality_status(
        provider.confidentiality_status().copy(
            update={
                "enabled": True,
                "verified": "false",
                "mode": "tinfoil",
                "verified_at": 1_700_000_000,
                "expires_at": 4_102_444_800,
                "verifier": "unit-test-verifier",
                "policy_digest": VALID_POLICY_DIGEST,
                "evidence_digest": VALID_EVIDENCE_DIGEST,
                "verified_claims": {"tee": "unit-test"},
            }
        )
    )

    status = provider.confidentiality_status()
    assert status.verified is False
    assert status.verified_at is None
    assert status.expires_at is None
    assert status.verified_claims == {}
    assert status.failure_reason == "confidentiality verifier verified must be true"


def test_confidentiality_status_rejects_string_proof_booleans() -> None:
    with pytest.raises(ValidationError):
        ConfidentialityStatus(
            enabled="true",
            verified="true",
            mode="tinfoil",
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            verifier="unit-test-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_claims={"tee": "unit-test"},
            model_ids=["gpt-secure"],
        )


def test_confidentiality_status_rejects_string_freshness_fields() -> None:
    with pytest.raises(ValidationError):
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="tinfoil",
            verified_at="1700000000",
            expires_at="4102444800",
            verifier="unit-test-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_claims={"tee": "unit-test"},
            model_ids=["gpt-secure"],
        )


def test_confidentiality_status_rejects_non_string_model_selectors() -> None:
    with pytest.raises(ValidationError):
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="tinfoil",
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            verifier="unit-test-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_claims={"tee": "unit-test"},
            model_ids=[123],
            model_id_prefixes=["private/"],
        )

    with pytest.raises(ValidationError):
        ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="tinfoil",
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            verifier="unit-test-verifier",
            policy_digest=VALID_POLICY_DIGEST,
            evidence_digest=VALID_EVIDENCE_DIGEST,
            verified_claims={"tee": "unit-test"},
            model_ids=["gpt-secure"],
            model_id_prefixes=[123],
        )


def test_runtime_status_rejects_copied_string_expiry() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="tinfoil",
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_claims={"tee": "unit-test"},
        model_ids=["gpt-secure"],
    ).copy(update={"expires_at": "4102444800"})

    provider.set_confidentiality_status(status)

    downgraded = provider.confidentiality_status()
    assert downgraded.verified is False
    assert downgraded.verified_at is None
    assert downgraded.expires_at is None
    assert downgraded.verified_claims == {}
    assert downgraded.failure_reason == (
        "confidentiality verifier expires_at must be an integer"
    )


def test_runtime_status_rejects_future_verified_at() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    now = int(time.time())
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="tinfoil",
        verified_at=now + 60,
        expires_at=now + 300,
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_claims={"tee": "unit-test"},
        model_ids=["gpt-secure"],
    )

    provider.set_confidentiality_status(status)

    downgraded = provider.confidentiality_status()
    assert downgraded.verified is False
    assert downgraded.verified_at is None
    assert downgraded.expires_at is None
    assert downgraded.verified_claims == {}
    assert downgraded.failure_reason == (
        "confidentiality verifier verified_at is in the future"
    )


def test_runtime_status_rejects_non_canonical_verified_claims() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="tinfoil",
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_claims={"tee": "unit-test"},
        model_ids=["gpt-secure"],
    ).copy(update={"verified_claims": {"bad": {1, 2}}})

    provider.set_confidentiality_status(status)

    downgraded = provider.confidentiality_status()
    assert downgraded.verified is False
    assert downgraded.verified_at is None
    assert downgraded.expires_at is None
    assert downgraded.evidence_digest is None
    assert downgraded.verified_claims == {}
    assert downgraded.failure_reason == (
        "confidentiality verifier did not return required evidence"
    )


def test_runtime_status_rejects_copied_non_string_verifier() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="tinfoil",
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_claims={"tee": "unit-test"},
        model_ids=["gpt-secure"],
    ).copy(update={"verifier": 123})

    provider.set_confidentiality_status(status)

    downgraded = provider.confidentiality_status()
    assert downgraded.verified is False
    assert downgraded.verifier is None
    assert downgraded.evidence_digest is None
    assert downgraded.verified_claims == {}
    assert downgraded.failure_reason == (
        "confidentiality verifier did not return required evidence"
    )


def test_runtime_status_rejects_copied_non_dict_verified_claims() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="tinfoil",
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        verifier="unit-test-verifier",
        policy_digest=VALID_POLICY_DIGEST,
        evidence_digest=VALID_EVIDENCE_DIGEST,
        verified_claims={"tee": "unit-test"},
        model_ids=["gpt-secure"],
    ).copy(update={"verified_claims": ["not", "claims"]})

    provider.set_confidentiality_status(status)

    downgraded = provider.confidentiality_status()
    assert downgraded.verified is False
    assert downgraded.verifier is None
    assert downgraded.evidence_digest is None
    assert downgraded.verified_claims == {}
    assert downgraded.failure_reason == (
        "confidentiality verifier did not return required evidence"
    )


def test_runtime_status_rejects_placeholder_evidence_digests() -> None:
    status = ConfidentialityStatus(
        enabled=True,
        verified=True,
        mode="tinfoil",
        verified_at=1_700_000_000,
        expires_at=4_102_444_800,
        verifier="unit-test-verifier",
        policy_digest="sha256:policy",
        evidence_digest="sha256:evidence",
        verified_claims={"tee": "unit-test"},
        model_ids=["gpt-secure"],
    )

    assert status.has_required_verification_evidence() is False

    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )
    provider.set_confidentiality_status(status)

    downgraded = provider.confidentiality_status()
    assert downgraded.verified is False
    assert downgraded.verified_at is None
    assert downgraded.expires_at is None
    assert downgraded.evidence_digest is None
    assert downgraded.verified_claims == {}
    assert (
        downgraded.failure_reason
        == "confidentiality verifier did not return required evidence"
    )


def test_runtime_status_requires_future_expiry_to_mark_verified() -> None:
    provider = BaseUpstreamProvider(
        base_url="https://example.test/v1",
        api_key="test",
    )

    provider.set_confidentiality_status(
        provider.confidentiality_status().copy(
            update={
                "enabled": True,
                "verified": True,
                "mode": "tinfoil",
                "verified_at": 1_700_000_000,
                "expires_at": None,
                "verifier": "unit-test-verifier",
                "policy_digest": VALID_POLICY_DIGEST,
                "evidence_digest": VALID_EVIDENCE_DIGEST,
                "verified_claims": {"tee": "unit-test"},
            }
        )
    )

    status = provider.confidentiality_status()
    assert status.verified is False
    assert status.verified_at is None
    assert status.expires_at is None
    assert status.verified_claims == {}
    assert status.failure_reason == (
        "confidentiality verifier did not return a future expires_at"
    )
