from __future__ import annotations

import hashlib
import json
import time

import pytest

from routstr.nostr import listing


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


VALID_POLICY_DIGEST = _digest("valid-policy")
VALID_EVIDENCE_DIGEST = _digest("valid-evidence")
VALID_ROUTSTR_PUBLIC_KEY_DIGEST = _digest("valid-routstr-public-key")
VALID_ROUTSTR_TEE_CLAIMS_DIGEST = _digest("valid-routstr-tee-claims")
VALID_ROUTSTR_TEE_EVIDENCE_DIGEST = _digest("valid-routstr-tee-evidence")
VALID_ROUTSTR_HPKE_KEY_CONFIG_DIGEST = _digest("valid-routstr-hpke-key-config")
VALID_ROUTSTR_HPKE_PUBLIC_KEY_DIGEST = _digest("valid-routstr-hpke-public-key")


def _sha256_json_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _tinfoil_public_proof_claims() -> dict[str, object]:
    return {
        "transport": "ehbp",
        "repo": "tinfoilsh/confidential-model-router",
        "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
        "attestation_report_digest": _digest("placeholder-3"),
        "attested_hpke_public_key_hex": "b" * 64,
        "code_measurement_fingerprint": _digest("placeholder-2"),
        "enclave_measurement_fingerprint": _digest("placeholder-1"),
        "payload_policy_digest": VALID_POLICY_DIGEST,
        "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
        "release_digest": _digest("placeholder-4"),
        "tls_public_key_fingerprint_sha256": _digest("placeholder-5"),
        "verification_steps": {
            "attested_transport_key_binding": True,
            "code_transparency": True,
            "freshness": True,
            "hardware_attestation_report": True,
            "hardware_certificate_chain": True,
            "measurement_match": True,
        },
        "model_attestations": {
            "gpt-secure": {
                "repo": "tinfoilsh/confidential-gpt-secure",
                "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
                "attestation_report_digest": _digest("placeholder-6"),
                "attested_hpke_public_key_hex": "c" * 64,
                "code_measurement_fingerprint": _digest("placeholder-7"),
                "enclave_measurement_fingerprint": _digest("placeholder-8"),
                "release_digest": _digest("placeholder-9"),
                "tls_public_key_fingerprint_sha256": _digest("placeholder-a"),
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


def _routstr_tee_status(*, ready: bool = True) -> dict[str, object]:
    return {
        "required": True,
        "ready": ready,
        "attestation_evidence_digest": VALID_ROUTSTR_TEE_EVIDENCE_DIGEST,
        "hpke_key_config_digest": VALID_ROUTSTR_HPKE_KEY_CONFIG_DIGEST,
        "hpke_public_key_digest": VALID_ROUTSTR_HPKE_PUBLIC_KEY_DIGEST,
        "client_confidentiality": {
            "mode": "attested-tls-termination",
            "tls_terminates_in_attested_tee": True,
            "inbound_ehbp_ohttp_request_decryption": False,
            "attested_tls_public_key_digest": VALID_ROUTSTR_PUBLIC_KEY_DIGEST,
        },
        "local_verification": {
            "verified": ready,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "evidence_digest": VALID_ROUTSTR_TEE_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_ROUTSTR_TEE_CLAIMS_DIGEST,
            "proof_claims": {
                "hpke_key_config_digest": VALID_ROUTSTR_HPKE_KEY_CONFIG_DIGEST,
                "hpke_public_key_digest": VALID_ROUTSTR_HPKE_PUBLIC_KEY_DIGEST,
                "public_key_digest": VALID_ROUTSTR_PUBLIC_KEY_DIGEST,
            },
        },
    }


def _ppq_private_public_proof_claims() -> dict[str, object]:
    return {
        "transport": "ehbp",
        "repo": "ppq-ai/private-tee",
        "attestation_bundle_url_digest": _digest("placeholder-5"),
        "attested_hpke_public_key_hex": "b" * 64,
        "code_measurement_fingerprint": _digest("placeholder-2"),
        "enclave_measurement_fingerprint": _digest("placeholder-1"),
        "payload_policy_digest": VALID_POLICY_DIGEST,
        "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
        "selected_model_ids": ["private/gpt-oss-120b"],
        "tls_public_key_fingerprint_sha256": _digest("placeholder-4"),
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
    expected_workload_identity_digest = _sha256_json_digest(
        {
            "ids": [],
            "sans": ["backend.privatemode.example"],
        }
    )
    model_workload_binding_digest = _sha256_json_digest(
        {
            "privatemode/gpt-oss-120b": {
                "workload_ids": [],
                "workload_sans": ["backend.privatemode.example"],
            }
        }
    )
    claims = {
        "transport": "privatemode-proxy",
        "manifest_digest": _digest("placeholder-1"),
        "manifest_log_manifest_digest": _digest("placeholder-1"),
        "proxy_binary_digest": _digest("placeholder-6"),
        "payload_policy_digest": VALID_POLICY_DIGEST,
        "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
        "coordinator_measurement": _digest("placeholder-3"),
        "secret_service_measurement": _digest("placeholder-4"),
        "ai_worker_measurement": _digest("placeholder-5"),
        "attested_workload_identity_digest": _digest("placeholder-f"),
        "attested_workload_policy_digest": _digest("placeholder-5"),
        "expected_workload_identity_digest": expected_workload_identity_digest,
        "model_workload_binding_digest": model_workload_binding_digest,
        "selected_model_ids": ["privatemode/gpt-oss-120b"],
        "gpu_attestation_policy": "nvidia-ocsp-good-only",
        "coordinator_attestation_doc_digest": _digest("placeholder-7"),
        "mesh_ca_digest": _digest("placeholder-8"),
        "secret_service_certificate_digest": _digest("placeholder-9"),
        "ai_worker_manifest_digest": _digest("placeholder-a"),
        "nvidia_ocsp_policy_header_digest": _digest("placeholder-b"),
        "nvidia_ocsp_policy_mac_digest": _digest("placeholder-c"),
        "prompt_encryption_ciphertext_digest": _digest("placeholder-d"),
        "inference_secret_id_digest": _digest("placeholder-e"),
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


def test_confidentiality_listing_metadata_omits_secrets_and_diagnostics() -> None:
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "tinfoil-prod",
                    "base_url": "https://inference.tinfoil.sh/v1",
                    "api_key": "SECRET",
                    "supported_endpoints": [
                        "/v1/chat/completions",
                        "/v1/audio",
                        "/v1/embeddings",
                        "/v1/responses",
                    ],
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": _digest("placeholder-4"),
                        "require_model_attestations": True,
                        "attestation_url": "https://router.tinfoil.example/attest",
                        "model_attestation_targets": {
                            "gpt-secure": {
                                "host": "gpt-secure.tinfoil.example",
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "expected_release_digest": _digest("placeholder-9"),
                                "hpke_keys_url": "https://gpt-secure.tinfoil.example/.well-known/hpke-keys",
                            }
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                        "failure_reason": "debug details",
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                        "proof_claims": {
                            "transport": "ehbp",
                            "repo": "tinfoilsh/confidential-model-router",
                            "attested_hpke_public_key_hex": "b" * 64,
                            "enclave_measurement_fingerprint": _digest("placeholder-1"),
                            "code_measurement_fingerprint": _digest("placeholder-2"),
                            "payload_policy_digest": VALID_POLICY_DIGEST,
                            "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
                            "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
                            "attestation_report_digest": _digest("placeholder-3"),
                            "release_digest": _digest("placeholder-4"),
                            "tls_public_key_fingerprint_sha256": _digest(
                                "placeholder-5"
                            ),
                            "ehbp_key_config_b64": "SECRET_KEY_CONFIG",
                            "verification_steps": {
                                "hardware_attestation_report": True,
                                "hardware_certificate_chain": True,
                                "code_transparency": True,
                                "measurement_match": True,
                                "attested_transport_key_binding": True,
                                "freshness": True,
                            },
                            "model_attestations": {
                                "gpt-secure": {
                                    "repo": "tinfoilsh/confidential-gpt-secure",
                                    "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
                                    "attestation_report_digest": _digest(
                                        "placeholder-6"
                                    ),
                                    "attested_hpke_public_key_hex": "c" * 64,
                                    "code_measurement_fingerprint": _digest(
                                        "placeholder-7"
                                    ),
                                    "enclave_measurement_fingerprint": _digest(
                                        "placeholder-8"
                                    ),
                                    "release_digest": _digest("placeholder-9"),
                                    "tls_public_key_fingerprint_sha256": _digest(
                                        "placeholder-a"
                                    ),
                                    "verification_steps": {
                                        "hardware_attestation_report": True,
                                        "hardware_certificate_chain": True,
                                        "code_transparency": True,
                                        "measurement_match": True,
                                        "attested_transport_key_binding": True,
                                        "freshness": True,
                                    },
                                }
                            },
                        },
                    },
                },
                {
                    "provider_type": "openai",
                    "upstream_name": "plain",
                    "confidentiality": {"enabled": False},
                },
            ],
        }
    )

    assert metadata == {
        "mode": "required",
        "required": True,
        "end_to_end_ready": False,
        "providers": [
            {
                "provider_type": "tinfoil",
                "upstream_name": "tinfoil-prod",
                "supported_endpoints": [
                    "/v1/chat/completions",
                    "/v1/audio",
                    "/v1/embeddings",
                    "/v1/responses",
                ],
                "confidentiality_policy": {
                    "expected_release_digest": _digest("placeholder-4"),
                    "model_attestation_targets": {
                        "gpt-secure": {
                            "expected_release_digest": _digest("placeholder-9"),
                            "host": "gpt-secure.tinfoil.example",
                            "hpke_keys_url": "https://gpt-secure.tinfoil.example/.well-known/hpke-keys",
                            "repo": "tinfoilsh/confidential-gpt-secure",
                        }
                    },
                    "repo": "tinfoilsh/confidential-model-router",
                    "require_model_attestations": True,
                },
                "confidentiality": {
                    "enabled": True,
                    "verified": True,
                    "mode": "tinfoil",
                    "verifier": "unit-test-verifier",
                    "policy_digest": VALID_POLICY_DIGEST,
                    "evidence_digest": VALID_EVIDENCE_DIGEST,
                    "verified_at": 1_700_000_000,
                    "expires_at": 1_800_000_000,
                    "model_ids": ["gpt-secure"],
                    "model_id_prefixes": [],
                    "proof_claims": {
                        "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
                        "attestation_report_digest": _digest("placeholder-3"),
                        "attested_hpke_public_key_hex": "b" * 64,
                        "code_measurement_fingerprint": _digest("placeholder-2"),
                        "enclave_measurement_fingerprint": _digest("placeholder-1"),
                        "payload_policy_digest": VALID_POLICY_DIGEST,
                        "payload_evidence_digest": VALID_EVIDENCE_DIGEST,
                        "release_digest": _digest("placeholder-4"),
                        "repo": "tinfoilsh/confidential-model-router",
                        "tls_public_key_fingerprint_sha256": _digest("placeholder-5"),
                        "transport": "ehbp",
                        "model_attestations": {
                            "gpt-secure": {
                                "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
                                "attestation_report_digest": _digest("placeholder-6"),
                                "attested_hpke_public_key_hex": "c" * 64,
                                "code_measurement_fingerprint": _digest(
                                    "placeholder-7"
                                ),
                                "enclave_measurement_fingerprint": _digest(
                                    "placeholder-8"
                                ),
                                "release_digest": _digest("placeholder-9"),
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "tls_public_key_fingerprint_sha256": _digest(
                                    "placeholder-a"
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
                        },
                        "verification_steps": {
                            "attested_transport_key_binding": True,
                            "code_transparency": True,
                            "freshness": True,
                            "hardware_attestation_report": True,
                            "hardware_certificate_chain": True,
                            "measurement_match": True,
                        },
                    },
                },
            }
        ],
    }
    assert "SECRET" not in str(metadata)
    assert "SECRET_KEY_CONFIG" not in str(metadata)
    assert "attestation_url" not in str(metadata)
    assert "failure_reason" not in str(metadata)
    assert "base_url" not in str(metadata)


def test_confidentiality_listing_metadata_includes_local_routstr_tee_boundary() -> None:
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "routstr_tee": {
                "required": True,
                "ready": True,
                "client_confidentiality": {
                    "mode": "attested-tls-termination",
                    "tls_terminates_in_attested_tee": True,
                    "inbound_ehbp_ohttp_request_decryption": False,
                    "attested_tls_public_key_digest": VALID_ROUTSTR_PUBLIC_KEY_DIGEST,
                },
                "failure_reason_digest": _digest("placeholder-1"),
                "local_verification": {
                    "verified": True,
                    "verified_claims": {"api_key": "SECRET_LOCAL_TEE"},
                },
            },
            "providers": [],
        }
    )

    assert metadata == {
        "mode": "required",
        "required": True,
        "end_to_end_ready": False,
        "routstr_tee": {
            "required": True,
            "ready": False,
            "client_confidentiality": {
                "mode": "attested-tls-termination",
                "tls_terminates_in_attested_tee": True,
                "inbound_ehbp_ohttp_request_decryption": False,
                "attested_tls_public_key_digest": VALID_ROUTSTR_PUBLIC_KEY_DIGEST,
            },
        },
        "providers": [],
    }
    assert "SECRET_LOCAL_TEE" not in str(metadata)


def test_confidentiality_listing_metadata_requires_local_tee_for_end_to_end_ready() -> (
    None
):
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "tinfoil-prod",
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": _digest("placeholder-4"),
                        "require_model_attestations": True,
                        "model_attestation_targets": {
                            "gpt-secure": {
                                "host": "gpt-secure.tinfoil.example",
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "expected_release_digest": _digest("placeholder-9"),
                            }
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                        "proof_claims": _tinfoil_public_proof_claims(),
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["providers"][0]["confidentiality"]["verified"] is True
    assert metadata["end_to_end_ready"] is False


def test_confidentiality_listing_metadata_requires_routstr_tee_for_end_to_end_ready() -> (
    None
):
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "routstr_tee": {
                "required": True,
                "ready": False,
                "client_confidentiality": {
                    "mode": "attested-tls-termination",
                    "tls_terminates_in_attested_tee": True,
                    "inbound_ehbp_ohttp_request_decryption": False,
                },
            },
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "tinfoil-prod",
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": _digest("placeholder-4"),
                        "require_model_attestations": True,
                        "model_attestation_targets": {
                            "gpt-secure": {
                                "host": "gpt-secure.tinfoil.example",
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "expected_release_digest": _digest("placeholder-9"),
                            }
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                        "proof_claims": _tinfoil_public_proof_claims(),
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["providers"][0]["confidentiality"]["verified"] is True
    assert metadata["routstr_tee"]["ready"] is False
    assert metadata["end_to_end_ready"] is False


def test_confidentiality_listing_metadata_requires_routstr_tee_boundary_for_end_to_end_ready() -> (
    None
):
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "routstr_tee": {
                "required": True,
                "ready": True,
            },
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": _digest("placeholder-4"),
                        "require_model_attestations": True,
                        "model_attestation_targets": {
                            "gpt-secure": {
                                "host": "gpt-secure.tinfoil.example",
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "expected_release_digest": _digest("placeholder-9"),
                            }
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                        "proof_claims": _tinfoil_public_proof_claims(),
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["providers"][0]["confidentiality"]["verified"] is True
    assert metadata["routstr_tee"]["ready"] is False
    assert "client_confidentiality" not in metadata["routstr_tee"]
    assert metadata["end_to_end_ready"] is False


def test_confidentiality_listing_metadata_marks_end_to_end_ready_with_local_tee() -> (
    None
):
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "routstr_tee": _routstr_tee_status(),
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": _digest("placeholder-4"),
                        "require_model_attestations": True,
                        "model_attestation_targets": {
                            "gpt-secure": {
                                "host": "gpt-secure.tinfoil.example",
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "expected_release_digest": _digest("placeholder-9"),
                            }
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                        "proof_claims": _tinfoil_public_proof_claims(),
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["providers"][0]["confidentiality"]["verified"] is True
    assert metadata["end_to_end_ready"] is True


def test_confidentiality_listing_metadata_exposes_exact_routable_full_attestation_models() -> (
    None
):
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "routstr_tee": _routstr_tee_status(),
            "routable_with_full_attestation": {
                "tinfoil": ["gpt-secure"],
                "ppq-private": [],
                "privatemode": [],
            },
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": _digest("placeholder-4"),
                        "require_model_attestations": True,
                        "model_attestation_targets": {
                            "gpt-secure": {
                                "host": "gpt-secure.tinfoil.example",
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "expected_release_digest": _digest("placeholder-9"),
                            }
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                        "proof_claims": _tinfoil_public_proof_claims(),
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["routable_with_full_attestation"] == {
        "tinfoil": ["gpt-secure"],
        "ppq-private": [],
        "privatemode": [],
    }
    assert metadata["end_to_end_ready"] is True


def test_confidentiality_listing_metadata_keeps_routable_full_attestation_empty_without_tee() -> (
    None
):
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "routstr_tee": _routstr_tee_status(ready=False),
            "routable_with_full_attestation": {
                "tinfoil": ["gpt-secure"],
                "ppq-private": [],
                "privatemode": [],
            },
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": _digest("placeholder-4"),
                        "require_model_attestations": True,
                        "model_attestation_targets": {
                            "gpt-secure": {
                                "host": "gpt-secure.tinfoil.example",
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "expected_release_digest": _digest("placeholder-9"),
                            }
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                        "proof_claims": _tinfoil_public_proof_claims(),
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["routable_with_full_attestation"] == {
        "tinfoil": [],
        "ppq-private": [],
        "privatemode": [],
    }
    assert metadata["end_to_end_ready"] is False


def test_confidentiality_listing_metadata_rejects_expired_local_tee_proof_for_end_to_end_ready() -> (
    None
):
    routstr_tee = _routstr_tee_status()
    local_verification = routstr_tee["local_verification"]
    assert isinstance(local_verification, dict)
    local_verification["verified_at"] = 1
    local_verification["expires_at"] = 2

    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "routstr_tee": routstr_tee,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": _digest("placeholder-4"),
                        "require_model_attestations": True,
                        "model_attestation_targets": {
                            "gpt-secure": {
                                "host": "gpt-secure.tinfoil.example",
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "expected_release_digest": _digest("placeholder-9"),
                            }
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                        "proof_claims": _tinfoil_public_proof_claims(),
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["providers"][0]["confidentiality"]["verified"] is True
    assert metadata["routstr_tee"]["ready"] is False
    assert metadata["end_to_end_ready"] is False


def test_confidentiality_listing_metadata_requires_local_tee_proof_for_end_to_end_ready() -> (
    None
):
    routstr_tee = _routstr_tee_status()
    routstr_tee.pop("local_verification")

    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "routstr_tee": routstr_tee,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": _digest("placeholder-4"),
                        "require_model_attestations": True,
                        "model_attestation_targets": {
                            "gpt-secure": {
                                "host": "gpt-secure.tinfoil.example",
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "expected_release_digest": _digest("placeholder-9"),
                            }
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                        "proof_claims": _tinfoil_public_proof_claims(),
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["providers"][0]["confidentiality"]["verified"] is True
    assert metadata["routstr_tee"]["ready"] is False
    assert metadata["end_to_end_ready"] is False


def test_confidentiality_listing_metadata_requires_attested_tls_key_digest_for_end_to_end_ready() -> (
    None
):
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "routstr_tee": {
                "required": True,
                "ready": True,
                "client_confidentiality": {
                    "mode": "attested-tls-termination",
                    "tls_terminates_in_attested_tee": True,
                    "inbound_ehbp_ohttp_request_decryption": False,
                },
            },
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": _digest("placeholder-4"),
                        "require_model_attestations": True,
                        "model_attestation_targets": {
                            "gpt-secure": {
                                "host": "gpt-secure.tinfoil.example",
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "expected_release_digest": _digest("placeholder-9"),
                            }
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                        "proof_claims": _tinfoil_public_proof_claims(),
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["providers"][0]["confidentiality"]["verified"] is True
    assert metadata["routstr_tee"]["ready"] is False
    assert metadata["end_to_end_ready"] is False


def test_confidentiality_listing_metadata_omits_policy_with_inline_secrets() -> None:
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": _digest("placeholder-4"),
                        "api_key": "SECRET_POLICY",
                        "model_attestation_targets": {
                            "gpt-secure": {
                                "host": "gpt-secure.tinfoil.example",
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "expected_release_digest": _digest("placeholder-9"),
                            }
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": False,
                        "mode": "tinfoil",
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                    },
                }
            ],
        }
    )

    assert "confidentiality_policy" not in str(metadata)
    assert "SECRET_POLICY" not in str(metadata)


def test_confidentiality_listing_metadata_includes_sanitized_privatemode_policy() -> (
    None
):
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "privatemode",
                    "upstream_name": "privatemode-prod",
                    "confidentiality_policy": {
                        "manifest_digest": _digest("placeholder-1"),
                        "proxy_image_digest": _digest("placeholder-2"),
                        "expected_workload_sans": ["backend.privatemode.example"],
                        "model_workload_bindings": {
                            "privatemode/gpt-oss-120b": {
                                "workload_sans": ["backend.privatemode.example"],
                            }
                        },
                        "expected_workload_identity_digest": _digest("placeholder-3"),
                        "model_workload_binding_digest": _digest("placeholder-4"),
                        "api_key_env": "PRIVATEMODE_API_KEY",
                        "manifest_url": (
                            "https://cdn.confidential.cloud/privatemode/v2/manifest.json"
                        ),
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": False,
                        "mode": "privatemode",
                        "model_ids": ["privatemode/gpt-oss-120b"],
                        "model_id_prefixes": [],
                    },
                }
            ],
        }
    )

    assert metadata is not None
    provider = metadata["providers"][0]
    assert provider["confidentiality_policy"] == {
        "expected_workload_identity_digest": _digest("placeholder-3"),
        "expected_workload_sans": ["backend.privatemode.example"],
        "manifest_digest": _digest("placeholder-1"),
        "model_workload_binding_digest": _digest("placeholder-4"),
        "model_workload_bindings": {
            "privatemode/gpt-oss-120b": {
                "workload_sans": ["backend.privatemode.example"],
            }
        },
        "proxy_image_digest": _digest("placeholder-2"),
    }
    serialized = json.dumps(metadata, sort_keys=True)
    assert "PRIVATEMODE_API_KEY" not in serialized
    assert "manifest_url" not in serialized


def test_confidentiality_listing_metadata_rejects_duplicate_tinfoil_target_keys() -> (
    None
):
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": _digest("placeholder-4"),
                        "require_model_attestations": True,
                        "model_attestation_targets": {
                            "gpt-secure": {
                                "host": "gpt-secure.tinfoil.example",
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "expected_release_digest": _digest("placeholder-9"),
                            },
                            " gpt-secure ": {
                                "host": "other.tinfoil.example",
                                "repo": "attacker/conflicting-gpt",
                                "expected_release_digest": _digest("placeholder-8"),
                            },
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": False,
                        "mode": "tinfoil",
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                    },
                }
            ],
        }
    )

    assert "confidentiality_policy" not in str(metadata)


def test_confidentiality_listing_metadata_rejects_case_variant_tinfoil_target_keys() -> (
    None
):
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": _digest("placeholder-4"),
                        "require_model_attestations": True,
                        "model_attestation_targets": {
                            "gpt-secure": {
                                "host": "gpt-secure.tinfoil.example",
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "expected_release_digest": _digest("placeholder-9"),
                            },
                            "GPT-secure": {
                                "host": "other.tinfoil.example",
                                "repo": "attacker/conflicting-gpt",
                                "expected_release_digest": _digest("placeholder-8"),
                            },
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": False,
                        "mode": "tinfoil",
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                    },
                }
            ],
        }
    )

    assert "confidentiality_policy" not in str(metadata)


def test_confidentiality_listing_metadata_rejects_duplicate_privatemode_binding_keys() -> (
    None
):
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "privatemode",
                    "confidentiality_policy": {
                        "manifest_digest": _digest("placeholder-1"),
                        "proxy_image_digest": _digest("placeholder-2"),
                        "expected_workload_sans": ["backend.privatemode.example"],
                        "model_workload_bindings": {
                            "privatemode/gpt-oss-120b": {
                                "workload_sans": ["backend.privatemode.example"],
                            },
                            " privatemode/gpt-oss-120b ": {
                                "workload_sans": ["other.privatemode.example"],
                            },
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": False,
                        "mode": "privatemode",
                        "model_ids": ["privatemode/gpt-oss-120b"],
                        "model_id_prefixes": [],
                    },
                }
            ],
        }
    )

    assert "confidentiality_policy" not in str(metadata)


def test_confidentiality_listing_metadata_rejects_case_variant_privatemode_binding_keys() -> (
    None
):
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "privatemode",
                    "confidentiality_policy": {
                        "manifest_digest": _digest("placeholder-1"),
                        "proxy_image_digest": _digest("placeholder-2"),
                        "expected_workload_sans": ["backend.privatemode.example"],
                        "model_workload_bindings": {
                            "privatemode/gpt-oss-120b": {
                                "workload_sans": ["backend.privatemode.example"],
                            },
                            "PRIVATEMODE/gpt-oss-120b": {
                                "workload_sans": ["other.privatemode.example"],
                            },
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": False,
                        "mode": "privatemode",
                        "model_ids": ["privatemode/gpt-oss-120b"],
                        "model_id_prefixes": [],
                    },
                }
            ],
        }
    )

    assert "confidentiality_policy" not in str(metadata)


def test_confidentiality_listing_metadata_rejects_nested_policy_lists_with_objects() -> (
    None
):
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": _digest("placeholder-4"),
                        "model_attestation_targets": {
                            "gpt-secure": {
                                "host": "gpt-secure.tinfoil.example",
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "allowed_release_digests": [
                                    _digest("placeholder-9"),
                                    {"Authorization": "SECRET_NESTED_LIST"},
                                ],
                            }
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": False,
                        "mode": "tinfoil",
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                    },
                }
            ],
        }
    )

    assert "confidentiality_policy" not in str(metadata)
    assert "SECRET_NESTED_LIST" not in str(metadata)


def test_confidentiality_listing_metadata_rejects_boolean_workload_bindings() -> None:
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "privatemode",
                    "confidentiality_policy": {
                        "manifest_digest": _digest("placeholder-1"),
                        "proxy_image_digest": _digest("placeholder-2"),
                        "model_workload_bindings": {
                            "privatemode/gpt-oss-120b": {
                                "workload_sans": True,
                            }
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": False,
                        "mode": "privatemode",
                        "model_ids": ["privatemode/gpt-oss-120b"],
                        "model_id_prefixes": [],
                    },
                }
            ],
        }
    )

    assert "confidentiality_policy" not in str(metadata)


def test_confidentiality_listing_metadata_rejects_boolean_policy_digest_pins() -> None:
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": True,
                        "require_model_attestations": True,
                        "model_attestation_targets": {
                            "gpt-secure": {
                                "host": "gpt-secure.tinfoil.example",
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "expected_release_digest": _digest("placeholder-9"),
                            }
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": False,
                        "mode": "tinfoil",
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                    },
                }
            ],
        }
    )

    assert "confidentiality_policy" not in str(metadata)


def test_confidentiality_listing_metadata_rejects_malformed_policy_digest_pins() -> (
    None
):
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": "not-a-digest",
                        "require_model_attestations": True,
                        "model_attestation_targets": {
                            "gpt-secure": {
                                "host": "gpt-secure.tinfoil.example",
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "expected_release_digest": _digest("placeholder-9"),
                            }
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": False,
                        "mode": "tinfoil",
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                    },
                }
            ],
        }
    )

    assert "confidentiality_policy" not in str(metadata)


def test_confidentiality_listing_metadata_rejects_malformed_measurement_policy_pins() -> (
    None
):
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": _digest("placeholder-4"),
                        "expected_code_measurement_fingerprint": "not-a-digest",
                        "require_model_attestations": True,
                        "model_attestation_targets": {
                            "gpt-secure": {
                                "host": "gpt-secure.tinfoil.example",
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "expected_release_digest": _digest("placeholder-9"),
                            }
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": False,
                        "mode": "tinfoil",
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                    },
                }
            ],
        }
    )

    assert "confidentiality_policy" not in str(metadata)


def test_confidentiality_listing_metadata_rejects_malformed_privatemode_component_pins() -> (
    None
):
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "privatemode",
                    "confidentiality_policy": {
                        "manifest_digest": _digest("placeholder-1"),
                        "proxy_image_digest": _digest("placeholder-2"),
                        "expected_coordinator_measurement": "not-a-digest",
                        "expected_workload_sans": ["backend.privatemode.example"],
                        "model_workload_bindings": {
                            "privatemode/gpt-oss-120b": {
                                "workload_sans": ["backend.privatemode.example"],
                            }
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": False,
                        "mode": "privatemode",
                        "model_ids": ["privatemode/gpt-oss-120b"],
                        "model_id_prefixes": [],
                    },
                }
            ],
        }
    )

    assert "confidentiality_policy" not in str(metadata)


def test_confidentiality_listing_metadata_filters_non_string_public_lists() -> None:
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "tinfoil-prod",
                    "supported_endpoints": [
                        "/v1/chat/completions",
                        {"Authorization": "SECRET_ENDPOINT_TOKEN"},
                        42,
                    ],
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                        "model_ids": [
                            "gpt-secure",
                            {"api_key": "SECRET_SELECTOR"},
                            123,
                        ],
                        "model_id_prefixes": [
                            "tinfoil/",
                            {"raw_prompt": "SECRET_PREFIX"},
                        ],
                    },
                }
            ],
        }
    )

    assert metadata is not None
    provider = metadata["providers"][0]
    assert provider["supported_endpoints"] == ["/v1/chat/completions"]
    assert provider["confidentiality"]["model_ids"] == ["gpt-secure"]
    assert provider["confidentiality"]["model_id_prefixes"] == ["tinfoil/"]

    serialized = json.dumps(metadata)
    assert "SECRET_ENDPOINT_TOKEN" not in serialized
    assert "SECRET_SELECTOR" not in serialized
    assert "SECRET_PREFIX" not in serialized


def test_confidentiality_listing_metadata_rejects_truthy_string_enabled() -> None:
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "tinfoil-prod",
                    "confidentiality": {
                        "enabled": "false",
                        "verified": "false",
                        "mode": "tinfoil",
                    },
                }
            ],
        }
    )

    assert metadata == {
        "mode": "required",
        "required": True,
        "end_to_end_ready": False,
        "providers": [],
    }


def test_confidentiality_listing_metadata_requires_strict_required_flag() -> None:
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": "false",
            "providers": [],
        }
    )

    assert metadata is None


def test_confidentiality_listing_metadata_rejects_non_string_provider_identity() -> (
    None
):
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": {"api_key": "SECRET_PROVIDER"},
                    "upstream_name": {"raw_prompt": "SECRET_UPSTREAM"},
                    "confidentiality": {
                        "enabled": True,
                        "verified": False,
                        "mode": "tinfoil",
                    },
                }
            ],
        }
    )

    assert metadata == {
        "mode": "required",
        "required": True,
        "end_to_end_ready": False,
        "providers": [],
    }
    assert "SECRET_PROVIDER" not in str(metadata)
    assert "SECRET_UPSTREAM" not in str(metadata)


def test_confidentiality_listing_metadata_downgrades_malformed_verified_proof() -> None:
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "tinfoil-prod",
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": "sha256:policy",
                        "evidence_digest": "sha256:evidence",
                        "expires_at": "1800000000",
                        "proof_claims": {
                            "transport": "ehbp",
                            "repo": "tinfoilsh/confidential-model-router",
                        },
                    },
                }
            ],
        }
    )

    assert metadata is not None
    confidentiality = metadata["providers"][0]["confidentiality"]
    assert confidentiality == {
        "enabled": True,
        "verified": False,
        "mode": "tinfoil",
        "model_ids": [],
        "model_id_prefixes": [],
    }


def test_confidentiality_listing_metadata_downgrades_non_boolean_verification_step() -> (
    None
):
    proof_claims = _tinfoil_public_proof_claims()
    proof_claims["verification_steps"]["Authorization"] = "SECRET_STEP"  # type: ignore[index]

    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "tinfoil-prod",
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                        "proof_claims": proof_claims,
                    },
                }
            ],
        }
    )

    assert metadata is not None
    confidentiality = metadata["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is False
    assert "policy_digest" not in confidentiality
    assert "proof_claims" not in confidentiality
    assert "evidence_digest" not in confidentiality
    assert "SECRET_STEP" not in str(metadata)


def test_confidentiality_listing_metadata_rejects_payload_policy_digest_mismatch() -> (
    None
):
    proof_claims = _tinfoil_public_proof_claims()
    proof_claims["payload_policy_digest"] = _digest("placeholder-9")

    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "tinfoil-prod",
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                        "proof_claims": proof_claims,
                    },
                }
            ],
        }
    )

    assert metadata is not None
    confidentiality = metadata["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is False
    assert "policy_digest" not in confidentiality
    assert "proof_claims" not in confidentiality
    assert "evidence_digest" not in confidentiality


def test_confidentiality_listing_metadata_rejects_payload_evidence_digest_mismatch() -> (
    None
):
    proof_claims = _tinfoil_public_proof_claims()
    proof_claims["payload_evidence_digest"] = _digest("placeholder-9")

    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "tinfoil-prod",
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                        "proof_claims": proof_claims,
                    },
                }
            ],
        }
    )

    assert metadata is not None
    confidentiality = metadata["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is False
    assert "policy_digest" not in confidentiality
    assert "proof_claims" not in confidentiality
    assert "evidence_digest" not in confidentiality


def test_confidentiality_listing_metadata_rejects_bound_payload_policy_mismatch() -> (
    None
):
    proof_claims = _tinfoil_public_proof_claims()
    proof_claims["payload_policy_digest"] = _digest("placeholder-9")

    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "tinfoil-prod",
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": [],
                        "proof_claims": proof_claims,
                    },
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": _digest("placeholder-4"),
                        "require_model_attestations": True,
                        "model_attestation_targets": {
                            "gpt-secure": {
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "expected_release_digest": _digest("placeholder-9"),
                            }
                        },
                    },
                }
            ],
        }
    )

    assert metadata is not None
    confidentiality = metadata["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is False
    assert "policy_digest" not in confidentiality
    assert "proof_claims" not in confidentiality
    assert "evidence_digest" not in confidentiality


def test_confidentiality_listing_metadata_rejects_provider_mode_mismatch() -> None:
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "custom",
                    "upstream_name": "custom-prod",
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                        "proof_claims": _tinfoil_public_proof_claims(),
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["providers"][0]["confidentiality"] == {
        "enabled": True,
        "verified": False,
        "mode": "tinfoil",
        "model_ids": [],
        "model_id_prefixes": [],
    }


def test_confidentiality_listing_metadata_rejects_malformed_verified_selectors() -> (
    None
):
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "tinfoil-prod",
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                        "model_ids": ["gpt-secure", {"api_key": "SECRET_SELECTOR"}],
                        "model_id_prefixes": [],
                        "proof_claims": _tinfoil_public_proof_claims(),
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["providers"][0]["confidentiality"] == {
        "enabled": True,
        "verified": False,
        "mode": "tinfoil",
        "model_ids": ["gpt-secure"],
        "model_id_prefixes": [],
    }


def test_confidentiality_listing_metadata_rejects_non_list_verified_selectors() -> None:
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "tinfoil-prod",
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                        "model_ids": ("gpt-secure",),
                        "model_id_prefixes": [],
                        "proof_claims": _tinfoil_public_proof_claims(),
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["providers"][0]["confidentiality"] == {
        "enabled": True,
        "verified": False,
        "mode": "tinfoil",
        "model_ids": ["gpt-secure"],
        "model_id_prefixes": [],
    }


def test_confidentiality_listing_metadata_downgrades_prefix_verified_selector() -> None:
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "tinfoil-prod",
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["gpt-secure"],
                        "model_id_prefixes": ["tinfoil/"],
                        "proof_claims": _tinfoil_public_proof_claims(),
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["providers"][0]["confidentiality"] == {
        "enabled": True,
        "verified": False,
        "mode": "tinfoil",
        "model_ids": ["gpt-secure"],
        "model_id_prefixes": ["tinfoil/"],
    }


@pytest.mark.parametrize("verified_at", [None, "1700000000"])
def test_confidentiality_listing_metadata_requires_verified_at(
    verified_at: object,
) -> None:
    confidentiality = {
        "enabled": True,
        "verified": True,
        "mode": "tinfoil",
        "verifier": "unit-test-verifier",
        "policy_digest": VALID_POLICY_DIGEST,
        "evidence_digest": VALID_EVIDENCE_DIGEST,
        "expires_at": 1_800_000_000,
    }
    if verified_at is not None:
        confidentiality["verified_at"] = verified_at

    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "tinfoil-prod",
                    "confidentiality": confidentiality,
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["providers"][0]["confidentiality"] == {
        "enabled": True,
        "verified": False,
        "mode": "tinfoil",
        "model_ids": [],
        "model_id_prefixes": [],
    }


def test_confidentiality_listing_metadata_rejects_expired_verified_proof() -> None:
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "tinfoil-prod",
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1,
                        "proof_claims": _tinfoil_public_proof_claims(),
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["providers"][0]["confidentiality"] == {
        "enabled": True,
        "verified": False,
        "mode": "tinfoil",
        "model_ids": [],
        "model_id_prefixes": [],
    }


def test_confidentiality_listing_metadata_rejects_future_verified_at() -> None:
    now = int(time.time())
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": now + 60,
                        "expires_at": now + 300,
                        "proof_claims": _tinfoil_public_proof_claims(),
                    },
                }
            ],
        }
    )

    assert metadata["providers"][0]["confidentiality"] == {
        "enabled": True,
        "verified": False,
        "mode": "tinfoil",
        "model_ids": [],
        "model_id_prefixes": [],
    }


def test_confidentiality_listing_metadata_requires_public_proof_claims() -> None:
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "tinfoil-prod",
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["providers"][0]["confidentiality"] == {
        "enabled": True,
        "verified": False,
        "mode": "tinfoil",
        "model_ids": [],
        "model_id_prefixes": [],
    }


def test_confidentiality_listing_metadata_requires_complete_public_proof_claims() -> (
    None
):
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "tinfoil-prod",
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                        "proof_claims": {
                            "transport": "ehbp",
                            "repo": "tinfoilsh/confidential-model-router",
                        },
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["providers"][0]["confidentiality"] == {
        "enabled": True,
        "verified": False,
        "mode": "tinfoil",
        "model_ids": [],
        "model_id_prefixes": [],
    }


def test_confidentiality_listing_metadata_requires_tinfoil_model_attestation_coverage() -> (
    None
):
    proof_claims = _tinfoil_public_proof_claims()

    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "tinfoil-prod",
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["tinfoil/gpt-secure"],
                        "model_id_prefixes": [],
                        "proof_claims": proof_claims,
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["providers"][0]["confidentiality"] == {
        "enabled": True,
        "verified": False,
        "mode": "tinfoil",
        "model_ids": ["tinfoil/gpt-secure"],
        "model_id_prefixes": [],
    }


def test_confidentiality_listing_metadata_requires_public_policy_binding() -> None:
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "tinfoil-prod",
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["tinfoil/gpt-secure"],
                        "model_id_prefixes": [],
                        "proof_claims": _tinfoil_public_proof_claims(),
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["providers"][0]["confidentiality"] == {
        "enabled": True,
        "verified": False,
        "mode": "tinfoil",
        "model_ids": ["tinfoil/gpt-secure"],
        "model_id_prefixes": [],
    }


def test_confidentiality_listing_metadata_rejects_weak_public_policy_binding() -> None:
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "ppq-private",
                    "upstream_name": "ppq-private-prod",
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "ppq-private-tee",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["private/gpt-oss-120b"],
                        "model_id_prefixes": [],
                        "proof_claims": _ppq_private_public_proof_claims(),
                    },
                    "confidentiality_policy": {
                        "repo": "ppq-ai/private-tee",
                        "expected_code_measurement_fingerprint": "sha256:" + ("2" * 64),
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["providers"][0]["confidentiality"] == {
        "enabled": True,
        "verified": False,
        "mode": "ppq-private-tee",
        "model_ids": ["private/gpt-oss-120b"],
        "model_id_prefixes": [],
    }


def test_confidentiality_listing_metadata_rejects_privatemode_workload_identity_mismatch() -> (
    None
):
    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "privatemode",
                    "upstream_name": "privatemode-prod",
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "privatemode",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["privatemode/gpt-oss-120b"],
                        "model_id_prefixes": [],
                        "proof_claims": _privatemode_public_proof_claims(),
                    },
                    "confidentiality_policy": {
                        "proxy_binary_digest": _digest("placeholder-6"),
                        "expected_workload_sans": ["wrong.privatemode.example"],
                        "model_workload_bindings": {
                            "privatemode/gpt-oss-120b": {
                                "workload_sans": ["backend.privatemode.example"],
                            }
                        },
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["providers"][0]["confidentiality"] == {
        "enabled": True,
        "verified": False,
        "mode": "privatemode",
        "model_ids": ["privatemode/gpt-oss-120b"],
        "model_id_prefixes": [],
    }


def test_confidentiality_listing_metadata_rejects_malformed_present_tls_alias() -> None:
    proof_claims = _tinfoil_public_proof_claims()
    proof_claims["tls_public_key_fingerprint_sha256"] = "not-a-digest"
    proof_claims["tls_public_key"] = _digest("placeholder-5")

    metadata = listing._build_confidentiality_listing_metadata(
        {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "tinfoil-prod",
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 1_800_000_000,
                        "proof_claims": proof_claims,
                    },
                }
            ],
        }
    )

    assert metadata is not None
    assert metadata["providers"][0]["confidentiality"] == {
        "enabled": True,
        "verified": False,
        "mode": "tinfoil",
        "model_ids": [],
        "model_id_prefixes": [],
    }


def test_build_listing_metadata_includes_confidentiality_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        listing,
        "_build_confidentiality_listing_metadata",
        lambda: {"mode": "required", "required": True, "providers": []},
    )

    metadata = listing.build_listing_metadata("Routstr", "AI proxy")

    assert metadata == {
        "name": "Routstr",
        "about": "AI proxy",
        "confidentiality": {
            "mode": "required",
            "required": True,
            "providers": [],
        },
    }
