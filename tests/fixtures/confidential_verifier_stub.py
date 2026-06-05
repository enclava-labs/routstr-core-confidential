from __future__ import annotations

import base64
import hashlib
import json
import os
import sys


def _ehbp_public_key_hex(key_config_b64: str) -> str:
    key_config = base64.b64decode(key_config_b64, validate=True)
    if len(key_config) < 35:
        raise ValueError("EHBP key config is too short")
    key_len = int.from_bytes(key_config[1:3], "big")
    public_key = key_config[3 : 3 + key_len]
    if len(public_key) != 32:
        raise ValueError("EHBP public key must be 32 bytes")
    return public_key.hex()


def _test_digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


def _sha256_file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _policy_string(policy: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = policy.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _sha256_json_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _normalized_string_policy_values(
    policy: dict[str, object],
    *keys: str,
) -> list[str]:
    values: list[str] = []
    for key in keys:
        if key not in policy:
            continue
        value = policy[key]
        raw_values = value if isinstance(value, list) else [value]
        for raw_value in raw_values:
            if not isinstance(raw_value, str):
                continue
            normalized = raw_value.strip()
            if normalized and normalized not in values:
                values.append(normalized)
    return sorted(values)


def _expected_workload_identity_digest(policy: dict[str, object]) -> str:
    return _sha256_json_digest(
        {
            "ids": _normalized_string_policy_values(
                policy,
                "expected_workload_ids",
                "expectedWorkloadIDs",
            ),
            "sans": _normalized_string_policy_values(
                policy,
                "expected_workload_sans",
                "expectedWorkloadSANs",
            ),
        }
    )


def _model_workload_binding_digest(policy: dict[str, object]) -> str:
    bindings = policy.get("model_workload_bindings")
    if not isinstance(bindings, dict):
        bindings = policy.get("modelWorkloadBindings")
    if not isinstance(bindings, dict):
        bindings = {}

    normalized: dict[str, dict[str, list[str]]] = {}
    for model_id, binding in bindings.items():
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        if not isinstance(binding, dict):
            continue
        normalized[model_id.strip()] = {
            "workload_ids": _normalized_string_policy_values(
                binding,
                "workload_ids",
                "workloadIDs",
            ),
            "workload_sans": _normalized_string_policy_values(
                binding,
                "workload_sans",
                "workloadSANs",
            ),
        }
    return _sha256_json_digest(dict(sorted(normalized.items())))


def _key_release_binding_digest(claims: dict[str, object]) -> str:
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


def _ehbp_verification_steps() -> dict[str, bool]:
    return {
        "attested_transport_key_binding": True,
        "code_transparency": True,
        "freshness": True,
        "hardware_attestation_report": True,
        "hardware_certificate_chain": True,
        "measurement_match": True,
    }


def _privatemode_verification_steps() -> dict[str, bool]:
    return {
        "ai_worker_attestation": True,
        "contrast_manifest": True,
        "coordinator_attestation": True,
        "gpu_attestation": True,
        "key_release_binding": True,
        "mesh_ca_binding": True,
        "nvidia_ocsp_revocation": True,
        "prompt_encryption": True,
        "secret_service_tls": True,
    }


def _routstr_tee_verification_steps() -> dict[str, bool]:
    return {
        "freshness": True,
        "hpke_key_binding": True,
        "measurement_match": True,
        "public_key_binding": True,
        "runtime_policy_binding": True,
        "tee_attestation_report": True,
        "tee_certificate_chain": True,
    }


def _ehbp_claims(payload: dict[str, object]) -> dict[str, object]:
    policy = payload.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("policy must be a JSON object")

    claims = {
        "attested_hpke_public_key_hex": _ehbp_public_key_hex(
            str(payload.get("hpke_key_config_b64") or "")
        ),
        "code_measurement_fingerprint": _test_digest("fixture-code-measurement"),
        "enclave_measurement_fingerprint": _test_digest("fixture-enclave-measurement"),
        "payload_evidence_digest": payload.get("evidence_digest"),
        "payload_policy_digest": payload.get("policy_digest"),
        "payload_verification_nonce": payload.get("verification_nonce"),
        "verification_steps": _ehbp_verification_steps(),
    }
    release_digest = policy.get("expected_release_digest")
    if release_digest is None:
        allowed_release_digests = policy.get("allowed_release_digests")
        if isinstance(allowed_release_digests, list) and allowed_release_digests:
            release_digest = allowed_release_digests[0]
    if release_digest is not None:
        claims["release_digest"] = release_digest
    mode = payload.get("mode")
    if mode == "tinfoil":
        attestation = payload.get("attestation")
        if not isinstance(attestation, dict):
            raise ValueError("attestation must be a JSON object")
        claims.update(
            {
                "attestation_body_encoding": attestation.get("body_encoding"),
                "attestation_format": attestation.get("format"),
                "attestation_report_digest": attestation.get("report_digest"),
                "repo": policy.get("repo") or policy.get("expected_repo"),
                "tls_public_key_fingerprint_sha256": _test_digest(
                    "fixture-tls-key-fingerprint"
                ),
            }
        )
        model_attestations = payload.get("model_attestations")
        if isinstance(model_attestations, list) and model_attestations:
            model_claims: dict[str, object] = {}
            targets = policy.get("model_attestation_targets")
            if not isinstance(targets, dict):
                targets = {}
            for raw_entry in model_attestations:
                if not isinstance(raw_entry, dict):
                    continue
                model_id = str(raw_entry.get("model_id") or "").strip()
                if not model_id:
                    continue
                target_policy = targets.get(model_id)
                if not isinstance(target_policy, dict):
                    target_policy = {}
                target_attestation = raw_entry.get("attestation")
                if not isinstance(target_attestation, dict):
                    target_attestation = {}
                model_release_digest = target_policy.get("expected_release_digest")
                if model_release_digest is None:
                    allowed_release_digests = target_policy.get(
                        "allowed_release_digests"
                    )
                    if (
                        isinstance(allowed_release_digests, list)
                        and allowed_release_digests
                    ):
                        model_release_digest = allowed_release_digests[0]
                model_claims[model_id] = {
                    "attested_hpke_public_key_hex": _ehbp_public_key_hex(
                        str(raw_entry.get("hpke_key_config_b64") or "")
                    ),
                    "attestation_format": target_attestation.get("format"),
                    "attestation_report_digest": target_attestation.get(
                        "report_digest"
                    ),
                    "code_measurement_fingerprint": _test_digest(
                        "fixture-code-measurement"
                    ),
                    "enclave_measurement_fingerprint": _test_digest(
                        "fixture-enclave-measurement"
                    ),
                    "release_digest": model_release_digest,
                    "repo": target_policy.get("repo")
                    or target_policy.get("expected_repo"),
                    "tls_public_key_fingerprint_sha256": _test_digest(
                        "fixture-tls-key-fingerprint"
                    ),
                    "verification_steps": _ehbp_verification_steps(),
                }
            claims["model_attestations"] = model_claims
    elif mode == "ppq-private-tee":
        claims["attestation_bundle_url"] = payload.get("attestation_bundle_url")
        claims["client_encryption_boundary"] = "routstr-tee-ehbp-proxy"
        claims["repo"] = policy.get("repo") or policy.get("expected_repo")
        claims["tls_public_key_fingerprint_sha256"] = _test_digest(
            "fixture-tls-key-fingerprint"
        )
        model_attestations = payload.get("model_attestations")
        if isinstance(model_attestations, list) and model_attestations:
            model_claims: dict[str, object] = {}
            targets = policy.get("model_attestation_targets")
            if not isinstance(targets, dict):
                targets = {}
            for raw_entry in model_attestations:
                if not isinstance(raw_entry, dict):
                    continue
                model_id = str(raw_entry.get("model_id") or "").strip()
                if not model_id:
                    continue
                target_policy = targets.get(model_id)
                if not isinstance(target_policy, dict):
                    target_policy = {}
                target_attestation = raw_entry.get("attestation")
                if not isinstance(target_attestation, dict):
                    target_attestation = {}
                model_release_digest = target_policy.get("expected_release_digest")
                if model_release_digest is None:
                    allowed_release_digests = target_policy.get(
                        "allowed_release_digests"
                    )
                    if (
                        isinstance(allowed_release_digests, list)
                        and allowed_release_digests
                    ):
                        model_release_digest = allowed_release_digests[0]
                model_claims[model_id] = {
                    "attested_hpke_public_key_hex": _ehbp_public_key_hex(
                        str(raw_entry.get("hpke_key_config_b64") or "")
                    ),
                    "attestation_format": target_attestation.get("format"),
                    "attestation_report_digest": target_attestation.get(
                        "report_digest"
                    ),
                    "code_measurement_fingerprint": _test_digest(
                        "fixture-code-measurement"
                    ),
                    "enclave_measurement_fingerprint": _test_digest(
                        "fixture-enclave-measurement"
                    ),
                    "release_digest": model_release_digest,
                    "repo": target_policy.get("repo")
                    or target_policy.get("expected_repo"),
                    "tls_public_key_fingerprint_sha256": _test_digest(
                        "fixture-tls-key-fingerprint"
                    ),
                    "verification_steps": _ehbp_verification_steps(),
                }
            claims["backend_model_attestations"] = model_claims
    else:
        raise ValueError(f"unsupported EHBP verifier mode: {mode}")
    return claims


def _privatemode_claims(payload: dict[str, object]) -> dict[str, object]:
    policy = payload.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("policy must be a JSON object")
    claims: dict[str, object] = {
        "ai_worker_measurement": _test_digest("fixture-attested-workload-policy"),
        "ai_worker_manifest_digest": _test_digest("fixture-ai-worker-manifest"),
        "attested_workload_identity_digest": _test_digest(
            "fixture-attested-workload-identity"
        ),
        "attested_workload_policy_digest": _test_digest(
            "fixture-attested-workload-policy"
        ),
        "coordinator_measurement": _test_digest("fixture-coordinator"),
        "coordinator_attestation_doc_digest": _test_digest("fixture-coordinator-doc"),
        "expected_workload_identity_digest": _expected_workload_identity_digest(policy),
        "gpu_attestation_policy": "nvidia-ocsp-good-only",
        "inference_secret_id_digest": _test_digest("fixture-inference-secret-id"),
        "manifest_digest": policy.get("manifest_digest"),
        "mesh_ca_digest": _test_digest("fixture-mesh-ca"),
        "model_workload_binding_digest": _model_workload_binding_digest(policy),
        "nvidia_ocsp_policy_header_digest": _test_digest("fixture-nvidia-ocsp-header"),
        "nvidia_ocsp_policy_mac_digest": _test_digest("fixture-nvidia-ocsp-mac"),
        "payload_evidence_digest": payload.get("evidence_digest"),
        "payload_policy_digest": payload.get("policy_digest"),
        "payload_verification_nonce": payload.get("verification_nonce"),
        "prompt_encryption_ciphertext_digest": _test_digest(
            "fixture-prompt-ciphertext"
        ),
        "proxy_image_digest": policy.get("proxy_image_digest"),
        "proxy_binary_digest": policy.get("proxy_binary_digest"),
        "proxy_base_url": payload.get("proxy_base_url"),
        "secret_service_certificate_digest": _test_digest(
            "fixture-secret-service-cert"
        ),
        "secret_service_measurement": _test_digest("fixture-secret-service"),
        "transport": "privatemode-proxy",
        "trust_tier": "app-e2ee",
        "verification_steps": _privatemode_verification_steps(),
    }
    claims["key_release_binding"] = _key_release_binding_digest(claims)
    return claims


def _routstr_tee_claims(payload: dict[str, object]) -> dict[str, object]:
    claims = {
        "attestation_evidence_digest": payload.get("attestation_evidence_digest"),
        "attestation_document_format": payload.get("attestation_document_format"),
        "hpke_key_config_digest": payload.get("hpke_key_config_digest"),
        "hpke_public_key_digest": payload.get("hpke_public_key_digest"),
        "payload_routing_policy_digest": payload.get("routing_policy_digest"),
        "payload_verification_nonce": payload.get("verification_nonce"),
        "routstr_code_measurement": _test_digest("fixture-routstr-code-measurement"),
        "routstr_config_measurement": payload.get("routing_policy_digest"),
        "tee_attestation_report_digest": payload.get("attestation_evidence_digest"),
        "tee_certificate_chain_digest": _test_digest("fixture-tee-certificate-chain"),
        "tee_report_data_hex": payload.get("tee_report_data_hex"),
        "tee_report_data_digest": payload.get("tee_report_data_digest"),
        "tee_report_nonce": payload.get("verification_nonce"),
        "tee_report_nonce_digest": _test_digest(
            str(payload.get("verification_nonce") or "")
        ),
        "verification_steps": _routstr_tee_verification_steps(),
    }
    if os.environ.get("ROUTSTR_VERIFIER_STUB_BEHAVIOR") != "omit_public_key_digest":
        public_key_digest = payload.get("public_key_digest")
        if public_key_digest is not None:
            claims["public_key_digest"] = public_key_digest
    local_artifacts: dict[str, object] = {}
    local_artifact_inputs = payload.get("local_artifacts")
    if not isinstance(local_artifact_inputs, dict):
        local_artifact_inputs = {}
    routing_policy = payload.get("routing_policy")
    if isinstance(routing_policy, dict):
        providers = routing_policy.get("providers")
        if isinstance(providers, list):
            for provider in providers:
                if not isinstance(provider, dict):
                    continue
                confidentiality = provider.get("confidentiality")
                if not isinstance(confidentiality, dict):
                    confidentiality = {}
                policy = provider.get("confidentiality_policy")
                if not isinstance(policy, dict):
                    continue
                if not (
                    provider.get("provider_type") == "privatemode"
                    or confidentiality.get("mode") == "privatemode"
                ):
                    continue
                if confidentiality.get("verified") is True:
                    proxy_digest = _policy_string(
                        policy,
                        "proxy_binary_digest",
                        "proxyBinaryDigest",
                    )
                    if not proxy_digest:
                        continue
                    proxy_path = _policy_string(
                        policy,
                        "proxy_binary_path",
                        "proxyBinaryPath",
                    )
                    if not proxy_path:
                        local_artifact = local_artifact_inputs.get(
                            "privatemode_proxy_binary"
                        )
                        if not isinstance(local_artifact, dict):
                            raise ValueError(
                                "local_artifacts.privatemode_proxy_binary is required "
                                "when proxy_binary_digest is set"
                            )
                        if (
                            _policy_string(local_artifact, "digest")
                            != proxy_digest
                        ):
                            raise ValueError(
                                "local_artifacts.privatemode_proxy_binary digest "
                                "does not match routing policy"
                            )
                        proxy_path = _policy_string(local_artifact, "path")
                    if not proxy_path:
                        raise ValueError(
                            "local_artifacts.privatemode_proxy_binary.path is required "
                            "when proxy_binary_digest is set"
                        )
                    try:
                        actual_digest = _sha256_file_digest(proxy_path)
                    except OSError as exc:
                        raise ValueError(
                            "hashing local_artifacts.privatemode_proxy_binary.path "
                            "failed"
                        ) from exc
                    if actual_digest != proxy_digest:
                        raise ValueError(
                            "proxy_binary_digest does not match proxy_binary_path"
                        )
                    if (
                        "privatemode_proxy_binary" in local_artifacts
                        and local_artifacts["privatemode_proxy_binary"] != proxy_digest
                    ):
                        raise ValueError(
                            "attested_local_artifacts.privatemode_proxy_binary has "
                            "conflicting routing policy digests"
                        )
                    local_artifacts["privatemode_proxy_binary"] = proxy_digest
    if local_artifacts:
        claims["attested_local_artifacts"] = local_artifacts
    return claims


def main() -> int:
    behavior = os.environ.get("ROUTSTR_VERIFIER_STUB_BEHAVIOR", "verified")
    if behavior == "non_json":
        print("not-json")
        return 0
    if behavior == "failure":
        sys.stderr.write("fixture verifier failure")
        return 7
    if behavior == "failure_secret":
        sys.stderr.write(
            "fixture verifier failure SECRET_PROMPT=hello "
            "sk-live-secret attestation=raw-quote"
        )
        return 7

    payload = json.load(sys.stdin)
    schema_version = payload.get("schema_version")
    if schema_version not in {
        "routstr-confidential-verifier-v1",
        "routstr-tee-verifier-v1",
    }:
        raise ValueError("unexpected schema_version")

    mode = payload.get("mode")
    if schema_version == "routstr-tee-verifier-v1":
        claims = _routstr_tee_claims(payload)
        result = {
            "attestation_evidence_digest": payload.get("attestation_evidence_digest"),
            "claims": claims,
            "expires_at": 4_102_444_800,
            "hpke_key_config_digest": payload.get("hpke_key_config_digest"),
            "routing_policy_digest": payload.get("routing_policy_digest"),
            "verified": behavior != "unverified",
            "verified_at": 1_700_000_000,
            "verification_nonce": payload.get("verification_nonce"),
            "verifier": "routstr-test-tee-verifier",
        }
    elif mode in {"tinfoil", "ppq-private-tee"}:
        claims = _ehbp_claims(payload)
        result = {
            "claims": claims,
            "evidence_digest": payload.get("evidence_digest"),
            "expires_at": 4_102_444_800,
            "policy_digest": payload.get("policy_digest"),
            "verified": behavior != "unverified",
            "verified_at": 1_700_000_000,
            "verification_nonce": payload.get("verification_nonce"),
            "verifier": "routstr-test-confidential-verifier",
        }
    elif mode == "privatemode":
        claims = _privatemode_claims(payload)
        result = {
            "claims": claims,
            "evidence_digest": payload.get("evidence_digest"),
            "expires_at": 4_102_444_800,
            "policy_digest": payload.get("policy_digest"),
            "verified": behavior != "unverified",
            "verified_at": 1_700_000_000,
            "verification_nonce": payload.get("verification_nonce"),
            "verifier": "routstr-test-confidential-verifier",
        }
    else:
        raise ValueError(f"unsupported verifier mode: {mode}")

    json.dump(
        result,
        sys.stdout,
        sort_keys=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
