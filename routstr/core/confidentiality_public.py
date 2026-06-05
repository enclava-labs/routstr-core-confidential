from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .policy_secrets import inline_policy_secret_violations

FULL_SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
RAW_SHA256_DIGEST_RE = re.compile(r"^[0-9a-fA-F]{64}$")
HEX_64_RE = re.compile(r"^[0-9a-fA-F]{64}$")

PUBLIC_PROVIDER_PROOF_CLAIM_KEYS = {
    "ai_worker_manifest_digest",
    "ai_worker_measurement",
    "attested_workload_identity_digest",
    "attested_workload_policy_digest",
    "attestation_format",
    "attestation_bundle_url_digest",
    "attestation_report_digest",
    "attested_hpke_public_key_hex",
    "backend_model_attestations",
    "client_encryption_boundary",
    "code_measurement_fingerprint",
    "coordinator_attestation_doc_digest",
    "coordinator_measurement",
    "ehbp_config_digest",
    "enclave_measurement_fingerprint",
    "expected_workload_identity_digest",
    "gpu_attestation_policy",
    "inference_secret_id_digest",
    "key_release_binding",
    "manifest_digest",
    "manifest_log_digest",
    "manifest_log_manifest_digest",
    "mesh_ca_digest",
    "model_workload_binding_digest",
    "nvidia_ocsp_policy_header_digest",
    "nvidia_ocsp_policy_mac_digest",
    "payload_evidence_digest",
    "payload_policy_digest",
    "prompt_encryption_ciphertext_digest",
    "proxy_binary_digest",
    "proxy_image_digest",
    "release_digest",
    "repo",
    "secret_service_certificate_digest",
    "secret_service_measurement",
    "selected_model_ids",
    "tls_public_key",
    "tls_public_key_fingerprint_sha256",
    "transport",
    "trust_tier",
}

PUBLIC_TINFOIL_MODEL_ATTESTATION_KEYS = {
    "attestation_format",
    "attestation_report_digest",
    "attested_hpke_public_key_hex",
    "code_measurement_fingerprint",
    "enclave_measurement_fingerprint",
    "release_digest",
    "repo",
    "tls_public_key",
    "tls_public_key_fingerprint_sha256",
}

PUBLIC_TINFOIL_MODEL_ATTESTATION_DIGEST_KEYS = {
    "attestation_report_digest",
    "code_measurement_fingerprint",
    "enclave_measurement_fingerprint",
    "release_digest",
    "tls_public_key",
    "tls_public_key_fingerprint_sha256",
}
REQUIRED_TINFOIL_MODEL_ATTESTATION_DIGEST_KEYS = (
    "attestation_report_digest",
    "code_measurement_fingerprint",
    "enclave_measurement_fingerprint",
    "release_digest",
)

PUBLIC_TINFOIL_MODEL_ATTESTATION_HEX64_KEYS = {
    "attested_hpke_public_key_hex",
}

PUBLIC_TINFOIL_MODEL_ATTESTATION_STRING_KEYS = {
    "attestation_format",
    "repo",
}

PUBLIC_PROVIDER_DIGEST_PROOF_CLAIM_KEYS = {
    "ai_worker_manifest_digest",
    "ai_worker_measurement",
    "attested_workload_identity_digest",
    "attested_workload_policy_digest",
    "attestation_bundle_url_digest",
    "attestation_report_digest",
    "code_measurement_fingerprint",
    "coordinator_attestation_doc_digest",
    "coordinator_measurement",
    "ehbp_config_digest",
    "enclave_measurement_fingerprint",
    "expected_workload_identity_digest",
    "inference_secret_id_digest",
    "key_release_binding",
    "manifest_digest",
    "manifest_log_digest",
    "manifest_log_manifest_digest",
    "mesh_ca_digest",
    "model_workload_binding_digest",
    "nvidia_ocsp_policy_header_digest",
    "nvidia_ocsp_policy_mac_digest",
    "payload_evidence_digest",
    "payload_policy_digest",
    "prompt_encryption_ciphertext_digest",
    "proxy_binary_digest",
    "proxy_image_digest",
    "release_digest",
    "secret_service_certificate_digest",
    "secret_service_measurement",
    "tls_public_key",
    "tls_public_key_fingerprint_sha256",
}

PUBLIC_PROVIDER_HEX64_PROOF_CLAIM_KEYS = {
    "attested_hpke_public_key_hex",
}

PUBLIC_PROVIDER_STRING_PROOF_CLAIM_KEYS = {
    "attestation_format",
    "client_encryption_boundary",
    "gpu_attestation_policy",
    "repo",
    "trust_tier",
}
PUBLIC_PROVIDER_STRING_LIST_PROOF_CLAIM_KEYS = {
    "selected_model_ids",
}

PUBLIC_PROVIDER_TRANSPORT_VALUES = {
    "ehbp",
    "privatemode-proxy",
}

PUBLIC_PROVIDER_VERIFICATION_STEP_KEYS = {
    "ai_worker_attestation",
    "attested_transport_key_binding",
    "code_transparency",
    "contrast_manifest",
    "coordinator_attestation",
    "freshness",
    "gpu_attestation",
    "hardware_attestation_report",
    "hardware_certificate_chain",
    "key_release_binding",
    "measurement_match",
    "mesh_ca_binding",
    "nvidia_ocsp_revocation",
    "prompt_encryption",
    "secret_service_tls",
}

REQUIRED_EHBP_PROVIDER_VERIFICATION_STEPS = (
    "hardware_attestation_report",
    "hardware_certificate_chain",
    "code_transparency",
    "measurement_match",
    "attested_transport_key_binding",
    "freshness",
)

REQUIRED_PRIVATEMODE_PROVIDER_VERIFICATION_STEPS = (
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
PRIVATEMODE_GPU_ATTESTATION_POLICY = "nvidia-ocsp-good-only"

REQUIRED_PRIVATEMODE_PUBLIC_DIGEST_CLAIMS = (
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
REQUIRED_PROVIDER_PAYLOAD_BINDING_CLAIMS = (
    "payload_evidence_digest",
    "payload_policy_digest",
)
PRIVATEMODE_KEY_RELEASE_BINDING_CLAIMS = (
    "attested_workload_policy_digest",
    "expected_workload_identity_digest",
    "model_workload_binding_digest",
    "manifest_digest",
    "mesh_ca_digest",
    "secret_service_certificate_digest",
    "inference_secret_id_digest",
    "nvidia_ocsp_policy_mac_digest",
)

PUBLIC_CONFIDENTIAL_PROVIDER_MODES = {
    "tinfoil": "tinfoil",
    "ppq-private": "ppq-private-tee",
    "privatemode": "privatemode",
}

TINFOIL_PUBLIC_RELEASE_POLICY_KEYS = (
    "expected_release_digest",
    "release_digest",
    "allowed_release_digest",
    "allowed_release_digests",
)
TINFOIL_PUBLIC_CODE_MEASUREMENT_POLICY_KEYS = (
    "expected_code_measurement_fingerprint",
    "code_measurement_fingerprint",
    "allowed_code_measurement_fingerprint",
    "allowed_code_measurement_fingerprints",
    "allowed_code_measurements",
)
TINFOIL_PUBLIC_ENCLAVE_MEASUREMENT_POLICY_KEYS = (
    "expected_enclave_measurement_fingerprint",
    "enclave_measurement_fingerprint",
    "allowed_enclave_measurement_fingerprint",
    "allowed_enclave_measurement_fingerprints",
    "allowed_enclave_measurements",
)
TINFOIL_PUBLIC_MODEL_TARGET_KEYS = (
    "model_attestation_targets",
    "modelAttestationTargets",
    "model_enclave_bindings",
    "modelEnclaveBindings",
)
PRIVATEMODE_PUBLIC_WORKLOAD_BINDING_KEYS = (
    "model_workload_bindings",
    "modelWorkloadBindings",
)


def _public_policy_string(policy: dict[str, Any], *keys: str) -> str | None:
    selected: str | None = None
    for key in keys:
        value = policy.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            return None
        normalized = value.strip()
        if selected is None:
            selected = normalized
        elif selected != normalized:
            return None
    return selected


def _public_policy_digest_values(policy: dict[str, Any], *keys: str) -> set[str] | None:
    values: set[str] = set()
    for key in keys:
        if key not in policy:
            continue
        value = policy[key]
        if isinstance(value, str):
            digest = _public_sha256_digest(value)
            if digest is None:
                return None
            values.add(digest)
        elif isinstance(value, list):
            if not value:
                return None
            for item in value:
                digest = _public_sha256_digest(item)
                if digest is None:
                    return None
                values.add(digest)
        else:
            return None
    return values


def _public_policy_digest_alias_values(
    policy: dict[str, Any],
    *keys: str,
) -> set[str] | None:
    selected_values: set[str] | None = None
    for key in keys:
        if key not in policy:
            continue
        values = _public_policy_digest_values(policy, key)
        if values is None or not values:
            return None
        if selected_values is None:
            selected_values = values
        elif selected_values != values:
            return None
    return selected_values or set()


def _public_policy_claim_matches(
    policy: dict[str, Any],
    claims: dict[str, Any],
    claim_key: str,
    *policy_keys: str,
) -> bool:
    expected = _public_policy_digest_values(policy, *policy_keys)
    if expected is None:
        return False
    if not expected:
        return True
    claim = _public_sha256_digest(claims.get(claim_key))
    return claim in expected


def _public_policy_has_digest_pin(policy: dict[str, Any], *keys: str) -> bool:
    values = _public_policy_digest_values(policy, *keys)
    return bool(values)


def _public_policy_claim_matches_digest_aliases(
    policy: dict[str, Any],
    claims: dict[str, Any],
    claim_key: str,
    *policy_keys: str,
) -> bool:
    expected = _public_policy_digest_alias_values(policy, *policy_keys)
    if not expected:
        return False
    claim = _public_sha256_digest(claims.get(claim_key))
    return claim in expected


def _public_policy_has_digest_alias_pin(policy: dict[str, Any], *keys: str) -> bool:
    values = _public_policy_digest_alias_values(policy, *keys)
    return bool(values)


def _public_ppq_attestation_bundle_policy_digests(
    policy: dict[str, Any],
) -> set[str] | None:
    digests = _public_policy_digest_values(policy, "attestation_bundle_url_digest")
    if digests is None:
        return None
    attestation_bundle_url = None
    if "attestation_bundle_url" in policy:
        attestation_bundle_url = _public_policy_string(
            policy,
            "attestation_bundle_url",
        )
        if attestation_bundle_url is None:
            return None
    if attestation_bundle_url:
        url_digest = _sha256_json_digest(attestation_bundle_url)
        if digests and url_digest not in digests:
            return None
        digests.add(url_digest)
    return digests


def _public_policy_targets(policy: dict[str, Any]) -> dict[str, Any] | None:
    selected: dict[str, Any] | None = None
    for key in TINFOIL_PUBLIC_MODEL_TARGET_KEYS:
        value = policy.get(key)
        if value is None:
            continue
        if not isinstance(value, dict) or not value:
            return None
        seen_model_ids = set()
        for raw_model_id in value:
            if not isinstance(raw_model_id, str) or not raw_model_id.strip():
                return None
            model_id = raw_model_id.strip().lower()
            if model_id in seen_model_ids:
                return None
            seen_model_ids.add(model_id)
        if selected is None:
            selected = value
        elif selected != value:
            return None
    return selected


def _public_policy_model_workload_bindings(
    policy: dict[str, Any],
) -> dict[str, Any] | None:
    selected: dict[str, Any] | None = None
    for key in PRIVATEMODE_PUBLIC_WORKLOAD_BINDING_KEYS:
        value = policy.get(key)
        if value is None:
            continue
        if not isinstance(value, dict) or not value:
            return None
        seen_model_ids = set()
        for raw_model_id in value:
            if not isinstance(raw_model_id, str) or not raw_model_id.strip():
                return None
            model_id = raw_model_id.strip().lower()
            if model_id in seen_model_ids:
                return None
            seen_model_ids.add(model_id)
        if selected is None:
            selected = value
        elif selected != value:
            return None
    return selected


def _public_policy_sorted_strings(policy: dict[str, Any], *keys: str) -> list[str] | None:
    values: list[str] = []
    for key in keys:
        if key not in policy:
            continue
        value = policy.get(key)
        if not isinstance(value, list) or not value:
            return None
        for item in value:
            if not isinstance(item, str) or not item.strip():
                return None
            values.append(item.strip())
    if len({value.lower() for value in values}) != len(values):
        return None
    return sorted(values)


def _public_privatemode_expected_workload_identity_digest(
    policy: dict[str, Any],
) -> str | None:
    workload_ids = _public_policy_sorted_strings(
        policy,
        "expected_workload_ids",
        "expectedWorkloadIDs",
    )
    workload_sans = _public_policy_sorted_strings(
        policy,
        "expected_workload_sans",
        "expectedWorkloadSANs",
    )
    if workload_ids is None or workload_sans is None:
        return None
    if not workload_ids and not workload_sans:
        return None
    return _sha256_json_digest(
        {
            "ids": workload_ids,
            "sans": workload_sans,
        }
    )


def _public_privatemode_expected_workload_sets(
    policy: dict[str, Any],
) -> tuple[set[str], set[str]] | None:
    workload_ids = _public_policy_sorted_strings(
        policy,
        "expected_workload_ids",
        "expectedWorkloadIDs",
    )
    workload_sans = _public_policy_sorted_strings(
        policy,
        "expected_workload_sans",
        "expectedWorkloadSANs",
    )
    if workload_ids is None or workload_sans is None:
        return None
    if not workload_ids and not workload_sans:
        return None
    return set(workload_ids), set(workload_sans)


def _public_privatemode_gpu_policy_matches(
    policy: dict[str, Any],
    proof_claims: dict[str, Any],
) -> bool:
    expected = _public_policy_string(
        policy,
        "expected_gpu_attestation_policy",
        "expectedGpuAttestationPolicy",
    )
    if expected is None:
        return False
    return proof_claims.get("gpu_attestation_policy") == expected


def _public_selected_model_set(model_ids: object) -> set[str] | None:
    if not isinstance(model_ids, list) or not model_ids:
        return None
    selected: list[str] = []
    for model_id in model_ids:
        if not isinstance(model_id, str) or not model_id.strip():
            return None
        selected.append(model_id.strip())
    if len({model_id.lower() for model_id in selected}) != len(selected):
        return None
    return set(selected)


def public_model_id_matches_verified_selector(
    model_id: object,
    model_ids: object,
    *,
    provider_type: object = None,
    mode: object = None,
) -> bool:
    """Return whether a local model ID is covered by an exact verified selector."""
    if not isinstance(model_id, str) or not model_id.strip():
        return False
    if not isinstance(model_ids, list) or not model_ids:
        return False
    normalized_model_id = model_id.strip().lower()
    normalized_selectors: list[str] = []
    for selector in model_ids:
        if not isinstance(selector, str) or not selector.strip():
            return False
        normalized_selectors.append(selector.strip().lower())

    if normalized_model_id in normalized_selectors:
        return True
    provider = provider_type.strip().lower() if isinstance(provider_type, str) else ""
    normalized_mode = mode.strip().lower() if isinstance(mode, str) else ""
    if provider != "tinfoil" and normalized_mode != "tinfoil":
        return False
    if "/" in normalized_model_id:
        return False
    return any(
        selector.startswith("tinfoil/")
        and selector.split("/", 1)[1] == normalized_model_id
        for selector in normalized_selectors
    )


def _public_tinfoil_model_attestation_claims_match_target(
    target: dict[str, Any],
    claims: dict[str, Any],
) -> bool:
    expected_repo = _public_policy_string(target, "repo", "expected_repo")
    if expected_repo is None or claims.get("repo") != expected_repo:
        return False
    if not _public_policy_has_digest_pin(
        target,
        *TINFOIL_PUBLIC_RELEASE_POLICY_KEYS,
    ):
        return False
    if not _public_policy_claim_matches(
        target,
        claims,
        "release_digest",
        *TINFOIL_PUBLIC_RELEASE_POLICY_KEYS,
    ):
        return False
    if not _public_policy_claim_matches(
        target,
        claims,
        "code_measurement_fingerprint",
        *TINFOIL_PUBLIC_CODE_MEASUREMENT_POLICY_KEYS,
    ):
        return False
    if not _public_policy_claim_matches(
        target,
        claims,
        "enclave_measurement_fingerprint",
        *TINFOIL_PUBLIC_ENCLAVE_MEASUREMENT_POLICY_KEYS,
    ):
        return False
    return True


def _public_tinfoil_policy_binds_proof(
    policy: dict[str, Any],
    proof_claims: dict[str, Any],
    model_ids: object,
) -> bool:
    selected = _public_selected_model_set(model_ids)
    if not selected:
        return False
    expected_repo = _public_policy_string(policy, "repo", "expected_repo")
    if expected_repo is None or proof_claims.get("repo") != expected_repo:
        return False
    if not _public_policy_has_digest_pin(policy, *TINFOIL_PUBLIC_RELEASE_POLICY_KEYS):
        return False
    if not _public_policy_claim_matches(
        policy,
        proof_claims,
        "release_digest",
        *TINFOIL_PUBLIC_RELEASE_POLICY_KEYS,
    ):
        return False
    if not _public_policy_claim_matches(
        policy,
        proof_claims,
        "code_measurement_fingerprint",
        *TINFOIL_PUBLIC_CODE_MEASUREMENT_POLICY_KEYS,
    ):
        return False
    if not _public_policy_claim_matches(
        policy,
        proof_claims,
        "enclave_measurement_fingerprint",
        *TINFOIL_PUBLIC_ENCLAVE_MEASUREMENT_POLICY_KEYS,
    ):
        return False
    if (
        policy.get("require_model_attestations") is not True
        and policy.get("requireModelAttestations") is not True
    ):
        return False
    targets = _public_policy_targets(policy)
    if not isinstance(targets, dict) or set(targets) != selected:
        return False
    model_attestations = proof_claims.get("model_attestations")
    if not isinstance(model_attestations, dict) or set(model_attestations) != selected:
        return False
    for model_id in selected:
        target = targets.get(model_id)
        claims = model_attestations.get(model_id)
        if not isinstance(target, dict) or not isinstance(claims, dict):
            return False
        if not _public_tinfoil_model_attestation_claims_match_target(target, claims):
            return False
    return True


def _public_tinfoil_tls_public_key_binding_required(
    policy: dict[str, Any] | None,
    *,
    target_policy: dict[str, Any] | None = None,
) -> bool:
    for raw_policy in (target_policy or {}, policy or {}):
        for key in ("transport_security", "tinfoil_transport_security"):
            value = raw_policy.get(key)
            if isinstance(value, str) and value.strip().lower() == "ehbp":
                return False
    return True


def _public_tls_public_key_binding_satisfies_mode(
    claims: dict[str, Any],
    *,
    required: bool,
) -> bool:
    has_tls_claim = (
        "tls_public_key_fingerprint_sha256" in claims
        or "tls_public_key" in claims
    )
    if required or has_tls_claim:
        return _matching_public_sha256_alias(
            claims,
            "tls_public_key_fingerprint_sha256",
            "tls_public_key",
        )
    return True


def _public_ppq_private_policy_binds_proof(
    policy: dict[str, Any],
    proof_claims: dict[str, Any],
    model_ids: object,
) -> bool:
    selected = _public_selected_model_set(model_ids)
    if not selected or not all(model_id.startswith("private/") for model_id in selected):
        return False
    expected_repo = _public_policy_string(policy, "repo", "expected_repo")
    if expected_repo is None or proof_claims.get("repo") != expected_repo:
        return False
    bundle_pins = _public_ppq_attestation_bundle_policy_digests(policy)
    if not bundle_pins:
        return False
    if _public_sha256_digest(
        proof_claims.get("attestation_bundle_url_digest")
    ) not in bundle_pins:
        return False

    release_pins = _public_policy_digest_values(
        policy,
        *TINFOIL_PUBLIC_RELEASE_POLICY_KEYS,
    )
    code_measurement_pins = _public_policy_digest_values(
        policy,
        *TINFOIL_PUBLIC_CODE_MEASUREMENT_POLICY_KEYS,
    )
    enclave_measurement_pins = _public_policy_digest_values(
        policy,
        *TINFOIL_PUBLIC_ENCLAVE_MEASUREMENT_POLICY_KEYS,
    )
    if (
        release_pins is None
        or code_measurement_pins is None
        or enclave_measurement_pins is None
    ):
        return False
    if not release_pins and not code_measurement_pins:
        return False
    if release_pins and _public_sha256_digest(
        proof_claims.get("release_digest")
    ) not in release_pins:
        return False
    if code_measurement_pins and _public_sha256_digest(
        proof_claims.get("code_measurement_fingerprint")
    ) not in code_measurement_pins:
        return False
    if enclave_measurement_pins and _public_sha256_digest(
        proof_claims.get("enclave_measurement_fingerprint")
    ) not in enclave_measurement_pins:
        return False
    if (
        policy.get("require_model_attestations") is not True
        and policy.get("requireModelAttestations") is not True
    ):
        return False
    targets = _public_policy_targets(policy)
    if not isinstance(targets, dict) or set(targets) != selected:
        return False
    backend_attestations = proof_claims.get("backend_model_attestations")
    if not isinstance(backend_attestations, dict) or set(backend_attestations) != selected:
        return False
    for model_id in selected:
        target = targets.get(model_id)
        claims = backend_attestations.get(model_id)
        if not isinstance(target, dict) or not isinstance(claims, dict):
            return False
        if not _public_tinfoil_model_attestation_claims_match_target(target, claims):
            return False
    return True


def _public_privatemode_policy_binds_proof(
    policy: dict[str, Any],
    proof_claims: dict[str, Any],
    model_ids: object,
) -> bool:
    selected = _public_selected_model_set(model_ids)
    if not selected:
        return False
    if not _public_policy_has_digest_alias_pin(
        policy,
        "proxy_binary_digest",
        "proxyBinaryDigest",
    ):
        return False
    has_manifest_digest_pin = _public_policy_has_digest_alias_pin(
        policy,
        "manifest_digest",
        "manifestDigest",
    )
    manifest_log_dir = _public_policy_string(
        policy,
        "manifest_log_dir",
        "manifestLogDir",
    )
    if not has_manifest_digest_pin and manifest_log_dir is None:
        return False
    if has_manifest_digest_pin and not _public_policy_claim_matches_digest_aliases(
        policy,
        proof_claims,
        "manifest_digest",
        "manifest_digest",
        "manifestDigest",
    ):
        return False
    if manifest_log_dir is not None and (
        not is_full_sha256_digest(proof_claims.get("manifest_log_digest"))
        or not is_full_sha256_digest(
            proof_claims.get("manifest_log_manifest_digest")
        )
        or proof_claims.get("manifest_log_manifest_digest")
        != proof_claims.get("manifest_digest")
    ):
        return False
    if not _public_policy_claim_matches_digest_aliases(
        policy,
        proof_claims,
        "proxy_binary_digest",
        "proxy_binary_digest",
        "proxyBinaryDigest",
    ):
        return False
    expected_workload_identity_digest = (
        _public_privatemode_expected_workload_identity_digest(policy)
    )
    if (
        expected_workload_identity_digest is None
        or proof_claims.get("expected_workload_identity_digest")
        != expected_workload_identity_digest
    ):
        return False
    if (
        _public_policy_string(policy, "expected_trust_tier") != "app-e2ee"
        or proof_claims.get("trust_tier") != "app-e2ee"
    ):
        return False
    if not _public_privatemode_gpu_policy_matches(policy, proof_claims):
        return False
    bindings = _public_policy_model_workload_bindings(policy)
    if not isinstance(bindings, dict) or set(bindings) != selected:
        return False
    expected_workloads = _public_privatemode_expected_workload_sets(policy)
    if expected_workloads is None:
        return False
    expected_ids, expected_sans = expected_workloads
    bound_ids: set[str] = set()
    bound_sans: set[str] = set()
    normalized_bindings: dict[str, dict[str, list[str]]] = {}
    for model_id in selected:
        binding = bindings.get(model_id)
        if not isinstance(binding, dict):
            return False
        workload_ids = sorted(
            set(_public_string_list(binding.get("workload_ids")) or [])
            | set(_public_string_list(binding.get("workloadIDs")) or [])
        )
        workload_sans = sorted(
            set(_public_string_list(binding.get("workload_sans")) or [])
            | set(_public_string_list(binding.get("workloadSANs")) or [])
        )
        if not workload_ids and not workload_sans:
            return False
        binding_ids = set(workload_ids)
        binding_sans = set(workload_sans)
        if binding_ids - expected_ids or binding_sans - expected_sans:
            return False
        bound_ids.update(binding_ids)
        bound_sans.update(binding_sans)
        normalized_bindings[model_id] = {
            "workload_ids": workload_ids,
            "workload_sans": workload_sans,
        }
    if expected_ids - bound_ids or expected_sans - bound_sans:
        return False
    expected_binding_digest = _sha256_json_digest(
        {
            model_id: normalized_bindings[model_id]
            for model_id in sorted(selected)
        }
    )
    return proof_claims.get("model_workload_binding_digest") == expected_binding_digest


def public_confidentiality_policy_binds_provider_proof(
    provider_type: object,
    mode: object,
    public_policy: object,
    proof_claims: object,
    model_ids: object,
) -> bool:
    """Return whether public policy pins prove the public verifier claims."""
    provider = provider_type.strip().lower() if isinstance(provider_type, str) else ""
    normalized_mode = mode.strip().lower() if isinstance(mode, str) else ""
    if (
        provider not in {"tinfoil", "ppq-private", "privatemode"}
        and normalized_mode not in {"tinfoil", "ppq-private-tee", "privatemode"}
    ):
        return True
    if not isinstance(public_policy, dict) or not isinstance(proof_claims, dict):
        return False
    if provider == "tinfoil" or normalized_mode == "tinfoil":
        return _public_tinfoil_policy_binds_proof(
            public_policy,
            proof_claims,
            model_ids,
        )
    if provider == "ppq-private" or normalized_mode == "ppq-private-tee":
        return _public_ppq_private_policy_binds_proof(
            public_policy,
            proof_claims,
            model_ids,
        )
    if provider == "privatemode" or normalized_mode == "privatemode":
        return _public_privatemode_policy_binds_proof(
            public_policy,
            proof_claims,
            model_ids,
        )
    return False


def public_verified_model_selectors_satisfy_provider(
    provider_type: object,
    model_ids: object,
    model_id_prefixes: object,
) -> bool:
    """Return whether verified public metadata is scoped to exact provider models."""
    if not isinstance(provider_type, str):
        return False
    normalized_provider_type = provider_type.strip().lower()
    if normalized_provider_type not in PUBLIC_CONFIDENTIAL_PROVIDER_MODES:
        return True

    if not isinstance(model_ids, list) or not model_ids:
        return False
    if not all(isinstance(model_id, str) and model_id.strip() for model_id in model_ids):
        return False
    normalized_model_ids = [model_id.strip() for model_id in model_ids]
    if len({model_id.lower() for model_id in normalized_model_ids}) != len(
        normalized_model_ids
    ):
        return False
    if not isinstance(model_id_prefixes, list) or model_id_prefixes:
        return False
    if normalized_provider_type == "ppq-private":
        return all(model_id.startswith("private/") for model_id in normalized_model_ids)
    if normalized_provider_type == "privatemode":
        return all(
            model_id.startswith("privatemode/") for model_id in normalized_model_ids
        )
    return True


def public_provider_proof_claims_cover_model_selectors(
    provider_type: object,
    mode: object,
    proof_claims: object,
    model_ids: object,
    public_policy: object | None = None,
) -> bool:
    """Return whether public proof claims cover exact selected models."""
    provider = provider_type.strip().lower() if isinstance(provider_type, str) else ""
    normalized_mode = mode.strip().lower() if isinstance(mode, str) else ""
    if (
        provider not in {"tinfoil", "ppq-private", "privatemode"}
        and normalized_mode not in {"tinfoil", "ppq-private-tee", "privatemode"}
    ):
        return True
    if not isinstance(proof_claims, dict):
        return False
    if not isinstance(model_ids, list) or not model_ids:
        return False
    if not all(isinstance(model_id, str) and model_id.strip() for model_id in model_ids):
        return False
    selected_values = [model_id.strip() for model_id in model_ids]
    if len({model_id.lower() for model_id in selected_values}) != len(
        selected_values
    ):
        return False
    selected = set(selected_values)
    if (
        provider in {"ppq-private", "privatemode"}
        or normalized_mode in {"ppq-private-tee", "privatemode"}
    ):
        proof_selected_model_ids = proof_claims.get("selected_model_ids")
        if not isinstance(proof_selected_model_ids, list):
            return False
        if not all(
            isinstance(model_id, str) and model_id.strip()
            for model_id in proof_selected_model_ids
        ):
            return False
        proof_selected_values = [
            model_id.strip()
            for model_id in proof_selected_model_ids
        ]
        if len({model_id.lower() for model_id in proof_selected_values}) != len(
            proof_selected_values
        ):
            return False
        proof_selected = set(proof_selected_values)
        return proof_selected == selected

    model_attestations = proof_claims.get("model_attestations")
    if not isinstance(model_attestations, dict):
        return False
    if not all(
        isinstance(model_id, str) and model_id.strip() for model_id in model_attestations
    ):
        return False
    attested = {
        model_id.strip()
        for model_id in model_attestations
    }
    if len({model_id.lower() for model_id in attested}) != len(attested):
        return False
    if attested != selected:
        return False
    targets = _public_policy_targets(public_policy) if isinstance(public_policy, dict) else {}
    return all(
        _public_tinfoil_model_attestation_claims_satisfy_mode(
            model_attestations.get(model_id),
            public_policy=public_policy if isinstance(public_policy, dict) else None,
            target_policy=targets.get(model_id)
            if isinstance(targets.get(model_id), dict)
            else None,
        )
        for model_id in selected
    )


def _public_tinfoil_model_attestation_claims_satisfy_mode(
    claims: object,
    *,
    public_policy: dict[str, Any] | None = None,
    target_policy: dict[str, Any] | None = None,
) -> bool:
    if not isinstance(claims, dict):
        return False
    return bool(
        all(
            is_full_sha256_digest(claims.get(key))
            for key in REQUIRED_TINFOIL_MODEL_ATTESTATION_DIGEST_KEYS
        )
        and _public_tls_public_key_binding_satisfies_mode(
            claims,
            required=_public_tinfoil_tls_public_key_binding_required(
                public_policy,
                target_policy=target_policy,
            ),
        )
        and all(
            _public_hex64(claims.get(key)) is not None
            for key in PUBLIC_TINFOIL_MODEL_ATTESTATION_HEX64_KEYS
        )
        and all(
            isinstance(claims.get(key), str) and bool(str(claims.get(key)).strip())
            for key in PUBLIC_TINFOIL_MODEL_ATTESTATION_STRING_KEYS
        )
        and _has_required_steps(
            claims,
            REQUIRED_EHBP_PROVIDER_VERIFICATION_STEPS,
        )
    )


def is_full_sha256_digest(value: object) -> bool:
    """Return True for a concrete prefixed SHA-256 digest."""
    if not isinstance(value, str):
        return False
    digest = value.strip()
    return bool(FULL_SHA256_DIGEST_RE.fullmatch(digest)) and not (
        _is_template_placeholder_digest_value(digest)
    )


def _public_sha256_digest(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    digest = value.strip().lower()
    if _is_template_placeholder_digest_value(digest):
        return None
    if FULL_SHA256_DIGEST_RE.fullmatch(digest):
        return digest
    if RAW_SHA256_DIGEST_RE.fullmatch(digest):
        return f"sha256:{digest}"
    return None


def _is_template_placeholder_digest_value(value: str) -> bool:
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    return bool(digest) and len(digest) in {64, 96} and len(set(digest)) == 1


def _matching_public_sha256_alias(claims: dict[str, Any], *keys: str) -> bool:
    normalized_values: list[str] = []
    for key in keys:
        if key not in claims:
            continue
        value = _public_sha256_digest(claims.get(key))
        if value is None:
            return False
        normalized_values.append(value)
    return bool(normalized_values) and len(set(normalized_values)) == 1


def _sha256_json_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


RUNTIME_PROVIDER_EVIDENCE_VOLATILE_CLAIMS = {
    "payload_policy_digest",
    "payload_evidence_digest",
    "payload_verification_nonce",
    "runtime_evidence_digest",
}


def provider_evidence_digest_matches_status(
    status_evidence_digest: object,
    claims: object,
    proof_claims: object,
) -> bool:
    """Return True when status evidence digest is bound to verifier claims."""
    if not is_full_sha256_digest(status_evidence_digest):
        return False
    if not isinstance(proof_claims, dict):
        return False

    if isinstance(claims, dict) and "runtime_evidence_digest" in claims:
        runtime_evidence_digest = claims.get("runtime_evidence_digest")
        if runtime_evidence_digest != status_evidence_digest:
            return False
        try:
            recomputed_digest = _sha256_json_digest(
                {
                    key: value
                    for key, value in claims.items()
                    if key not in RUNTIME_PROVIDER_EVIDENCE_VOLATILE_CLAIMS
                }
            )
        except (TypeError, ValueError):
            return False
        return recomputed_digest == status_evidence_digest

    return proof_claims.get("payload_evidence_digest") == status_evidence_digest


def _privatemode_key_release_binding_matches(claims: dict[str, Any]) -> bool:
    expected = _sha256_json_digest(
        {key: claims.get(key) for key in PRIVATEMODE_KEY_RELEASE_BINDING_CLAIMS}
    )
    return claims.get("key_release_binding") == expected


def _public_hex64(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    hex_value = value.strip().lower()
    if HEX_64_RE.fullmatch(hex_value):
        return hex_value
    return None


def _public_string_list(value: object) -> list[str] | None:
    if not isinstance(value, list) or not value:
        return None
    strings: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return None
        strings.append(item.strip())
    if len({item.lower() for item in strings}) != len(strings):
        return None
    return strings


def _public_verification_steps(value: object) -> dict[str, bool] | None:
    if not isinstance(value, dict):
        return None
    public_steps = {
        step: bool(value[step])
        for step in sorted(PUBLIC_PROVIDER_VERIFICATION_STEP_KEYS)
        if isinstance(value.get(step), bool)
    }
    return public_steps or None


def _public_verification_steps_source_is_well_formed(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return all(isinstance(step, str) and step.strip() for step in value) and all(
        isinstance(step_value, bool) for step_value in value.values()
    )


def _public_tinfoil_model_attestation_claims(
    claims: object,
) -> dict[str, Any] | None:
    if not isinstance(claims, dict) or not claims:
        return None
    if not _public_verification_steps_source_is_well_formed(
        claims.get("verification_steps")
    ):
        return None

    public_claims: dict[str, Any] = {}
    for key in sorted(PUBLIC_TINFOIL_MODEL_ATTESTATION_KEYS):
        value = claims.get(key)
        if key in PUBLIC_TINFOIL_MODEL_ATTESTATION_DIGEST_KEYS:
            public_value = _public_sha256_digest(value)
        elif key in PUBLIC_TINFOIL_MODEL_ATTESTATION_HEX64_KEYS:
            public_value = _public_hex64(value)
        elif key in PUBLIC_TINFOIL_MODEL_ATTESTATION_STRING_KEYS:
            public_value = (
                value.strip() if isinstance(value, str) and value.strip() else None
            )
        else:
            public_value = None
        if public_value:
            public_claims[key] = public_value

    public_steps = _public_verification_steps(claims.get("verification_steps"))
    if public_steps:
        public_claims["verification_steps"] = public_steps

    return public_claims or None


def _public_tinfoil_model_attestations(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not value:
        return None
    public_attestations: dict[str, Any] = {}
    seen_model_ids: set[str] = set()
    for raw_model_id, raw_claims in value.items():
        if not isinstance(raw_model_id, str) or not raw_model_id.strip():
            return None
        model_id = raw_model_id.strip()
        normalized_model_id = model_id.lower()
        if normalized_model_id in seen_model_ids:
            return None
        seen_model_ids.add(normalized_model_id)
        public_claims = _public_tinfoil_model_attestation_claims(raw_claims)
        if public_claims is None:
            return None
        public_attestations[model_id] = public_claims
    return dict(sorted(public_attestations.items()))


def _tinfoil_model_attestation_source_is_well_formed(value: object) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    seen_model_ids: set[str] = set()
    for raw_model_id, raw_claims in value.items():
        if not isinstance(raw_model_id, str) or not raw_model_id.strip():
            return False
        model_id = raw_model_id.strip()
        normalized_model_id = model_id.lower()
        if normalized_model_id in seen_model_ids:
            return False
        seen_model_ids.add(normalized_model_id)
        if not isinstance(raw_claims, dict) or not raw_claims:
            return False
        for key in PUBLIC_TINFOIL_MODEL_ATTESTATION_KEYS:
            if key not in raw_claims:
                continue
            claim_value = raw_claims.get(key)
            if key in PUBLIC_TINFOIL_MODEL_ATTESTATION_DIGEST_KEYS:
                if _public_sha256_digest(claim_value) is None:
                    return False
            elif key in PUBLIC_TINFOIL_MODEL_ATTESTATION_HEX64_KEYS:
                if _public_hex64(claim_value) is None:
                    return False
            elif key in PUBLIC_TINFOIL_MODEL_ATTESTATION_STRING_KEYS:
                if not (isinstance(claim_value, str) and claim_value.strip()):
                    return False
        verification_steps = raw_claims.get("verification_steps")
        if verification_steps is not None:
            if not _public_verification_steps_source_is_well_formed(
                verification_steps
            ):
                return False
    return True


def public_provider_proof_claims(claims: object) -> dict[str, Any] | None:
    """Return non-secret provider proof claims for public routing surfaces."""
    if not isinstance(claims, dict) or not claims:
        return None

    proof_claims: dict[str, Any] = {}
    for key in sorted(PUBLIC_PROVIDER_PROOF_CLAIM_KEYS):
        value = claims.get(key)
        if key in PUBLIC_PROVIDER_DIGEST_PROOF_CLAIM_KEYS:
            public_value = _public_sha256_digest(value)
        elif key in PUBLIC_PROVIDER_HEX64_PROOF_CLAIM_KEYS:
            public_value = _public_hex64(value)
        elif key == "transport":
            public_value = (
                value.strip().lower()
                if isinstance(value, str)
                and value.strip().lower() in PUBLIC_PROVIDER_TRANSPORT_VALUES
                else None
            )
        elif key in PUBLIC_PROVIDER_STRING_PROOF_CLAIM_KEYS:
            public_value = (
                value.strip() if isinstance(value, str) and value.strip() else None
            )
        elif key in PUBLIC_PROVIDER_STRING_LIST_PROOF_CLAIM_KEYS:
            public_value = _public_string_list(value)
        else:
            public_value = None
        if public_value:
            proof_claims[key] = public_value

    verification_steps = claims.get("verification_steps")
    if isinstance(verification_steps, dict):
        public_steps = _public_verification_steps(verification_steps)
        if public_steps:
            proof_claims["verification_steps"] = public_steps

    if "model_attestations" in claims:
        model_attestations = _public_tinfoil_model_attestations(
            claims.get("model_attestations")
        )
        if model_attestations is None:
            return None
        proof_claims["model_attestations"] = model_attestations
    if "backend_model_attestations" in claims:
        backend_model_attestations = _public_tinfoil_model_attestations(
            claims.get("backend_model_attestations")
        )
        if backend_model_attestations is None:
            return None
        proof_claims["backend_model_attestations"] = backend_model_attestations

    return proof_claims or None


def _public_provider_proof_source_is_well_formed(
    mode: object,
    claims: object,
) -> bool:
    if not isinstance(claims, dict) or not claims:
        return False
    normalized_mode = mode.strip().lower() if isinstance(mode, str) else ""
    ignored_claims = (
        {"proxy_binary_digest"} if normalized_mode == "ppq-private-tee" else set()
    )
    public_source = {
        key: claims[key]
        for key in PUBLIC_PROVIDER_PROOF_CLAIM_KEYS
        if key in claims and key not in ignored_claims
    }
    for key in (
        "verification_steps",
        "model_attestations",
        "backend_model_attestations",
    ):
        if key in claims:
            public_source[key] = claims[key]
    if inline_policy_secret_violations(
        public_source,
        policy_name="provider proof claims",
    ):
        return False

    for key in PUBLIC_PROVIDER_PROOF_CLAIM_KEYS:
        if key in ignored_claims:
            continue
        if key not in claims:
            continue
        value = claims.get(key)
        if key in PUBLIC_PROVIDER_DIGEST_PROOF_CLAIM_KEYS:
            if _public_sha256_digest(value) is None:
                return False
        elif key in PUBLIC_PROVIDER_HEX64_PROOF_CLAIM_KEYS:
            if _public_hex64(value) is None:
                return False
        elif key == "transport":
            if not (
                isinstance(value, str)
                and value.strip().lower() in PUBLIC_PROVIDER_TRANSPORT_VALUES
            ):
                return False
        elif key in PUBLIC_PROVIDER_STRING_PROOF_CLAIM_KEYS:
            if not (isinstance(value, str) and value.strip()):
                return False
        elif key in PUBLIC_PROVIDER_STRING_LIST_PROOF_CLAIM_KEYS:
            if _public_string_list(value) is None:
                return False

    verification_steps = claims.get("verification_steps")
    if verification_steps is not None:
        if not _public_verification_steps_source_is_well_formed(verification_steps):
            return False

    model_attestations = claims.get("model_attestations")
    if model_attestations is not None and not _tinfoil_model_attestation_source_is_well_formed(
        model_attestations
    ):
        return False
    backend_model_attestations = claims.get("backend_model_attestations")
    if (
        backend_model_attestations is not None
        and not _tinfoil_model_attestation_source_is_well_formed(
            backend_model_attestations
        )
    ):
        return False

    return True


def verified_public_provider_proof_claims(
    mode: object,
    claims: object,
    public_policy: object | None = None,
    reject_secret_source: bool = False,
) -> dict[str, Any] | None:
    """Return sanitized proof claims only when the source is valid for verified use."""
    if reject_secret_source and isinstance(claims, dict):
        if inline_policy_secret_violations(
            claims,
            policy_name="provider proof claims",
        ):
            return None
    if not _public_provider_proof_source_is_well_formed(mode, claims):
        return None
    proof_claims = public_provider_proof_claims(claims)
    if isinstance(mode, str) and mode.strip().lower() == "ppq-private-tee":
        if isinstance(proof_claims, dict):
            proof_claims.pop("proxy_binary_digest", None)
    if public_provider_proof_claims_satisfy_mode(
        mode,
        proof_claims,
        public_policy=public_policy,
    ):
        return proof_claims
    return None


def _has_required_steps(claims: dict[str, Any], steps: tuple[str, ...]) -> bool:
    verification_steps = claims.get("verification_steps")
    if not isinstance(verification_steps, dict):
        return False
    return all(verification_steps.get(step) is True for step in steps)


def _has_required_public_digest_claims(
    claims: dict[str, Any],
    digest_claims: tuple[str, ...],
) -> bool:
    return all(is_full_sha256_digest(claims.get(claim)) for claim in digest_claims)


def _strict_public_non_empty_string_list(value: object) -> list[str] | None:
    if not isinstance(value, list) or not value:
        return None
    values: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return None
        values.append(item.strip())
    if len({value.lower() for value in values}) != len(values):
        return None
    return values


def public_provider_type_satisfies_mode(
    provider_type: object,
    mode: object,
) -> bool:
    if not isinstance(provider_type, str) or not isinstance(mode, str):
        return False
    return (
        PUBLIC_CONFIDENTIAL_PROVIDER_MODES.get(provider_type.strip().lower())
        == mode.strip().lower()
    )


def public_provider_proof_claims_satisfy_mode(
    mode: object,
    proof_claims: object,
    public_policy: object | None = None,
) -> bool:
    """Return whether sanitized public proof claims are sufficient for a mode."""
    if not isinstance(mode, str) or not isinstance(proof_claims, dict):
        return False
    normalized_mode = mode.strip().lower()
    if not normalized_mode or not proof_claims:
        return False

    if normalized_mode in {"tinfoil", "ppq-private-tee"}:
        tls_binding_required = True
        if normalized_mode == "tinfoil" and isinstance(public_policy, dict):
            tls_binding_required = _public_tinfoil_tls_public_key_binding_required(
                public_policy,
            )
        ehbp_proof = (
            proof_claims.get("transport") == "ehbp"
            and _has_required_public_digest_claims(
                proof_claims,
                REQUIRED_PROVIDER_PAYLOAD_BINDING_CLAIMS,
            )
            and isinstance(proof_claims.get("repo"), str)
            and bool(str(proof_claims.get("repo")).strip())
            and _public_hex64(proof_claims.get("attested_hpke_public_key_hex"))
            is not None
            and is_full_sha256_digest(
                proof_claims.get("enclave_measurement_fingerprint")
            )
            and is_full_sha256_digest(proof_claims.get("code_measurement_fingerprint"))
            and _public_tls_public_key_binding_satisfies_mode(
                proof_claims,
                required=tls_binding_required,
            )
            and _has_required_steps(
                proof_claims,
                REQUIRED_EHBP_PROVIDER_VERIFICATION_STEPS,
            )
        )
        if not ehbp_proof:
            return False
        if normalized_mode == "ppq-private-tee":
            selected_model_ids = _strict_public_non_empty_string_list(
                proof_claims.get("selected_model_ids")
            )
            backend_model_attestations = proof_claims.get("backend_model_attestations")
            return bool(
                proof_claims.get("client_encryption_boundary")
                == "routstr-tee-ehbp-proxy"
                and is_full_sha256_digest(
                    proof_claims.get("attestation_bundle_url_digest")
                )
                and selected_model_ids
                and all(
                    model_id.startswith("private/")
                    for model_id in selected_model_ids
                )
                and isinstance(backend_model_attestations, dict)
                and set(backend_model_attestations)
                == set(selected_model_ids)
                and all(
                    _public_tinfoil_model_attestation_claims_satisfy_mode(
                        backend_model_attestations.get(model_id)
                    )
                    for model_id in selected_model_ids
                )
            )

        return bool(
            isinstance(proof_claims.get("attestation_format"), str)
            and str(proof_claims.get("attestation_format")).strip()
            and is_full_sha256_digest(proof_claims.get("attestation_report_digest"))
            and is_full_sha256_digest(proof_claims.get("release_digest"))
        )

    if normalized_mode == "privatemode":
        selected_model_ids = _strict_public_non_empty_string_list(
            proof_claims.get("selected_model_ids")
        )
        return bool(
            proof_claims.get("transport") == "privatemode-proxy"
            and proof_claims.get("trust_tier") == "app-e2ee"
            and _has_required_public_digest_claims(
                proof_claims,
                REQUIRED_PROVIDER_PAYLOAD_BINDING_CLAIMS,
            )
            and _matching_public_sha256_alias(
                proof_claims,
                "manifest_digest",
                "manifest_log_manifest_digest",
            )
            and is_full_sha256_digest(proof_claims.get("proxy_binary_digest"))
            and (
                "proxy_image_digest" not in proof_claims
                or is_full_sha256_digest(proof_claims.get("proxy_image_digest"))
            )
            and is_full_sha256_digest(proof_claims.get("coordinator_measurement"))
            and is_full_sha256_digest(proof_claims.get("secret_service_measurement"))
            and is_full_sha256_digest(proof_claims.get("ai_worker_measurement"))
            and proof_claims.get("ai_worker_measurement")
            == proof_claims.get("attested_workload_policy_digest")
            and proof_claims.get("gpu_attestation_policy")
            == PRIVATEMODE_GPU_ATTESTATION_POLICY
            and is_full_sha256_digest(proof_claims.get("key_release_binding"))
            and _privatemode_key_release_binding_matches(proof_claims)
            and selected_model_ids
            and all(
                model_id.startswith("privatemode/")
                for model_id in selected_model_ids
            )
            and all(
                is_full_sha256_digest(proof_claims.get(claim))
                for claim in REQUIRED_PRIVATEMODE_PUBLIC_DIGEST_CLAIMS
            )
            and _has_required_steps(
                proof_claims,
                REQUIRED_PRIVATEMODE_PROVIDER_VERIFICATION_STEPS,
            )
        )

    return False
