#!/usr/bin/env python3
"""
Listing: Routstr Provider Discoverability Implementation
Automatically announces this Routstr proxy instance to Nostr relays.
"""

import asyncio
import json
import os
import random
import ssl
import time
from typing import Any, cast

from nostr.event import Event
from nostr.filter import Filter, Filters
from nostr.key import PrivateKey
from nostr.message_type import ClientMessageType
from nostr.relay_manager import RelayManager

from ..core import get_logger
from ..core.confidentiality_public import (
    is_full_sha256_digest,
    public_confidentiality_policy_binds_provider_proof,
    public_provider_proof_claims_cover_model_selectors,
    public_provider_type_satisfies_mode,
    public_verified_model_selectors_satisfy_provider,
    verified_public_provider_proof_claims,
)
from ..core.policy_secrets import inline_policy_secret_violations
from ..core.settings import settings

logger = get_logger(__name__)


LISTING_PUBLIC_CONFIDENTIALITY_POLICY_KEYS = {
    "allowed_code_measurement_fingerprint",
    "allowed_code_measurement_fingerprints",
    "allowed_code_measurements",
    "allowed_enclave_measurement_fingerprint",
    "allowed_enclave_measurement_fingerprints",
    "allowed_enclave_measurements",
    "allowed_release_digest",
    "allowed_release_digests",
    "allowed_ai_worker_measurements",
    "allowed_coordinator_measurements",
    "allowed_gpu_attestation_policies",
    "allowed_key_release_bindings",
    "allowed_secret_service_measurements",
    "attestation_bundle_url_digest",
    "code_measurement_fingerprint",
    "enclave_measurement_fingerprint",
    "expectedWorkloadIDs",
    "expectedWorkloadSANs",
    "expected_ai_worker_measurement",
    "expected_code_measurement_fingerprint",
    "expected_coordinator_measurement",
    "expected_enclave_measurement_fingerprint",
    "expected_gpu_attestation_policy",
    "expected_key_release_binding",
    "expected_release_digest",
    "expected_repo",
    "expected_secret_service_measurement",
    "expected_workload_ids",
    "expected_workload_identity_digest",
    "expected_workload_sans",
    "manifestDigest",
    "manifest_digest",
    "manifestLogDir",
    "manifest_log_dir",
    "modelAttestationTargets",
    "modelEnclaveBindings",
    "modelWorkloadBindings",
    "model_attestation_targets",
    "model_enclave_bindings",
    "model_workload_binding_digest",
    "model_workload_bindings",
    "proxyBinaryDigest",
    "proxyImageDigest",
    "proxy_binary_digest",
    "proxy_image_digest",
    "release_digest",
    "repo",
    "requireModelAttestations",
    "require_model_attestations",
    "tinfoil_transport_security",
    "transport_security",
}
LISTING_PUBLIC_TINFOIL_TARGET_POLICY_KEYS = {
    "allowed_code_measurement_fingerprint",
    "allowed_code_measurement_fingerprints",
    "allowed_code_measurements",
    "allowed_enclave_measurement_fingerprint",
    "allowed_enclave_measurement_fingerprints",
    "allowed_enclave_measurements",
    "allowed_release_digest",
    "allowed_release_digests",
    "code_measurement_fingerprint",
    "enclave_measurement_fingerprint",
    "expected_code_measurement_fingerprint",
    "expected_enclave_host",
    "expected_enclave_measurement_fingerprint",
    "expected_release_digest",
    "expected_repo",
    "host",
    "hpke_keys_url",
    "release_digest",
    "repo",
    "tinfoil_transport_security",
    "transport_security",
}
LISTING_PUBLIC_PRIVATEMODE_WORKLOAD_BINDING_POLICY_KEYS = {
    "workloadIDs",
    "workloadSANs",
    "workload_ids",
    "workload_sans",
}
LISTING_PUBLIC_BOOLEAN_CONFIDENTIALITY_POLICY_KEYS = {
    "requireModelAttestations",
    "require_model_attestations",
}
LISTING_PUBLIC_DIGEST_CONFIDENTIALITY_POLICY_KEYS = {
    "attestation_bundle_url_digest",
    "code_measurement_fingerprint",
    "enclave_measurement_fingerprint",
    "expected_ai_worker_measurement",
    "expected_code_measurement_fingerprint",
    "expected_coordinator_measurement",
    "expected_enclave_measurement_fingerprint",
    "expected_key_release_binding",
    "expected_release_digest",
    "expected_secret_service_measurement",
    "expected_workload_identity_digest",
    "manifestDigest",
    "manifest_digest",
    "model_workload_binding_digest",
    "proxyBinaryDigest",
    "proxyImageDigest",
    "proxy_binary_digest",
    "proxy_image_digest",
    "release_digest",
}
LISTING_PUBLIC_DIGEST_LIST_CONFIDENTIALITY_POLICY_KEYS = {
    "allowed_ai_worker_measurements",
    "allowed_code_measurement_fingerprint",
    "allowed_code_measurement_fingerprints",
    "allowed_code_measurements",
    "allowed_coordinator_measurements",
    "allowed_enclave_measurement_fingerprint",
    "allowed_enclave_measurement_fingerprints",
    "allowed_enclave_measurements",
    "allowed_key_release_bindings",
    "allowed_release_digest",
    "allowed_release_digests",
    "allowed_secret_service_measurements",
}
LISTING_PUBLIC_DIGEST_TINFOIL_TARGET_POLICY_KEYS = {
    "code_measurement_fingerprint",
    "enclave_measurement_fingerprint",
    "expected_code_measurement_fingerprint",
    "expected_enclave_measurement_fingerprint",
    "expected_release_digest",
    "release_digest",
}
LISTING_PUBLIC_DIGEST_LIST_TINFOIL_TARGET_POLICY_KEYS = {
    "allowed_code_measurement_fingerprint",
    "allowed_code_measurement_fingerprints",
    "allowed_code_measurements",
    "allowed_enclave_measurement_fingerprint",
    "allowed_enclave_measurement_fingerprints",
    "allowed_enclave_measurements",
    "allowed_release_digest",
    "allowed_release_digests",
}
LISTING_ROUTABLE_FULL_ATTESTATION_PROVIDERS = (
    "tinfoil",
    "ppq-private",
    "privatemode",
)


def get_app_version() -> str | None:
    try:
        from ..core.version import __version__ as imported_version

        return imported_version
    except Exception:
        return None


def _event_to_dict(ev: Event) -> dict[str, Any]:
    return {
        "id": ev.id,
        "pubkey": ev.public_key,
        "created_at": ev.created_at,
        "kind": int(ev.kind) if not isinstance(ev.kind, int) else ev.kind,
        "tags": ev.tags,
        "content": ev.content,
        "sig": ev.signature,
    }


def nsec_to_keypair(nsec: str) -> tuple[str, str] | None:
    """
    Convert a Nostr private key (nsec) to a keypair (privkey_hex, pubkey_hex).

    Args:
        nsec: Nostr private key in nsec format or hex format

    Returns:
        Tuple of (private_key_hex, public_key_hex) or None if invalid
    """
    try:
        if nsec.startswith("nsec"):
            pk = PrivateKey.from_nsec(nsec)
            return (pk.hex(), pk.public_key.hex())

        if len(nsec) == 64:
            pk = PrivateKey(bytes.fromhex(nsec))
            return (pk.hex(), pk.public_key.hex())

        logger.error(f"Invalid private key format/length: {len(nsec)}")
        return None
    except Exception as e:
        logger.error(f"Failed to convert nsec to keypair: {e}")
        return None


def create_listing_event(
    private_key_hex: str,
    provider_id: str,
    endpoint_urls: list[str],
    mint_urls: list[str] | None = None,
    version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a listing provider announcement event (kind:38421).

    Args:
        private_key_hex: 32-byte hex private key for signing
        provider_id: Unique identifier for this provider (d tag)
        endpoint_urls: List of URLs to connect to the provider
        mint_urls: Optional list of ecash mint URLs for payments
        version: Provider software version
        metadata: Optional metadata dictionary (name, picture, about, etc.)

    Returns:
        Complete signed nostr event as a dict ready for publishing
    """
    pk = PrivateKey(bytes.fromhex(private_key_hex))

    tags = [["d", provider_id]]
    for url in endpoint_urls:
        tags.append(["u", url])
    if mint_urls:
        for m in mint_urls:
            if m:
                tags.append(["mint", m])
    if version:
        tags.append(["version", version])

    content = json.dumps(metadata, separators=(",", ":")) if metadata else ""

    ev = Event(pk.public_key.hex(), content, kind=38421, tags=tags)
    pk.sign_event(ev)
    return _event_to_dict(ev)


def _safe_listing_confidentiality_status(
    confidentiality: dict[str, Any],
    *,
    provider_type: str,
    public_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mode = _public_string(confidentiality.get("mode")) or "none"
    strict_model_ids = _strict_public_string_list(confidentiality.get("model_ids"))
    strict_model_id_prefixes = _strict_public_string_list(
        confidentiality.get("model_id_prefixes")
    )
    public_status = {
        "enabled": confidentiality.get("enabled", False) is True,
        "verified": False,
        "mode": mode,
        "model_ids": _public_string_list(confidentiality.get("model_ids")),
        "model_id_prefixes": _public_string_list(
            confidentiality.get("model_id_prefixes")
        ),
    }

    verified = confidentiality.get("verified", False) is True
    verifier = _public_string(confidentiality.get("verifier"))
    policy_digest = confidentiality.get("policy_digest")
    evidence_digest = confidentiality.get("evidence_digest")
    verified_at = _public_int(confidentiality.get("verified_at"))
    expires_at = _public_int(confidentiality.get("expires_at"))

    proof_claim_source = confidentiality.get("proof_claims")
    if not isinstance(proof_claim_source, dict):
        proof_claim_source = confidentiality.get("verified_claims")
    proof_claims = verified_public_provider_proof_claims(
        mode,
        proof_claim_source,
        public_policy=public_policy,
    )
    now = int(time.time())

    if (
        verified
        and verifier
        and is_full_sha256_digest(policy_digest)
        and is_full_sha256_digest(evidence_digest)
        and verified_at is not None
        and expires_at is not None
        and verified_at <= now
        and expires_at > now
        and public_provider_type_satisfies_mode(provider_type, mode)
        and proof_claims is not None
        and proof_claims.get("payload_policy_digest") == policy_digest
        and proof_claims.get("payload_evidence_digest") == evidence_digest
        and strict_model_ids is not None
        and strict_model_id_prefixes is not None
        and public_verified_model_selectors_satisfy_provider(
            provider_type,
            strict_model_ids,
            strict_model_id_prefixes,
        )
        and public_provider_proof_claims_cover_model_selectors(
            provider_type,
            mode,
            proof_claims,
            strict_model_ids,
            public_policy=public_policy,
        )
        and public_confidentiality_policy_binds_provider_proof(
            provider_type,
            mode,
            public_policy,
            proof_claims,
            strict_model_ids,
        )
    ):
        public_status.update(
            {
                "verified": True,
                "verifier": verifier,
                "policy_digest": policy_digest,
                "evidence_digest": evidence_digest,
                "verified_at": verified_at,
                "expires_at": expires_at,
            }
        )
        public_status["proof_claims"] = proof_claims
    return public_status


def _safe_listing_confidentiality_policy(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not value:
        return None
    if inline_policy_secret_violations(value):
        return None

    public: dict[str, Any] = {}
    invalid_policy_value = object()

    def public_policy_value(raw_value: object, *, allow_bool: bool = True) -> object:
        if isinstance(raw_value, str):
            return raw_value
        if allow_bool and isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, list) and all(
            isinstance(list_item, str) for list_item in raw_value
        ):
            return list(raw_value)
        return invalid_policy_value

    def public_digest_policy_value(raw_value: object) -> object:
        if isinstance(raw_value, str) and is_full_sha256_digest(raw_value):
            return raw_value
        return invalid_policy_value

    def public_digest_list_policy_value(raw_value: object) -> object:
        if isinstance(raw_value, str) and is_full_sha256_digest(raw_value):
            return raw_value
        if isinstance(raw_value, list) and all(
            isinstance(list_item, str) and is_full_sha256_digest(list_item)
            for list_item in raw_value
        ):
            return list(raw_value)
        return invalid_policy_value

    for key in sorted(LISTING_PUBLIC_CONFIDENTIALITY_POLICY_KEYS):
        item = value.get(key)
        if item is None:
            continue
        if key in {
            "model_attestation_targets",
            "modelAttestationTargets",
            "model_enclave_bindings",
            "modelEnclaveBindings",
        }:
            if not isinstance(item, dict):
                return None
            public_targets: dict[str, dict[str, Any]] = {}
            seen_model_ids: set[str] = set()
            for raw_model_id, raw_target in item.items():
                if not isinstance(raw_model_id, str) or not raw_model_id.strip():
                    return None
                model_id = raw_model_id.strip()
                normalized_model_id = model_id.lower()
                if normalized_model_id in seen_model_ids:
                    return None
                seen_model_ids.add(normalized_model_id)
                if not isinstance(raw_target, dict):
                    return None
                public_target = {}
                for target_key in sorted(LISTING_PUBLIC_TINFOIL_TARGET_POLICY_KEYS):
                    if target_key not in raw_target:
                        continue
                    if target_key in LISTING_PUBLIC_DIGEST_TINFOIL_TARGET_POLICY_KEYS:
                        public_value = public_digest_policy_value(
                            raw_target[target_key]
                        )
                    elif (
                        target_key
                        in LISTING_PUBLIC_DIGEST_LIST_TINFOIL_TARGET_POLICY_KEYS
                    ):
                        public_value = public_digest_list_policy_value(
                            raw_target[target_key]
                        )
                    else:
                        public_value = public_policy_value(
                            raw_target[target_key],
                            allow_bool=False,
                        )
                    if public_value is invalid_policy_value:
                        return None
                    public_target[target_key] = public_value
                if not public_target:
                    return None
                public_targets[model_id] = public_target
            public[key] = dict(sorted(public_targets.items()))
        elif key in {"model_workload_bindings", "modelWorkloadBindings"}:
            if not isinstance(item, dict):
                return None
            public_bindings: dict[str, dict[str, Any]] = {}
            seen_model_ids: set[str] = set()
            for raw_model_id, raw_binding in item.items():
                if not isinstance(raw_model_id, str) or not raw_model_id.strip():
                    return None
                model_id = raw_model_id.strip()
                normalized_model_id = model_id.lower()
                if normalized_model_id in seen_model_ids:
                    return None
                seen_model_ids.add(normalized_model_id)
                if not isinstance(raw_binding, dict):
                    return None
                public_binding = {}
                for binding_key in sorted(
                    LISTING_PUBLIC_PRIVATEMODE_WORKLOAD_BINDING_POLICY_KEYS
                ):
                    if binding_key not in raw_binding:
                        continue
                    public_value = public_policy_value(
                        raw_binding[binding_key],
                        allow_bool=False,
                    )
                    if public_value is invalid_policy_value:
                        return None
                    public_binding[binding_key] = public_value
                if not public_binding:
                    return None
                public_bindings[model_id] = public_binding
            public[key] = dict(sorted(public_bindings.items()))
        else:
            if key in LISTING_PUBLIC_DIGEST_CONFIDENTIALITY_POLICY_KEYS:
                public_value = public_digest_policy_value(item)
            elif key in LISTING_PUBLIC_DIGEST_LIST_CONFIDENTIALITY_POLICY_KEYS:
                public_value = public_digest_list_policy_value(item)
            else:
                public_value = public_policy_value(
                    item,
                    allow_bool=key
                    in LISTING_PUBLIC_BOOLEAN_CONFIDENTIALITY_POLICY_KEYS,
                )
            if public_value is invalid_policy_value:
                return None
            public[key] = public_value
    return public or None


def _public_string_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _strict_public_string_list(value: object) -> list[str] | None:
    if value is None:
        return []
    if not isinstance(value, list):
        return None
    values: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return None
        values.append(item.strip())
    if len({value.lower() for value in values}) != len(values):
        return None
    return values


def _public_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _public_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _safe_listing_client_confidentiality(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if value.get("mode") != "attested-tls-termination":
        return None
    if value.get("tls_terminates_in_attested_tee") is not True:
        return None
    if value.get("inbound_ehbp_ohttp_request_decryption") is not False:
        return None
    attested_tls_public_key_digest = value.get("attested_tls_public_key_digest")
    if not is_full_sha256_digest(attested_tls_public_key_digest):
        return None
    return {
        "mode": "attested-tls-termination",
        "tls_terminates_in_attested_tee": True,
        "inbound_ehbp_ohttp_request_decryption": False,
        "attested_tls_public_key_digest": attested_tls_public_key_digest,
    }


def _local_tee_status_has_listing_proof(
    value: dict[str, Any],
    *,
    client_confidentiality: dict[str, Any],
) -> bool:
    attestation_evidence_digest = value.get("attestation_evidence_digest")
    hpke_key_config_digest = value.get("hpke_key_config_digest")
    hpke_public_key_digest = value.get("hpke_public_key_digest")
    if not (
        is_full_sha256_digest(attestation_evidence_digest)
        and is_full_sha256_digest(hpke_key_config_digest)
        and is_full_sha256_digest(hpke_public_key_digest)
    ):
        return False

    local_verification = value.get("local_verification")
    if not isinstance(local_verification, dict):
        return False
    if local_verification.get("verified") is not True:
        return False
    verified_at = local_verification.get("verified_at")
    expires_at = local_verification.get("expires_at")
    if type(verified_at) is not int or type(expires_at) is not int:
        return False
    now = int(time.time())
    if verified_at > now or expires_at <= now:
        return False
    if local_verification.get("evidence_digest") != attestation_evidence_digest:
        return False
    if not is_full_sha256_digest(local_verification.get("verified_claims_digest")):
        return False

    proof_claims = local_verification.get("proof_claims")
    if not isinstance(proof_claims, dict):
        return False
    if proof_claims.get("hpke_key_config_digest") != hpke_key_config_digest:
        return False
    if proof_claims.get("hpke_public_key_digest") != hpke_public_key_digest:
        return False
    if proof_claims.get("public_key_digest") != client_confidentiality.get(
        "attested_tls_public_key_digest"
    ):
        return False
    return True


def _safe_listing_routstr_tee_status(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    public_status: dict[str, Any] = {
        "required": value.get("required", False) is True,
        "ready": value.get("ready", False) is True,
    }
    client_confidentiality = _safe_listing_client_confidentiality(
        value.get("client_confidentiality")
    )
    if client_confidentiality is not None:
        public_status["client_confidentiality"] = client_confidentiality
    elif public_status["ready"] is True:
        public_status["ready"] = False
    if public_status["ready"] is True and (
        client_confidentiality is None
        or not _local_tee_status_has_listing_proof(
            value,
            client_confidentiality=client_confidentiality,
        )
    ):
        public_status["ready"] = False
    return public_status


def _empty_listing_routable_with_full_attestation() -> dict[str, list[str]]:
    return {
        provider_type: []
        for provider_type in LISTING_ROUTABLE_FULL_ATTESTATION_PROVIDERS
    }


def _safe_listing_routable_with_full_attestation(
    value: object,
    *,
    required: bool,
    routstr_tee_ready: bool,
    verified_provider_models: dict[str, set[str]],
) -> dict[str, list[str]]:
    routable = _empty_listing_routable_with_full_attestation()
    if not required or not routstr_tee_ready or not isinstance(value, dict):
        return routable

    for provider_type in LISTING_ROUTABLE_FULL_ATTESTATION_PROVIDERS:
        raw_model_ids = value.get(provider_type, [])
        model_ids = _strict_public_string_list(raw_model_ids)
        if model_ids is None:
            return _empty_listing_routable_with_full_attestation()
        verified_model_ids = verified_provider_models.get(provider_type, set())
        for model_id in model_ids:
            if model_id not in verified_model_ids:
                return _empty_listing_routable_with_full_attestation()
            routable[provider_type].append(model_id)
    return {
        provider_type: sorted(model_ids)
        for provider_type, model_ids in routable.items()
    }


def _build_confidentiality_listing_metadata(
    status: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if status is None:
        try:
            from ..proxy import get_confidentiality_status

            status = get_confidentiality_status()
        except Exception:
            return None

    providers: list[dict[str, Any]] = []
    verified_provider_count = 0
    verified_provider_models: dict[str, set[str]] = {}
    for provider in status.get("providers", []):
        if not isinstance(provider, dict):
            continue
        confidentiality = provider.get("confidentiality") or {}
        if not isinstance(confidentiality, dict):
            continue
        if confidentiality.get("enabled", False) is not True:
            continue
        provider_type = _public_string(provider.get("provider_type"))
        if provider_type is None:
            continue
        confidentiality_policy = _safe_listing_confidentiality_policy(
            provider.get("confidentiality_policy")
        )
        provider_metadata = {
            "provider_type": provider_type,
            "confidentiality": _safe_listing_confidentiality_status(
                confidentiality,
                provider_type=provider_type,
                public_policy=confidentiality_policy,
            ),
        }
        if provider_metadata["confidentiality"].get("verified") is True:
            verified_provider_count += 1
            verified_provider_models.setdefault(provider_type, set()).update(
                cast(
                    list[str],
                    provider_metadata["confidentiality"].get("model_ids", []),
                )
            )
        upstream_name = _public_string(provider.get("upstream_name"))
        if upstream_name is not None:
            provider_metadata["upstream_name"] = upstream_name
        supported_endpoints = _public_string_list(provider.get("supported_endpoints"))
        if supported_endpoints:
            provider_metadata["supported_endpoints"] = supported_endpoints
        if confidentiality_policy:
            provider_metadata["confidentiality_policy"] = confidentiality_policy
        providers.append(provider_metadata)

    required = status.get("required", False) is True
    if not required and not providers:
        return None
    mode = _public_string(status.get("mode")) or "disabled"
    metadata: dict[str, Any] = {
        "mode": mode,
        "required": required,
        "end_to_end_ready": False,
        "providers": providers,
    }
    routstr_tee = _safe_listing_routstr_tee_status(status.get("routstr_tee"))
    if routstr_tee is not None:
        metadata["routstr_tee"] = routstr_tee
    routstr_tee_ready = (
        isinstance(routstr_tee, dict) and routstr_tee.get("ready") is True
    )
    if "routable_with_full_attestation" in status:
        routable_with_full_attestation = _safe_listing_routable_with_full_attestation(
            status.get("routable_with_full_attestation"),
            required=required,
            routstr_tee_ready=routstr_tee_ready,
            verified_provider_models=verified_provider_models,
        )
        metadata["routable_with_full_attestation"] = routable_with_full_attestation
        metadata["end_to_end_ready"] = required and any(
            routable_with_full_attestation.values()
        )
    else:
        metadata["end_to_end_ready"] = (
            required and verified_provider_count > 0 and routstr_tee_ready
        )
    return metadata


def build_listing_metadata(provider_name: str, provider_about: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "name": provider_name,
        "about": provider_about,
    }
    confidentiality = _build_confidentiality_listing_metadata()
    if confidentiality:
        metadata["confidentiality"] = confidentiality
    return metadata


def _get_tag_values(event: dict[str, Any], key: str) -> list[str]:
    tags = event.get("tags", [])
    values: list[str] = []
    for tag in tags:
        if isinstance(tag, list) and tag and tag[0] == key and len(tag) >= 2:
            values.append(tag[1])
    return values


def _get_single_tag_value(event: dict[str, Any], key: str) -> str | None:
    values = _get_tag_values(event, key)
    return values[0] if values else None


def _parse_content_json(content: str) -> dict[str, Any]:
    if not content:
        return {}
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def events_semantically_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.get("kind") != b.get("kind"):
        return False

    if _get_single_tag_value(a, "d") != _get_single_tag_value(b, "d"):
        return False

    urls_a = set(_get_tag_values(a, "u"))
    urls_b = set(_get_tag_values(b, "u"))
    if urls_a != urls_b:
        return False

    mints_a = set(_get_tag_values(a, "mint"))
    mints_b = set(_get_tag_values(b, "mint"))
    if mints_a != mints_b:
        return False

    if _get_single_tag_value(a, "version") != _get_single_tag_value(b, "version"):
        return False

    content_a = _parse_content_json(cast(str, a.get("content", "")))
    content_b = _parse_content_json(cast(str, b.get("content", "")))
    if content_a != content_b:
        return False

    return True


async def query_listing_events(
    relay_url: str,
    pubkey: str,
    provider_id: str | None = None,
    timeout: int = 30,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Query a Nostr relay for listing provider announcements (kind:38421) via nostr library.

    Returns a tuple of (events, ok) where ok indicates whether the relay interaction
    succeeded without transport-level errors.
    """

    def _sync_query() -> tuple[list[dict[str, Any]], bool]:
        rm = RelayManager()
        rm.add_relay(relay_url)
        events_out: list[dict[str, Any]] = []
        ok = True
        try:
            rm.open_connections({"cert_reqs": ssl.CERT_NONE})
            time.sleep(1.0)

            flt = Filter(kinds=[38421], authors=[pubkey], limit=10)
            filters = Filters([flt])
            sub_id = f"routstr_listing_{int(time.time())}"
            rm.add_subscription(sub_id, filters)
            req: list[Any] = [ClientMessageType.REQUEST, sub_id]
            req.extend(filters.to_json_array())
            rm.publish_message(json.dumps(req))

            start = time.time()
            last_event_ts = start
            while time.time() - start < timeout:
                drained = False
                while rm.message_pool.has_events():
                    drained = True
                    ev_msg = rm.message_pool.get_event()
                    ev = ev_msg.event
                    ev_dict = _event_to_dict(ev)
                    if provider_id is not None:
                        tags = ev_dict.get("tags", [])
                        if not any(
                            isinstance(t, list)
                            and len(t) >= 2
                            and t[0] == "d"
                            and t[1] == provider_id
                            for t in tags
                        ):
                            continue
                    events_out.append(ev_dict)
                    logger.debug(
                        f"Found listing event: {ev_dict.get('id', '')[:6]}...{ev_dict.get('id', '')[-6:]}"
                    )
                if drained:
                    last_event_ts = time.time()

                while rm.message_pool.has_notices():
                    notice = rm.message_pool.get_notice()
                    try:
                        content = getattr(notice, "content", notice)
                        s = str(content)
                        if len(s) > 200:
                            s = s[:200] + "..."
                        logger.debug(f"Relay notice: {s}")
                    except Exception:
                        pass

                if time.time() - last_event_ts > 2.5:
                    break

                time.sleep(0.1)
        except Exception as e:
            ok = False
            logger.debug(f"Failed to query relay {relay_url}: {type(e).__name__}")
        finally:
            try:
                rm.close_connections()
            except Exception:
                pass
        return events_out, ok

    return await asyncio.to_thread(_sync_query)


def discover_onion_url_from_tor(base_dir: str = "/var/lib/tor") -> str | None:
    """Discover onion URL by reading Tor hidden service hostname files.

    Tries common paths first, then scans recursively for any 'hostname' file.
    Returns an http URL like 'http://<host>.onion' if found.
    """
    common_candidates = [
        os.path.join(base_dir, "hs", "router", "hostname"),
        os.path.join(base_dir, "hs", "ROUTER", "hostname"),
        os.path.join(base_dir, "hidden_service", "hostname"),
    ]

    for candidate in common_candidates:
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                host = f.readline().strip()
            if host and host.endswith(".onion"):
                return f"http://{host}"
        except Exception:
            pass

    try:
        for root, _dirs, files in os.walk(base_dir):
            if "hostname" in files:
                path = os.path.join(root, "hostname")
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        host = f.readline().strip()
                    if host and host.endswith(".onion"):
                        return f"http://{host}"
                except Exception:
                    continue
    except Exception:
        pass

    return None


async def _determine_provider_id(public_key_hex: str, relay_urls: list[str]) -> str:
    explicit = settings.provider_id
    if explicit:
        logger.info(f"Using configured provider_id from env: {explicit}")
        return explicit

    async def query_single_relay(relay_url: str) -> list[dict[str, Any]]:
        try:
            events, _ok = await query_listing_events(relay_url, public_key_hex, None)
            return events
        except Exception:
            return []

    # Query all relays concurrently
    all_events_lists = await asyncio.gather(
        *[query_single_relay(relay_url) for relay_url in relay_urls]
    )

    latest_event: dict[str, Any] | None = None
    latest_ts = -1

    for events_list in all_events_lists:
        for ev in events_list:
            ts = int(ev.get("created_at", 0))
            if ts > latest_ts:
                latest_event = ev
                latest_ts = ts

    existing_d = _get_single_tag_value(latest_event, "d") if latest_event else None
    if existing_d:
        logger.info(f"Reusing existing provider_id from relay: {existing_d}")
        return existing_d

    fallback = public_key_hex[:12]
    logger.info(f"No existing provider_id found; using fallback: {fallback}")
    return fallback


async def publish_to_relay(
    relay_url: str,
    event: dict[str, Any],
    timeout: int = 30,
) -> bool:
    """
    Publish a listing event to a nostr relay via nostr library.
    """

    def _sync_publish() -> bool:
        rm = RelayManager()
        rm.add_relay(relay_url)
        try:
            rm.open_connections({"cert_reqs": ssl.CERT_NONE})
            time.sleep(1.0)
            # Publish the event as-is via publish_message to preserve signature
            rm.publish_message(json.dumps(["EVENT", event]))
            logger.debug(f"Sent listing event {event.get('id', '')} to {relay_url}")
            time.sleep(1.0)
            return True
        except Exception as e:
            logger.debug(f"Failed to publish to {relay_url}: {type(e).__name__}")
            return False
        finally:
            try:
                rm.close_connections()
            except Exception:
                pass

    return await asyncio.to_thread(_sync_publish)


async def announce_provider() -> None:
    """
    Background task to announce this Routstr provider to Nostr relays.
    Checks for existing announcements and creates new ones if needed.
    """
    # Check for NSEC in environment (use NSEC only)
    nsec = settings.nsec
    if not nsec:
        logger.info("Nostr private key not found (NSEC), skipping listing announcement")
        return

    # Convert NSEC to keypair
    keypair = nsec_to_keypair(nsec)
    if not keypair:
        logger.error("Failed to parse NSEC, skipping listing announcement")
        return

    private_key_hex, public_key_hex = keypair
    logger.info(f"Using Nostr pubkey: {public_key_hex}")

    # Resolve settings and determine if we can publish BEFORE touching relays
    try:
        base_url: str | None = settings.http_url
        onion_url: str | None = settings.onion_url
        provider_name = settings.name or "Routstr Proxy"
        provider_about = settings.description or "Privacy-preserving AI proxy via Nostr"
        cashu_mints = [m.strip() for m in settings.cashu_mints if m.strip()]
    except Exception:
        base_url = settings.http_url or None
        onion_url = settings.onion_url or None
        provider_name = settings.name or "Routstr Proxy"
        provider_about = settings.description or "Privacy-preserving AI proxy via Nostr"
        cashu_mints = [m.strip() for m in settings.cashu_mints if m.strip()]
    if not onion_url:
        discovered = discover_onion_url_from_tor()
        if discovered:
            onion_url = discovered
            logger.info(f"Discovered onion URL via Tor volume: {onion_url}")
    mint_urls = cashu_mints if cashu_mints else None

    endpoint_urls: list[str] = []
    if base_url and base_url.strip() and base_url.strip() != "http://localhost:8000":
        endpoint_urls.append(base_url.strip())
    if onion_url and onion_url.strip():
        ou = onion_url.strip()
        if ou.endswith(".onion") and not (
            ou.startswith("http://") or ou.startswith("https://")
        ):
            ou = f"http://{ou}"
        endpoint_urls.append(ou)

    if not endpoint_urls:
        logger.warning(
            "No valid endpoints configured (HTTP_URL/ONION_URL). Skipping listing publish."
        )
        return

    # Only now configure relays and determine provider_id (may query relays)
    relay_urls = [u.strip() for u in getattr(settings, "relays", []) if u.strip()]
    if not relay_urls:
        relay_urls = [
            "wss://relay.nostr.band",
            "wss://relay.damus.io",
            "wss://relay.routstr.com",
            "wss://nos.lol",
        ]

    provider_id = await _determine_provider_id(public_key_hex, relay_urls)
    logger.info(f"Using provider_id: {provider_id}")

    # Build metadata
    metadata = build_listing_metadata(provider_name, provider_about)

    # Create the candidate event that we would publish
    version_str = get_app_version()
    candidate_event = create_listing_event(
        private_key_hex=private_key_hex,
        provider_id=provider_id,
        endpoint_urls=endpoint_urls,
        mint_urls=mint_urls,
        version=version_str,
        metadata=metadata,
    )

    # Backoff configuration and state (sensible defaults)
    backoff_base = 5.0
    backoff_max = 900.0
    backoff_jitter_ratio = 0.2
    relay_next_allowed: dict[str, float] = {}
    relay_current_delay: dict[str, float] = {}

    def _should_skip(relay: str) -> bool:
        return time.time() < relay_next_allowed.get(relay, 0.0)

    def _register_success(relay: str) -> None:
        relay_current_delay[relay] = 0.0
        relay_next_allowed[relay] = time.time()

    def _register_failure(relay: str) -> None:
        previous = relay_current_delay.get(relay, 0.0)
        delay = backoff_base if previous <= 0.0 else min(backoff_max, previous * 2.0)
        jitter = delay * backoff_jitter_ratio * (2.0 * random.random() - 1.0)
        scheduled = time.time() + max(0.0, delay + jitter)
        relay_current_delay[relay] = delay
        relay_next_allowed[relay] = scheduled
        logger.debug(
            f"Backoff: {relay} delay={delay:.1f}s jitter={jitter:.1f}s next={int(scheduled)}"
        )

    # Fetch existing events for this provider_id
    existing_events: list[dict[str, Any]] = []
    for relay_url in relay_urls:
        if _should_skip(relay_url):
            logger.debug(f"Skipping {relay_url} due to backoff")
            continue
        events, ok = await query_listing_events(relay_url, public_key_hex, provider_id)
        if ok:
            _register_success(relay_url)
            existing_events.extend(events)
        else:
            _register_failure(relay_url)

    # Decide whether to publish: publish if none exist or any differ from candidate
    found_any = len(existing_events) > 0
    all_match = found_any and all(
        events_semantically_equal(ev, candidate_event) for ev in existing_events
    )

    if not all_match:
        logger.debug(
            "No matching listing announcement found or differences detected; publishing update"
        )
        success_count = 0
        for relay_url in relay_urls:
            if _should_skip(relay_url):
                logger.debug(f"Skipping publish to {relay_url} due to backoff")
                continue
            if await publish_to_relay(relay_url, candidate_event):
                _register_success(relay_url)
                success_count += 1
            else:
                _register_failure(relay_url)
        logger.info(
            f"Published listing announcement to {success_count}/{len(relay_urls)} relays"
        )
    else:
        logger.debug(
            "Matching listing announcement already present; skipping publish on startup"
        )

    # Re-announce periodically (every 24 hours)
    announcement_interval = 24 * 60 * 60

    while True:
        try:
            await asyncio.sleep(announcement_interval)

            # Build fresh candidate event for comparison
            version_str = get_app_version()
            metadata = build_listing_metadata(provider_name, provider_about)
            candidate_event = create_listing_event(
                private_key_hex=private_key_hex,
                provider_id=provider_id,
                endpoint_urls=endpoint_urls,
                mint_urls=mint_urls,
                version=version_str,
                metadata=metadata,
            )

            # Fetch existing events for this provider_id
            existing_events = []
            for relay_url in relay_urls:
                if _should_skip(relay_url):
                    logger.debug(f"Skipping {relay_url} due to backoff")
                    continue
                events, ok = await query_listing_events(
                    relay_url, public_key_hex, provider_id
                )
                if ok:
                    _register_success(relay_url)
                    existing_events.extend(events)
                else:
                    _register_failure(relay_url)

            found_any = len(existing_events) > 0
            all_match = found_any and all(
                events_semantically_equal(ev, candidate_event) for ev in existing_events
            )

            if all_match:
                logger.debug(
                    "Matching listing announcement already present; skipping periodic re-announce"
                )
                continue

            logger.debug(
                f"Re-announcing provider due to differences or absence: {candidate_event['id']}"
            )
            for relay_url in relay_urls:
                if _should_skip(relay_url):
                    logger.debug(f"Skipping publish to {relay_url} due to backoff")
                    continue
                ok = await publish_to_relay(relay_url, candidate_event)
                if ok:
                    _register_success(relay_url)
                else:
                    _register_failure(relay_url)

        except asyncio.CancelledError:
            logger.info("Listing announcement task cancelled")
            break
        except Exception as e:
            logger.debug(f"Error in listing announcement loop: {type(e).__name__}")
            # Continue running despite errors
