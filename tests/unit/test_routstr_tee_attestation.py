from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from routstr.core.attestation import (
    _accepted_routstr_tee_verification,
    _clear_cap_attestation_status_cache,
    _private_local_artifacts_snapshot,
    _routstr_tee_report_data_digest,
    _routstr_tee_report_data_hex,
    _routstr_tee_verifier_policy,
    _safe_provider_confidentiality,
    _validate_routstr_tee_verifier_result,
    _verify_routstr_tee_evidence,
    get_public_routstr_tee_status,
    get_routstr_attestation_statement,
    get_routstr_tee_readiness,
)
from routstr.core.settings import settings

VERIFIER_STUB_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "confidential_verifier_stub.py"
)
ATTESTATION_STUB_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "routstr_tee_attestation_stub.py"
)


def _named_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


VALID_POLICY_DIGEST = _named_digest("routing-policy")
VALID_OTHER_POLICY_DIGEST = _named_digest("other-routing-policy")
VALID_QUOTE_DIGEST = _named_digest("routstr-tee-quote")
VALID_HPKE_KEY_CONFIG_DIGEST = _named_digest("hpke-key-config")
VALID_HPKE_PUBLIC_KEY_DIGEST = _named_digest("hpke-public-key")
VALID_PUBLIC_KEY_DIGEST = _named_digest("routstr-attestation-public-key")
VALID_CODE_MEASUREMENT = _named_digest("routstr-code-measurement")
VALID_CODE_MEASUREMENT_A = _named_digest("routstr-code-measurement-a")
VALID_CODE_MEASUREMENT_B = _named_digest("routstr-code-measurement-b")
VALID_UNEXPECTED_CODE_MEASUREMENT = _named_digest("unexpected-routstr-code-measurement")
VALID_PRIVATEMODE_PROXY_DIGEST = _named_digest("privatemode-proxy-binary")
VALID_QUOTE_REPORT_DIGEST = _named_digest("tee-attestation-report")
VALID_QUOTE_CERTIFICATE_CHAIN_DIGEST = _named_digest("tee-certificate-chain")
VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST = _named_digest("local-runtime-evidence")
VALID_PROVIDER_CLAIMS_DIGEST = _named_digest("provider-claims")
FIXTURE_ROUTSTR_CODE_MEASUREMENT = (
    "sha256:" + hashlib.sha256(b"fixture-routstr-code-measurement").hexdigest()
)


def _sha256_file_digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _json_digest(value: object) -> str:
    data = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _provider_policy_digest(
    *,
    provider_type: str = "tinfoil",
    mode: str = "tinfoil",
    base_url: str = "https://inference.tinfoil.sh/v1",
    policy: dict[str, object] | None = None,
    model_ids: list[str] | None = None,
    model_id_prefixes: list[str] | None = None,
) -> str:
    return _json_digest(
        {
            "provider_type": provider_type,
            "mode": mode,
            "base_url": base_url,
            "policy": policy or {},
            "model_ids": model_ids or [],
            "model_id_prefixes": model_id_prefixes or [],
        }
    )


def _ehbp_key_config(public_key: bytes | None = None) -> bytes:
    public_key = public_key or (b"\x55" * 32)
    return b"\x00" + b"\x00\x20" + public_key + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"


def _configure_routstr_tee_command_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)
    public_key = "-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----"

    monkeypatch.setattr(settings, "routstr_attestation_document_path", "")
    monkeypatch.setattr(settings, "routstr_attestation_document_format", "tdx_quote")
    monkeypatch.setattr(settings, "routstr_attestation_public_key", public_key)
    monkeypatch.setattr(settings, "routstr_attestation_public_key_path", "")
    monkeypatch.setattr(
        settings,
        "routstr_attestation_hpke_key_config_b64",
        base64.b64encode(hpke_key_config).decode("ascii"),
    )
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_path", "")
    monkeypatch.setattr(
        settings,
        "routstr_tee_attestation_command",
        json.dumps([sys.executable, str(ATTESTATION_STUB_PATH)]),
        raising=False,
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_attestation_artifact_path",
        str(ATTESTATION_STUB_PATH),
        raising=False,
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_attestation_command_digest",
        _sha256_file_digest(ATTESTATION_STUB_PATH),
        raising=False,
    )
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
            {"expected_routstr_code_measurement": FIXTURE_ROUTSTR_CODE_MEASUREMENT}
        ),
    )
    monkeypatch.setattr(settings, "routstr_tee_verifier_timeout_seconds", 2.0)


def _verification_steps() -> dict[str, bool]:
    return {
        "freshness": True,
        "hpke_key_binding": True,
        "measurement_match": True,
        "public_key_binding": True,
        "runtime_policy_binding": True,
        "tee_attestation_report": True,
        "tee_certificate_chain": True,
    }


def _proof_claims(
    nonce: str = "nonce",
    attestation_document_format: str = "tdx_quote",
    routing_policy_digest: str = VALID_POLICY_DIGEST,
    hpke_key_config_digest: str = VALID_HPKE_KEY_CONFIG_DIGEST,
    hpke_public_key_digest: str = VALID_HPKE_PUBLIC_KEY_DIGEST,
    public_key_digest: str = VALID_PUBLIC_KEY_DIGEST,
    include_report_data_hex: bool = True,
) -> dict[str, str]:
    claims = {
        "attestation_document_format": attestation_document_format,
        "hpke_key_config_digest": hpke_key_config_digest,
        "tee_attestation_report_digest": VALID_QUOTE_REPORT_DIGEST,
        "tee_certificate_chain_digest": VALID_QUOTE_CERTIFICATE_CHAIN_DIGEST,
        "tee_report_data_digest": _routstr_tee_report_data_digest(
            routing_policy_digest=routing_policy_digest,
            hpke_key_config_digest=hpke_key_config_digest,
            hpke_public_key_digest=hpke_public_key_digest,
            public_key_digest=public_key_digest,
            verification_nonce=nonce,
        ),
        "tee_report_nonce": nonce,
        "tee_report_nonce_digest": (
            "sha256:" + hashlib.sha256(nonce.encode("utf-8")).hexdigest()
        ),
    }
    if include_report_data_hex:
        claims["tee_report_data_hex"] = _routstr_tee_report_data_hex(
            routing_policy_digest=routing_policy_digest,
            hpke_key_config_digest=hpke_key_config_digest,
            hpke_public_key_digest=hpke_public_key_digest,
            public_key_digest=public_key_digest,
            verification_nonce=nonce,
        )
    return claims


def _provider_proof_claims() -> dict[str, object]:
    return {
        "transport": "ehbp",
        "repo": "tinfoilsh/confidential-model-router",
        "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
        "attestation_report_digest": _named_digest("tinfoil-router-attestation-report"),
        "attested_hpke_public_key_hex": "a" * 64,
        "enclave_measurement_fingerprint": _named_digest(
            "tinfoil-router-enclave-measurement"
        ),
        "code_measurement_fingerprint": _named_digest(
            "tinfoil-router-code-measurement"
        ),
        "payload_policy_digest": VALID_POLICY_DIGEST,
        "payload_evidence_digest": _named_digest("provider-runtime-evidence"),
        "release_digest": _named_digest("tinfoil-router-release"),
        "tls_public_key_fingerprint_sha256": _named_digest("tinfoil-router-tls-key"),
        "verification_steps": {
            "hardware_attestation_report": True,
            "hardware_certificate_chain": True,
            "code_transparency": True,
            "measurement_match": True,
            "attested_transport_key_binding": True,
            "freshness": True,
        },
        "model_attestations": {
            "secure-model": {
                "repo": "tinfoilsh/confidential-secure-model",
                "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
                "attestation_report_digest": _named_digest(
                    "secure-model-attestation-report"
                ),
                "attested_hpke_public_key_hex": "f" * 64,
                "enclave_measurement_fingerprint": _named_digest(
                    "secure-model-enclave-measurement"
                ),
                "code_measurement_fingerprint": _named_digest(
                    "secure-model-code-measurement"
                ),
                "release_digest": _named_digest("secure-model-release"),
                "tls_public_key_fingerprint_sha256": _named_digest(
                    "secure-model-tls-key"
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
    }


def _ppq_private_provider_proof_claims(policy_digest: str) -> dict[str, object]:
    return {
        "transport": "ehbp",
        "client_encryption_boundary": "routstr-tee-ehbp-proxy",
        "repo": "ppq-ai/private-tee",
        "attestation_bundle_url_digest": _json_digest("https://api.ppq.ai/private"),
        "attested_hpke_public_key_hex": "b" * 64,
        "code_measurement_fingerprint": _named_digest("ppq-provider-code-measurement"),
        "enclave_measurement_fingerprint": _named_digest(
            "ppq-provider-enclave-measurement"
        ),
        "payload_policy_digest": policy_digest,
        "payload_evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
        "release_digest": _named_digest("ppq-provider-release"),
        "selected_model_ids": ["private/gpt-oss-120b"],
        "tls_public_key_fingerprint_sha256": _named_digest("ppq-provider-tls-key"),
        "verification_steps": {
            "hardware_attestation_report": True,
            "hardware_certificate_chain": True,
            "code_transparency": True,
            "measurement_match": True,
            "attested_transport_key_binding": True,
            "freshness": True,
        },
        "backend_model_attestations": {
            "private/gpt-oss-120b": {
                "repo": "tinfoilsh/confidential-gpt-oss-120b",
                "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
                "attestation_report_digest": _named_digest(
                    "ppq-backend-attestation-report"
                ),
                "attested_hpke_public_key_hex": "c" * 64,
                "code_measurement_fingerprint": _named_digest(
                    "ppq-backend-code-measurement"
                ),
                "enclave_measurement_fingerprint": _named_digest(
                    "ppq-backend-enclave-measurement"
                ),
                "release_digest": _named_digest("ppq-backend-release"),
                "tls_public_key_fingerprint_sha256": _named_digest(
                    "ppq-backend-tls-key"
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
    }


def test_routstr_tee_verifier_requires_config_claim_to_bind_policy() -> None:
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_OTHER_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }

    with pytest.raises(
        ValueError,
        match="routstr_config_measurement claim does not match routing policy",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_rejects_non_boolean_verification_step() -> None:
    claims = {
        "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
        "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
        "routstr_code_measurement": VALID_CODE_MEASUREMENT,
        "routstr_config_measurement": VALID_POLICY_DIGEST,
        "verification_steps": {
            **_verification_steps(),
            "diagnostic": "SECRET_PROMPT",
        },
        **_proof_claims(),
    }
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
        "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
        "verification_nonce": "nonce",
        "claims": claims,
    }

    with pytest.raises(
        ValueError,
        match="Routstr TEE verification_steps values must be JSON booleans",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_requires_expected_code_measurement_policy() -> None:
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }

    with pytest.raises(
        ValueError,
        match="expected_routstr_code_measurement or allowed_routstr_code_measurements is required",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={},
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_rejects_non_string_code_measurement_policy_entries() -> (
    None
):
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }

    with pytest.raises(
        ValueError,
        match="routstr_code_measurement policy value must be a sha256 digest",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": {"unexpected": "object"},
                "allowed_routstr_code_measurements": [VALID_CODE_MEASUREMENT],
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_rejects_blank_code_measurement_policy_alias() -> None:
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }

    with pytest.raises(
        ValueError,
        match="routstr_code_measurement policy value must be a sha256 digest",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": "",
                "allowed_routstr_code_measurements": [VALID_CODE_MEASUREMENT],
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_rejects_code_measurement_outside_policy() -> None:
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_UNEXPECTED_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }

    with pytest.raises(
        ValueError,
        match="routstr_code_measurement claim does not match policy",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "allowed_routstr_code_measurements": [
                    VALID_CODE_MEASUREMENT_A,
                    VALID_CODE_MEASUREMENT_B,
                ],
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_defaults_missing_expiry_to_bounded_freshness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.core.attestation.time.time",
        lambda: 1_700_000_010,
    )
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "verified_at": 1_700_000_000,
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }

    result = _validate_routstr_tee_verifier_result(
        verifier_result=verifier_result,
        verifier_policy={
            "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
        },
        routing_policy_digest=VALID_POLICY_DIGEST,
        attestation_evidence_digest=VALID_QUOTE_DIGEST,
        attestation_document_format="tdx_quote",
        hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
        hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
        public_key_digest=VALID_PUBLIC_KEY_DIGEST,
        verification_nonce="nonce",
    )

    assert result["expires_at"] == 1_700_000_300


def test_routstr_tee_verifier_rejects_stale_verified_at_despite_future_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.core.attestation.time.time",
        lambda: 1_700_000_400,
    )
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "verified_at": 1_700_000_000,
        "expires_at": 4_102_444_800,
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }

    with pytest.raises(ValueError, match="Routstr TEE verifier result is stale"):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_rejects_future_verified_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.core.attestation.time.time",
        lambda: 1_700_000_010,
    )
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "verified_at": 1_700_000_020,
        "expires_at": 1_700_000_320,
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }

    with pytest.raises(
        ValueError,
        match="Routstr TEE verifier result verified_at is in the future",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_rejects_string_expiry_claim() -> None:
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "verified_at": 1_700_000_000,
        "expires_at": "4102444800",
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }

    with pytest.raises(ValueError, match="expires_at must be an integer"):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


@pytest.mark.parametrize(
    ("policy_fragment", "expected_message"),
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
def test_routstr_tee_verifier_rejects_malformed_max_age_policy(
    policy_fragment: dict[str, object],
    expected_message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.core.attestation.time.time",
        lambda: 1_700_000_010,
    )
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "verified_at": 1_700_000_000,
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
                **policy_fragment,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_rejects_placeholder_proof_digests() -> None:
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
            "tee_attestation_report_digest": "sha256:" + ("8" * 64),
        },
    }

    with pytest.raises(ValueError, match="must not be a placeholder digest"):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


@pytest.mark.parametrize(
    ("claim_name", "claim_value", "extra_result"),
    (
        ("routing_policy_digest", VALID_OTHER_POLICY_DIGEST, {}),
        ("payload_routing_policy_digest", VALID_OTHER_POLICY_DIGEST, {}),
        ("attestation_evidence_digest", VALID_OTHER_POLICY_DIGEST, {}),
        ("attestation_document_format", "sev_snp_report", {}),
        ("hpke_key_config_digest", VALID_OTHER_POLICY_DIGEST, {}),
        ("hpke_public_key_digest", VALID_OTHER_POLICY_DIGEST, {}),
        (
            "public_key_digest",
            VALID_OTHER_POLICY_DIGEST,
            {"public_key_digest": VALID_PUBLIC_KEY_DIGEST},
        ),
        ("verification_nonce", "wrong-nonce", {}),
        ("payload_verification_nonce", "wrong-nonce", {}),
    ),
)
def test_routstr_tee_verifier_rejects_mismatched_payload_binding_claims(
    claim_name: str,
    claim_value: str,
    extra_result: dict[str, str],
) -> None:
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "attestation_document_format": "tdx_quote",
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
        "verification_nonce": "nonce",
        **extra_result,
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
            claim_name: claim_value,
        },
    }

    with pytest.raises(ValueError, match=claim_name):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_readiness_rejects_verified_status_without_future_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_path = tmp_path / "quote.bin"
    evidence_path.write_bytes(b"fake-tdx-quote")
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)

    monkeypatch.setattr(
        "routstr.core.attestation.time.time",
        lambda: 1_700_000_010,
    )
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
        return {
            "verified": True,
            "verifier": "bad-adapter",
            "verified_at": 1_700_000_000,
            "expires_at": None,
            "failure_reason": None,
            "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
            "verified_claims": {
                "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
                "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
                "routstr_code_measurement": VALID_CODE_MEASUREMENT,
                "routstr_config_measurement": VALID_POLICY_DIGEST,
                "verification_steps": _verification_steps(),
                **_proof_claims(),
            },
        }

    monkeypatch.setattr(
        "routstr.core.attestation._verify_routstr_tee_evidence",
        fake_verify_routstr_tee_evidence,
    )

    readiness = get_routstr_tee_readiness()

    assert readiness["ready"] is False
    assert readiness["local_verification"]["verified"] is False
    assert "future expires_at" in str(readiness["failure_reason"] or "")


def test_routstr_tee_readiness_rejects_static_evidence_in_required_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_path = tmp_path / "quote.bin"
    evidence_path.write_bytes(b"fake-tdx-quote")
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)
    verifier_called = False

    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
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
    monkeypatch.setattr(settings, "routstr_tee_attestation_command", "")
    monkeypatch.setattr(settings, "routstr_tee_attestation_command_digest", "")
    monkeypatch.setattr(settings, "routstr_tee_attestation_artifact_path", "")

    def fake_verify_routstr_tee_evidence(**kwargs: object) -> dict[str, object]:
        nonlocal verifier_called
        verifier_called = True
        return {
            "verified": True,
            "verifier": "bad-static-verifier",
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "failure_reason": None,
            "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
            "verified_claims": {
                "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
                "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
                "routstr_code_measurement": VALID_CODE_MEASUREMENT,
                "routstr_config_measurement": VALID_POLICY_DIGEST,
                "verification_steps": _verification_steps(),
                **_proof_claims(),
            },
        }

    monkeypatch.setattr(
        "routstr.core.attestation._verify_routstr_tee_evidence",
        fake_verify_routstr_tee_evidence,
    )

    readiness = get_routstr_tee_readiness()

    assert readiness["ready"] is False
    assert readiness["local_verification"]["verified"] is False
    assert verifier_called is False
    assert "static evidence is only suitable for fixtures" in str(
        readiness["failure_reason"] or ""
    )


def test_routstr_tee_readiness_requires_client_confidentiality_boundary_in_required_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_routstr_tee_command_stubs(monkeypatch)
    monkeypatch.setattr(
        "routstr.core.attestation.time.time",
        lambda: 1_700_000_010,
    )
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    monkeypatch.setattr(settings, "routstr_tee_client_confidentiality_boundary", "")

    readiness = get_routstr_tee_readiness()

    assert readiness["ready"] is False
    assert readiness["local_verification"]["verified"] is False
    assert "ROUTSTR_TEE_CLIENT_CONFIDENTIALITY_BOUNDARY" in str(
        readiness["failure_reason"] or ""
    )


def test_routstr_tee_readiness_accepts_client_confidentiality_boundary_in_required_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_routstr_tee_command_stubs(monkeypatch)
    monkeypatch.setattr(
        "routstr.core.attestation.time.time",
        lambda: 1_700_000_010,
    )
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    monkeypatch.setattr(
        settings,
        "routstr_tee_client_confidentiality_boundary",
        "attested-tls-termination",
    )

    readiness = get_routstr_tee_readiness()

    assert readiness["ready"] is True
    assert readiness["failure_reason"] is None
    assert readiness["local_verification"]["verified"] is True


def test_routstr_tee_readiness_rejects_attested_tls_public_key_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_routstr_tee_command_stubs(monkeypatch)
    monkeypatch.setattr(
        "routstr.core.attestation.time.time",
        lambda: 1_700_000_010,
    )
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    monkeypatch.setattr(
        settings,
        "routstr_tee_client_confidentiality_boundary",
        "attested-tls-termination",
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_attested_tls_public_key_digest",
        VALID_OTHER_POLICY_DIGEST,
        raising=False,
    )

    readiness = get_routstr_tee_readiness()

    assert readiness["ready"] is False
    assert readiness["local_verification"]["verified"] is False
    assert "ROUTSTR_TEE_ATTESTED_TLS_PUBLIC_KEY_DIGEST does not match" in str(
        readiness["failure_reason"] or ""
    )


def test_routstr_tee_acceptance_rejects_placeholder_runtime_digests() -> None:
    accepted, failure_reason = _accepted_routstr_tee_verification(
        {
            "verified": True,
            "verifier": "bad-adapter",
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "failure_reason": None,
            "evidence_digest": "sha256:local-runtime-evidence",
            "verified_claims": {
                "routstr_code_measurement": "sha256:code",
            },
        }
    )

    assert accepted["verified"] is False
    assert failure_reason == (
        "Routstr TEE local verification evidence_digest must be a sha256 digest"
    )


def test_routstr_tee_acceptance_rejects_truthy_string_verified() -> None:
    accepted, failure_reason = _accepted_routstr_tee_verification(
        {
            "verified": "false",
            "verifier": "bad-adapter",
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "failure_reason": None,
            "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
            "verified_claims": {
                "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
                "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
                "routstr_code_measurement": VALID_CODE_MEASUREMENT,
                "routstr_config_measurement": VALID_POLICY_DIGEST,
                "verification_steps": _verification_steps(),
                **_proof_claims(),
            },
        }
    )

    assert accepted["verified"] is False
    assert failure_reason == "Routstr TEE local verification verified must be true"


def test_routstr_tee_acceptance_requires_hpke_key_config_digest_claim() -> None:
    verified_claims = {
        "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
        "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
        "routstr_code_measurement": VALID_CODE_MEASUREMENT,
        "routstr_config_measurement": VALID_POLICY_DIGEST,
        "verification_steps": _verification_steps(),
        **_proof_claims(),
    }
    verified_claims.pop("hpke_key_config_digest", None)

    accepted, failure_reason = _accepted_routstr_tee_verification(
        {
            "verified": True,
            "verifier": "bad-adapter",
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "failure_reason": None,
            "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
            "verified_claims": verified_claims,
        }
    )

    assert accepted["verified"] is False
    assert failure_reason == (
        "Routstr TEE local verification "
        "hpke_key_config_digest claim must be a sha256 digest"
    )


@pytest.mark.parametrize("verifier", [None, "", {"name": "unit-test-verifier"}])
def test_routstr_tee_acceptance_requires_string_verifier_identity(
    verifier: object,
) -> None:
    accepted, failure_reason = _accepted_routstr_tee_verification(
        {
            "verified": True,
            "verifier": verifier,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "failure_reason": None,
            "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
            "verified_claims": {
                "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
                "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
                "routstr_code_measurement": VALID_CODE_MEASUREMENT,
                "routstr_config_measurement": VALID_POLICY_DIGEST,
                "verification_steps": _verification_steps(),
                **_proof_claims(),
            },
        }
    )

    assert accepted["verified"] is False
    assert failure_reason == (
        "Routstr TEE local verification verifier must be a non-empty string"
    )


@pytest.mark.parametrize(
    ("verified_at", "expected_failure"),
    [
        (None, "Routstr TEE local verification must include verified_at"),
        (
            "1700000000",
            "Routstr TEE local verification verified_at must be an integer",
        ),
    ],
)
def test_routstr_tee_acceptance_requires_integer_verified_at(
    verified_at: object,
    expected_failure: str,
) -> None:
    accepted, failure_reason = _accepted_routstr_tee_verification(
        {
            "verified": True,
            "verifier": "bad-adapter",
            "verified_at": verified_at,
            "expires_at": 4_102_444_800,
            "failure_reason": None,
            "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
            "verified_claims": {
                "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
                "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
                "routstr_code_measurement": VALID_CODE_MEASUREMENT,
                "routstr_config_measurement": VALID_POLICY_DIGEST,
                "verification_steps": _verification_steps(),
                **_proof_claims(),
            },
        }
    )

    assert accepted["verified"] is False
    assert failure_reason == expected_failure


def test_routstr_tee_acceptance_rejects_future_verified_at() -> None:
    now = int(time.time())
    accepted, failure_reason = _accepted_routstr_tee_verification(
        {
            "verified": True,
            "verifier": "bad-adapter",
            "verified_at": now + 60,
            "expires_at": now + 300,
            "failure_reason": None,
            "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
            "verified_claims": {
                "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
                "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
                "routstr_code_measurement": VALID_CODE_MEASUREMENT,
                "routstr_config_measurement": VALID_POLICY_DIGEST,
                "verification_steps": _verification_steps(),
                **_proof_claims(),
            },
        }
    )

    assert accepted["verified"] is False
    assert failure_reason == (
        "Routstr TEE local verification verified_at is in the future"
    )


class _StringCoercibleProofField:
    def __init__(self, value: str):
        self.value = value

    def __str__(self) -> str:
        return self.value


class _NonSerializableProofClaim:
    pass


def test_routstr_tee_verification_rejects_string_coercible_hpke_key_config_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            {"expected_routstr_code_measurement": FIXTURE_ROUTSTR_CODE_MEASUREMENT}
        ),
    )
    monkeypatch.setattr(settings, "routstr_tee_verifier_timeout_seconds", 2.0)

    status = _verify_routstr_tee_evidence(
        tee={
            "attestation_evidence_digest": VALID_QUOTE_DIGEST,
            "evidence_format": "tdx_quote",
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "hpke_key_config": {
                "key_config_digest": _StringCoercibleProofField(
                    VALID_HPKE_KEY_CONFIG_DIGEST
                ),
                "public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            },
        },
        routing_policy={"mode": "required", "providers": []},
    )

    assert status["verified"] is False
    assert "Routstr TEE HPKE key config digest must be a sha256 digest" in str(
        status["failure_reason"] or ""
    )


@pytest.mark.parametrize(
    "attestation_document_b64",
    (
        _StringCoercibleProofField(base64.b64encode(b"fake-tdx-quote").decode("ascii")),
        "not-base64",
    ),
)
def test_routstr_tee_verification_rejects_malformed_attestation_document(
    attestation_document_b64: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            {"expected_routstr_code_measurement": FIXTURE_ROUTSTR_CODE_MEASUREMENT}
        ),
    )
    monkeypatch.setattr(settings, "routstr_tee_verifier_timeout_seconds", 2.0)

    status = _verify_routstr_tee_evidence(
        tee={
            "attestation_document_b64": attestation_document_b64,
            "attestation_evidence_digest": VALID_QUOTE_DIGEST,
            "evidence_format": "tdx_quote",
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "hpke_key_config": {
                "key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
                "public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            },
        },
        routing_policy={"mode": "required", "providers": []},
    )

    assert status["verified"] is False
    assert "Routstr TEE attestation document must be base64 evidence" in str(
        status["failure_reason"] or ""
    )


@pytest.mark.parametrize(
    ("path", "value", "expected_failure"),
    [
        (
            ("evidence_digest",),
            _StringCoercibleProofField(VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST),
            "Routstr TEE local verification evidence_digest must be a sha256 digest",
        ),
        (
            ("verified_claims", "hpke_public_key_digest"),
            _StringCoercibleProofField(VALID_HPKE_PUBLIC_KEY_DIGEST),
            "Routstr TEE local verification hpke_public_key_digest claim must be a sha256 digest",
        ),
        (
            ("verified_claims", "attestation_document_format"),
            _StringCoercibleProofField("tdx_quote"),
            "Routstr TEE local verification attestation_document_format claim must be a non-empty string",
        ),
    ],
)
def test_routstr_tee_acceptance_rejects_string_coercible_proof_fields(
    path: tuple[str, ...],
    value: object,
    expected_failure: str,
) -> None:
    status: dict[str, object] = {
        "verified": True,
        "verifier": "bad-adapter",
        "verified_at": 1_700_000_000,
        "expires_at": 4_102_444_800,
        "failure_reason": None,
        "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
        "verified_claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }
    target: dict[str, object] = status
    for segment in path[:-1]:
        target = target[segment]  # type: ignore[assignment]
    target[path[-1]] = value

    accepted, failure_reason = _accepted_routstr_tee_verification(status)

    assert accepted["verified"] is False
    assert failure_reason == expected_failure


@pytest.mark.parametrize(
    ("claim_name", "claim_value", "expected_failure"),
    [
        (
            "tee_report_nonce",
            {"nonce": "nonce"},
            "Routstr TEE local verification tee_report_nonce claim must be a non-empty string",
        ),
        (
            "tee_report_nonce_digest",
            "sha256:" + ("0" * 63),
            "Routstr TEE local verification tee_report_nonce_digest claim must be a sha256 digest",
        ),
    ],
)
def test_routstr_tee_acceptance_rejects_malformed_optional_public_claims(
    claim_name: str,
    claim_value: object,
    expected_failure: str,
) -> None:
    accepted, failure_reason = _accepted_routstr_tee_verification(
        {
            "verified": True,
            "verifier": "bad-adapter",
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "failure_reason": None,
            "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
            "verified_claims": {
                "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
                "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
                "routstr_code_measurement": VALID_CODE_MEASUREMENT,
                "routstr_config_measurement": VALID_POLICY_DIGEST,
                "verification_steps": _verification_steps(),
                **_proof_claims(),
                claim_name: claim_value,
            },
        }
    )

    assert accepted["verified"] is False
    assert failure_reason == expected_failure


def test_routstr_tee_acceptance_rejects_non_json_digestible_claims() -> None:
    accepted, failure_reason = _accepted_routstr_tee_verification(
        {
            "verified": True,
            "verifier": "bad-adapter",
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "failure_reason": None,
            "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
            "verified_claims": {
                "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
                "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
                "routstr_code_measurement": VALID_CODE_MEASUREMENT,
                "routstr_config_measurement": VALID_POLICY_DIGEST,
                "verification_steps": _verification_steps(),
                "debug": _NonSerializableProofClaim(),
                **_proof_claims(),
            },
        }
    )

    assert accepted["verified"] is False
    assert failure_reason == (
        "Routstr TEE local verification verified_claims must be canonical JSON"
    )


def test_routstr_tee_acceptance_rejects_non_canonical_claims() -> None:
    accepted, failure_reason = _accepted_routstr_tee_verification(
        {
            "verified": True,
            "verifier": "bad-adapter",
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "failure_reason": None,
            "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
            "verified_claims": {
                "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
                "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
                "routstr_code_measurement": VALID_CODE_MEASUREMENT,
                "routstr_config_measurement": VALID_POLICY_DIGEST,
                "verification_steps": _verification_steps(),
                "non_canonical": float("nan"),
                **_proof_claims(),
            },
        }
    )

    assert accepted["verified"] is False
    assert failure_reason == (
        "Routstr TEE local verification verified_claims must be canonical JSON"
    )


def test_routstr_tee_verifier_policy_rejects_non_standard_json_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "routstr_tee_verifier_policy_json",
        (
            '{"expected_routstr_code_measurement":"'
            f'{VALID_CODE_MEASUREMENT}",'
            '"non_finite":NaN}'
        ),
    )

    with pytest.raises(ValueError, match="must not contain NaN"):
        _routstr_tee_verifier_policy()


def test_routstr_tee_verifier_policy_rejects_placeholder_policy_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "routstr_tee_verifier_policy_json",
        json.dumps(
            {
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
                "expected_routstr_runtime_artifact_digest": "sha256:" + ("a" * 64),
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="expected_routstr_runtime_artifact_digest must not be a placeholder digest",
    ):
        _routstr_tee_verifier_policy()


def test_routstr_tee_readiness_rejects_truthy_string_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_path = tmp_path / "quote.bin"
    evidence_path.write_bytes(b"fake-tdx-quote")
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)

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
        return {
            "verified": "false",
            "verifier": "bad-adapter",
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "failure_reason": None,
            "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
            "verified_claims": {
                "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
                "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
                "routstr_code_measurement": VALID_CODE_MEASUREMENT,
                "routstr_config_measurement": VALID_POLICY_DIGEST,
                "verification_steps": _verification_steps(),
                **_proof_claims(),
            },
        }

    monkeypatch.setattr(
        "routstr.core.attestation._verify_routstr_tee_evidence",
        fake_verify_routstr_tee_evidence,
    )

    readiness = get_routstr_tee_readiness()

    assert readiness["ready"] is False
    assert readiness["local_verification"]["verified"] is False
    assert "verified must be true" in str(readiness["failure_reason"] or "")


def test_public_routstr_tee_status_requires_strict_ready_boolean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.core.attestation.get_routstr_tee_readiness",
        lambda: {
            "ready": "false",
            "failure_reason": "malformed readiness",
            "attestation_evidence_digest": VALID_QUOTE_DIGEST,
            "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "local_verification": {"verified": False},
        },
    )

    status = get_public_routstr_tee_status()

    assert status["ready"] is False
    assert status["failure_reason_digest"].startswith("sha256:")


def test_public_routstr_tee_status_publishes_client_confidentiality_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public_key = "-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----"
    public_key_digest = (
        "sha256:" + hashlib.sha256(public_key.encode("utf-8")).hexdigest()
    )
    monkeypatch.setattr(settings, "routstr_attestation_public_key", public_key)
    monkeypatch.setattr(settings, "routstr_attestation_public_key_path", "")
    monkeypatch.setattr(
        settings,
        "routstr_tee_client_confidentiality_boundary",
        "attested-tls-termination",
    )
    monkeypatch.setattr(
        "routstr.core.attestation.get_routstr_tee_readiness",
        lambda: {
            "ready": True,
            "failure_reason": None,
            "attestation_evidence_digest": VALID_QUOTE_DIGEST,
            "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "local_verification": {"verified": True},
        },
    )

    status = get_public_routstr_tee_status()

    assert status["client_confidentiality"] == {
        "mode": "attested-tls-termination",
        "tls_terminates_in_attested_tee": True,
        "inbound_ehbp_ohttp_request_decryption": False,
        "attested_tls_public_key_digest": public_key_digest,
    }


def test_public_routstr_tee_status_drops_unverified_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.core.attestation.get_routstr_tee_readiness",
        lambda: {
            "ready": False,
            "failure_reason": "verification failed",
            "attestation_evidence_digest": VALID_QUOTE_DIGEST,
            "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "local_verification": {
                "verified": False,
                "failure_reason": "verification failed",
                "verified_claims": {
                    "tee_report_nonce": "nonce",
                    "debug": _NonSerializableProofClaim(),
                },
            },
        },
    )

    status = get_public_routstr_tee_status()

    assert status["ready"] is False
    local_verification = status["local_verification"]
    assert local_verification["verified"] is False
    assert "failure_reason_digest" in local_verification
    assert "verified_claims_digest" not in local_verification
    assert "proof_claims" not in local_verification


def test_public_routstr_tee_status_publishes_hpke_key_config_digest_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified_claims: dict[str, object] = {
        "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
        "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
        "routstr_code_measurement": VALID_CODE_MEASUREMENT,
        "routstr_config_measurement": VALID_POLICY_DIGEST,
        "verification_steps": _verification_steps(),
        **_proof_claims(),
    }
    monkeypatch.setattr(
        "routstr.core.attestation.get_routstr_tee_readiness",
        lambda: {
            "ready": True,
            "failure_reason": None,
            "attestation_evidence_digest": VALID_QUOTE_DIGEST,
            "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "local_verification": {
                "verified": True,
                "verifier": "unit-test-routstr-tee-verifier",
                "verified_at": 1_700_000_000,
                "expires_at": 4_102_444_800,
                "failure_reason": None,
                "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                "verified_claims": verified_claims,
            },
        },
    )

    status = get_public_routstr_tee_status()

    proof_claims = status["local_verification"]["proof_claims"]
    assert proof_claims["hpke_key_config_digest"] == VALID_HPKE_KEY_CONFIG_DIGEST


def test_public_routstr_tee_status_downgrades_malformed_verified_public_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified_claims: dict[str, object] = {
        "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
        "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
        "routstr_code_measurement": VALID_CODE_MEASUREMENT,
        "routstr_config_measurement": VALID_POLICY_DIGEST,
        "verification_steps": _verification_steps(),
        **_proof_claims(),
    }
    verified_claims["tee_attestation_report_digest"] = {
        "api_key": "SECRET_PUBLIC_PROOF_CLAIM"
    }

    monkeypatch.setattr(
        "routstr.core.attestation.get_routstr_tee_readiness",
        lambda: {
            "ready": True,
            "failure_reason": None,
            "attestation_evidence_digest": VALID_QUOTE_DIGEST,
            "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "local_verification": {
                "verified": True,
                "verifier": "unit-test-routstr-tee-verifier",
                "verified_at": 1_700_000_000,
                "expires_at": 4_102_444_800,
                "failure_reason": None,
                "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                "verified_claims": verified_claims,
            },
        },
    )

    status = get_public_routstr_tee_status()

    assert status["ready"] is False
    local_verification = status["local_verification"]
    assert local_verification["verified"] is False
    assert "failure_reason_digest" in local_verification
    assert "verified_claims_digest" not in local_verification
    assert "proof_claims" not in local_verification
    assert "SECRET_PUBLIC_PROOF_CLAIM" not in json.dumps(status)


def test_public_routstr_tee_status_downgrades_future_verified_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = int(time.time())

    monkeypatch.setattr(
        "routstr.core.attestation.get_routstr_tee_readiness",
        lambda: {
            "ready": True,
            "failure_reason": None,
            "attestation_evidence_digest": VALID_QUOTE_DIGEST,
            "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "local_verification": {
                "verified": True,
                "verifier": "unit-test-routstr-tee-verifier",
                "verified_at": now + 60,
                "expires_at": now + 300,
                "failure_reason": None,
                "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                "verified_claims": {
                    "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
                    "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
                    "routstr_code_measurement": VALID_CODE_MEASUREMENT,
                    "routstr_config_measurement": VALID_POLICY_DIGEST,
                    "verification_steps": _verification_steps(),
                    **_proof_claims(),
                },
            },
        },
    )

    status = get_public_routstr_tee_status()

    local_verification = status["local_verification"]
    assert local_verification["verified"] is False
    assert local_verification["verified_at"] is None
    assert local_verification["expires_at"] is None
    assert "failure_reason_digest" in local_verification


def test_routstr_attestation_statement_strips_provider_url_userinfo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        assert kwargs == {
            "include_provider_urls": True,
            "include_routstr_tee": False,
        }
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "https://operator:secret-token@inference.tinfoil.sh/v1",
                    "db_id": 7,
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["secure-model"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    provider = statement["routing_policy"]["providers"][0]
    assert provider["base_url"] == "https://inference.tinfoil.sh/v1"
    encoded_statement = json.dumps(statement, sort_keys=True)
    assert "secret-token" not in encoded_statement
    assert "operator:" not in encoded_statement


def test_routstr_attestation_statement_drops_tinfoil_non_default_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        assert kwargs == {
            "include_provider_urls": True,
            "include_routstr_tee": False,
        }
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "https://attacker.example/v1",
                    "db_id": 7,
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["secure-model"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    provider = statement["routing_policy"]["providers"][0]
    assert provider["base_url"] is None


def test_routstr_attestation_statement_drops_privatemode_remote_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        assert kwargs == {
            "include_provider_urls": True,
            "include_routstr_tee": False,
        }
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "privatemode",
                    "upstream_name": "privatemode-local",
                    "base_url": "https://api.privatemode.ai/v1",
                    "db_id": 8,
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "privatemode",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["privatemode/secure-model"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    provider = statement["routing_policy"]["providers"][0]
    assert provider["base_url"] is None


def test_routstr_attestation_statement_binds_client_confidentiality_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        assert kwargs == {
            "include_provider_urls": True,
            "include_routstr_tee": False,
        }
        return {
            "mode": "required",
            "required": True,
            "providers": [],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_client_confidentiality_boundary",
        "attested-tls-termination",
        raising=False,
    )
    monkeypatch.setattr(
        settings,
        "routstr_attestation_public_key",
        "-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----",
    )
    monkeypatch.setattr(settings, "routstr_attestation_public_key_path", "")

    statement = get_routstr_attestation_statement()
    public_key_digest = (
        "sha256:"
        + hashlib.sha256(
            b"-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----"
        ).hexdigest()
    )

    assert statement["routing_policy"]["client_confidentiality"] == {
        "mode": "attested-tls-termination",
        "tls_terminates_in_attested_tee": True,
        "inbound_ehbp_ohttp_request_decryption": False,
        "attested_tls_public_key_digest": public_key_digest,
    }
    assert statement["routing_policy_digest"] == _json_digest(
        statement["routing_policy"]
    )


def test_routstr_attestation_statement_binds_exact_routable_model_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        assert kwargs == {
            "include_provider_urls": True,
            "include_routstr_tee": False,
        }
        return {
            "mode": "required",
            "required": True,
            "routable_with_full_attestation": {
                "tinfoil": ["secure-model"],
                "ppq-private": ["private/gpt-oss-120b"],
                "privatemode": ["privatemode/llama"],
            },
            "providers": [],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    assert statement["routing_policy"]["routable_with_full_attestation"] == {
        "tinfoil": ["secure-model"],
        "ppq-private": ["private/gpt-oss-120b"],
        "privatemode": ["privatemode/llama"],
    }
    assert statement["routing_policy_digest"] == _json_digest(
        statement["routing_policy"]
    )


def test_routstr_attestation_statement_does_not_synthesize_exact_map_without_embedded_tee(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    calls: list[bool] = []

    def fake_routable_with_full_attestation(
        *,
        routstr_tee_ready: bool,
    ) -> dict[str, list[str]]:
        calls.append(routstr_tee_ready)
        if not routstr_tee_ready:
            return {"tinfoil": [], "ppq-private": [], "privatemode": []}
        return {
            "tinfoil": ["secure-model"],
            "ppq-private": [],
            "privatemode": [],
        }

    monkeypatch.setattr(
        "routstr.proxy._routable_with_full_attestation",
        fake_routable_with_full_attestation,
    )

    statement = get_routstr_attestation_statement()

    assert calls == [False]
    assert statement["routing_policy"]["routable_with_full_attestation"] == {
        "tinfoil": [],
        "ppq-private": [],
        "privatemode": [],
    }


def test_routstr_attestation_statement_clears_malformed_routable_model_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        assert kwargs == {
            "include_provider_urls": True,
            "include_routstr_tee": False,
        }
        return {
            "mode": "required",
            "required": True,
            "routable_with_full_attestation": {
                "tinfoil": ["secure-model", " SECURE-MODEL "],
                "ppq-private": ["private/gpt-oss-120b"],
                "privatemode": ["privatemode/llama"],
            },
            "providers": [],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    assert statement["routing_policy"]["routable_with_full_attestation"] == {
        "tinfoil": [],
        "ppq-private": [],
        "privatemode": [],
    }


def test_routstr_attestation_statement_downgrades_missing_client_confidentiality_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        assert kwargs == {
            "include_provider_urls": True,
            "include_routstr_tee": False,
        }
        return {
            "mode": "required",
            "required": True,
            "providers": [],
        }

    _configure_routstr_tee_command_stubs(monkeypatch)
    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )
    monkeypatch.setattr(
        "routstr.core.attestation.time.time",
        lambda: 1_700_000_010,
    )
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    monkeypatch.setattr(settings, "routstr_tee_client_confidentiality_boundary", "")

    statement = get_routstr_attestation_statement()

    assert (
        statement["routing_policy"]["client_confidentiality"]["mode"] == "unspecified"
    )
    assert statement["tee"]["local_verification"]["verified"] is False
    assert "failure_reason_digest" in statement["tee"]["local_verification"]


def test_routstr_attestation_statement_drops_malformed_provider_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        assert kwargs == {
            "include_provider_urls": True,
            "include_routstr_tee": False,
        }
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "relative/private/path",
                    "db_id": 7,
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["secure-model"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    provider = statement["routing_policy"]["providers"][0]
    assert provider["base_url"] is None


def test_routstr_attestation_statement_drops_provider_base_url_with_invalid_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        assert kwargs == {
            "include_provider_urls": True,
            "include_routstr_tee": False,
        }
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "https://verified.example:invalid/v1",
                    "db_id": 7,
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["secure-model"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    provider = statement["routing_policy"]["providers"][0]
    assert provider["base_url"] is None


def test_routstr_attestation_statement_drops_provider_base_url_malformed_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        assert kwargs == {
            "include_provider_urls": True,
            "include_routstr_tee": False,
        }
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "https://[verified.example/v1",
                    "db_id": 7,
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["secure-model"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    provider = statement["routing_policy"]["providers"][0]
    assert provider["base_url"] is None


def test_routstr_attestation_statement_drops_provider_base_url_with_secret_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        assert kwargs == {
            "include_provider_urls": True,
            "include_routstr_tee": False,
        }
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": (
                        "https://verified.example/v1?"
                        "tenant=one&api_key=SECRET_PROVIDER_KEY"
                    ),
                    "db_id": 7,
                    "confidentiality": {
                        "enabled": True,
                        "verified": False,
                        "mode": "tinfoil",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "model_ids": ["secure-model"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    provider = statement["routing_policy"]["providers"][0]
    assert provider["base_url"] is None
    serialized = json.dumps(statement, sort_keys=True)
    assert "SECRET_PROVIDER_KEY" not in serialized
    assert "api_key" not in serialized


def test_routstr_attestation_statement_drops_provider_base_url_duplicate_secret_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        assert kwargs == {
            "include_provider_urls": True,
            "include_routstr_tee": False,
        }
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": (
                        "https://verified.example/v1?"
                        "tenant=sk-secret-provider-token&tenant=public"
                    ),
                    "db_id": 7,
                    "confidentiality": {
                        "enabled": True,
                        "verified": False,
                        "mode": "tinfoil",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "model_ids": ["secure-model"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    provider = statement["routing_policy"]["providers"][0]
    assert provider["base_url"] is None
    serialized = json.dumps(statement, sort_keys=True)
    assert "sk-secret-provider-token" not in serialized


def test_routstr_attestation_statement_drops_non_https_remote_provider_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        assert kwargs == {
            "include_provider_urls": True,
            "include_routstr_tee": False,
        }
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "http://verified.example/v1",
                    "db_id": 7,
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["secure-model"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    provider = statement["routing_policy"]["providers"][0]
    assert provider["base_url"] is None


def test_routstr_attestation_statement_rejects_malformed_policy_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        return {
            "mode": {"raw_prompt": "SECRET_MODE"},
            "required": "false",
            "providers": [
                {
                    "provider_type": {"api_key": "SECRET_PROVIDER"},
                    "upstream_name": {"raw_prompt": "SECRET_UPSTREAM"},
                    "base_url": "https://verified.example/v1",
                    "db_id": {"token": "SECRET_DB_ID"},
                    "confidentiality": {
                        "enabled": True,
                        "verified": False,
                        "mode": "tinfoil",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "model_ids": ["secure-model"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    assert statement["routing_policy"]["mode"] == "disabled"
    assert statement["routing_policy"]["required"] is False
    assert statement["routing_policy"]["providers"] == []
    serialized = json.dumps(statement, sort_keys=True)
    assert "SECRET_MODE" not in serialized
    assert "SECRET_PROVIDER" not in serialized
    assert "SECRET_UPSTREAM" not in serialized
    assert "SECRET_DB_ID" not in serialized


def test_routstr_attestation_statement_filters_malformed_provider_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "https://verified.example/v1",
                    "db_id": 7,
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": "sha256:" + ("d" * 64),
                        "require_model_attestations": True,
                        "model_attestation_targets": {
                            "secure-model": {
                                "repo": "tinfoilsh/confidential-secure-model",
                                "expected_release_digest": "sha256:" + ("4" * 64),
                            },
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["secure-model", {"api_key": "SECRET_SELECTOR"}],
                        "model_id_prefixes": ["tinfoil/", 123],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    confidentiality = statement["routing_policy"]["providers"][0]["confidentiality"]
    assert confidentiality["model_ids"] == ["secure-model"]
    assert confidentiality["model_id_prefixes"] == ["tinfoil/"]
    assert "SECRET_SELECTOR" not in json.dumps(statement, sort_keys=True)


def test_routstr_attestation_statement_drops_case_variant_provider_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "https://verified.example/v1",
                    "db_id": 7,
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": "sha256:" + ("d" * 64),
                        "require_model_attestations": True,
                        "model_attestation_targets": {
                            "secure-model": {
                                "repo": "tinfoilsh/confidential-secure-model",
                                "expected_release_digest": "sha256:" + ("4" * 64),
                            },
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": False,
                        "mode": "tinfoil",
                        "model_ids": ["secure-model", "Secure-Model"],
                        "model_id_prefixes": ["tinfoil/", "TINFOIL/"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    confidentiality = statement["routing_policy"]["providers"][0]["confidentiality"]
    assert confidentiality["model_ids"] == []
    assert confidentiality["model_id_prefixes"] == []


def test_routstr_attestation_statement_downgrades_verified_malformed_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_proof_claims = _provider_proof_claims()

    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "https://verified.example/v1",
                    "db_id": 7,
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": "sha256:" + ("d" * 64),
                        "require_model_attestations": True,
                        "model_attestation_targets": {
                            "secure-model": {
                                "repo": "tinfoilsh/confidential-secure-model",
                                "expected_release_digest": "sha256:" + ("4" * 64),
                            },
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                        "verified_claims_digest": VALID_PROVIDER_CLAIMS_DIGEST,
                        "proof_claims": provider_proof_claims,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": [
                            "secure-model",
                            {"api_key": "SECRET_SELECTOR"},
                        ],
                        "model_id_prefixes": ["tinfoil/", 123],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    confidentiality = statement["routing_policy"]["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert confidentiality["verified_at"] is None
    assert confidentiality["expires_at"] is None
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality
    assert "SECRET_SELECTOR" not in json.dumps(statement, sort_keys=True)


def test_routstr_attestation_statement_downgrades_known_provider_prefix_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_proof_claims = _provider_proof_claims()

    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "https://verified.example/v1",
                    "db_id": 7,
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                        "verified_claims_digest": VALID_PROVIDER_CLAIMS_DIGEST,
                        "proof_claims": provider_proof_claims,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["secure-model"],
                        "model_id_prefixes": ["tinfoil/"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    confidentiality = statement["routing_policy"]["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_routstr_attestation_statement_binds_provider_public_proof_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_proof_claims = _provider_proof_claims()
    confidentiality_policy = {
        "repo": "tinfoilsh/confidential-model-router",
        "expected_release_digest": _named_digest("tinfoil-router-release"),
        "require_model_attestations": True,
        "model_attestation_targets": {
            "secure-model": {
                "repo": "tinfoilsh/confidential-secure-model",
                "expected_release_digest": _named_digest("secure-model-release"),
            },
        },
    }
    provider_policy_digest = _provider_policy_digest(
        policy=confidentiality_policy,
        model_ids=["secure-model"],
    )
    provider_proof_claims["payload_policy_digest"] = provider_policy_digest
    provider_proof_claims["payload_evidence_digest"] = (
        VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST
    )

    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "https://inference.tinfoil.sh/v1",
                    "db_id": 7,
                    "confidentiality_policy": confidentiality_policy,
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": provider_policy_digest,
                        "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                        "verified_claims_digest": VALID_PROVIDER_CLAIMS_DIGEST,
                        "proof_claims": provider_proof_claims,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["secure-model"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    confidentiality = statement["routing_policy"]["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is True
    assert confidentiality["verified_claims_digest"] == VALID_PROVIDER_CLAIMS_DIGEST
    assert confidentiality["proof_claims"] == provider_proof_claims


def test_safe_provider_confidentiality_accepts_runtime_provider_evidence_digest() -> (
    None
):
    provider_proof_claims = _provider_proof_claims()
    confidentiality_policy = {
        "repo": "tinfoilsh/confidential-model-router",
        "expected_release_digest": _named_digest("tinfoil-router-release"),
        "require_model_attestations": True,
        "model_attestation_targets": {
            "secure-model": {
                "repo": "tinfoilsh/confidential-secure-model",
                "expected_release_digest": _named_digest("secure-model-release"),
            },
        },
    }
    provider_policy_digest = _provider_policy_digest(
        policy=confidentiality_policy,
        model_ids=["secure-model"],
    )
    runtime_claims = {
        **provider_proof_claims,
        "payload_policy_digest": provider_policy_digest,
        "runtime_evidence_digest": "placeholder",
        "payload_verification_nonce": "runtime-nonce",
    }
    runtime_digest = _json_digest(
        {
            key: value
            for key, value in runtime_claims.items()
            if key
            not in {
                "payload_evidence_digest",
                "payload_policy_digest",
                "payload_verification_nonce",
                "runtime_evidence_digest",
            }
        }
    )
    runtime_claims["runtime_evidence_digest"] = runtime_digest
    provider_proof_claims["payload_policy_digest"] = provider_policy_digest

    confidentiality = _safe_provider_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": provider_policy_digest,
            "evidence_digest": runtime_digest,
            "verified_claims_digest": VALID_PROVIDER_CLAIMS_DIGEST,
            "proof_claims": provider_proof_claims,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "model_ids": ["secure-model"],
            "_verified_claims_for_validation": runtime_claims,
        },
        provider_type="tinfoil",
        provider_base_url="https://inference.tinfoil.sh/v1",
        public_policy=confidentiality_policy,
    )
    assert confidentiality["verified"] is True
    assert confidentiality["evidence_digest"] == runtime_digest
    assert confidentiality["proof_claims"] == provider_proof_claims
    assert "_verified_claims_for_validation" not in json.dumps(confidentiality)
    assert "runtime-nonce" not in json.dumps(confidentiality)


def test_routstr_attestation_statement_accepts_ppq_raw_policy_digest_with_public_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_policy = {
        "repo": "ppq-ai/private-tee",
        "attestation_bundle_url": "https://api.ppq.ai/private",
        "release_digest": _named_digest("ppq-provider-release"),
        "code_measurement_fingerprint": _named_digest("ppq-provider-code-measurement"),
        "enclave_measurement_fingerprint": _named_digest(
            "ppq-provider-enclave-measurement"
        ),
        "require_model_attestations": True,
        "model_attestation_targets": {
            "private/gpt-oss-120b": {
                "repo": "tinfoilsh/confidential-gpt-oss-120b",
                "expected_release_digest": _named_digest("ppq-backend-release"),
            },
        },
    }
    public_policy = {
        "repo": "ppq-ai/private-tee",
        "attestation_bundle_url_digest": _json_digest("https://api.ppq.ai/private"),
        "release_digest": _named_digest("ppq-provider-release"),
        "code_measurement_fingerprint": _named_digest("ppq-provider-code-measurement"),
        "enclave_measurement_fingerprint": _named_digest(
            "ppq-provider-enclave-measurement"
        ),
        "require_model_attestations": True,
        "model_attestation_targets": {
            "private/gpt-oss-120b": {
                "repo": "tinfoilsh/confidential-gpt-oss-120b",
                "expected_release_digest": _named_digest("ppq-backend-release"),
            },
        },
    }
    provider_policy_digest = _provider_policy_digest(
        provider_type="ppq-private",
        mode="ppq-private-tee",
        base_url="https://api.ppq.ai/private/v1",
        policy=raw_policy,
        model_ids=["private/gpt-oss-120b"],
    )
    provider_proof_claims = _ppq_private_provider_proof_claims(provider_policy_digest)

    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "ppq-private",
                    "upstream_name": "ppq-private",
                    "base_url": "https://api.ppq.ai/private/v1",
                    "db_id": 8,
                    "confidentiality_policy": public_policy,
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "ppq-private-tee",
                        "verifier": "unit-test-ppq-verifier",
                        "policy_digest": provider_policy_digest,
                        "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                        "verified_claims_digest": VALID_PROVIDER_CLAIMS_DIGEST,
                        "proof_claims": provider_proof_claims,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["private/gpt-oss-120b"],
                        "model_id_prefixes": [],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    confidentiality = statement["routing_policy"]["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is True
    assert confidentiality["policy_digest"] == provider_policy_digest
    assert confidentiality["proof_claims"] == provider_proof_claims


def test_routstr_attestation_statement_downgrades_secret_bearing_provider_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_proof_claims = _provider_proof_claims()
    confidentiality_policy = {
        "repo": "tinfoilsh/confidential-model-router",
        "expected_release_digest": _named_digest("tinfoil-router-release"),
        "require_model_attestations": True,
        "model_attestation_targets": {
            "secure-model": {
                "repo": "tinfoilsh/confidential-secure-model",
                "expected_release_digest": _named_digest("secure-model-release"),
            },
        },
    }
    provider_policy_digest = _provider_policy_digest(
        policy=confidentiality_policy,
        model_ids=["secure-model"],
    )
    provider_proof_claims["payload_policy_digest"] = provider_policy_digest

    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "https://inference.tinfoil.sh/v1",
                    "db_id": 7,
                    "confidentiality_policy": confidentiality_policy,
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": provider_policy_digest,
                        "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                        "verified_claims_digest": VALID_PROVIDER_CLAIMS_DIGEST,
                        "proof_claims": {
                            **provider_proof_claims,
                            "api_key": "SECRET_PROVIDER_KEY",
                        },
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["secure-model"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    confidentiality = statement["routing_policy"]["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is False
    assert "proof_claims" not in confidentiality
    assert "SECRET_PROVIDER_KEY" not in json.dumps(statement, sort_keys=True)


def test_routstr_attestation_statement_downgrades_provider_policy_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    confidentiality_policy = {
        "repo": "tinfoilsh/confidential-model-router",
        "expected_release_digest": "sha256:" + ("d" * 64),
        "require_model_attestations": True,
        "model_attestation_targets": {
            "secure-model": {
                "repo": "tinfoilsh/confidential-secure-model",
                "expected_release_digest": "sha256:" + ("4" * 64),
            },
        },
    }

    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "https://inference.tinfoil.sh/v1",
                    "db_id": 7,
                    "confidentiality_policy": confidentiality_policy,
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_OTHER_POLICY_DIGEST,
                        "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                        "verified_claims_digest": VALID_PROVIDER_CLAIMS_DIGEST,
                        "proof_claims": _provider_proof_claims(),
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["secure-model"],
                        "model_id_prefixes": [],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    confidentiality = statement["routing_policy"]["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert "policy_digest" not in confidentiality
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_routstr_attestation_statement_downgrades_policy_mismatched_public_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_proof_claims = _provider_proof_claims()

    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "https://verified.example/v1",
                    "db_id": 7,
                    "confidentiality_policy": {
                        "repo": "tinfoilsh/confidential-model-router",
                        "expected_release_digest": "sha256:" + ("0" * 64),
                        "require_model_attestations": True,
                        "model_attestation_targets": {
                            "secure-model": {
                                "repo": "tinfoilsh/confidential-secure-model",
                                "expected_release_digest": "sha256:" + ("4" * 64),
                            },
                        },
                    },
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                        "verified_claims_digest": VALID_PROVIDER_CLAIMS_DIGEST,
                        "proof_claims": provider_proof_claims,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["secure-model"],
                        "model_id_prefixes": [],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    confidentiality = statement["routing_policy"]["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_routstr_attestation_statement_downgrades_tinfoil_selected_model_without_model_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "https://inference.tinfoil.sh/v1",
                    "db_id": 7,
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                        "verified_claims_digest": VALID_PROVIDER_CLAIMS_DIGEST,
                        "proof_claims": _provider_proof_claims(),
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["tinfoil/secure-model"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    confidentiality = statement["routing_policy"]["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_routstr_attestation_statement_downgrades_malformed_present_proof_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_proof_claims = _provider_proof_claims()
    provider_proof_claims["tls_public_key_fingerprint_sha256"] = "not-a-digest"
    provider_proof_claims["tls_public_key"] = "sha256:" + ("e" * 64)

    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "https://verified.example/v1",
                    "db_id": 7,
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                        "verified_claims_digest": VALID_PROVIDER_CLAIMS_DIGEST,
                        "proof_claims": provider_proof_claims,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["secure-model"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    confidentiality = statement["routing_policy"]["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert confidentiality["verified_at"] is None
    assert confidentiality["expires_at"] is None
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_routstr_attestation_provider_confidentiality_rejects_non_list_model_selectors() -> (
    None
):
    public_status = _safe_provider_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_PROVIDER_CLAIMS_DIGEST,
            "proof_claims": _provider_proof_claims(),
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "model_ids": ("secure-model",),
            "model_id_prefixes": [],
        },
        provider_type="tinfoil",
    )

    assert public_status["verified"] is False
    assert public_status["verifier"] is None
    assert "policy_digest" not in public_status
    assert public_status["evidence_digest"] is None
    assert "verified_claims_digest" not in public_status
    assert "proof_claims" not in public_status


def test_routstr_attestation_provider_confidentiality_rejects_payload_policy_mismatch() -> (
    None
):
    proof_claims = _provider_proof_claims()
    proof_claims["payload_policy_digest"] = VALID_OTHER_POLICY_DIGEST

    public_status = _safe_provider_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_PROVIDER_CLAIMS_DIGEST,
            "proof_claims": proof_claims,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "model_ids": ["secure-model"],
            "model_id_prefixes": [],
        },
        provider_type="tinfoil",
    )

    assert public_status["verified"] is False
    assert public_status["verifier"] is None
    assert "policy_digest" not in public_status
    assert public_status["evidence_digest"] is None
    assert "verified_claims_digest" not in public_status
    assert "proof_claims" not in public_status


def test_routstr_attestation_provider_confidentiality_rejects_payload_evidence_mismatch() -> (
    None
):
    proof_claims = _provider_proof_claims()
    proof_claims["payload_evidence_digest"] = VALID_QUOTE_DIGEST
    confidentiality_policy = {
        "repo": "tinfoilsh/confidential-model-router",
        "expected_release_digest": _named_digest("tinfoil-router-release"),
        "require_model_attestations": True,
        "model_attestation_targets": {
            "secure-model": {
                "repo": "tinfoilsh/confidential-secure-model",
                "expected_release_digest": _named_digest("secure-model-release"),
            },
        },
    }

    public_status = _safe_provider_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": VALID_POLICY_DIGEST,
            "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
            "verified_claims_digest": VALID_PROVIDER_CLAIMS_DIGEST,
            "proof_claims": proof_claims,
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "model_ids": ["secure-model"],
            "model_id_prefixes": [],
        },
        provider_type="tinfoil",
        public_policy=confidentiality_policy,
    )

    assert public_status["verified"] is False
    assert public_status["verifier"] is None
    assert "policy_digest" not in public_status
    assert public_status["evidence_digest"] is None
    assert "verified_claims_digest" not in public_status
    assert "proof_claims" not in public_status


def test_routstr_attestation_statement_requires_strict_provider_status_booleans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "https://verified.example/v1",
                    "db_id": 7,
                    "confidentiality": {
                        "enabled": "false",
                        "verified": "false",
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["secure-model"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    confidentiality = statement["routing_policy"]["providers"][0]["confidentiality"]
    assert confidentiality["enabled"] is False
    assert confidentiality["verified"] is False


def test_routstr_attestation_statement_downgrades_malformed_provider_proof_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "https://verified.example/v1",
                    "db_id": 7,
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": {"raw_prompt": "SECRET_MODE"},
                        "verifier": {"api_key": "SECRET_VERIFIER"},
                        "policy_digest": "sha256:policy",
                        "evidence_digest": "sha256:evidence",
                        "verified_at": "1700000000",
                        "expires_at": "4102444800",
                        "model_ids": ["secure-model"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    confidentiality = statement["routing_policy"]["providers"][0]["confidentiality"]
    assert confidentiality == {
        "enabled": True,
        "verified": False,
        "mode": "none",
        "verifier": None,
        "evidence_digest": None,
        "verified_at": None,
        "expires_at": None,
        "model_ids": ["secure-model"],
        "model_id_prefixes": [],
    }
    serialized = json.dumps(statement, sort_keys=True)
    assert "SECRET_MODE" not in serialized
    assert "SECRET_VERIFIER" not in serialized


def test_routstr_attestation_statement_downgrades_provider_mode_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "custom",
                    "upstream_name": "secure-custom",
                    "base_url": "https://verified.example/v1",
                    "db_id": 7,
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_LOCAL_RUNTIME_EVIDENCE_DIGEST,
                        "verified_at": 1_700_000_000,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["secure-model"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    confidentiality = statement["routing_policy"]["providers"][0]["confidentiality"]
    assert confidentiality["enabled"] is True
    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert confidentiality["verified_at"] is None
    assert confidentiality["expires_at"] is None


def test_routstr_attestation_statement_downgrades_provider_without_verified_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "https://verified.example/v1",
                    "db_id": 7,
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_QUOTE_DIGEST,
                        "verified_at": None,
                        "expires_at": 4_102_444_800,
                        "model_ids": ["secure-model"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    confidentiality = statement["routing_policy"]["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert confidentiality["verified_at"] is None
    assert confidentiality["expires_at"] is None


def test_routstr_attestation_statement_downgrades_expired_provider_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "https://verified.example/v1",
                    "db_id": 7,
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_QUOTE_DIGEST,
                        "verified_claims_digest": VALID_PROVIDER_CLAIMS_DIGEST,
                        "proof_claims": _provider_proof_claims(),
                        "verified_at": 1_700_000_000,
                        "expires_at": 1,
                        "model_ids": ["secure-model"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    confidentiality = statement["routing_policy"]["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert confidentiality["verified_at"] is None
    assert confidentiality["expires_at"] is None
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_routstr_attestation_statement_downgrades_future_provider_verified_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = int(time.time())

    def fake_get_confidentiality_status(**kwargs: object) -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "upstream_name": "secure-tinfoil",
                    "base_url": "https://verified.example/v1",
                    "db_id": 7,
                    "confidentiality": {
                        "enabled": True,
                        "verified": True,
                        "mode": "tinfoil",
                        "verifier": "unit-test-verifier",
                        "policy_digest": VALID_POLICY_DIGEST,
                        "evidence_digest": VALID_QUOTE_DIGEST,
                        "verified_claims_digest": VALID_PROVIDER_CLAIMS_DIGEST,
                        "proof_claims": _provider_proof_claims(),
                        "verified_at": now + 60,
                        "expires_at": now + 300,
                        "model_ids": ["secure-model"],
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_get_confidentiality_status,
    )

    statement = get_routstr_attestation_statement()

    confidentiality = statement["routing_policy"]["providers"][0]["confidentiality"]
    assert confidentiality["verified"] is False
    assert confidentiality["verifier"] is None
    assert confidentiality["evidence_digest"] is None
    assert confidentiality["verified_at"] is None
    assert confidentiality["expires_at"] is None
    assert "verified_claims_digest" not in confidentiality
    assert "proof_claims" not in confidentiality


def test_routstr_tee_verifier_rejects_missing_public_key_binding() -> None:
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }

    with pytest.raises(
        ValueError,
        match="public_key_digest does not match Routstr TEE verifier payload",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_rejects_non_string_verifier_identity() -> None:
    verifier_result = {
        "verified": True,
        "verifier": 123,
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }

    with pytest.raises(
        ValueError,
        match="Routstr TEE verifier identity must be a string",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_requires_configured_public_key_digest() -> None:
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }

    with pytest.raises(
        ValueError,
        match="Routstr TEE public key digest is required",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=None,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_requires_concrete_proof_artifacts() -> None:
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            "attestation_document_format": "tdx_quote",
        },
    }

    with pytest.raises(
        ValueError,
        match="tee_attestation_report_digest claim is required",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_requires_privatemode_proxy_inside_attested_runtime() -> (
    None
):
    routing_policy = {
        "mode": "required",
        "required": True,
        "providers": [
            {
                "provider_type": "privatemode",
                "base_url": "http://127.0.0.1:8080/v1",
                "confidentiality": {
                    "verified": True,
                    "mode": "privatemode",
                },
                "confidentiality_policy": {
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                },
            }
        ],
    }
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }

    with pytest.raises(
        ValueError,
        match="attested_local_artifacts.privatemode_proxy_binary claim is required",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            routing_policy=routing_policy,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_accepts_ppq_private_without_local_proxy_artifact() -> (
    None
):
    routing_policy = {
        "mode": "required",
        "required": True,
        "providers": [
            {
                "provider_type": "ppq-private",
                "base_url": "https://api.ppq.ai/private/v1",
                "confidentiality": {
                    "verified": True,
                    "mode": "ppq-private-tee",
                },
                "confidentiality_policy": {},
            }
        ],
    }
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }

    status = _validate_routstr_tee_verifier_result(
        verifier_result=verifier_result,
        verifier_policy={
            "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
        },
        routing_policy_digest=VALID_POLICY_DIGEST,
        routing_policy=routing_policy,
        attestation_evidence_digest=VALID_QUOTE_DIGEST,
        attestation_document_format="tdx_quote",
        hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
        hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
        public_key_digest=VALID_PUBLIC_KEY_DIGEST,
        verification_nonce="nonce",
    )

    assert status["verified"] is True


def test_routstr_tee_verifier_rejects_placeholder_local_proxy_policy_pin() -> None:
    routing_policy = {
        "mode": "required",
        "required": True,
        "providers": [
            {
                "provider_type": "privatemode",
                "base_url": "http://127.0.0.1:8080/v1",
                "confidentiality": {
                    "verified": True,
                    "mode": "privatemode",
                },
                "confidentiality_policy": {
                    "proxy_binary_digest": "sha256:" + ("7" * 64),
                },
            }
        ],
    }
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "attested_local_artifacts": {
                "privatemode_proxy_binary": VALID_PRIVATEMODE_PROXY_DIGEST,
            },
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }

    with pytest.raises(
        ValueError,
        match=(
            "confidentiality_policy.proxy_binary_digest must not be a "
            "placeholder digest"
        ),
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            routing_policy=routing_policy,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_accepts_privatemode_proxy_attested_runtime_binding() -> (
    None
):
    routing_policy = {
        "mode": "required",
        "required": True,
        "providers": [
            {
                "provider_type": "privatemode",
                "base_url": "http://127.0.0.1:8080/v1",
                "confidentiality": {
                    "verified": True,
                    "mode": "privatemode",
                },
                "confidentiality_policy": {
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                },
            }
        ],
    }
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "attested_local_artifacts": {
                "privatemode_proxy_binary": VALID_PRIVATEMODE_PROXY_DIGEST,
            },
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }

    result = _validate_routstr_tee_verifier_result(
        verifier_result=verifier_result,
        verifier_policy={
            "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
        },
        routing_policy_digest=VALID_POLICY_DIGEST,
        routing_policy=routing_policy,
        attestation_evidence_digest=VALID_QUOTE_DIGEST,
        attestation_document_format="tdx_quote",
        hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
        hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
        public_key_digest=VALID_PUBLIC_KEY_DIGEST,
        verification_nonce="nonce",
    )

    assert result["verified_claims"]["attested_local_artifacts"] == {
        "privatemode_proxy_binary": VALID_PRIVATEMODE_PROXY_DIGEST,
    }


def test_private_local_artifacts_snapshot_uses_live_provider_policy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy_binary = tmp_path / "privatemode-proxy"
    proxy_binary.write_bytes(b"privatemode proxy binary")
    proxy_digest = _sha256_file_digest(proxy_binary)
    provider = SimpleNamespace(
        provider_type="privatemode",
        confidentiality_status=lambda: SimpleNamespace(
            verified=True,
            mode="privatemode",
        ),
        confidentiality_policy=lambda: SimpleNamespace(
            policy={
                "proxy_binary_digest": proxy_digest,
                "proxy_binary_path": str(proxy_binary),
            }
        ),
    )
    monkeypatch.setattr("routstr.proxy._upstreams", [provider])

    assert _private_local_artifacts_snapshot() == {
        "privatemode_proxy_binary": {
            "digest": proxy_digest,
            "path": str(proxy_binary),
        }
    }


def test_routstr_tee_fixture_rejects_public_only_local_proxy_digest() -> None:
    payload = {
        "schema_version": "routstr-tee-verifier-v1",
        "routing_policy": {
            "mode": "required",
            "required": True,
            "providers": [
                {
                    "provider_type": "privatemode",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "confidentiality": {
                        "verified": True,
                        "mode": "privatemode",
                    },
                    "confidentiality_policy": {
                        "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                    },
                }
            ],
        },
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "attestation_document_format": "tdx_quote",
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
        "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
        "verification_nonce": "nonce",
        "tee_report_data_hex": _routstr_tee_report_data_hex(
            routing_policy_digest=VALID_POLICY_DIGEST,
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        ),
        "tee_report_data_digest": _routstr_tee_report_data_digest(
            routing_policy_digest=VALID_POLICY_DIGEST,
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        ),
        "local_artifacts": {},
    }

    result = subprocess.run(
        [sys.executable, str(VERIFIER_STUB_PATH)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "local_artifacts.privatemode_proxy_binary" in result.stderr


def test_routstr_tee_verifier_rejects_conflicting_local_proxy_policy_digests() -> None:
    other_proxy_digest = _named_digest("other-privatemode-proxy-binary")
    routing_policy = {
        "mode": "required",
        "required": True,
        "providers": [
            {
                "provider_type": "privatemode",
                "base_url": "http://127.0.0.1:8080/v1",
                "confidentiality": {
                    "verified": True,
                    "mode": "privatemode",
                },
                "confidentiality_policy": {
                    "proxy_binary_digest": VALID_PRIVATEMODE_PROXY_DIGEST,
                },
            },
            {
                "provider_type": "privatemode",
                "base_url": "http://127.0.0.1:9090/v1",
                "confidentiality": {
                    "verified": True,
                    "mode": "privatemode",
                },
                "confidentiality_policy": {
                    "proxy_binary_digest": other_proxy_digest,
                },
            },
        ],
    }
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "attested_local_artifacts": {
                "privatemode_proxy_binary": VALID_PRIVATEMODE_PROXY_DIGEST,
            },
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }

    with pytest.raises(
        ValueError,
        match="attested_local_artifacts.privatemode_proxy_binary has conflicting routing policy digests",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            routing_policy=routing_policy,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_requires_report_data_binding_to_policy_and_keys() -> None:
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            "tee_attestation_report_digest": VALID_QUOTE_REPORT_DIGEST,
            "tee_certificate_chain_digest": VALID_QUOTE_CERTIFICATE_CHAIN_DIGEST,
            "attestation_document_format": "tdx_quote",
            "tee_report_data_hex": _routstr_tee_report_data_hex(
                routing_policy_digest=VALID_POLICY_DIGEST,
                hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
                hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
                public_key_digest=VALID_PUBLIC_KEY_DIGEST,
                verification_nonce="nonce",
            ),
            "tee_report_nonce": "nonce",
        },
    }

    with pytest.raises(
        ValueError,
        match="tee_report_data_digest claim is required",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_requires_exact_report_data_hex_binding() -> None:
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(include_report_data_hex=False),
        },
    }

    with pytest.raises(
        ValueError,
        match="tee_report_data_hex claim is required",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_rejects_string_coercible_report_data_hex() -> None:
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
            "tee_report_data_hex": _StringCoercibleProofField(
                _routstr_tee_report_data_hex(
                    routing_policy_digest=VALID_POLICY_DIGEST,
                    hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
                    hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
                    public_key_digest=VALID_PUBLIC_KEY_DIGEST,
                    verification_nonce="nonce",
                )
            ),
        },
    }

    with pytest.raises(
        ValueError,
        match="tee_report_data_hex claim must be 128 hex characters",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_rejects_report_data_hex_mismatch() -> None:
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
            "tee_report_data_hex": "00" * 64,
        },
    }

    with pytest.raises(
        ValueError,
        match="tee_report_data_hex claim does not match policy and keys",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_rejects_report_nonce_mismatch() -> None:
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
            "tee_report_nonce": "wrong-nonce",
        },
    }

    with pytest.raises(
        ValueError,
        match="tee_report_nonce does not match Routstr TEE verifier payload",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_requires_report_nonce_for_remote_audit() -> None:
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }
    verifier_result["claims"].pop("tee_report_nonce")

    with pytest.raises(
        ValueError,
        match="tee_report_nonce claim is required",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_requires_report_nonce_digest_for_remote_audit() -> None:
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
        },
    }
    verifier_result["claims"].pop("tee_report_nonce_digest")

    with pytest.raises(
        ValueError,
        match="tee_report_nonce_digest claim is required",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_rejects_report_nonce_digest_mismatch() -> None:
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(),
            "tee_report_nonce_digest": "sha256:" + ("0" * 64),
        },
    }

    with pytest.raises(
        ValueError,
        match="tee_report_nonce_digest does not match Routstr TEE verifier payload",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_verifier_requires_format_claim_to_match_evidence() -> None:
    verifier_result = {
        "verified": True,
        "verifier": "unit-test-routstr-tee-verifier",
        "routing_policy_digest": VALID_POLICY_DIGEST,
        "attestation_evidence_digest": VALID_QUOTE_DIGEST,
        "hpke_key_config_digest": VALID_HPKE_KEY_CONFIG_DIGEST,
        "verification_nonce": "nonce",
        "claims": {
            "hpke_public_key_digest": VALID_HPKE_PUBLIC_KEY_DIGEST,
            "public_key_digest": VALID_PUBLIC_KEY_DIGEST,
            "routstr_code_measurement": VALID_CODE_MEASUREMENT,
            "routstr_config_measurement": VALID_POLICY_DIGEST,
            "verification_steps": _verification_steps(),
            **_proof_claims(attestation_document_format="sev_snp_quote"),
        },
    }

    with pytest.raises(
        ValueError,
        match="attestation_document_format does not match Routstr TEE verifier payload",
    ):
        _validate_routstr_tee_verifier_result(
            verifier_result=verifier_result,
            verifier_policy={
                "expected_routstr_code_measurement": VALID_CODE_MEASUREMENT,
            },
            routing_policy_digest=VALID_POLICY_DIGEST,
            attestation_evidence_digest=VALID_QUOTE_DIGEST,
            attestation_document_format="tdx_quote",
            hpke_key_config_digest=VALID_HPKE_KEY_CONFIG_DIGEST,
            hpke_public_key_digest=VALID_HPKE_PUBLIC_KEY_DIGEST,
            public_key_digest=VALID_PUBLIC_KEY_DIGEST,
            verification_nonce="nonce",
        )


def test_routstr_tee_attestation_command_generates_nonce_bound_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.core.attestation.time.time",
        lambda: 1_700_000_010,
    )
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)
    public_key = "-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----"

    monkeypatch.setattr(settings, "routstr_attestation_document_path", "")
    monkeypatch.setattr(settings, "routstr_attestation_document_format", "tdx_quote")
    monkeypatch.setattr(settings, "routstr_attestation_public_key", public_key)
    monkeypatch.setattr(settings, "routstr_attestation_public_key_path", "")
    monkeypatch.setattr(
        settings,
        "routstr_attestation_hpke_key_config_b64",
        base64.b64encode(hpke_key_config).decode("ascii"),
    )
    monkeypatch.setattr(settings, "routstr_attestation_hpke_key_config_path", "")
    monkeypatch.setattr(
        settings,
        "routstr_tee_attestation_command",
        json.dumps([sys.executable, str(ATTESTATION_STUB_PATH)]),
        raising=False,
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_attestation_artifact_path",
        str(ATTESTATION_STUB_PATH),
        raising=False,
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_attestation_command_digest",
        _sha256_file_digest(ATTESTATION_STUB_PATH),
        raising=False,
    )
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

    from routstr.core.attestation import get_routstr_attestation_statement

    statement = get_routstr_attestation_statement()

    tee = statement["tee"]
    local_verification = tee["local_verification"]
    assert tee["available"] is True
    assert local_verification["verified"] is True
    generated_quote = json.loads(base64.b64decode(tee["attestation_document_b64"]))
    claims = local_verification["proof_claims"]
    assert "verified_claims" not in local_verification
    assert generated_quote["schema_version"] == "fixture-routstr-tee-quote-v1"
    assert generated_quote["report_data_hex"] == claims["tee_report_data_hex"]
    assert generated_quote["report_data_digest"] == claims["tee_report_data_digest"]
    assert generated_quote["verification_nonce"] == claims["tee_report_nonce"]


def test_routstr_tee_attestation_command_rejects_non_string_document_b64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_routstr_tee_command_stubs(monkeypatch)
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_STUB_BEHAVIOR",
        "numeric_document_b64",
    )

    readiness = get_routstr_tee_readiness()

    assert readiness["ready"] is False
    assert "attestation_document_b64 must be a string" in str(
        readiness["failure_reason"] or ""
    )


def test_routstr_tee_attestation_command_rejects_non_string_document_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_routstr_tee_command_stubs(monkeypatch)
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTATION_STUB_BEHAVIOR",
        "numeric_document_format",
    )

    readiness = get_routstr_tee_readiness()

    assert readiness["ready"] is False
    assert "attestation_document_format must be a string" in str(
        readiness["failure_reason"] or ""
    )


def test_cap_attestation_status_is_cached_for_request_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_cap_attestation_status_cache()
    status_payload = {
        "claims_error": None,
        "claims_instance_id": "routstr-core-prod",
        "claims_verified": True,
        "config_ready": True,
        "error": None,
        "instance_id": "cap-org-routstr-core-prod",
        "mode": "password",
        "state": "unlocked",
        "tenant_id": "cap-org-routstr-core-prod",
        "tenant_instance_identity_hash": "a" * 64,
    }
    calls = 0

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, limit: int = -1) -> bytes:
            return json.dumps(status_payload).encode("utf-8")

    def fake_urlopen(*args: object, **kwargs: object) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse()

    monkeypatch.setattr(settings, "routstr_tee_cap_attestation_enabled", True)
    monkeypatch.setattr(
        settings, "routstr_tee_cap_status_url", "http://127.0.0.1:8081/status"
    )
    monkeypatch.setattr(settings, "routstr_tee_cap_tee_domain", "example.tee.test")
    monkeypatch.setattr(settings, "routstr_tee_cap_public_base_url", "")
    monkeypatch.setattr(
        settings,
        "routstr_tee_client_confidentiality_boundary",
        "attested-tls-termination",
    )
    monkeypatch.setattr("routstr.core.attestation.urlopen", fake_urlopen)

    try:
        first = get_routstr_tee_readiness()
        second = get_routstr_tee_readiness()

        assert first["ready"] is True
        assert second["ready"] is True
        assert calls == 1
    finally:
        _clear_cap_attestation_status_cache()


def test_cap_attestation_status_cache_survives_transient_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_cap_attestation_status_cache()
    now_wall = 1_800_000_000
    now_monotonic = 5_000.0
    calls = 0
    status_payload = {
        "claims_error": None,
        "claims_instance_id": "routstr-core-prod",
        "claims_verified": True,
        "config_ready": True,
        "error": None,
        "instance_id": "cap-org-routstr-core-prod",
        "mode": "password",
        "state": "unlocked",
        "tenant_id": "cap-org-routstr-core-prod",
        "tenant_instance_identity_hash": "a" * 64,
    }

    def fake_fetch() -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return status_payload
        raise TimeoutError("status endpoint stalled")

    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    monkeypatch.setattr(settings, "routstr_tee_attestation_required", True)
    monkeypatch.setattr(settings, "routstr_tee_cap_attestation_enabled", True)
    monkeypatch.setattr(settings, "routstr_tee_cap_verifier_max_age_seconds", 300)
    monkeypatch.setattr(settings, "routstr_tee_cap_status_cache_seconds", 30.0)
    monkeypatch.setattr(
        settings, "routstr_tee_cap_status_url", "http://127.0.0.1:8081/status"
    )
    monkeypatch.setattr(settings, "routstr_tee_cap_tee_domain", "example.tee.test")
    monkeypatch.setattr(settings, "routstr_tee_cap_public_base_url", "")
    monkeypatch.setattr(
        settings,
        "routstr_tee_client_confidentiality_boundary",
        "attested-tls-termination",
    )
    monkeypatch.setattr(
        "routstr.core.attestation._fetch_cap_attestation_status", fake_fetch
    )
    monkeypatch.setattr("routstr.core.attestation.time.time", lambda: now_wall)
    monkeypatch.setattr(
        "routstr.core.attestation.time.monotonic", lambda: now_monotonic
    )

    try:
        first = get_routstr_tee_readiness()

        assert first["ready"] is True
        assert first["local_verification"]["verified_at"] == now_wall

        now_wall += 60
        now_monotonic += 60
        second = get_routstr_tee_readiness()

        assert calls == 2
        assert second["ready"] is True
        assert second["failure_reason"] is None
        assert second["local_verification"]["verified_at"] == 1_800_000_000
        assert second["local_verification"]["expires_at"] == 1_800_000_300

        now_wall = 1_800_000_301
        now_monotonic += 241
        expired = get_routstr_tee_readiness()

        assert expired["ready"] is False
        assert "status endpoint stalled" in str(expired["failure_reason"])
    finally:
        _clear_cap_attestation_status_cache()


def test_routstr_tee_attestation_command_rejects_secret_argv_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_routstr_tee_command_stubs(monkeypatch)
    monkeypatch.setattr(
        settings,
        "routstr_tee_attestation_command",
        json.dumps(
            [
                sys.executable,
                str(ATTESTATION_STUB_PATH),
                "--token=sk-local-attestation-secret",
            ]
        ),
        raising=False,
    )
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_STUB_BEHAVIOR", "failure_secret")

    readiness = get_routstr_tee_readiness()

    failure_reason = str(readiness["failure_reason"] or "")
    assert readiness["ready"] is False
    assert "secret-like value must not be embedded" in failure_reason
    assert "sk-local-attestation-secret" not in failure_reason
    assert "Routstr TEE attestation command failed with exit code" not in failure_reason


def test_routstr_tee_attestation_command_requires_digest_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)

    monkeypatch.setattr(settings, "routstr_attestation_document_path", "")
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
    monkeypatch.setattr(
        settings,
        "routstr_tee_attestation_command",
        json.dumps([sys.executable, str(ATTESTATION_STUB_PATH)]),
        raising=False,
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_attestation_artifact_path",
        str(ATTESTATION_STUB_PATH),
        raising=False,
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_attestation_command_digest",
        "",
        raising=False,
    )
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

    readiness = get_routstr_tee_readiness()

    assert readiness["ready"] is False
    assert "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST is required" in str(
        readiness["failure_reason"] or ""
    )


def test_routstr_tee_verifier_rejects_malformed_digest_before_hashing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_path = tmp_path / "quote.bin"
    evidence_path.write_bytes(b"fake-tdx-quote")
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)

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
    monkeypatch.setattr(
        settings,
        "routstr_tee_verifier_command",
        json.dumps([sys.executable, str(tmp_path / "missing-verifier")]),
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_verifier_artifact_path",
        str(tmp_path / "missing-verifier"),
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_verifier_command_digest",
        "not-a-sha256-digest",
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_verifier_policy_json",
        json.dumps(
            {"expected_routstr_code_measurement": (FIXTURE_ROUTSTR_CODE_MEASUREMENT)}
        ),
    )
    monkeypatch.setattr(settings, "routstr_tee_verifier_timeout_seconds", 2.0)

    readiness = get_routstr_tee_readiness()

    failure_reason = str(readiness["failure_reason"] or "")
    assert readiness["ready"] is False
    assert "ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST must be a sha256 digest" in (
        failure_reason
    )
    assert "failed to hash Routstr TEE verifier artifact" not in failure_reason


def test_routstr_tee_attestation_command_rejects_malformed_digest_before_hashing_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)

    monkeypatch.setattr(settings, "routstr_attestation_document_path", "")
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
    monkeypatch.setattr(
        settings,
        "routstr_tee_attestation_command",
        json.dumps([sys.executable, str(tmp_path / "missing-attester")]),
        raising=False,
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_attestation_artifact_path",
        str(tmp_path / "missing-attester"),
        raising=False,
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_attestation_command_digest",
        "not-a-sha256-digest",
        raising=False,
    )
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

    readiness = get_routstr_tee_readiness()

    failure_reason = str(readiness["failure_reason"] or "")
    assert readiness["ready"] is False
    assert "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST must be a sha256 digest" in (
        failure_reason
    )
    assert "failed to hash Routstr TEE attestation artifact" not in failure_reason


def test_routstr_tee_verifier_rejects_unbound_artifact_path_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_path = tmp_path / "quote.bin"
    evidence_path.write_bytes(b"fake-tdx-quote")
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)

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
    monkeypatch.setattr(
        settings,
        "routstr_tee_verifier_command",
        json.dumps([sys.executable, str(tmp_path / "missing-verifier")]),
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

    readiness = get_routstr_tee_readiness()

    failure_reason = str(readiness["failure_reason"] or "")
    assert readiness["ready"] is False
    assert "ROUTSTR_TEE_VERIFIER_ARTIFACT_PATH must match" in failure_reason
    assert "Routstr TEE verifier command failed with exit code" not in failure_reason


def test_routstr_tee_attestation_command_rejects_unbound_artifact_path_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)

    monkeypatch.setattr(settings, "routstr_attestation_document_path", "")
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
    monkeypatch.setattr(
        settings,
        "routstr_tee_attestation_command",
        json.dumps([sys.executable, str(tmp_path / "missing-attester")]),
        raising=False,
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_attestation_artifact_path",
        str(ATTESTATION_STUB_PATH),
        raising=False,
    )
    monkeypatch.setattr(
        settings,
        "routstr_tee_attestation_command_digest",
        _sha256_file_digest(ATTESTATION_STUB_PATH),
        raising=False,
    )
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

    readiness = get_routstr_tee_readiness()

    failure_reason = str(readiness["failure_reason"] or "")
    assert readiness["ready"] is False
    assert "ROUTSTR_TEE_ATTESTATION_ARTIFACT_PATH must match" in failure_reason
    assert "Routstr TEE attestation command failed with exit code" not in failure_reason


def test_routstr_tee_attestation_command_stderr_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)

    monkeypatch.setattr(settings, "routstr_attestation_document_path", "")
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
    monkeypatch.setenv("ROUTSTR_TEE_ATTESTATION_STUB_BEHAVIOR", "failure_secret")

    readiness = get_routstr_tee_readiness()

    assert readiness["ready"] is False
    failure_reason = str(readiness["failure_reason"] or "")
    assert "Routstr TEE attestation command failed with exit code 9" in failure_reason
    assert "stderr redacted" in failure_reason
    assert "SECRET_PROMPT" not in failure_reason
    assert "sk-live-secret" not in failure_reason
    assert "report_data=raw" not in failure_reason


def test_routstr_tee_readiness_requires_attestation_document_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_path = tmp_path / "quote.bin"
    evidence_path.write_bytes(b"fake-tdx-quote")
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)

    monkeypatch.setattr(
        settings, "routstr_attestation_document_path", str(evidence_path)
    )
    monkeypatch.setattr(settings, "routstr_attestation_document_format", "")
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

    readiness = get_routstr_tee_readiness()

    assert readiness["ready"] is False
    assert "Routstr TEE attestation document format is not configured" in str(
        readiness["failure_reason"] or ""
    )


def test_routstr_tee_verifier_policy_rejects_nested_inline_secret_before_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_path = tmp_path / "quote.bin"
    evidence_path.write_bytes(b"fake-tdx-quote")
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)

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
            {
                "expected_routstr_code_measurement": (FIXTURE_ROUTSTR_CODE_MEASUREMENT),
                "nested": {
                    "Authorization": "Bearer SECRET_LOCAL_POLICY",
                },
            }
        ),
    )
    monkeypatch.setattr(settings, "routstr_tee_verifier_timeout_seconds", 2.0)
    monkeypatch.setenv("ROUTSTR_VERIFIER_STUB_BEHAVIOR", "failure_secret")

    readiness = get_routstr_tee_readiness()

    failure_reason = str(readiness["failure_reason"] or "")
    assert readiness["ready"] is False
    assert "authorization must not be embedded in Routstr TEE verifier policy" in (
        failure_reason
    )
    assert "SECRET_LOCAL_POLICY" not in failure_reason
    assert "sk-live-secret" not in failure_reason


def test_routstr_tee_verifier_policy_rejects_refresh_token_before_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_path = tmp_path / "quote.bin"
    evidence_path.write_bytes(b"fake-tdx-quote")
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)

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
            {
                "expected_routstr_code_measurement": (FIXTURE_ROUTSTR_CODE_MEASUREMENT),
                "credentials": {
                    "refreshToken": "refresh-secret",
                },
            }
        ),
    )
    monkeypatch.setattr(settings, "routstr_tee_verifier_timeout_seconds", 2.0)
    monkeypatch.setenv("ROUTSTR_VERIFIER_STUB_BEHAVIOR", "failure_secret")

    readiness = get_routstr_tee_readiness()

    failure_reason = str(readiness["failure_reason"] or "")
    assert readiness["ready"] is False
    assert "refresh_token must not be embedded in Routstr TEE verifier policy" in (
        failure_reason
    )
    assert "refresh-secret" not in failure_reason
    assert "sk-live-secret" not in failure_reason


def test_routstr_tee_verifier_policy_rejects_inline_secret_value_before_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_path = tmp_path / "quote.bin"
    evidence_path.write_bytes(b"fake-tdx-quote")
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)

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
            {
                "expected_routstr_code_measurement": (FIXTURE_ROUTSTR_CODE_MEASUREMENT),
                "metadata": {
                    "credential_hint": "sk-should-not-enter-policy",
                },
            }
        ),
    )
    monkeypatch.setattr(settings, "routstr_tee_verifier_timeout_seconds", 2.0)
    monkeypatch.setenv("ROUTSTR_VERIFIER_STUB_BEHAVIOR", "failure_secret")

    readiness = get_routstr_tee_readiness()

    failure_reason = str(readiness["failure_reason"] or "")
    assert readiness["ready"] is False
    assert (
        "secret-like value must not be embedded in Routstr TEE verifier policy"
        in failure_reason
    )
    assert "sk-should-not-enter-policy" not in failure_reason
    assert "sk-live-secret" not in failure_reason


def test_routstr_tee_verifier_command_rejects_secret_argv_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_path = tmp_path / "quote.bin"
    evidence_path.write_bytes(b"fake-tdx-quote")
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)

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
    monkeypatch.setattr(
        settings,
        "routstr_tee_verifier_command",
        json.dumps(
            [
                sys.executable,
                str(VERIFIER_STUB_PATH),
                "--token=sk-local-verifier-secret",
            ]
        ),
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
    monkeypatch.setenv("ROUTSTR_VERIFIER_STUB_BEHAVIOR", "failure_secret")

    readiness = get_routstr_tee_readiness()

    failure_reason = str(readiness["failure_reason"] or "")
    assert readiness["ready"] is False
    assert "secret-like value must not be embedded" in failure_reason
    assert "sk-local-verifier-secret" not in failure_reason
    assert "Routstr TEE verifier command failed with exit code" not in failure_reason


def test_routstr_tee_verifier_stderr_is_redacted_from_failure_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_path = tmp_path / "quote.bin"
    evidence_path.write_bytes(b"fake-tdx-quote")
    hpke_key_config = _ehbp_key_config(public_key=b"\x77" * 32)

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
    monkeypatch.setenv("ROUTSTR_VERIFIER_STUB_BEHAVIOR", "failure_secret")

    readiness = get_routstr_tee_readiness()

    assert readiness["ready"] is False
    failure_reason = str(readiness["failure_reason"] or "")
    assert "Routstr TEE verifier command failed with exit code 7" in failure_reason
    assert "stderr redacted" in failure_reason
    assert "SECRET_PROMPT" not in failure_reason
    assert "sk-live-secret" not in failure_reason
    assert "raw-quote" not in failure_reason
