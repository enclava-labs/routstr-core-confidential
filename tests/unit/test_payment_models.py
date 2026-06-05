from typing import Any

import pytest

import routstr.core.attestation as attestation
import routstr.payment.models as payment_models_module
import routstr.proxy as proxy
from routstr.core.db import ModelRow
from routstr.payment.models import (
    Model,
    _public_model_confidentiality,
    _row_to_model,
    remote_model_without_public_proof,
)


def _model_row(**updates: object) -> ModelRow:
    data = {
        "id": "gpt-secure",
        "upstream_provider_id": 1,
        "name": "GPT Secure",
        "created": 1,
        "description": "test",
        "context_length": 4096,
        "architecture": (
            '{"modality":"text->text","input_modalities":["text"],'
            '"output_modalities":["text"],"tokenizer":"gpt",'
            '"instruct_type":null}'
        ),
        "pricing": (
            '{"prompt":0.0,"completion":0.0,"request":0.0,'
            '"image":0.0,"web_search":0.0,"internal_reasoning":0.0}'
        ),
        "enabled": True,
    }
    data.update(updates)
    return ModelRow(**data)


def _catalog_model(
    model_id: str,
    *,
    confidentiality: dict[str, Any] | None = None,
) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        created=1,
        description="test",
        context_length=4096,
        architecture={
            "modality": "text->text",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "tokenizer": "gpt",
            "instruct_type": None,
        },
        pricing={
            "prompt": 0.0,
            "completion": 0.0,
            "request": 0.0,
            "image": 0.0,
            "web_search": 0.0,
            "internal_reasoning": 0.0,
        },
        confidentiality=confidentiality,
    )


def test_row_to_model_rejects_non_standard_pricing_json() -> None:
    row = _model_row(
        pricing=(
            '{"prompt":NaN,"completion":0.0,"request":0.0,'
            '"image":0.0,"web_search":0.0,"internal_reasoning":0.0}'
        )
    )

    with pytest.raises(ValueError, match="model pricing JSON must not contain NaN"):
        _row_to_model(row)


def test_row_to_model_rejects_string_non_finite_pricing_values() -> None:
    row = _model_row(
        pricing=(
            '{"prompt":"NaN","completion":0.0,"request":0.0,'
            '"image":0.0,"web_search":0.0,"internal_reasoning":0.0}'
        )
    )

    with pytest.raises(ValueError, match="pricing values must be finite"):
        _row_to_model(row)


def test_row_to_model_rejects_negative_pricing_values() -> None:
    row = _model_row(
        pricing=(
            '{"prompt":-0.1,"completion":0.0,"request":0.0,'
            '"image":0.0,"web_search":0.0,"internal_reasoning":0.0}'
        )
    )

    with pytest.raises(ValueError, match="pricing values must be non-negative"):
        _row_to_model(row)


@pytest.mark.asyncio
async def test_models_endpoint_lists_only_routable_confidential_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plain_model = _catalog_model("gpt-public")
    unrouted_confidential_model = _catalog_model(
        "gpt-secure-cold",
        confidentiality={"mode": "tinfoil"},
    )
    routed_confidential_model = _catalog_model(
        "gpt-secure",
        confidentiality={"mode": "tinfoil"},
    )

    def fake_public_model_confidentiality(
        value: object,
        *,
        model_id: str | None = None,
    ) -> dict[str, Any] | None:
        if value == {"mode": "tinfoil"}:
            return {
                "verified": True,
                "mode": "tinfoil",
                "provider_type": "tinfoil",
                "evidence_digest": "sha256:" + ("a" * 64),
            }
        return None

    monkeypatch.setattr(
        proxy,
        "get_unique_models",
        lambda: [plain_model, unrouted_confidential_model, routed_confidential_model],
    )
    monkeypatch.setattr(
        proxy,
        "is_routable_confidential_model",
        lambda model_id, model: model_id == "gpt-secure",
    )
    monkeypatch.setattr(
        attestation,
        "get_public_routstr_tee_status",
        lambda: {"required": False, "ready": False},
    )
    monkeypatch.setattr(
        payment_models_module,
        "_public_model_confidentiality",
        fake_public_model_confidentiality,
    )

    response = await payment_models_module.models()

    assert [model["id"] for model in response["data"]] == ["gpt-secure"]
    assert response["data"][0]["confidential"] is True
    assert response["data"][0]["attestation_status"] == "verified"


def test_public_model_confidentiality_rejects_non_list_verified_selectors() -> None:
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "verifier": "unit-test-verifier",
            "policy_digest": "sha256:" + ("a" * 64),
            "evidence_digest": "sha256:" + ("b" * 64),
            "verified_claims_digest": "sha256:" + ("c" * 64),
            "verified_at": 1_700_000_000,
            "expires_at": 4_102_444_800,
            "model_ids": ("gpt-secure",),
            "model_id_prefixes": [],
            "proof_claims": {
                "transport": "ehbp",
                "repo": "tinfoilsh/confidential-model-router",
                "attestation_format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
                "attestation_report_digest": "sha256:" + ("d" * 64),
                "attested_hpke_public_key_hex": "a" * 64,
                "code_measurement_fingerprint": "sha256:" + ("e" * 64),
                "enclave_measurement_fingerprint": "sha256:" + ("f" * 64),
                "release_digest": "sha256:" + ("1" * 64),
                "tls_public_key_fingerprint_sha256": "sha256:" + ("2" * 64),
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
                        "attestation_report_digest": "sha256:" + ("3" * 64),
                        "attested_hpke_public_key_hex": "b" * 64,
                        "code_measurement_fingerprint": "sha256:" + ("4" * 64),
                        "enclave_measurement_fingerprint": "sha256:" + ("5" * 64),
                        "release_digest": "sha256:" + ("6" * 64),
                        "tls_public_key_fingerprint_sha256": "sha256:" + ("7" * 64),
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
            },
        }
    )

    assert confidentiality is None


def test_public_model_confidentiality_rejects_unverified_proof_digests() -> None:
    confidentiality = _public_model_confidentiality(
        {
            "enabled": True,
            "verified": False,
            "attestation_status": "unavailable",
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "policy_digest": "sha256:" + ("a" * 64),
            "evidence_digest": "sha256:" + ("b" * 64),
            "model_ids": ["gpt-secure"],
            "model_id_prefixes": [],
        }
    )

    assert confidentiality is None


def test_remote_model_without_public_proof_strips_copied_attestation_fields() -> None:
    model = remote_model_without_public_proof(
        {
            "id": "gpt-secure",
            "name": "GPT Secure",
            "confidential": True,
            "attestation_status": "verified",
            "attestation_provider": "tinfoil",
            "provider_attestation_status": "verified",
            "attestation_evidence_digest": "sha256:" + ("a" * 64),
            "routstr_tee": {"ready": True},
            "confidentiality": {
                "verified": True,
                "proof_claims": {"raw": "remote proof is not local proof"},
            },
        }
    )

    assert model == {
        "id": "gpt-secure",
        "name": "GPT Secure",
    }
