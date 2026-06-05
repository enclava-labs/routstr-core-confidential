import pytest

from routstr.upstream.openrouter import OpenRouterUpstreamProvider


def _remote_openrouter_model_with_copied_proof() -> dict[str, object]:
    return {
        "id": "openai/gpt-secure",
        "name": "GPT Secure",
        "created": 1,
        "description": "remote model",
        "context_length": 4096,
        "architecture": {
            "modality": "text->text",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "tokenizer": "gpt",
            "instruct_type": None,
        },
        "pricing": {
            "prompt": 0.0,
            "completion": 0.0,
            "request": 0.0,
            "image": 0.0,
            "web_search": 0.0,
            "internal_reasoning": 0.0,
        },
        "confidential": True,
        "attestation_status": "verified",
        "attestation_provider": "tinfoil",
        "provider_attestation_status": "verified",
        "attestation_evidence_digest": "sha256:" + ("a" * 64),
        "routstr_tee": {"ready": True},
        "confidentiality": {
            "enabled": True,
            "verified": True,
            "mode": "tinfoil",
            "provider_type": "tinfoil",
            "proof_claims": {"raw": "remote proof is not local proof"},
        },
    }


@pytest.mark.asyncio
async def test_openrouter_fetch_models_strips_remote_confidentiality_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch_openrouter_models(
        source_filter: str | None = None,
    ) -> list[dict[str, object]]:
        assert source_filter is None
        return [_remote_openrouter_model_with_copied_proof()]

    monkeypatch.setattr(
        "routstr.upstream.openrouter.async_fetch_openrouter_models",
        fake_fetch_openrouter_models,
    )

    provider = OpenRouterUpstreamProvider(api_key="sk-test")

    models = await provider.fetch_models()

    assert len(models) == 1
    assert models[0].id == "openai/gpt-secure"
    assert models[0].confidentiality is None
