from types import TracebackType
from unittest.mock import Mock

import httpx
import pytest

from routstr.upstream.routstr import RoutstrUpstreamProvider


class DummyAsyncClient:
    def __init__(self, response: Mock | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def __aenter__(self) -> "DummyAsyncClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False

    async def get(self, url: str, headers: dict[str, str], timeout: float) -> Mock:
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


@pytest.mark.asyncio
async def test_get_balance_omits_auth_header_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Mock()
    response.json.return_value = {"balance_msats": 42000}
    response.raise_for_status.return_value = None

    client = DummyAsyncClient(response=response)
    monkeypatch.setattr("routstr.upstream.routstr.httpx.AsyncClient", lambda: client)

    provider = RoutstrUpstreamProvider(base_url="https://node.example", api_key="")

    balance = await provider.get_balance()

    assert balance == 42.0
    assert client.calls == [
        {
            "url": "https://node.example/v1/balance/info",
            "headers": {},
            "timeout": 10.0,
        }
    ]


@pytest.mark.asyncio
async def test_get_balance_returns_none_on_connect_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = DummyAsyncClient(error=httpx.ConnectTimeout("timed out"))
    monkeypatch.setattr("routstr.upstream.routstr.httpx.AsyncClient", lambda: client)

    provider = RoutstrUpstreamProvider(
        base_url="https://node.example",
        api_key="secret",
    )

    balance = await provider.get_balance()

    assert balance is None


def test_normalize_request_path_keeps_v1_prefix() -> None:
    """Routstr upstream stores ``base_url`` without ``/v1``; the prefix
    must stay on the path so ``build_request_url`` produces ``/v1/<endpoint>``
    instead of ``/<endpoint>`` (which the upstream Routstr 404s with HTML)."""
    provider = RoutstrUpstreamProvider(
        base_url="https://privateprovider.xyz", api_key="key"
    )

    assert provider.normalize_request_path("v1/messages") == "v1/messages"
    assert provider.normalize_request_path("/v1/messages") == "v1/messages"
    assert (
        provider.normalize_request_path("v1/chat/completions") == "v1/chat/completions"
    )


def test_build_request_url_for_v1_messages() -> None:
    """Forwarding ``/v1/messages`` must hit the upstream's ``/v1/messages``."""
    provider = RoutstrUpstreamProvider(
        base_url="https://privateprovider.xyz", api_key="key"
    )

    normalized = provider.normalize_request_path("v1/messages")

    assert (
        provider.build_request_url(normalized)
        == "https://privateprovider.xyz/v1/messages"
    )


def test_supports_anthropic_messages_natively() -> None:
    """Routstr nodes serve ``/v1/messages`` directly, so the proxy must
    forward as-is instead of round-tripping through litellm."""
    assert RoutstrUpstreamProvider.supports_anthropic_messages is True


@pytest.mark.asyncio
async def test_fetch_models_strips_remote_confidentiality_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": [
            {
                "id": "gpt-secure",
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
                "confidentiality": {
                    "enabled": True,
                    "verified": True,
                    "mode": "tinfoil",
                    "provider_type": "tinfoil",
                    "verifier": "remote-verifier",
                    "policy_digest": "sha256:" + ("b" * 64),
                    "evidence_digest": "sha256:" + ("c" * 64),
                    "verified_at": 1_700_000_000,
                    "expires_at": 4_102_444_800,
                    "model_ids": ["gpt-secure"],
                    "model_id_prefixes": [],
                    "proof_claims": {"raw": "remote proof is not local proof"},
                },
            }
        ],
        "routstr_tee": {
            "required": True,
            "ready": True,
        },
    }

    client = DummyAsyncClient(response=response)
    monkeypatch.setattr("routstr.upstream.routstr.httpx.AsyncClient", lambda: client)

    provider = RoutstrUpstreamProvider(base_url="https://node.example", api_key="")

    models = await provider.fetch_models()

    assert len(models) == 1
    assert models[0].id == "gpt-secure"
    assert models[0].confidentiality is None
