from __future__ import annotations

import base64
import gzip
import hashlib
from typing import Any

import pytest

from routstr.upstream.base import BaseUpstreamProvider, ConfidentialityStatus
from routstr.upstream.ppqai import PPQAIUpstreamProvider, PPQPrivateUpstreamProvider


def _test_digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


VALID_VERIFIER_DIGEST = _test_digest("ppq-private-verifier")
PLACEHOLDER_DIGEST = "sha256:" + ("b" * 64)
EHBP_CODE_MEASUREMENT = _test_digest("ehbp-code-measurement")
EHBP_ENCLAVE_MEASUREMENT = _test_digest("ehbp-enclave-measurement")
EHBP_OTHER_CODE_MEASUREMENT = _test_digest("ehbp-other-code-measurement")
EHBP_OTHER_ENCLAVE_MEASUREMENT = _test_digest("ehbp-other-enclave-measurement")
TINFOIL_EHBP_ATTESTATION_FORMAT = "https://tinfoil.sh/predicate/sev-snp-guest/v2"
PPQ_PRIVATE_SELECTED_MODEL_ID = "private/gpt-oss-120b"
PPQ_PRIVATE_BACKEND_HOST = "gpt-oss-120b-1.inf10.tinfoil.sh"
PPQ_PRIVATE_BACKEND_REPO = "tinfoilsh/confidential-gpt-oss-120b"
PPQ_PRIVATE_BACKEND_RELEASE_DIGEST = hashlib.sha256(
    b"ppq-private-backend-release"
).hexdigest()
PPQ_PRIVATE_PROXY_BINARY_PATH = "/opt/ppq/ppq-private-mode-proxy"


def _ehbp_verification_steps() -> dict[str, bool]:
    return {
        "hardware_attestation_report": True,
        "hardware_certificate_chain": True,
        "code_transparency": True,
        "measurement_match": True,
        "attested_transport_key_binding": True,
        "freshness": True,
    }


def _ppq_backend_model_attestation_policy() -> dict[str, Any]:
    return {
        "require_model_attestations": True,
        "model_attestation_targets": {
            PPQ_PRIVATE_SELECTED_MODEL_ID: {
                "host": PPQ_PRIVATE_BACKEND_HOST,
                "repo": PPQ_PRIVATE_BACKEND_REPO,
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "expected_release_digest": PPQ_PRIVATE_BACKEND_RELEASE_DIGEST,
            }
        },
    }


def _ppq_backend_model_attestation_claims(
    payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    model_attestations = payload.get("model_attestations")
    if not isinstance(model_attestations, list):
        return {}
    claims: dict[str, dict[str, Any]] = {}
    for raw_attestation in model_attestations:
        if not isinstance(raw_attestation, dict):
            continue
        model_id = raw_attestation.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            continue
        attestation = raw_attestation.get("attestation")
        if not isinstance(attestation, dict):
            attestation = {}
        claims[model_id] = {
            "repo": PPQ_PRIVATE_BACKEND_REPO,
            "attestation_format": attestation.get("format"),
            "attestation_report_digest": attestation.get("report_digest"),
            "attested_hpke_public_key_hex": "33" * 32,
                "client_encryption_boundary": "routstr-tee-ehbp-proxy",
            "tls_public_key_fingerprint_sha256": _test_digest(
                "ppq-private-backend-tls-key"
            ),
            "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
            "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
            "release_digest": PPQ_PRIVATE_BACKEND_RELEASE_DIGEST,
            "verification_steps": _ehbp_verification_steps(),
        }
    return claims


class DummyResponse:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._data


class DummyAsyncClient:
    calls: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def __aenter__(self) -> "DummyAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> DummyResponse:
        self.calls.append(
            {"url": url, "headers": headers or {}, "params": params or {}}
        )
        return DummyResponse(
            {
                "data": [
                    {
                        "id": "openai/gpt-5-mini",
                        "provider": "OpenAI",
                        "name": "GPT 5 Mini",
                        "created_at": 1_700_000_000_000,
                        "context_length": 128000,
                        "pricing": {
                            "api": {
                                "input_per_1M": 1.0,
                                "output_per_1M": 2.0,
                            }
                        },
                    },
                    {
                        "id": "private/gpt-oss-120b",
                        "provider": "PPQ Private",
                        "name": "GPT OSS 120B Private",
                        "created_at": 1_700_000_000_000,
                        "context_length": 131072,
                        "pricing": {
                            "api": {
                                "input_per_1M": 0.15,
                                "output_per_1M": 0.6,
                            }
                        },
                    },
                ]
            }
        )


class NonCanonicalPPQPrivateCatalogClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def __aenter__(self) -> "NonCanonicalPPQPrivateCatalogClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> DummyResponse:
        return DummyResponse(
            {
                "data": [
                    {
                        "id": "private/nan-priced-model",
                        "provider": "PPQ Private",
                        "name": "NaN Priced Private",
                        "created_at": 1_700_000_000_000,
                        "context_length": 131072,
                        "pricing": {
                            "api": {
                                "input_per_1M": float("nan"),
                                "output_per_1M": 0.6,
                            }
                        },
                    }
                ]
            }
        )


class MalformedPublicPPQCatalogClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def __aenter__(self) -> "MalformedPublicPPQCatalogClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> DummyResponse:
        return DummyResponse(
            {
                "data": [
                    {
                        "id": "public-provider/malformed",
                    },
                    {
                        "id": "private/gpt-oss-120b",
                        "provider": "PPQ Private",
                        "name": "GPT OSS 120B Private",
                        "created_at": 1_700_000_000_000,
                        "context_length": 131072,
                        "pricing": {
                            "api": {
                                "input_per_1M": 0.15,
                                "output_per_1M": 0.6,
                            }
                        },
                    },
                ]
            }
        )


class DummyHealthResponse:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._data


class DummyHealthClient:
    calls: list[str] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def __aenter__(self) -> "DummyHealthClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def get(self, url: str) -> DummyHealthResponse:
        self.calls.append(url)
        return DummyHealthResponse({"status": "ok", "attestation": True})


class DummyPPQVerifierResponse:
    def __init__(self, content: bytes = b"", data: dict[str, Any] | None = None) -> None:
        self.content = content
        self._data = data or {}

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._data


class DummyPPQVerifierClient:
    calls: list[str] = []
    hpke_keys: bytes = (
        b"\x00\x00\x20" + (b"\x33" * 32) + b"\x00\x04" + b"\x00\x01" + b"\x00\x02"
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def __aenter__(self) -> "DummyPPQVerifierClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def get(self, url: str) -> DummyPPQVerifierResponse:
        self.calls.append(url)
        if url.endswith("/.well-known/tinfoil-attestation"):
            body = base64.b64encode(gzip.compress(b"model-attestation-report")).decode(
                "ascii"
            )
            return DummyPPQVerifierResponse(
                data={
                    "format": TINFOIL_EHBP_ATTESTATION_FORMAT,
                    "body": body,
                }
            )
        if url.endswith("/.well-known/hpke-keys"):
            return DummyPPQVerifierResponse(content=self.hpke_keys)
        raise AssertionError(f"unexpected URL: {url}")


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_prefix_selectors_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_id_prefixes": ["private/"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert DummyPPQVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert "model_id_prefixes are not supported" in (
        status.failure_reason or ""
    )
    assert "requires exact model_ids" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_non_private_model_ids_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["openai/gpt-5-mini"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert DummyPPQVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert "PPQ private model_ids must start with private/" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_malformed_selectors_without_default_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.time.time",
        lambda: 1_700_000_010,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        return {
            "verified": True,
            "verifier": "unit-test-ppq-private-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "verified_at": 1_700_000_000,
            "claims": {
                "attestation_bundle_url": "https://api.ppq.ai/private",
                "attested_hpke_public_key_hex": "33" * 32,
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                "tls_public_key_fingerprint_sha256": _test_digest(
                    "ppq-private-tls-key"
                ),
                "verification_steps": _ehbp_verification_steps(),
                "backend_model_attestations": _ppq_backend_model_attestation_claims(
                    payload
                ),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_id_prefixes": [{"unexpected": "private/"}],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.model_id_prefixes == []
    assert "model_id_prefixes" in (status.failure_reason or "")
    assert verifier_payloads == []
    assert DummyPPQVerifierClient.calls == []


@pytest.mark.asyncio
async def test_ppq_private_refresh_returns_sanitized_runtime_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_refresh_ppq_private_status(
        provider: PPQPrivateUpstreamProvider,
    ) -> ConfidentialityStatus:
        return ConfidentialityStatus(
            enabled=True,
            verified=True,
            mode="ppq-private-tee",
            verified_at=1_700_000_000,
            expires_at=4_102_444_800,
            verifier="test-verifier",
            policy_digest=_test_digest("policy"),
            evidence_digest=_test_digest("evidence"),
            verified_claims={"transport": "ehbp", "non_canonical": float("nan")},
            model_id_prefixes=["private/"],
        )

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.refresh_ppq_private_status",
        fake_refresh_ppq_private_status,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )

    status = await provider.refresh_confidentiality_status()
    stored_status = provider.confidentiality_status()

    assert status is stored_status
    assert status.verified is False
    assert status.verifier is None
    assert status.evidence_digest is None
    assert status.verified_claims == {}
    assert status.model_id_prefixes == ["private/"]
    assert (
        status.failure_reason
        == "confidentiality verifier did not return required evidence"
    )


@pytest.mark.asyncio
async def test_ppqai_provider_excludes_private_tee_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyAsyncClient.calls = []
    monkeypatch.setattr("routstr.upstream.ppqai.httpx.AsyncClient", DummyAsyncClient)

    async def no_openrouter_models() -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(
        "routstr.upstream.ppqai.async_fetch_openrouter_models",
        no_openrouter_models,
    )

    provider = PPQAIUpstreamProvider(api_key="test")

    models = await provider.fetch_models()

    assert [model.id for model in models] == ["openai/gpt-5-mini"]
    assert all(not model.id.startswith("private/") for model in models)


@pytest.mark.asyncio
async def test_ppq_private_provider_fetches_only_private_models_from_all_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyAsyncClient.calls = []
    monkeypatch.setattr("routstr.upstream.ppqai.httpx.AsyncClient", DummyAsyncClient)

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )

    models = await provider.fetch_models()

    assert [model.id for model in models] == ["private/gpt-oss-120b"]
    assert models[0].supported_endpoints == ["/v1/chat/completions"]
    assert models[0].pricing.prompt == 0.15 / 1_000_000
    assert DummyAsyncClient.calls == [
        {
            "url": "https://api.ppq.ai/v1/models",
            "headers": {"Authorization": "Bearer test"},
            "params": {"type": "all"},
        }
    ]

    status = provider.confidentiality_status()
    assert status.enabled is True
    assert status.verified is False
    assert status.mode == "ppq-private-tee"
    assert status.failure_reason == "runtime PPQ private verifier has not run"
    assert status.model_id_prefixes == ["private/"]
    assert provider.requires_verified_ehbp_transport is True
    assert provider.transform_model_name("private/gpt-oss-120b") == "gpt-oss-120b"


@pytest.mark.asyncio
async def test_ppq_private_fetch_models_ignores_malformed_public_rows_without_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.upstream.ppqai.httpx.AsyncClient",
        MalformedPublicPPQCatalogClient,
    )
    warnings: list[str] = []

    def record_warning(message: str, *args: Any, **kwargs: Any) -> None:
        warnings.append(message)

    monkeypatch.setattr("routstr.upstream.ppqai.logger.warning", record_warning)

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )

    models = await provider.fetch_models()

    assert [model.id for model in models] == ["private/gpt-oss-120b"]
    assert warnings == []


@pytest.mark.asyncio
async def test_ppq_private_fetch_models_rejects_non_canonical_catalog_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "routstr.upstream.ppqai.httpx.AsyncClient",
        NonCanonicalPPQPrivateCatalogClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )

    assert await provider.fetch_models() == []


def test_ppq_private_provider_does_not_inherit_ppq_account_management() -> None:
    normal_provider = PPQAIUpstreamProvider(api_key="test")
    private_provider = PPQPrivateUpstreamProvider(api_key="test")

    assert (
        type(normal_provider).create_account is not BaseUpstreamProvider.create_account
    )
    assert (
        type(normal_provider).initiate_topup is not BaseUpstreamProvider.initiate_topup
    )
    assert type(private_provider).create_account is BaseUpstreamProvider.create_account
    assert type(private_provider).initiate_topup is BaseUpstreamProvider.initiate_topup


def test_ppq_private_from_db_row_rejects_non_standard_provider_settings_json() -> None:
    provider_row = type(
        "ProviderRow",
        (),
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "api_key": "test",
            "provider_fee": 1.0,
            "provider_settings": (
                '{"catalog_base_url":"https://attacker.example/v1",'
                '"ignored_non_standard":NaN}'
            ),
        },
    )()

    provider = PPQPrivateUpstreamProvider.from_db_row(provider_row)

    assert provider.catalog_base_url == "https://api.ppq.ai/v1"


def test_ppq_private_from_db_row_ignores_non_string_catalog_base_url() -> None:
    provider_row = type(
        "ProviderRow",
        (),
        {
            "base_url": "https://api.ppq.ai/private/v1",
            "api_key": "test",
            "provider_fee": 1.0,
            "provider_settings": (
                '{"catalog_base_url":{"url":"http://169.254.169.254/v1"}}'
            ),
        },
    )()

    provider = PPQPrivateUpstreamProvider.from_db_row(provider_row)

    assert provider.catalog_base_url == "https://api.ppq.ai/v1"


@pytest.mark.asyncio
async def test_ppq_private_fetch_models_rejects_catalog_origin_mismatch_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyAsyncClient.calls = []
    monkeypatch.setattr("routstr.upstream.ppqai.httpx.AsyncClient", DummyAsyncClient)

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
        catalog_base_url="https://catalog.attacker.example/v1",
    )

    assert await provider.fetch_models() == []
    assert DummyAsyncClient.calls == []


@pytest.mark.asyncio
async def test_ppq_private_fetch_models_rejects_unsafe_base_url_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyAsyncClient.calls = []
    monkeypatch.setattr("routstr.upstream.ppqai.httpx.AsyncClient", DummyAsyncClient)

    provider = PPQPrivateUpstreamProvider(
        base_url="https://operator:secret-token@api.ppq.ai/private/v1",
        api_key="test",
    )

    assert await provider.fetch_models() == []
    assert DummyAsyncClient.calls == []


@pytest.mark.asyncio
async def test_ppq_private_fetch_models_rejects_unowned_private_base_url_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyAsyncClient.calls = []
    monkeypatch.setattr("routstr.upstream.ppqai.httpx.AsyncClient", DummyAsyncClient)

    provider = PPQPrivateUpstreamProvider(
        base_url="https://attacker.example/private/v1",
        api_key="test",
    )

    assert await provider.fetch_models() == []
    assert DummyAsyncClient.calls == []


@pytest.mark.asyncio
async def test_ppq_private_refresh_requires_verifier_command_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.enabled is True
    assert status.verified is False
    assert status.policy_digest is not None
    assert status.evidence_digest is None
    assert status.model_ids == ["private/gpt-oss-120b"]
    assert (
        status.failure_reason
        == "verifier_command or ppq_private_verifier_command is required"
    )
    assert DummyPPQVerifierClient.calls == []


@pytest.mark.asyncio
async def test_ppq_private_refresh_requires_backend_model_attestation_targets_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": "/opt/routstr/bin/confidential-verifier",
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert DummyPPQVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert (
        "PPQ private require_model_attestations must be true for selected model routing"
        in (status.failure_reason or "")
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_policy_mode_mismatch_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "tinfoil",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": "/opt/routstr/bin/confidential-verifier",
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert DummyPPQVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert (
        status.failure_reason
        == "confidentiality mode must be ppq-private-tee for ppq-private provider"
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_inline_api_key_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "api_key": "sk-should-not-enter-policy",
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert DummyPPQVerifierClient.calls == []
    assert "api_key must not be embedded in verifier policy" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_base_url_credentials_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://operator:secret-token@api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert DummyPPQVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert "provider base_url must not include credentials" in (
        status.failure_reason or ""
    )
    assert "secret-token" not in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_non_private_base_url_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert DummyPPQVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert "PPQ private base_url must include a private path segment" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_non_private_attestation_bundle_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/attestation",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert DummyPPQVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert "attestation_bundle_url must include a private path segment" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_non_private_hpke_keys_url_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "hpke_keys_url": "https://api.ppq.ai/.well-known/hpke-keys",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert DummyPPQVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert "hpke_keys_url must include a private path segment" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_non_https_base_url_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="http://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert DummyPPQVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert "provider base_url must use https" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_invalid_base_url_port_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai:bad/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert DummyPPQVerifierClient.calls == []
    assert status.verified is False
    assert "provider base_url must include a valid port" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_unowned_private_base_url_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://attacker.example/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert DummyPPQVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert "PPQ private base_url host must be ppq.ai or a ppq.ai subdomain" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_non_https_policy_urls_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "http://api.ppq.ai/private",
                    "hpke_keys_url": "http://api.ppq.ai/private/.well-known/hpke-keys",
                    "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "allowed_release_digests": [_test_digest("release")],
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    failure_reason = status.failure_reason or ""
    assert DummyPPQVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert "attestation_bundle_url must use https" in failure_reason
    assert "hpke_keys_url must use https" in failure_reason


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_policy_host_mismatch_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://attest.ppq.ai/private",
                    "hpke_keys_url": "https://attest.ppq.ai/private/.well-known/hpke-keys",
                    "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "allowed_release_digests": [_test_digest("release")],
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    failure_reason = status.failure_reason or ""
    assert DummyPPQVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert "attestation_bundle_url host must match provider base_url host" in (
        failure_reason
    )
    assert "hpke_keys_url host must match provider base_url host" in failure_reason


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_policy_origin_mismatch_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai:8443/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "hpke_keys_url": (
                        "https://api.ppq.ai/private/.well-known/hpke-keys"
                    ),
                    "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "allowed_release_digests": [_test_digest("release")],
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    failure_reason = status.failure_reason or ""
    assert DummyPPQVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert (
        "attestation_bundle_url origin must match provider base_url origin"
        in failure_reason
    )
    assert "hpke_keys_url origin must match provider base_url origin" in failure_reason


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_enclave_host_mismatch_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "expected_enclave_host": "attest.ppq.ai",
                    "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "allowed_release_digests": [_test_digest("release")],
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    failure_reason = status.failure_reason or ""
    assert DummyPPQVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert "expected_enclave_host must match provider base_url host" in failure_reason


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_non_string_policy_urls_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": {"url": "https://api.ppq.ai/private"},
                    "hpke_keys_url": 123,
                    "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "allowed_release_digests": [_test_digest("release")],
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    failure_reason = status.failure_reason or ""
    assert DummyPPQVerifierClient.calls == []
    assert status.verified is False
    assert status.evidence_digest is None
    assert "attestation_bundle_url must be a string" in failure_reason
    assert "hpke_keys_url must be a string" in failure_reason


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_malformed_release_digest_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "allowed_release_digests": [{"unexpected": "object"}],
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyPPQVerifierClient.calls == []
    assert "release digest policy values must be sha256 digests" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_malformed_measurement_pin_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": "code-fp",
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyPPQVerifierClient.calls == []
    assert "measurement fingerprint policy values must be sha256 digests" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_malformed_verifier_digest_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": "not-a-sha256-digest",
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyPPQVerifierClient.calls == []
    assert "verifier_command_digest must be a sha256 digest" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_placeholder_verifier_digest_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": PLACEHOLDER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyPPQVerifierClient.calls == []
    assert "verifier_command_digest must not be a placeholder digest" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_conflicting_verifier_digest_aliases_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    "verifier_binary_digest": _test_digest("other-ppq-verifier"),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyPPQVerifierClient.calls == []
    assert "verifier_command_digest aliases must match" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_unbound_verifier_artifact_path_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/bin/python3", "/tmp/evil-verifier.py"],
                    "verifier_artifact_path": "/opt/routstr/bin/ppq-verifier.py",
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyPPQVerifierClient.calls == []
    assert (
        "verifier_artifact_path must match verifier command executable or argument"
        in (status.failure_reason or "")
    )


@pytest.mark.asyncio
async def test_ppq_private_evidence_probe_failure_redacts_secret_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingPPQVerifierClient(DummyPPQVerifierClient):
        calls: list[str] = []

        async def get(self, url: str) -> DummyPPQVerifierResponse:
            self.calls.append(url)
            raise RuntimeError(
                'probe failed {"raw_prompt":"SECRET_PROMPT"} '
                "api_key=SECRET_PROVIDER_KEY sk-live-secretvalue "
                "url=https://operator:secret-token@api.ppq.ai/private"
            )

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        FailingPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    failure_reason = status.failure_reason or ""
    assert status.verified is False
    assert "PPQ private evidence probe failed" in failure_reason
    assert "api_key: [REDACTED]" in failure_reason
    assert "raw_prompt: [REDACTED]" in failure_reason
    assert "https://api.ppq.ai/private" in failure_reason
    assert "SECRET_PROVIDER_KEY" not in failure_reason
    assert "SECRET_PROMPT" not in failure_reason
    assert "sk-live-secretvalue" not in failure_reason
    assert "secret-token" not in failure_reason


@pytest.mark.asyncio
async def test_ppq_private_refresh_marks_verified_when_command_verifies_and_key_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.time.time",
        lambda: 1_700_000_010,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )
    verifier_payloads: list[dict[str, Any]] = []

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        verifier_payloads.append(payload)
        assert command == ["/usr/local/bin/routstr-ppq-private-verifier"]
        return {
            "verified": True,
            "verifier": "unit-test-ppq-private-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "verified_at": 1_700_000_000,
            "claims": {
                "attestation_bundle_url": "https://api.ppq.ai/private",
                "attested_hpke_public_key_hex": "33" * 32,
                "client_encryption_boundary": "routstr-tee-ehbp-proxy",
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "selected_model_ids": ["private/gpt-oss-120b"],
                "tls_public_key_fingerprint_sha256": _test_digest(
                    "ppq-private-tls-key"
                ),
                "verification_steps": _ehbp_verification_steps(),
                "backend_model_attestations": _ppq_backend_model_attestation_claims(
                    payload
                ),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is True
    assert status.failure_reason is None
    assert status.verifier == "unit-test-ppq-private-verifier"
    assert status.verified_at == 1_700_000_000
    assert status.expires_at == 1_700_000_300
    assert status.verified_claims["transport"] == "ehbp"
    assert status.verified_claims["attested_hpke_public_key_hex"] == "33" * 32
    assert status.verified_claims["ehbp_key_config_b64"] == base64.b64encode(
        DummyPPQVerifierClient.hpke_keys
    ).decode("ascii")
    assert verifier_payloads[0]["mode"] == "ppq-private-tee"
    assert verifier_payloads[0]["attestation_bundle_url"] == (
        "https://api.ppq.ai/private"
    )
    assert verifier_payloads[0]["model_ids"] == ["private/gpt-oss-120b"]
    assert status.verified_claims["selected_model_ids"] == ["private/gpt-oss-120b"]
    assert (
        status.verified_claims["client_encryption_boundary"]
        == "routstr-tee-ehbp-proxy"
    )
    assert (
        status.verified_claims["payload_policy_digest"]
        == verifier_payloads[0]["policy_digest"]
    )
    assert (
        status.verified_claims["payload_evidence_digest"]
        == verifier_payloads[0]["evidence_digest"]
    )
    assert (
        status.verified_claims["payload_verification_nonce"]
        == verifier_payloads[0]["verification_nonce"]
    )
    assert verifier_payloads[0]["hpke_key_config_b64"] == base64.b64encode(
        DummyPPQVerifierClient.hpke_keys
    ).decode("ascii")
    assert verifier_payloads[0]["verifier_command_digest"] == VALID_VERIFIER_DIGEST
    assert isinstance(verifier_payloads[0]["verification_nonce"], str)
    assert verifier_payloads[0]["verification_nonce"]


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_verifier_without_client_encryption_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-ppq-private-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "attestation_bundle_url": "https://api.ppq.ai/private",
                "attested_hpke_public_key_hex": "33" * 32,
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                "tls_public_key_fingerprint_sha256": _test_digest(
                    "ppq-private-tls-key"
                ),
                "selected_model_ids": ["private/gpt-oss-120b"],
                "verification_steps": _ehbp_verification_steps(),
                "backend_model_attestations": _ppq_backend_model_attestation_claims(
                    payload
                ),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "verifier client_encryption_boundary claim does not match policy" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_verifier_without_selected_model_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.time.time",
        lambda: 1_700_000_010,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-ppq-private-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "verified_at": 1_700_000_000,
            "claims": {
                "attestation_bundle_url": "https://api.ppq.ai/private",
                "attested_hpke_public_key_hex": "33" * 32,
                "client_encryption_boundary": "routstr-tee-ehbp-proxy",
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                "tls_public_key_fingerprint_sha256": _test_digest(
                    "ppq-private-tls-key"
                ),
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "selected_model_ids claim is required" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_duplicate_selected_model_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.time.time",
        lambda: 1_700_000_010,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-ppq-private-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "verified_at": 1_700_000_000,
            "claims": {
                "attestation_bundle_url": "https://api.ppq.ai/private",
                "attested_hpke_public_key_hex": "33" * 32,
                "client_encryption_boundary": "routstr-tee-ehbp-proxy",
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                "selected_model_ids": [
                    "private/gpt-oss-120b",
                    "private/gpt-oss-120b",
                ],
                "tls_public_key_fingerprint_sha256": _test_digest(
                    "ppq-private-tls-key"
                ),
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "selected_model_ids claim must not contain duplicates" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_case_variant_duplicate_selected_model_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.time.time",
        lambda: 1_700_000_010,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-ppq-private-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "verified_at": 1_700_000_000,
            "claims": {
                "attestation_bundle_url": "https://api.ppq.ai/private",
                "attested_hpke_public_key_hex": "33" * 32,
                "client_encryption_boundary": "routstr-tee-ehbp-proxy",
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                "selected_model_ids": [
                    "PRIVATE/gpt-oss-120b",
                    "private/gpt-oss-120b",
                ],
                "tls_public_key_fingerprint_sha256": _test_digest(
                    "ppq-private-tls-key"
                ),
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "selected_model_ids claim must not contain duplicates" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_verifier_claims_without_tls_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.time.time",
        lambda: 1_700_000_010,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-ppq-private-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "verified_at": 1_700_000_000,
            "claims": {
                "attestation_bundle_url": "https://api.ppq.ai/private",
                "attested_hpke_public_key_hex": "33" * 32,
                "client_encryption_boundary": "routstr-tee-ehbp-proxy",
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert "TLS public key binding claim is required" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_ppq_private_refresh_accepts_prefixed_measurement_pin_with_raw_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )
    raw_code_measurement = EHBP_CODE_MEASUREMENT.removeprefix("sha256:")
    raw_enclave_measurement = EHBP_ENCLAVE_MEASUREMENT.removeprefix("sha256:")

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-ppq-private-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "attestation_bundle_url": "https://api.ppq.ai/private",
                "attested_hpke_public_key_hex": "33" * 32,
                "client_encryption_boundary": "routstr-tee-ehbp-proxy",
                "enclave_measurement_fingerprint": raw_enclave_measurement,
                "code_measurement_fingerprint": raw_code_measurement,
                "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                "tls_public_key_fingerprint_sha256": _test_digest(
                    "ppq-private-tls-key"
                ),
                "selected_model_ids": ["private/gpt-oss-120b"],
                "verification_steps": _ehbp_verification_steps(),
                "backend_model_attestations": _ppq_backend_model_attestation_claims(
                    payload
                ),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is True
    assert status.failure_reason is None
    assert (
        status.verified_claims["code_measurement_fingerprint"] == raw_code_measurement
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_evidence_digest_tracks_verifier_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )
    call_count = 0

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {
            "verified": True,
            "verifier": "unit-test-ppq-private-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "attestation_bundle_url": "https://api.ppq.ai/private",
                "attested_hpke_public_key_hex": "33" * 32,
                "client_encryption_boundary": "routstr-tee-ehbp-proxy",
                "enclave_measurement_fingerprint": _test_digest(
                    f"ehbp-enclave-measurement-{call_count}"
                ),
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "tls_public_key_fingerprint_sha256": _test_digest(
                    "ppq-private-tls-key"
                ),
                "selected_model_ids": ["private/gpt-oss-120b"],
                "verification_steps": _ehbp_verification_steps(),
                "backend_model_attestations": _ppq_backend_model_attestation_claims(
                    payload
                ),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    first_status = await provider.refresh_confidentiality_status()
    second_status = await provider.refresh_confidentiality_status()

    assert first_status.verified is True
    assert second_status.verified is True
    assert first_status.policy_digest == second_status.policy_digest
    assert first_status.evidence_digest != second_status.evidence_digest
    assert (
        first_status.evidence_digest
        == first_status.verified_claims["runtime_evidence_digest"]
    )
    assert (
        second_status.evidence_digest
        == second_status.verified_claims["runtime_evidence_digest"]
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_verifier_without_full_step_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-ppq-private-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "attestation_bundle_url": "https://api.ppq.ai/private",
                "attested_hpke_public_key_hex": "33" * 32,
                "client_encryption_boundary": "routstr-tee-ehbp-proxy",
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "verification_steps must be a JSON object" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_placeholder_measurement_fingerprint_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-ppq-private-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "attestation_bundle_url": "https://api.ppq.ai/private",
                "attested_hpke_public_key_hex": "33" * 32,
                "client_encryption_boundary": "routstr-tee-ehbp-proxy",
                "enclave_measurement_fingerprint": "enclave-fp",
                "code_measurement_fingerprint": "code-fp",
                "release_digest": hashlib.sha256(
                    b"placeholder-measurement-release"
                ).hexdigest(),
                "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_release_digest": hashlib.sha256(
                        b"placeholder-measurement-release"
                    ).hexdigest(),
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "enclave_measurement_fingerprint claim must be a sha256 digest" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_verifier_output_bound_to_other_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-ppq-private-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": "stale-nonce",
            "claims": {
                "attestation_bundle_url": "https://api.ppq.ai/private",
                "attested_hpke_public_key_hex": "33" * 32,
                "client_encryption_boundary": "routstr-tee-ehbp-proxy",
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "repo": "ppq-ai/private-tee",
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "verification_nonce does not match verifier payload" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_verifier_key_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-ppq-private-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "attestation_bundle_url": "https://api.ppq.ai/private",
                "attested_hpke_public_key_hex": "44" * 32,
                "client_encryption_boundary": "routstr-tee-ehbp-proxy",
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "attested HPKE public key does not match" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_verifier_repo_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-ppq-private-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "attestation_bundle_url": "https://api.ppq.ai/private",
                "attested_hpke_public_key_hex": "33" * 32,
                "client_encryption_boundary": "routstr-tee-ehbp-proxy",
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "repo": "unexpected/repo",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "verifier repo claim does not match policy" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_release_digest_outside_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-ppq-private-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "attestation_bundle_url": "https://api.ppq.ai/private",
                "attested_hpke_public_key_hex": "33" * 32,
                "client_encryption_boundary": "routstr-tee-ehbp-proxy",
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "release_digest": hashlib.sha256(b"actual-ppq-release").hexdigest(),
                "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_release_digest": hashlib.sha256(
                        b"expected-ppq-release"
                    ).hexdigest(),
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "release_digest claim does not match policy" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_ppq_private_refresh_rejects_code_measurement_outside_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    async def fake_verifier_command(
        command: object,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "verified": True,
            "verifier": "unit-test-ppq-private-verifier",
            "policy_digest": payload["policy_digest"],
            "evidence_digest": payload["evidence_digest"],
            "verification_nonce": payload["verification_nonce"],
            "claims": {
                "attestation_bundle_url": "https://api.ppq.ai/private",
                "attested_hpke_public_key_hex": "33" * 32,
                "client_encryption_boundary": "routstr-tee-ehbp-proxy",
                "enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                "code_measurement_fingerprint": EHBP_OTHER_CODE_MEASUREMENT,
                "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                "verification_steps": _ehbp_verification_steps(),
            },
        }

    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._run_confidential_verifier_command",
        fake_verifier_command,
        raising=False,
    )
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers._sha256_file_digest",
        lambda path: VALID_VERIFIER_DIGEST,
        raising=False,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "expected_enclave_measurement_fingerprint": EHBP_ENCLAVE_MEASUREMENT,
                    "allowed_code_measurement_fingerprints": [EHBP_CODE_MEASUREMENT],
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                    "proxy_binary_path": PPQ_PRIVATE_PROXY_BINARY_PATH,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                    **_ppq_backend_model_attestation_policy(),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.verified_claims == {}
    assert "code_measurement_fingerprint claim does not match policy" in (
        status.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_requires_remote_attestation_bundle_url() -> None:
    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {},
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert "attestation_bundle_url is required" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_ppq_private_refresh_requires_expected_repo() -> None:
    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert "repo or expected_repo is required" in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_ppq_private_refresh_requires_artifact_identity_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                "proxy_binary_digest": _test_digest("ppq-private-proxy"),
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyPPQVerifierClient.calls == []
    assert (
        "release_digests, expected_code_measurement_fingerprint, or "
        "allowed_code_measurement_fingerprints is required"
        in (status.failure_reason or "")
    )


@pytest.mark.asyncio
async def test_ppq_private_refresh_without_proxy_binary_reaches_backend_guardrail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DummyPPQVerifierClient.calls = []
    monkeypatch.setattr(
        "routstr.upstream.confidential_verifiers.httpx.AsyncClient",
        DummyPPQVerifierClient,
    )

    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "verifier_command": ["/usr/local/bin/routstr-ppq-private-verifier"],
                    "verifier_command_digest": VALID_VERIFIER_DIGEST,
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert DummyPPQVerifierClient.calls == []
    assert (
        "PPQ private require_model_attestations must be true for selected model routing"
    ) in (status.failure_reason or "")


@pytest.mark.asyncio
async def test_ppq_private_refresh_requires_ehbp_transport() -> None:
    provider = PPQPrivateUpstreamProvider(
        base_url="https://api.ppq.ai/private/v1",
        api_key="test",
    )
    provider.configure_confidentiality_from_settings(
        {
            "confidentiality": {
                "enabled": True,
                "mode": "ppq-private-tee",
                "model_ids": ["private/gpt-oss-120b"],
                "policy": {
                    "attestation_bundle_url": "https://api.ppq.ai/private",
                    "repo": "ppq-ai/private-tee",
                    "expected_code_measurement_fingerprint": EHBP_CODE_MEASUREMENT,
                    "transport": "plaintext",
                },
            }
        }
    )

    status = await provider.refresh_confidentiality_status()

    assert status.verified is False
    assert status.evidence_digest is None
    assert "transport must be ehbp" in (status.failure_reason or "")
