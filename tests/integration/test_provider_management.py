"""
Integration tests for provider management functionality.
Tests GET /v1/providers/ endpoint for listing and managing providers.
"""

import hashlib
import json
import time
from types import TracebackType
from typing import Any, Generator
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlmodel import select

from routstr.core.admin import admin_sessions
from routstr.core.db import ModelRow, UpstreamProviderRow
from routstr.nostr.discovery import _PROVIDERS_CACHE

from .utils import ResponseValidator


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


VALID_POLICY_DIGEST = _digest("valid-policy")
VALID_EVIDENCE_DIGEST = _digest("valid-evidence")
VALID_ROUTSTR_PUBLIC_KEY_DIGEST = _digest("valid-routstr-public-key")
VALID_ROUTSTR_TEE_CLAIMS_DIGEST = _digest("valid-routstr-tee-claims")
VALID_ROUTSTR_TEE_EVIDENCE_DIGEST = _digest("valid-routstr-tee-evidence")
VALID_ROUTSTR_HPKE_KEY_CONFIG_DIGEST = _digest("valid-routstr-hpke-key-config")
VALID_ROUTSTR_HPKE_PUBLIC_KEY_DIGEST = _digest("valid-routstr-hpke-public-key")


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


def _confidential_provider_listing_metadata(
    *,
    routstr_tee_ready: bool = True,
) -> dict[str, object]:
    return {
        "name": "Attested Provider",
        "about": "Publishes a confidential Routstr listing",
        "confidentiality": {
            "mode": "required",
            "required": True,
            "routstr_tee": _routstr_tee_status(ready=routstr_tee_ready),
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
        },
        "raw_verifier_claims": {"api_key": "SECRET_ANNOUNCEMENT_CONTENT"},
    }


@pytest.fixture(autouse=True)
def _clear_providers_cache() -> None:
    _PROVIDERS_CACHE.clear()


@pytest.fixture(autouse=True)
def _enable_provider_discovery() -> Generator[None, Any, Any]:
    """Enable provider discovery for all tests in this module"""
    with patch(
        "routstr.core.settings.settings.providers_refresh_interval_seconds", 300
    ):
        yield


@pytest.mark.integration
@pytest.mark.asyncio
async def test_providers_endpoint_default_response(
    integration_client: AsyncClient, db_snapshot: Any
) -> None:
    """Test GET /v1/providers/ endpoint returns list of providers in default format"""

    # Capture initial database state
    await db_snapshot.capture()

    # Mock the Nostr relay queries and onion fetching to avoid external dependencies
    mock_events: list[dict[str, Any]] = [
        {
            "id": "event1",
            "pubkey": "test_pubkey1",
            "kind": 38421,  # NIP-91 event kind
            "created_at": 1234567890,
            "content": '{"name": "Provider 1", "about": "Test provider 1"}',
            "tags": [
                ["d", "provider1"],
                ["u", "http://provider1.onion"],
            ],
        },
        {
            "id": "event2",
            "pubkey": "test_pubkey2",
            "kind": 38421,  # NIP-91 event kind
            "created_at": 1234567891,
            "content": '{"name": "Provider 2", "about": "Test provider 2"}',
            "tags": [
                ["d", "provider2"],
                ["u", "http://provider2.onion"],
            ],
        },
    ]

    # Mock the healthy provider check
    mock_fetch_responses = {
        "http://provider1.onion": {"status_code": 200, "json": {"status": "healthy"}},
        "http://provider2.onion": {"status_code": 200, "json": {"status": "healthy"}},
    }

    with patch(
        "routstr.nostr.discovery.query_nostr_relay_for_providers",
        return_value=mock_events,
    ):
        with patch("routstr.nostr.discovery.fetch_provider_health") as mock_fetch:
            # Configure mock to return appropriate responses
            mock_fetch.side_effect = lambda url: mock_fetch_responses.get(
                url, {"status_code": 500, "json": {"error": "Unknown provider"}}
            )

            response = await integration_client.get("/v1/providers/")

            assert response.status_code == 200
            data = response.json()

            # Validate response structure
            assert "providers" in data
            assert isinstance(data["providers"], list)

            # In default format, should return list of provider objects
            for provider in data["providers"]:
                assert isinstance(provider, dict)
                assert "endpoint_url" in provider
                assert provider["endpoint_url"].endswith(".onion")

    # Verify no database state changes
    diff = await db_snapshot.diff()
    assert len(diff["api_keys"]["added"]) == 0
    assert len(diff["api_keys"]["modified"]) == 0
    assert len(diff["api_keys"]["removed"]) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_providers_endpoint_with_include_json(
    integration_client: AsyncClient, db_snapshot: Any
) -> None:
    """Test GET /v1/providers/ with include_json=true returns full provider details"""

    # Capture initial database state
    await db_snapshot.capture()

    # Mock events with provider URLs
    mock_events: list[dict[str, Any]] = [
        {
            "id": "event1",
            "pubkey": "test_pubkey",
            "kind": 38421,  # NIP-91 event kind
            "created_at": 1234567890,
            "content": '{"name": "Test Provider", "about": "A test provider"}',
            "tags": [
                ["d", "test-provider"],
                ["u", "http://test-provider.onion"],
            ],
        }
    ]

    # Mock provider health check response
    mock_provider_response = {
        "status": "online",
        "name": "Test Provider",
        "models": ["gpt-3.5-turbo", "gpt-4"],
        "pricing": {"gpt-3.5-turbo": "0.002", "gpt-4": "0.03"},
    }

    with patch(
        "routstr.nostr.discovery.query_nostr_relay_for_providers",
        return_value=mock_events,
    ):
        with patch("routstr.nostr.discovery.fetch_provider_health") as mock_fetch:
            mock_fetch.return_value = {
                "status_code": 200,
                "json": mock_provider_response,
            }

            response = await integration_client.get("/v1/providers/?include_json=true")

            assert response.status_code == 200
            data = response.json()

            # Validate response structure
            assert "providers" in data
            assert isinstance(data["providers"], list)

            # With include_json=true, should return list of dictionaries with provider and health info
            for provider_data in data["providers"]:
                assert isinstance(provider_data, dict)
                # Each provider should have 'provider' and 'health' keys
                assert "provider" in provider_data
                assert "health" in provider_data

                provider_info = provider_data["provider"]
                assert "endpoint_url" in provider_info
                assert provider_info["endpoint_url"].endswith(".onion")

                health_info = provider_data["health"]
                assert isinstance(health_info, dict)
                assert "status_code" in health_info

    # Verify no database state changes
    diff = await db_snapshot.diff()
    assert len(diff["api_keys"]["added"]) == 0
    assert len(diff["api_keys"]["modified"]) == 0
    assert len(diff["api_keys"]["removed"]) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_providers_data_structure_validation(
    integration_client: AsyncClient,
) -> None:
    """Test provider data structure contains expected fields"""

    # Mock NIP-91 provider announcement event
    mock_events: list[dict[str, Any]] = [
        {
            "id": "event1",
            "pubkey": "test_pubkey",
            "created_at": 1234567890,
            "kind": 38421,  # NIP-91 event kind
            "content": '{"name": "Comprehensive Provider", "about": "A comprehensive AI provider"}',
            "tags": [
                ["d", "provider-123"],
                ["u", "https://api.provider.example/v1"],
                ["models", "gpt-3.5-turbo", "gpt-4"],
            ],
        }
    ]

    mock_health_response = {
        "status_code": 200,
        "endpoint": "models",
        "json": {
            "data": [
                {"id": "gpt-3.5-turbo", "object": "model"},
                {"id": "gpt-4", "object": "model"},
            ]
        },
    }

    with patch(
        "routstr.nostr.discovery.query_nostr_relay_for_providers",
        return_value=mock_events,
    ):
        with patch("routstr.nostr.discovery.fetch_provider_health") as mock_fetch:
            mock_fetch.return_value = mock_health_response

            response = await integration_client.get("/v1/providers/?include_json=true")
            assert response.status_code == 200

            data = response.json()
            providers = data["providers"]

            # Validate that provider data contains expected fields
            assert len(providers) > 0
            for provider_data in providers:
                # Should have provider and health keys based on actual implementation
                assert "provider" in provider_data
                assert "health" in provider_data

                provider_info = provider_data["provider"]
                # health_info = provider_data["health"]

                # Expected fields from NIP-91 parser (supported_models removed)
                expected_fields = ["id", "name", "endpoint_url"]
                for field in expected_fields:
                    assert field in provider_info


@pytest.mark.integration
@pytest.mark.asyncio
async def test_providers_endpoint_exposes_sanitized_confidentiality_listing_metadata(
    integration_client: AsyncClient,
) -> None:
    mock_events: list[dict[str, Any]] = [
        {
            "id": "attested-event",
            "pubkey": "attested_pubkey",
            "kind": 38421,
            "created_at": 1234567890,
            "content": json.dumps(_confidential_provider_listing_metadata()),
            "tags": [
                ["d", "attested-provider"],
                ["u", "http://attested-provider.onion"],
            ],
        }
    ]

    with patch(
        "routstr.nostr.discovery.query_nostr_relay_for_providers",
        return_value=mock_events,
    ):
        with patch("routstr.nostr.discovery.fetch_provider_health") as mock_fetch:
            mock_fetch.return_value = {
                "status_code": 200,
                "endpoint": "root",
                "json": {"status": "online"},
            }
            response = await integration_client.get("/v1/providers/")

    assert response.status_code == 200
    provider = response.json()["providers"][0]

    assert "content" not in provider
    assert "SECRET_ANNOUNCEMENT_CONTENT" not in json.dumps(provider)
    assert provider["metadata"]["confidentiality"][
        "routable_with_full_attestation"
    ] == {
        "tinfoil": ["gpt-secure"],
        "ppq-private": [],
        "privatemode": [],
    }
    assert provider["metadata"]["confidentiality"]["end_to_end_ready"] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_providers_endpoint_clears_copied_confidentiality_routes_without_tee(
    integration_client: AsyncClient,
) -> None:
    mock_events: list[dict[str, Any]] = [
        {
            "id": "copied-attested-event",
            "pubkey": "attested_pubkey",
            "kind": 38421,
            "created_at": 1234567890,
            "content": json.dumps(
                _confidential_provider_listing_metadata(routstr_tee_ready=False)
            ),
            "tags": [
                ["d", "copied-attested-provider"],
                ["u", "http://copied-attested-provider.onion"],
            ],
        }
    ]

    with patch(
        "routstr.nostr.discovery.query_nostr_relay_for_providers",
        return_value=mock_events,
    ):
        with patch("routstr.nostr.discovery.fetch_provider_health") as mock_fetch:
            mock_fetch.return_value = {
                "status_code": 200,
                "endpoint": "root",
                "json": {"status": "online"},
            }
            response = await integration_client.get("/v1/providers/")

    assert response.status_code == 200
    provider = response.json()["providers"][0]

    assert provider["metadata"]["confidentiality"][
        "routable_with_full_attestation"
    ] == {
        "tinfoil": [],
        "ppq-private": [],
        "privatemode": [],
    }
    assert provider["metadata"]["confidentiality"]["end_to_end_ready"] is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_providers_endpoint_no_providers_found(
    integration_client: AsyncClient,
) -> None:
    """Test providers endpoint when no providers are found"""

    # Force empty discovery by returning events that are filtered out
    mock_events: list[dict[str, Any]] = [
        {
            "id": "localhost-event",
            "pubkey": "ignored_pubkey",
            "kind": 38421,
            "created_at": 1234567899,
            "content": '{"name": "Local"}',
            "tags": [["d", "local"], ["u", "http://localhost:8000"]],
        }
    ]

    with patch(
        "routstr.nostr.discovery.query_nostr_relay_for_providers",
        return_value=mock_events,
    ):
        response = await integration_client.get("/v1/providers/")

        assert response.status_code == 200
        data = response.json()

        # Should return empty list
        assert "providers" in data
        assert isinstance(data["providers"], list)
        assert len(data["providers"]) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_providers_endpoint_rejects_private_discovery_endpoints(
    integration_client: AsyncClient,
) -> None:
    """Untrusted Nostr announcements must not trigger local/private SSRF probes."""
    mock_events: list[dict[str, Any]] = [
        {
            "id": "loopback-event",
            "pubkey": "attacker_pubkey",
            "kind": 38421,
            "created_at": 1234567890,
            "content": '{"name": "Loopback"}',
            "tags": [["d", "loopback"], ["u", "http://127.0.0.1:8080"]],
        },
        {
            "id": "metadata-event",
            "pubkey": "attacker_pubkey",
            "kind": 38421,
            "created_at": 1234567891,
            "content": '{"name": "Metadata"}',
            "tags": [["d", "metadata"], ["u", "http://169.254.169.254/latest"]],
        },
        {
            "id": "credential-event",
            "pubkey": "attacker_pubkey",
            "kind": 38421,
            "created_at": 1234567892,
            "content": '{"name": "Credential URL"}',
            "tags": [
                [
                    "d",
                    "credential-url",
                ],
                [
                    "u",
                    "https://user:password@example.com",
                ],
            ],
        },
        {
            "id": "valid-event",
            "pubkey": "provider_pubkey",
            "kind": 38421,
            "created_at": 1234567893,
            "content": '{"name": "Valid"}',
            "tags": [["d", "valid"], ["u", "http://provider.onion"]],
        },
    ]

    with patch(
        "routstr.nostr.discovery.query_nostr_relay_for_providers",
        return_value=mock_events,
    ):
        with patch("routstr.nostr.discovery.fetch_provider_health") as mock_fetch:
            mock_fetch.return_value = {
                "status_code": 200,
                "endpoint": "root",
                "json": {"status": "online"},
            }
            response = await integration_client.get("/v1/providers/?include_json=true")

    assert response.status_code == 200
    data = response.json()
    assert [item["provider"]["endpoint_url"] for item in data["providers"]] == [
        "http://provider.onion"
    ]
    mock_fetch.assert_awaited_once_with("http://provider.onion")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_health_rejects_hostname_resolving_to_private_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DNS names resolving to local/private IPs must not reach the HTTP client."""
    import socket

    from routstr.nostr.discovery import fetch_provider_health

    def fake_getaddrinfo(
        host: str,
        port: int | None,
        *args: object,
        **kwargs: object,
    ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
        assert host == "attacker.example"
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("127.0.0.1", port or 80),
            )
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    with patch("routstr.nostr.discovery.httpx.AsyncClient") as client_cls:
        health = await fetch_provider_health("http://attacker.example/v1")

    assert health["status_code"] == 400
    assert health["endpoint"] == "error"
    assert "Unsafe provider endpoint" in health["json"]["error"]
    client_cls.assert_not_called()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_providers_endpoint_omits_hostname_resolving_to_private_ip(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider discovery must not publish hostnames that resolve to private IPs."""
    import socket

    mock_events: list[dict[str, Any]] = [
        {
            "id": "private-dns-event",
            "pubkey": "attacker_pubkey",
            "kind": 38421,
            "created_at": 1234567890,
            "content": '{"name": "Private DNS"}',
            "tags": [["d", "private-dns"], ["u", "http://attacker.example/v1"]],
        },
        {
            "id": "valid-event",
            "pubkey": "provider_pubkey",
            "kind": 38421,
            "created_at": 1234567891,
            "content": '{"name": "Valid"}',
            "tags": [["d", "valid"], ["u", "http://provider.onion"]],
        },
    ]

    def fake_getaddrinfo(
        host: str,
        port: int | None,
        *args: object,
        **kwargs: object,
    ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
        assert host == "attacker.example"
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("127.0.0.1", port or 80),
            )
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    with patch(
        "routstr.nostr.discovery.query_nostr_relay_for_providers",
        return_value=mock_events,
    ):
        response = await integration_client.get("/v1/providers/")

    assert response.status_code == 200
    data = response.json()
    assert [provider["endpoint_url"] for provider in data["providers"]] == [
        "http://provider.onion"
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_providers_endpoint_offline_providers(
    integration_client: AsyncClient,
) -> None:
    """Test providers endpoint handling of offline/unhealthy providers"""

    mock_events: list[dict[str, Any]] = [
        {
            "id": "event1",
            "pubkey": "healthy_provider_pubkey",
            "kind": 38421,  # NIP-91 event kind
            "created_at": 1234567890,
            "content": '{"name": "Healthy Provider", "about": "Healthy provider announcement"}',
            "tags": [
                ["d", "healthy-provider"],
                ["u", "http://healthy-provider.onion"],
            ],
        },
        {
            "id": "event2",
            "pubkey": "offline_provider_pubkey",
            "kind": 38421,  # NIP-91 event kind
            "created_at": 1234567891,
            "content": '{"name": "Offline Provider", "about": "Offline provider announcement"}',
            "tags": [
                ["d", "offline-provider"],
                ["u", "http://offline-provider.onion"],
            ],
        },
    ]

    # Mock one healthy and one offline provider
    def mock_fetch_provider_health(url: str) -> dict[str, Any]:
        if "healthy" in url:
            return {
                "status_code": 200,
                "endpoint": "root",
                "json": {"status": "online"},
            }
        else:
            return {
                "status_code": 500,
                "endpoint": "error",
                "json": {"error": "Service unavailable"},
            }

    with patch(
        "routstr.nostr.discovery.query_nostr_relay_for_providers",
        return_value=mock_events,
    ):
        with patch(
            "routstr.nostr.discovery.fetch_provider_health",
            side_effect=mock_fetch_provider_health,
        ):
            response = await integration_client.get("/v1/providers/?include_json=true")

            assert response.status_code == 200
            data = response.json()

            # Should include both providers regardless of status
            assert len(data["providers"]) == 2

            # Verify that offline providers are still included but marked appropriately
            for provider_data in data["providers"]:
                assert "provider" in provider_data
                assert "health" in provider_data

                provider_info = provider_data["provider"]
                health_info = provider_data["health"]

                if "offline" in provider_info["endpoint_url"]:
                    # Offline provider should have error information in health
                    assert health_info["status_code"] == 500
                    assert "error" in health_info["json"]
                else:
                    # Healthy provider should have successful health check
                    assert health_info["status_code"] == 200
                    assert (
                        "status" in health_info["json"]
                        or "error" not in health_info["json"]
                    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_providers_endpoint_duplicate_urls(
    integration_client: AsyncClient,
) -> None:
    """Test providers endpoint handles duplicate URLs correctly"""

    # Mock events with duplicate provider events (same event ID) - should be deduplicated by relay query logic
    mock_events: list[dict[str, Any]] = [
        {
            "id": "event1",
            "pubkey": "provider_pubkey",
            "kind": 38421,  # NIP-91 event kind
            "created_at": 1234567890,
            "content": '{"name": "Provider", "about": "Provider announcement"}',
            "tags": [
                ["d", "provider-1"],
                ["u", "http://provider.onion"],
            ],
        },
        {
            "id": "event2",
            "pubkey": "other_provider_pubkey",
            "kind": 38421,  # NIP-91 event kind
            "created_at": 1234567892,
            "content": '{"name": "Other Provider", "about": "Different provider announcement"}',
            "tags": [
                ["d", "other-provider"],
                ["u", "http://other-provider.onion"],
            ],
        },
    ]

    with patch(
        "routstr.nostr.discovery.query_nostr_relay_for_providers",
        return_value=mock_events,
    ):
        with patch("routstr.nostr.discovery.fetch_provider_health") as mock_fetch:
            mock_fetch.return_value = {
                "status_code": 200,
                "endpoint": "root",
                "json": {"status": "online"},
            }

            response = await integration_client.get("/v1/providers/")

            assert response.status_code == 200
            data = response.json()

            # Should return 2 unique providers based on events
            providers = data["providers"]
            assert len(providers) == 2  # 2 unique events

            # Verify all providers are unique by endpoint_url
            endpoint_urls = []
            for provider_data in providers:
                endpoint_urls.append(provider_data["endpoint_url"])

            unique_endpoints = set(endpoint_urls)
            assert len(unique_endpoints) == len(endpoint_urls)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_providers_endpoint_nostr_relay_failures(
    integration_client: AsyncClient,
) -> None:
    """Test providers endpoint handles Nostr relay failures gracefully"""

    # Mock relay failure
    async def failing_query(*args: Any, **kwargs: Any) -> None:
        raise Exception("Connection to relay failed")

    with patch(
        "routstr.nostr.discovery.query_nostr_relay_for_providers",
        side_effect=failing_query,
    ):
        response = await integration_client.get("/v1/providers/")

        # Should still return 200 with empty providers list
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert isinstance(data["providers"], list)
        assert len(data["providers"]) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_providers_endpoint_malformed_urls(
    integration_client: AsyncClient,
) -> None:
    """Test providers endpoint handles malformed URLs in Nostr events"""

    mock_events: list[dict[str, Any]] = [
        {
            "id": "event1",
            "content": "Valid provider: http://good-provider.onion",
            "created_at": 1234567890,
        },
        {
            "id": "event2",
            "content": "Invalid URL: not-a-valid-url.onion",
            "created_at": 1234567891,
        },
        {
            "id": "event3",
            "content": "No URLs here, just text",
            "created_at": 1234567892,
        },
    ]

    with patch(
        "routstr.nostr.discovery.query_nostr_relay_for_providers",
        return_value=mock_events,
    ):
        with patch("routstr.nostr.discovery.fetch_provider_health") as mock_fetch:
            mock_fetch.return_value = {"status_code": 200, "json": {"status": "online"}}

            response = await integration_client.get("/v1/providers/")

            assert response.status_code == 200
            data = response.json()

            # With NIP-91-only parsing, events without required tags are ignored
            assert "providers" in data
            assert isinstance(data["providers"], list)
            assert len(data["providers"]) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_providers_endpoint_response_format(
    integration_client: AsyncClient,
) -> None:
    """Test providers endpoint response format consistency"""

    mock_events: list[dict[str, Any]] = [
        {
            "id": "event1",
            "content": "Provider: http://test-provider.onion",
            "created_at": 1234567890,
        }
    ]

    with patch(
        "routstr.nostr.discovery.query_nostr_relay_for_providers",
        return_value=mock_events,
    ):
        with patch("routstr.nostr.discovery.fetch_provider_health") as mock_fetch:
            mock_fetch.return_value = {"status_code": 200, "json": {"status": "online"}}

            # Test default format
            response = await integration_client.get("/v1/providers/")
            assert response.status_code == 200

            validator = ResponseValidator()
            validation = validator.validate_success_response(
                response, expected_status=200, required_fields=["providers"]
            )
            assert validation["valid"]

            data = response.json()
            assert isinstance(data, dict)
            assert "providers" in data
            assert isinstance(data["providers"], list)

            # Test include_json format
            response_json = await integration_client.get(
                "/v1/providers/?include_json=true"
            )
            assert response_json.status_code == 200

            data_json = response_json.json()
            assert isinstance(data_json, dict)
            assert "providers" in data_json
            assert isinstance(data_json["providers"], list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_providers_endpoint_concurrent_requests(
    integration_client: AsyncClient,
) -> None:
    """Test providers endpoint handles concurrent requests correctly"""

    from .utils import ConcurrencyTester

    mock_events: list[dict[str, Any]] = [
        {
            "id": "event1",
            "content": "Provider: http://concurrent-provider.onion",
            "created_at": 1234567890,
        }
    ]

    with patch(
        "routstr.nostr.discovery.query_nostr_relay_for_providers",
        return_value=mock_events,
    ):
        with patch("routstr.nostr.discovery.fetch_provider_health") as mock_fetch:
            mock_fetch.return_value = {"status_code": 200, "json": {"status": "online"}}

            # Create concurrent requests
            requests = [{"method": "GET", "url": "/v1/providers/"} for _ in range(10)]

            tester = ConcurrencyTester()
            responses = await tester.run_concurrent_requests(
                integration_client, requests, max_concurrent=5
            )

            # All should succeed
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert "providers" in data


@pytest.mark.integration
@pytest.mark.asyncio
async def test_providers_endpoint_parameter_validation(
    integration_client: AsyncClient,
) -> None:
    """Test providers endpoint parameter handling"""

    mock_events: list[dict[str, Any]] = [
        {
            "id": "event1",
            "pubkey": "param_pubkey",
            "kind": 38421,
            "created_at": 1234567890,
            "content": '{"name": "Param Test Provider"}',
            "tags": [
                ["d", "param-test-provider"],
                ["u", "http://param-test-provider.onion"],
            ],
        }
    ]

    with patch(
        "routstr.nostr.discovery.query_nostr_relay_for_providers",
        return_value=mock_events,
    ):
        with patch("routstr.nostr.discovery.fetch_provider_health") as mock_fetch:
            mock_fetch.return_value = {"status_code": 200, "json": {"status": "online"}}

            # Test various parameter values
            test_cases = [
                ("/v1/providers/", False),  # Default
                ("/v1/providers/?include_json=false", False),  # Explicit false
                ("/v1/providers/?include_json=true", True),  # Explicit true
                ("/v1/providers/?include_json=1", True),  # Truthy value
                ("/v1/providers/?include_json=0", False),  # Falsy value
            ]

            for url, expected_json_format in test_cases:
                response = await integration_client.get(url)
                assert response.status_code == 200

                data = response.json()
                providers = data["providers"]

                if len(providers) > 0:
                    if expected_json_format:
                        # Should be list of {provider, health} dictionaries
                        for item in providers:
                            assert isinstance(item, dict)
                            assert "provider" in item and "health" in item
                    else:
                        # Should be list of provider objects
                        for provider in providers:
                            assert isinstance(provider, dict)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_database_changes_during_provider_operations(
    integration_client: AsyncClient, db_snapshot: Any
) -> None:
    """Comprehensive test that provider operations don't modify database state"""

    # Capture initial state
    await db_snapshot.capture()

    mock_events: list[dict[str, Any]] = [
        {
            "id": "event1",
            "content": "Provider: http://no-db-change-provider.onion",
            "created_at": 1234567890,
        }
    ]

    with patch(
        "routstr.nostr.discovery.query_nostr_relay_for_providers",
        return_value=mock_events,
    ):
        with patch("routstr.nostr.discovery.fetch_provider_health") as mock_fetch:
            mock_fetch.return_value = {"status_code": 200, "json": {"status": "online"}}

            # Make multiple requests with different parameters
            endpoints = [
                "/v1/providers/",
                "/v1/providers/?include_json=true",
                "/v1/providers/?include_json=false",
            ]

            for endpoint in endpoints:
                response = await integration_client.get(endpoint)
                assert response.status_code == 200

                # Check no database changes after each request
                current_diff = await db_snapshot.diff()
                assert len(current_diff["api_keys"]["added"]) == 0
                assert len(current_diff["api_keys"]["modified"]) == 0
                assert len(current_diff["api_keys"]["removed"]) == 0

    # Final verification - database state should be identical
    final_diff = await db_snapshot.diff()
    assert final_diff["api_keys"]["added"] == []
    assert final_diff["api_keys"]["modified"] == []
    assert final_diff["api_keys"]["removed"] == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_provider_types_include_confidential_integrations(
    integration_client: AsyncClient,
) -> None:
    admin_token = "test-admin-token-provider-types"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        response = await integration_client.get("/admin/api/provider-types")
    finally:
        admin_sessions.pop(admin_token, None)

    assert response.status_code == 200
    provider_types = {item["id"]: item for item in response.json()}

    assert provider_types["tinfoil"]["default_base_url"] == (
        "https://inference.tinfoil.sh/v1"
    )
    assert provider_types["tinfoil"]["fixed_base_url"] is True
    assert provider_types["ppq-private"]["default_base_url"] == (
        "https://api.ppq.ai/private/v1"
    )
    assert provider_types["ppq-private"]["can_create_account"] is False
    assert provider_types["privatemode"]["default_base_url"] == (
        "http://127.0.0.1:8080/v1"
    )
    assert provider_types["privatemode"]["fixed_base_url"] is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_unknown_upstream_provider_type(
    integration_client: AsyncClient,
) -> None:
    admin_token = "test-admin-token-unknown-provider-type"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        response = await integration_client.post(
            "/admin/api/upstream-providers",
            json={
                "provider_type": "confidential-looking-but-unregistered",
                "base_url": "https://example.com/v1",
                "api_key": "sk-test",
                "provider_fee": 1.0,
            },
        )
    finally:
        admin_sessions.pop(admin_token, None)

    assert response.status_code == 400
    assert response.json()["detail"] == "Provider type is not registered"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_fixed_base_url_provider_redirect(
    integration_client: AsyncClient,
) -> None:
    admin_token = "test-admin-token-fixed-base-url"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        response = await integration_client.post(
            "/admin/api/upstream-providers",
            json={
                "provider_type": "tinfoil",
                "base_url": "https://attacker.example/v1",
                "api_key": "sk-test",
                "provider_fee": 1.0,
            },
        )
    finally:
        admin_sessions.pop(admin_token, None)

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Provider type tinfoil requires base_url https://inference.tinfoil.sh/v1"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_update_rejects_fixed_base_url_provider_redirect(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    provider = UpstreamProviderRow(
        provider_type="generic",
        base_url="https://example.com/v1",
        api_key="sk-test",
        enabled=True,
        provider_fee=1.0,
    )
    integration_session.add(provider)
    await integration_session.commit()
    await integration_session.refresh(provider)

    admin_token = "test-admin-token-update-fixed-base-url"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        response = await integration_client.patch(
            f"/admin/api/upstream-providers/{provider.id}",
            json={
                "provider_type": "tinfoil",
                "base_url": "https://attacker.example/v1",
            },
        )
    finally:
        admin_sessions.pop(admin_token, None)

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Provider type tinfoil requires base_url https://inference.tinfoil.sh/v1"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_remote_privatemode_proxy_base_url(
    integration_client: AsyncClient,
) -> None:
    admin_token = "test-admin-token-remote-privatemode"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        response = await integration_client.post(
            "/admin/api/upstream-providers",
            json={
                "provider_type": "privatemode",
                "base_url": "https://api.privatemode.ai/v1",
                "api_key": "sk-test",
                "provider_fee": 1.0,
            },
        )
    finally:
        admin_sessions.pop(admin_token, None)

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Privatemode proxy base_url must use a loopback host"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_update_rejects_remote_privatemode_proxy_base_url(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    provider = UpstreamProviderRow(
        provider_type="privatemode",
        base_url="http://127.0.0.1:8080/v1",
        api_key="sk-test",
        enabled=True,
        provider_fee=1.0,
    )
    integration_session.add(provider)
    await integration_session.commit()
    await integration_session.refresh(provider)

    admin_token = "test-admin-token-update-remote-privatemode"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        response = await integration_client.patch(
            f"/admin/api/upstream-providers/{provider.id}",
            json={"base_url": "https://api.privatemode.ai/v1"},
        )
    finally:
        admin_sessions.pop(admin_token, None)

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Privatemode proxy base_url must use a loopback host"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_non_private_ppq_private_base_url(
    integration_client: AsyncClient,
) -> None:
    admin_token = "test-admin-token-ppq-private-base-url"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        response = await integration_client.post(
            "/admin/api/upstream-providers",
            json={
                "provider_type": "ppq-private",
                "base_url": "https://api.ppq.ai/v1",
                "api_key": "sk-test",
                "provider_fee": 1.0,
            },
        )
    finally:
        admin_sessions.pop(admin_token, None)

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "PPQ private base_url must include a private path segment"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_unowned_ppq_private_base_url(
    integration_client: AsyncClient,
) -> None:
    admin_token = "test-admin-token-ppq-private-unowned-base-url"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        response = await integration_client.post(
            "/admin/api/upstream-providers",
            json={
                "provider_type": "ppq-private",
                "base_url": "https://attacker.example/private/v1",
                "api_key": "sk-test",
                "provider_fee": 1.0,
            },
        )
    finally:
        admin_sessions.pop(admin_token, None)

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "PPQ private base_url host must be ppq.ai or a ppq.ai subdomain"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_update_rejects_non_private_ppq_private_base_url(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    provider = UpstreamProviderRow(
        provider_type="ppq-private",
        base_url="https://api.ppq.ai/private/v1",
        api_key="sk-test",
        enabled=True,
        provider_fee=1.0,
    )
    integration_session.add(provider)
    await integration_session.commit()
    await integration_session.refresh(provider)

    admin_token = "test-admin-token-update-ppq-private-base-url"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        response = await integration_client.patch(
            f"/admin/api/upstream-providers/{provider.id}",
            json={"base_url": "https://api.ppq.ai/v1"},
        )
    finally:
        admin_sessions.pop(admin_token, None)

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "PPQ private base_url must include a private path segment"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_ppq_private_catalog_origin_mismatch(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token-ppq-private-catalog-origin"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.post(
                "/admin/api/upstream-providers",
                json={
                    "provider_type": "ppq-private",
                    "base_url": "https://api.ppq.ai/private/v1",
                    "api_key": "sk-test",
                    "provider_fee": 1.0,
                    "provider_settings": {
                        "catalog_base_url": "https://catalog.attacker.example/v1"
                    },
                },
            )

        assert response.status_code == 400
        assert response.json()["detail"] == (
            "PPQ private catalog_base_url origin must match provider base_url origin"
        )

        rows = (
            await integration_session.exec(
                select(UpstreamProviderRow).where(
                    UpstreamProviderRow.provider_type == "ppq-private"
                )
            )
        ).all()
        assert rows == []
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_update_rejects_ppq_private_catalog_origin_mismatch(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    provider = UpstreamProviderRow(
        provider_type="ppq-private",
        base_url="https://api.ppq.ai/private/v1",
        api_key="sk-test",
        enabled=True,
        provider_fee=1.0,
    )
    integration_session.add(provider)
    await integration_session.commit()
    await integration_session.refresh(provider)

    admin_token = "test-admin-token-update-ppq-private-catalog-origin"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.patch(
                f"/admin/api/upstream-providers/{provider.id}",
                json={
                    "provider_settings": {
                        "catalog_base_url": "https://catalog.attacker.example/v1"
                    },
                },
            )

        assert response.status_code == 400
        assert response.json()["detail"] == (
            "PPQ private catalog_base_url origin must match provider base_url origin"
        )

        await integration_session.refresh(provider)
        assert provider.provider_settings is None
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_update_rejects_ppq_private_base_url_that_breaks_stored_catalog(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    provider = UpstreamProviderRow(
        provider_type="ppq-private",
        base_url="https://api.ppq.ai/private/v1",
        api_key="sk-test",
        enabled=True,
        provider_fee=1.0,
        provider_settings='{"catalog_base_url":"https://api.ppq.ai/v1"}',
    )
    integration_session.add(provider)
    await integration_session.commit()
    await integration_session.refresh(provider)

    admin_token = "test-admin-token-update-ppq-private-stored-catalog"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.patch(
                f"/admin/api/upstream-providers/{provider.id}",
                json={"base_url": "https://tenant.ppq.ai/private/v1"},
            )

        assert response.status_code == 400
        assert response.json()["detail"] == (
            "PPQ private catalog_base_url origin must match provider base_url origin"
        )

        await integration_session.refresh(provider)
        assert provider.base_url == "https://api.ppq.ai/private/v1"
        assert (
            provider.provider_settings == '{"catalog_base_url":"https://api.ppq.ai/v1"}'
        )
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_update_ignores_malformed_stored_provider_settings(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    provider = UpstreamProviderRow(
        provider_type="ppq-private",
        base_url="https://api.ppq.ai/private/v1",
        api_key="sk-test",
        enabled=True,
        provider_fee=1.0,
        provider_settings='{"catalog_base_url":"https://api.ppq.ai/v1"',
    )
    integration_session.add(provider)
    await integration_session.commit()
    await integration_session.refresh(provider)

    admin_token = "test-admin-token-update-malformed-provider-settings"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.patch(
                f"/admin/api/upstream-providers/{provider.id}",
                json={"enabled": False},
            )

        assert response.status_code == 200
        assert response.json()["enabled"] is False

        await integration_session.refresh(provider)
        assert provider.enabled is False
        assert (
            provider.provider_settings == '{"catalog_base_url":"https://api.ppq.ai/v1"'
        )
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_update_ignores_non_standard_stored_provider_settings(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    provider = UpstreamProviderRow(
        provider_type="ppq-private",
        base_url="https://api.ppq.ai/private/v1",
        api_key="sk-test",
        enabled=True,
        provider_fee=1.0,
        provider_settings='{"catalog_base_url":"https://api.ppq.ai/v1","ttl":NaN}',
    )
    integration_session.add(provider)
    await integration_session.commit()
    await integration_session.refresh(provider)

    admin_token = "test-admin-token-update-non-standard-provider-settings"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.patch(
                f"/admin/api/upstream-providers/{provider.id}",
                json={"enabled": False},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["enabled"] is False
        assert body["provider_settings"] is None

        await integration_session.refresh(provider)
        assert provider.enabled is False
        assert provider.provider_settings == (
            '{"catalog_base_url":"https://api.ppq.ai/v1","ttl":NaN}'
        )
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_list_ignores_malformed_stored_provider_settings(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    provider = UpstreamProviderRow(
        provider_type="ppq-private",
        base_url="https://api.ppq.ai/private/v1",
        api_key="sk-test",
        enabled=True,
        provider_fee=1.0,
        provider_settings='{"catalog_base_url":"https://api.ppq.ai/v1"',
    )
    integration_session.add(provider)
    await integration_session.commit()
    await integration_session.refresh(provider)

    admin_token = "test-admin-token-list-malformed-provider-settings"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        response = await integration_client.get("/admin/api/upstream-providers")
    finally:
        admin_sessions.pop(admin_token, None)

    assert response.status_code == 200
    rows = response.json()
    matching = [row for row in rows if row["id"] == provider.id]
    assert len(matching) == 1
    assert matching[0]["provider_settings"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_detail_ignores_malformed_stored_provider_settings(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    provider = UpstreamProviderRow(
        provider_type="ppq-private",
        base_url="https://api.ppq.ai/private/v1",
        api_key="sk-test",
        enabled=True,
        provider_fee=1.0,
        provider_settings='{"catalog_base_url":"https://api.ppq.ai/v1"',
    )
    integration_session.add(provider)
    await integration_session.commit()
    await integration_session.refresh(provider)

    admin_token = "test-admin-token-detail-malformed-provider-settings"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        response = await integration_client.get(
            f"/admin/api/upstream-providers/{provider.id}"
        )
    finally:
        admin_sessions.pop(admin_token, None)

    assert response.status_code == 200
    assert response.json()["provider_settings"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_non_finite_provider_fee(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token-provider-fee"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.post(
                "/admin/api/upstream-providers",
                content=(
                    '{"provider_type":"tinfoil",'
                    '"base_url":"https://inference.tinfoil.sh/v1",'
                    '"api_key":"sk-test",'
                    '"provider_fee":NaN}'
                ),
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 400
        assert response.json()["detail"] == "provider_fee must be a finite number"

        rows = (
            await integration_session.exec(
                select(UpstreamProviderRow).where(
                    UpstreamProviderRow.base_url == "https://inference.tinfoil.sh/v1"
                )
            )
        ).all()
        assert rows == []
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_non_positive_provider_fee(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token-provider-negative-fee"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.post(
                "/admin/api/upstream-providers",
                json={
                    "provider_type": "tinfoil",
                    "base_url": "https://inference.tinfoil.sh/v1",
                    "api_key": "sk-test",
                    "provider_fee": -1.0,
                },
            )

        assert response.status_code == 400
        assert response.json()["detail"] == "provider_fee must be positive"

        rows = (
            await integration_session.exec(
                select(UpstreamProviderRow).where(
                    UpstreamProviderRow.base_url == "https://inference.tinfoil.sh/v1"
                )
            )
        ).all()
        assert rows == []
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_non_standard_model_override_json(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token-model-override"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    provider = UpstreamProviderRow(
        provider_type="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        api_key="sk-upstream-test",
        enabled=True,
        provider_fee=1.01,
    )
    integration_session.add(provider)
    await integration_session.commit()
    await integration_session.refresh(provider)

    try:
        with patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock):
            response = await integration_client.post(
                f"/admin/api/upstream-providers/{provider.id}/models",
                content=(
                    '{"id":"gpt-secure","name":"GPT Secure","description":"test",'
                    '"created":1,"context_length":4096,'
                    '"architecture":{"modality":"text->text",'
                    '"input_modalities":["text"],"output_modalities":["text"],'
                    '"tokenizer":"gpt","instruct_type":null},'
                    '"pricing":{"prompt":NaN,"completion":0.0,"request":0.0,'
                    '"image":0.0,"web_search":0.0,"internal_reasoning":0.0}}'
                ),
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 400
        assert response.json()["detail"] == (
            "model pricing must be strict standard JSON"
        )

        rows = (
            await integration_session.exec(
                select(ModelRow).where(ModelRow.id == "gpt-secure")
            )
        ).all()
        assert rows == []
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_string_non_finite_model_override_pricing(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token-model-override-string-nan"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    provider = UpstreamProviderRow(
        provider_type="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        api_key="sk-upstream-test",
        enabled=True,
        provider_fee=1.01,
    )
    integration_session.add(provider)
    await integration_session.commit()
    await integration_session.refresh(provider)

    try:
        with patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock):
            response = await integration_client.post(
                f"/admin/api/upstream-providers/{provider.id}/models",
                json={
                    "id": "gpt-secure",
                    "name": "GPT Secure",
                    "description": "test",
                    "created": 1,
                    "context_length": 4096,
                    "architecture": {
                        "modality": "text->text",
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                        "tokenizer": "gpt",
                        "instruct_type": None,
                    },
                    "pricing": {
                        "prompt": "NaN",
                        "completion": 0.0,
                        "request": 0.0,
                        "image": 0.0,
                        "web_search": 0.0,
                        "internal_reasoning": 0.0,
                    },
                },
            )

        assert response.status_code == 400
        assert response.json()["detail"] == "model pricing values must be finite"

        rows = (
            await integration_session.exec(
                select(ModelRow).where(ModelRow.id == "gpt-secure")
            )
        ).all()
        assert rows == []
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_non_standard_provider_settings_json(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token-provider-settings"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.post(
                "/admin/api/upstream-providers",
                content=(
                    '{"provider_type":"tinfoil",'
                    '"base_url":"https://inference.tinfoil.sh/v1",'
                    '"api_key":"sk-test",'
                    '"provider_settings":{'
                    '"confidentiality":{"policy":{"max_verifier_age_seconds":NaN}}'
                    "}}"
                ),
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 400
        assert response.json()["detail"] == (
            "provider_settings must be strict standard JSON"
        )

        rows = (
            await integration_session.exec(
                select(UpstreamProviderRow).where(
                    UpstreamProviderRow.base_url == "https://inference.tinfoil.sh/v1"
                )
            )
        ).all()
        assert rows == []
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_inline_secret_confidential_provider_settings_on_create(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token-provider-settings-secret"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"
    base_url = "https://secret-policy-create.tinfoil.example/v1"

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.post(
                "/admin/api/upstream-providers",
                json={
                    "provider_type": "tinfoil",
                    "base_url": base_url,
                    "api_key": "sk-test",
                    "provider_settings": {
                        "confidentiality": {
                            "mode": "tinfoil",
                            "model_ids": ["tinfoil/gpt-secure"],
                            "policy": {
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "headers": {
                                    "Authorization": "Bearer sk-secret-provider-token"
                                },
                            },
                        }
                    },
                },
            )

        assert response.status_code == 400
        assert "authorization must not be embedded" in response.json()["detail"]

        rows = (
            await integration_session.exec(
                select(UpstreamProviderRow).where(
                    UpstreamProviderRow.base_url == base_url
                )
            )
        ).all()
        assert rows == []
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_secret_anywhere_in_provider_settings_on_create(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token-provider-settings-top-secret"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"
    base_url = "https://secret-provider-settings.tinfoil.example/v1"

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.post(
                "/admin/api/upstream-providers",
                json={
                    "provider_type": "tinfoil",
                    "base_url": base_url,
                    "api_key": "sk-test",
                    "provider_settings": {
                        "headers": {"Authorization": "Bearer sk-secret-provider-token"},
                        "confidentiality": {
                            "mode": "tinfoil",
                            "model_ids": ["tinfoil/gpt-secure"],
                            "policy": {
                                "repo": "tinfoilsh/confidential-gpt-secure",
                            },
                        },
                    },
                },
            )

        assert response.status_code == 400
        assert "authorization must not be embedded" in response.json()["detail"]
        assert "sk-secret-provider-token" not in response.text

        rows = (
            await integration_session.exec(
                select(UpstreamProviderRow).where(
                    UpstreamProviderRow.base_url == base_url
                )
            )
        ).all()
        assert rows == []
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_placeholder_confidential_policy_pin_on_create(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token-placeholder-policy-pin"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"
    base_url = "https://inference.tinfoil.sh/v1"

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.post(
                "/admin/api/upstream-providers",
                json={
                    "provider_type": "tinfoil",
                    "base_url": base_url,
                    "api_key": "sk-test-placeholder-pin",
                    "provider_settings": {
                        "confidentiality": {
                            "mode": "tinfoil",
                            "model_ids": ["tinfoil/gpt-secure"],
                            "policy": {
                                "repo": "tinfoilsh/confidential-model-router",
                                "expected_release_digest": "sha256:" + ("a" * 64),
                            },
                        }
                    },
                },
            )

        assert response.status_code == 400
        assert "placeholder digest" in response.json()["detail"]

        rows = (
            await integration_session.exec(
                select(UpstreamProviderRow).where(
                    UpstreamProviderRow.api_key == "sk-test-placeholder-pin"
                )
            )
        ).all()
        assert rows == []
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_placeholder_confidential_policy_pin_on_update(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token-placeholder-policy-pin-update"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    provider = UpstreamProviderRow(
        provider_type="tinfoil",
        base_url="https://inference.tinfoil.sh/v1",
        api_key="sk-upstream-placeholder-update",
        enabled=True,
        provider_fee=1.01,
        provider_settings=(
            '{"confidentiality":{"mode":"tinfoil",'
            '"model_ids":["tinfoil/gpt-secure"],'
            '"policy":{"repo":"tinfoilsh/confidential-model-router"}}}'
        ),
    )
    integration_session.add(provider)
    await integration_session.commit()
    await integration_session.refresh(provider)
    original_provider_settings = provider.provider_settings

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.patch(
                f"/admin/api/upstream-providers/{provider.id}",
                json={
                    "provider_settings": {
                        "confidentiality": {
                            "mode": "tinfoil",
                            "model_ids": ["tinfoil/gpt-secure"],
                            "policy": {
                                "repo": "tinfoilsh/confidential-model-router",
                                "expected_release_digest": "sha256:" + ("b" * 64),
                            },
                        }
                    }
                },
            )

        assert response.status_code == 400
        assert "placeholder digest" in response.json()["detail"]

        await integration_session.refresh(provider)
        assert provider.provider_settings == original_provider_settings
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_placeholder_top_level_attestation_policy_pin(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token-placeholder-attestation-policy"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"
    base_url = "https://inference.tinfoil.sh/v1"

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.post(
                "/admin/api/upstream-providers",
                json={
                    "provider_type": "tinfoil",
                    "base_url": base_url,
                    "api_key": "sk-test-placeholder-alias",
                    "provider_settings": {
                        "confidentiality": {
                            "mode": "tinfoil",
                            "model_ids": ["tinfoil/gpt-secure"],
                        },
                        "attestation_policy": {
                            "repo": "tinfoilsh/confidential-model-router",
                            "allowed_release_digests": ["sha256:" + ("c" * 64)],
                        },
                    },
                },
            )

        assert response.status_code == 400
        assert "placeholder digest" in response.json()["detail"]

        rows = (
            await integration_session.exec(
                select(UpstreamProviderRow).where(
                    UpstreamProviderRow.api_key == "sk-test-placeholder-alias"
                )
            )
        ).all()
        assert rows == []
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_confidentiality_mode_provider_type_mismatch(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token-confidential-mode-mismatch"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.post(
                "/admin/api/upstream-providers",
                json={
                    "provider_type": "ppq-private",
                    "base_url": "https://api.ppq.ai/private/v1",
                    "api_key": "sk-test-confidential-mode-mismatch",
                    "provider_settings": {
                        "confidentiality": {
                            "mode": "tinfoil",
                            "model_ids": ["private/gpt-oss-120b"],
                            "policy": {
                                "repo": "ppq-ai/private-tee",
                                "expected_release_digest": (
                                    "sha256:ebb921d8572bc8f6b4e04812a7e65bffcf6ac1a2fabe8a55115a13d61e901b85"
                                ),
                            },
                        }
                    },
                },
            )

        assert response.status_code == 400
        assert response.json()["detail"] == (
            "confidentiality mode tinfoil does not match provider_type ppq-private"
        )

        rows = (
            await integration_session.exec(
                select(UpstreamProviderRow).where(
                    UpstreamProviderRow.api_key == "sk-test-confidential-mode-mismatch"
                )
            )
        ).all()
        assert rows == []
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_unknown_confidentiality_mode(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token-unknown-confidential-mode"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.post(
                "/admin/api/upstream-providers",
                json={
                    "provider_type": "tinfoil",
                    "base_url": "https://inference.tinfoil.sh/v1",
                    "api_key": "sk-test-unknown-confidential-mode",
                    "provider_settings": {
                        "confidentiality": {
                            "mode": "custom-confidential-mode",
                            "model_ids": ["tinfoil/gpt-secure"],
                            "policy": {
                                "repo": "tinfoilsh/confidential-model-router",
                                "expected_release_digest": (
                                    "sha256:ebb921d8572bc8f6b4e04812a7e65bffcf6ac1a2fabe8a55115a13d61e901b85"
                                ),
                            },
                        }
                    },
                },
            )

        assert response.status_code == 400
        assert response.json()["detail"] == (
            "confidentiality mode must be tinfoil, ppq-private-tee, or privatemode"
        )

        rows = (
            await integration_session.exec(
                select(UpstreamProviderRow).where(
                    UpstreamProviderRow.api_key == "sk-test-unknown-confidential-mode"
                )
            )
        ).all()
        assert rows == []
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_tinfoil_selected_model_policy_without_model_attestations(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token-tinfoil-missing-model-attestations"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.post(
                "/admin/api/upstream-providers",
                json={
                    "provider_type": "tinfoil",
                    "base_url": "https://inference.tinfoil.sh/v1",
                    "api_key": "sk-test-tinfoil-missing-model-attestations",
                    "provider_settings": {
                        "confidentiality": {
                            "mode": "tinfoil",
                            "model_ids": ["tinfoil/gpt-secure"],
                            "policy": {
                                "repo": "tinfoilsh/confidential-model-router",
                                "expected_release_digest": (
                                    "sha256:ebb921d8572bc8f6b4e04812a7e65bffcf6ac1a2fabe8a55115a13d61e901b85"
                                ),
                                "expected_code_measurement_fingerprint": (
                                    "sha256:f9bdc70e21e119a7446f5d62f1304234a151919b5c5cd4a1b9d511b3a9525a73"
                                ),
                            },
                        }
                    },
                },
            )

        assert response.status_code == 400
        assert (
            "Tinfoil require_model_attestations must be true for selected model routing"
            in response.json()["detail"]
        )

        rows = (
            await integration_session.exec(
                select(UpstreamProviderRow).where(
                    UpstreamProviderRow.api_key
                    == "sk-test-tinfoil-missing-model-attestations"
                )
            )
        ).all()
        assert rows == []
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_ppq_private_non_private_selected_model(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token-ppq-private-non-private-model"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.post(
                "/admin/api/upstream-providers",
                json={
                    "provider_type": "ppq-private",
                    "base_url": "https://api.ppq.ai/private/v1",
                    "api_key": "sk-test-ppq-private-non-private-model",
                    "provider_settings": {
                        "confidentiality": {
                            "mode": "ppq-private-tee",
                            "model_ids": ["tinfoil/gpt-secure"],
                            "policy": {
                                "attestation_bundle_url": (
                                    "https://api.ppq.ai/private/v1/attestation-bundle"
                                ),
                                "repo": "ppq-ai/private-tee",
                                "expected_release_digest": (
                                    "sha256:ebb921d8572bc8f6b4e04812a7e65bffcf6ac1a2fabe8a55115a13d61e901b85"
                                ),
                                "expected_code_measurement_fingerprint": (
                                    "sha256:f9bdc70e21e119a7446f5d62f1304234a151919b5c5cd4a1b9d511b3a9525a73"
                                ),
                                "proxy_binary_digest": (
                                    "sha256:6bd4e7d891d73c4f24db0cd79d8e82cbcb0714b56eddf9a5207845440d8fe4ab"
                                ),
                                "proxy_binary_path": "/opt/ppq/ppq-private-mode-proxy",
                                "require_model_attestations": True,
                                "model_attestation_targets": {
                                    "tinfoil/gpt-secure": {
                                        "host": "gpt-secure.tinfoil.example",
                                        "repo": "tinfoilsh/confidential-gpt-secure",
                                        "expected_release_digest": (
                                            "sha256:2f1edb5242e12ac12be60356f28ccf9d1b5fadc2e5fbb6a4a672bd556b327d76"
                                        ),
                                        "expected_code_measurement_fingerprint": (
                                            "sha256:47ab8df4d5c88cf9482f077481453e114e1a3192c115000106f168b6842aa8b3"
                                        ),
                                    }
                                },
                            },
                        }
                    },
                },
            )

        assert response.status_code == 400
        assert (
            "PPQ private model_ids must start with private/"
            in (response.json()["detail"])
        )

        rows = (
            await integration_session.exec(
                select(UpstreamProviderRow).where(
                    UpstreamProviderRow.api_key
                    == "sk-test-ppq-private-non-private-model"
                )
            )
        ).all()
        assert rows == []
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_runtime_unloadable_confidential_policy_shape(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token-runtime-unloadable-confidential-policy"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.post(
                "/admin/api/upstream-providers",
                json={
                    "provider_type": "tinfoil",
                    "base_url": "https://inference.tinfoil.sh/v1",
                    "api_key": "sk-test-runtime-unloadable-confidential-policy",
                    "provider_settings": {
                        "mode": "tinfoil",
                        "model_ids": ["tinfoil/gpt-secure"],
                        "policy": {
                            "repo": "tinfoilsh/confidential-model-router",
                            "expected_release_digest": "sha256:" + ("a" * 64),
                            "verifier_command": "/opt/routstr/bin/confidential-verifier",
                            "verifier_command_digest": "sha256:" + ("b" * 64),
                        },
                    },
                },
            )

        assert response.status_code == 400
        assert "runtime-loadable" in response.json()["detail"]

        rows = (
            await integration_session.exec(
                select(UpstreamProviderRow).where(
                    UpstreamProviderRow.api_key
                    == "sk-test-runtime-unloadable-confidential-policy"
                )
            )
        ).all()
        assert rows == []
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_confidential_mode_without_verifier_policy(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token-confidential-mode-without-policy"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.post(
                "/admin/api/upstream-providers",
                json={
                    "provider_type": "tinfoil",
                    "base_url": "https://inference.tinfoil.sh/v1",
                    "api_key": "sk-test-confidential-mode-without-policy",
                    "provider_settings": {
                        "confidentiality": {
                            "mode": "tinfoil",
                            "model_ids": ["tinfoil/gpt-secure"],
                        },
                    },
                },
            )

        assert response.status_code == 400
        assert "runtime-loadable" in response.json()["detail"]

        rows = (
            await integration_session.exec(
                select(UpstreamProviderRow).where(
                    UpstreamProviderRow.api_key
                    == "sk-test-confidential-mode-without-policy"
                )
            )
        ).all()
        assert rows == []
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_inline_secret_in_malformed_confidential_policy(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token-provider-settings-malformed-secret"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"
    base_url = "https://secret-policy-list.tinfoil.example/v1"

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.post(
                "/admin/api/upstream-providers",
                json={
                    "provider_type": "tinfoil",
                    "base_url": base_url,
                    "api_key": "sk-test",
                    "provider_settings": {
                        "confidentiality": {
                            "mode": "tinfoil",
                            "model_ids": ["tinfoil/gpt-secure"],
                            "policy": [
                                {
                                    "repo": "tinfoilsh/confidential-gpt-secure",
                                    "api_key": "sk-secret-provider-token",
                                }
                            ],
                        }
                    },
                },
            )

        assert response.status_code == 400
        assert "api_key must not be embedded" in response.json()["detail"]

        rows = (
            await integration_session.exec(
                select(UpstreamProviderRow).where(
                    UpstreamProviderRow.base_url == base_url
                )
            )
        ).all()
        assert rows == []
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_embedded_secret_in_confidential_verifier_command(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token-provider-settings-command-secret"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"
    base_url = "https://secret-command-policy.tinfoil.example/v1"

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.post(
                "/admin/api/upstream-providers",
                json={
                    "provider_type": "tinfoil",
                    "base_url": base_url,
                    "api_key": "sk-test",
                    "provider_settings": {
                        "confidentiality": {
                            "mode": "tinfoil",
                            "model_ids": ["tinfoil/gpt-secure"],
                            "policy": {
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "verifier_command": [
                                    "/opt/routstr/bin/tinfoil-verifier",
                                    "--token=sk-secret-provider-token",
                                ],
                            },
                        }
                    },
                },
            )

        assert response.status_code == 400
        assert "secret-like value must not be embedded" in response.json()["detail"]

        rows = (
            await integration_session.exec(
                select(UpstreamProviderRow).where(
                    UpstreamProviderRow.base_url == base_url
                )
            )
        ).all()
        assert rows == []
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_rejects_inline_secret_confidential_provider_settings_on_update(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token-provider-settings-update-secret"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    provider = UpstreamProviderRow(
        provider_type="tinfoil",
        base_url="https://secret-policy-update.tinfoil.example/v1",
        api_key="sk-upstream-test",
        enabled=True,
        provider_fee=1.01,
        provider_settings=(
            '{"confidentiality":{"mode":"tinfoil",'
            '"model_ids":["tinfoil/gpt-secure"],'
            '"policy":{"repo":"tinfoilsh/confidential-gpt-secure"}}}'
        ),
    )
    integration_session.add(provider)
    await integration_session.commit()
    await integration_session.refresh(provider)
    original_provider_settings = provider.provider_settings

    try:
        with (
            patch("routstr.core.admin.reinitialize_upstreams", new_callable=AsyncMock),
            patch("routstr.core.admin.refresh_model_maps", new_callable=AsyncMock),
        ):
            response = await integration_client.patch(
                f"/admin/api/upstream-providers/{provider.id}",
                json={
                    "provider_settings": {
                        "confidentiality": {
                            "mode": "tinfoil",
                            "model_ids": ["tinfoil/gpt-secure"],
                            "policy": {
                                "repo": "tinfoilsh/confidential-gpt-secure",
                                "api_key": "sk-secret-provider-token",
                            },
                        }
                    }
                },
            )

        assert response.status_code == 400
        assert "api_key must not be embedded" in response.json()["detail"]

        await integration_session.refresh(provider)
        assert provider.provider_settings == original_provider_settings
    finally:
        admin_sessions.pop(admin_token, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_routstr_topup_retries_transient_upstream_failure(
    integration_client: AsyncClient,
    integration_session: Any,
) -> None:
    admin_token = "test-admin-token"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    provider = UpstreamProviderRow(
        provider_type="routstr",
        base_url="https://node.example",
        api_key="sk-upstream-test",
        enabled=True,
        provider_fee=1.01,
    )
    integration_session.add(provider)
    await integration_session.commit()
    await integration_session.refresh(provider)

    class MockResponse:
        def __init__(self, status_code: int, data: dict[str, Any] | None = None):
            self.status_code = status_code
            self._data = data or {}
            self.text = str(self._data)

        def json(self) -> dict[str, Any]:
            return self._data

    class MockAsyncClient:
        def __init__(self) -> None:
            self.calls = 0

        async def __aenter__(self) -> "MockAsyncClient":
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            return None

        async def post(
            self, url: str, json: dict[str, Any], headers: dict[str, str]
        ) -> MockResponse:
            self.calls += 1
            assert url == "https://node.example/v1/balance/lightning/invoice"
            assert json["amount_sats"] == 10
            assert json["purpose"] == "topup"
            assert json["api_key"] == "sk-upstream-test"
            assert headers["Authorization"] == "Bearer sk-upstream-test"

            if self.calls == 1:
                return MockResponse(500, {"detail": "warmup failure"})

            return MockResponse(
                200,
                {
                    "bolt11": "lnbc1testinvoice",
                    "invoice_id": "invoice-123",
                },
            )

    mock_client = MockAsyncClient()

    try:
        with patch("httpx.AsyncClient", return_value=mock_client):
            response = await integration_client.post(
                f"/admin/api/upstream-providers/{provider.id}/topup",
                json={"amount": 10},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["topup_data"]["payment_request"] == "lnbc1testinvoice"
        assert data["topup_data"]["invoice_id"] == "invoice-123"
        assert mock_client.calls == 2
    finally:
        admin_sessions.pop(admin_token, None)
