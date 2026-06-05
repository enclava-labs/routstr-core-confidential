"""
Integration tests for general information endpoints that don't require authentication.
Tests GET /, GET /v1/models, and GET /admin/ endpoints.
"""

import hashlib
import time
from typing import Any

import pytest
from httpx import AsyncClient

from routstr.core.admin import admin_sessions
from routstr.core.settings import settings

from .utils import PerformanceValidator


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_root_endpoint_structure_and_performance(
    integration_client: AsyncClient, db_snapshot: Any
) -> None:
    """Test GET / endpoint response structure and performance requirements"""

    # Capture initial database state
    await db_snapshot.capture()

    # Test performance
    validator = PerformanceValidator()

    # Run multiple requests to get reliable timing
    responses = []
    for i in range(10):
        start = validator.start_timing("root_endpoint")
        response = await integration_client.get("/v1/info")
        duration = validator.end_timing("root_endpoint", start)
        responses.append(response)

        # Each individual request should be fast
        assert duration < 1.0, f"Single request took {duration:.3f}s (too slow)"

    # All requests should succeed
    for response in responses:
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    # Validate performance requirement: 95th percentile < 500ms
    perf_result = validator.validate_response_time(
        "root_endpoint", max_duration=0.5, percentile=0.95
    )
    assert perf_result["valid"], (
        f"Performance requirement failed: 95th percentile was "
        f"{perf_result['percentile_time']:.3f}s (required < 0.5s)"
    )

    # Validate response structure using the last response
    response = responses[-1]
    data = response.json()

    # Required fields in response
    required_fields = [
        "name",
        "description",
        "version",
        "npub",
        "mints",
        "http_url",
        "onion_url",
    ]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"

    # Validate field types
    assert isinstance(data["name"], str)
    assert isinstance(data["description"], str)
    assert isinstance(data["version"], str)
    assert isinstance(data["npub"], str)
    assert isinstance(data["mints"], list)
    assert isinstance(data["http_url"], str)
    assert isinstance(data["onion_url"], str)

    # Ensure models field is not present (removed as per issue #184)
    assert "models" not in data, "Models field should not be present in base URL output"

    # Verify no database state changes
    diff = await db_snapshot.diff()
    assert len(diff["api_keys"]["added"]) == 0
    assert len(diff["api_keys"]["removed"]) == 0
    assert len(diff["api_keys"]["modified"]) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_root_endpoint_environment_variables(
    integration_client: AsyncClient,
    test_mode: str,
) -> None:
    """Test that root endpoint reflects environment variable configuration"""

    response = await integration_client.get("/v1/info")
    assert response.status_code == 200

    data = response.json()

    # Check that environment variables are reflected in response
    # In mock mode, URLs are adjusted to localhost
    if test_mode == "docker":
        assert "http://mint:3338" in data["mints"]
    else:
        assert "http://localhost:3338" in data["mints"]

    # Name should have a default value or be configurable
    assert len(data["name"]) > 0

    # Description should have a default value
    assert len(data["description"]) > 0

    # Version should be set
    assert len(data["version"]) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_info_endpoint_requires_local_tee_proof_for_end_to_end_ready(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_confidentiality_status() -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "end_to_end_ready": True,
            "routable_with_full_attestation": {
                "tinfoil": ["gpt-secure"],
                "ppq-private": [],
                "privatemode": [],
            },
            "routstr_tee": {
                "required": True,
                "ready": True,
                "client_confidentiality": {
                    "mode": "attested-tls-termination",
                    "tls_terminates_in_attested_tee": True,
                    "inbound_ehbp_ohttp_request_decryption": False,
                },
            },
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_confidentiality_status,
    )

    response = await integration_client.get("/v1/info")

    assert response.status_code == 200
    confidentiality = response.json()["confidentiality"]
    assert confidentiality["end_to_end_ready"] is False
    assert confidentiality["routstr_tee"]["ready"] is False
    assert confidentiality["routable_with_full_attestation"] == {
        "tinfoil": [],
        "ppq-private": [],
        "privatemode": [],
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_info_endpoint_requires_local_tee_summary_for_exact_map(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_confidentiality_status() -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "end_to_end_ready": True,
            "routable_with_full_attestation": {
                "tinfoil": ["gpt-secure"],
                "ppq-private": [],
                "privatemode": [],
            },
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_confidentiality_status,
    )

    response = await integration_client.get("/v1/info")

    assert response.status_code == 200
    confidentiality = response.json()["confidentiality"]
    assert confidentiality["end_to_end_ready"] is False
    assert "routstr_tee" not in confidentiality
    assert confidentiality["routable_with_full_attestation"] == {
        "tinfoil": [],
        "ppq-private": [],
        "privatemode": [],
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_info_endpoint_exposes_exact_confidential_routable_model_summary(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_confidentiality_status() -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "end_to_end_ready": True,
            "routable_with_full_attestation": {
                "tinfoil": ["gpt-secure"],
                "ppq-private": [],
                "privatemode": ["privatemode/gpt-oss-120b"],
            },
            "routstr_tee": {
                "required": True,
                "ready": True,
                "attestation_evidence_digest": _digest("tee-evidence"),
                "hpke_key_config_digest": _digest("hpke-key-config"),
                "hpke_public_key_digest": _digest("hpke-public-key"),
                "client_confidentiality": {
                    "mode": "attested-tls-termination",
                    "tls_terminates_in_attested_tee": True,
                    "inbound_ehbp_ohttp_request_decryption": False,
                    "attested_tls_public_key_digest": _digest("public-key"),
                },
                "local_verification": {
                    "verified": True,
                    "verified_at": 1_700_000_000,
                    "expires_at": int(time.time()) + 300,
                    "evidence_digest": _digest("tee-evidence"),
                    "verified_claims_digest": _digest("routstr-tee-claims"),
                    "proof_claims": {
                        "hpke_key_config_digest": _digest("hpke-key-config"),
                        "hpke_public_key_digest": _digest("hpke-public-key"),
                        "public_key_digest": _digest("public-key"),
                    },
                },
            },
            "providers": [
                {
                    "provider_type": "tinfoil",
                    "confidentiality": {
                        "verified": True,
                        "proof_claims": {"raw": "not-for-info-summary"},
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_confidentiality_status,
    )

    response = await integration_client.get("/v1/info")

    assert response.status_code == 200
    confidentiality = response.json()["confidentiality"]
    assert confidentiality == {
        "mode": "required",
        "required": True,
        "end_to_end_ready": True,
        "routable_with_full_attestation": {
            "tinfoil": ["gpt-secure"],
            "ppq-private": [],
            "privatemode": ["privatemode/gpt-oss-120b"],
        },
        "routstr_tee": {
            "required": True,
            "ready": True,
            "client_confidentiality": {
                "mode": "attested-tls-termination",
                "tls_terminates_in_attested_tee": True,
                "inbound_ehbp_ohttp_request_decryption": False,
            },
        },
    }
    assert "providers" not in confidentiality
    assert "not-for-info-summary" not in response.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_info_endpoint_clears_malformed_confidential_routable_model_summary(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_confidentiality_status() -> dict[str, object]:
        return {
            "mode": "required",
            "required": True,
            "end_to_end_ready": True,
            "routable_with_full_attestation": {
                "tinfoil": ["gpt-secure", {"bad": "selector"}],
                "ppq-private": ["private/gpt-oss-120b"],
                "privatemode": "privatemode/gpt-oss-120b",
            },
        }

    monkeypatch.setattr(
        "routstr.proxy.get_confidentiality_status",
        fake_confidentiality_status,
    )

    response = await integration_client.get("/v1/info")

    assert response.status_code == 200
    confidentiality = response.json()["confidentiality"]
    assert confidentiality["end_to_end_ready"] is False
    assert confidentiality["routable_with_full_attestation"] == {
        "tinfoil": [],
        "ppq-private": [],
        "privatemode": [],
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_models_endpoint_structure_and_performance(
    integration_client: AsyncClient, db_snapshot: Any
) -> None:
    """Test GET /v1/models endpoint with OpenAI-compatible structure"""

    # Capture initial database state
    await db_snapshot.capture()

    # Test performance
    validator = PerformanceValidator()

    # Run multiple requests for performance measurement
    responses = []
    for i in range(10):
        start = validator.start_timing("models_endpoint")
        response = await integration_client.get("/v1/models")
        duration = validator.end_timing("models_endpoint", start)
        responses.append(response)

        # Each request should be reasonably fast
        assert duration < 1.0, f"Models request took {duration:.3f}s (too slow)"

    # All requests should succeed
    for response in responses:
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    # Validate performance requirement
    perf_result = validator.validate_response_time(
        "models_endpoint", max_duration=0.5, percentile=0.95
    )
    assert perf_result["valid"], (
        f"Models endpoint performance failed: 95th percentile was "
        f"{perf_result['percentile_time']:.3f}s (required < 0.5s)"
    )

    # Validate response structure
    response = responses[-1]
    data = response.json()

    # Should have OpenAI-compatible structure
    assert "data" in data
    assert isinstance(data["data"], list)

    # Validate each model structure
    for model in data["data"]:
        # Required OpenAI model fields
        required_fields = ["id", "name", "created"]
        for field in required_fields:
            assert field in model, f"Model missing required field: {field}"

        # Validate field types
        assert isinstance(model["id"], str)
        assert isinstance(model["name"], str)
        assert isinstance(model["created"], (int, float))

        # Check for additional expected fields
        optional_fields = [
            "description",
            "context_length",
            "architecture",
            "pricing",
            "sats_pricing",
        ]
        for field in optional_fields:
            if field in model:
                if field == "pricing" or field == "sats_pricing":
                    # Pricing fields can be dict or None
                    assert isinstance(model[field], (dict, type(None)))
                elif field == "context_length":
                    assert isinstance(model[field], (int, type(None)))
                elif field == "architecture":
                    assert isinstance(model[field], (dict, type(None)))

    # Verify no database state changes
    diff = await db_snapshot.diff()
    assert len(diff["api_keys"]["added"]) == 0
    assert len(diff["api_keys"]["removed"]) == 0
    assert len(diff["api_keys"]["modified"]) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_models_endpoint_pricing_structure(
    integration_client: AsyncClient,
) -> None:
    """Test that models endpoint includes proper pricing information"""

    response = await integration_client.get("/v1/models")
    assert response.status_code == 200

    data = response.json()

    # If models exist, validate pricing structure
    for model in data["data"]:
        if "pricing" in model and model["pricing"]:
            pricing = model["pricing"]

            # Common pricing fields
            expected_pricing_fields = ["prompt", "completion", "request"]
            for field in expected_pricing_fields:
                if field in pricing:
                    # Should be numeric string or number
                    assert isinstance(pricing[field], (str, int, float))

        if "sats_pricing" in model and model["sats_pricing"]:
            sats_pricing = model["sats_pricing"]

            # Sats pricing should be numeric
            for key, value in sats_pricing.items():
                if value is not None:
                    assert isinstance(value, (int, float, str))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_models_endpoint_accept_headers(integration_client: AsyncClient) -> None:
    """Test models endpoint with different Accept headers"""

    # Test JSON accept header (should work)
    response = await integration_client.get(
        "/v1/models", headers={"Accept": "application/json"}
    )
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]
    data = response.json()
    assert "data" in data

    # Test HTML accept header (should still return JSON)
    response = await integration_client.get(
        "/v1/models", headers={"Accept": "text/html"}
    )
    assert response.status_code == 200
    # Endpoint always returns JSON regardless of Accept header
    assert "application/json" in response.headers["content-type"]

    # Test wildcard accept header
    response = await integration_client.get("/v1/models", headers={"Accept": "*/*"})
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_endpoint_unauthenticated(
    integration_client: AsyncClient, db_snapshot: Any
) -> None:
    """Test GET /admin/ endpoint redirects to /"""
    await db_snapshot.capture()

    response = await integration_client.get("/admin/api/settings")

    assert response.status_code == 403

    diff = await db_snapshot.diff()
    assert len(diff["api_keys"]["added"]) == 0
    assert len(diff["api_keys"]["removed"]) == 0
    assert len(diff["api_keys"]["modified"]) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_settings_redacts_local_tee_verifier_material(
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admin_token = "admin-tee-settings-redaction"
    admin_sessions[admin_token] = int(time.time()) + 3600
    integration_client.headers["Authorization"] = f"Bearer {admin_token}"

    verifier_policy = '{"measurement":"secret-local-policy"}'
    verifier_command = "/opt/routstr/bin/verify-tee --token sk-local-secret"
    attestation_command = "/opt/routstr/bin/attest-tee --nonce local-secret"
    hpke_key_config = "local-hpke-key-config"
    attestation_public_key = "local-attestation-public-key"
    monkeypatch.setattr(settings, "routstr_tee_verifier_policy_json", verifier_policy)
    monkeypatch.setattr(settings, "routstr_tee_verifier_command", verifier_command)
    monkeypatch.setattr(
        settings, "routstr_tee_attestation_command", attestation_command
    )
    monkeypatch.setattr(
        settings, "routstr_attestation_hpke_key_config_b64", hpke_key_config
    )
    monkeypatch.setattr(
        settings, "routstr_attestation_public_key", attestation_public_key
    )

    try:
        response = await integration_client.get("/admin/api/settings")
        assert response.status_code == 200
        data = response.json()
        assert data["routstr_tee_verifier_policy_json"] == "[REDACTED]"
        assert data["routstr_tee_verifier_command"] == "[REDACTED]"
        assert data["routstr_tee_attestation_command"] == "[REDACTED]"
        assert data["routstr_attestation_hpke_key_config_b64"] == "[REDACTED]"
        assert data["routstr_attestation_public_key"] == "[REDACTED]"
        assert verifier_policy not in response.text
        assert verifier_command not in response.text
        assert attestation_command not in response.text
        assert hpke_key_config not in response.text
        assert attestation_public_key not in response.text

    finally:
        admin_sessions.pop(admin_token, None)
        integration_client.headers.pop("Authorization", None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_all_info_endpoints_no_database_changes(
    integration_client: AsyncClient, db_snapshot: Any
) -> None:
    """Verify that all info endpoints don't modify database state"""

    # Capture initial state
    initial_state = await db_snapshot.capture()

    # Make requests to all info endpoints
    endpoints = ["/v1/info", "/v1/models"]

    for endpoint in endpoints:
        response = await integration_client.get(endpoint)
        assert response.status_code == 200

        # Check no database changes after each request
        diff = await db_snapshot.diff()
        assert len(diff["api_keys"]["added"]) == 0, (
            f"Endpoint {endpoint} added API keys"
        )
        assert len(diff["api_keys"]["removed"]) == 0, (
            f"Endpoint {endpoint} removed API keys"
        )
        assert len(diff["api_keys"]["modified"]) == 0, (
            f"Endpoint {endpoint} modified API keys"
        )

    # Final verification - database state should be identical
    final_state = await db_snapshot.capture()
    assert final_state == initial_state


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_info_endpoint_requests(
    integration_client: AsyncClient,
) -> None:
    """Test concurrent requests to info endpoints don't cause issues"""

    from .utils import ConcurrencyTester

    # Create concurrent requests to all endpoints
    requests = []
    for endpoint in ["/", "/v1/models"]:
        for _ in range(5):  # 5 requests per endpoint
            # Use /v1/info instead of / for JSON API
            url = "/v1/info" if endpoint == "/" else endpoint
            requests.append({"method": "GET", "url": url})

    # Execute concurrently
    tester = ConcurrencyTester()
    responses = await tester.run_concurrent_requests(
        integration_client, requests, max_concurrent=10
    )

    # All should succeed
    assert len(responses) == 10  # 2 endpoints × 5 requests each
    for response in responses:
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_info_endpoints_response_consistency(
    integration_client: AsyncClient,
) -> None:
    """Test that info endpoints return consistent responses across multiple calls"""

    # Test root endpoint consistency
    responses = []
    for _ in range(5):
        response = await integration_client.get("/v1/info")
        assert response.status_code == 200
        responses.append(response.json())

    # All responses should be identical (assuming no background updates)
    first_response = responses[0]
    for response in responses[1:]:
        # Core fields should remain consistent
        for field in ["name", "description", "version"]:
            assert response[field] == first_response[field]  # type: ignore[index]

    # Test models endpoint consistency
    model_responses = []
    for _ in range(5):
        response = await integration_client.get("/v1/models")
        assert response.status_code == 200
        model_responses.append(response.json())

    # Model structure should be consistent
    first_models = model_responses[0]["data"]
    for response in model_responses[1:]:
        models = response["data"]  # type: ignore[index]
        assert len(models) == len(first_models)

        # Model IDs should be the same
        first_ids = {m["id"] for m in first_models}
        response_ids = {m["id"] for m in models}
        assert first_ids == response_ids
