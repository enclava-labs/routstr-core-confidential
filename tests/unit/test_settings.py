import os
from pathlib import Path

import pytest
from pydantic.v1 import ValidationError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, select, text
from sqlmodel.ext.asyncio.session import AsyncSession

from routstr.core.db import UpstreamProviderRow
from routstr.core.settings import Settings, SettingsService
from routstr.upstream.helpers import _seed_providers_from_settings

ROOT = Path(__file__).resolve().parents[2]
CONFIDENTIAL_ENV_VARS = {
    "CONFIDENTIAL_ROUTING_MODE",
    "ROUTSTR_TEE_CLIENT_CONFIDENTIALITY_BOUNDARY",
    "ROUTSTR_TEE_ATTESTATION_DOCUMENT_FORMAT",
    "ROUTSTR_TEE_VERIFIER_COMMAND",
    "ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST",
    "ROUTSTR_TEE_ATTESTATION_COMMAND",
    "ROUTSTR_TEE_ATTESTATION_COMMAND_DIGEST",
    "ROUTSTR_TEE_ATTESTATION_REQUIRED",
    "ROUTSTR_TEE_PUBLIC_KEY_PATH",
    "ROUTSTR_TEE_ATTESTED_TLS_PUBLIC_KEY_DIGEST",
    "ROUTSTR_TEE_HPKE_KEY_CONFIG_PATH",
    "ROUTSTR_TEE_VERIFIER_POLICY_JSON",
}


def test_env_example_surfaces_confidential_routing_deployment_knobs() -> None:
    contents = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "examples/confidential-routing/" in contents
    for env_var in CONFIDENTIAL_ENV_VARS:
        assert env_var in contents


def test_confidential_routing_required_defaults_to_provider_attestation_only(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ROUTSTR_TEE_ATTESTATION_REQUIRED", raising=False)

    settings = Settings()

    assert settings.routstr_tee_attestation_required is False


@pytest.mark.asyncio
async def test_settings_seed_from_env_and_persist() -> None:
    os.environ["UPSTREAM_BASE_URL"] = "https://api.test/v1"
    os.environ.pop("ONION_URL", None)
    os.environ.pop("ENABLE_ANALYTICS_SHARING", None)

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        settings = await SettingsService.initialize(session)

        assert settings.upstream_base_url == "https://api.test/v1"
        # ONION_URL may be empty if not discoverable
        assert isinstance(settings.onion_url, str)
        assert settings.enable_analytics_sharing is True


@pytest.mark.asyncio
async def test_settings_db_precedence_over_env() -> None:
    os.environ["UPSTREAM_BASE_URL"] = "https://api.env/v1"
    os.environ["ENABLE_ANALYTICS_SHARING"] = "true"

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        _ = await SettingsService.initialize(session)
        updated = await SettingsService.update(
            {"name": "DBName", "enable_analytics_sharing": False}, session
        )
        assert updated.name == "DBName"
        assert updated.enable_analytics_sharing is False

        # Change env and re-initialize; DB should still win
        os.environ["NAME"] = "EnvName"
        os.environ["ENABLE_ANALYTICS_SHARING"] = "true"
        again = await SettingsService.initialize(session)
        assert again.name == "DBName"
        assert again.enable_analytics_sharing is False


@pytest.mark.asyncio
async def test_confidential_routing_deployment_controls_are_env_authoritative(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CONFIDENTIAL_ROUTING_MODE", "required")
    monkeypatch.setenv(
        "ROUTSTR_TEE_CLIENT_CONFIDENTIALITY_BOUNDARY",
        "attested-tls-termination",
    )
    monkeypatch.setenv("ROUTSTR_TEE_VERIFIER_COMMAND", "/opt/routstr/verifier")
    monkeypatch.setenv(
        "ROUTSTR_TEE_VERIFIER_COMMAND_DIGEST",
        "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    )
    monkeypatch.setenv(
        "ROUTSTR_TEE_ATTESTED_TLS_PUBLIC_KEY_DIGEST",
        "sha256:fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
    )

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await SettingsService.initialize(session)

        await session.exec(  # type: ignore
            text(
                "UPDATE settings SET data = :data WHERE id = 1"
            ).bindparams(
                data=(
                    '{"name":"DBName",'
                    '"confidential_routing_mode":"disabled",'
                    '"routstr_tee_client_confidentiality_boundary":"plaintext-proxy",'
                    '"routstr_tee_attested_tls_public_key_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
                    '"routstr_tee_verifier_command":"/tmp/unpinned-verifier",'
                    '"routstr_tee_verifier_command_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}'
                )
            )
        )
        await session.commit()

        reloaded = await SettingsService.initialize(session)

        assert reloaded.name == "DBName"
        assert reloaded.confidential_routing_mode == "required"
        assert (
            reloaded.routstr_tee_client_confidentiality_boundary
            == "attested-tls-termination"
        )
        assert reloaded.routstr_tee_verifier_command == "/opt/routstr/verifier"
        assert (
            reloaded.routstr_tee_verifier_command_digest
            == "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        )
        assert (
            reloaded.routstr_tee_attested_tls_public_key_digest
            == "sha256:fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
        )

        updated = await SettingsService.update(
            {
                "confidential_routing_mode": "disabled",
                "routstr_tee_client_confidentiality_boundary": "plaintext-proxy",
                "routstr_tee_attested_tls_public_key_digest": (
                    "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                ),
                "routstr_tee_verifier_command": "/tmp/other-verifier",
            },
            session,
        )

        assert updated.confidential_routing_mode == "required"
        assert (
            updated.routstr_tee_client_confidentiality_boundary
            == "attested-tls-termination"
        )
        assert updated.routstr_tee_verifier_command == "/opt/routstr/verifier"
        assert (
            updated.routstr_tee_attested_tls_public_key_digest
            == "sha256:fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
        )

        await session.exec(  # type: ignore
            text(
                "UPDATE settings SET data = :data WHERE id = 1"
            ).bindparams(
                data=(
                    '{"confidential_routing_mode":"disabled",'
                    '"routstr_tee_client_confidentiality_boundary":"plaintext-proxy"}'
                )
            )
        )
        await session.commit()

        loaded = await SettingsService.reload_from_db(session)

        assert loaded.confidential_routing_mode == "required"
        assert (
            loaded.routstr_tee_client_confidentiality_boundary
            == "attested-tls-termination"
        )


def test_payout_settings_have_sensible_defaults() -> None:
    s = Settings()
    assert s.min_payout_sat == 210
    assert s.payout_interval_seconds == 900


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("min_payout_sat", 0),
        ("min_payout_sat", -1),
        ("payout_interval_seconds", 0),
        ("payout_interval_seconds", -10),
    ],
)
def test_payout_settings_reject_invalid_values(field: str, bad_value: int) -> None:
    kwargs: dict[str, object] = {field: bad_value}
    with pytest.raises(ValidationError):
        Settings(**kwargs)  # type: ignore[arg-type]


def test_payout_settings_accept_custom_positive_values() -> None:
    s = Settings(min_payout_sat=500, payout_interval_seconds=60)
    assert s.min_payout_sat == 500
    assert s.payout_interval_seconds == 60


def test_confidential_routing_mode_rejects_unknown_values() -> None:
    with pytest.raises(ValidationError):
        Settings(confidential_routing_mode="requiredd")


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), 0.0, -1.0])
def test_upstream_provider_fee_rejects_invalid_values(bad_value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(upstream_provider_fee=bad_value)


@pytest.mark.asyncio
async def test_seed_providers_from_settings_preserves_configured_provider_fee(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        await _seed_providers_from_settings(
            session,
            Settings(upstream_provider_fee=1.23),
        )
        await session.commit()

        result = await session.exec(
            select(UpstreamProviderRow).where(
                UpstreamProviderRow.provider_type == "openai"
            )
        )
        provider = result.one()

    assert provider.provider_fee == 1.23


@pytest.mark.asyncio
async def test_seed_providers_from_settings_adds_confidential_provider_env_vars(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TINFOIL_API_KEY", "sk-test-tinfoil")
    monkeypatch.setenv("PPQ_API_KEY", "sk-test-ppq")
    monkeypatch.setenv("PRIVATEMODE_API_KEY", "sk-test-privatemode")
    monkeypatch.setenv("PRIVATEMODE_PROXY_URL", "http://127.0.0.1:18080/v1")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        await _seed_providers_from_settings(
            session,
            Settings(upstream_provider_fee=1.23),
        )
        await session.commit()

        result = await session.exec(select(UpstreamProviderRow))
        providers = {provider.provider_type: provider for provider in result.all()}

    assert providers["tinfoil"].base_url == "https://inference.tinfoil.sh/v1"
    assert providers["tinfoil"].api_key == "sk-test-tinfoil"
    assert providers["tinfoil"].provider_fee == 1.23
    assert providers["ppq-private"].base_url == "https://api.ppq.ai/private/v1"
    assert providers["ppq-private"].api_key == "sk-test-ppq"
    assert providers["ppq-private"].provider_fee == 1.23
    assert providers["privatemode"].base_url == "http://127.0.0.1:18080/v1"
    assert providers["privatemode"].api_key == "sk-test-privatemode"
    assert providers["privatemode"].provider_fee == 1.23


@pytest.mark.asyncio
async def test_payout_settings_persist_via_settings_service() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await SettingsService.initialize(session)
        updated = await SettingsService.update(
            {"min_payout_sat": 1000, "payout_interval_seconds": 300}, session
        )
        assert updated.min_payout_sat == 1000
        assert updated.payout_interval_seconds == 300


@pytest.mark.asyncio
async def test_payout_settings_update_rejects_invalid() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await SettingsService.initialize(session)
        with pytest.raises(ValidationError):
            await SettingsService.update({"min_payout_sat": 0}, session)


@pytest.mark.asyncio
async def test_settings_initialize_discards_unknown_keys() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        _ = await SettingsService.initialize(session)

        # Simulate older persisted key name and an unknown key.
        await session.exec(  # type: ignore
            text(
                "UPDATE settings SET data = :data WHERE id = 1"
            ).bindparams(
                data='{"name":"LegacyNode","nostr_analytics_enabled":false,"unknown_key":123}'
            )
        )
        await session.commit()

        reloaded = await SettingsService.initialize(session)
        assert reloaded.name == "LegacyNode"
        assert reloaded.enable_analytics_sharing is True

        row = await session.exec(text("SELECT data FROM settings WHERE id = 1"))  # type: ignore
        stored_data = row.first()[0]
        assert '"enable_analytics_sharing": true' in stored_data
        assert "nostr_analytics_enabled" not in stored_data
        assert "unknown_key" not in stored_data
