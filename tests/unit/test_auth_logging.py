from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from routstr.auth import validate_bearer_key
from routstr.payment.helpers import check_token_balance
from routstr.proxy import get_bearer_token_key


@pytest.mark.asyncio
async def test_proxy_bearer_validation_failure_logs_redacted_diagnostics() -> None:
    bearer_key = "sk-live-secretvalue-abcdefghi"
    secret_error = (
        'validation failed {"raw_prompt":"SECRET_PROMPT"} '
        "api_key=SECRET_PROVIDER_KEY sk-live-secretvalue-abcdefghi"
    )
    mock_validate = AsyncMock(side_effect=RuntimeError(secret_error))

    with (
        patch("routstr.proxy.validate_bearer_key", mock_validate),
        patch("routstr.proxy.logger.error") as log_error,
    ):
        with pytest.raises(RuntimeError):
            await get_bearer_token_key(
                headers={},
                path="v1/chat/completions",
                session=MagicMock(),
                auth=f"Bearer {bearer_key}",
                min_cost=1,
                model_id="secure-model",
            )

    log_error.assert_called_once()
    args, kwargs = log_error.call_args
    serialized = f"{args} {kwargs}"
    assert "api_key: [REDACTED]" in serialized
    assert "raw_prompt: [REDACTED]" in serialized
    assert "SECRET_PROVIDER_KEY" not in serialized
    assert "SECRET_PROMPT" not in serialized
    assert "sk-live-secretvalue" not in serialized
    assert kwargs["extra"]["bearer_key_fingerprint"].startswith("sha256:")
    assert "bearer_key_preview" not in kwargs["extra"]


@pytest.mark.asyncio
async def test_validate_bearer_key_logs_fingerprint_not_api_key_preview() -> None:
    bearer_key = "sk-live-secretvalue-abcdefghi"
    session = MagicMock()
    session.get = AsyncMock(return_value=None)

    with (
        patch("routstr.auth.logger.warning") as log_warning,
        patch("routstr.auth.logger.error") as log_error,
    ):
        with pytest.raises(Exception):
            await validate_bearer_key(bearer_key, session)

    serialized = f"{log_warning.call_args} {log_error.call_args}"
    assert "sk-live-secretvalue" not in serialized
    assert "key_preview" not in serialized
    assert log_warning.call_args.kwargs["extra"]["key_fingerprint"].startswith(
        "sha256:"
    )
    assert log_error.call_args.kwargs["extra"]["key_fingerprint"].startswith("sha256:")


@pytest.mark.asyncio
async def test_cashu_redemption_failure_logs_and_returns_redacted_error() -> None:
    cashu_token = "cashuAsecretvalue-abcdefghi"
    secret_error = (
        'cashu redemption failed {"raw_prompt":"SECRET_PROMPT"} '
        "api_key=SECRET_PROVIDER_KEY "
        "https://operator:secret-token@example.test/redeem"
    )

    with (
        patch(
            "routstr.auth.deserialize_token_from_string",
            side_effect=RuntimeError(secret_error),
        ),
        patch("routstr.auth.logger.error") as log_error,
    ):
        with pytest.raises(Exception) as exc_info:
            await validate_bearer_key(cashu_token, MagicMock())

    serialized = f"{log_error.call_args} {exc_info.value}"
    assert "api_key: [REDACTED]" in serialized
    assert "raw_prompt: [REDACTED]" in serialized
    assert "SECRET_PROVIDER_KEY" not in serialized
    assert "SECRET_PROMPT" not in serialized
    assert "cashuAsecretvalue" not in serialized
    assert "secret-token" not in serialized
    assert "https://example.test/redeem" in serialized
    assert log_error.call_args.kwargs["extra"]["token_fingerprint"].startswith(
        "sha256:"
    )
    assert "token_preview" not in log_error.call_args.kwargs["extra"]


def test_preflight_payment_logs_fingerprints_not_token_previews() -> None:
    cashu_token = "sk-live-secretvalue-abcdefghi"

    with patch("routstr.payment.helpers.logger.debug") as log_debug:
        check_token_balance({"x-cashu": cashu_token}, {"model": "secure"}, 1)

    serialized = f"{log_debug.call_args}"
    assert "sk-live-secretvalue" not in serialized
    assert "token_preview" not in serialized
    assert log_debug.call_args.kwargs["extra"]["token_fingerprint"].startswith(
        "sha256:"
    )


def test_preflight_authorization_logs_fingerprint_not_header_preview() -> None:
    auth_header = "Bearer sk-live-secretvalue-abcdefghi"

    with patch("routstr.payment.helpers.logger.debug") as log_debug:
        check_token_balance({"authorization": auth_header}, {"model": "secure"}, 1)

    serialized = f"{log_debug.call_args}"
    assert "sk-live-secretvalue" not in serialized
    assert "auth_preview" not in serialized
    assert log_debug.call_args.kwargs["extra"]["auth_fingerprint"].startswith(
        "sha256:"
    )
