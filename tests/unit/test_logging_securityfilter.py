"""Unit tests for the logging SecurityFilter.

This module tests that the SecurityFilter correctly identifies and redacts
sensitive information from log messages without causing false positives.

"""

import json
import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from routstr.core.logging import (
    SecurityFilter,
    get_logger,
    redact_sensitive_text,
    setup_logging,
)


@pytest.fixture
def security_filter() -> SecurityFilter:
    """Provide an instance of the SecurityFilter for testing."""
    return SecurityFilter()


@pytest.fixture
def filter_message(security_filter: SecurityFilter) -> Callable[[str], str]:
    """A helper fixture to apply the filter to a message string."""

    def _filter(msg: str) -> str:
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        security_filter.filter(record)
        return record.getMessage()

    return _filter


def test_redacts_unquoted_key_value_pairs(filter_message: Callable[[str], str]) -> None:
    """Test that an unquoted key-value pair is correctly redacted."""
    original = "Processing request with api_key=sk-12345abcdef"
    expected = "Processing request with api_key: [REDACTED]"
    assert filter_message(original) == expected


def test_redacts_quoted_key_value_pairs(filter_message: Callable[[str], str]) -> None:
    """Test that a quoted token is correctly redacted."""
    original = 'User authenticated with token="cashuA123abc"'
    expected = "User authenticated with token: [REDACTED]"
    assert filter_message(original) == expected


def test_redacts_bearer_token(filter_message: Callable[[str], str]) -> None:
    """Test that a Bearer token of sufficient length is redacted."""
    original = "Authorization: Bearer abc1234567890xyzabcdefg"
    expected = "Authorization: [REDACTED]"
    assert filter_message(original) == expected


def test_redacts_cashu_token(filter_message: Callable[[str], str]) -> None:
    """Test that a Cashu token is redacted."""
    original = "Received cashuTOKENeyJ0b2tlbiI6W3siaWQiOiI"
    expected = "Received [REDACTED]"
    assert filter_message(original) == expected


def test_redacts_nsec_key(filter_message: Callable[[str], str]) -> None:
    """Test that a full-length Nostr private key is redacted."""
    original = "Private key is nsec1a8d9f8s7d9f8a7s6d5f4a3s2d1f9a8s7d6f5a4s3d2f1a9s8d7f6a5s4d3f"
    expected = "Private key is [REDACTED]"
    assert filter_message(original) == expected


def test_redacts_standalone_provider_api_key(
    filter_message: Callable[[str], str],
) -> None:
    """Test that standalone provider-style API keys are redacted."""
    original = "Verifier failed with upstream token sk-live-secretvalue"
    expected = "Verifier failed with upstream token [REDACTED]"
    assert filter_message(original) == expected


def test_redact_sensitive_text_redacts_standalone_provider_api_key() -> None:
    original = "Verifier failed with upstream token sk-live-secretvalue"
    redacted = redact_sensitive_text(original)

    assert redacted == "Verifier failed with upstream token [REDACTED]"
    assert "sk-live-secretvalue" not in redacted


def test_redacts_json_style_secret_fields(
    filter_message: Callable[[str], str],
) -> None:
    """Test that quoted JSON-style secret keys are redacted."""
    original = (
        'Verifier failed with {"api_key":"sk-live-secretvalue",'
        '"raw_prompt":"SECRET PROMPT"}'
    )
    redacted = filter_message(original)

    assert "api_key: [REDACTED]" in redacted
    assert "raw_prompt: [REDACTED]" in redacted
    assert "sk-live-secretvalue" not in redacted
    assert "SECRET PROMPT" not in redacted


def test_redact_sensitive_text_redacts_json_style_secret_fields() -> None:
    original = (
        'Verifier failed with {"api_key":"sk-live-secretvalue",'
        '"raw_prompt":"SECRET PROMPT"}'
    )
    redacted = redact_sensitive_text(original)

    assert "api_key: [REDACTED]" in redacted
    assert "raw_prompt: [REDACTED]" in redacted
    assert "sk-live-secretvalue" not in redacted
    assert "SECRET PROMPT" not in redacted


def test_security_filter_redacts_url_userinfo(
    filter_message: Callable[[str], str],
) -> None:
    original = "Verifier failed at https://user:pass@verified.example/v1"
    redacted = filter_message(original)

    assert redacted == "Verifier failed at https://verified.example/v1"
    assert "user:pass" not in redacted


def test_security_filter_redacts_structured_extra_fields(
    security_filter: SecurityFilter,
) -> None:
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Provider refresh failed",
        args=(),
        exc_info=None,
    )
    record.error = (
        'verifier failed api_key=SECRET_PROVIDER_KEY '
        '"raw_prompt":"SECRET PROMPT" https://user:pass@verified.example/v1'
    )
    record.api_key = "SECRET_PROVIDER_KEY"
    record.context = {
        "Authorization": "Bearer sk-live-secretvalue",
        "raw_prompt": "SECRET PROMPT",
        "url": "https://user:pass@verified.example/v1",
    }

    security_filter.filter(record)

    assert "SECRET_PROVIDER_KEY" not in record.error
    assert "SECRET PROMPT" not in record.error
    assert "user:pass" not in record.error
    assert "api_key: [REDACTED]" in record.error
    assert "raw_prompt: [REDACTED]" in record.error
    assert "https://verified.example/v1" in record.error
    assert record.api_key == "[REDACTED]"
    assert record.context["Authorization"] == "[REDACTED]"
    assert record.context["raw_prompt"] == "[REDACTED]"
    assert record.context["url"] == "https://verified.example/v1"


def test_redacts_camel_case_secret_keys(
    security_filter: SecurityFilter,
) -> None:
    original = (
        'Verifier failed with {"apiKey":"SECRET_API_KEY",'
        '"clientSecret":"SECRET CLIENT","rawPrompt":"SECRET PROMPT"}'
    )
    redacted = redact_sensitive_text(original)

    assert "apikey: [REDACTED]" in redacted
    assert "clientsecret: [REDACTED]" in redacted
    assert "rawprompt: [REDACTED]" in redacted
    assert "SECRET_API_KEY" not in redacted
    assert "SECRET CLIENT" not in redacted
    assert "SECRET PROMPT" not in redacted

    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Provider refresh failed",
        args=(),
        exc_info=None,
    )
    record.context = {
        "apiKey": "SECRET_API_KEY",
        "clientSecret": "SECRET CLIENT",
        "rawPrompt": "SECRET PROMPT",
    }

    security_filter.filter(record)

    assert record.context["apiKey"] == "[REDACTED]"
    assert record.context["clientSecret"] == "[REDACTED]"
    assert record.context["rawPrompt"] == "[REDACTED]"


def test_ignores_non_sensitive_message(filter_message: Callable[[str], str]) -> None:
    """Test that a message with no sensitive data is left untouched."""
    original = "No token pricing configured, using base cost"
    expected = "No token pricing configured, using base cost"
    assert filter_message(original) == expected


def test_setup_logging_uses_configured_log_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    log_dir = tmp_path / "state" / "logs"
    monkeypatch.chdir(app_dir)
    monkeypatch.setenv("ROUTSTR_LOG_DIR", str(log_dir))

    setup_logging()
    logger = get_logger("routstr")
    logger.info("configured log dir test")
    for handler in logger.handlers:
        handler.flush()

    assert list(log_dir.glob("app_*.log"))
    assert not (app_dir / "logs").exists()


def test_log_manager_reads_configured_log_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    log_dir = tmp_path / "state" / "logs"
    log_dir.mkdir(parents=True)
    monkeypatch.chdir(app_dir)
    monkeypatch.setenv("ROUTSTR_LOG_DIR", str(log_dir))

    request_id = "configured-log-dir-request"
    (log_dir / "app_2026-06-06.log").write_text(
        json.dumps(
            {
                "asctime": "2026-06-06 12:00:00",
                "levelname": "ERROR",
                "message": "configured log dir lookup",
                "request_id": request_id,
            }
        )
        + "\n"
    )

    from routstr.core.log_manager import LogManager

    assert [entry["request_id"] for entry in LogManager().search_logs()] == [
        request_id
    ]
    assert not (app_dir / "logs").exists()


def test_multiple_secrets_in_one_message(filter_message: Callable[[str], str]) -> None:
    """Test that multiple different secrets in one message are all redacted."""
    original = 'Auth with Bearer abcdefghijklmnopqrstuvwxyz and api_key="sk-12345"'
    expected = "Auth with [REDACTED] and api_key: [REDACTED]"
    assert filter_message(original) == expected


def test_redacts_key_with_no_value(filter_message: Callable[[str], str]) -> None:
    """Test that a key with no value is not redacted."""
    original = "Request contains api_key and secret."
    expected = "Request contains api_key and secret."
    assert filter_message(original) == expected


def test_redacts_key_value_with_spaces(filter_message: Callable[[str], str]) -> None:
    """Test that key-value pairs with extra spaces are correctly redacted."""
    original = "Auth info:   api_key   =   'sk-12345'"
    expected = "Auth info:   api_key: [REDACTED]"
    assert filter_message(original) == expected


def test_is_case_insensitive_for_keys(filter_message: Callable[[str], str]) -> None:
    """Test that key matching is case-insensitive."""
    original = "TOKEN=sk-abcdef12345"
    expected = "token: [REDACTED]"
    assert filter_message(original) == expected


def test_is_case_insensitive_for_standalone(
    filter_message: Callable[[str], str],
) -> None:
    """Test that standalone matching is case-insensitive."""
    original = "Using NSEC1a8d9f8s7d9f8a7s6d5f4a3s2d1f9a8s7d6f5a4s3d2f1a9s8d7f6a5s4d3f and CaShuA123abc"
    expected = "Using [REDACTED] and [REDACTED]"
    assert filter_message(original) == expected
