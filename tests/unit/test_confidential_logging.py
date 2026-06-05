from unittest.mock import patch

import pytest
from fastapi import HTTPException

from routstr.core.settings import settings
from routstr.proxy import parse_request_body_json


def test_confidential_required_invalid_json_redacts_body_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "confidential_routing_mode", "required")
    body = b'{"model":"secure","messages":[{"role":"user","content":"SECRET_PROMPT"}]'

    with patch("routstr.proxy.logger.error") as log_error:
        with pytest.raises(HTTPException):
            parse_request_body_json(body, "v1/chat/completions")

    log_error.assert_called_once()
    _, kwargs = log_error.call_args
    assert "SECRET_PROMPT" not in str(kwargs)
    assert kwargs["extra"]["body_preview"] is None
    assert kwargs["extra"]["confidential"] is True


def test_request_model_selector_must_be_string() -> None:
    body = b'{"model":{"id":"secure"},"messages":[]}'

    with pytest.raises(HTTPException) as exc_info:
        parse_request_body_json(body, "v1/chat/completions")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "error": {"type": "invalid_request_error", "code": "invalid_model"}
    }


def test_request_body_json_rejects_non_standard_constants() -> None:
    body = b'{"model":"secure","messages":[],"temperature":NaN}'

    with pytest.raises(HTTPException) as exc_info:
        parse_request_body_json(body, "v1/chat/completions")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "error": {"type": "invalid_request_error", "code": "invalid_json"}
    }


@pytest.mark.parametrize("value", ["true", "0", "-1"])
def test_request_max_tokens_must_be_positive_integer(value: str) -> None:
    body = f'{{"model":"secure","messages":[],"max_tokens":{value}}}'.encode()

    with pytest.raises(HTTPException) as exc_info:
        parse_request_body_json(body, "v1/chat/completions")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "error": {
            "type": "invalid_request_error",
            "code": "invalid_max_tokens",
        }
    }


@pytest.mark.parametrize("value", ["true", "0", "-1", "2"])
def test_request_n_must_be_single_completion(value: str) -> None:
    body = f'{{"model":"secure","messages":[],"n":{value}}}'.encode()

    with pytest.raises(HTTPException) as exc_info:
        parse_request_body_json(body, "v1/chat/completions")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "error": {
            "type": "invalid_request_error",
            "code": "invalid_n",
        }
    }


def test_responses_input_model_selector_must_be_string() -> None:
    body = b'{"input":{"model":123,"content":"test"}}'

    with pytest.raises(HTTPException) as exc_info:
        parse_request_body_json(body, "v1/responses")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "error": {"type": "invalid_request_error", "code": "invalid_model"}
    }


def test_responses_model_selectors_must_match() -> None:
    body = (
        b'{"model":"tinfoil/secure-model",'
        b'"input":{"model":"private/gpt-oss-120b","content":"SECRET_PROMPT"}}'
    )

    with pytest.raises(HTTPException) as exc_info:
        parse_request_body_json(body, "v1/responses")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "error": {"type": "invalid_request_error", "code": "invalid_model"}
    }


def test_responses_matching_model_selectors_are_accepted() -> None:
    body = (
        b'{"model":"tinfoil/secure-model",'
        b'"input":{"model":" tinfoil/secure-model ","content":"test"}}'
    )

    parsed = parse_request_body_json(body, "v1/responses")

    assert parsed["model"] == "tinfoil/secure-model"
    assert parsed["input"]["model"] == " tinfoil/secure-model "
