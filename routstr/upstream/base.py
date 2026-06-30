from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import json
import re
import time
import traceback
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, Mapping, cast
from urllib.parse import urlparse

import httpx
from fastapi import BackgroundTasks, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic.v1 import (
    BaseModel,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    validator,
)
from sqlmodel import select

from ..auth import adjust_payment_for_tokens
from ..core import get_logger
from ..core.confidentiality_public import (
    is_full_sha256_digest,
    public_confidentiality_policy_binds_provider_proof,
    verified_public_provider_proof_claims,
)
from ..core.db import (
    ApiKey,
    AsyncSession,
    ModelRow,
    UpstreamProviderRow,
    canonical_json,
    create_session,
    store_cashu_transaction,
)
from ..core.exceptions import UpstreamError
from ..core.logging import (
    credential_fingerprint,
    redact_sensitive_text,
    redact_url_userinfo,
)
from ..core.policy_secrets import inline_policy_secret_violations
from ..core.settings import settings
from ..payment.cost_calculation import (
    CostData,
    CostDataError,
    MaxCostData,
    calculate_cost,
)
from ..payment.helpers import create_error_response
from ..payment.models import (
    Model,
    Pricing,
    _calculate_usd_max_costs,
    _update_model_sats_pricing,
    list_models,
    remote_model_without_public_proof,
)
from ..payment.price import sats_usd_price
from ..wallet import recieve_token, send_token
from . import messages_dispatch
from .count_tokens import count_tokens_locally
from .ehbp import (
    EhbpEncryptedRequest,
    decrypt_ehbp_response_body,
    encrypt_ehbp_request_body,
    parse_ehbp_key_config,
)
from .litellm_routing import detect_litellm_prefix

logger = get_logger(__name__)
SUPPORTED_CONFIDENTIALITY_MODES = (
    "tinfoil",
    "ppq-private-tee",
    "privatemode",
)
CONFIDENTIAL_PROVIDER_MODES = {
    "tinfoil": "tinfoil",
    "ppq-private": "ppq-private-tee",
    "privatemode": "privatemode",
}
CONFIDENTIAL_REDACTION_TEXT = "<redacted: confidential route>"


def _is_json_content_type(content_type: str | None) -> bool:
    """Return True when the upstream response should be parsed as JSON."""
    if not content_type:
        return False
    main = content_type.split(";", 1)[0].strip().lower()
    if main in ("application/json", "text/json"):
        return True
    return main.startswith("application/") and main.endswith("+json")


def _redact_confidential_json(value: Any) -> Any:
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, list):
        return [_redact_confidential_json(item) for item in value]
    if isinstance(value, dict):
        return {
            (
                redact_sensitive_text(key) if isinstance(key, str) else key
            ): _redact_confidential_json(item)
            for key, item in value.items()
        }
    return value


def _safe_diagnostic_text(value: object, *, confidential: bool) -> str:
    text = str(value)
    if confidential and text:
        return CONFIDENTIAL_REDACTION_TEXT
    return str(redact_sensitive_text(text))


def _is_multipart_form_data(content_type: str | None) -> bool:
    if not content_type:
        return False
    return content_type.split(";", 1)[0].strip().lower() == "multipart/form-data"


def _multipart_boundary(content_type: str | None) -> bytes | None:
    if not _is_multipart_form_data(content_type):
        return None
    match = re.search(r'(?:^|;)\s*boundary=(?:"([^"]+)"|([^;]+))', content_type or "")
    if not match:
        return None
    boundary = (match.group(1) or match.group(2) or "").strip()
    if not boundary:
        return None
    try:
        return boundary.encode("ascii")
    except UnicodeEncodeError:
        return None


def _multipart_part_field_name(headers: bytes) -> str | None:
    header_text = headers.decode("latin1", errors="ignore")
    for line in header_text.splitlines():
        if not line.lower().startswith("content-disposition:"):
            continue
        match = re.search(r'(?:^|;)\s*name=(?:"([^"]+)"|([^;]+))', line)
        if not match:
            return None
        return (match.group(1) or match.group(2) or "").strip()
    return None


def _replace_multipart_form_field(
    body: bytes,
    content_type: str | None,
    field_name: str,
    value: str,
) -> bytes | None:
    boundary = _multipart_boundary(content_type)
    if not boundary:
        return None

    delimiter = b"--" + boundary
    segments = body.split(delimiter)
    if len(segments) < 3:
        return None

    replacement = value.encode("utf-8")
    replaced = False
    rewritten: list[bytes] = [segments[0]]

    for segment in segments[1:]:
        if segment.startswith(b"--"):
            rewritten.append(segment)
            continue

        prefix = b""
        payload = segment
        if payload.startswith(b"\r\n"):
            prefix = b"\r\n"
            payload = payload[2:]
        elif payload.startswith(b"\n"):
            prefix = b"\n"
            payload = payload[1:]

        separator = b"\r\n\r\n" if b"\r\n\r\n" in payload else b"\n\n"
        if separator not in payload:
            rewritten.append(segment)
            continue

        part_headers, _, part_body = payload.partition(separator)
        if _multipart_part_field_name(part_headers) != field_name:
            rewritten.append(segment)
            continue

        suffix = b""
        if part_body.endswith(b"\r\n"):
            suffix = b"\r\n"
        elif part_body.endswith(b"\n"):
            suffix = b"\n"

        rewritten.append(prefix + part_headers + separator + replacement + suffix)
        replaced = True

    if not replaced:
        return None
    return delimiter.join(rewritten)


class TopupData(BaseModel):
    """Universal top-up data schema for Lightning Network invoices."""

    invoice_id: str
    payment_request: str
    amount: int
    currency: str
    expires_at: int | None = None
    checkout_url: str | None = None


class ConfidentialityStatus(BaseModel):
    """Runtime status for provider-side confidential inference verification."""

    enabled: StrictBool = False
    verified: StrictBool = False
    mode: StrictStr = "none"
    verified_at: StrictInt | None = None
    expires_at: StrictInt | None = None
    failure_reason: StrictStr | None = None
    model_ids: list[StrictStr] = Field(default_factory=list)
    model_id_prefixes: list[StrictStr] = Field(default_factory=list)
    verifier: StrictStr | None = None
    policy_digest: StrictStr | None = None
    evidence_digest: StrictStr | None = None
    verified_claims: dict[str, Any] = Field(default_factory=dict)

    def has_required_verification_evidence(self) -> bool:
        """Return True when verified status has concrete evidence metadata."""
        if not isinstance(self.verifier, str) or not self.verifier.strip():
            return False
        if not isinstance(self.verified_claims, dict) or not self.verified_claims:
            return False
        try:
            _sha256_json_digest(self.verified_claims)
        except (TypeError, ValueError):
            return False
        return bool(
            is_full_sha256_digest(self.policy_digest)
            and is_full_sha256_digest(self.evidence_digest)
        )


def _strict_confidential_selector_list(value: object) -> list[str] | None:
    if not isinstance(value, list):
        return None
    selectors: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return None
        selectors.append(item.strip().lower())
    if len(set(selectors)) != len(selectors):
        return None
    return selectors


def _strict_optional_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"{label} must be an integer")


def _is_loopback_hostname(hostname: str | None) -> bool:
    if not hostname:
        return False
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _privatemode_proxy_base_url_violation(base_url: str) -> str | None:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return "Privatemode proxy base_url must be an absolute URL"
    if parsed.username or parsed.password:
        return "Privatemode proxy base_url must not contain userinfo"
    if parsed.scheme not in {"http", "https"}:
        return "Privatemode proxy base_url must use http or https"
    if not parsed.hostname:
        return "Privatemode proxy base_url must include a host"
    try:
        parsed.port
    except ValueError:
        return "Privatemode proxy base_url must include a valid port"
    if not _is_loopback_hostname(parsed.hostname):
        return "Privatemode proxy base_url must use a loopback host"
    return None


def _safe_policy_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        if len(errors) == 1:
            message = errors[0].get("msg")
            if isinstance(message, str) and message.strip():
                return str(redact_sensitive_text(message.strip()))
    return str(redact_sensitive_text(str(exc)))


class ConfidentialVerifierPolicy(BaseModel):
    """Provider-specific verifier policy normalized for hashing and status."""

    provider_type: StrictStr
    mode: StrictStr
    base_url: StrictStr
    policy: dict[str, Any] = Field(default_factory=dict)
    model_ids: list[StrictStr] = Field(default_factory=list)
    model_id_prefixes: list[StrictStr] = Field(default_factory=list)

    @validator("mode")
    def _reject_unsupported_mode(cls, value: str) -> str:
        mode = value.strip().lower()
        if mode not in SUPPORTED_CONFIDENTIALITY_MODES:
            supported = (
                f"{', '.join(SUPPORTED_CONFIDENTIALITY_MODES[:-1])}, "
                f"or {SUPPORTED_CONFIDENTIALITY_MODES[-1]}"
            )
            raise ValueError(f"confidentiality mode must be {supported}")
        return mode

    @validator("policy")
    def _reject_inline_policy_secrets(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        violations = inline_policy_secret_violations(value)
        if violations:
            raise ValueError("; ".join(violations))
        try:
            _sha256_json_digest(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "confidential verifier policy must be canonical JSON"
            ) from exc
        return value

    @validator("model_ids", "model_id_prefixes")
    def _reject_invalid_model_selectors(
        cls,
        value: list[str],
        field: Any,
    ) -> list[str]:
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(f"{field.name} must be a list of non-empty strings")
        normalized = [item.strip() for item in value]
        if len({item.lower() for item in normalized}) != len(normalized):
            raise ValueError(f"{field.name} must not contain duplicates")
        return normalized

    @property
    def digest(self) -> str:
        payload = {
            "provider_type": self.provider_type,
            "mode": self.mode,
            "base_url": str(redact_url_userinfo(self.base_url)),
            "policy": self.policy,
            "model_ids": self.model_ids,
            "model_id_prefixes": self.model_id_prefixes,
        }
        return _sha256_json_digest(payload)

    @classmethod
    def from_provider_settings(
        cls,
        *,
        provider_type: str,
        base_url: str,
        provider_settings: Mapping[str, Any] | None,
    ) -> "ConfidentialVerifierPolicy | None":
        if not provider_settings:
            return None

        raw_confidentiality = provider_settings.get("confidentiality")
        mode = "none"
        model_ids: list[str] = []
        model_id_prefixes: list[str] = []
        policy_input: object | None = None

        if isinstance(raw_confidentiality, Mapping):
            mode = _confidentiality_mode(raw_confidentiality.get("mode"), default=mode)
            model_ids = _policy_string_list(raw_confidentiality, "model_ids")
            model_id_prefixes = _policy_string_list(
                raw_confidentiality, "model_id_prefixes"
            )
            policy_input = raw_confidentiality.get("policy")
            if policy_input is None:
                policy_input = raw_confidentiality.get("attestation_policy")

        for key in ("confidentiality_policy", "attestation_policy"):
            if policy_input is None and key in provider_settings:
                policy_input = provider_settings[key]

        if policy_input is None:
            if (
                isinstance(raw_confidentiality, Mapping)
                and ("mode" in raw_confidentiality or model_ids or model_id_prefixes)
                and mode.strip().lower() != "none"
            ):
                raise ValueError(
                    "confidential verifier policy is required when "
                    "confidentiality mode or model selectors are configured"
                )
            return None
        if not isinstance(policy_input, Mapping):
            raise ValueError("confidential verifier policy must be a JSON object")
        normalized_provider_type = provider_type.strip().lower()
        if (
            normalized_provider_type in CONFIDENTIAL_PROVIDER_MODES
            and model_id_prefixes
        ):
            raise ValueError(
                "confidential verifier policy for "
                f"{normalized_provider_type} requires exact model_ids; "
                "model_id_prefixes are not supported"
            )
        policy_dict = dict(policy_input)
        secret_violations = inline_policy_secret_violations(policy_dict)
        if secret_violations:
            raise ValueError("; ".join(secret_violations))

        return cls(
            provider_type=provider_type,
            mode=mode,
            base_url=base_url,
            policy=policy_dict,
            model_ids=model_ids,
            model_id_prefixes=model_id_prefixes,
        )


def _policy_string_list(source: Mapping[str, Any], key: str) -> list[str]:
    if key not in source:
        return []
    value = source[key]
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{key} must be a list of non-empty strings")
            values.append(item.strip())
        if len({value.lower() for value in values}) != len(values):
            raise ValueError(f"{key} must not contain duplicates")
        return values
    raise ValueError(f"{key} must be a list of non-empty strings")


def _confidentiality_mode(value: object, *, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError("confidentiality mode must be a non-empty string")


def _validate_confidentiality_status_mode(raw_status: Mapping[str, Any]) -> None:
    if "mode" in raw_status:
        _confidentiality_mode(raw_status.get("mode"), default="none")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_json_digest(value: object) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


async def _single_chunk_async_iter(content: bytes) -> AsyncIterator[bytes]:
    yield content


class BaseUpstreamProvider:
    """Provider for forwarding requests to an upstream AI service API."""

    provider_type: str = "base"
    upstream_name: str = "base"
    default_base_url: str | None = None
    platform_url: str | None = None

    supports_anthropic_messages: bool = False
    supports_audio_api: bool = True
    supports_audio_transcriptions_api: bool = True
    supports_audio_translations_api: bool = True
    supports_audio_speech_api: bool = True
    supports_completions_api: bool = True
    supports_embeddings_api: bool = True
    supports_images_api: bool = True
    supports_moderations_api: bool = True
    supports_responses_api: bool = True
    requires_verified_confidential_transport: bool = False
    requires_verified_ehbp_transport: bool = False
    # When None, the prefix is detected from `base_url` at dispatch time
    # (see `get_litellm_provider_prefix`). Subclasses set this to lock the
    # provider regardless of URL.
    litellm_provider_prefix: str | None = None

    base_url: str
    api_key: str
    provider_fee: float = 1.05
    _models_cache: list[Model] = []
    _models_by_id: dict[str, Model] = {}

    def __init__(self, base_url: str, api_key: str, provider_fee: float = 1.01):
        """Initialize the upstream provider.

        Args:
            base_url: Base URL of the upstream API endpoint
            api_key: API key for authenticating with the upstream service
            provider_fee: Provider fee multiplier (default 1.01 for 1% fee)
        """
        self.base_url = base_url
        self.api_key = api_key
        self.provider_fee = provider_fee
        self._models_cache = []
        self._models_by_id = {}
        self._confidentiality_status = ConfidentialityStatus()
        self._confidentiality_policy: ConfidentialVerifierPolicy | None = None
        self._confidentiality_policy_error: str | None = None
        self._last_model_catalog_refresh_succeeded: bool = True
        self._last_model_catalog_refresh_error: str | None = None

    def configure_confidentiality_from_settings(
        self, provider_settings: Mapping[str, Any] | None
    ) -> None:
        """Load confidential routing metadata from provider_settings.

        Provider-specific attestation adapters can replace this runtime status
        after they verify fresh evidence. The default remains fail-closed.
        """
        if not provider_settings:
            return

        self._confidentiality_policy = None
        self._confidentiality_policy_error = None
        try:
            policy = ConfidentialVerifierPolicy.from_provider_settings(
                provider_type=self.provider_type,
                base_url=self.base_url,
                provider_settings=provider_settings,
            )
            if policy:
                self._confidentiality_policy = policy
        except Exception as exc:
            safe_error = _safe_policy_error(exc)
            self._confidentiality_policy_error = safe_error
            logger.warning(
                "Ignoring invalid confidential verifier policy",
                extra={
                    "provider": self.provider_type,
                    "error": safe_error,
                    "error_type": type(exc).__name__,
                },
            )
            policy = None

        raw_status = provider_settings.get("confidentiality")
        if raw_status is None:
            if policy:
                self._confidentiality_status = self._confidentiality_status.copy(
                    update={"policy_digest": policy.digest}
                )
            elif self._confidentiality_policy_error:
                self._confidentiality_status = ConfidentialityStatus(
                    enabled=False,
                    verified=False,
                    mode="none",
                    failure_reason=self._confidentiality_policy_error,
                )
            return

        if isinstance(raw_status, bool):
            if self._confidentiality_policy_error and policy is None:
                self._confidentiality_status = ConfidentialityStatus(
                    enabled=False,
                    verified=False,
                    mode="none",
                    failure_reason=self._confidentiality_policy_error,
                )
                return
            self._confidentiality_status = ConfidentialityStatus(
                enabled=raw_status,
                verified=False,
                mode="manual" if raw_status else "none",
                failure_reason=self._confidentiality_policy_error,
                policy_digest=policy.digest if policy else None,
            )
            return

        if not isinstance(raw_status, Mapping):
            logger.warning(
                "Ignoring invalid confidentiality provider_settings",
                extra={"provider": self.provider_type},
            )
            self._confidentiality_status = ConfidentialityStatus(
                enabled=False,
                verified=False,
                mode="none",
                failure_reason="confidentiality provider_settings must be a JSON object",
                policy_digest=policy.digest if policy else None,
            )
            return

        try:
            _validate_confidentiality_status_mode(raw_status)
            status = ConfidentialityStatus.parse_obj(dict(raw_status))
            if self._confidentiality_policy_error and policy is None:
                self._confidentiality_status = ConfidentialityStatus(
                    enabled=False,
                    verified=False,
                    mode="none",
                    failure_reason=self._confidentiality_policy_error,
                )
                return
            # Static configuration can declare intended confidential transport,
            # model selectors, and policy metadata, but it is not evidence.
            # Only a runtime verifier/provider adapter may set verified=True.
            requested_verified = status.verified
            status.verified = False
            status.verified_at = None
            status.expires_at = None
            status.verifier = None
            status.evidence_digest = None
            status.verified_claims = {}
            if policy:
                status.policy_digest = policy.digest
            elif self._confidentiality_policy_error and not status.failure_reason:
                status.failure_reason = self._confidentiality_policy_error
            if requested_verified and not status.failure_reason:
                status.failure_reason = (
                    "static configuration is not attestation evidence"
                )
            self._confidentiality_status = status
        except Exception as exc:
            safe_error = str(redact_sensitive_text(str(exc)))
            failure_reason = self._confidentiality_policy_error or safe_error
            logger.warning(
                "Ignoring invalid confidentiality provider_settings",
                extra={
                    "provider": self.provider_type,
                    "error": safe_error,
                    "error_type": type(exc).__name__,
                },
            )
            self._confidentiality_status = ConfidentialityStatus(
                enabled=False,
                verified=False,
                mode="none",
                failure_reason=failure_reason,
                policy_digest=policy.digest if policy else None,
            )

    def set_confidentiality_status(self, status: ConfidentialityStatus) -> None:
        downgrade_reason = None
        if status.enabled is not True:
            status = status.copy(
                update={
                    "enabled": False,
                    "verified": False,
                    "verified_at": None,
                    "expires_at": None,
                    "verifier": None,
                    "evidence_digest": None,
                    "verified_claims": {},
                }
            )
        elif status.verified is False:
            pass
        elif status.verified is not True:
            downgrade_reason = (
                status.failure_reason
                or "confidentiality verifier verified must be true"
            )
        elif not status.has_required_verification_evidence():
            downgrade_reason = (
                status.failure_reason
                or "confidentiality verifier did not return required evidence"
            )
        else:
            try:
                verified_at = _strict_optional_int(status.verified_at, "verified_at")
                expires_at = _strict_optional_int(
                    status.expires_at,
                    "expires_at",
                )
            except ValueError as exc:
                downgrade_reason = f"confidentiality verifier {exc}"
                expires_at = None
                verified_at = None
            if downgrade_reason is None and verified_at is None:
                downgrade_reason = (
                    status.failure_reason
                    or "confidentiality verifier did not return verified_at"
                )
            now = int(time.time())
            if (
                downgrade_reason is None
                and verified_at is not None
                and verified_at > now
            ):
                downgrade_reason = (
                    status.failure_reason
                    or "confidentiality verifier verified_at is in the future"
                )
            if downgrade_reason is None and (expires_at is None or expires_at <= now):
                downgrade_reason = (
                    status.failure_reason
                    or "confidentiality verifier did not return a future expires_at"
                )
            expected_provider_mode = CONFIDENTIAL_PROVIDER_MODES.get(
                str(getattr(self, "provider_type", "") or "").strip().lower()
            )
            provider_type = (
                str(getattr(self, "provider_type", "") or "").strip().lower()
            )
            try:
                current_policy = self.confidentiality_policy()
            except Exception:
                current_policy = None
            if (
                downgrade_reason is None
                and expected_provider_mode is not None
                and current_policy is None
            ):
                downgrade_reason = (
                    status.failure_reason
                    or "confidentiality verifier policy is required for provider"
                )
            if downgrade_reason is None and current_policy is not None:
                policy = current_policy
                policy_digest = getattr(policy, "digest", None)
                policy_provider_type = getattr(policy, "provider_type", None)
                policy_mode = getattr(policy, "mode", None)
                policy_model_ids = getattr(policy, "model_ids", None)
                policy_model_id_prefixes = getattr(
                    policy,
                    "model_id_prefixes",
                    None,
                )
                if (
                    not isinstance(policy_digest, str)
                    or not isinstance(policy_mode, str)
                    or not isinstance(policy_model_ids, list)
                    or not isinstance(policy_model_id_prefixes, list)
                ):
                    downgrade_reason = (
                        status.failure_reason
                        or "confidentiality verifier policy is incomplete"
                    )
                elif (
                    isinstance(policy_provider_type, str)
                    and policy_provider_type.strip().lower() != provider_type
                ):
                    downgrade_reason = (
                        status.failure_reason
                        or "confidentiality verifier policy provider_type does not match provider"
                    )
                elif (
                    expected_provider_mode is not None
                    and policy_mode != expected_provider_mode
                ):
                    downgrade_reason = (
                        status.failure_reason
                        or "confidentiality verifier mode does not match provider_type"
                    )
                elif status.policy_digest != policy_digest:
                    downgrade_reason = (
                        status.failure_reason
                        or "confidentiality verifier policy digest does not match policy"
                    )
                elif status.mode.strip().lower() != policy_mode:
                    downgrade_reason = (
                        status.failure_reason
                        or "confidentiality verifier mode does not match policy"
                    )
                elif expected_provider_mode is not None and policy_model_id_prefixes:
                    downgrade_reason = (
                        status.failure_reason
                        or "confidentiality verifier prefix selectors are not valid provider evidence"
                    )
                elif expected_provider_mode is not None and (
                    _strict_confidential_selector_list(policy_model_ids) is None
                    or _strict_confidential_selector_list(policy_model_id_prefixes)
                    is None
                    or _strict_confidential_selector_list(list(status.model_ids))
                    is None
                    or _strict_confidential_selector_list(
                        list(status.model_id_prefixes)
                    )
                    is None
                ):
                    downgrade_reason = (
                        status.failure_reason
                        or "confidentiality verifier model selectors are invalid"
                    )
                elif (
                    expected_provider_mode is not None
                    and not _strict_confidential_selector_list(policy_model_ids)
                ):
                    downgrade_reason = (
                        status.failure_reason
                        or "confidentiality verifier exact model selectors are required for provider"
                    )
                elif list(status.model_ids) != list(policy_model_ids) or list(
                    status.model_id_prefixes
                ) != list(policy_model_id_prefixes):
                    downgrade_reason = (
                        status.failure_reason
                        or "confidentiality verifier model selectors do not match policy"
                    )
                elif expected_provider_mode is not None and policy_model_ids:
                    raw_policy = getattr(policy, "policy", None)
                    public_proof_claims = verified_public_provider_proof_claims(
                        status.mode,
                        status.verified_claims,
                        public_policy=raw_policy
                        if isinstance(raw_policy, dict)
                        else None,
                    )
                    if public_proof_claims is None:
                        downgrade_reason = (
                            status.failure_reason
                            or "confidentiality verifier proof claims do not satisfy provider policy"
                        )
                    elif not public_confidentiality_policy_binds_provider_proof(
                        provider_type,
                        status.mode,
                        raw_policy,
                        public_proof_claims,
                        policy_model_ids,
                    ):
                        downgrade_reason = (
                            status.failure_reason
                            or "confidentiality verifier proof claims do not satisfy provider policy"
                        )

        if downgrade_reason:
            status = status.copy(
                update={
                    "verified": False,
                    "verified_at": None,
                    "expires_at": None,
                    "failure_reason": downgrade_reason,
                    "verifier": None,
                    "evidence_digest": None,
                    "verified_claims": {},
                }
            )
        self._confidentiality_status = status

    def confidentiality_status(self) -> ConfidentialityStatus:
        return self._confidentiality_status

    def confidentiality_policy(self) -> ConfidentialVerifierPolicy | None:
        return self._confidentiality_policy

    def confidentiality_policy_error(self) -> str | None:
        return self._confidentiality_policy_error

    def _require_current_confidential_transport(self) -> ConfidentialityStatus:
        status = self.confidentiality_status()
        if status.enabled is not True or status.verified is not True:
            raise UpstreamError(
                "verified confidential transport is required for this provider",
                status_code=424,
            )
        if not status.has_required_verification_evidence():
            raise UpstreamError(
                "verified confidential transport evidence is incomplete",
                status_code=424,
            )
        try:
            expires_at = _strict_optional_int(
                status.expires_at,
                "expires_at",
            )
        except ValueError:
            expires_at = None
        if expires_at is None or expires_at <= int(time.time()):
            raise UpstreamError(
                "verified confidential transport attestation has expired",
                status_code=424,
            )
        if getattr(self, "provider_type", None) == "privatemode" and (
            violation := _privatemode_proxy_base_url_violation(
                str(getattr(self, "base_url", "") or "")
            )
        ):
            raise UpstreamError(violation, status_code=424)
        from ..algorithm import has_current_confidential_verification

        if not has_current_confidential_verification(self):
            raise UpstreamError(
                "verified confidential transport evidence is incomplete",
                status_code=424,
            )
        return status

    def _verified_ehbp_key_config(self) -> bytes:
        status = self.confidentiality_status()
        if status.enabled is not True or status.verified is not True:
            raise UpstreamError(
                "verified EHBP transport is required for this provider",
                status_code=424,
            )
        if not status.has_required_verification_evidence():
            raise UpstreamError(
                "verified EHBP transport evidence is incomplete",
                status_code=424,
            )
        try:
            expires_at = _strict_optional_int(
                status.expires_at,
                "expires_at",
            )
        except ValueError:
            expires_at = None
        if expires_at is None or expires_at <= int(time.time()):
            raise UpstreamError(
                "verified EHBP transport attestation has expired",
                status_code=424,
            )
        from ..algorithm import has_current_confidential_verification

        if not has_current_confidential_verification(self):
            raise UpstreamError(
                "verified EHBP transport evidence is incomplete",
                status_code=424,
            )
        transport = status.verified_claims.get("transport")
        key_config_b64 = status.verified_claims.get("ehbp_key_config_b64")
        if (
            not isinstance(transport, str)
            or transport.strip().lower() != "ehbp"
            or not isinstance(key_config_b64, str)
        ):
            raise UpstreamError(
                "verified EHBP transport is required for this provider",
                status_code=424,
            )
        try:
            key_config = base64.b64decode(key_config_b64, validate=True)
            parsed_key_config = parse_ehbp_key_config(key_config)
        except Exception as exc:
            raise UpstreamError(
                "verified EHBP key config is invalid",
                status_code=424,
            ) from exc

        expected_public_key_hex = parsed_key_config.public_key.hex()
        public_key_claims = []
        for claim_name in (
            "attested_hpke_public_key_hex",
            "hpke_public_key_hex",
            "hpke_public_key",
        ):
            if claim_name not in status.verified_claims:
                continue
            claim_value = status.verified_claims.get(claim_name)
            if not isinstance(claim_value, str):
                raise UpstreamError(
                    "verified EHBP key config does not match attested key",
                    status_code=424,
                )
            normalized = claim_value.strip().lower()
            if normalized:
                public_key_claims.append(normalized)
        if not public_key_claims or any(
            claim != expected_public_key_hex for claim in public_key_claims
        ):
            raise UpstreamError(
                "verified EHBP key config does not match attested key",
                status_code=424,
            )

        digest_value = status.verified_claims.get("ehbp_public_key_digest")
        if digest_value is not None and (
            not isinstance(digest_value, str)
            or digest_value.strip().lower() != parsed_key_config.public_key_digest
        ):
            raise UpstreamError(
                "verified EHBP key config does not match attested key",
                status_code=424,
            )
        return key_config

    def _prepare_ehbp_request_body(
        self,
        headers: dict[str, str],
        body: bytes | None,
    ) -> tuple[bytes, EhbpEncryptedRequest]:
        if not body:
            raise UpstreamError(
                "verified EHBP transport requires a non-empty request body",
                status_code=424,
            )
        encrypted = encrypt_ehbp_request_body(self._verified_ehbp_key_config(), body)
        headers.pop("content-length", None)
        headers.pop("Content-Length", None)
        headers["Ehbp-Encapsulated-Key"] = encrypted.encapsulated_key_hex
        headers["Transfer-Encoding"] = "chunked"
        return encrypted.body, encrypted

    def _require_confidential_transport_covers_model(
        self,
        model_obj: Model,
        *,
        route_alias: str | None = None,
    ) -> None:
        if not (
            self.requires_verified_confidential_transport
            or self.requires_verified_ehbp_transport
        ):
            return

        if self.requires_verified_confidential_transport:
            self._require_current_confidential_transport()
        if self.requires_verified_ehbp_transport:
            self._verified_ehbp_key_config()

        from ..algorithm import is_confidential_provider_for_model

        if not is_confidential_provider_for_model(self, model_obj, route_alias):
            raise UpstreamError(
                "verified confidential transport does not cover requested model",
                status_code=424,
            )

    async def _decrypt_ehbp_response(
        self,
        response: httpx.Response,
        encrypted_request: EhbpEncryptedRequest,
    ) -> httpx.Response:
        nonce_hex = response.headers.get("Ehbp-Response-Nonce")
        if not nonce_hex:
            await response.aclose()
            raise UpstreamError(
                "confidential upstream response missing EHBP response nonce",
                status_code=502,
            )
        try:
            response_nonce = bytes.fromhex(nonce_hex)
            encrypted_body = await response.aread()
            decrypted_body = decrypt_ehbp_response_body(
                encrypted_request.exported_secret,
                encrypted_request.encapsulated_key,
                response_nonce,
                encrypted_body,
            )
        except Exception as exc:
            await response.aclose()
            raise UpstreamError(
                "confidential upstream response decryption failed",
                status_code=502,
            ) from exc

        headers = dict(response.headers)
        for header_name in (
            "Ehbp-Response-Nonce",
            "ehbp-response-nonce",
            "content-length",
            "Content-Length",
            "transfer-encoding",
            "Transfer-Encoding",
            "content-encoding",
            "Content-Encoding",
        ):
            headers.pop(header_name, None)

        decrypted_response = httpx.Response(
            status_code=response.status_code,
            headers=headers,
            content=decrypted_body,
            request=response.request,
        )
        await response.aclose()
        return decrypted_response

    def confidential_logging_enabled(self) -> bool:
        """Return True when upstream payloads must not be written to logs."""
        mode = str(getattr(settings, "confidential_routing_mode", "") or "")
        if mode.strip().lower() in {"required", "require", "strict", "enforced"}:
            return True
        status = self.confidentiality_status()
        return bool(
            status.enabled
            or status.verified
            or (status.mode and status.mode.strip().lower() != "none")
        )

    async def refresh_confidentiality_status(self) -> ConfidentialityStatus:
        """Refresh provider attestation state.

        Base providers have no verifier, so they return the current fail-closed
        status. Confidential provider adapters override this with real
        attestation and encryption key verification.
        """
        return self._confidentiality_status

    def get_litellm_provider_prefix(self) -> str:
        """Resolve the litellm provider prefix for this provider instance.

        1. If the subclass pinned `litellm_provider_prefix`, use it.
        2. Otherwise infer from `base_url` (e.g. ``api.fireworks.ai`` →
           ``fireworks_ai/``) so custom/generic rows reach the correct
           litellm backend instead of falling back to ``openai/``.
        3. Default ``openai/`` for unknown OpenAI-compatible servers.
        """
        if self.__class__.litellm_provider_prefix:
            return self.__class__.litellm_provider_prefix
        return detect_litellm_prefix(self.base_url)

    @classmethod
    def from_db_row(
        cls, provider_row: "UpstreamProviderRow"
    ) -> "BaseUpstreamProvider | None":
        """Factory method to instantiate provider from database row.

        Args:
            provider_row: Database row containing provider configuration

        Returns:
            Instantiated provider or None if instantiation fails
        """
        return cls(
            base_url=provider_row.base_url,
            api_key=provider_row.api_key,
            provider_fee=provider_row.provider_fee,
        )

    @classmethod
    def get_provider_metadata(cls) -> dict[str, object]:
        """Get metadata about this provider type for API responses.

        Returns:
            Dict with provider type metadata including id, name, default_base_url, fixed_base_url, platform_url, can_create_account, can_topup, can_show_balance
        """
        return {
            "id": cls.provider_type,
            "name": cls.provider_type.title(),
            "default_base_url": cls.default_base_url or "",
            "fixed_base_url": bool(cls.default_base_url),
            "platform_url": cls.platform_url,
            "can_create_account": False,
            "can_topup": False,
            "can_show_balance": False,
        }

    @staticmethod
    def _fold_cache_into_input_tokens(usage: object) -> None:
        """Fold cache token counts into ``input_tokens`` / ``prompt_tokens``.

        Cost calculation has already used the per-bucket counts to bill the
        request correctly; what the client sees in the visible token total
        should be a single rolled-up prompt count *including* the cache
        portion. The standalone ``cache_read_input_tokens`` /
        ``cache_creation_input_tokens`` fields are left in place for clients
        that want the breakdown.

        For Anthropic-shaped responses (``input_tokens`` present), the cache
        fields are forced to ``0`` when the upstream omitted them, so the
        client always sees a consistent shape.
        """
        if not isinstance(usage, dict):
            return

        # Normalise missing cache fields to 0 on Anthropic-shaped usage so
        # downstream consumers can rely on them being present.
        if "input_tokens" in usage:
            usage.setdefault("cache_read_input_tokens", 0)
            usage.setdefault("cache_creation_input_tokens", 0)

        try:
            cache_read = int(usage.get("cache_read_input_tokens") or 0)
            cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
        except (TypeError, ValueError):
            return
        extra = cache_read + cache_creation
        if extra <= 0:
            return
        if "input_tokens" in usage:
            try:
                usage["input_tokens"] = int(usage.get("input_tokens") or 0) + extra
            except (TypeError, ValueError):
                pass
        if "prompt_tokens" in usage:
            try:
                usage["prompt_tokens"] = int(usage.get("prompt_tokens") or 0) + extra
            except (TypeError, ValueError):
                pass

    def _apply_provider_field(self, response_json: object) -> None:
        """Stamp the routstr ``provider`` field onto an upstream response payload.

        Format is ``"<provider_type>:<upstream_provider>"`` when the upstream
        already reported its own provider (e.g. OpenRouter returns
        ``"provider": "Fireworks"``), otherwise just ``"<provider_type>"``
        for direct upstreams.
        """
        if not isinstance(response_json, dict):
            return
        existing = response_json.get("provider")
        if isinstance(existing, str) and existing.strip():
            response_json["provider"] = f"{self.provider_type}:{existing.strip()}"
        else:
            response_json["provider"] = self.provider_type

    def inject_cost_metadata(
        self,
        response_json: dict,
        cost_data: CostData | MaxCostData | dict,
        key: ApiKey,
    ) -> None:
        """Unifies the injection of cost and usage metadata across all completion types."""
        self._apply_provider_field(response_json)
        if isinstance(cost_data, dict):
            total_msats = cost_data.get("total_msats", 0)
            total_usd = cost_data.get("total_usd", 0.0)
            cost_dict = cost_data
        else:
            total_msats = cost_data.total_msats
            total_usd = cost_data.total_usd
            cost_dict = cost_data.dict()

        sats_cost = total_msats // 1000

        # Inject into top-level usage block (OpenAI/Anthropic style)
        if "usage" in response_json:
            response_json["usage"]["cost"] = total_usd
            response_json["usage"]["cost_sats"] = sats_cost
            response_json["usage"]["remaining_balance_msats"] = key.balance
            self._fold_cache_into_input_tokens(response_json["usage"])

        # Inject into Anthropic nested usage block if present
        if (
            "message" in response_json
            and isinstance(response_json["message"], dict)
            and "usage" in response_json["message"]
        ):
            response_json["message"]["usage"]["sats_cost"] = sats_cost
            self._fold_cache_into_input_tokens(response_json["message"]["usage"])

        # Unified Routstr metadata
        response_json["metadata"] = response_json.get("metadata", {})
        response_json["metadata"]["routstr"] = {
            "cost": cost_dict,
            "sats_cost": sats_cost,
            "remaining_balance_msats": key.balance,
        }

        # Legacy/Compatibility fields
        response_json["cost"] = cost_dict.copy()
        response_json["cost"]["sats_cost"] = sats_cost
        response_json["cost"]["remaining_balance_msats"] = key.balance

    def prepare_headers(self, request_headers: dict) -> dict:
        """Prepare headers for upstream request by removing proxy-specific headers and adding authentication.

        Args:
            request_headers: Original request headers from the client

        Returns:
            Headers dict ready for upstream forwarding with authentication added
        """
        logger.debug(
            "Preparing upstream headers",
            extra={
                "original_headers_count": len(request_headers),
                "has_upstream_api_key": bool(self.api_key),
            },
        )

        headers = dict(request_headers)
        removed_headers = []

        for header in [
            "host",
            "content-length",
            "refund-lnurl",
            "key-expiry-time",
            "x-cashu",
        ]:
            if headers.pop(header, None) is not None:
                removed_headers.append(header)

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            if headers.pop("authorization", None) is not None:
                removed_headers.append("authorization (replaced with upstream key)")
        else:
            for auth_header in ["Authorization", "authorization"]:
                if headers.pop(auth_header, None) is not None:
                    removed_headers.append(auth_header)

        for header in ["authorization", "accept-encoding"]:
            if headers.pop(header, None) is not None:
                removed_headers.append(f"{header} (replaced with routstr-safe version)")

        # Explicitly define the list of supported compression encodings
        headers["accept-encoding"] = "gzip, deflate, br, identity"

        logger.debug(
            "Headers prepared for upstream",
            extra={
                "final_headers_count": len(headers),
                "removed_headers": removed_headers,
                "added_upstream_auth": bool(self.api_key),
            },
        )

        return headers

    def prepare_params(
        self, path: str, query_params: Mapping[str, str] | None
    ) -> Mapping[str, str]:
        """Prepare query parameters for upstream request.

        Base implementation passes through query params unchanged. Override in subclasses for provider-specific params.

        Args:
            path: Request path
            query_params: Original query parameters from the client

        Returns:
            Query parameters dict ready for upstream forwarding
        """
        return query_params or {}

    def transform_model_name(self, model_id: str) -> str:
        """Transform model ID for this provider's API format.

        Base implementation returns model_id unchanged. Override in subclasses for provider-specific transformations.

        Args:
            model_id: Model identifier (may include provider prefix)

        Returns:
            Transformed model ID for this provider
        """
        return model_id

    def request_model_id_for_transform(self, model_obj: Model) -> str:
        """Return the model identity used for upstream request-body transforms."""
        return model_obj.id

    def normalize_request_path(self, path: str, model_obj: Model | None = None) -> str:
        """Normalize request path before forwarding to upstream."""
        if path.startswith("v1/"):
            return path.replace("v1/", "", 1)
        return path

    def _require_supported_endpoint_for_path(
        self,
        path: str,
        model_obj: Model | None = None,
    ) -> None:
        from ..algorithm import (
            provider_endpoint_requirement_for_path,
            provider_requires_known_model_endpoint,
            provider_supports_required_endpoint,
        )

        requirement = provider_endpoint_requirement_for_path(path)
        if requirement is None:
            if provider_requires_known_model_endpoint(self):
                raise UpstreamError(
                    f"{self.provider_type} does not support unrecognized API paths",
                    status_code=400,
                )
            return

        support_attr, endpoint_name = requirement
        if provider_supports_required_endpoint(
            self,
            support_attr,
            model=model_obj,
        ):
            return

        raise UpstreamError(
            f"{self.provider_type} does not support the {endpoint_name}",
            status_code=400,
        )

    def _require_x_cashu_preconditions_before_token(
        self,
        path: str,
        model_obj: Model,
        *,
        route_alias: str | None = None,
    ) -> None:
        normalized_path = self.normalize_request_path(path, model_obj)
        self._require_supported_endpoint_for_path(normalized_path, model_obj)

        if self.requires_verified_confidential_transport:
            self._require_current_confidential_transport()
        if self.requires_verified_ehbp_transport:
            self._verified_ehbp_key_config()
        self._require_confidential_transport_covers_model(
            model_obj,
            route_alias=route_alias or model_obj.forwarded_model_id or model_obj.id,
        )

    def get_request_base_url(self, path: str, model_obj: Model | None = None) -> str:
        """Get upstream base URL used when building forwarding URL."""
        return self.base_url.rstrip("/")

    def build_request_url(self, path: str, model_obj: Model | None = None) -> str:
        """Build full upstream URL from normalized path."""
        clean_path = path.lstrip("/")
        return f"{self.get_request_base_url(path, model_obj)}/{clean_path}"

    def prepare_responses_request_body(
        self, body: bytes | None, model_obj: Model
    ) -> bytes | None:
        """Transform request body for Responses API specific requirements.

        Handles Responses API specific transformations while maintaining model name transforms.

        Args:
            body: Original request body bytes
            model_obj: Model object containing the original model information

        Returns:
            Transformed request body bytes
        """
        if not body:
            return body

        try:
            data = json.loads(body)
            if isinstance(data, dict):
                # Handle model transformation in various locations
                if "model" in data:
                    original_model = self.request_model_id_for_transform(model_obj)
                    transformed_model = self.transform_model_name(original_model)
                    data["model"] = transformed_model

                    logger.debug(
                        "Transformed model name in Responses API request",
                        extra={
                            "original": original_model,
                            "transformed": transformed_model,
                            "provider": self.provider_type or self.base_url,
                        },
                    )

                # Handle model in input field (alternative format)
                if (
                    "input" in data
                    and isinstance(data["input"], dict)
                    and "model" in data["input"]
                ):
                    original_model = self.request_model_id_for_transform(model_obj)
                    transformed_model = self.transform_model_name(original_model)
                    data["input"]["model"] = transformed_model

                # Ensure proper Responses API structure
                # Add any Responses-specific transformations here

                return json.dumps(data).encode()
        except Exception as e:
            logger.debug(
                "Could not transform Responses API request body",
                extra={
                    "error": str(e),
                    "provider": self.provider_type or self.base_url,
                },
            )

        return body

    def prepare_request_body(
        self,
        body: bytes | None,
        model_obj: Model,
        path: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> bytes | None:
        """Transform request body for provider-specific requirements.

        Automatically transforms model names and, for streaming chat
        completions, opts the upstream into emitting per-chunk ``usage``
        so cost tracking can read real token counts instead of falling
        back to ``MaxCostData``.

        Args:
            body: Original request body bytes

        Returns:
            Transformed request body bytes
        """
        if not body:
            return body

        try:
            data = json.loads(body)
        except Exception as e:
            content_type = None
            if headers is not None:
                content_type = headers.get("content-type") or headers.get(
                    "Content-Type"
                )
            original_model = self.request_model_id_for_transform(model_obj)
            transformed_model = self.transform_model_name(original_model)
            if transformed_model and _is_multipart_form_data(content_type):
                transformed_body = _replace_multipart_form_field(
                    body,
                    content_type,
                    "model",
                    transformed_model,
                )
                if transformed_body is not None:
                    logger.debug(
                        "Transformed model name in multipart request",
                        extra={
                            "original": original_model,
                            "transformed": transformed_model,
                            "path": path,
                            "provider": self.provider_type or self.base_url,
                        },
                    )
                    return transformed_body

            logger.debug(
                "Could not parse request body for transformation",
                extra={
                    "error": str(e),
                    "provider": self.provider_type or self.base_url,
                },
            )
            return body

        if not isinstance(data, dict):
            return body

        changed = False

        if "model" in data:
            original_model = self.request_model_id_for_transform(model_obj)
            transformed_model = self.transform_model_name(original_model)
            if data["model"] != transformed_model:
                data["model"] = transformed_model
                logger.debug(
                    "Transformed model name in request",
                    extra={
                        "original": original_model,
                        "transformed": transformed_model,
                        "provider": self.provider_type or self.base_url,
                    },
                )
                changed = True

        # OpenAI-compatible streaming responses omit ``usage`` unless the
        # request sets ``stream_options.include_usage = true``. Without it
        # we can't reconcile token counts at end of stream and the
        # request gets billed at max-cost with zero tokens. Discriminate
        # chat-completions-shaped requests by the ``messages`` field so we
        # don't poke unrelated endpoints.
        if (
            data.get("stream") is True
            and "messages" in data
            and isinstance(data.get("messages"), list)
        ):
            existing = data.get("stream_options")
            merged = dict(existing) if isinstance(existing, dict) else {}
            if merged.get("include_usage") is not True:
                merged["include_usage"] = True
                data["stream_options"] = merged
                changed = True

        if changed:
            return json.dumps(data).encode()
        return body

    def prepare_request_headers(
        self,
        headers: dict[str, str],
        path: str,
        request_body: bytes | None,
        model_obj: Model,
    ) -> None:
        """Allow providers to add request headers that depend on model context."""
        for auth_header in ("Authorization", "authorization"):
            headers.pop(auth_header, None)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

    def _extract_upstream_error_message(
        self, body_bytes: bytes
    ) -> tuple[str, str | None]:
        """Extract error message and code from upstream error response body.

        Args:
            body_bytes: Raw response body bytes from upstream

        Returns:
            Tuple of (error_message, error_code), where error_code may be None
        """
        message: str = "Upstream request failed"
        upstream_code: str | None = None
        if not body_bytes:
            return message, upstream_code
        try:
            data = json.loads(body_bytes)
            if isinstance(data, dict):
                err = data.get("error")
                if isinstance(err, dict):
                    raw_msg = (
                        err.get("message") or err.get("detail") or err.get("error")
                    )
                    if isinstance(raw_msg, (str, int, float)):
                        message = str(raw_msg)
                    upstream_code_raw = err.get("code") or err.get("type")
                    if isinstance(upstream_code_raw, (str, int, float)):
                        upstream_code = str(upstream_code_raw)
                elif "message" in data and isinstance(
                    data["message"], (str, int, float)
                ):
                    message = str(data["message"])  # type: ignore[arg-type]
                elif "detail" in data and isinstance(data["detail"], (str, int, float)):
                    message = str(data["detail"])  # type: ignore[arg-type]
        except Exception:
            preview = body_bytes.decode("utf-8", errors="ignore").strip()
            if preview:
                message = preview[:500]
        return message, upstream_code

    async def on_upstream_error_redirect(
        self, status_code: int, error_message: str
    ) -> None:
        """Hook called when the proxy redirects to another provider due to an error.

        Subclasses can implement this to perform actions like disabling the provider
        if it's out of balance.

        Args:
            status_code: The HTTP status code returned by the upstream
            error_message: The error message extracted from the upstream response
        """
        pass

    async def forward_upstream_error_response(
        self,
        request: Request,
        path: str,
        upstream_response: httpx.Response,
        model_id: str | None = None,
    ) -> Response:
        """Log upstream errors and forward the response in a JSON envelope."""
        status_code = upstream_response.status_code
        headers = dict(upstream_response.headers)
        content_type = headers.get("content-type") or headers.get("Content-Type", "")
        upstream_request_id = (
            headers.get("request-id")
            or headers.get("Request-Id")
            or headers.get("x-request-id")
            or headers.get("X-Request-Id")
            or headers.get("anthropic-request-id")
            or headers.get("openai-request-id")
        )

        body_read_error = None
        try:
            body_bytes = await upstream_response.aread()
        except Exception as exc:
            body_bytes = b""
            body_read_error = f"{type(exc).__name__}: {exc}"

        message, upstream_code = self._extract_upstream_error_message(body_bytes)
        body_preview = body_bytes.decode("utf-8", errors="ignore").strip()[:500]
        is_json_body = _is_json_content_type(content_type)
        redact_body = self.confidential_logging_enabled()
        safe_log_preview = (
            CONFIDENTIAL_REDACTION_TEXT
            if redact_body and (message or body_preview)
            else (message or body_preview or "<empty>")[:300]
        )

        logger.warning(
            "Upstream %s returned %s for model=%s path=%s: %s",
            self.provider_type,
            status_code,
            model_id or "unknown",
            path,
            safe_log_preview,
            extra={
                "path": path,
                "provider": self.provider_type,
                "model": model_id or "unknown",
                "upstream_status": status_code,
                "upstream_code": upstream_code,
                "upstream_content_type": content_type,
                "upstream_request_id": upstream_request_id,
                "message_preview": None if redact_body else message[:200],
                "body_preview": None if redact_body else body_preview,
                "body_read_error": body_read_error,
                "method": request.method,
                "json_normalized": not is_json_body,
                "confidential": redact_body,
            },
        )

        for header_name in (
            "content-length",
            "Content-Length",
            "transfer-encoding",
            "Transfer-Encoding",
            "content-encoding",
            "Content-Encoding",
            "connection",
            "Connection",
            "keep-alive",
            "Keep-Alive",
            "proxy-authenticate",
            "Proxy-Authenticate",
            "proxy-authorization",
            "Proxy-Authorization",
            "te",
            "TE",
            "trailer",
            "Trailer",
            "upgrade",
            "Upgrade",
        ):
            headers.pop(header_name, None)

        if is_json_body:
            if not content_type:
                headers.pop("content-type", None)
                headers.pop("Content-Type", None)
            media_type = content_type or None
            if redact_body:
                try:
                    redacted_json = _redact_confidential_json(json.loads(body_bytes))
                except Exception:
                    for header_name in ("content-type", "Content-Type"):
                        headers.pop(header_name, None)
                    envelope = {
                        "error": {
                            "message": (
                                "Upstream returned an unreadable JSON error response "
                                "on a confidential route"
                            ),
                            "type": "upstream_error",
                            "code": upstream_code or status_code,
                            "upstream_status": status_code,
                            "upstream_content_type": content_type or None,
                            "upstream_body_preview": None,
                        },
                        "request_id": getattr(request.state, "request_id", None),
                    }
                    return Response(
                        content=json.dumps(envelope).encode(),
                        status_code=status_code,
                        headers=headers,
                        media_type="application/json",
                    )
                body_bytes = json.dumps(redacted_json).encode()
            return Response(
                content=body_bytes,
                status_code=status_code,
                headers=headers,
                media_type=media_type,
            )

        # Non-JSON upstream error (HTML, plain text, empty, ...). Wrap it in
        # the standard JSON envelope so callers don't need a second parser.
        for header_name in ("content-type", "Content-Type"):
            headers.pop(header_name, None)

        envelope = {
            "error": {
                "message": (
                    "Upstream returned a non-JSON error response on a confidential route"
                    if redact_body
                    else message or "Upstream returned a non-JSON error response"
                ),
                "type": "upstream_error",
                "code": upstream_code or status_code,
                "upstream_status": status_code,
                "upstream_content_type": content_type or None,
                "upstream_body_preview": None if redact_body else body_preview or None,
            },
            "request_id": getattr(request.state, "request_id", None),
        }

        return Response(
            content=json.dumps(envelope).encode(),
            status_code=status_code,
            headers=headers,
            media_type="application/json",
        )

    async def handle_streaming_chat_completion(
        self,
        response: httpx.Response,
        key: ApiKey,
        max_cost_for_model: int,
        background_tasks: BackgroundTasks,
        requested_model: str | None = None,
    ) -> StreamingResponse:
        """Handle streaming chat completion responses with token usage tracking and cost adjustment.

        Args:
            response: Streaming response from upstream
            key: API key for the authenticated user
            max_cost_for_model: Maximum cost deducted upfront for the model

        Returns:
            StreamingResponse with cost data injected at the end
        """
        logger.debug(
            "Processing streaming chat completion",
            extra={
                "key_hash": key.hashed_key[:8] + "...",
                "key_balance": key.balance,
                "response_status": response.status_code,
            },
        )

        async def stream_with_cost(
            max_cost_for_model: int,
        ) -> AsyncGenerator[bytes, None]:
            usage_finalized: bool = False
            last_model_seen: str | None = None
            usage_chunk_data: dict | None = None
            done_seen: bool = False

            async def finalize_db_only() -> None:
                nonlocal usage_finalized
                if usage_finalized:
                    return
                async with create_session() as new_session:
                    fresh_key = await new_session.get(key.__class__, key.hashed_key)
                    if not fresh_key:
                        return
                    try:
                        await adjust_payment_for_tokens(
                            fresh_key,
                            {"model": last_model_seen or "unknown", "usage": None},
                            new_session,
                            max_cost_for_model,
                        )
                        usage_finalized = True
                    except Exception:
                        pass

            try:
                async for chunk in response.aiter_bytes():
                    # Split chunk into SSE events
                    parts = re.split(b"data: ", chunk)
                    for i, part in enumerate(parts):
                        if not part:
                            continue

                        stripped_part = part.strip()
                        if not stripped_part:
                            continue

                        if stripped_part == b"[DONE]":
                            done_seen = True
                            continue

                        try:
                            # Only parse if it looks like a JSON object to avoid SSE control messages or partials
                            if part.strip().startswith(b"{") and part.strip().endswith(
                                b"}"
                            ):
                                obj = json.loads(part)
                                if isinstance(obj, dict):
                                    self._apply_provider_field(obj)
                                    if obj.get("model"):
                                        last_model_seen = str(obj.get("model"))
                                    if requested_model:
                                        obj["model"] = requested_model
                                    if (
                                        "id" not in obj
                                        or not isinstance(obj["id"], str)
                                        or obj["id"] == "existing-id"
                                    ):
                                        if not hasattr(self, "_current_stream_id"):
                                            self._current_stream_id = (
                                                f"chatcmpl-{uuid.uuid4()}"
                                            )
                                        obj["id"] = self._current_stream_id
                                    if isinstance(obj.get("usage"), dict):
                                        usage_chunk_data = obj
                                        continue
                                    yield b"data: " + json.dumps(obj).encode() + b"\n\n"
                                    continue
                        except Exception:
                            pass

                        prefix = (
                            b"data: " if (i > 0 or chunk.startswith(b"data: ")) else b""
                        )
                        yield prefix + part

                async with create_session() as session:
                    fresh_key = await session.get(key.__class__, key.hashed_key)
                    if fresh_key:
                        cost_data: dict
                        try:
                            adjustment_input = (
                                usage_chunk_data
                                if usage_chunk_data is not None
                                else {
                                    "model": last_model_seen or "unknown",
                                    "usage": None,
                                }
                            )
                            cost_data = await adjust_payment_for_tokens(
                                fresh_key,
                                adjustment_input,
                                session,
                                max_cost_for_model,
                            )
                            usage_finalized = True
                        except Exception as e:
                            logger.exception(
                                "Error during usage finalization",
                                extra={
                                    "key_hash": key.hashed_key[:8] + "...",
                                    "error": str(e),
                                },
                            )

                            # Fall back so we still emit a non-zero sats cost downstream.
                            cost_data = {
                                "base_msats": 0,
                                "input_msats": 0,
                                "output_msats": 0,
                                "total_msats": 0,
                                "total_usd": 0.0,
                                "input_tokens": 0,
                                "output_tokens": 0,
                            }

                        if usage_chunk_data is None:
                            if not hasattr(self, "_current_stream_id"):
                                self._current_stream_id = f"chatcmpl-{uuid.uuid4()}"
                            usage_chunk_data = {
                                "id": self._current_stream_id,
                                "object": "chat.completion.chunk",
                                "model": last_model_seen or "unknown",
                                "choices": [],
                                "usage": {
                                    "prompt_tokens": cost_data.get("input_tokens", 0),
                                    "completion_tokens": cost_data.get(
                                        "output_tokens", 0
                                    ),
                                    "total_tokens": cost_data.get("input_tokens", 0)
                                    + cost_data.get("output_tokens", 0),
                                },
                            }

                        try:
                            self.inject_cost_metadata(
                                usage_chunk_data, cost_data, fresh_key
                            )
                        except Exception:
                            logger.exception(
                                "Failed to inject cost metadata into streaming chunk",
                                extra={
                                    "key_hash": key.hashed_key[:8] + "...",
                                },
                            )

                        yield f"data: {json.dumps(usage_chunk_data)}\n\n".encode()

                if done_seen:
                    yield b"data: [DONE]\n\n"

            except Exception as stream_error:
                logger.warning(
                    "Streaming interrupted; finalizing in background",
                    extra={
                        "error": str(stream_error),
                        "key_hash": key.hashed_key[:8] + "...",
                    },
                )
                raise
            finally:
                if not usage_finalized:
                    # Create a background task to ensure finalization happens
                    # even if the generator is closed early
                    background_tasks.add_task(finalize_db_only)

        # Remove inaccurate encoding headers from upstream response
        response_headers = dict(response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)

        return StreamingResponse(
            stream_with_cost(max_cost_for_model),
            status_code=response.status_code,
            headers=response_headers,
        )

    async def handle_non_streaming_chat_completion(
        self,
        response: httpx.Response,
        key: ApiKey,
        session: AsyncSession,
        deducted_max_cost: int,
        requested_model: str | None = None,
    ) -> Response:
        """Handle non-streaming chat completion responses with token usage tracking and cost adjustment.

        Args:
            response: Response from upstream
            key: API key for the authenticated user
            session: Database session for updating balance
            deducted_max_cost: Maximum cost deducted upfront

        Returns:
            Response with cost data added to JSON body
        """
        logger.debug(
            "Processing non-streaming chat completion",
            extra={
                "key_hash": key.hashed_key[:8] + "...",
                "key_balance": key.balance,
                "response_status": response.status_code,
            },
        )

        content: bytes | None = None
        try:
            content = await response.aread()
            response_json = json.loads(content)
            self._apply_provider_field(response_json)

            logger.debug(
                "Parsed response JSON",
                extra={
                    "key_hash": key.hashed_key[:8] + "...",
                    "model": response_json.get("model", "unknown"),
                    "has_usage": "usage" in response_json,
                },
            )

            if requested_model:
                response_json["model"] = requested_model
            if "id" not in response_json or not isinstance(response_json["id"], str):
                response_json["id"] = f"chatcmpl-{uuid.uuid4()}"

            cost_data = await adjust_payment_for_tokens(
                key, response_json, session, deducted_max_cost
            )

            await session.refresh(key)
            remaining_balance_msats = key.balance

            # Merge cost into usage for OpenCode
            if "usage" in response_json:
                response_json["usage"]["cost"] = cost_data.get("total_usd", 0.0)
                response_json["usage"]["cost_sats"] = (
                    cost_data.get("total_msats", 0) // 1000
                )
                response_json["usage"]["remaining_balance_msats"] = (
                    remaining_balance_msats
                )
                self._fold_cache_into_input_tokens(response_json["usage"])

            # Keep detailed cost
            response_json["metadata"] = response_json.get("metadata", {})
            response_json["metadata"]["routstr"] = {"cost": cost_data}
            response_json["metadata"]["routstr"]["cost"]["sats_cost"] = (
                cost_data.get("total_msats", 0) // 1000
            )
            response_json["metadata"]["routstr"]["cost"]["remaining_balance_msats"] = (
                remaining_balance_msats
            )
            response_json["cost"] = cost_data
            response_json["cost"]["sats_cost"] = cost_data.get("total_msats", 0) // 1000
            response_json["cost"]["remaining_balance_msats"] = remaining_balance_msats

            logger.debug(
                "Payment adjustment completed for non-streaming",
                extra={
                    "key_hash": key.hashed_key[:8] + "...",
                    "cost_data": cost_data,
                    "model": response_json.get("model", "unknown"),
                    "balance_after_adjustment": key.balance,
                },
            )

            allowed_headers = {
                "content-type",
                "cache-control",
                "date",
                "vary",
                "access-control-allow-origin",
                "access-control-allow-methods",
                "access-control-allow-headers",
                "access-control-allow-credentials",
                "access-control-expose-headers",
                "access-control-max-age",
            }

            response_headers = {
                k: v
                for k, v in response.headers.items()
                if k.lower() in allowed_headers
            }

            if requested_model:
                response_json["model"] = requested_model
            return Response(
                content=json.dumps(response_json).encode(),
                status_code=response.status_code,
                headers=response_headers,
                media_type="application/json",
            )
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse JSON from upstream response",
                extra={
                    "error": str(e),
                    "key_hash": key.hashed_key[:8] + "...",
                    "content_preview": content[:200].decode(errors="ignore")
                    if content
                    else "empty",
                },
            )
            raise
        except Exception as e:
            logger.error(
                "Error processing non-streaming chat completion",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "key_hash": key.hashed_key[:8] + "...",
                },
            )
            raise

    async def handle_streaming_responses_completion(
        self,
        response: httpx.Response,
        key: ApiKey,
        max_cost_for_model: int,
        requested_model: str | None = None,
    ) -> StreamingResponse:
        """Handle streaming Responses API responses with token usage tracking and cost adjustment.

        Args:
            response: Streaming response from upstream
            key: API key for the authenticated user
            max_cost_for_model: Maximum cost deducted upfront for the model

        Returns:
            StreamingResponse with cost data injected at the end
        """
        logger.debug(
            "Processing streaming Responses API completion",
            extra={
                "key_hash": key.hashed_key[:8] + "...",
                "key_balance": key.balance,
                "response_status": response.status_code,
            },
        )

        async def stream_with_responses_cost(
            max_cost_for_model: int,
        ) -> AsyncGenerator[bytes, None]:
            usage_finalized: bool = False
            last_model_seen: str | None = None
            reasoning_tokens: int = 0
            usage_chunk_data: dict | None = None
            done_seen: bool = False

            async def finalize_db_only() -> None:
                nonlocal usage_finalized
                if usage_finalized:
                    return
                async with create_session() as new_session:
                    fresh_key = await new_session.get(key.__class__, key.hashed_key)
                    if not fresh_key:
                        return
                    try:
                        await adjust_payment_for_tokens(
                            fresh_key,
                            {"model": last_model_seen or "unknown", "usage": None},
                            new_session,
                            max_cost_for_model,
                        )
                        usage_finalized = True
                    except Exception:
                        pass

            try:
                async for chunk in response.aiter_bytes():
                    # Split chunk into SSE events
                    parts = re.split(b"data: ", chunk)
                    for i, part in enumerate(parts):
                        if not part:
                            continue

                        stripped_part = part.strip()
                        if not stripped_part:
                            continue

                        if stripped_part == b"[DONE]":
                            done_seen = True
                            continue

                        try:
                            obj = json.loads(part)
                            if isinstance(obj, dict):
                                self._apply_provider_field(obj)
                                if obj.get("model"):
                                    last_model_seen = str(obj.get("model"))
                                if requested_model:
                                    obj["model"] = requested_model

                                # Track reasoning tokens for Responses API
                                if usage := obj.get("usage", {}):
                                    if (
                                        isinstance(usage, dict)
                                        and "reasoning_tokens" in usage
                                    ):
                                        reasoning_tokens += usage.get(
                                            "reasoning_tokens", 0
                                        )

                                # Responses API usage is in response.completed/incomplete events
                                chunk_type = obj.get("type", "")
                                if chunk_type in (
                                    "response.completed",
                                    "response.incomplete",
                                ):
                                    usage_chunk_data = obj
                                    continue
                        except json.JSONDecodeError:
                            pass

                        prefix = (
                            b"data: " if (i > 0 or chunk.startswith(b"data: ")) else b""
                        )
                        yield prefix + part

                # Always emit a cost-bearing data chunk
                async with create_session() as session:
                    fresh_key = await session.get(key.__class__, key.hashed_key)
                    if fresh_key:
                        cost_data: dict
                        try:
                            adjustment_input = (
                                usage_chunk_data
                                if usage_chunk_data is not None
                                else {
                                    "model": last_model_seen or "unknown",
                                    "usage": None,
                                }
                            )
                            cost_data = await adjust_payment_for_tokens(
                                fresh_key,
                                adjustment_input,
                                session,
                                max_cost_for_model,
                            )
                            usage_finalized = True
                        except Exception as e:
                            logger.exception(
                                "Error during Responses API usage finalization",
                                extra={
                                    "key_hash": key.hashed_key[:8] + "...",
                                    "error": str(e),
                                },
                            )
                            cost_data = {
                                "base_msats": 0,
                                "input_msats": 0,
                                "output_msats": 0,
                                "total_msats": 0,
                                "total_usd": 0.0,
                                "input_tokens": 0,
                                "output_tokens": 0,
                            }

                        if usage_chunk_data is None:
                            usage_chunk_data = {
                                "type": "response.completed",
                                "response": {
                                    "model": last_model_seen or "unknown",
                                    "usage": {
                                        "input_tokens": cost_data.get(
                                            "input_tokens", 0
                                        ),
                                        "output_tokens": cost_data.get(
                                            "output_tokens", 0
                                        ),
                                        "total_tokens": cost_data.get("input_tokens", 0)
                                        + cost_data.get("output_tokens", 0),
                                    },
                                },
                                "usage": {
                                    "input_tokens": cost_data.get("input_tokens", 0),
                                    "output_tokens": cost_data.get("output_tokens", 0),
                                    "total_tokens": cost_data.get("input_tokens", 0)
                                    + cost_data.get("output_tokens", 0),
                                },
                            }

                        remaining_balance_msats = fresh_key.balance
                        sats_cost = cost_data.get("total_msats", 0) // 1000

                        if (
                            "response" in usage_chunk_data
                            and isinstance(usage_chunk_data["response"], dict)
                            and "usage" in usage_chunk_data["response"]
                        ):
                            usage_chunk_data["response"]["usage"]["cost"] = (
                                cost_data.get("total_usd", 0.0)
                            )
                            usage_chunk_data["response"]["usage"]["cost_sats"] = (
                                sats_cost
                            )
                            usage_chunk_data["response"]["usage"][
                                "remaining_balance_msats"
                            ] = remaining_balance_msats

                        try:
                            self.inject_cost_metadata(
                                usage_chunk_data, cost_data, fresh_key
                            )
                        except Exception:
                            logger.exception(
                                "Failed to inject cost metadata into Responses streaming chunk",
                                extra={
                                    "key_hash": key.hashed_key[:8] + "...",
                                },
                            )

                        yield f"data: {json.dumps(usage_chunk_data)}\n\n".encode()

                if done_seen:
                    yield b"data: [DONE]\n\n"

            except Exception as stream_error:
                logger.warning(
                    "Responses API streaming interrupted; finalizing in background",
                    extra={
                        "error": str(stream_error),
                        "key_hash": key.hashed_key[:8] + "...",
                    },
                )
                raise
            finally:
                if not usage_finalized:
                    await finalize_db_only()

        # Remove inaccurate encoding headers from upstream response
        response_headers = dict(response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)

        return StreamingResponse(
            stream_with_responses_cost(max_cost_for_model),
            status_code=response.status_code,
            headers=response_headers,
        )

    async def handle_non_streaming_responses_completion(
        self,
        response: httpx.Response,
        key: ApiKey,
        session: AsyncSession,
        deducted_max_cost: int,
        requested_model: str | None = None,
    ) -> Response:
        """Handle non-streaming Responses API responses with token usage tracking and cost adjustment.

        Args:
            response: Response from upstream
            key: API key for the authenticated user
            session: Database session for updating balance
            deducted_max_cost: Maximum cost deducted upfront

        Returns:
            Response with cost data added to JSON body
        """
        logger.debug(
            "Processing non-streaming Responses API completion",
            extra={
                "key_hash": key.hashed_key[:8] + "...",
                "key_balance": key.balance,
                "response_status": response.status_code,
            },
        )

        content: bytes | None = None
        try:
            content = await response.aread()
            response_json = json.loads(content)
            self._apply_provider_field(response_json)

            logger.debug(
                "Parsed Responses API response JSON",
                extra={
                    "key_hash": key.hashed_key[:8] + "...",
                    "model": response_json.get("model", "unknown"),
                    "has_usage": "usage" in response_json,
                    "has_reasoning_tokens": "usage" in response_json
                    and isinstance(response_json.get("usage"), dict)
                    and "reasoning_tokens" in response_json["usage"],
                },
            )

            if requested_model:
                response_json["model"] = requested_model
            if "id" not in response_json or not isinstance(response_json["id"], str):
                response_json["id"] = f"chatcmpl-{uuid.uuid4()}"

            cost_data = await adjust_payment_for_tokens(
                key, response_json, session, deducted_max_cost
            )

            await session.refresh(key)
            remaining_balance_msats = key.balance

            # Merge cost into usage for OpenCode
            if "usage" in response_json:
                response_json["usage"]["cost"] = cost_data.get("total_usd", 0.0)
                response_json["usage"]["cost_sats"] = (
                    cost_data.get("total_msats", 0) // 1000
                )
                response_json["usage"]["remaining_balance_msats"] = (
                    remaining_balance_msats
                )
                self._fold_cache_into_input_tokens(response_json["usage"])

            # Keep detailed cost
            response_json["metadata"] = response_json.get("metadata", {})
            response_json["metadata"]["routstr"] = {"cost": cost_data}
            response_json["metadata"]["routstr"]["cost"]["sats_cost"] = (
                cost_data.get("total_msats", 0) // 1000
            )
            response_json["metadata"]["routstr"]["cost"]["remaining_balance_msats"] = (
                remaining_balance_msats
            )
            response_json["cost"] = cost_data
            response_json["cost"]["sats_cost"] = cost_data.get("total_msats", 0) // 1000
            response_json["cost"]["remaining_balance_msats"] = remaining_balance_msats

            logger.debug(
                "Payment adjustment completed for non-streaming Responses API",
                extra={
                    "key_hash": key.hashed_key[:8] + "...",
                    "cost_data": cost_data,
                    "model": response_json.get("model", "unknown"),
                    "balance_after_adjustment": key.balance,
                },
            )

            allowed_headers = {
                "content-type",
                "cache-control",
                "date",
                "vary",
                "access-control-allow-origin",
                "access-control-allow-methods",
                "access-control-allow-headers",
                "access-control-allow-credentials",
                "access-control-expose-headers",
                "access-control-max-age",
            }

            response_headers = {
                k: v
                for k, v in response.headers.items()
                if k.lower() in allowed_headers
            }

            if requested_model:
                response_json["model"] = requested_model
            return Response(
                content=json.dumps(response_json).encode(),
                status_code=response.status_code,
                headers=response_headers,
                media_type="application/json",
            )
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse JSON from upstream Responses API response",
                extra={
                    "error": str(e),
                    "key_hash": key.hashed_key[:8] + "...",
                    "content_preview": content[:200].decode(errors="ignore")
                    if content
                    else "empty",
                },
            )
            raise
        except Exception as e:
            logger.error(
                "Error processing non-streaming Responses API completion",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "key_hash": key.hashed_key[:8] + "...",
                },
            )
            raise

    async def _finalize_generic_streaming_payment(
        self, key_hash: str, max_cost: int, path: str
    ) -> None:
        """Background task to finalize payment for generic streaming requests."""
        async with create_session() as session:
            key = await session.get(ApiKey, key_hash)
            if not key:
                logger.warning(
                    "Key not found during background payment finalization",
                    extra={"key_hash": key_hash[:8] + "..."},
                )
                return

            try:
                # Finalize with "unknown" model and no usage to release reservation/charge max cost
                await adjust_payment_for_tokens(
                    key,
                    {"model": "unknown", "usage": None},
                    session,
                    max_cost,
                )
                logger.debug(
                    "Finalized generic streaming payment in background",
                    extra={
                        "path": path,
                        "key_hash": key_hash[:8] + "...",
                    },
                )
            except Exception as e:
                logger.error(
                    "Error finalizing generic streaming payment in background",
                    extra={
                        "error": str(e),
                        "key_hash": key_hash[:8] + "...",
                        "path": path,
                    },
                )

    async def handle_streaming_messages_completion(
        self,
        response: httpx.Response,
        key: ApiKey,
        max_cost_for_model: int,
        requested_model: str | None = None,
    ) -> StreamingResponse:
        async def stream_with_cost(
            max_cost_for_model: int,
        ) -> AsyncGenerator[bytes, None]:
            stored_chunks: list[bytes] = []
            usage_finalized: bool = False
            last_model_seen: str | None = None
            input_tokens: int = 0
            output_tokens: int = 0
            cache_read_input_tokens: int = 0
            cache_creation_input_tokens: int = 0
            total_cost: float = 0.0
            input_cost: float = 0.0
            output_cost: float = 0.0

            def _coerce_usd(value: object) -> float:
                if value is None or isinstance(value, bool):
                    return 0.0
                if not isinstance(value, (int, float, str)):
                    return 0.0
                try:
                    return max(0.0, float(value))
                except (TypeError, ValueError):
                    return 0.0

            def _absorb_usd(usage_or_root: dict) -> None:
                nonlocal total_cost, input_cost, output_cost
                cd = usage_or_root.get("cost_details")
                if isinstance(cd, dict):
                    total_cost = max(
                        total_cost,
                        _coerce_usd(cd.get("total_cost")),
                    )
                    input_cost = max(
                        input_cost,
                        _coerce_usd(cd.get("input_cost")),
                    )
                    output_cost = max(
                        output_cost,
                        _coerce_usd(cd.get("output_cost")),
                    )
                for field in ("total_cost", "cost"):
                    total_cost = max(total_cost, _coerce_usd(usage_or_root.get(field)))

            async def finalize_without_usage() -> bytes | None:
                nonlocal usage_finalized
                if usage_finalized:
                    return None
                async with create_session() as new_session:
                    fresh_key = await new_session.get(key.__class__, key.hashed_key)
                    if not fresh_key:
                        usage_finalized = True
                        return None
                    try:
                        fallback: dict = {
                            "model": last_model_seen or "unknown",
                            "usage": None,
                        }
                        cost_data = await adjust_payment_for_tokens(
                            fresh_key, fallback, new_session, max_cost_for_model
                        )
                        usage_finalized = True
                        return f"event: cost\ndata: {json.dumps({'cost': cost_data})}\n\n".encode()
                    except Exception:
                        usage_finalized = True
                        return None

            try:
                async for chunk in response.aiter_bytes():
                    stored_chunks.append(chunk)
                    try:
                        decoded_chunk = chunk.decode("utf-8", errors="ignore")
                        modified_lines = []
                        changed = False
                        for line in decoded_chunk.split("\n"):
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    if isinstance(data, dict):
                                        msg = data.get("message", {})
                                        if msg and msg.get("model"):
                                            last_model_seen = str(msg.get("model"))

                                        provider_added = "provider" not in data
                                        self._apply_provider_field(data)

                                        if requested_model:
                                            # Apply requested_model override
                                            model_updated = False
                                            if msg:
                                                msg["model"] = requested_model
                                                model_updated = True
                                            if data.get("model"):
                                                data["model"] = requested_model
                                                model_updated = True

                                            if model_updated or provider_added:
                                                line = "data: " + json.dumps(data)
                                                changed = True
                                        elif provider_added:
                                            line = "data: " + json.dumps(data)
                                            changed = True

                                        if usage := msg.get("usage"):
                                            input_tokens += usage.get("input_tokens", 0)
                                            output_tokens += usage.get(
                                                "output_tokens", 0
                                            )
                                            # Anthropic's `message_start.usage`
                                            # carries the cumulative cache
                                            # snapshot for the prompt — pick
                                            # the max() so subsequent
                                            # `message_delta.usage` events
                                            # (which only restate the same
                                            # numbers) don't double-count.
                                            cache_read_input_tokens = max(
                                                cache_read_input_tokens,
                                                int(
                                                    usage.get(
                                                        "cache_read_input_tokens", 0
                                                    )
                                                    or 0
                                                ),
                                            )
                                            cache_creation_input_tokens = max(
                                                cache_creation_input_tokens,
                                                int(
                                                    usage.get(
                                                        "cache_creation_input_tokens",
                                                        0,
                                                    )
                                                    or 0
                                                ),
                                            )
                                            _absorb_usd(usage)

                                        if usage := data.get("usage"):
                                            input_tokens += usage.get("input_tokens", 0)
                                            output_tokens += usage.get(
                                                "output_tokens", 0
                                            )
                                            cache_read_input_tokens = max(
                                                cache_read_input_tokens,
                                                int(
                                                    usage.get(
                                                        "cache_read_input_tokens", 0
                                                    )
                                                    or 0
                                                ),
                                            )
                                            cache_creation_input_tokens = max(
                                                cache_creation_input_tokens,
                                                int(
                                                    usage.get(
                                                        "cache_creation_input_tokens",
                                                        0,
                                                    )
                                                    or 0
                                                ),
                                            )
                                            _absorb_usd(usage)
                                        # Some upstreams attach cost fields at
                                        # the event root rather than nested
                                        # under `usage`.
                                        _absorb_usd(data)
                                except json.JSONDecodeError:
                                    pass
                            modified_lines.append(line)

                        if changed:
                            yield "\n".join(modified_lines).encode("utf-8")
                        else:
                            yield chunk
                    except Exception:
                        yield chunk

                usage_data = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_input_tokens": cache_read_input_tokens,
                    "cache_creation_input_tokens": cache_creation_input_tokens,
                }
                messages_dispatch.embed_usd_costs(
                    usage_data,
                    total_cost,
                    input_cost,
                    output_cost,
                )

                if (
                    input_tokens > 0
                    or output_tokens > 0
                    or cache_read_input_tokens > 0
                    or cache_creation_input_tokens > 0
                    or total_cost > 0
                ):
                    async with create_session() as new_session:
                        fresh_key = await new_session.get(key.__class__, key.hashed_key)
                        if fresh_key:
                            try:
                                combined_data = {
                                    "model": last_model_seen or "unknown",
                                    "usage": usage_data,
                                }
                                cost_data = await adjust_payment_for_tokens(
                                    fresh_key,
                                    combined_data,
                                    new_session,
                                    max_cost_for_model,
                                )

                                self.inject_cost_metadata(
                                    combined_data, cost_data, fresh_key
                                )

                                usage_finalized = True
                                # Emit the full combined_data as the cost
                                yield f"event: cost\ndata: {json.dumps(combined_data)}\n\n".encode()
                            except Exception:
                                pass

                if not usage_finalized:
                    maybe_cost_event = await finalize_without_usage()
                    if maybe_cost_event is not None:
                        yield maybe_cost_event

            except httpx.ReadError:
                if not usage_finalized:
                    await finalize_without_usage()
                # Upstream dropped the connection mid-stream; response already started, swallow silently
            except Exception:
                if not usage_finalized:
                    await finalize_without_usage()
                raise
            finally:
                if not usage_finalized:
                    await finalize_without_usage()

        response_headers = dict(response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)

        return StreamingResponse(
            stream_with_cost(max_cost_for_model),
            status_code=response.status_code,
            headers=response_headers,
        )

    async def handle_non_streaming_messages_completion(
        self,
        response: httpx.Response,
        key: ApiKey,
        session: AsyncSession,
        deducted_max_cost: int,
        path: str,
        requested_model: str | None = None,
    ) -> Response:
        try:
            content = await response.aread()
            response_json = json.loads(content)

            if requested_model:
                if "model" in response_json:
                    response_json["model"] = requested_model
                if (
                    "message" in response_json
                    and isinstance(response_json["message"], dict)
                    and "model" in response_json["message"]
                ):
                    response_json["message"]["model"] = requested_model

            if path.endswith("count_tokens") and "usage" not in response_json:
                input_tokens = response_json.get("input_tokens", 0)
                response_json["usage"] = {"input_tokens": input_tokens}

            cost_data = await adjust_payment_for_tokens(
                key, response_json, session, deducted_max_cost
            )

            self.inject_cost_metadata(response_json, cost_data, key)

            allowed_headers = {
                "content-type",
                "cache-control",
                "date",
                "vary",
                "access-control-allow-origin",
                "access-control-allow-methods",
                "access-control-allow-headers",
                "access-control-allow-credentials",
                "access-control-expose-headers",
                "access-control-max-age",
            }

            response_headers = {
                k: v
                for k, v in response.headers.items()
                if k.lower() in allowed_headers
            }

            return Response(
                content=json.dumps(response_json).encode(),
                status_code=response.status_code,
                headers=response_headers,
                media_type="application/json",
            )
        except Exception:
            raise

    # ------------------------------------------------------------------
    # Litellm /v1/messages dispatch (thin wrappers)
    #
    # The actual translation logic lives in ``messages_dispatch``. These
    # method shims exist so subclasses and tests can keep the original
    # provider-bound API.
    # ------------------------------------------------------------------

    _coerce_litellm_payload = staticmethod(messages_dispatch.coerce_litellm_payload)
    _parse_sse_blocks = staticmethod(messages_dispatch.parse_sse_blocks)
    _events_from_chunk = staticmethod(messages_dispatch.events_from_chunk)

    async def _aggregate_anthropic_events_to_message(
        self, iterator: AsyncIterator[Any]
    ) -> dict:
        return await messages_dispatch.aggregate_anthropic_events_to_message(iterator)

    async def _dispatch_anthropic_messages(
        self,
        request_body: bytes | None,
        model_obj: Model,
        *,
        log_extra: dict[str, Any] | None = None,
    ) -> tuple[bool, Any, str | None]:
        return await messages_dispatch.dispatch_anthropic_messages(
            request_body=request_body,
            model_obj=model_obj,
            base_url=self.base_url,
            api_key=self.api_key,
            provider_prefix=self.get_litellm_provider_prefix(),
            transform_model_name=self.transform_model_name,
            log_extra=log_extra,
        )

    async def _forward_messages_via_litellm(
        self,
        request_body: bytes | None,
        key: ApiKey,
        session: AsyncSession,
        max_cost_for_model: int,
        model_obj: Model,
    ) -> Response | StreamingResponse:
        """Translate /v1/messages to upstream chat/completions via litellm.

        Used when the upstream provider does not natively serve Anthropic
        Messages (i.e. supports_anthropic_messages is False). Cost
        tracking and metadata injection mirror the native messages path.
        """
        stream, result, requested_model = await self._dispatch_anthropic_messages(
            request_body,
            model_obj,
            log_extra={
                "key_hash": key.hashed_key[:8] + "...",
                "confidential": self.confidential_logging_enabled(),
            },
        )

        if stream:
            return self._stream_litellm_messages(
                cast(AsyncIterator[Any], result),
                key,
                max_cost_for_model,
                requested_model,
            )

        response_json = messages_dispatch.coerce_litellm_payload(result)
        if requested_model and "model" in response_json:
            response_json["model"] = requested_model

        cost_data = await adjust_payment_for_tokens(
            key, response_json, session, max_cost_for_model
        )
        self.inject_cost_metadata(response_json, cost_data, key)

        return Response(
            content=json.dumps(response_json).encode(),
            status_code=200,
            media_type="application/json",
        )

    async def _forward_x_cashu_messages_via_litellm(
        self,
        request_body: bytes,
        amount: int,
        unit: str,
        max_cost_for_model: int,
        model_obj: Model,
        mint: str | None = None,
        request_id: str | None = None,
    ) -> Response | StreamingResponse:
        """Dispatch /v1/messages via litellm for x-cashu payments.

        Computes cost from upstream usage, refunds the unspent balance via
        an X-Cashu response header, and returns the Anthropic-shaped body.
        """
        stream, result, requested_model = await self._dispatch_anthropic_messages(
            request_body,
            model_obj,
            log_extra={
                "payment_unit": unit,
                "payment_amount": amount,
                "confidential": self.confidential_logging_enabled(),
            },
        )

        if stream:
            return await self._stream_x_cashu_litellm_messages(
                cast(AsyncIterator[Any], result),
                amount,
                unit,
                max_cost_for_model,
                requested_model,
                mint,
                request_id,
            )

        response_json = messages_dispatch.coerce_litellm_payload(result)
        self._apply_provider_field(response_json)
        if requested_model and "model" in response_json:
            response_json["model"] = requested_model

        cost_data = await self.get_x_cashu_cost(response_json, max_cost_for_model)

        if (
            cost_data
            and "usage" in response_json
            and isinstance(response_json["usage"], dict)
        ):
            response_json["usage"]["cost_sats"] = cost_data.total_msats // 1000
            self._fold_cache_into_input_tokens(response_json["usage"])

        response_headers: dict[str, str] = {}
        if cost_data:
            refund_amount = messages_dispatch.compute_refund(
                amount, unit, cost_data.total_msats
            )
            if refund_amount > 0:
                refund_token = await self.send_refund(
                    refund_amount,
                    unit,
                    mint,
                    request_id=request_id,
                )
                response_headers["X-Cashu"] = refund_token
                logger.info(
                    "Refund processed for non-streaming /v1/messages via litellm",
                    extra={
                        "refund_amount": refund_amount,
                        "unit": unit,
                        "model": response_json.get("model", "unknown"),
                    },
                )

        return Response(
            content=json.dumps(response_json).encode(),
            status_code=200,
            headers=response_headers,
            media_type="application/json",
        )

    _compute_refund = staticmethod(messages_dispatch.compute_refund)

    def _stream_litellm_messages(
        self,
        iterator: AsyncIterator[Any],
        key: ApiKey,
        max_cost_for_model: int,
        requested_model: str | None,
    ) -> StreamingResponse:
        """Re-emit a litellm Anthropic-event iterator as live SSE bytes
        with cost reconciliation appended at end of stream."""

        async def stream_with_cost() -> AsyncGenerator[bytes, None]:
            usage_finalized = False
            last_model_seen: str | None = None
            input_tokens = 0
            output_tokens = 0
            cache_read_input_tokens = 0
            cache_creation_input_tokens = 0
            total_cost = 0.0
            input_cost = 0.0
            output_cost = 0.0

            async def finalize_without_usage() -> bytes | None:
                nonlocal usage_finalized
                if usage_finalized:
                    return None
                logger.warning(
                    "Finalizing /v1/messages stream with no usage data — "
                    "client will be billed at max-cost with zero tokens. "
                    "Likely cause: upstream omitted `usage` from the SSE "
                    "stream (check that the request includes "
                    "`stream_options.include_usage=true` and that the "
                    "upstream actually emits a final usage chunk).",
                    extra={
                        "key_hash": key.hashed_key[:8] + "...",
                        "model": last_model_seen or "unknown",
                        "provider": self.provider_type or self.base_url,
                        "max_cost_msats": max_cost_for_model,
                    },
                )
                async with create_session() as new_session:
                    fresh_key = await new_session.get(key.__class__, key.hashed_key)
                    if not fresh_key:
                        usage_finalized = True
                        return None
                    try:
                        fallback: dict = {
                            "model": last_model_seen or "unknown",
                            "usage": None,
                        }
                        cost_data = await adjust_payment_for_tokens(
                            fresh_key,
                            fallback,
                            new_session,
                            max_cost_for_model,
                        )
                        usage_finalized = True
                        return (
                            f"event: cost\ndata: {json.dumps({'cost': cost_data})}\n\n"
                        ).encode()
                    except Exception:
                        usage_finalized = True
                        return None

            try:
                async for annotated in messages_dispatch.stream_annotated_events(
                    iterator, requested_model
                ):
                    if annotated.model:
                        last_model_seen = annotated.model
                    # Anthropic SSE reports usage cumulatively across
                    # message_start + message_delta — take the max snapshot
                    # rather than summing, otherwise input tokens
                    # double-count.
                    input_tokens = max(input_tokens, annotated.input_tokens)
                    output_tokens = max(output_tokens, annotated.output_tokens)
                    cache_read_input_tokens = max(
                        cache_read_input_tokens,
                        annotated.cache_read_input_tokens,
                    )
                    cache_creation_input_tokens = max(
                        cache_creation_input_tokens,
                        annotated.cache_creation_input_tokens,
                    )
                    total_cost = max(total_cost, annotated.total_cost)
                    input_cost = max(input_cost, annotated.input_cost)
                    output_cost = max(output_cost, annotated.output_cost)
                    yield annotated.sse_bytes

                if (
                    input_tokens > 0
                    or output_tokens > 0
                    or cache_read_input_tokens > 0
                    or cache_creation_input_tokens > 0
                    or total_cost > 0
                ):
                    async with create_session() as new_session:
                        fresh_key = await new_session.get(key.__class__, key.hashed_key)
                        if fresh_key:
                            try:
                                rebuilt_usage: dict = {
                                    "input_tokens": input_tokens,
                                    "output_tokens": output_tokens,
                                    "cache_read_input_tokens": (
                                        cache_read_input_tokens
                                    ),
                                    "cache_creation_input_tokens": (
                                        cache_creation_input_tokens
                                    ),
                                }
                                messages_dispatch.embed_usd_costs(
                                    rebuilt_usage,
                                    total_cost,
                                    input_cost,
                                    output_cost,
                                )
                                combined_data: dict = {
                                    "model": last_model_seen or "unknown",
                                    "usage": rebuilt_usage,
                                }
                                cost_data = await adjust_payment_for_tokens(
                                    fresh_key,
                                    combined_data,
                                    new_session,
                                    max_cost_for_model,
                                )
                                self.inject_cost_metadata(
                                    combined_data, cost_data, fresh_key
                                )
                                usage_finalized = True
                                yield (
                                    f"event: cost\ndata: "
                                    f"{json.dumps({'cost': cost_data})}\n\n"
                                ).encode()
                            except Exception:
                                pass

                if not usage_finalized:
                    cost_event = await finalize_without_usage()
                    if cost_event is not None:
                        yield cost_event

            except Exception:
                if not usage_finalized:
                    await finalize_without_usage()
                raise

        return StreamingResponse(
            stream_with_cost(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def _stream_x_cashu_litellm_messages(
        self,
        iterator: AsyncIterator[Any],
        amount: int,
        unit: str,
        max_cost_for_model: int,
        requested_model: str | None,
        mint: str | None,
        request_id: str | None,
    ) -> StreamingResponse:
        """Buffer a litellm stream end-to-end, compute cost, then replay.

        Note this is **not** true streaming — the full event sequence is
        accumulated into memory before a single byte is sent to the
        client. The constraint is the ``X-Cashu`` refund token, which must
        be set as a response *header* and therefore has to be known before
        the response begins. The bearer-key path
        (:meth:`_stream_litellm_messages`) avoids this by emitting cost as
        a trailing ``event: cost`` SSE message; switching x-cashu to the
        same trailing-event contract would let this path stream live, at
        the cost of a wire-format change for clients that read ``X-Cashu``
        from headers today.
        """
        buffered: list[bytes] = []
        last_model_seen: str | None = None
        input_tokens = 0
        output_tokens = 0
        cache_read_input_tokens = 0
        cache_creation_input_tokens = 0
        total_cost = 0.0
        input_cost = 0.0
        output_cost = 0.0

        async for annotated in messages_dispatch.stream_annotated_events(
            iterator, requested_model
        ):
            if annotated.model:
                last_model_seen = annotated.model
            # See _stream_litellm_messages for why this is max() not +=.
            input_tokens = max(input_tokens, annotated.input_tokens)
            output_tokens = max(output_tokens, annotated.output_tokens)
            cache_read_input_tokens = max(
                cache_read_input_tokens, annotated.cache_read_input_tokens
            )
            cache_creation_input_tokens = max(
                cache_creation_input_tokens,
                annotated.cache_creation_input_tokens,
            )
            total_cost = max(total_cost, annotated.total_cost)
            input_cost = max(input_cost, annotated.input_cost)
            output_cost = max(output_cost, annotated.output_cost)
            buffered.append(annotated.sse_bytes)

        response_headers: dict[str, str] = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }

        if (
            input_tokens == 0
            and output_tokens == 0
            and cache_read_input_tokens == 0
            and cache_creation_input_tokens == 0
            and total_cost == 0
        ):
            logger.warning(
                "x-cashu /v1/messages stream finished with no usage data "
                "— refund cannot be computed and the client effectively "
                "pays the full cashu amount. Likely cause: upstream "
                "omitted `usage` from the SSE stream.",
                extra={
                    "model": last_model_seen or "unknown",
                    "provider": self.provider_type or self.base_url,
                    "amount": amount,
                    "unit": unit,
                },
            )

        if (
            input_tokens > 0
            or output_tokens > 0
            or cache_read_input_tokens > 0
            or cache_creation_input_tokens > 0
            or total_cost > 0
        ):
            rebuilt_usage: dict = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read_input_tokens,
                "cache_creation_input_tokens": cache_creation_input_tokens,
            }
            messages_dispatch.embed_usd_costs(
                rebuilt_usage, total_cost, input_cost, output_cost
            )
            response_data: dict = {
                "model": last_model_seen or "unknown",
                "usage": rebuilt_usage,
            }
            try:
                cost_data = await self.get_x_cashu_cost(
                    response_data, max_cost_for_model
                )
                if cost_data:
                    refund_amount = messages_dispatch.compute_refund(
                        amount, unit, cost_data.total_msats
                    )
                    if refund_amount > 0:
                        refund_token = await self.send_refund(
                            refund_amount,
                            unit,
                            mint,
                            request_id=request_id,
                        )
                        response_headers["X-Cashu"] = refund_token
                        logger.info(
                            "Refund processed for streaming /v1/messages via litellm",
                            extra={
                                "refund_amount": refund_amount,
                                "unit": unit,
                                "model": last_model_seen,
                            },
                        )
            except Exception as exc:
                logger.error(
                    "Error calculating cost for streaming /v1/messages",
                    extra={
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "amount": amount,
                        "unit": unit,
                    },
                )

        async def replay() -> AsyncGenerator[bytes, None]:
            for chunk in buffered:
                yield chunk

        return StreamingResponse(
            replay(),
            media_type="text/event-stream",
            headers=response_headers,
        )

    async def forward_request(
        self,
        request: Request,
        path: str,
        headers: dict,
        request_body: bytes | None,
        key: ApiKey,
        max_cost_for_model: int,
        session: AsyncSession,
        model_obj: Model,
        *,
        route_alias: str | None = None,
    ) -> Response | StreamingResponse:
        """Forward authenticated request to upstream service with cost tracking.

        Args:
            request: Original FastAPI request
            path: Request path
            headers: Prepared headers for upstream
            request_body: Request body bytes, if any
            key: API key for authenticated user
            max_cost_for_model: Maximum cost deducted upfront
            session: Database session for balance updates

        Returns:
            Response or StreamingResponse from upstream with cost tracking
        """
        path = self.normalize_request_path(path, model_obj)
        self._require_supported_endpoint_for_path(path, model_obj)
        if self.requires_verified_confidential_transport:
            self._require_current_confidential_transport()

        requested_model_id = route_alias or model_obj.forwarded_model_id or model_obj.id
        self._require_confidential_transport_covers_model(
            model_obj,
            route_alias=requested_model_id,
        )

        if (
            path.endswith("messages/count_tokens")
            and not self.supports_anthropic_messages
        ):
            return count_tokens_locally(request_body, model_obj)

        if (
            path.endswith("messages")
            and not path.endswith("count_tokens")
            and not self.supports_anthropic_messages
        ):
            if self.requires_verified_ehbp_transport:
                raise UpstreamError(
                    "EHBP transport does not support /v1/messages translation yet",
                    status_code=424,
                )
            return await self._forward_messages_via_litellm(
                request_body=request_body,
                key=key,
                session=session,
                max_cost_for_model=max_cost_for_model,
                model_obj=model_obj,
            )

        url = self.build_request_url(path, model_obj)

        transformed_body = self.prepare_request_body(
            request_body, model_obj, path=path, headers=headers
        )
        self.prepare_request_headers(headers, path, request_body, model_obj)
        ehbp_request: EhbpEncryptedRequest | None = None
        outbound_body = (
            transformed_body if transformed_body is not None else request_body
        )
        if self.requires_verified_ehbp_transport:
            outbound_body, ehbp_request = self._prepare_ehbp_request_body(
                headers,
                outbound_body,
            )

        logger.debug(
            "Forwarding request to upstream",
            extra={
                "url": url,
                "method": request.method,
                "path": path,
                "model": requested_model_id,
                "provider": self.provider_type,
                "key_hash": key.hashed_key[:8] + "...",
            },
        )

        client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=1),
            timeout=None,
        )

        try:
            if outbound_body is not None:
                outbound_content: Any = (
                    _single_chunk_async_iter(outbound_body)
                    if ehbp_request is not None
                    else outbound_body
                )
                response = await client.send(
                    client.build_request(
                        request.method,
                        url,
                        headers=headers,
                        content=outbound_content,
                        params=self.prepare_params(path, request.query_params),
                    ),
                    stream=True,
                )
            else:
                response = await client.send(
                    client.build_request(
                        request.method,
                        url,
                        headers=headers,
                        content=request.stream(),
                        params=self.prepare_params(path, request.query_params),
                    ),
                    stream=True,
                )

            if ehbp_request is not None:
                try:
                    response = await self._decrypt_ehbp_response(response, ehbp_request)
                except UpstreamError:
                    await client.aclose()
                    raise

            if response.status_code != 200:
                if response.status_code >= 500:
                    try:
                        body_bytes = await response.aread()
                    except Exception:
                        body_bytes = b""
                    body_preview = body_bytes.decode("utf-8", errors="ignore").strip()[
                        :500
                    ]
                    redact_body = self.confidential_logging_enabled()
                    safe_body_preview = (
                        CONFIDENTIAL_REDACTION_TEXT
                        if redact_body and body_preview
                        else body_preview
                    )
                    logger.error(
                        "Upstream %s returned %s for model=%s path=%s: %s",
                        self.provider_type,
                        response.status_code,
                        requested_model_id,
                        path,
                        safe_body_preview or "<empty>",
                        extra={
                            "provider": self.provider_type,
                            "model": requested_model_id,
                            "status_code": response.status_code,
                            "reason_phrase": response.reason_phrase,
                            "path": path,
                            "body_preview": None if redact_body else body_preview,
                            "confidential": redact_body,
                        },
                    )
                    await response.aclose()
                    await client.aclose()
                    if redact_body:
                        raise UpstreamError(
                            f"Upstream {self.provider_type} returned {response.status_code} "
                            f"for model {requested_model_id}: "
                            "confidential upstream body redacted",
                            status_code=response.status_code,
                        )
                    raise UpstreamError(
                        f"Upstream {self.provider_type} returned {response.status_code} "
                        f"for model {requested_model_id}: "
                        f"{body_preview[:200] or '<empty>'}",
                        status_code=response.status_code,
                    )

                try:
                    mapped_error = await self.forward_upstream_error_response(
                        request, path, response, model_id=requested_model_id
                    )
                finally:
                    await response.aclose()
                    await client.aclose()
                return mapped_error

            if (
                path.endswith("chat/completions")
                or path.endswith("embeddings")
                or path.endswith("messages")
                or path.endswith("messages/count_tokens")
            ):
                if path.endswith("messages"):
                    client_wants_streaming = False
                    if request_body:
                        try:
                            request_data = json.loads(request_body)
                            client_wants_streaming = request_data.get("stream", False)
                        except json.JSONDecodeError:
                            pass

                    content_type = response.headers.get("content-type", "")
                    upstream_is_streaming = "text/event-stream" in content_type
                    is_streaming = client_wants_streaming and upstream_is_streaming

                    if is_streaming and response.status_code == 200:
                        result = await self.handle_streaming_messages_completion(
                            response,
                            key,
                            max_cost_for_model,
                            requested_model=requested_model_id,
                        )
                        background_tasks = BackgroundTasks()
                        background_tasks.add_task(response.aclose)
                        background_tasks.add_task(client.aclose)
                        result.background = background_tasks
                        return result

                    if response.status_code == 200:
                        try:
                            return await self.handle_non_streaming_messages_completion(
                                response,
                                key,
                                session,
                                max_cost_for_model,
                                path,
                                requested_model=requested_model_id,
                            )
                        finally:
                            await response.aclose()
                            await client.aclose()

                if path.endswith("messages/count_tokens"):
                    if response.status_code == 200:
                        try:
                            return await self.handle_non_streaming_messages_completion(
                                response,
                                key,
                                session,
                                max_cost_for_model,
                                path,
                                requested_model=requested_model_id,
                            )
                        finally:
                            await response.aclose()
                            await client.aclose()

                if path.endswith("chat/completions"):
                    client_wants_streaming = False
                    if request_body:
                        try:
                            request_data = json.loads(request_body)
                            client_wants_streaming = request_data.get("stream", False)
                            logger.debug(
                                "Chat completion request analysis",
                                extra={
                                    "client_wants_streaming": client_wants_streaming,
                                    "model": request_data.get("model", "unknown"),
                                    "key_hash": key.hashed_key[:8] + "...",
                                },
                            )
                        except json.JSONDecodeError:
                            logger.warning(
                                "Failed to parse request body JSON for streaming detection"
                            )

                    content_type = response.headers.get("content-type", "")
                    upstream_is_streaming = "text/event-stream" in content_type
                    is_streaming = client_wants_streaming and upstream_is_streaming

                    logger.debug(
                        "Response type analysis",
                        extra={
                            "is_streaming": is_streaming,
                            "client_wants_streaming": client_wants_streaming,
                            "upstream_is_streaming": upstream_is_streaming,
                            "content_type": content_type,
                            "key_hash": key.hashed_key[:8] + "...",
                        },
                    )

                    if is_streaming and response.status_code == 200:
                        background_tasks = BackgroundTasks()
                        background_tasks.add_task(response.aclose)
                        background_tasks.add_task(client.aclose)
                        result = await self.handle_streaming_chat_completion(
                            response,
                            key,
                            max_cost_for_model,
                            background_tasks,
                            requested_model=requested_model_id,
                        )
                        result.background = background_tasks
                        return result

                # Handle both non-streaming chat completions and embeddings
                if response.status_code == 200:
                    try:
                        return await self.handle_non_streaming_chat_completion(
                            response,
                            key,
                            session,
                            max_cost_for_model,
                            requested_model=requested_model_id,
                        )
                    finally:
                        await response.aclose()
                        await client.aclose()

            background_tasks = BackgroundTasks()
            background_tasks.add_task(response.aclose)
            background_tasks.add_task(client.aclose)
            background_tasks.add_task(
                self._finalize_generic_streaming_payment,
                key.hashed_key,
                max_cost_for_model,
                path,
            )

            logger.debug(
                "Streaming non-chat response",
                extra={
                    "path": path,
                    "status_code": response.status_code,
                    "key_hash": key.hashed_key[:8] + "...",
                },
            )

            return StreamingResponse(
                response.aiter_bytes(),
                status_code=response.status_code,
                headers=dict(response.headers),
                background=background_tasks,
            )

        except UpstreamError:
            raise

        except httpx.RequestError as exc:
            await client.aclose()
            error_type = type(exc).__name__
            redact_body = self.confidential_logging_enabled()
            error_details = _safe_diagnostic_text(exc, confidential=redact_body)

            logger.error(
                "HTTP request error to upstream",
                extra={
                    "error_type": error_type,
                    "error_details": error_details,
                    "method": request.method,
                    "url": url,
                    "path": path,
                    "query_params": dict(request.query_params),
                    "key_hash": key.hashed_key[:8] + "...",
                    "confidential": redact_body,
                },
            )

            # Don't revert here — proxy.py owns payment revert to avoid double-revert
            if isinstance(exc, httpx.ConnectError):
                error_message = "Unable to connect to upstream service"
            elif isinstance(exc, httpx.TimeoutException):
                error_message = "Upstream service request timed out"
            elif isinstance(exc, httpx.NetworkError):
                error_message = "Network error while connecting to upstream service"
            else:
                error_message = f"Error connecting to upstream service: {error_type}"

            raise UpstreamError(error_message, status_code=502)

        except Exception as exc:
            await client.aclose()
            redact_body = self.confidential_logging_enabled()
            safe_error = _safe_diagnostic_text(exc, confidential=redact_body)
            safe_traceback = _safe_diagnostic_text(
                traceback.format_exc(), confidential=redact_body
            )

            logger.error(
                "Unexpected error in upstream forwarding",
                extra={
                    "error": safe_error,
                    "error_type": type(exc).__name__,
                    "method": request.method,
                    "url": url,
                    "path": path,
                    "query_params": dict(request.query_params),
                    "key_hash": key.hashed_key[:8] + "...",
                    "traceback": safe_traceback,
                    "confidential": redact_body,
                },
            )

            # Don't revert here — proxy.py owns payment revert to avoid double-revert
            raise UpstreamError("An unexpected server error occurred", status_code=500)

    async def forward_responses_request(
        self,
        request: Request,
        path: str,
        headers: dict,
        request_body: bytes | None,
        key: ApiKey,
        max_cost_for_model: int,
        session: AsyncSession,
        model_obj: Model,
        *,
        route_alias: str | None = None,
    ) -> Response | StreamingResponse:
        """Forward authenticated Responses API request to upstream service with cost tracking.

        Args:
            request: Original FastAPI request
            path: Request path
            headers: Prepared headers for upstream
            request_body: Request body bytes, if any
            key: API key for authenticated user
            max_cost_for_model: Maximum cost deducted upfront
            session: Database session for balance updates
            model_obj: Model object for the request

        Returns:
            Response or StreamingResponse from upstream with cost tracking
        """
        path = self.normalize_request_path(path, model_obj)
        self._require_supported_endpoint_for_path(path, model_obj)
        if self.requires_verified_confidential_transport:
            self._require_current_confidential_transport()
        url = self.build_request_url(path, model_obj)

        original_model_id = (
            route_alias or (model_obj.forwarded_model_id or model_obj.id)
            if model_obj
            else route_alias
        )
        self._require_confidential_transport_covers_model(
            model_obj,
            route_alias=original_model_id,
        )

        transformed_body = self.prepare_responses_request_body(request_body, model_obj)
        self.prepare_request_headers(headers, path, request_body, model_obj)
        ehbp_request: EhbpEncryptedRequest | None = None
        outbound_body = (
            transformed_body if transformed_body is not None else request_body
        )
        if self.requires_verified_ehbp_transport:
            outbound_body, ehbp_request = self._prepare_ehbp_request_body(
                headers,
                outbound_body,
            )

        logger.debug(
            "Forwarding Responses API request to upstream",
            extra={
                "url": url,
                "method": request.method,
                "path": path,
                "model": original_model_id or "unknown",
                "provider": self.provider_type,
                "key_hash": key.hashed_key[:8] + "...",
            },
        )

        client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=1),
            timeout=None,
        )

        try:
            if outbound_body is not None:
                outbound_content: Any = (
                    _single_chunk_async_iter(outbound_body)
                    if ehbp_request is not None
                    else outbound_body
                )
                response = await client.send(
                    client.build_request(
                        request.method,
                        url,
                        headers=headers,
                        content=outbound_content,
                        params=self.prepare_params(path, request.query_params),
                    ),
                    stream=True,
                )
            else:
                response = await client.send(
                    client.build_request(
                        request.method,
                        url,
                        headers=headers,
                        content=request.stream(),
                        params=self.prepare_params(path, request.query_params),
                    ),
                    stream=True,
                )

            if ehbp_request is not None:
                try:
                    response = await self._decrypt_ehbp_response(response, ehbp_request)
                except UpstreamError:
                    await client.aclose()
                    raise

            if response.status_code != 200:
                if response.status_code >= 500:
                    try:
                        body_bytes = await response.aread()
                    except Exception:
                        body_bytes = b""
                    body_preview = body_bytes.decode("utf-8", errors="ignore").strip()[
                        :500
                    ]
                    redact_body = self.confidential_logging_enabled()
                    safe_body_preview = (
                        CONFIDENTIAL_REDACTION_TEXT
                        if redact_body and body_preview
                        else body_preview
                    )
                    logger.error(
                        "Upstream %s returned %s for model=%s path=%s: %s",
                        self.provider_type,
                        response.status_code,
                        original_model_id or "unknown",
                        path,
                        safe_body_preview or "<empty>",
                        extra={
                            "provider": self.provider_type,
                            "model": original_model_id or "unknown",
                            "status_code": response.status_code,
                            "path": path,
                            "body_preview": None if redact_body else body_preview,
                            "confidential": redact_body,
                        },
                    )
                    await response.aclose()
                    await client.aclose()
                    if redact_body:
                        raise UpstreamError(
                            f"Upstream {self.provider_type} returned {response.status_code} "
                            f"for model {original_model_id or 'unknown'}: "
                            "confidential upstream body redacted",
                            status_code=response.status_code,
                        )
                    raise UpstreamError(
                        f"Upstream {self.provider_type} returned {response.status_code} "
                        f"for model {original_model_id or 'unknown'}: "
                        f"{body_preview[:200] or '<empty>'}",
                        status_code=response.status_code,
                    )

                try:
                    mapped_error = await self.forward_upstream_error_response(
                        request, path, response, model_id=original_model_id
                    )
                finally:
                    await response.aclose()
                    await client.aclose()
                return mapped_error

            if path.startswith("responses"):
                content_type = response.headers.get("content-type", "")
                is_streaming = "text/event-stream" in content_type

                logger.debug(
                    "Responses API response type analysis",
                    extra={
                        "is_streaming": is_streaming,
                        "content_type": content_type,
                        "key_hash": key.hashed_key[:8] + "...",
                    },
                )

                if is_streaming and response.status_code == 200:
                    result = await self.handle_streaming_responses_completion(
                        response,
                        key,
                        max_cost_for_model,
                        requested_model=original_model_id,
                    )
                    background_tasks = BackgroundTasks()
                    background_tasks.add_task(response.aclose)
                    background_tasks.add_task(client.aclose)
                    result.background = background_tasks
                    return result

                if response.status_code == 200:
                    try:
                        return await self.handle_non_streaming_responses_completion(
                            response,
                            key,
                            session,
                            max_cost_for_model,
                            requested_model=original_model_id,
                        )
                    finally:
                        await response.aclose()
                        await client.aclose()

            background_tasks = BackgroundTasks()
            background_tasks.add_task(response.aclose)
            background_tasks.add_task(client.aclose)
            background_tasks.add_task(
                self._finalize_generic_streaming_payment,
                key.hashed_key,
                max_cost_for_model,
                path,
            )

            logger.debug(
                "Streaming non-Responses API response",
                extra={
                    "path": path,
                    "status_code": response.status_code,
                    "key_hash": key.hashed_key[:8] + "...",
                },
            )

            return StreamingResponse(
                response.aiter_bytes(),
                status_code=response.status_code,
                headers=dict(response.headers),
                background=background_tasks,
            )

        except UpstreamError:
            raise

        except httpx.RequestError as exc:
            await client.aclose()
            error_type = type(exc).__name__
            redact_body = self.confidential_logging_enabled()
            error_details = _safe_diagnostic_text(exc, confidential=redact_body)

            logger.error(
                "HTTP request error to upstream Responses API",
                extra={
                    "error_type": error_type,
                    "error_details": error_details,
                    "method": request.method,
                    "url": url,
                    "path": path,
                    "query_params": dict(request.query_params),
                    "key_hash": key.hashed_key[:8] + "...",
                    "confidential": redact_body,
                },
            )

            # Don't revert here — proxy.py owns payment revert to avoid double-revert
            if isinstance(exc, httpx.ConnectError):
                error_message = "Unable to connect to upstream service"
            elif isinstance(exc, httpx.TimeoutException):
                error_message = "Upstream service request timed out"
            elif isinstance(exc, httpx.NetworkError):
                error_message = "Network error while connecting to upstream service"
            else:
                error_message = f"Error connecting to upstream service: {error_type}"

            raise UpstreamError(error_message, status_code=502)

        except Exception as exc:
            await client.aclose()
            redact_body = self.confidential_logging_enabled()
            safe_error = _safe_diagnostic_text(exc, confidential=redact_body)
            safe_traceback = _safe_diagnostic_text(
                traceback.format_exc(), confidential=redact_body
            )

            logger.error(
                "Unexpected error in upstream Responses API forwarding",
                extra={
                    "error": safe_error,
                    "error_type": type(exc).__name__,
                    "method": request.method,
                    "url": url,
                    "path": path,
                    "query_params": dict(request.query_params),
                    "key_hash": key.hashed_key[:8] + "...",
                    "traceback": safe_traceback,
                    "confidential": redact_body,
                },
            )

            # Don't revert here — proxy.py owns payment revert to avoid double-revert
            raise UpstreamError("An unexpected server error occurred", status_code=500)

    async def forward_get_request(
        self,
        request: Request,
        path: str,
        headers: dict,
    ) -> Response | StreamingResponse:
        """Forward unauthenticated GET request to upstream service.

        Args:
            request: Original FastAPI request
            path: Request path
            headers: Prepared headers for upstream

        Returns:
            StreamingResponse from upstream
        """
        path = self.normalize_request_path(path)
        if (
            self.requires_verified_confidential_transport
            or self.requires_verified_ehbp_transport
        ) and not path.startswith("tee/"):
            raise UpstreamError(
                f"{self.provider_type} does not support unrecognized API paths",
                status_code=400,
            )
        if self.requires_verified_confidential_transport:
            self._require_current_confidential_transport()
        if self.requires_verified_ehbp_transport:
            self._verified_ehbp_key_config()

        headers = self.prepare_headers(headers)
        url = self.build_request_url(path)

        logger.debug(
            "Forwarding GET request to upstream",
            extra={
                "url": url,
                "method": request.method,
                "path": path,
                "provider": self.provider_type,
            },
        )

        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=1),
            timeout=None,
        ) as client:
            try:
                response = await client.send(
                    client.build_request(
                        request.method,
                        url,
                        headers=headers,
                        content=request.stream(),
                        params=self.prepare_params(path, request.query_params),
                    ),
                )

                logger.debug(
                    "GET request forwarded",
                    extra={
                        "path": path,
                        "status_code": response.status_code,
                        "provider": self.provider_type,
                    },
                )
                if response.status_code != 200:
                    try:
                        mapped = await self.forward_upstream_error_response(
                            request, path, response
                        )
                    finally:
                        await response.aclose()
                    return mapped

                response_headers = dict(response.headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)
                return StreamingResponse(
                    response.aiter_bytes(),
                    status_code=response.status_code,
                    headers=response_headers,
                )
            except Exception as exc:
                tb = traceback.format_exc()
                logger.error(
                    "Error forwarding GET request",
                    extra={
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "method": request.method,
                        "url": url,
                        "path": path,
                        "query_params": dict(request.query_params),
                        "traceback": tb,
                    },
                )
                return create_error_response(
                    "internal_error",
                    "An unexpected server error occurred",
                    500,
                    request=request,
                )

    async def get_x_cashu_cost(
        self, response_data: dict, max_cost_for_model: int
    ) -> MaxCostData | CostData | None:
        """Calculate cost for X-Cashu payment based on response data.

        Args:
            response_data: Response data containing model and usage information
            max_cost_for_model: Maximum cost for the model

        Returns:
            Cost data object (MaxCostData or CostData) or None if calculation fails
        """
        model = response_data.get("model", None)
        logger.debug(
            "Calculating cost for response",
            extra={"model": model, "has_usage": "usage" in response_data},
        )

        async with create_session() as session:
            match await calculate_cost(response_data, max_cost_for_model, session):
                case MaxCostData() as cost:
                    logger.debug(
                        "Using max cost pricing",
                        extra={"model": model, "max_cost_msats": cost.total_msats},
                    )
                    return cost
                case CostData() as cost:
                    logger.debug(
                        "Using token-based pricing",
                        extra={
                            "model": model,
                            "total_cost_msats": cost.total_msats,
                            "input_msats": cost.input_msats,
                            "output_msats": cost.output_msats,
                        },
                    )
                    return cost
                case CostDataError() as error:
                    logger.error(
                        "Cost calculation error",
                        extra={
                            "model": model,
                            "error_message": error.message,
                            "error_code": error.code,
                        },
                    )
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": {
                                "message": error.message,
                                "type": "invalid_request_error",
                                "code": error.code,
                            }
                        },
                    )
        return None

    async def send_refund(
        self,
        amount: int,
        unit: str,
        mint: str | None = None,
        request_id: str | None = None,
    ) -> str:
        """Create and send a refund token to the user.

        Args:
            amount: Refund amount
            unit: Unit of the refund (sat or msat)
            mint: Optional mint URL for the refund token
            request_id: Optional HTTP request ID for tracking

        Returns:
            Refund token string
        """
        logger.debug(
            "Creating refund token",
            extra={"amount": amount, "unit": unit, "mint": mint},
        )

        max_retries = 3
        last_exception = None

        for attempt in range(max_retries):
            try:
                refund_token = await send_token(amount, unit=unit, mint_url=mint)

                logger.info(
                    "Refund token created successfully",
                    extra={
                        "amount": amount,
                        "unit": unit,
                        "mint": mint,
                        "attempt": attempt + 1,
                        "token_fingerprint": credential_fingerprint(refund_token),
                    },
                )

                try:
                    await store_cashu_transaction(
                        token=refund_token,
                        amount=amount,
                        unit=unit,
                        mint_url=mint,
                        typ="out",
                        request_id=request_id,
                    )
                except Exception:
                    pass  # store_cashu_transaction already logs

                return refund_token
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        "Refund token creation failed, retrying",
                        extra={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "amount": amount,
                            "unit": unit,
                            "mint": mint,
                        },
                    )
                else:
                    logger.error(
                        "Failed to create refund token after all retries",
                        extra={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "amount": amount,
                            "unit": unit,
                            "mint": mint,
                        },
                    )

        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": f"failed to create refund after {max_retries} attempts: {str(last_exception)}",
                    "type": "invalid_request_error",
                    "code": "send_token_failed",
                }
            },
        )

    async def handle_x_cashu_streaming_response(
        self,
        content_str: str,
        response: httpx.Response,
        amount: int,
        unit: str,
        max_cost_for_model: int,
        mint: str | None = None,
        request_id: str | None = None,
        requested_model: str | None = None,
    ) -> StreamingResponse:
        """Handle streaming response for X-Cashu payment, calculating refund if needed.

        Args:
            content_str: Response content as string
            response: Original httpx response
            amount: Payment amount received
            unit: Payment unit (sat or msat)
            max_cost_for_model: Maximum cost for the model

        Returns:
            StreamingResponse with refund token in header if applicable
        """
        logger.debug(
            "Processing streaming response",
            extra={
                "amount": amount,
                "unit": unit,
                "content_lines": len(content_str.strip().split("\n")),
            },
        )

        response_headers = dict(response.headers)
        if "transfer-encoding" in response_headers:
            del response_headers["transfer-encoding"]
        if "content-encoding" in response_headers:
            del response_headers["content-encoding"]

        usage_data = None
        model = None
        cost_data: CostData | MaxCostData | None = None

        lines = content_str.strip().split("\n")
        for line in lines:
            if line.startswith("data: "):
                try:
                    data_json = json.loads(line[6:])
                    # OpenAI format: usage and model at top level
                    if "usage" in data_json:
                        usage_data = data_json["usage"]
                        model = data_json.get("model") or model
                    elif "model" in data_json and not model:
                        model = data_json["model"]
                    # Anthropic format: model and input usage inside "message" key
                    if "message" in data_json:
                        msg = data_json["message"]
                        if not model and msg.get("model"):
                            model = msg["model"]
                        if msg.get("usage") and not usage_data:
                            usage_data = msg["usage"]
                        elif msg.get("usage") and usage_data:
                            # Merge: message_start has input_tokens, message_delta has output_tokens
                            merged = dict(usage_data)
                            for k, v in msg["usage"].items():
                                merged[k] = merged.get(k, 0) + v
                            usage_data = merged
                except json.JSONDecodeError:
                    continue

        if usage_data and model:
            if requested_model:
                model = requested_model
            logger.debug(
                "Found usage data in streaming response",
                extra={
                    "model": model,
                    "usage_data": usage_data,
                    "amount": amount,
                    "unit": unit,
                },
            )

            response_data = {"usage": usage_data, "model": model}
            try:
                cost_data = await self.get_x_cashu_cost(
                    response_data, max_cost_for_model
                )
                if cost_data:
                    if unit == "msat":
                        refund_amount = amount - cost_data.total_msats
                    elif unit == "sat":
                        refund_amount = amount - (cost_data.total_msats + 999) // 1000
                    else:
                        raise ValueError(f"Invalid unit: {unit}")

                    if refund_amount > 0:
                        logger.debug(
                            "Processing refund for streaming response",
                            extra={
                                "original_amount": amount,
                                "cost_msats": cost_data.total_msats,
                                "refund_amount": refund_amount,
                                "unit": unit,
                                "model": model,
                            },
                        )

                        refund_token = await self.send_refund(
                            refund_amount,
                            unit,
                            mint,
                            request_id=request_id,
                        )
                        response_headers["X-Cashu"] = refund_token

                        logger.info(
                            "Refund processed for streaming response",
                            extra={
                                "refund_amount": refund_amount,
                                "unit": unit,
                                "refund_token_fingerprint": credential_fingerprint(
                                    refund_token
                                ),
                            },
                        )
                    else:
                        logger.debug(
                            "No refund needed for streaming response",
                            extra={
                                "amount": amount,
                                "cost_msats": cost_data.total_msats,
                                "model": model,
                            },
                        )
            except Exception as e:
                logger.error(
                    "Error calculating cost for streaming response",
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "model": model,
                        "amount": amount,
                        "unit": unit,
                    },
                )

        for i, line in enumerate(lines):
            if line.startswith("data: "):
                try:
                    data_json = json.loads(line[6:])
                    if not isinstance(data_json, dict):
                        continue
                    changed = False
                    if "provider" not in data_json:
                        self._apply_provider_field(data_json)
                        changed = True
                    if requested_model and data_json.get("model"):
                        data_json["model"] = requested_model
                        changed = True
                    if cost_data and "usage" in data_json and data_json["usage"]:
                        data_json["usage"]["cost_sats"] = cost_data.total_msats // 1000
                        changed = True
                    if changed:
                        lines[i] = "data: " + json.dumps(data_json)
                except json.JSONDecodeError:
                    pass

        async def generate() -> AsyncGenerator[bytes, None]:
            for line in lines:
                yield (line + "\n").encode("utf-8")

        return StreamingResponse(
            generate(),
            status_code=response.status_code,
            headers=response_headers,
            media_type="text/plain",
        )

    async def handle_x_cashu_non_streaming_response(
        self,
        content_str: str,
        response: httpx.Response,
        amount: int,
        unit: str,
        max_cost_for_model: int,
        mint: str | None = None,
        request_id: str | None = None,
        requested_model: str | None = None,
    ) -> Response:
        """Handle non-streaming response for X-Cashu payment, calculating refund if needed.

        Args:
            content_str: Response content as string
            response: Original httpx response
            amount: Payment amount received
            unit: Payment unit (sat or msat)
            max_cost_for_model: Maximum cost for the model

        Returns:
            Response with refund token in header if applicable
        """
        logger.debug(
            "Processing non-streaming response",
            extra={"amount": amount, "unit": unit, "content_length": len(content_str)},
        )

        try:
            response_json = json.loads(content_str)
            self._apply_provider_field(response_json)
            if requested_model:
                response_json["model"] = requested_model
            cost_data = await self.get_x_cashu_cost(response_json, max_cost_for_model)

            if cost_data and "usage" in response_json:
                response_json["usage"]["cost_sats"] = cost_data.total_msats // 1000

            if not cost_data:
                logger.error(
                    "Failed to calculate cost for response",
                    extra={
                        "amount": amount,
                        "unit": unit,
                        "response_model": response_json.get("model", "unknown"),
                    },
                )
                return Response(
                    content=json.dumps(
                        {
                            "error": {
                                "message": "Error forwarding request to upstream",
                                "type": "upstream_error",
                                "code": response.status_code,
                            }
                        }
                    ),
                    status_code=response.status_code,
                    media_type="application/json",
                )

            response_headers = dict(response.headers)
            if "transfer-encoding" in response_headers:
                del response_headers["transfer-encoding"]
            if "content-encoding" in response_headers:
                del response_headers["content-encoding"]

            if unit == "msat":
                refund_amount = amount - cost_data.total_msats
            elif unit == "sat":
                refund_amount = amount - (cost_data.total_msats + 999) // 1000
            else:
                raise ValueError(f"Invalid unit: {unit}")

            logger.debug(
                "Processing non-streaming response cost calculation",
                extra={
                    "original_amount": amount,
                    "cost_msats": cost_data.total_msats,
                    "refund_amount": refund_amount,
                    "unit": unit,
                    "model": response_json.get("model", "unknown"),
                },
            )

            if refund_amount > 0:
                refund_token = await self.send_refund(
                    refund_amount,
                    unit,
                    mint,
                    request_id=request_id,
                )
                response_headers["X-Cashu"] = refund_token

                logger.info(
                    "Refund processed for non-streaming response",
                    extra={
                        "refund_amount": refund_amount,
                        "unit": unit,
                        "refund_token_fingerprint": credential_fingerprint(
                            refund_token
                        ),
                    },
                )

            return Response(
                content=json.dumps(response_json),
                status_code=response.status_code,
                headers=response_headers,
                media_type="application/json",
            )
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse JSON from upstream response",
                extra={
                    "error": str(e),
                    "content_preview": content_str[:200] + "..."
                    if len(content_str) > 200
                    else content_str,
                    "amount": amount,
                    "unit": unit,
                },
            )

            emergency_refund = amount
            refund_token = await send_token(emergency_refund, unit=unit, mint_url=mint)
            response.headers["X-Cashu"] = refund_token
            try:
                await store_cashu_transaction(
                    token=refund_token,
                    amount=emergency_refund,
                    unit=unit,
                    mint_url=mint,
                    typ="out",
                    request_id=request_id,
                )
            except Exception:
                pass

            logger.warning(
                "Emergency refund issued due to JSON parse error",
                extra={
                    "original_amount": amount,
                    "refund_amount": emergency_refund,
                },
            )

            return Response(
                content=content_str,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type="application/json",
            )

    async def handle_x_cashu_chat_completion(
        self,
        response: httpx.Response,
        amount: int,
        unit: str,
        max_cost_for_model: int,
        mint: str | None = None,
        request_id: str | None = None,
        requested_model: str | None = None,
    ) -> StreamingResponse | Response:
        """Handle chat completion response for X-Cashu payment, detecting streaming vs non-streaming.

        Args:
            response: Response from upstream
            amount: Payment amount received
            unit: Payment unit (sat or msat)
            max_cost_for_model: Maximum cost for the model

        Returns:
            StreamingResponse or Response depending on response type
        """
        logger.debug(
            "Handling chat completion response",
            extra={"amount": amount, "unit": unit, "status_code": response.status_code},
        )

        try:
            content = await response.aread()
            content_str = (
                content.decode("utf-8") if isinstance(content, bytes) else content
            )
            is_streaming = content_str.startswith("data:") or "data:" in content_str

            logger.debug(
                "Chat completion response analysis",
                extra={
                    "is_streaming": is_streaming,
                    "content_length": len(content_str),
                    "amount": amount,
                    "unit": unit,
                },
            )

            if is_streaming:
                return await self.handle_x_cashu_streaming_response(
                    content_str,
                    response,
                    amount,
                    unit,
                    max_cost_for_model,
                    mint,
                    request_id=request_id,
                    requested_model=requested_model,
                )
            else:
                return await self.handle_x_cashu_non_streaming_response(
                    content_str,
                    response,
                    amount,
                    unit,
                    max_cost_for_model,
                    mint,
                    request_id=request_id,
                    requested_model=requested_model,
                )

        except Exception as e:
            logger.error(
                "Error processing chat completion response",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "amount": amount,
                    "unit": unit,
                },
            )
            return StreamingResponse(
                response.aiter_bytes(),
                status_code=response.status_code,
                headers=dict(response.headers),
            )

    async def forward_x_cashu_request(
        self,
        request: Request,
        path: str,
        headers: dict,
        amount: int,
        unit: str,
        max_cost_for_model: int,
        model_obj: Model,
        mint: str | None = None,
        *,
        route_alias: str | None = None,
    ) -> Response | StreamingResponse:
        """Forward request paid with X-Cashu token to upstream service.

        Args:
            request: Original FastAPI request
            path: Request path
            headers: Prepared headers for upstream
            amount: Payment amount from X-Cashu token
            unit: Payment unit (sat or msat)
            max_cost_for_model: Maximum cost for the model
            model_obj: Model object for the request

        Returns:
            Response or StreamingResponse with refund if applicable
        """
        if path.startswith("v1/"):
            path = path.replace("v1/", "")

        self._require_supported_endpoint_for_path(path, model_obj)
        if self.requires_verified_confidential_transport:
            self._require_current_confidential_transport()

        request_body = await request.body()

        requested_model_id = route_alias or model_obj.forwarded_model_id or model_obj.id
        self._require_confidential_transport_covers_model(
            model_obj,
            route_alias=requested_model_id,
        )

        if (
            path.endswith("messages/count_tokens")
            and not self.supports_anthropic_messages
        ):
            return count_tokens_locally(request_body, model_obj)

        if (
            path.endswith("messages")
            and not path.endswith("count_tokens")
            and not self.supports_anthropic_messages
        ):
            if self.requires_verified_ehbp_transport:
                raise UpstreamError(
                    "EHBP transport does not support /v1/messages translation yet",
                    status_code=424,
                )
            return await self._forward_x_cashu_messages_via_litellm(
                request_body=request_body,
                amount=amount,
                unit=unit,
                max_cost_for_model=max_cost_for_model,
                model_obj=model_obj,
                mint=mint,
                request_id=getattr(request.state, "request_id", None),
            )

        url = f"{self.base_url}/{path}"

        transformed_body = self.prepare_request_body(
            request_body, model_obj, path=path, headers=headers
        )
        self.prepare_request_headers(headers, path, request_body, model_obj)
        ehbp_request: EhbpEncryptedRequest | None = None
        outbound_body = (
            transformed_body if transformed_body is not None else request_body
        )
        if self.requires_verified_ehbp_transport:
            outbound_body, ehbp_request = self._prepare_ehbp_request_body(
                headers,
                outbound_body,
            )

        logger.debug(
            "Forwarding request to upstream",
            extra={
                "url": url,
                "method": request.method,
                "path": path,
                "amount": amount,
                "unit": unit,
            },
        )

        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=1),
            timeout=None,
        ) as client:
            try:
                response = await client.send(
                    client.build_request(
                        request.method,
                        url,
                        headers=headers,
                        content=_single_chunk_async_iter(outbound_body)
                        if ehbp_request is not None
                        else outbound_body,
                        params=self.prepare_params(path, request.query_params),
                    ),
                    stream=True,
                )

                if ehbp_request is not None:
                    response = await self._decrypt_ehbp_response(response, ehbp_request)

                if response.status_code != 200:
                    logger.error(
                        "Received upstream response",
                        extra={
                            "reason_phrase": response.reason_phrase,
                            "status_code": response.status_code,
                            "path": path,
                            "response_headers": dict(response.headers),
                        },
                    )
                else:
                    logger.debug(
                        "Received upstream response",
                        extra={
                            "status_code": response.status_code,
                            "path": path,
                            "response_headers": dict(response.headers),
                        },
                    )

                if response.status_code != 200:
                    logger.warning(
                        "Upstream request failed, processing refund",
                        extra={
                            "status_code": response.status_code,
                            "path": path,
                            "amount": amount,
                            "unit": unit,
                        },
                    )

                    refund_token = await self.send_refund(
                        amount,
                        unit,
                        mint,
                        request_id=getattr(request.state, "request_id", None),
                    )

                    logger.info(
                        "Refund processed for failed upstream request",
                        extra={
                            "status_code": response.status_code,
                            "refund_amount": amount,
                            "unit": unit,
                            "refund_token_fingerprint": credential_fingerprint(
                                refund_token
                            ),
                        },
                    )

                    error_response = Response(
                        content=json.dumps(
                            {
                                "error": {
                                    "message": "Error forwarding request to upstream",
                                    "type": "upstream_error",
                                    "code": response.status_code,
                                    "refund_token": refund_token,
                                }
                            }
                        ),
                        status_code=response.status_code,
                        media_type="application/json",
                    )
                    error_response.headers["X-Cashu"] = refund_token
                    return error_response

                if (
                    path.endswith("chat/completions")
                    or path.endswith("embeddings")
                    or path.endswith("messages")
                    or path.endswith("messages/count_tokens")
                ):
                    logger.debug(
                        "Processing completion/embeddings/messages response",
                        extra={"path": path, "amount": amount, "unit": unit},
                    )

                    result = await self.handle_x_cashu_chat_completion(
                        response,
                        amount,
                        unit,
                        max_cost_for_model,
                        mint,
                        request_id=getattr(request.state, "request_id", None),
                        requested_model=requested_model_id,
                    )
                    background_tasks = BackgroundTasks()
                    background_tasks.add_task(response.aclose)
                    result.background = background_tasks
                    return result

                background_tasks = BackgroundTasks()
                background_tasks.add_task(response.aclose)
                background_tasks.add_task(client.aclose)

                logger.debug(
                    "Streaming non-chat response",
                    extra={"path": path, "status_code": response.status_code},
                )

                return StreamingResponse(
                    response.aiter_bytes(),
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    background=background_tasks,
                )
            except Exception as exc:
                redact_body = self.confidential_logging_enabled()
                safe_error = _safe_diagnostic_text(exc, confidential=redact_body)
                safe_traceback = _safe_diagnostic_text(
                    traceback.format_exc(), confidential=redact_body
                )
                logger.error(
                    "Unexpected error in upstream forwarding",
                    extra={
                        "error": safe_error,
                        "error_type": type(exc).__name__,
                        "method": request.method,
                        "url": url,
                        "path": path,
                        "query_params": dict(request.query_params),
                        "traceback": safe_traceback,
                        "confidential": redact_body,
                    },
                )
                return create_error_response(
                    "internal_error",
                    "An unexpected server error occurred",
                    500,
                    request=request,
                )

    async def handle_x_cashu_responses(
        self,
        request: Request,
        x_cashu_token: str,
        path: str,
        max_cost_for_model: int,
        model_obj: Model,
        *,
        route_alias: str | None = None,
    ) -> Response | StreamingResponse:
        """Handle X-Cashu payment for Responses API requests.

        Args:
            request: Original FastAPI request
            x_cashu_token: X-Cashu token from request header
            path: Request path
            max_cost_for_model: Maximum cost for the model
            model_obj: Model object for the request

        Returns:
            Response or StreamingResponse from upstream with refund if applicable
        """
        logger.debug(
            "Processing X-Cashu payment for Responses API",
            extra={
                "path": path,
                "method": request.method,
                "token_fingerprint": credential_fingerprint(x_cashu_token),
            },
        )

        self._require_x_cashu_preconditions_before_token(
            path,
            model_obj,
            route_alias=route_alias,
        )

        try:
            headers = dict(request.headers)
            amount, unit, mint = await recieve_token(x_cashu_token)
            headers = self.prepare_headers(dict(request.headers))

            request_id = getattr(request.state, "request_id", None)
            try:
                await store_cashu_transaction(
                    token=x_cashu_token,
                    amount=amount,
                    unit=unit,
                    mint_url=mint,
                    typ="in",
                    request_id=request_id,
                    collected=True,
                )
            except Exception:
                pass

            logger.info(
                "X-Cashu token redeemed for Responses API",
                extra={"amount": amount, "unit": unit, "path": path, "mint": mint},
            )

            return await self.forward_x_cashu_responses_request(
                request,
                path,
                headers,
                amount,
                unit,
                max_cost_for_model,
                model_obj,
                mint,
                route_alias=route_alias,
            )
        except Exception as e:
            error_message = str(e)
            logger.error(
                "X-Cashu payment for Responses API failed",
                extra={
                    "error": error_message,
                    "error_type": type(e).__name__,
                    "path": path,
                    "method": request.method,
                },
            )

            # Use same error handling as regular X-Cashu
            if "already spent" in error_message.lower():
                return create_error_response(
                    "token_already_spent",
                    "The provided CASHU token has already been spent",
                    400,
                    request=request,
                    token=x_cashu_token,
                )

            if "invalid token" in error_message.lower():
                return create_error_response(
                    "invalid_token",
                    "The provided CASHU token is invalid",
                    400,
                    request=request,
                    token=x_cashu_token,
                )

            if "mint error" in error_message.lower():
                return create_error_response(
                    "mint_error",
                    f"CASHU mint error: {error_message}",
                    422,
                    request=request,
                    token=x_cashu_token,
                )

            return create_error_response(
                "cashu_error",
                f"CASHU token processing failed: {error_message}",
                400,
                request=request,
                token=x_cashu_token,
            )

    async def forward_x_cashu_responses_request(
        self,
        request: Request,
        path: str,
        headers: dict,
        amount: int,
        unit: str,
        max_cost_for_model: int,
        model_obj: Model,
        mint: str | None = None,
        *,
        route_alias: str | None = None,
    ) -> Response | StreamingResponse:
        """Forward Responses API request paid with X-Cashu token to upstream service.

        Args:
            request: Original FastAPI request
            path: Request path
            headers: Prepared headers for upstream
            amount: Payment amount from X-Cashu token
            unit: Payment unit (sat or msat)
            max_cost_for_model: Maximum cost for the model
            model_obj: Model object for the request
            mint: Mint URL for refund tokens

        Returns:
            Response or StreamingResponse with refund if applicable
        """
        if path.startswith("v1/"):
            path = path.replace("v1/", "")

        self._require_supported_endpoint_for_path(path, model_obj)
        if self.requires_verified_confidential_transport:
            self._require_current_confidential_transport()

        url = f"{self.base_url}/{path}"
        requested_model_id = route_alias or model_obj.forwarded_model_id or model_obj.id
        self._require_confidential_transport_covers_model(
            model_obj,
            route_alias=requested_model_id,
        )

        request_body = await request.body()
        transformed_body = self.prepare_responses_request_body(request_body, model_obj)
        self.prepare_request_headers(headers, path, request_body, model_obj)
        ehbp_request: EhbpEncryptedRequest | None = None
        outbound_body = (
            transformed_body if transformed_body is not None else request_body
        )
        if self.requires_verified_ehbp_transport:
            outbound_body, ehbp_request = self._prepare_ehbp_request_body(
                headers,
                outbound_body,
            )

        logger.debug(
            "Forwarding Responses API request to upstream with X-Cashu payment",
            extra={
                "url": url,
                "method": request.method,
                "path": path,
                "amount": amount,
                "unit": unit,
            },
        )

        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=1),
            timeout=None,
        ) as client:
            try:
                response = await client.send(
                    client.build_request(
                        request.method,
                        url,
                        headers=headers,
                        content=_single_chunk_async_iter(outbound_body)
                        if ehbp_request is not None
                        else outbound_body,
                        params=self.prepare_params(path, request.query_params),
                    ),
                    stream=True,
                )

                if ehbp_request is not None:
                    response = await self._decrypt_ehbp_response(response, ehbp_request)

                logger.debug(
                    "Received upstream Responses API response",
                    extra={
                        "status_code": response.status_code,
                        "path": path,
                        "response_headers": dict(response.headers),
                    },
                )

                if response.status_code != 200:
                    logger.warning(
                        "Upstream Responses API request failed, processing refund",
                        extra={
                            "status_code": response.status_code,
                            "path": path,
                            "amount": amount,
                            "unit": unit,
                        },
                    )

                    refund_token = await self.send_refund(
                        amount,
                        unit,
                        mint,
                        request_id=getattr(request.state, "request_id", None),
                    )

                    logger.info(
                        "Refund processed for failed upstream Responses API request",
                        extra={
                            "status_code": response.status_code,
                            "refund_amount": amount,
                            "unit": unit,
                            "refund_token_fingerprint": credential_fingerprint(
                                refund_token
                            ),
                        },
                    )

                    error_response = Response(
                        content=json.dumps(
                            {
                                "error": {
                                    "message": "Error forwarding Responses API request to upstream",
                                    "type": "upstream_error",
                                    "code": response.status_code,
                                    "refund_token": refund_token,
                                }
                            }
                        ),
                        status_code=response.status_code,
                        media_type="application/json",
                    )
                    error_response.headers["X-Cashu"] = refund_token
                    return error_response

                if path.startswith("responses"):
                    logger.debug(
                        "Processing Responses API response",
                        extra={"path": path, "amount": amount, "unit": unit},
                    )

                    result = await self.handle_x_cashu_responses_completion(
                        response,
                        amount,
                        unit,
                        max_cost_for_model,
                        mint,
                        request_id=getattr(request.state, "request_id", None),
                        requested_model=requested_model_id,
                    )
                    background_tasks = BackgroundTasks()
                    background_tasks.add_task(response.aclose)
                    result.background = background_tasks
                    return result

                background_tasks = BackgroundTasks()
                background_tasks.add_task(response.aclose)
                background_tasks.add_task(client.aclose)

                logger.debug(
                    "Streaming non-responses response",
                    extra={"path": path, "status_code": response.status_code},
                )

                return StreamingResponse(
                    response.aiter_bytes(),
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    background=background_tasks,
                )
            except Exception as exc:
                redact_body = self.confidential_logging_enabled()
                safe_error = _safe_diagnostic_text(exc, confidential=redact_body)
                safe_traceback = _safe_diagnostic_text(
                    traceback.format_exc(), confidential=redact_body
                )
                logger.error(
                    "Unexpected error in upstream Responses API forwarding",
                    extra={
                        "error": safe_error,
                        "error_type": type(exc).__name__,
                        "method": request.method,
                        "url": url,
                        "path": path,
                        "query_params": dict(request.query_params),
                        "traceback": safe_traceback,
                        "confidential": redact_body,
                    },
                )
                return create_error_response(
                    "internal_error",
                    "An unexpected server error occurred",
                    500,
                    request=request,
                )

    async def handle_x_cashu_responses_completion(
        self,
        response: httpx.Response,
        amount: int,
        unit: str,
        max_cost_for_model: int,
        mint: str | None = None,
        request_id: str | None = None,
        requested_model: str | None = None,
    ) -> StreamingResponse | Response:
        """Handle Responses API completion response for X-Cashu payment.

        Args:
            response: Response from upstream
            amount: Payment amount received
            unit: Payment unit (sat or msat)
            max_cost_for_model: Maximum cost for the model
            mint: Mint URL for refund tokens

        Returns:
            StreamingResponse or Response depending on response type
        """
        logger.debug(
            "Handling Responses API completion response",
            extra={"amount": amount, "unit": unit, "status_code": response.status_code},
        )

        try:
            content = await response.aread()
            content_str = (
                content.decode("utf-8") if isinstance(content, bytes) else content
            )
            is_streaming = content_str.startswith("data:") or "data:" in content_str

            logger.debug(
                "Responses API completion response analysis",
                extra={
                    "is_streaming": is_streaming,
                    "content_length": len(content_str),
                    "amount": amount,
                    "unit": unit,
                },
            )

            if is_streaming:
                return await self.handle_x_cashu_streaming_responses_response(
                    content_str,
                    response,
                    amount,
                    unit,
                    max_cost_for_model,
                    mint,
                    request_id=request_id,
                    requested_model=requested_model,
                )
            else:
                return await self.handle_x_cashu_non_streaming_responses_response(
                    content_str,
                    response,
                    amount,
                    unit,
                    max_cost_for_model,
                    mint,
                    request_id=request_id,
                    requested_model=requested_model,
                )

        except Exception as e:
            logger.error(
                "Error processing Responses API completion response",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "amount": amount,
                    "unit": unit,
                },
            )
            return StreamingResponse(
                response.aiter_bytes(),
                status_code=response.status_code,
                headers=dict(response.headers),
            )

    async def handle_x_cashu_streaming_responses_response(
        self,
        content_str: str,
        response: httpx.Response,
        amount: int,
        unit: str,
        max_cost_for_model: int,
        mint: str | None = None,
        request_id: str | None = None,
        requested_model: str | None = None,
    ) -> StreamingResponse:
        """Handle streaming Responses API response for X-Cashu payment.

        Similar to regular streaming but handles Responses API specific tokens like reasoning_tokens.
        """
        logger.debug(
            "Processing streaming Responses API response",
            extra={
                "amount": amount,
                "unit": unit,
                "content_lines": len(content_str.strip().split("\\n")),
            },
        )

        response_headers = dict(response.headers)
        if "transfer-encoding" in response_headers:
            del response_headers["transfer-encoding"]
        if "content-encoding" in response_headers:
            del response_headers["content-encoding"]

        usage_data = None
        model = None
        reasoning_tokens = 0

        lines = content_str.strip().split("\\n")
        for line in lines:
            if line.startswith("data: "):
                try:
                    data_json = json.loads(line[6:])
                    if "usage" in data_json:
                        usage_data = data_json["usage"]
                        model = data_json.get("model")
                        # Track reasoning tokens for Responses API
                        if (
                            isinstance(usage_data, dict)
                            and "reasoning_tokens" in usage_data
                        ):
                            reasoning_tokens = usage_data.get("reasoning_tokens", 0)
                    elif "model" in data_json and not model:
                        model = data_json["model"]
                except json.JSONDecodeError:
                    continue

        if usage_data and model:
            if requested_model:
                model = requested_model
            logger.debug(
                "Found usage data in streaming Responses API response",
                extra={
                    "model": model,
                    "usage_data": usage_data,
                    "reasoning_tokens": reasoning_tokens,
                    "amount": amount,
                    "unit": unit,
                },
            )

            response_data = {"usage": usage_data, "model": model}
            try:
                cost_data = await self.get_x_cashu_cost(
                    response_data, max_cost_for_model
                )
                if cost_data:
                    if unit == "msat":
                        refund_amount = amount - cost_data.total_msats
                    elif unit == "sat":
                        refund_amount = amount - (cost_data.total_msats + 999) // 1000
                    else:
                        raise ValueError(f"Invalid unit: {unit}")

                    if refund_amount > 0:
                        logger.debug(
                            "Processing refund for streaming Responses API response",
                            extra={
                                "original_amount": amount,
                                "cost_msats": cost_data.total_msats,
                                "refund_amount": refund_amount,
                                "unit": unit,
                                "model": model,
                                "reasoning_tokens": reasoning_tokens,
                            },
                        )

                        refund_token = await self.send_refund(
                            refund_amount,
                            unit,
                            mint,
                            request_id=request_id,
                        )
                        response_headers["X-Cashu"] = refund_token

                        logger.info(
                            "Refund processed for streaming Responses API response",
                            extra={
                                "refund_amount": refund_amount,
                                "unit": unit,
                                "refund_token_fingerprint": credential_fingerprint(
                                    refund_token
                                ),
                            },
                        )
                    else:
                        logger.debug(
                            "No refund needed for streaming Responses API response",
                            extra={
                                "amount": amount,
                                "cost_msats": cost_data.total_msats,
                                "model": model,
                            },
                        )
            except Exception as e:
                logger.error(
                    "Error calculating cost for streaming Responses API response",
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "model": model,
                        "amount": amount,
                        "unit": unit,
                    },
                )

        for i, line in enumerate(lines):
            if line.startswith("data: "):
                try:
                    data_json = json.loads(line[6:])
                    if not isinstance(data_json, dict):
                        continue
                    changed = False
                    if "provider" not in data_json:
                        self._apply_provider_field(data_json)
                        changed = True
                    if requested_model and data_json.get("model"):
                        data_json["model"] = requested_model
                        changed = True
                    if cost_data and "usage" in data_json and data_json["usage"]:
                        data_json["usage"]["cost_sats"] = cost_data.total_msats // 1000
                        changed = True
                    if changed:
                        lines[i] = "data: " + json.dumps(data_json)
                except json.JSONDecodeError:
                    pass

        async def generate() -> AsyncGenerator[bytes, None]:
            for line in lines:
                yield (line + "\\n").encode("utf-8")

        return StreamingResponse(
            generate(),
            status_code=response.status_code,
            headers=response_headers,
            media_type="text/plain",
        )

    async def handle_x_cashu_non_streaming_responses_response(
        self,
        content_str: str,
        response: httpx.Response,
        amount: int,
        unit: str,
        max_cost_for_model: int,
        mint: str | None = None,
        request_id: str | None = None,
        requested_model: str | None = None,
    ) -> Response:
        """Handle non-streaming Responses API response for X-Cashu payment."""
        logger.debug(
            "Processing non-streaming Responses API response",
            extra={"amount": amount, "unit": unit, "content_length": len(content_str)},
        )

        try:
            response_json = json.loads(content_str)
            self._apply_provider_field(response_json)
            if requested_model:
                response_json["model"] = requested_model
            cost_data = await self.get_x_cashu_cost(response_json, max_cost_for_model)

            if cost_data and "usage" in response_json:
                response_json["usage"]["cost_sats"] = cost_data.total_msats // 1000

            if not cost_data:
                logger.error(
                    "Failed to calculate cost for Responses API response",
                    extra={
                        "amount": amount,
                        "unit": unit,
                        "response_model": response_json.get("model", "unknown"),
                    },
                )
                return Response(
                    content=json.dumps(
                        {
                            "error": {
                                "message": "Error forwarding Responses API request to upstream",
                                "type": "upstream_error",
                                "code": response.status_code,
                            }
                        }
                    ),
                    status_code=response.status_code,
                    media_type="application/json",
                )

            response_headers = dict(response.headers)
            if "transfer-encoding" in response_headers:
                del response_headers["transfer-encoding"]
            if "content-encoding" in response_headers:
                del response_headers["content-encoding"]

            if unit == "msat":
                refund_amount = amount - cost_data.total_msats
            elif unit == "sat":
                refund_amount = amount - (cost_data.total_msats + 999) // 1000
            else:
                raise ValueError(f"Invalid unit: {unit}")

            logger.debug(
                "Processing non-streaming Responses API cost calculation",
                extra={
                    "original_amount": amount,
                    "cost_msats": cost_data.total_msats,
                    "refund_amount": refund_amount,
                    "unit": unit,
                    "model": response_json.get("model", "unknown"),
                },
            )

            if refund_amount > 0:
                refund_token = await self.send_refund(
                    refund_amount,
                    unit,
                    mint,
                    request_id=request_id,
                )
                response_headers["X-Cashu"] = refund_token

                logger.info(
                    "Refund processed for non-streaming Responses API response",
                    extra={
                        "refund_amount": refund_amount,
                        "unit": unit,
                        "refund_token_fingerprint": credential_fingerprint(
                            refund_token
                        ),
                    },
                )

            return Response(
                content=json.dumps(response_json),
                status_code=response.status_code,
                headers=response_headers,
                media_type="application/json",
            )
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse JSON from upstream Responses API response",
                extra={
                    "error": str(e),
                    "content_preview": content_str[:200] + "..."
                    if len(content_str) > 200
                    else content_str,
                    "amount": amount,
                    "unit": unit,
                },
            )

            emergency_refund = amount
            refund_token = await send_token(emergency_refund, unit=unit, mint_url=mint)
            response.headers["X-Cashu"] = refund_token
            try:
                await store_cashu_transaction(
                    token=refund_token,
                    amount=emergency_refund,
                    unit=unit,
                    mint_url=mint,
                    typ="out",
                    request_id=request_id,
                )
            except Exception:
                pass

            logger.warning(
                "Emergency refund issued for Responses API due to JSON parse error",
                extra={
                    "original_amount": amount,
                    "refund_amount": emergency_refund,
                },
            )

            return Response(
                content=content_str,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type="application/json",
            )

    async def handle_x_cashu(
        self,
        request: Request,
        x_cashu_token: str,
        path: str,
        max_cost_for_model: int,
        model_obj: Model,
        *,
        route_alias: str | None = None,
    ) -> Response | StreamingResponse:
        """Handle request with X-Cashu token payment, redeeming token and forwarding request.

        Args:
            request: Original FastAPI request
            x_cashu_token: X-Cashu token from request header
            path: Request path
            max_cost_for_model: Maximum cost for the model
            model_obj: Model object for the request

        Returns:
            Response or StreamingResponse from upstream with refund if applicable
        """
        logger.debug(
            "Processing X-Cashu payment request",
            extra={
                "path": path,
                "method": request.method,
                "token_fingerprint": credential_fingerprint(x_cashu_token),
            },
        )

        self._require_x_cashu_preconditions_before_token(
            path,
            model_obj,
            route_alias=route_alias,
        )

        try:
            headers = dict(request.headers)
            amount, unit, mint = await recieve_token(x_cashu_token)
            headers = self.prepare_headers(dict(request.headers))

            request_id = getattr(request.state, "request_id", None)
            try:
                await store_cashu_transaction(
                    token=x_cashu_token,
                    amount=amount,
                    unit=unit,
                    mint_url=mint,
                    typ="in",
                    request_id=request_id,
                    collected=True,
                )
            except Exception:
                pass

            logger.info(
                "X-Cashu token redeemed successfully",
                extra={"amount": amount, "unit": unit, "path": path, "mint": mint},
            )

            return await self.forward_x_cashu_request(
                request,
                path,
                headers,
                amount,
                unit,
                max_cost_for_model,
                model_obj,
                mint,
                route_alias=route_alias,
            )
        except Exception as e:
            error_message = str(e)
            logger.error(
                "X-Cashu payment request failed",
                extra={
                    "error": error_message,
                    "error_type": type(e).__name__,
                    "path": path,
                    "method": request.method,
                },
            )

            if "already spent" in error_message.lower():
                return create_error_response(
                    "token_already_spent",
                    "The provided CASHU token has already been spent",
                    400,
                    request=request,
                    token=x_cashu_token,
                )

            if "invalid token" in error_message.lower():
                return create_error_response(
                    "invalid_token",
                    "The provided CASHU token is invalid",
                    400,
                    request=request,
                    token=x_cashu_token,
                )

            if "mint error" in error_message.lower():
                return create_error_response(
                    "mint_error",
                    f"CASHU mint error: {error_message}",
                    422,
                    request=request,
                    token=x_cashu_token,
                )

            return create_error_response(
                "cashu_error",
                f"CASHU token processing failed: {error_message}",
                400,
                request=request,
                token=x_cashu_token,
            )

    def _apply_provider_fee_to_model(self, model: Model) -> Model:
        """Apply provider fee to model's USD pricing and calculate max costs.

        Args:
            model: Model object to update

        Returns:
            Model with provider fee applied to pricing and max costs calculated
        """
        adjusted_pricing = Pricing.parse_obj(
            {k: v * self.provider_fee for k, v in model.pricing.dict().items()}
        )

        temp_model = Model(
            id=model.id,
            name=model.name,
            created=model.created,
            description=model.description,
            context_length=model.context_length,
            architecture=model.architecture,
            pricing=adjusted_pricing,
            sats_pricing=None,
            per_request_limits=model.per_request_limits,
            top_provider=model.top_provider,
            enabled=model.enabled,
            upstream_provider_id=model.upstream_provider_id,
            canonical_slug=model.canonical_slug,
            alias_ids=model.alias_ids,
            forwarded_model_id=model.forwarded_model_id,
            supported_endpoints=model.supported_endpoints,
            confidentiality=model.confidentiality,
        )

        (
            adjusted_pricing.max_prompt_cost,
            adjusted_pricing.max_completion_cost,
            adjusted_pricing.max_cost,
        ) = _calculate_usd_max_costs(temp_model)

        return Model(
            id=model.id,
            name=model.name,
            created=model.created,
            description=model.description,
            context_length=model.context_length,
            architecture=model.architecture,
            pricing=adjusted_pricing,
            sats_pricing=model.sats_pricing,
            per_request_limits=model.per_request_limits,
            top_provider=model.top_provider,
            enabled=model.enabled,
            upstream_provider_id=model.upstream_provider_id,
            canonical_slug=model.canonical_slug,
            alias_ids=model.alias_ids,
            forwarded_model_id=model.forwarded_model_id,
            supported_endpoints=model.supported_endpoints,
            confidentiality=model.confidentiality,
        )

    async def fetch_models(self) -> list[Model]:
        """Fetch available models from upstream API and update cache.

        Returns:
            List of Model objects with pricing
        """

        try:
            or_models, provider_models_response = await asyncio.gather(
                self._fetch_openrouter_models(),
                self._fetch_provider_models(),
            )

            provider_model_ids = self._parse_model_ids(provider_models_response)

            found_models = []
            not_found_models = []

            for model_id in provider_model_ids:
                or_model = self._match_model(model_id, or_models)
                if or_model:
                    try:
                        safe_model = remote_model_without_public_proof(or_model)
                        if safe_model is None:
                            continue
                        model = Model(**safe_model)
                        found_models.append(model)
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse model {model_id}",
                            extra={"error": str(e), "error_type": type(e).__name__},
                        )
                else:
                    not_found_models.append(model_id)

            if not_found_models:
                logger.debug(
                    f"({len(not_found_models)}/{len(provider_model_ids)}) unmatched models for {self.provider_type or self.base_url}",
                    extra={"not_found_models": not_found_models},
                )

            return found_models

        except Exception as e:
            logger.error(
                f"Error fetching models for {self.provider_type or self.base_url}",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            return []

    async def _fetch_openrouter_models(self) -> list[dict]:
        """Fetch models from OpenRouter API."""
        url = "https://openrouter.ai/api/v1/models"
        embeddings_url = "https://openrouter.ai/api/v1/embeddings/models"

        async with httpx.AsyncClient(timeout=30.0) as client:
            models_response, embeddings_response = await asyncio.gather(
                client.get(url), client.get(embeddings_url), return_exceptions=True
            )

            all_models = []

            def process_models_response(
                response: httpx.Response | BaseException,
            ) -> list[dict]:
                if not isinstance(response, BaseException):
                    response.raise_for_status()
                    data = response.json()
                    return [
                        model
                        for model in data.get("data", [])
                        if ":free" not in model.get("id", "").lower()
                    ]
                return []

            all_models.extend(process_models_response(models_response))
            all_models.extend(process_models_response(embeddings_response))

            return all_models

    async def _fetch_provider_models(self) -> dict:
        """Fetch models from provider's API."""
        url = f"{self.base_url.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

    def _parse_model_ids(self, response: dict) -> list[str]:
        """Parse model IDs from provider response."""
        return [model.get("id") for model in response.get("data", []) if "id" in model]

    def _match_model(self, model_id: str, or_models: list[dict]) -> dict | None:
        """Match provider model ID with OpenRouter model."""
        return next(
            (
                model
                for model in or_models
                if (model.get("id") == model_id)
                or (model.get("id", "").split("/")[-1] == model_id)
                or (model.get("canonical_slug") == model_id)
                or (model.get("canonical_slug", "").split("/")[-1] == model_id)
            ),
            None,
        )

    async def refresh_models_cache(self) -> None:
        """Refresh the in-memory models cache from upstream API."""
        try:
            async with create_session() as session:
                stmt = select(UpstreamProviderRow).where(
                    UpstreamProviderRow.base_url == self.base_url,
                    UpstreamProviderRow.api_key == self.api_key,
                )
                result = await session.exec(stmt)

                # .first() returns the object or None if not found
                provider = result.first()
                if not provider or not provider.id:
                    raise HTTPException(status_code=404, detail="Provider not found")

                db_models = await list_models(
                    session=session,
                    upstream_id=provider.id,
                    include_disabled=False,
                    apply_fees=False,
                )
                db_model_ids: set[str] = {model.id for model in db_models}
                models = await self.fetch_models()
                if self._last_model_catalog_refresh_succeeded and models:
                    await self._persist_discovered_models(
                        session,
                        provider_id=provider.id,
                        models=models,
                    )
                model_ids = [model.id for model in models]
                diff = set(db_model_ids) - set(model_ids)

                for db_model_id in diff:
                    found_db_model = next(
                        (
                            model_obj
                            for model_obj in db_models
                            if model_obj.id == db_model_id
                        )
                    )
                    models.append(found_db_model)

                models_with_fees = [
                    self._apply_provider_fee_to_model(m) for m in models
                ]

                try:
                    sats_to_usd = sats_usd_price()
                    self._models_cache = [
                        _update_model_sats_pricing(m, sats_to_usd)
                        for m in models_with_fees
                    ]
                except Exception:
                    self._models_cache = models_with_fees

                self._models_by_id = {
                    m.forwarded_model_id or m.id: m for m in self._models_cache
                }

        except Exception as e:
            logger.error(
                f"Failed to refresh models cache for {self.provider_type or self.base_url}",
                extra={"error": str(e), "error_type": type(e).__name__},
            )

    def get_cached_models(self) -> list[Model]:
        """Get cached models for this provider.

        Returns:
            List of cached Model objects
        """
        return self._models_cache

    async def _persist_discovered_models(
        self,
        session: AsyncSession,
        *,
        provider_id: int,
        models: list[Model],
    ) -> None:
        """Persist successful provider model discovery without overwriting manuals."""
        now = int(time.time())
        seen_ids = {model.id for model in models if isinstance(model.id, str)}

        for model in models:
            row = await session.get(ModelRow, (model.id, provider_id))
            capabilities_json = canonical_json(
                {
                    "supported_endpoints": model.supported_endpoints,
                    "architecture": model.architecture.dict(),
                    "top_provider": model.top_provider.dict()
                    if model.top_provider
                    else None,
                }
            )
            if row is None:
                row = ModelRow(
                    id=model.id,
                    upstream_provider_id=provider_id,
                    name=model.name,
                    created=model.created,
                    description=model.description,
                    context_length=model.context_length,
                    architecture=canonical_json(model.architecture.dict()),
                    pricing=canonical_json(model.pricing.dict()),
                    sats_pricing=canonical_json(model.sats_pricing.dict())
                    if model.sats_pricing
                    else None,
                    per_request_limits=canonical_json(model.per_request_limits)
                    if model.per_request_limits is not None
                    else None,
                    top_provider=canonical_json(model.top_provider.dict())
                    if model.top_provider
                    else None,
                    canonical_slug=model.canonical_slug,
                    alias_ids=canonical_json(model.alias_ids)
                    if model.alias_ids is not None
                    else None,
                    enabled=model.enabled,
                    forwarded_model_id=model.forwarded_model_id,
                    last_seen_at=now,
                    availability_status="available",
                    capabilities_json=capabilities_json,
                )
                session.add(row)
                continue

            previous_status = row.availability_status
            row.last_seen_at = now
            row.capabilities_json = capabilities_json

            if previous_status in {"available", "unavailable"}:
                row.availability_status = "available"
                row.name = model.name
                row.created = model.created
                row.description = model.description
                row.context_length = model.context_length
                row.architecture = canonical_json(model.architecture.dict())
                row.pricing = canonical_json(model.pricing.dict())
                row.sats_pricing = (
                    canonical_json(model.sats_pricing.dict())
                    if model.sats_pricing
                    else None
                )
                row.per_request_limits = (
                    canonical_json(model.per_request_limits)
                    if model.per_request_limits is not None
                    else None
                )
                row.top_provider = (
                    canonical_json(model.top_provider.dict())
                    if model.top_provider
                    else None
                )
                row.canonical_slug = model.canonical_slug
                row.alias_ids = (
                    canonical_json(model.alias_ids)
                    if model.alias_ids is not None
                    else None
                )
                row.forwarded_model_id = model.forwarded_model_id
            else:
                row.availability_status = previous_status or "manual"

        unavailable_stmt = select(ModelRow).where(
            ModelRow.upstream_provider_id == provider_id,
            ModelRow.availability_status == "available",
        )
        existing_available = (await session.exec(unavailable_stmt)).all()
        for row in existing_available:
            if row.id not in seen_ids:
                row.availability_status = "unavailable"

        await session.commit()

    def get_cached_model_by_id(self, model_id: str) -> Model | None:
        """Get a specific cached model by ID.

        Args:
            model_id: Model identifier

        Returns:
            Model object or None if not found
        """
        return self._models_by_id.get(model_id)

    @classmethod
    async def create_account_static(cls) -> dict[str, object]:
        """Create a new account with the provider (class method, no instance needed).

        Returns:
            Dict with account creation details including api_key

        Raises:
            NotImplementedError: If provider does not support account creation
        """
        raise NotImplementedError(
            f"Provider {cls.provider_type} does not support account creation"
        )

    async def create_account(self) -> dict[str, object]:
        """Create a new account with the provider.

        Returns:
            Dict with account creation details including api_key

        Raises:
            NotImplementedError: If provider does not support account creation
        """
        raise NotImplementedError(
            f"Provider {self.provider_type} does not support account creation"
        )

    async def initiate_topup(self, amount: int) -> TopupData:
        """Initiate a Lightning Network top-up for the provider account.

        Args:
            amount: Amount in currency units to top up

        Returns:
            TopupData with standardized invoice information

        Raises:
            NotImplementedError: If provider does not support top-up
        """
        raise NotImplementedError(
            f"Provider {self.provider_type} does not support top-up"
        )

    async def get_balance(self) -> float | None:
        """Get the current account balance from the provider.

        Returns:
            Float representing the balance amount, or None if not supported/available.
            Typically in USD or the provider's credit unit.

        Raises:
            NotImplementedError: If provider does not support balance checking (default behavior)
        """
        raise NotImplementedError(
            f"Provider {self.provider_type} does not support balance checking"
        )
