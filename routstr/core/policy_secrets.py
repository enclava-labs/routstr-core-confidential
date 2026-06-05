from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, urlparse

INLINE_SECRET_POLICY_KEYS = (
    ("api_key", "api_key"),
    ("apiKey", "api_key"),
    ("authorization", "authorization"),
    ("Authorization", "authorization"),
    ("bearer_token", "bearer_token"),
    ("bearerToken", "bearer_token"),
    ("access_token", "access_token"),
    ("accessToken", "access_token"),
    ("refresh_token", "refresh_token"),
    ("refreshToken", "refresh_token"),
    ("token", "token"),
    ("password", "password"),
    ("client_secret", "client_secret"),
    ("clientSecret", "client_secret"),
    ("secret", "secret"),
    ("upstream_api_key", "upstream_api_key"),
    ("upstreamApiKey", "upstream_api_key"),
)
INLINE_SECRET_POLICY_LABELS = {
    key.lower(): label for key, label in INLINE_SECRET_POLICY_KEYS
}
SECRET_LIKE_VALUE_PATTERN = re.compile(
    r"(?i)(?:bearer\s+\S+|(?:^|[=:\s,;])sk-[A-Za-z0-9][A-Za-z0-9._-]*)"
)


def _secret_like_string(value: str) -> bool:
    stripped = value.strip()
    return bool(SECRET_LIKE_VALUE_PATTERN.search(stripped))


def _credential_bearing_url_query_keys(value: str) -> list[str]:
    try:
        parsed = urlparse(value.strip())
    except ValueError:
        return []
    if not (parsed.scheme and parsed.netloc):
        return []

    found: list[str] = []
    seen: set[str] = set()
    for raw_key, raw_value in parse_qsl(parsed.query, keep_blank_values=True):
        key = raw_key.replace("-", "_").lower()
        label = INLINE_SECRET_POLICY_LABELS.get(key)
        if label and raw_value and label not in seen:
            found.append(label)
            seen.add(label)
    return found


def _credential_bearing_url(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    try:
        parsed = urlparse(stripped)
    except ValueError:
        return False
    return bool(
        parsed.scheme
        and parsed.netloc
        and (
            parsed.username is not None
            or parsed.password is not None
            or "@" in parsed.netloc
        )
    )


def inline_policy_secret_violations(
    policy: dict[str, Any],
    *,
    policy_name: str = "verifier policy",
) -> list[str]:
    violations: list[str] = []
    seen: set[str] = set()

    def visit(value: object, *, path_label: str | None = None) -> None:
        if isinstance(value, dict):
            for raw_key, raw_value in value.items():
                key_text = str(raw_key)
                key = key_text.lower()
                label = INLINE_SECRET_POLICY_LABELS.get(key)
                if label and label not in seen and raw_value not in (None, ""):
                    seen.add(label)
                    violations.append(
                        f"{label} must not be embedded in {policy_name}"
                    )
                visit(raw_value, path_label=key_text)
        elif isinstance(value, list):
            for item in value:
                visit(item, path_label=path_label)
        elif isinstance(value, str):
            if _credential_bearing_url(value):
                label = (
                    f"{path_label} URL credentials"
                    if path_label
                    else "url credentials"
                )
                if label not in seen:
                    seen.add(label)
                    if path_label:
                        violations.append(
                            f"{path_label} must not include credentials in "
                            f"{policy_name}"
                        )
                    else:
                        violations.append(
                            f"url credentials must not be embedded in {policy_name}"
                        )
            for label in _credential_bearing_url_query_keys(value):
                query_label = f"url {label}"
                if query_label not in seen:
                    seen.add(query_label)
                    violations.append(
                        f"{label} must not be embedded in {policy_name} URL query"
                    )
            if _secret_like_string(value):
                label = "secret-like value"
                if label not in seen:
                    seen.add(label)
                    violations.append(
                        f"{label} must not be embedded in {policy_name}"
                    )

    visit(policy)
    return violations
