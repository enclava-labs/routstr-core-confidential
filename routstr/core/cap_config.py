from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sqlmodel.ext.asyncio.session import AsyncSession

from .logging import get_logger

logger = get_logger(__name__)

DEFAULT_CAP_CONFIG_DIR = Path("/state/.enclava/config")
DEFAULT_BOOTSTRAP_API_KEY_BALANCE_MSATS = 1_000_000_000


def _env_value(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def cap_config_dir(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    configured = _env_value(env, "ROUTSTR_CAP_CONFIG_DIR") or _env_value(
        env, "CAP_CONFIG_DIR"
    )
    return Path(configured) if configured else DEFAULT_CAP_CONFIG_DIR


def read_cap_config_text(
    key: str,
    environ: Mapping[str, str] | None = None,
    *,
    config_dir: Path | None = None,
) -> str | None:
    env = os.environ if environ is None else environ
    if value := _env_value(env, key):
        return value

    base = config_dir or cap_config_dir(env)
    path = base / key
    try:
        if not path.is_file():
            return None
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning(
            "Failed to read CAP config key", extra={"key": key, "error": str(exc)}
        )
        return None
    return value or None


def read_cap_config_json(
    key: str,
    environ: Mapping[str, str] | None = None,
    *,
    config_dir: Path | None = None,
) -> dict[str, Any] | None:
    raw = read_cap_config_text(key, environ, config_dir=config_dir)
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Ignoring invalid JSON CAP config key",
            extra={"key": key, "error": str(exc)},
        )
        return None
    if not isinstance(value, dict):
        logger.warning("Ignoring non-object JSON CAP config key", extra={"key": key})
        return None
    return value


def configured_api_key(
    environ: Mapping[str, str] | None = None,
    *,
    config_dir: Path | None = None,
) -> str | None:
    raw = read_cap_config_text("ROUTSTR_API_KEY", environ, config_dir=config_dir)
    if not raw:
        return None
    return raw[3:] if raw.startswith("sk-") else raw


def configured_api_key_balance_msats(
    environ: Mapping[str, str] | None = None,
    *,
    config_dir: Path | None = None,
) -> int:
    raw = read_cap_config_text(
        "ROUTSTR_API_KEY_BALANCE_MSATS", environ, config_dir=config_dir
    )
    if not raw:
        return DEFAULT_BOOTSTRAP_API_KEY_BALANCE_MSATS
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid ROUTSTR_API_KEY_BALANCE_MSATS")
        return DEFAULT_BOOTSTRAP_API_KEY_BALANCE_MSATS
    return max(value, 0)


async def seed_configured_api_key(
    session: AsyncSession,
    environ: Mapping[str, str] | None = None,
    *,
    config_dir: Path | None = None,
) -> bool:
    key = configured_api_key(environ, config_dir=config_dir)
    if not key:
        return False

    from .db import ApiKey

    existing = await session.get(ApiKey, key)
    if existing:
        return False

    api_key = ApiKey(
        hashed_key=key,
        balance=configured_api_key_balance_msats(environ, config_dir=config_dir),
        reserved_balance=0,
    )
    session.add(api_key)
    await session.commit()
    logger.info(
        "Seeded CAP-configured Routstr API key", extra={"key_hash": key[:8] + "..."}
    )
    return True
