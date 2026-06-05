from __future__ import annotations

import json
from typing import Any


def _json_constant_rejecter(label: str):
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} must not contain {value}")

    return reject_constant


def sha256_json_payload(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def require_canonical_json(value: object, label: str) -> None:
    try:
        sha256_json_payload(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be canonical JSON") from exc


def loads_strict_json(data: str | bytes, label: str) -> Any:
    return json.loads(data, parse_constant=_json_constant_rejecter(label))
