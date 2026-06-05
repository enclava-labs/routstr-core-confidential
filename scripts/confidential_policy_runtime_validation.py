from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class RuntimePolicyValidationError(ValueError):
    pass


def validate_runtime_policy_document(
    document: dict[str, Any],
    *,
    provider_type: str,
) -> dict[str, Any]:
    """Fail generator output that the runtime verifier policy cannot load."""
    try:
        base_url = document["base_url"]
        confidentiality = document["confidentiality"]
    except KeyError as exc:
        raise RuntimePolicyValidationError(
            f"generated policy is missing {exc.args[0]}"
        ) from exc
    if not isinstance(base_url, str) or not base_url.strip():
        raise RuntimePolicyValidationError("generated policy base_url must be a string")
    if not isinstance(confidentiality, dict):
        raise RuntimePolicyValidationError(
            "generated policy confidentiality must be an object"
        )

    from routstr.upstream.base import ConfidentialVerifierPolicy

    try:
        policy = ConfidentialVerifierPolicy.from_provider_settings(
            provider_type=provider_type,
            base_url=base_url,
            provider_settings={"confidentiality": confidentiality},
        )
    except Exception as exc:
        raise RuntimePolicyValidationError(
            f"generated policy is not runtime-loadable: {exc}"
        ) from exc
    if policy is None:
        raise RuntimePolicyValidationError(
            "generated policy did not produce a runtime verifier policy"
        )
    return document
