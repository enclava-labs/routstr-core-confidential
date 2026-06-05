from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from ..core.logging import get_logger
from ..payment.models import Architecture, Model, Pricing, TopProvider
from .base import ConfidentialityStatus
from .generic import GenericUpstreamProvider
from .strict_json import require_canonical_json
from .url_validation import https_url_violation

if TYPE_CHECKING:
    from ..core.db import UpstreamProviderRow

logger = get_logger(__name__)


def _catalog_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _catalog_number(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _catalog_int(value: object, default: int = 0) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def _catalog_endpoints(value: object) -> set[str] | None:
    if value is None:
        return set()
    if not isinstance(value, list):
        return None
    endpoints: set[str] = set()
    for item in value:
        endpoint = _catalog_string(item)
        if endpoint is None:
            return None
        endpoints.add(endpoint.lower())
    return endpoints


def _architecture_from_catalog(
    *,
    endpoints: set[str],
    model_type: str | None,
    multimodal: bool,
) -> Architecture:
    if "/v1/embeddings" in endpoints or model_type == "embedding":
        return Architecture(
            modality="text->embedding",
            input_modalities=["text"],
            output_modalities=["embedding"],
            tokenizer="unknown",
            instruct_type=None,
        )
    if "/v1/audio/speech" in endpoints or model_type == "tts":
        return Architecture(
            modality="text->audio",
            input_modalities=["text"],
            output_modalities=["audio"],
            tokenizer="unknown",
            instruct_type=None,
        )
    if (
        "/v1/audio/transcriptions" in endpoints
        or "/v1/audio/translations" in endpoints
        or model_type == "audio"
    ):
        input_modalities = ["audio"]
        if "/v1/chat/completions" in endpoints:
            input_modalities.append("text")
        return Architecture(
            modality="+".join(input_modalities) + "->text",
            input_modalities=input_modalities,
            output_modalities=["text"],
            tokenizer="unknown",
            instruct_type=None,
        )
    if model_type == "document" or "/v1/convert/file" in endpoints:
        return Architecture(
            modality="file->text",
            input_modalities=["file"],
            output_modalities=["text"],
            tokenizer="unknown",
            instruct_type=None,
        )
    if model_type == "tool":
        return Architecture(
            modality="tool",
            input_modalities=["tool"],
            output_modalities=["tool"],
            tokenizer="unknown",
            instruct_type=None,
        )
    input_modalities = ["text", "image"] if multimodal else ["text"]
    return Architecture(
        modality="+".join(input_modalities) + "->text",
        input_modalities=input_modalities,
        output_modalities=["text"],
        tokenizer="unknown",
        instruct_type=None,
    )


def _tinfoil_catalog_model_to_routstr(model_data: dict[str, Any]) -> Model | None:
    model_id = _catalog_string(model_data.get("id"))
    if model_id is None:
        return None
    endpoints = _catalog_endpoints(model_data.get("endpoints"))
    if endpoints is None:
        endpoints = set()
    model_type = _catalog_string(model_data.get("type"))
    if model_type is not None:
        model_type = model_type.lower()
    context_length = _catalog_int(
        model_data.get("context_window"),
        _catalog_int(model_data.get("context_length"), 4096),
    )
    pricing = model_data.get("pricing")
    if not isinstance(pricing, dict):
        pricing = {}
    input_price = _catalog_number(pricing.get("inputTokenPricePer1M")) / 1_000_000
    output_price = _catalog_number(pricing.get("outputTokenPricePer1M")) / 1_000_000
    request_price = _catalog_number(pricing.get("requestPrice"))
    name = _catalog_string(model_data.get("name")) or model_id
    owner = _catalog_string(model_data.get("owned_by")) or "tinfoil"
    return Model(
        id=model_id,
        name=name,
        created=_catalog_int(model_data.get("created")),
        description=f"{owner} {model_type or 'model'} model",
        context_length=context_length,
        architecture=_architecture_from_catalog(
            endpoints=endpoints,
            model_type=model_type,
            multimodal=model_data.get("multimodal") is True,
        ),
        pricing=Pricing(
            prompt=input_price,
            completion=output_price,
            request=request_price,
            image=0.0,
            web_search=0.0,
            internal_reasoning=0.0,
        ),
        top_provider=TopProvider(
            context_length=context_length,
            max_completion_tokens=context_length // 2 if context_length else None,
            is_moderated=model_type == "safety",
        ),
        supported_endpoints=sorted(endpoints),
    )


class TinfoilUpstreamProvider(GenericUpstreamProvider):
    """Tinfoil provider boundary for verified encrypted inference."""

    provider_type = "tinfoil"
    default_base_url = "https://inference.tinfoil.sh/v1"
    platform_url = "https://docs.tinfoil.sh/sdk/overview"
    requires_verified_ehbp_transport = True
    supports_audio_api = True
    supports_audio_translations_api = True
    supports_audio_speech_api = False
    supports_completions_api = False
    supports_images_api = False
    supports_moderations_api = False

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        provider_fee: float = 1.0,
    ):
        super().__init__(
            base_url=base_url or self.default_base_url,
            api_key=api_key,
            provider_fee=provider_fee,
            upstream_name=self.provider_type,
        )
        self.set_confidentiality_status(
            ConfidentialityStatus(
                enabled=True,
                verified=False,
                mode="tinfoil",
                failure_reason="runtime Tinfoil verifier has not run",
            )
        )

    @classmethod
    def from_db_row(
        cls, provider_row: "UpstreamProviderRow"
    ) -> "TinfoilUpstreamProvider":
        return cls(
            api_key=provider_row.api_key,
            base_url=provider_row.base_url,
            provider_fee=provider_row.provider_fee,
        )

    @classmethod
    def get_provider_metadata(cls) -> dict[str, object]:
        return {
            "id": cls.provider_type,
            "name": "Tinfoil",
            "default_base_url": cls.default_base_url,
            "fixed_base_url": True,
            "platform_url": cls.platform_url,
            "can_create_account": False,
            "can_topup": False,
            "can_show_balance": False,
        }

    def transform_model_name(self, model_id: str) -> str:
        return model_id.removeprefix("tinfoil/")

    def prepare_request_headers(
        self,
        headers: dict[str, str],
        path: str,
        request_body: bytes | None,
        model_obj: Model,
    ) -> None:
        super().prepare_request_headers(headers, path, request_body, model_obj)
        headers["X-Tinfoil-Request-Usage-Metrics"] = "true"

    async def fetch_models(self) -> list[Model]:
        """Fetch Tinfoil catalog models while preserving endpoint architecture."""
        if violation := https_url_violation(self.base_url, "Tinfoil base_url"):
            logger.error(
                "Refusing to fetch Tinfoil models from invalid base_url",
                extra={"error": violation},
            )
            return []

        url = f"{self.base_url.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                require_canonical_json(data, "Tinfoil model catalog response")

            models: list[Model] = []
            for model_data in data.get("data", []):
                if not isinstance(model_data, dict):
                    continue
                try:
                    model = _tinfoil_catalog_model_to_routstr(model_data)
                except Exception as exc:
                    logger.warning(
                        "Failed to parse Tinfoil model",
                        extra={
                            "model_id": model_data.get("id", "unknown"),
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )
                    continue
                if model is not None:
                    models.append(model)
            return models
        except Exception as exc:
            logger.error(
                "Error fetching models from Tinfoil",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
            return []

    async def refresh_confidentiality_status(self) -> ConfidentialityStatus:
        from .confidential_verifiers import refresh_tinfoil_status

        status = await refresh_tinfoil_status(self)
        self.set_confidentiality_status(status)
        return self.confidentiality_status()
