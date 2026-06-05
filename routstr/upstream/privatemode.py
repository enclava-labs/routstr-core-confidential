from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from ..core.logging import get_logger
from ..payment.models import Architecture, Model, Pricing, TopProvider
from .base import ConfidentialityStatus, _privatemode_proxy_base_url_violation
from .generic import GenericUpstreamProvider
from .strict_json import require_canonical_json

if TYPE_CHECKING:
    from ..core.db import UpstreamProviderRow

logger = get_logger(__name__)


def _catalog_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _catalog_int(value: object, default: int = 0) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def _catalog_string_set(value: object) -> set[str] | None:
    if value is None:
        return set()
    if not isinstance(value, list):
        return None
    values: set[str] = set()
    for item in value:
        value_item = _catalog_string(item)
        if value_item is None:
            return None
        values.add(value_item.lower())
    return values


def _privatemode_supported_endpoints(
    *,
    tasks: set[str],
    endpoints: set[str],
) -> set[str]:
    supported = set(endpoints)
    if tasks & {"generate", "tool_calling"}:
        supported.update(
            {
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/messages",
            }
        )
    if tasks & {"embed", "embedding", "embeddings"}:
        supported.add("/v1/embeddings")
    if tasks & {
        "transcribe",
        "transcription",
        "speech-to-text",
        "speech_to_text",
        "audio-transcription",
        "audio_transcription",
    }:
        supported.add("/v1/audio/transcriptions")
    return supported


def _privatemode_architecture_from_catalog(
    *,
    tasks: set[str],
    endpoints: set[str],
) -> Architecture:
    if "/v1/embeddings" in endpoints or tasks & {"embed", "embedding", "embeddings"}:
        return Architecture(
            modality="text->embedding",
            input_modalities=["text"],
            output_modalities=["embedding"],
            tokenizer="unknown",
            instruct_type=None,
        )
    if "/v1/audio/transcriptions" in endpoints or tasks & {
        "transcribe",
        "transcription",
        "speech-to-text",
        "speech_to_text",
        "audio-transcription",
        "audio_transcription",
    }:
        return Architecture(
            modality="audio->text",
            input_modalities=["audio"],
            output_modalities=["text"],
            tokenizer="unknown",
            instruct_type=None,
        )
    if endpoints & {"/v1/chat/completions", "/v1/completions", "/v1/messages"} or (
        tasks & {"generate", "tool_calling"}
    ):
        input_modalities = ["text"]
        if "vision" in tasks:
            input_modalities.append("image")
        return Architecture(
            modality="+".join(input_modalities) + "->text",
            input_modalities=input_modalities,
            output_modalities=["text"],
            tokenizer="unknown",
            instruct_type=None,
        )
    return Architecture(
        modality="unknown",
        input_modalities=[],
        output_modalities=[],
        tokenizer="unknown",
        instruct_type=None,
    )


class PrivatemodeUpstreamProvider(GenericUpstreamProvider):
    """Privatemode provider boundary for a local verified proxy."""

    provider_type = "privatemode"
    default_base_url = "http://127.0.0.1:8080/v1"
    platform_url = "https://docs.privatemode.ai/api/overview/"
    supports_anthropic_messages = True
    supports_audio_api = True
    supports_audio_translations_api = False
    supports_audio_speech_api = False
    supports_completions_api = True
    supports_embeddings_api = True
    supports_images_api = False
    supports_moderations_api = False
    supports_responses_api = False
    requires_verified_confidential_transport = True

    def __init__(
        self,
        api_key: str = "",
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
                mode="privatemode",
                failure_reason="runtime Privatemode verifier has not run",
            )
        )

    @classmethod
    def from_db_row(
        cls, provider_row: "UpstreamProviderRow"
    ) -> "PrivatemodeUpstreamProvider":
        return cls(
            api_key=provider_row.api_key,
            base_url=provider_row.base_url,
            provider_fee=provider_row.provider_fee,
        )

    @classmethod
    def get_provider_metadata(cls) -> dict[str, object]:
        return {
            "id": cls.provider_type,
            "name": "Privatemode",
            "default_base_url": cls.default_base_url,
            "fixed_base_url": False,
            "platform_url": cls.platform_url,
            "can_create_account": False,
            "can_topup": False,
            "can_show_balance": False,
        }

    def transform_model_name(self, model_id: str) -> str:
        return model_id.removeprefix("privatemode/")

    async def fetch_models(self) -> list[Model]:
        """Fetch Privatemode proxy models while preserving task capability ceilings."""
        if violation := _privatemode_proxy_base_url_violation(self.base_url):
            logger.error(
                "Refusing to fetch Privatemode models from invalid proxy base_url",
                extra={"error": violation},
            )
            return []

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url.rstrip('/')}/models",
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
                require_canonical_json(data, "Privatemode model catalog response")

            models: list[Model] = []
            for model_data in data.get("data", []):
                if not isinstance(model_data, dict):
                    continue
                model_id = _catalog_string(model_data.get("id"))
                if model_id is None:
                    continue

                tasks = _catalog_string_set(model_data.get("tasks"))
                endpoints = _catalog_string_set(model_data.get("endpoints"))
                malformed_capabilities = tasks is None or endpoints is None
                if tasks is None:
                    tasks = set()
                if endpoints is None:
                    endpoints = set()
                supported_endpoints = _privatemode_supported_endpoints(
                    tasks=tasks,
                    endpoints=endpoints,
                )
                if malformed_capabilities:
                    supported_endpoints = set()
                architecture = _privatemode_architecture_from_catalog(
                    tasks=tasks,
                    endpoints=supported_endpoints,
                )
                context_length = _catalog_int(
                    model_data.get("context_window"),
                    _catalog_int(model_data.get("context_length"), 4096),
                )
                model_name = _catalog_string(model_data.get("name")) or model_id
                owned_by = _catalog_string(model_data.get("owned_by")) or "privatemode"
                created = _catalog_int(model_data.get("created"))

                models.append(
                    Model(
                        id=model_id,
                        name=model_name,
                        created=created,
                        description=f"{model_name} via {owned_by}",
                        context_length=context_length,
                        architecture=architecture,
                        pricing=Pricing(
                            prompt=0.0,
                            completion=0.0,
                            request=0.0,
                            image=0.0,
                            web_search=0.0,
                            internal_reasoning=0.0,
                            max_prompt_cost=0.001,
                            max_completion_cost=0.001,
                            max_cost=0.001,
                        ),
                        top_provider=TopProvider(
                            context_length=context_length,
                            max_completion_tokens=(
                                context_length // 2 if context_length else None
                            ),
                            is_moderated=False,
                        ),
                        supported_endpoints=sorted(supported_endpoints),
                    )
                )

            logger.info(
                "Fetched Privatemode proxy models",
                extra={"model_count": len(models), "base_url": self.base_url},
            )
            return models
        except Exception as exc:
            logger.error(
                "Failed to fetch Privatemode proxy models",
                extra={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "base_url": self.base_url,
                },
            )
            return []

    async def refresh_confidentiality_status(self) -> ConfidentialityStatus:
        from .confidential_verifiers import refresh_privatemode_status

        status = await refresh_privatemode_status(self)
        self.set_confidentiality_status(status)
        return self.confidentiality_status()
