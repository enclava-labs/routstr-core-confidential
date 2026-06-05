from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from urllib.parse import urlparse

import httpx
from pydantic.v1 import BaseModel, Field

from ..core.logging import get_logger
from ..payment.models import (
    Architecture,
    Model,
    Pricing,
    async_fetch_openrouter_models,
    remote_model_without_public_proof,
)
from .base import BaseUpstreamProvider, ConfidentialityStatus, TopupData
from .strict_json import loads_strict_json, require_canonical_json
from .url_validation import https_origin_tuple, https_url_violation

if TYPE_CHECKING:
    from ..core.db import UpstreamProviderRow

logger = get_logger(__name__)


def _is_ppq_owned_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    normalized = hostname.rstrip(".").lower()
    return normalized == "ppq.ai" or normalized.endswith(".ppq.ai")


def _url_has_private_path(value: str) -> bool:
    parsed = urlparse(value.strip())
    return any(
        segment.lower() == "private" for segment in parsed.path.split("/") if segment
    )


def _ppq_private_base_url_violation(value: object) -> str | None:
    if violation := https_url_violation(value, "PPQ private base_url"):
        return violation
    parsed = urlparse(value.strip()) if isinstance(value, str) else urlparse("")
    if not _is_ppq_owned_host(parsed.hostname):
        return "PPQ private base_url host must be ppq.ai or a ppq.ai subdomain"
    if not isinstance(value, str) or not _url_has_private_path(value):
        return "PPQ private base_url must include a private path segment"
    return None


def _ppq_private_catalog_url_violation(
    *,
    provider_base_url: object,
    catalog_base_url: object,
) -> str | None:
    if violation := _ppq_private_base_url_violation(provider_base_url):
        return violation
    if violation := https_url_violation(
        catalog_base_url,
        "PPQ private catalog_base_url",
    ):
        return violation
    if not isinstance(provider_base_url, str) or not isinstance(catalog_base_url, str):
        return "PPQ private catalog_base_url must be a non-empty string"
    if https_origin_tuple(provider_base_url) != https_origin_tuple(catalog_base_url):
        return "PPQ private catalog_base_url origin must match provider base_url origin"
    return None


class PPQAIModelPricing(BaseModel):
    ui: Optional[dict[str, float]] = None
    api: Optional[dict[str, float]] = None
    input_per_1M_tokens: Optional[float] = Field(None, alias="input_per_1M_tokens")
    output_per_1M_tokens: Optional[float] = Field(None, alias="output_per_1M_tokens")


class PPQAIModel(BaseModel):
    id: str
    provider: Optional[str] = None
    name: str
    created_at: int
    context_length: int
    pricing: PPQAIModelPricing
    popular: bool = False


def _ppq_model_to_routstr_model(ppqai_model: PPQAIModel) -> Model:
    input_price = 0.0
    if ppqai_model.pricing.api:
        input_price = ppqai_model.pricing.api.get("input_per_1M", 0.0)
    elif ppqai_model.pricing.input_per_1M_tokens:
        input_price = ppqai_model.pricing.input_per_1M_tokens

    output_price = 0.0
    if ppqai_model.pricing.api:
        output_price = ppqai_model.pricing.api.get("output_per_1M", 0.0)
    elif ppqai_model.pricing.output_per_1M_tokens:
        output_price = ppqai_model.pricing.output_per_1M_tokens

    return Model(
        id=ppqai_model.id,
        name=ppqai_model.name,
        created=ppqai_model.created_at // 1000,
        description=f"{ppqai_model.provider or 'PPQ.AI'} model",
        context_length=ppqai_model.context_length,
        architecture=Architecture(
            modality="text->text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="Unknown",
            instruct_type=None,
        ),
        pricing=Pricing(
            prompt=input_price / 1_000_000,
            completion=output_price / 1_000_000,
            request=0.0,
            image=0.0,
            web_search=0.0,
            internal_reasoning=0.0,
        ),
        supported_endpoints=["/v1/chat/completions"]
        if ppqai_model.id.startswith("private/")
        else None,
    )


class PPQAIUpstreamProvider(BaseUpstreamProvider):
    """Upstream provider for PPQ.AI API with Lightning Network top-up support."""

    provider_type = "ppqai"
    default_base_url = "https://api.ppq.ai"
    platform_url = "https://ppq.ai/api-docs"
    IGNORED_MODEL_IDS: list[str] = ["auto"]

    def __init__(self, api_key: str, provider_fee: float = 1.0):
        super().__init__(
            base_url=self.default_base_url, api_key=api_key, provider_fee=provider_fee
        )

    @classmethod
    def from_db_row(
        cls, provider_row: "UpstreamProviderRow"
    ) -> "PPQAIUpstreamProvider":
        return cls(
            api_key=provider_row.api_key,
            provider_fee=provider_row.provider_fee,
        )

    @classmethod
    def get_provider_metadata(cls) -> dict[str, object]:
        return {
            "id": cls.provider_type,
            "name": "PPQ.AI",
            "default_base_url": cls.default_base_url,
            "fixed_base_url": True,
            "platform_url": cls.platform_url,
            "can_create_account": True,
            "can_topup": True,
            "can_show_balance": True,
        }

    def transform_model_name(self, model_id: str) -> str:
        return model_id

    @classmethod
    async def create_account_static(cls) -> dict[str, object]:
        """Create a new PPQ.AI account without requiring an instance.

        Returns:
            Dict containing 'credit_id' and 'api_key' for the new account.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        url = f"{cls.default_base_url}/accounts/create"

        logger.info("Creating new PPQ.AI account", extra={"url": url})

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url)
            response.raise_for_status()
            account_data = response.json()

            logger.info(
                "Successfully created PPQ.AI account",
                extra={
                    "credit_id": account_data.get("credit_id"),
                    "has_api_key": bool(account_data.get("api_key")),
                },
            )

            return account_data

    async def fetch_models(self) -> list[Model]:
        """Fetch models from PPQ.AI API."""
        url = f"{self.base_url}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()

                models_data = data.get("data", [])

                or_models = [
                    Model(**safe_model)
                    for model in await async_fetch_openrouter_models()
                    if (safe_model := remote_model_without_public_proof(model))
                    is not None
                ]

                models = []
                for model_data in models_data:
                    try:
                        ppqai_model = PPQAIModel.parse_obj(model_data)
                        if ppqai_model.id in self.IGNORED_MODEL_IDS:
                            continue
                        if ppqai_model.id.startswith("private/"):
                            continue

                        or_model = next(
                            (
                                model
                                for model in or_models
                                if (model.id == ppqai_model.id)
                                or (model.id.split("/")[-1] == ppqai_model.id)
                                or (model.id == ppqai_model.id.split("/")[-1])
                            ),
                            None,
                        )

                        if or_model:
                            input_price = None
                            if ppqai_model.pricing.api:
                                input_price = ppqai_model.pricing.api.get(
                                    "input_per_1M"
                                )
                            elif ppqai_model.pricing.input_per_1M_tokens:
                                input_price = ppqai_model.pricing.input_per_1M_tokens

                            if input_price is not None:
                                or_model.pricing.prompt = input_price / 1_000_000

                            output_price = None
                            if ppqai_model.pricing.api:
                                output_price = ppqai_model.pricing.api.get(
                                    "output_per_1M"
                                )
                            elif ppqai_model.pricing.output_per_1M_tokens:
                                output_price = ppqai_model.pricing.output_per_1M_tokens

                            if output_price is not None:
                                or_model.pricing.completion = output_price / 1_000_000

                            if cl := ppqai_model.context_length:
                                or_model.context_length = cl
                            models.append(or_model)
                        else:
                            models.append(_ppq_model_to_routstr_model(ppqai_model))
                    except Exception as e:
                        logger.warning(
                            "Failed to parse PPQ.AI model",
                            extra={
                                "model_id": model_data.get("id", "unknown"),
                                "error": str(e),
                                "error_type": type(e).__name__,
                            },
                        )

                return models

        except Exception as e:
            logger.error(
                "Error fetching models from PPQ.AI",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            return []

    async def on_upstream_error_redirect(
        self, status_code: int, error_message: str
    ) -> None:
        if "insufficient balance" in error_message.lower():
            logger.warning(
                f"Disabling PPQ.AI provider ({self.base_url}) due to insufficient balance",
                extra={"error": error_message},
            )
            from sqlmodel import select

            from ..core.db import UpstreamProviderRow, create_session

            async with create_session() as session:
                statement = select(UpstreamProviderRow).where(
                    UpstreamProviderRow.base_url == self.base_url,
                    UpstreamProviderRow.api_key == self.api_key,
                )
                result = await session.exec(statement)
                provider = result.first()

                if provider:
                    provider.enabled = False
                    session.add(provider)
                    await session.commit()

                    # Trigger re-initialization of providers
                    # Import here to avoid circular dependency
                    from ..proxy import reinitialize_upstreams

                    await reinitialize_upstreams()

    async def create_account(self) -> dict[str, object]:
        """Create a new PPQ.AI account.

        Returns:
            Dict containing 'credit_id' and 'api_key' for the new account.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        url = f"{self.base_url}/accounts/create"

        logger.info("Creating new PPQ.AI account", extra={"url": url})

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url)
            response.raise_for_status()
            account_data = response.json()

            logger.info(
                "Successfully created PPQ.AI account",
                extra={
                    "credit_id": account_data.get("credit_id"),
                    "has_api_key": bool(account_data.get("api_key")),
                },
            )

            return account_data

    async def create_lightning_topup(
        self, amount: int, currency: str
    ) -> dict[str, object]:
        """Create a Lightning Network top-up invoice for this account.

        Args:
            amount: Amount to top up (in the specified currency)
            currency: Currency for the top-up (default: "USD")

        Returns:
            Dict containing invoice details including 'invoice_id', 'payment_request', etc.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        url = f"{self.base_url}/topup/create/btc-lightning"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"amount": amount, "currency": currency}

        logger.info(
            "Creating Lightning top-up invoice",
            extra={"url": url, "amount": amount, "currency": currency},
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            invoice_data = response.json()

            logger.info(
                "Successfully created Lightning top-up invoice",
                extra={
                    "invoice_id": invoice_data.get("invoice_id"),
                    "amount": amount,
                    "currency": currency,
                },
            )

            return invoice_data

    async def check_topup_status(self, invoice_id: str) -> bool:
        """Check the status of a Lightning top-up invoice.

        Args:
            invoice_id: The invoice ID to check

        Returns:
            True if the invoice is paid (status == "Settled"), False otherwise

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        url = f"{self.base_url}/topup/status/{invoice_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        logger.debug(
            "Checking Lightning top-up status",
            extra={"url": url, "invoice_id": invoice_id},
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            status_data = response.json()

            is_paid = status_data.get("status") == "Settled"

            logger.debug(
                "Retrieved Lightning top-up status",
                extra={
                    "invoice_id": invoice_id,
                    "status": status_data.get("status"),
                    "is_paid": is_paid,
                },
            )

            return is_paid

    async def initiate_topup(self, amount: int) -> TopupData:
        """Initiate a Lightning Network top-up for the PPQ.AI account.

        Args:
            amount: Amount in currency units to top up (will be sent to PPQ.AI API)

        Returns:
            TopupData with standardized invoice information

        Raises:
            httpx.HTTPStatusError: If the API request fails
        """
        ppq_response = await self.create_lightning_topup(amount, "USD")

        logger.info(
            "PPQ.AI top-up response",
            extra={
                "ppq_response": ppq_response,
                "invoice_id": ppq_response.get("invoice_id"),
                "has_lightning_invoice": "lightning_invoice" in ppq_response,
            },
        )

        expires_at_value = ppq_response.get("expires_at")
        checkout_url_value = ppq_response.get("checkout_url")

        topup_data = TopupData(
            invoice_id=str(ppq_response["invoice_id"]),
            payment_request=str(ppq_response["lightning_invoice"]),
            amount=int(ppq_response["amount"])
            if isinstance(ppq_response["amount"], (int, float, str))
            else 0,
            currency=str(ppq_response["currency"]),
            expires_at=int(expires_at_value)
            if isinstance(expires_at_value, (int, float, str))
            and expires_at_value is not None
            else None,
            checkout_url=str(checkout_url_value)
            if checkout_url_value is not None
            else None,
        )

        logger.info(
            "Created TopupData",
            extra={
                "invoice_id": topup_data.invoice_id,
                "payment_request_length": len(topup_data.payment_request),
                "amount": topup_data.amount,
            },
        )

        return topup_data

    async def get_balance(self) -> float | None:
        """Get the current account balance from PPQ.AI.

        Returns:
            Float representing the balance amount (in USD), or None if unavailable.

        Raises:
            httpx.HTTPStatusError: If the API request fails
        """
        data = await self.check_balance()
        balance = data.get("balance")
        if isinstance(balance, (int, float)):
            return float(balance)
        return None

    async def check_balance(self) -> dict[str, object]:
        """Check the account balance for this PPQ.AI account.

        Returns:
            Dict containing balance information

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        url = f"{self.base_url}/credits/balance"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        logger.debug("Checking PPQ.AI account balance", extra={"url": url})

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json={})
            response.raise_for_status()
            balance_data = response.json()

            logger.debug(
                "Retrieved PPQ.AI account balance",
                extra={"balance": balance_data.get("balance")},
            )

            return balance_data


class PPQPrivateUpstreamProvider(BaseUpstreamProvider):
    """PPQ private TEE provider for direct verified EHBP transport.

    The provider discovers PPQ `private/*` models from the public PPQ catalog,
    but it intentionally remains unverified until a runtime verifier proves the
    private enclave path and sets `verified=True`.
    """

    provider_type = "ppq-private"
    default_base_url = "https://api.ppq.ai/private/v1"
    platform_url = "https://ppq.ai/api-docs#private-tee"
    catalog_base_url = "https://api.ppq.ai/v1"
    requires_verified_ehbp_transport = True
    supports_audio_api = False
    supports_audio_transcriptions_api = False
    supports_audio_translations_api = False
    supports_audio_speech_api = False
    supports_completions_api = False
    supports_embeddings_api = False
    supports_images_api = False
    supports_moderations_api = False
    supports_responses_api = False

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str = "",
        provider_fee: float = 1.0,
        catalog_base_url: str | None = None,
    ):
        super().__init__(
            base_url=(base_url or self.default_base_url),
            api_key=api_key,
            provider_fee=provider_fee,
        )
        if isinstance(catalog_base_url, str) and catalog_base_url.strip():
            self.catalog_base_url = catalog_base_url.rstrip("/")
        else:
            self.catalog_base_url = self.catalog_base_url.rstrip("/")
        self.set_confidentiality_status(
            ConfidentialityStatus(
                enabled=True,
                verified=False,
                mode="ppq-private-tee",
                failure_reason="runtime PPQ private verifier has not run",
                model_id_prefixes=["private/"],
            )
        )

    @classmethod
    def from_db_row(
        cls, provider_row: "UpstreamProviderRow"
    ) -> "PPQPrivateUpstreamProvider":
        catalog_base_url = None
        if provider_row.provider_settings:
            try:
                settings = loads_strict_json(
                    provider_row.provider_settings,
                    "PPQ private provider_settings JSON",
                )
                if isinstance(settings, dict):
                    raw_catalog_base_url = settings.get("catalog_base_url")
                    if isinstance(raw_catalog_base_url, str):
                        catalog_base_url = raw_catalog_base_url
            except Exception:
                pass

        return cls(
            base_url=provider_row.base_url,
            api_key=provider_row.api_key,
            provider_fee=provider_row.provider_fee,
            catalog_base_url=catalog_base_url,
        )

    @classmethod
    def get_provider_metadata(cls) -> dict[str, object]:
        return {
            "id": cls.provider_type,
            "name": "PPQ Private TEE",
            "default_base_url": cls.default_base_url,
            "fixed_base_url": False,
            "platform_url": cls.platform_url,
            "can_create_account": False,
            "can_topup": False,
            "can_show_balance": False,
        }

    def transform_model_name(self, model_id: str) -> str:
        if model_id.startswith("ppq/private/"):
            return model_id.removeprefix("ppq/private/")
        if model_id.startswith("private/"):
            return model_id.removeprefix("private/")
        if model_id.startswith("ppq/"):
            return model_id.removeprefix("ppq/")
        return model_id

    def request_model_id_for_transform(self, model_obj: Model) -> str:
        return model_obj.forwarded_model_id or model_obj.id

    def prepare_request_headers(
        self,
        headers: dict[str, str],
        path: str,
        request_body: bytes | None,
        model_obj: Model,
    ) -> None:
        super().prepare_request_headers(headers, path, request_body, model_obj)
        for auth_header in ("Authorization", "authorization"):
            headers.pop(auth_header, None)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        private_model_id = model_obj.forwarded_model_id or model_obj.id
        if private_model_id.startswith("ppq/private/"):
            private_model_id = private_model_id.removeprefix("ppq/")
        if not private_model_id.startswith("private/"):
            private_model_id = f"private/{self.transform_model_name(private_model_id)}"
        headers["X-Private-Model"] = private_model_id
        headers["x-query-source"] = "api"

    async def fetch_models(self) -> list[Model]:
        """Fetch only PPQ private TEE models from the all-model catalog."""
        if violation := _ppq_private_catalog_url_violation(
            provider_base_url=self.base_url,
            catalog_base_url=self.catalog_base_url,
        ):
            logger.error(
                "Refusing to fetch PPQ private models from invalid catalog URL",
                extra={"error": violation},
            )
            return []

        url = f"{self.catalog_base_url}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    url, headers=headers, params={"type": "all"}
                )
                response.raise_for_status()
                data = response.json()
                require_canonical_json(data, "PPQ private model catalog response")

            models: list[Model] = []
            for model_data in data.get("data", []):
                if not isinstance(model_data, dict):
                    continue
                raw_model_id = model_data.get("id")
                if not isinstance(raw_model_id, str) or not raw_model_id.startswith(
                    "private/"
                ):
                    continue
                try:
                    ppqai_model = PPQAIModel.parse_obj(model_data)
                    models.append(_ppq_model_to_routstr_model(ppqai_model))
                except Exception as e:
                    logger.warning(
                        "Failed to parse PPQ private model",
                        extra={
                            "model_id": model_data.get("id", "unknown"),
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
            return models
        except Exception as e:
            logger.error(
                "Error fetching PPQ private models",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            return []

    async def refresh_confidentiality_status(self) -> ConfidentialityStatus:
        from .confidential_verifiers import refresh_ppq_private_status

        status = await refresh_ppq_private_status(self)
        self.set_confidentiality_status(status)
        return self.confidentiality_status()
