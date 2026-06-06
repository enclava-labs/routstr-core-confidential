import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.responses import Response as StarletteResponse
from starlette.types import Scope

from ..auth import periodic_key_reset
from ..balance import balance_router, deprecated_wallet_router
from ..lightning import lightning_router, periodic_invoice_watcher
from ..nostr import (
    announce_provider,
    providers_cache_refresher,
    publish_usage_analytics,
)
from ..nostr.discovery import providers_router
from ..payment.models import models_router, update_sats_pricing
from ..payment.price import update_prices_periodically
from ..proxy import initialize_upstreams, proxy_router, refresh_model_maps_periodically
from ..upstream.auto_topup import periodic_auto_topup
from ..upstream.litellm_routing import configure_litellm
from ..wallet import periodic_payout, periodic_refund_sweep, periodic_routstr_fee_payout
from .admin import admin_router
from .attestation import (
    get_routstr_attestation_statement,
    read_routstr_hpke_key_config,
)
from .confidentiality_public import is_full_sha256_digest
from .db import create_session, init_db, run_migrations
from .exceptions import general_exception_handler, http_exception_handler
from .logging import get_logger, setup_logging
from .middleware import LoggingMiddleware
from .not_found import _NOT_FOUND_HTML, not_found_catch_all  # noqa: F401
from .settings import SettingsService
from .settings import settings as global_settings
from .version import __version__

# Initialize logging first
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Application startup initiated", extra={"version": __version__})

    btc_price_task = None
    pricing_task = None
    payout_task = None
    nip91_task = None
    analytics_task = None
    providers_task = None
    models_refresh_task = None
    model_maps_refresh_task = None
    key_reset_task = None
    auto_topup_task = None
    refund_sweep_task = None
    routstr_fee_task = None
    invoice_watcher_task = None

    try:
        # Apply litellm-wide settings (drop_params, chat-completions URL,
        # debug logging) before any upstream provider dispatches a request.
        configure_litellm()

        # Run database migrations on startup
        run_migrations()

        # Initialize database connection pools
        # This creates any tables that might not be tracked by migrations yet
        await init_db()

        # Initialize application settings (env -> computed -> DB precedence)
        async with create_session() as session:
            s = await SettingsService.initialize(session)
            if s.reset_reserved_balance_on_startup:
                from .db import reset_all_reserved_balances

                await reset_all_reserved_balances(session)

        if not s.admin_password:
            logger.warning(
                f"Admin password is not set. Visit {s.http_url or 'http://localhost:8000'}/admin to set the password."
            )

        # Apply app metadata from settings
        try:
            app.title = s.name
            app.description = s.description
        except Exception:
            pass

        # await ensure_models_bootstrapped()

        from ..payment.price import _update_prices
        from ..proxy import get_upstreams
        from ..upstream.helpers import refresh_upstreams_models_periodically

        _update_prices_task = asyncio.create_task(_update_prices())
        _initialize_upstreams_task = asyncio.create_task(initialize_upstreams())

        # ensure both setup tasks complete
        await asyncio.gather(
            _update_prices_task, _initialize_upstreams_task, return_exceptions=True
        )

        btc_price_task = asyncio.create_task(update_prices_periodically())
        pricing_task = asyncio.create_task(update_sats_pricing())
        if global_settings.models_refresh_interval_seconds > 0:
            # Pass the accessor (not its current value) so the loop sees providers
            # added/changed via reinitialize_upstreams() instead of staying pinned
            # to the startup snapshot.
            models_refresh_task = asyncio.create_task(
                refresh_upstreams_models_periodically(get_upstreams)
            )
        model_maps_refresh_task = asyncio.create_task(refresh_model_maps_periodically())
        payout_task = asyncio.create_task(periodic_payout())
        if global_settings.nsec:
            nip91_task = asyncio.create_task(announce_provider())
        analytics_task = asyncio.create_task(publish_usage_analytics())
        if global_settings.providers_refresh_interval_seconds > 0:
            providers_task = asyncio.create_task(providers_cache_refresher())
        key_reset_task = asyncio.create_task(periodic_key_reset())
        auto_topup_task = asyncio.create_task(periodic_auto_topup())
        refund_sweep_task = asyncio.create_task(periodic_refund_sweep())
        routstr_fee_task = asyncio.create_task(periodic_routstr_fee_payout())
        invoice_watcher_task = asyncio.create_task(periodic_invoice_watcher())

        yield

    except asyncio.CancelledError:
        # Expected during shutdown
        pass
    except Exception as e:
        logger.error(
            "Application startup failed",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        raise
    finally:
        logger.info("Application shutdown initiated")

        if btc_price_task is not None:
            btc_price_task.cancel()
        if pricing_task is not None:
            pricing_task.cancel()
        if payout_task is not None:
            payout_task.cancel()
        if nip91_task is not None:
            nip91_task.cancel()
        if analytics_task is not None:
            analytics_task.cancel()
        if providers_task is not None:
            providers_task.cancel()
        if models_refresh_task is not None:
            models_refresh_task.cancel()
        if model_maps_refresh_task is not None:
            model_maps_refresh_task.cancel()
        if key_reset_task is not None:
            key_reset_task.cancel()
        if auto_topup_task is not None:
            auto_topup_task.cancel()
        if refund_sweep_task is not None:
            refund_sweep_task.cancel()
        if routstr_fee_task is not None:
            routstr_fee_task.cancel()
        if invoice_watcher_task is not None:
            invoice_watcher_task.cancel()

        try:
            tasks_to_wait = []
            if btc_price_task is not None:
                tasks_to_wait.append(btc_price_task)
            if pricing_task is not None:
                tasks_to_wait.append(pricing_task)
            if payout_task is not None:
                tasks_to_wait.append(payout_task)
            if nip91_task is not None:
                tasks_to_wait.append(nip91_task)
            if analytics_task is not None:
                tasks_to_wait.append(analytics_task)
            if providers_task is not None:
                tasks_to_wait.append(providers_task)
            if models_refresh_task is not None:
                tasks_to_wait.append(models_refresh_task)
            if model_maps_refresh_task is not None:
                tasks_to_wait.append(model_maps_refresh_task)
            if key_reset_task is not None:
                tasks_to_wait.append(key_reset_task)
            if auto_topup_task is not None:
                tasks_to_wait.append(auto_topup_task)
            if refund_sweep_task is not None:
                tasks_to_wait.append(refund_sweep_task)
            if routstr_fee_task is not None:
                tasks_to_wait.append(routstr_fee_task)
            if invoice_watcher_task is not None:
                tasks_to_wait.append(invoice_watcher_task)

            if tasks_to_wait:
                await asyncio.gather(*tasks_to_wait, return_exceptions=True)
            logger.info("Background tasks stopped successfully")
        except Exception as e:
            logger.error(
                "Error stopping background tasks",
                extra={"error": str(e), "error_type": type(e).__name__},
            )


class _ImmutableStaticFiles(StaticFiles):
    """Static files with long Cache-Control for content-hashed Next.js assets.

    Files under `/_next/static/` are emitted with content hashes in their
    filenames and never mutate, so we serve them with a one-year immutable
    cache header so browsers and CDNs stop revalidating on every reload.
    """

    async def get_response(self, path: str, scope: Scope) -> StarletteResponse:
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


app = FastAPI(version=__version__, lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=global_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-routstr-request-id", "x-cashu"],
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Add exception handlers
app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore
app.add_exception_handler(Exception, general_exception_handler)


INFO_CONFIDENTIALITY_PROVIDERS = ("tinfoil", "ppq-private", "privatemode")


def _empty_info_routable_with_full_attestation() -> dict[str, list[str]]:
    return {provider: [] for provider in INFO_CONFIDENTIALITY_PROVIDERS}


def _safe_info_routable_with_full_attestation(value: object) -> dict[str, list[str]]:
    routable = _empty_info_routable_with_full_attestation()
    if not isinstance(value, dict):
        return routable
    for provider in INFO_CONFIDENTIALITY_PROVIDERS:
        raw_models = value.get(provider, [])
        if not isinstance(raw_models, list):
            return _empty_info_routable_with_full_attestation()
        model_ids: list[str] = []
        seen: set[str] = set()
        for raw_model_id in raw_models:
            if not isinstance(raw_model_id, str) or not raw_model_id.strip():
                return _empty_info_routable_with_full_attestation()
            model_id = raw_model_id.strip()
            normalized = model_id.lower()
            if normalized in seen:
                return _empty_info_routable_with_full_attestation()
            seen.add(normalized)
            model_ids.append(model_id)
        routable[provider] = sorted(model_ids)
    return routable


def _safe_info_client_confidentiality(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if value.get("mode") != "attested-tls-termination":
        return None
    if value.get("tls_terminates_in_attested_tee") is not True:
        return None
    if value.get("inbound_ehbp_ohttp_request_decryption") is not False:
        return None
    return {
        "mode": "attested-tls-termination",
        "tls_terminates_in_attested_tee": True,
        "inbound_ehbp_ohttp_request_decryption": False,
    }


def _safe_info_public_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _local_tee_status_has_info_cap_proof(
    proof_claims: dict[str, Any],
    *,
    attestation_evidence_digest: object,
) -> bool:
    if proof_claims.get("attestation_document_format") != "cap-attestation-proxy-status":
        return False
    if proof_claims.get("cap_claims_verified") is not True:
        return False
    if proof_claims.get("cap_state") != "unlocked":
        return False
    if proof_claims.get("client_confidentiality_boundary") != "attested-tls-termination":
        return False
    if proof_claims.get("tls_terminates_in_attested_tee") is not True:
        return False
    for claim in (
        "cap_attestation_url",
        "cap_claims_instance_id",
        "cap_status_url",
        "cap_tee_domain",
        "cap_tenant_id",
    ):
        if _safe_info_public_string(proof_claims.get(claim)) is None:
            return False
    for url_claim in ("cap_attestation_url", "cap_status_url"):
        claim_url = _safe_info_public_string(proof_claims.get(url_claim))
        if claim_url is None or urlsplit(claim_url).scheme != "https":
            return False
    if proof_claims.get("cap_status_digest") != attestation_evidence_digest:
        return False
    if not is_full_sha256_digest(proof_claims.get("routstr_config_measurement")):
        return False
    verification_steps = proof_claims.get("verification_steps")
    if not isinstance(verification_steps, dict):
        return False
    for step in (
        "cap_status_verified",
        "cap_claims_verified",
        "cap_state_unlocked",
        "tls_terminates_in_attested_tee",
        "freshness",
    ):
        if verification_steps.get(step) is not True:
            return False
    return True


def _local_tee_status_has_info_proof(value: dict[str, Any]) -> bool:
    attestation_evidence_digest = value.get("attestation_evidence_digest")
    if not is_full_sha256_digest(attestation_evidence_digest):
        return False

    local_verification = value.get("local_verification")
    if not isinstance(local_verification, dict):
        return False
    if local_verification.get("verified") is not True:
        return False
    verified_at = local_verification.get("verified_at")
    expires_at = local_verification.get("expires_at")
    if (
        not isinstance(verified_at, int)
        or isinstance(verified_at, bool)
        or not isinstance(expires_at, int)
        or isinstance(expires_at, bool)
        or verified_at > int(time.time())
        or expires_at <= int(time.time())
    ):
        return False
    if local_verification.get("evidence_digest") != attestation_evidence_digest:
        return False
    if not is_full_sha256_digest(local_verification.get("verified_claims_digest")):
        return False

    proof_claims = local_verification.get("proof_claims")
    if not isinstance(proof_claims, dict):
        return False
    if local_verification.get("verifier") == "cap-attestation-proxy":
        return _local_tee_status_has_info_cap_proof(
            proof_claims,
            attestation_evidence_digest=attestation_evidence_digest,
        )

    hpke_key_config_digest = value.get("hpke_key_config_digest")
    hpke_public_key_digest = value.get("hpke_public_key_digest")
    if not (
        is_full_sha256_digest(hpke_key_config_digest)
        and is_full_sha256_digest(hpke_public_key_digest)
    ):
        return False
    client_confidentiality = value.get("client_confidentiality")
    if not isinstance(client_confidentiality, dict):
        return False
    return (
        proof_claims.get("hpke_key_config_digest") == hpke_key_config_digest
        and proof_claims.get("hpke_public_key_digest") == hpke_public_key_digest
        and proof_claims.get("public_key_digest")
        == client_confidentiality.get("attested_tls_public_key_digest")
    )


def _safe_info_routstr_tee(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    public: dict[str, Any] = {
        "required": value.get("required") is True,
        "ready": value.get("ready") is True,
    }
    client_confidentiality = _safe_info_client_confidentiality(
        value.get("client_confidentiality")
    )
    if client_confidentiality is not None:
        public["client_confidentiality"] = client_confidentiality
    elif public["ready"] is True:
        public["ready"] = False
    if public["ready"] is True and not _local_tee_status_has_info_proof(value):
        public["ready"] = False
    return public


def _safe_info_confidentiality_summary(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    required = value.get("required") is True
    routable = _safe_info_routable_with_full_attestation(
        value.get("routable_with_full_attestation")
    )
    mode = value.get("mode")
    summary: dict[str, Any] = {
        "mode": mode if isinstance(mode, str) and mode.strip() else "disabled",
        "required": required,
        "end_to_end_ready": required
        and value.get("end_to_end_ready") is True
        and any(routable.values()),
        "routable_with_full_attestation": routable,
    }
    routstr_tee = _safe_info_routstr_tee(value.get("routstr_tee"))
    if routstr_tee is not None:
        summary["routstr_tee"] = routstr_tee
    if not isinstance(routstr_tee, dict) or routstr_tee.get("ready") is not True:
        summary["end_to_end_ready"] = False
        summary["routable_with_full_attestation"] = (
            _empty_info_routable_with_full_attestation()
        )
    return summary


@app.get("/v1/info")
async def info() -> dict:
    response = {
        "name": global_settings.name,
        "description": global_settings.description,
        "version": __version__,
        "npub": global_settings.npub,
        "mints": global_settings.cashu_mints,
        "http_url": global_settings.http_url,
        "onion_url": global_settings.onion_url,
        "child_key_cost_msats": global_settings.child_key_cost,
    }
    try:
        from ..proxy import get_confidentiality_status

        confidentiality = _safe_info_confidentiality_summary(
            get_confidentiality_status()
        )
    except Exception:
        confidentiality = None
    if confidentiality is not None and (
        confidentiality["required"] is True
        or any(confidentiality["routable_with_full_attestation"].values())
    ):
        response["confidentiality"] = confidentiality
    return response


@app.get("/v1/providers")
async def providers() -> RedirectResponse:
    return RedirectResponse("/v1/providers/")


@app.get("/v1/confidentiality/status")
async def confidentiality_status() -> dict:
    from ..proxy import get_confidentiality_status

    return get_confidentiality_status()


@app.get("/.well-known/routstr-attestation")
async def routstr_attestation_statement() -> dict:
    return get_routstr_attestation_statement()


@app.get("/v1/confidentiality/attestation")
async def routstr_confidentiality_attestation_statement() -> dict:
    return get_routstr_attestation_statement()


@app.get("/.well-known/hpke-keys")
async def routstr_hpke_keys() -> StarletteResponse:
    try:
        key_config = read_routstr_hpke_key_config()
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StarletteResponse(
        content=key_config,
        media_type="application/ohttp-keys",
    )


UI_DIST_PATH = Path(__file__).parent.parent.parent / "ui_out"

if UI_DIST_PATH.exists() and UI_DIST_PATH.is_dir():
    logger.info(f"Serving static UI from {UI_DIST_PATH}")

    app.mount(
        "/_next",
        _ImmutableStaticFiles(directory=UI_DIST_PATH / "_next", check_dir=True),
        name="next-static",
    )

    @app.get("/", include_in_schema=False)
    async def serve_root_ui() -> FileResponse:
        return FileResponse(UI_DIST_PATH / "index.html")

    # Serve the App Router RSC payload for the home page.
    @app.get("/index.txt", include_in_schema=False)
    async def serve_root_rsc() -> FileResponse:
        return FileResponse(UI_DIST_PATH / "index.txt", media_type="text/x-component")

    # Next.js is built with `trailingSlash: true`, so all UI page URLs end
    # with a slash (e.g. `/login/`). The proxy router catches `/{path:path}`
    # before FastAPI's `redirect_slashes` logic can normalize the URL, so we
    # must register both the with-slash and without-slash variants here.
    UI_PAGES = (
        "dashboard",
        "login",
        "model",
        "providers",
        "settings",
        "transactions",
        "balances",
        "logs",
        "usage",
        "unauthorized",
    )

    def _register_ui_page(name: str) -> None:
        page_dir = UI_DIST_PATH / name
        index_html = page_dir / "index.html"
        index_txt = page_dir / "index.txt"

        async def serve_page() -> FileResponse:
            return FileResponse(index_html)

        async def serve_page_rsc() -> FileResponse:
            return FileResponse(index_txt, media_type="text/x-component")

        app.add_api_route(
            f"/{name}",
            serve_page,
            methods=["GET"],
            include_in_schema=False,
            name=f"serve_{name}_ui",
        )
        app.add_api_route(
            f"/{name}/",
            serve_page,
            methods=["GET"],
            include_in_schema=False,
            name=f"serve_{name}_ui_slash",
        )
        app.add_api_route(
            f"/{name}/index.txt",
            serve_page_rsc,
            methods=["GET"],
            include_in_schema=False,
            name=f"serve_{name}_rsc",
        )

    for _page in UI_PAGES:
        _register_ui_page(_page)

    @app.get("/admin")
    async def admin_redirect() -> FileResponse:
        return FileResponse(UI_DIST_PATH / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def serve_favicon() -> FileResponse:
        icon_path = UI_DIST_PATH / "icon.ico"
        if icon_path.exists():
            return FileResponse(icon_path)
        return FileResponse(UI_DIST_PATH / "favicon.ico")

    @app.get("/icon.ico", include_in_schema=False)
    async def serve_icon() -> FileResponse:
        return FileResponse(UI_DIST_PATH / "icon.ico")

else:
    logger.warning(
        f"UI dist directory not found at {UI_DIST_PATH}, skipping static file serving"
    )

    @app.get("/", include_in_schema=False)
    async def root_fallback() -> dict:
        return {
            "name": global_settings.name,
            "description": global_settings.description,
            "version": __version__,
            "status": "running",
            "ui": "not available",
        }


app.include_router(models_router)
app.include_router(admin_router)
app.include_router(balance_router)
app.include_router(lightning_router)
app.include_router(deprecated_wallet_router)
app.include_router(providers_router)
app.include_router(proxy_router)
