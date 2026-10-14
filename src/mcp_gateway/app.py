"""Application assembly: FastAPI outer app + FastMCP gateway."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mcp_gateway.config import GatewayConfig, load_config
from mcp_gateway.gateway import build_gateway
from mcp_gateway.oauth_server import GatewayOAuthProvider
from mcp_gateway.ratelimit import RateLimiter
from mcp_gateway.state import GatewayState
from mcp_gateway.storage import Storage
from mcp_gateway.upstream import BackendManager
from mcp_gateway.users import SessionManager
from mcp_gateway.web import build_auth_router, build_oauth_router, build_ui_router

logger = logging.getLogger(__name__)

# Strip fastmcp's default handler and let it propagate to root logger
_fastmcp_logger = logging.getLogger("fastmcp")
_fastmcp_logger.handlers.clear()
_fastmcp_logger.propagate = True

# Housekeeping cadence: expired tokens/codes/transactions, revoked-session
# records, and unused DCR/CIMD client registrations are reclaimed on this interval.
PURGE_INTERVAL_SECONDS = 15 * 60

# Anonymous, expensive endpoints: bounded per source IP.
LOGIN_RATE_LIMIT = (10, 60.0)  # 10 attempts / 60s per IP
REGISTER_RATE_LIMIT = (20, 60.0)  # 20 registrations / 60s per IP


def _warn_on_plaintext_passwords(config: GatewayConfig) -> None:
    plaintext_users = [u.username for u in config.auth.users if u.password is not None]
    if plaintext_users:
        logger.warning(
            "User(s) %s configured with a plaintext 'password' in the config file. "
            "This is intended for local testing only -- use 'password_hash' "
            "(mcp-gateway hash-password) for any real deployment.",
            ", ".join(sorted(plaintext_users)),
        )


def create_app(config: GatewayConfig | str) -> FastAPI:
    if isinstance(config, str):
        config = load_config(config)

    _warn_on_plaintext_passwords(config)

    logger.debug("Opening storage at %s", config.storage.path)
    storage = Storage(config.storage.path, config.auth.encryption_key)
    provider = GatewayOAuthProvider(config, storage)
    manager = BackendManager(config, storage)

    clients = {
        name: manager.build_client(name, backend)
        for name, backend in config.backends.items()
        if backend.enabled
    }
    logger.debug("Built %d backend client(s)", len(clients))

    mcp = build_gateway(config, provider, manager, clients)
    # The MCP endpoint lives at <public_url>/mcp; auth + well-known routes sit
    # at the root of the same app per RFC 8414/9728.
    mcp_app = mcp.http_app(path="/mcp")

    async def _purge_loop() -> None:
        while True:
            await asyncio.sleep(PURGE_INTERVAL_SECONDS)
            try:
                storage.purge_expired()
            except Exception:
                logger.exception("Periodic storage purge failed")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting up: purging expired tokens/codes/transactions")
        storage.purge_expired()
        purge_task = asyncio.create_task(_purge_loop())
        logger.info("MCP Gateway ready at %s", config.server.public_url)
        async with mcp_app.lifespan(mcp_app):
            yield
        purge_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await purge_task
        logger.info("Shutting down: closing storage")
        storage.close()

    app = FastAPI(title="MCP Gateway", lifespan=lifespan, docs_url=None, redoc_url=None)

    session_secret = config.auth.session_secret or storage.get_or_create_secret(
        "session_secret"
    )
    app.state = GatewayState()
    app.state.config = config
    app.state.storage = storage
    app.state.oauth_provider = provider
    app.state.backend_manager = manager
    app.state.backend_clients = clients
    app.state.sessions = SessionManager(
        session_secret, config.auth.login_session_expiry_seconds, storage
    )
    app.state.login_limiter = RateLimiter(*LOGIN_RATE_LIMIT)
    app.state.register_limiter = RateLimiter(*REGISTER_RATE_LIMIT)

    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        if request.url.path == "/register" and request.method == "POST":
            ip = request.client.host if request.client else "unknown"
            if not app.state.register_limiter.allow(ip):
                logger.warning("Client registration rate limit exceeded for %s", ip)
                return JSONResponse(
                    {"error": "invalid_request", "error_description": "Too many registrations"},
                    status_code=429,
                )

        response = await call_next(request)

        response.headers["X-Frame-Options"] = "DENY"
        response.headers.setdefault(
            "Content-Security-Policy",
            # Allow form-action to https: / http: so cross-origin OAuth callback
            # redirects (e.g. to https://oauth-redirect.googleusercontent.com)
            # are never blocked by strict browser CSP enforcement.
            "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self' https: http:",
        )
        # Use strict-origin-when-cross-origin so Google's OAuth popup relay
        # retains origin context during the callback handshake.
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Content-Type-Options"] = "nosniff"

        if request.url.path.startswith("/auth/api"):
            response.headers["Cache-Control"] = "no-store"

        return response

    app.include_router(build_auth_router())
    app.include_router(build_oauth_router())
    app.include_router(build_ui_router())

    @app.get("/")
    async def root():
        from fastapi.responses import RedirectResponse

        return RedirectResponse("/ui/")

    # Everything else (MCP endpoint, /authorize, /token, /register, /revoke,
    # /.well-known/*) is handled by the FastMCP app mounted as catch-all.
    app.mount("/", mcp_app)

    return app
