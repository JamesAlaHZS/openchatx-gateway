"""Client-facing OAuth 2.1 authorization server for the gateway.

Implements the MCP authorization spec (2025-11-25) for the client leg, with
full compatibility for Claude, Gemini Web (Custom Connected Apps / Google
Account Linking), Gemini Enterprise, and local CLI clients:

- OAuth 2.1 authorization code flow with mandatory PKCE (S256)
- Dynamic Client Registration (RFC 7591) at /register
- Pre-registered Static OAuth Clients (for Gemini Web Advanced features & Enterprise)
- Client ID Metadata Documents (CIMD) with multi-origin redirect support for
  Google Account Linking (accountlinking.google.com -> oauth-redirect.googleusercontent.com)
- Flexible Token Endpoint Client Authentication (accepts client_secret_basic,
  client_secret_post, private_key_jwt, and none seamlessly)
- Authorization Server Metadata (RFC 8414) & Protected Resource Metadata (RFC 9728)
- Short-lived opaque access tokens and rotating refresh tokens, stored hashed
"""

from __future__ import annotations

import base64
import hmac
import logging
import secrets
import time
from typing import Any
from urllib.parse import unquote, urlparse

from fastmcp.server.auth.auth import (
    AccessToken as FastMCPAccessToken,
)
from fastmcp.server.auth.auth import (
    OAuthProvider,
    PrivateKeyJWTClientAuthenticator,
    TokenHandler,
)
from fastmcp.server.auth.cimd import CIMDClientManager
from fastmcp.server.auth.oauth_proxy.models import ProxyDCRClient
from fastmcp.server.auth.redirect_validation import validate_redirect_uri
from mcp.server.auth.handlers.metadata import MetadataHandler
from mcp.server.auth.middleware.client_auth import AuthenticationError
from mcp.server.auth.provider import (
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.server.auth.routes import build_metadata, cors_middleware
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyUrl
from starlette.requests import Request
from starlette.routing import Route

from mcp_gateway.config import DEFAULT_ALLOWED_REDIRECT_URIS, GatewayConfig
from mcp_gateway.storage import Storage

logger = logging.getLogger(__name__)

CIMD_REFRESH_SECONDS = 3600


class GatewayClient(ProxyDCRClient):
    """Registered client with lenient scope and multi-origin redirect validation.

    - Accepts any requested scopes (including Google Account Linking's
      ``ACCESS_VIEW_MANAGE_MCP_CONTENT`` and ``offline_access``).
    - Validates redirect URIs against ``allowed_redirect_uri_patterns`` first
      (allowing Google's ``oauth-redirect*.googleusercontent.com`` callbacks
      even when the client was registered via CIMD on ``accountlinking.google.com``
      or via static config).
    """

    def validate_scope(self, requested_scope: str | None) -> list[str] | None:
        if requested_scope is None:
            if self.scope:
                return self.scope.split(" ")
            return None
        return [s for s in requested_scope.split(" ") if s]

    def validate_redirect_uri(self, redirect_uri: AnyUrl | None) -> AnyUrl:
        if redirect_uri is not None and self.allowed_redirect_uri_patterns:
            if validate_redirect_uri(
                redirect_uri=redirect_uri,
                allowed_patterns=self.allowed_redirect_uri_patterns,
            ):
                return redirect_uri
        return super().validate_redirect_uri(redirect_uri)


class FlexibleClientAuthenticator(PrivateKeyJWTClientAuthenticator):
    """Client authenticator that seamlessly handles client_secret_basic,
    client_secret_post, private_key_jwt, and public (none) PKCE exchanges.

    Google's OAuth relay (Gemini Web / Gemini Enterprise) may register a client
    with ``client_secret_post`` (or omit ``token_endpoint_auth_method``, which
    defaults to ``client_secret_basic`` or ``client_secret_post``) and then
    present credentials via HTTP Basic Auth header OR POST form body, or rely
    purely on PKCE without sending the secret. Standard SDK validation rejects
    mismatched transport methods with 401/400; this authenticator accepts any
    valid credential presentation for the registered client.
    """

    async def authenticate_request(self, request: Request) -> OAuthClientInformationFull:
        form_data = await request.form()
        grant_type = str(form_data.get("grant_type", ""))

        # 1. If client_assertion is present (CIMD private_key_jwt), delegate to parent
        if "client_assertion" in form_data:
            return await super().authenticate_request(request)

        # 2. Extract client_id and optional client_secret from Basic header or form body
        client_id: str | None = None
        provided_secret: str | None = None

        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("basic "):
            try:
                decoded = base64.b64decode(auth_header[6:].strip()).decode("utf-8")
                if ":" in decoded:
                    raw_id, raw_secret = decoded.split(":", 1)
                    client_id = unquote(raw_id)
                    provided_secret = unquote(raw_secret)
            except Exception as exc:
                raise AuthenticationError("Invalid Basic Authorization header") from exc

        form_client_id = form_data.get("client_id")
        if isinstance(form_client_id, str) and form_client_id:
            if client_id is None:
                client_id = form_client_id
            elif client_id != form_client_id:
                raise AuthenticationError("Mismatched client_id in header and body")

        form_secret = form_data.get("client_secret")
        if isinstance(form_secret, str) and form_secret and provided_secret is None:
            provided_secret = form_secret

        # 3. Fallback: if Gemini's token exchange omitted client_id on authorization_code
        # or refresh_token grant, infer client_id from the stored code/refresh_token
        if not client_id and hasattr(self, "provider"):
            provider = getattr(self, "provider", None)
            if provider is not None and hasattr(provider, "storage"):
                if grant_type == "authorization_code":
                    code_val = form_data.get("code")
                    if isinstance(code_val, str) and code_val:
                        code_data = provider.storage.get_auth_code(code_val)
                        if code_data:
                            client_id = code_data.get("client_id")
                elif grant_type == "refresh_token":
                    rt_val = form_data.get("refresh_token")
                    if isinstance(rt_val, str) and rt_val:
                        rt_data = provider.storage.get_refresh_token(rt_val)
                        if rt_data:
                            client_id = rt_data.get("client_id")

        if not client_id:
            raise AuthenticationError("Missing client_id in token request")

        client = await self.provider.get_client(client_id)
        if client is None:
            raise AuthenticationError(f"Invalid or unregistered client_id: {client_id}")

        # 4. If the client has a configured client_secret and the caller provided one,
        # verify it in constant time. If the caller omitted client_secret (e.g. public
        # PKCE client behaviour), allow PKCE code_verifier validation in TokenHandler
        # to authenticate the code exchange.
        if client.client_secret and provided_secret is not None:
            if not hmac.compare_digest(
                client.client_secret.encode("utf-8"),
                provided_secret.encode("utf-8"),
            ):
                raise AuthenticationError("Invalid client_secret")

        return client


class GatewayOAuthProvider(OAuthProvider):
    """A self-contained OAuth 2.1 AS backed by SQLite and config-file users."""

    def __init__(self, config: GatewayConfig, storage: Storage):
        public_url = config.server.public_url
        super().__init__(
            base_url=public_url,
            issuer_url=public_url,
            client_registration_options=ClientRegistrationOptions(
                enabled=True,
                valid_scopes=None,  # accept any; scopes are informational here
                default_scopes=list(config.auth.scopes_supported),
            ),
            revocation_options=RevocationOptions(enabled=True),
            required_scopes=None,
        )
        self.config = config
        self.storage = storage
        # Always merge user-configured redirect URIs with the built-in defaults
        # so existing config.yaml files created for Claude also allow Gemini Web.
        configured_uris = config.auth.allowed_client_redirect_uris
        if configured_uris is not None:
            merged: list[str] = list(configured_uris)
            for default_uri in DEFAULT_ALLOWED_REDIRECT_URIS:
                if default_uri not in merged:
                    merged.append(default_uri)
            self._allowed_redirect_patterns: list[str] | None = merged
        else:
            self._allowed_redirect_patterns = None

        self.cimd_manager = CIMDClientManager(
            enable_cimd=True,
            default_scope=" ".join(config.auth.scopes_supported),
            allowed_redirect_uri_patterns=self._allowed_redirect_patterns,
        )
        self._static_clients: dict[str, GatewayClient] = {}
        self._init_static_clients()

    def _init_static_clients(self) -> None:
        """Register static OAuth clients from config.yaml into memory and storage."""
        default_redirects = [
            AnyUrl("https://oauth-redirect.googleusercontent.com/r/openchatx"),
            AnyUrl("https://vertexaisearch.cloud.google.com/oauth-redirect"),
            AnyUrl("https://claude.ai/api/mcp/auth_callback"),
            AnyUrl("http://localhost:8080/callback"),
        ]
        for sc in self.config.auth.static_clients:
            redirect_uris = (
                [AnyUrl(u) for u in sc.redirect_uris]
                if sc.redirect_uris
                else default_redirects
            )
            auth_method = "client_secret_post" if sc.client_secret else "none"
            client = GatewayClient(
                client_id=sc.client_id,
                client_secret=sc.client_secret,
                client_name=sc.client_name,
                redirect_uris=redirect_uris,
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                token_endpoint_auth_method=auth_method,
                scope=" ".join(self.config.auth.scopes_supported),
                allowed_redirect_uri_patterns=self._allowed_redirect_patterns,
            )
            self._static_clients[sc.client_id] = client
            self.storage.save_client(sc.client_id, client.model_dump(mode="json"))
            self.storage.mark_client_used(sc.client_id)
            logger.info("Loaded static OAuth client %r (client_id=%s)", sc.client_name, sc.client_id)

    # ------------------------------------------------------------------ clients

    def _client_from_record(self, record: dict[str, Any]) -> GatewayClient:
        client = GatewayClient.model_validate(record)
        # Always apply the current allowlist patterns so updating config.yaml
        # immediately takes effect for existing cached clients in gateway.db.
        client.allowed_redirect_uri_patterns = self._allowed_redirect_patterns
        return client

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        if client_id in self._static_clients:
            return self._static_clients[client_id]

        record = self.storage.get_client(client_id)
        if record is not None:
            client = self._client_from_record(record)
            if client.cimd_document is not None and (
                client.cimd_fetched_at is None
                or time.time() - client.cimd_fetched_at > CIMD_REFRESH_SECONDS
            ):
                logger.debug("Refreshing stale CIMD client document for %s", client_id)
                refreshed = await self._fetch_cimd_client(client_id)
                if refreshed is not None:
                    return refreshed
            else:
                logger.debug("Loaded cached client %s from storage", client_id)
            return client

        if self.cimd_manager.is_cimd_client_id(client_id):
            logger.debug("Client %s not found in storage; fetching from CIMD", client_id)
            cimd_client = await self._fetch_cimd_client(client_id)
            if cimd_client is not None:
                return cimd_client
            # Fallback for Google Account Linking CIMD URLs (e.g.
            # https://accountlinking.google.com/clientidmetadata/...) if the
            # metadata document cannot be fetched or fails strict same-origin checks:
            parsed = urlparse(client_id)
            if parsed.scheme == "https" and (
                parsed.hostname == "accountlinking.google.com"
                or (parsed.hostname or "").endswith(".google.com")
                or (parsed.hostname or "").endswith(".googleusercontent.com")
            ):
                logger.info(
                    "Synthesizing fallback CIMD client for Google Account Linking (%s)",
                    client_id,
                )
                fallback = GatewayClient(
                    client_id=client_id,
                    client_secret=None,
                    client_name="Google Gemini Connected App",
                    redirect_uris=[
                        AnyUrl("https://oauth-redirect.googleusercontent.com/r/gemini")
                    ],
                    grant_types=["authorization_code", "refresh_token"],
                    response_types=["code"],
                    token_endpoint_auth_method="none",
                    scope=" ".join(self.config.auth.scopes_supported),
                    allowed_redirect_uri_patterns=self._allowed_redirect_patterns,
                )
                self.storage.save_client(
                    client_id, fallback.model_dump(mode="json"), is_cimd=True
                )
                return fallback

        logger.debug("Client %s is not a known or CIMD client, return None", client_id)
        return None

    async def _fetch_cimd_client(self, client_id: str) -> GatewayClient | None:
        try:
            cimd_client = await self.cimd_manager.get_client(client_id)
        except Exception as exc:
            logger.warning("Failed to fetch CIMD document for %s: %s", client_id, exc)
            return None
        if cimd_client is None:
            return None
        client = GatewayClient.model_validate(cimd_client.model_dump(mode="json"))
        client.allowed_redirect_uri_patterns = self._allowed_redirect_patterns
        self.storage.save_client(client_id, client.model_dump(mode="json"), is_cimd=True)
        return client

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        client = GatewayClient.model_validate(client_info.model_dump(mode="json"))
        client.allowed_redirect_uri_patterns = self._allowed_redirect_patterns
        self.storage.save_client(client.client_id, client.model_dump(mode="json"))
        logger.info(
            "Registered new OAuth client %r via DCR (%s, redirect_uris=%s)",
            client_info.client_name or client_info.client_id,
            client.client_id,
            [str(u) for u in (client.redirect_uris or [])],
        )

    # ------------------------------------------------------------------ authorize

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        txn_id = secrets.token_urlsafe(32)
        client_name = getattr(client, "client_name", None)
        self.storage.save_txn(
            txn_id,
            {
                "client_id": client.client_id,
                "client_name": client_name,
                "redirect_uri": str(params.redirect_uri),
                "redirect_uri_provided_explicitly": params.redirect_uri_provided_explicitly,
                "state": params.state,
                "code_challenge": params.code_challenge,
                "scopes": params.scopes or [],
                "resource": params.resource,
                "created_at": time.time(),
            },
            ttl_seconds=600,
        )
        return f"{str(self.base_url).rstrip('/')}/ui/authorize?txn={txn_id}"

    def describe_txn(self, txn_id: str) -> dict[str, Any] | None:
        """Public transaction info for the login/consent UI."""
        txn = self.storage.get_txn(txn_id)
        if txn is None:
            return None

        redirect = urlparse(txn["redirect_uri"])
        return {
            "txn_id": txn_id,
            "client_id": txn["client_id"],
            "client_name": txn.get("client_name"),
            "redirect_uri": txn["redirect_uri"],
            "redirect_host": redirect.hostname or "",
            "is_loopback_redirect": (redirect.hostname or "").lower()
            in {"localhost", "127.0.0.1", "::1"},
            "scopes": txn.get("scopes") or [],
        }

    def complete_authorization(self, txn_id: str, *, subject: str, approve: bool) -> str:
        """Finish a parked authorization request; returns the redirect URL."""
        txn = self.storage.get_txn(txn_id)
        if txn is None:
            raise AuthorizeError(
                error="invalid_request", error_description="Unknown or expired transaction"
            )
        self.storage.delete_txn(txn_id)

        if not approve:
            logger.info(
                "Authorization denied by %s for client %s", subject, txn["client_id"]
            )
            return construct_redirect_uri(
                txn["redirect_uri"], error="access_denied", state=txn.get("state")
            )

        logger.info("Authorization approved by %s for client %s", subject, txn["client_id"])
        code = secrets.token_urlsafe(32)
        expires_at = time.time() + self.config.auth.authorization_code_expiry_seconds
        self.storage.save_auth_code(
            code,
            {
                "client_id": txn["client_id"],
                "redirect_uri": txn["redirect_uri"],
                "redirect_uri_provided_explicitly": txn["redirect_uri_provided_explicitly"],
                "code_challenge": txn["code_challenge"],
                "scopes": txn.get("scopes") or [],
                "resource": txn.get("resource"),
                "subject": subject,
                "expires_at": expires_at,
            },
            expires_at=expires_at,
        )
        return construct_redirect_uri(
            txn["redirect_uri"], code=code, state=txn.get("state")
        )

    # ------------------------------------------------------------------ codes & tokens

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        data = self.storage.get_auth_code(authorization_code)
        if data is None or data["client_id"] != client.client_id:
            return None
        return AuthorizationCode(
            code=authorization_code,
            scopes=data["scopes"],
            expires_at=data["expires_at"],
            client_id=data["client_id"],
            code_challenge=data["code_challenge"],
            redirect_uri=AnyUrl(data["redirect_uri"]),
            redirect_uri_provided_explicitly=data["redirect_uri_provided_explicitly"],
            resource=data.get("resource"),
            subject=data.get("subject"),
        )

    def _issue_tokens(
        self,
        *,
        client_id: str,
        scopes: list[str],
        subject: str | None,
        resource: str | None,
    ) -> OAuthToken:
        access_token = secrets.token_urlsafe(43)
        refresh_token = secrets.token_urlsafe(43)
        now = time.time()
        access_expiry = self.config.auth.access_token_expiry_seconds
        effective_scopes = scopes if scopes else list(self.config.auth.scopes_supported)
        self.storage.save_access_token(
            access_token,
            client_id=client_id,
            scopes=effective_scopes,
            subject=subject,
            resource=resource,
            expires_at=now + access_expiry,
        )
        self.storage.save_refresh_token(
            refresh_token,
            client_id=client_id,
            scopes=effective_scopes,
            subject=subject,
            resource=resource,
            expires_at=now + self.config.auth.refresh_token_expiry_seconds,
        )
        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=access_expiry,
            scope=" ".join(effective_scopes) if effective_scopes else "mcp",
            refresh_token=refresh_token,
        )

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        if self.storage.get_auth_code(authorization_code.code) is None:
            raise TokenError(error="invalid_grant", error_description="Authorization code expired")
        self.storage.delete_auth_code(authorization_code.code)
        self.storage.mark_client_used(client.client_id)
        logger.info(
            "Issuing access/refresh token pair to client %s (subject=%s)",
            client.client_id,
            authorization_code.subject,
        )
        return self._issue_tokens(
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            subject=authorization_code.subject,
            resource=authorization_code.resource,
        )

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        data = self.storage.get_refresh_token(refresh_token)
        if data is None or data["client_id"] != client.client_id:
            return None
        return RefreshToken(
            token=refresh_token,
            client_id=data["client_id"],
            scopes=data["scopes"],
            expires_at=int(data["expires_at"]),
            subject=data.get("subject"),
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        data = self.storage.get_refresh_token(refresh_token.token)
        if data is None or data["client_id"] != client.client_id:
            raise TokenError(error="invalid_grant", error_description="Invalid refresh token")
        self.storage.revoke_refresh_token(refresh_token.token)
        effective_scopes = scopes or data["scopes"]
        logger.debug("Rotating refresh token for client %s", client.client_id)
        return self._issue_tokens(
            client_id=client.client_id,
            scopes=effective_scopes,
            subject=data.get("subject"),
            resource=data.get("resource"),
        )

    async def load_access_token(self, token: str) -> FastMCPAccessToken | None:
        data = self.storage.get_access_token(token)
        if data is None:
            return None
        return FastMCPAccessToken(
            token=token,
            client_id=data["client_id"],
            scopes=data["scopes"],
            expires_at=int(data["expires_at"]),
            resource=data.get("resource"),
            subject=data.get("subject"),
            claims={"iss": str(self.issuer_url), "sub": data.get("subject")},
        )

    async def revoke_token(self, token: FastMCPAccessToken | RefreshToken) -> None:
        logger.info("Revoking token for client %s", token.client_id)
        self.storage.delete_access_token(token.token)
        self.storage.revoke_refresh_token(token.token)

    # ------------------------------------------------------------------ routes

    def get_routes(self, mcp_path: str | None = None) -> list[Route]:
        """Standard OAuth routes with CIMD and FlexibleClientAuthenticator patched in."""
        routes = super().get_routes(mcp_path)
        token_endpoint_url = f"{str(self.base_url).rstrip('/')}/token"
        patched: list[Route] = []
        for route in routes:
            if (
                isinstance(route, Route)
                and route.path == "/token"
                and route.methods is not None
                and "POST" in route.methods
            ):
                authenticator = FlexibleClientAuthenticator(
                    provider=self,
                    cimd_manager=self.cimd_manager,
                    token_endpoint_url=token_endpoint_url,
                )
                handler = TokenHandler(provider=self, client_authenticator=authenticator)
                patched.append(
                    Route(
                        path="/token",
                        endpoint=cors_middleware(handler.handle, ["POST", "OPTIONS"]),
                        methods=["POST", "OPTIONS"],
                    )
                )
            elif isinstance(route, Route) and route.path.startswith(
                "/.well-known/oauth-authorization-server"
            ):
                metadata = build_metadata(
                    self.base_url,
                    self.service_documentation_url,
                    self.client_registration_options or ClientRegistrationOptions(),
                    self.revocation_options or RevocationOptions(),
                )
                metadata.client_id_metadata_document_supported = True
                existing = metadata.token_endpoint_auth_methods_supported or []
                methods = [
                    *existing,
                    *(
                        m
                        for m in (
                            "client_secret_basic",
                            "client_secret_post",
                            "private_key_jwt",
                            "none",
                        )
                        if m not in existing
                    ),
                ]
                metadata.token_endpoint_auth_methods_supported = methods
                if self.config.auth.scopes_supported:
                    metadata.scopes_supported = list(self.config.auth.scopes_supported)
                handler = MetadataHandler(metadata)
                patched.append(
                    Route(
                        path=route.path,
                        endpoint=cors_middleware(handler.handle, ["GET", "OPTIONS"]),
                        methods=route.methods or ["GET", "OPTIONS"],
                        name=route.name,
                        include_in_schema=route.include_in_schema,
                    )
                )
                patched.append(
                    Route(
                        path="/.well-known/openid-configuration",
                        endpoint=cors_middleware(handler.handle, ["GET", "OPTIONS"]),
                        methods=["GET", "OPTIONS"],
                    )
                )
            else:
                patched.append(route)
        return patched
