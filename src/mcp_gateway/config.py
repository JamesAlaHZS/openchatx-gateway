"""Configuration loading for the MCP gateway.

Everything is driven by a single YAML file (see config.example.yaml).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

DEFAULT_ALLOWED_REDIRECT_URIS = [
    "https://claude.ai/api/mcp/auth_callback",
    "https://claude.com/api/mcp/auth_callback",
    "https://oauth-redirect.googleusercontent.com/*",
    "https://oauth-redirect-sandbox.googleusercontent.com/*",
    "https://oauth-redirect-test.googleusercontent.com/*",
    "https://gemini.google.com/*",
    "https://vertexaisearch.cloud.google.com/oauth-redirect",
    "https://vertexaisearch.cloud.google.com/static/oauth/oauth.html",
    "http://localhost:*/callback",
    "http://localhost:*/*",
    "http://127.0.0.1:*/callback",
    "http://127.0.0.1:*/*",
]

DEFAULT_SCOPES_SUPPORTED = [
    "mcp",
    "offline_access",
    "ACCESS_VIEW_MANAGE_MCP_CONTENT",
]


def _expand_env(value: str) -> str:
    """Expand ${VAR} / ${VAR:-default} references in config values."""

    def repl(match: re.Match[str]) -> str:
        var, default = match.group(1), match.group(2)
        if var in os.environ:
            return os.environ[var]
        if default is not None:
            return default
        raise ValueError(f"Config references undefined environment variable: {var}")

    return _ENV_PATTERN.sub(repl, value)


def _expand_tree(node: object) -> object:
    if isinstance(node, str):
        return _expand_env(node)
    if isinstance(node, list):
        return [_expand_tree(item) for item in node]
    if isinstance(node, dict):
        return {key: _expand_tree(value) for key, value in node.items()}
    return node


class UserConfig(BaseModel):
    """A local user allowed to log in to the gateway's authorization server."""

    username: str
    password_hash: str | None = None
    password: str | None = None

    @model_validator(mode="after")
    def _check_password(self) -> UserConfig:
        if bool(self.password_hash) == bool(self.password):
            raise ValueError(
                f"User {self.username!r}: set exactly one of 'password_hash' or 'password'"
            )
        return self


class StaticOAuthClientConfig(BaseModel):
    """A pre-registered OAuth client for platforms that require or fall back to
    manual Client ID / Client Secret entry (e.g. Gemini Web Advanced features
    or Gemini Enterprise)."""

    client_id: str
    client_secret: str | None = None
    client_name: str = "OpenChatX Gemini Client"
    redirect_uris: list[str] = Field(default_factory=list)


class ServerConfig(BaseModel):
    public_url: str
    host: str = "0.0.0.0"
    port: int = 8000
    trusted_proxy_ips: str = "127.0.0.1"

    @field_validator("public_url")
    @classmethod
    def _normalize_public_url(cls, v: str) -> str:
        return v.rstrip("/")


class AuthConfig(BaseModel):
    users: list[UserConfig]
    encryption_key: str

    @field_validator("encryption_key")
    @classmethod
    def _check_encryption_key(cls, v: str) -> str:
        if not v:
            raise ValueError(
                "auth.encryption_key is empty -- set MCP_GATEWAY_ENCRYPTION_KEY or "
                "MCP_GATEWAY_ENCRYPTION_KEY_FILE"
            )
        return v

    session_secret: str | None = None
    access_token_expiry_seconds: int = 3600
    refresh_token_expiry_seconds: int = 60 * 60 * 24 * 30
    authorization_code_expiry_seconds: int = 300
    login_session_expiry_seconds: int = 60 * 60 * 8
    allowed_client_redirect_uris: list[str] | None = Field(
        default_factory=lambda: list(DEFAULT_ALLOWED_REDIRECT_URIS)
    )
    static_clients: list[StaticOAuthClientConfig] = Field(default_factory=list)
    scopes_supported: list[str] = Field(
        default_factory=lambda: list(DEFAULT_SCOPES_SUPPORTED)
    )


class BackendAuthConfig(BaseModel):
    """How the gateway authenticates against an upstream MCP server."""

    type: Literal["none", "bearer", "headers", "oauth"] = "none"
    token: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    scopes: list[str] | None = None
    prefer_dcr: bool = False
    client_id: str | None = None
    client_secret: str | None = None

    @model_validator(mode="after")
    def _check(self) -> BackendAuthConfig:
        if self.type == "bearer" and not self.token:
            raise ValueError("backend auth type 'bearer' requires 'token'")
        if self.type == "headers" and not self.headers:
            raise ValueError("backend auth type 'headers' requires 'headers'")
        if self.type != "oauth" and self.client_secret:
            raise ValueError("client_secret is only valid for auth type 'oauth'")
        return self


class BackendConfig(BaseModel):
    """An upstream MCP server exposed through the gateway."""

    url: str
    enabled: bool = True
    auth: BackendAuthConfig = Field(default_factory=BackendAuthConfig)
    headers: dict[str, str] = Field(default_factory=dict)


class StorageConfig(BaseModel):
    path: str = "data/gateway.db"


class GatewayConfig(BaseModel):
    server: ServerConfig
    auth: AuthConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
    backends: dict[str, BackendConfig] = Field(default_factory=dict)

    @field_validator("backends")
    @classmethod
    def _validate_backend_names(cls, v: dict[str, BackendConfig]) -> dict[str, BackendConfig]:
        for name in v:
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name):
                raise ValueError(
                    f"Backend name {name!r} must be alphanumeric with '-'/'_' "
                    "(it is used as a tool-name prefix)"
                )
        return v


def _read_encryption_key_file() -> str | None:
    key_file = os.environ.get("MCP_GATEWAY_ENCRYPTION_KEY_FILE")
    if not key_file:
        return None
    try:
        return Path(key_file).read_text().strip()
    except OSError as exc:
        raise ValueError(
            f"MCP_GATEWAY_ENCRYPTION_KEY_FILE={key_file!r} could not be read: {exc}"
        ) from exc


def load_config(path: str | Path) -> GatewayConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise TypeError(f"Config file {path} must contain a YAML mapping")
    expanded = _expand_tree(raw)

    key_from_file = _read_encryption_key_file()
    if key_from_file is not None:
        auth = expanded.setdefault("auth", {}) if isinstance(expanded, dict) else {}
        if isinstance(auth, dict):
            auth["encryption_key"] = key_from_file

    return GatewayConfig.model_validate(expanded)


def load_storage_path(path: str | Path) -> str:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise TypeError(f"Config file {path} must contain a YAML mapping")
    storage_raw = _expand_tree(raw.get("storage") or {})
    return str(storage_raw.get("path", StorageConfig().path))
