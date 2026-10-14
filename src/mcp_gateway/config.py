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
    """Expand ${VAR} /${VAR:-default} references in config values."""

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
    # Exactly one of these must be set. `password_hash` is a bcrypt hash
    # (generate with `mcp-gateway hash-password`); `password` is plaintext and
    # only meant for quick local testing.
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
