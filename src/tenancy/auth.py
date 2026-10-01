"""Keycloak authentication: obtain a token, verify it, extract the tenants.

Tenancy is derived ONLY from a cryptographically verified access token (the
`groups` claim, populated by a Keycloak group-membership mapper) - never from a
value the caller supplies. `TenantContext` is the verified result handed to the
retrieval layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import jwt
import requests
from jwt import PyJWKClient


def _base() -> str:
    return os.environ.get("KEYCLOAK_URL", "http://localhost:8081").rstrip("/")


def _realm_url() -> str:
    return f"{_base()}/realms/{os.environ.get('KEYCLOAK_REALM', 'rag')}"


def _client_id() -> str:
    return os.environ.get("KEYCLOAK_CLIENT_ID", "rag-api")


@dataclass(frozen=True)
class TenantContext:
    username: str
    tenants: tuple[str, ...]


def get_access_token(username: str, password: str | None = None) -> str:
    """Password-grant login against the local Keycloak realm (demo/dev only)."""
    password = password or os.environ["RAG_DEMO_PASSWORD"]
    resp = requests.post(
        f"{_realm_url()}/protocol/openid-connect/token",
        data={
            "grant_type": "password",
            "client_id": _client_id(),
            "username": username,
            "password": password,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def verify_token(token: str) -> TenantContext:
    """Verify signature (JWKS), issuer, audience and expiry; return the tenants."""
    signing_key = PyJWKClient(f"{_realm_url()}/protocol/openid-connect/certs").get_signing_key_from_jwt(token)
    claims = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience=_client_id(),
        issuer=_realm_url(),
    )
    tenants = tuple(claims.get("groups", ()))
    if not tenants:
        raise PermissionError(f"{claims.get('preferred_username')} belongs to no tenant")
    return TenantContext(claims.get("preferred_username", "?"), tenants)
