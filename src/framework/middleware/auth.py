"""Development stub authentication middleware.

Provides get_auth_token and authenticate_user used by various routers.
In production this should be replaced with real validation (JWT/OAuth etc.).
For now we accept any bearer token (or missing token) and return a static user.
"""
from __future__ import annotations

from typing import Optional

from fastapi import Header, HTTPException

# In dev we allow everything unless FKS_REQUIRE_AUTH=true
import os
REQUIRE = os.getenv("FKS_REQUIRE_AUTH", "false").lower() in {"1","true","yes"}

class AuthenticatedUser(dict):
    @property
    def id(self) -> str:  # type: ignore
        return self.get("id", "dev-user")  # type: ignore

DEV_USER = AuthenticatedUser({
    "id": "dev-user",
    "email": "dev@example.com",
    "name": "Developer",
    "roles": ["developer"],
    "scopes": ["*"],
})

async def get_auth_token(authorization: Optional[str] = Header(default=None)) -> Optional[str]:
    """Extract raw bearer token (if provided)."""
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return authorization

def authenticate_user(token: Optional[str]) -> AuthenticatedUser:
    """Validate token or raise HTTPException.

    Development: always returns DEV_USER unless REQUIRE flag set and no token.
    """
    if REQUIRE and not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    # Future: decode/validate token here
    return DEV_USER

__all__ = ["get_auth_token", "authenticate_user", "AuthenticatedUser"]
