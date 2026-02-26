from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt
from jwt import PyJWTError

from app.core.config import settings


@dataclass(frozen=True)
class TokenPair:
    access_token: str
    refresh_token: str


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _encode(payload: Dict[str, Any], *, token_type: str, expires_in: timedelta) -> str:
    now = _utcnow()
    exp = now + expires_in

    to_encode = dict(payload)
    to_encode.update(
        {
            "type": token_type,
            "iat": int(now.timestamp()),
            "exp": int(exp.timestamp()),
        }
    )

    return jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def create_access_token(payload: Dict[str, Any]) -> str:
    return _encode(
        payload,
        token_type="access",
        expires_in=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(payload: Dict[str, Any]) -> str:
    return _encode(
        payload,
        token_type="refresh",
        expires_in=timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )


def create_token_pair(*, sub: str, extra_claims: Optional[Dict[str, Any]] = None) -> TokenPair:
    base: Dict[str, Any] = {"sub": str(sub)}
    if extra_claims:
        base.update(extra_claims)

    return TokenPair(
        access_token=create_access_token(base),
        refresh_token=create_refresh_token(base),
    )


def decode_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["exp", "iat"]},
        )
    except PyJWTError as e:
        raise ValueError("Token inválido") from e


def assert_token_type(claims: Dict[str, Any], expected_type: str) -> None:
    t = claims.get("type")
    if t != expected_type:
        raise ValueError("Tipo de token inválido")


def get_sub(claims: Dict[str, Any]) -> str:
    sub = claims.get("sub")
    if not sub:
        raise ValueError("Token sem 'sub'")
    return str(sub)


def decode_and_validate(token: str, *, expected_type: str) -> Dict[str, Any]:
    claims = decode_token(token)
    assert_token_type(claims, expected_type)
    _ = get_sub(claims)
    return claims