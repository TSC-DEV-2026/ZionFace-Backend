import uuid
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from app.core.config import settings


def criar_token(payload: dict, expires_in_minutes: int) -> str:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=int(expires_in_minutes))

    to_encode = dict(payload)
    to_encode["exp"] = int(exp.timestamp())
    to_encode["iat"] = int(now.timestamp())
    to_encode["jti"] = uuid.uuid4().hex

    return jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_token(token: str) -> dict:
    return jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
    )


def verificar_token(token: str) -> dict | None:
    try:
        return decode_token(token)
    except JWTError:
        return None