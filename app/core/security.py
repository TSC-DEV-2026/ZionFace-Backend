from typing import Optional
from fastapi import Request, HTTPException, status
from jose import jwt, JWTError

from app.core.config import settings

def get_token_from_request(request: Request) -> Optional[str]:
    # 1) cookie
    token = request.cookies.get(settings.AUTH_COOKIE_NAME)
    if token:
        return token

    # 2) Authorization: Bearer <token>
    auth = request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()

    return None

def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALG],
            options={"verify_aud": False},
        )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido ou expirado",
        )