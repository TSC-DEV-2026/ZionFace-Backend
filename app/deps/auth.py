from fastapi import Depends, Request, HTTPException, status
from sqlalchemy.orm import Session
from jose import jwt, JWTError

from app.database.connection import get_db
from app.core.config import settings
from app.models.user import User


def _get_token_from_request(request: Request) -> str | None:
    # 1) cookie (ajuste o nome do cookie no settings)
    cookie_name = getattr(settings, "AUTH_COOKIE_NAME", "access_token")
    token = request.cookies.get(cookie_name)
    if token:
        return token

    # 2) Authorization: Bearer <token>
    auth = request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()

    return None


def _decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[getattr(settings, "JWT_ALG", "HS256")],
            options={"verify_aud": False},
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido ou expirado",
        )


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
    token = _get_token_from_request(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Não autenticado",
        )

    payload = _decode_access_token(token)

    # Ajuste aqui conforme o que você grava no JWT:
    # - recomendado: "sub" com o id do user
    user_id = payload.get("sub") or payload.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token sem subject (sub)",
        )

    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuário não encontrado",
        )

    return user