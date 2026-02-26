from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.core.config import settings
from app.deps.auth import get_current_user

from app.models.user import User
from app.models.pessoa import Pessoa
from app.schemas.user import CadastroPayload, LoginPayload, MeResponse

from app.utils.password import gerar_hash_senha, verificar_senha
from app.utils.jwt_handler import create_access_token, create_refresh_token  # ajuste se o nome for diferente


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register")
def register(payload: CadastroPayload, db: Session = Depends(get_db)) -> JSONResponse:

    nome = payload.pessoa.nome.strip()
    cpf = payload.pessoa.cpf.strip()
    data_nascimento = payload.pessoa.data_nascimento

    email = payload.usuario.email.lower().strip()
    senha = payload.usuario.senha

    # validações simples
    if not cpf:
        raise HTTPException(status_code=400, detail="CPF é obrigatório")

    # email já existe?
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=409, detail="E-mail já cadastrado")

    # cpf já existe?
    if db.query(Pessoa).filter(Pessoa.cpf == cpf).first():
        raise HTTPException(status_code=409, detail="CPF já cadastrado")

    # cria pessoa
    pessoa = Pessoa(
        nome=nome,
        cpf=cpf,
        data_nascimento=data_nascimento,
    )
    db.add(pessoa)
    db.flush()  # pega ID da pessoa sem commit ainda

    # cria user vinculado
    user = User(
        email=email,
        senha_hash=gerar_hash_senha(senha),
        pessoa_id=pessoa.id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    db.refresh(pessoa)

    return JSONResponse(
        status_code=201,
        content={
            "id": user.id,
            "email": user.email,
            "cpf": pessoa.cpf,
            "data_nascimento": pessoa.data_nascimento.isoformat() if pessoa.data_nascimento else None,
        },
    )


@router.post("/login")
def login(payload: LoginPayload, db: Session = Depends(get_db)) -> JSONResponse:
    email = payload.email.lower().strip()
    senha = payload.senha

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=401, detail="Credenciais inválidas")

    if not verificar_senha(senha, user.senha_hash):
        raise HTTPException(status_code=401, detail="Credenciais inválidas")

    access_token = create_access_token({"sub": str(user.id)})
    refresh_token = create_refresh_token({"sub": str(user.id)})

    resp = JSONResponse(content={"ok": True})

    # cookies (ajuste nomes/flags conforme seu padrão)
    resp.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=False if settings.ENVIRONMENT == "dev" else True,
        samesite="lax",
        max_age=60 * settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        path="/",
    )
    resp.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=False if settings.ENVIRONMENT == "dev" else True,
        samesite="lax",
        max_age=60 * 60 * 24 * 7,
        path="/",
    )

    return resp


@router.get("/me")
def me(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # garante que a pessoa está carregada
    db.refresh(user)

    pessoa = getattr(user, "pessoa", None)

    return {
        "usuario": {
            "id": user.id,
            "email": user.email,
            "pessoa_id": user.pessoa_id,
        },
        "pessoa": None if not pessoa else {
            "id": pessoa.id,
            "nome": pessoa.nome,
            "cpf": pessoa.cpf,
            "data_nascimento": pessoa.data_nascimento.isoformat() if pessoa.data_nascimento else None,
        },
    }