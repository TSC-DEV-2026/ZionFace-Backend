import re
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.core.config import settings

from app.models.user import Pessoa
from app.models.user import User
from app.models.blacklist import TokenBlacklist
from app.models.img import Img

from app.schemas.user import CadastroPayload, LoginPayload, MeResponse
from app.schemas.img import ImgCreate, ImgResponse

from app.utils.password import gerar_hash_senha, verificar_senha
from app.utils.jwt_handler import criar_token, verificar_token, decode_token


router = APIRouter(prefix="/auth", tags=["auth"])


def _is_email(v: str) -> bool:
    return re.match(r"[^@]+@[^@]+\.[^@]+", v or "") is not None


def _cpf_digits(v: str) -> str:
    return "".join(ch for ch in (v or "") if ch.isdigit())


def _cookie_env():
    is_prod = settings.ENVIRONMENT == "prod"
    domain = settings.COOKIE_DOMAIN if settings.COOKIE_DOMAIN else (None if not is_prod else settings.COOKIE_DOMAIN)
    secure = bool(settings.COOKIE_SECURE) if settings.COOKIE_DOMAIN is not None else (True if is_prod else False)
    return {"domain": domain, "secure": secure, "samesite": settings.COOKIE_SAMESITE}


def _set_auth_cookies(response: JSONResponse, access_token: str, refresh_token: str):
    env = _cookie_env()

    response.set_cookie(
        "access_token",
        access_token,
        httponly=True,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
        **env,
    )
    response.set_cookie(
        "refresh_token",
        refresh_token,
        httponly=True,
        max_age=int(settings.REFRESH_TOKEN_EXPIRE_DAYS) * 24 * 60 * 60,
        path="/",
        **env,
    )
    response.set_cookie(
        "logged_user",
        "true",
        httponly=False,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
        **env,
    )


def _clear_auth_cookies(response: JSONResponse):
    env = _cookie_env()
    domain = env.get("domain")

    response.delete_cookie("access_token", path="/", domain=domain)
    response.delete_cookie("refresh_token", path="/", domain=domain)
    response.delete_cookie("logged_user", path="/", domain=domain)


def _require_access_token(request: Request, db: Session) -> dict:
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Token ausente")

    payload = verificar_token(token)
    if not payload or payload.get("tipo") != "access":
        raise HTTPException(status_code=401, detail="Token inválido")

    jti = payload.get("jti")
    if not jti:
        raise HTTPException(status_code=401, detail="Token inválido")

    if db.query(TokenBlacklist).filter(TokenBlacklist.jti == jti).first():
        raise HTTPException(status_code=401, detail="Token expirado ou inválido")

    return payload


@router.post("/register", status_code=201)
def register(payload: CadastroPayload, db: Session = Depends(get_db)):
    email = payload.usuario.email.lower().strip()
    cpf = _cpf_digits(payload.pessoa.cpf) if payload.pessoa.cpf else None

    if cpf and db.query(Pessoa).filter(Pessoa.cpf == cpf).first():
        raise HTTPException(status_code=400, detail="CPF já cadastrado")

    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email já cadastrado")

    data_nascimento = None
    if payload.pessoa.data_nascimento:
        try:
            data_nascimento = datetime.strptime(payload.pessoa.data_nascimento, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="data_nascimento inválida (use YYYY-MM-DD)")

    pessoa = Pessoa(
        nome=payload.pessoa.nome.strip(),
        cpf=cpf,
        data_nascimento=data_nascimento,
    )
    db.add(pessoa)
    db.commit()
    db.refresh(pessoa)

    user = User(
        id_pessoa=pessoa.id,
        email=email,
        senha_hash=gerar_hash_senha(payload.usuario.senha),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {"ok": True, "id_pessoa": pessoa.id, "user_id": user.id}


@router.post("/login")
def login(payload: LoginPayload, db: Session = Depends(get_db)):
    u = payload.usuario.strip().lower()

    user: User | None = None

    if _is_email(u):
        user = db.query(User).filter(User.email == u).first()
    else:
        cpf = _cpf_digits(u)
        pessoa = db.query(Pessoa).filter(Pessoa.cpf == cpf).first()
        if pessoa:
            user = db.query(User).filter(User.id_pessoa == pessoa.id).first()

    if not user:
        raise HTTPException(status_code=401, detail="Usuário ou senha inválidos")

    if not verificar_senha(payload.senha, user.senha_hash):
        raise HTTPException(status_code=401, detail="Usuário ou senha inválidos")

    access_token = criar_token(
        {"id_pessoa": user.id_pessoa, "sub": user.email, "tipo": "access"},
        expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
    )
    refresh_token = criar_token(
        {"id_pessoa": user.id_pessoa, "sub": user.email, "tipo": "refresh"},
        expires_in_minutes=int(settings.REFRESH_TOKEN_EXPIRE_DAYS) * 24 * 60,
    )

    response = JSONResponse(content={"message": "Login com sucesso"})
    _set_auth_cookies(response, access_token, refresh_token)
    return response


@router.get("/me", response_model=MeResponse)
def me(request: Request, db: Session = Depends(get_db)):
    payload = _require_access_token(request, db)

    id_pessoa = payload.get("id_pessoa")
    if not id_pessoa:
        raise HTTPException(status_code=401, detail="Token inválido")

    pessoa = db.query(Pessoa).filter(Pessoa.id == id_pessoa).first()
    if not pessoa:
        raise HTTPException(status_code=401, detail="Pessoa não encontrada")

    user = db.query(User).filter(User.id_pessoa == pessoa.id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")

    return MeResponse(
        id_pessoa=pessoa.id,
        nome=pessoa.nome,
        cpf=pessoa.cpf,
        data_nascimento=pessoa.data_nascimento.isoformat() if pessoa.data_nascimento else None,
        user_id=user.id,
        email=user.email,
    )


@router.post("/refresh")
def refresh(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("refresh_token")
    if not token:
        raise HTTPException(status_code=400, detail="refresh_token não fornecido")

    payload = verificar_token(token)
    if not payload or payload.get("tipo") != "refresh":
        raise HTTPException(status_code=401, detail="refresh_token inválido ou expirado")

    email = payload.get("sub")
    id_pessoa = payload.get("id_pessoa")
    if not email or not id_pessoa:
        raise HTTPException(status_code=401, detail="Token inválido")

    user = db.query(User).filter(User.email == email).first()
    if not user or user.id_pessoa != id_pessoa:
        raise HTTPException(status_code=401, detail="Usuário não encontrado")

    novo_access = criar_token(
        {"id_pessoa": user.id_pessoa, "sub": user.email, "tipo": "access"},
        expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
    )

    response = JSONResponse(content={"message": "Token renovado"})
    env = _cookie_env()
    response.set_cookie(
        "access_token",
        novo_access,
        httponly=True,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
        **env,
    )
    response.set_cookie(
        "logged_user",
        "true",
        httponly=False,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
        **env,
    )
    return response


@router.post("/logout")
def logout(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if token:
        try:
            payload = decode_token(token)
            jti = payload.get("jti")
            exp = payload.get("exp")
            if jti and exp:
                exp_dt = datetime.fromtimestamp(int(exp), tz=timezone.utc).replace(tzinfo=None)
                db.add(TokenBlacklist(jti=jti, expira_em=exp_dt))
                db.commit()
        except Exception:
            pass

    response = JSONResponse(content={"message": "Logout realizado com sucesso"})
    _clear_auth_cookies(response)
    return response


@router.post("/img", response_model=ImgResponse, status_code=201)
def img_add(request: Request, body: ImgCreate, db: Session = Depends(get_db)):
    payload = _require_access_token(request, db)
    id_pessoa = payload.get("id_pessoa")
    if not id_pessoa:
        raise HTTPException(status_code=401, detail="Token inválido")

    if db.query(Img).filter(Img.codigo == body.codigo).first():
        raise HTTPException(status_code=400, detail="codigo já existe")

    img = Img(
        id_pessoa=int(id_pessoa),
        codigo=body.codigo,
        tipo=body.tipo,
        extensao=body.extensao,
        tamanho=body.tamanho,
        sha256=body.sha256,
        status=body.status or "ativo",
    )
    db.add(img)
    db.commit()
    db.refresh(img)

    return ImgResponse(
        id=img.id,
        id_pessoa=img.id_pessoa,
        codigo=img.codigo,
        tipo=img.tipo,
        extensao=img.extensao,
        tamanho=img.tamanho,
        sha256=img.sha256,
        status=img.status,
        criado_em=img.criado_em.isoformat(),
        atualizado_em=img.atualizado_em.isoformat(),
    )


@router.get("/img", response_model=list[ImgResponse])
def img_list(request: Request, db: Session = Depends(get_db)):
    payload = _require_access_token(request, db)
    id_pessoa = payload.get("id_pessoa")
    if not id_pessoa:
        raise HTTPException(status_code=401, detail="Token inválido")

    imgs = (
        db.query(Img)
        .filter(Img.id_pessoa == int(id_pessoa))
        .order_by(Img.id.desc())
        .all()
    )

    return [
        ImgResponse(
            id=i.id,
            id_pessoa=i.id_pessoa,
            codigo=i.codigo,
            tipo=i.tipo,
            extensao=i.extensao,
            tamanho=i.tamanho,
            sha256=i.sha256,
            status=i.status,
            criado_em=i.criado_em.isoformat(),
            atualizado_em=i.atualizado_em.isoformat(),
        )
        for i in imgs
    ]