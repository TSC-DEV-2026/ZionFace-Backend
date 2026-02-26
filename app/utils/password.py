# app/utils/password.py
from __future__ import annotations

import bcrypt


def gerar_hash_senha(senha: str) -> str:
    if not isinstance(senha, str):
        raise ValueError("senha inválida")

    senha = senha.strip()
    if not senha:
        raise ValueError("senha vazia")

    # bcrypt tem limite de 72 bytes; trate antes (melhor: recusar do que truncar)
    if len(senha.encode("utf-8")) > 72:
        raise ValueError("senha muito longa (máximo 72 bytes)")

    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(senha.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verificar_senha(senha: str, senha_hash: str) -> bool:
    if not senha or not senha_hash:
        return False
    try:
        return bcrypt.checkpw(senha.encode("utf-8"), senha_hash.encode("utf-8"))
    except Exception:
        return False