# app/schemas/user.py
from __future__ import annotations

from datetime import datetime, date
from pydantic import BaseModel, Field, EmailStr, field_validator, model_validator


# -----------------------------
# Payloads de entrada
# -----------------------------

class UsuarioPayload(BaseModel):
    email: EmailStr = Field(..., examples=["user@example.com"])
    senha: str = Field(..., min_length=6, examples=["string"])

class PessoaPayload(BaseModel):
    nome: str = Field(..., min_length=2, examples=["string"])
    cpf: str = Field(..., min_length=11, max_length=14, examples=["12345678901"])
    data_nascimento: date = Field(..., examples=["25-02-2026"])

    @field_validator("data_nascimento", mode="before")
    @classmethod
    def parse_data(cls, v):
        if isinstance(v, date):
            return v
        try:
            return datetime.strptime(v, "%d-%m-%Y").date()
        except ValueError:
            raise ValueError("data_nascimento inválida (use DD-MM-YYYY)")

class CadastroPayload(BaseModel):
    pessoa: PessoaPayload
    usuario: UsuarioPayload

    # Aceita payload antigo e normaliza para o novo:
    # {"usuario": {"nome": "...", ...}, "pessoa": {...}}
    # -> move usuario.nome para pessoa.nome
    @model_validator(mode="before")
    @classmethod
    def normalizar_payload_antigo(cls, data):
        if not isinstance(data, dict):
            return data

        pessoa = data.get("pessoa") or {}
        usuario = data.get("usuario") or {}

        nome_pessoa = pessoa.get("nome")
        nome_usuario = usuario.get("nome")

        if not nome_pessoa and nome_usuario:
            pessoa["nome"] = nome_usuario
            usuario.pop("nome", None)

        data["pessoa"] = pessoa
        data["usuario"] = usuario
        return data

    @field_validator("pessoa")
    @classmethod
    def validar_nome_obrigatorio(cls, pessoa: PessoaPayload) -> PessoaPayload:
        if not pessoa.nome or not pessoa.nome.strip():
            raise ValueError("pessoa.nome é obrigatório")
        return pessoa


class LoginPayload(BaseModel):
    email: EmailStr = Field(..., examples=["user@example.com"])
    senha: str = Field(..., examples=["string"])


# -----------------------------
# Responses
# -----------------------------

class MeResponse(BaseModel):
    id: int | None = None
    email: EmailStr
    nome: str
    cpf: str | None = None
    data_nascimento: date | None = None

    class Config:
        from_attributes = True