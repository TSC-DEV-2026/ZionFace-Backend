from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional


class PessoaCreate(BaseModel):
    nome: str = Field(min_length=1)
    cpf: Optional[str] = None
    data_nascimento: Optional[str] = None  # "YYYY-MM-DD"


class UserCreate(BaseModel):
    email: EmailStr
    senha: str = Field(min_length=1)


class CadastroPayload(BaseModel):
    pessoa: PessoaCreate
    usuario: UserCreate


class LoginPayload(BaseModel):
    usuario: str = Field(min_length=1)  # email ou cpf
    senha: str = Field(min_length=1)


class MeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id_pessoa: int
    nome: str
    cpf: Optional[str] = None
    data_nascimento: Optional[str] = None

    user_id: int
    email: str