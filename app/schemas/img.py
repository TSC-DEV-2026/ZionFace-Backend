from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class ImgCreate(BaseModel):
    codigo: str = Field(min_length=1, max_length=120)
    tipo: Optional[str] = None
    extensao: Optional[str] = None
    tamanho: Optional[int] = None
    sha256: Optional[str] = None
    status: Optional[str] = "ativo"


class ImgResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    id_pessoa: int
    codigo: str
    tipo: Optional[str] = None
    extensao: Optional[str] = None
    tamanho: Optional[int] = None
    sha256: Optional[str] = None
    status: str
    criado_em: str
    atualizado_em: str