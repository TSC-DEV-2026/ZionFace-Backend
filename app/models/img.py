from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.database.connection import Base


class Img(Base):
    __tablename__ = "tb_img"

    id = Column(Integer, primary_key=True, index=True)
    id_pessoa = Column(Integer, ForeignKey("tb_pessoas.id"), nullable=False, index=True)

    codigo = Column(String(120), nullable=False, unique=True)

    tipo = Column(String(50), nullable=True)
    extensao = Column(String(10), nullable=True)
    tamanho = Column(Integer, nullable=True)
    sha256 = Column(String(64), nullable=True)
    status = Column(String(20), nullable=False, default="ativo")

    criado_em = Column(DateTime, nullable=False, server_default=func.now())
    atualizado_em = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())