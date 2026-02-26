from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database.connection import Base


class Img(Base):
    __tablename__ = "tb_img"

    id = Column(Integer, primary_key=True, index=True)
    id_pessoa = Column(Integer, ForeignKey("tb_pessoas.id", onupdate="CASCADE", ondelete="RESTRICT"), nullable=False)

    codigo = Column(String(120), unique=True, nullable=False, index=True)

    tipo = Column(String(50), nullable=True)
    extensao = Column(String(10), nullable=True)
    tamanho = Column(Integer, nullable=True)
    sha256 = Column(String(64), nullable=True)
    status = Column(String(20), nullable=False, server_default="ativo")

    criado_em = Column(DateTime, nullable=False, server_default=func.now())
    atualizado_em = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())

    pessoa = relationship("Pessoa", back_populates="imgs")