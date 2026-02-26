from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.database.connection import Base


class User(Base):
    __tablename__ = "tb_usuario"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    senha_hash = Column(String(255), nullable=False)

    pessoa_id = Column(Integer, ForeignKey("tb_pessoas.id"), unique=True, nullable=False)
    pessoa = relationship("Pessoa", back_populates="user", uselist=False)