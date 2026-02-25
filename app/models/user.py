from sqlalchemy import Column, Integer, String, ForeignKey
from app.database.connection import Base


class User(Base):
    __tablename__ = "tb_user"

    id = Column(Integer, primary_key=True, index=True)
    id_pessoa = Column(Integer, ForeignKey("tb_pessoas.id"), nullable=False, index=True)

    email = Column(String(100), nullable=False, unique=True, index=True)
    senha_hash = Column(String, nullable=False)