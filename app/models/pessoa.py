from sqlalchemy import Column, Date, Integer, String
from sqlalchemy.orm import relationship

from app.database.connection import Base


class Pessoa(Base):
    __tablename__ = "tb_pessoas"

    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String(100), nullable=False)

    cpf = Column(String(14), unique=True, nullable=True, index=True)
    data_nascimento = Column(Date, nullable=True)

    user = relationship("User", back_populates="pessoa", uselist=False)
    # imgs = relationship("Img", back_populates="pessoa")