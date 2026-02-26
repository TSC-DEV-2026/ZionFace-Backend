from sqlalchemy import Column, DateTime, Integer, String
from app.database.connection import Base


class TokenBlacklist(Base):
    __tablename__ = "tb_blacklist"

    id = Column(Integer, primary_key=True)
    jti = Column(String(64), unique=True, nullable=False, index=True)
    expira_em = Column(DateTime, nullable=False, index=True)