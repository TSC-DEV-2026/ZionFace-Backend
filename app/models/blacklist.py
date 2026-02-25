from sqlalchemy import Column, Integer, String, DateTime, Index
from app.database.connection import Base


class TokenBlacklist(Base):
    __tablename__ = "tb_blacklist"

    id = Column(Integer, primary_key=True, index=True)
    jti = Column(String(64), nullable=False, unique=True, index=True)
    expira_em = Column(DateTime, nullable=False)


Index("ix_tb_blacklist_jti", TokenBlacklist.jti)
Index("ix_tb_blacklist_expira_em", TokenBlacklist.expira_em)