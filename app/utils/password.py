from passlib.context import CryptContext


_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


def gerar_hash_senha(senha: str) -> str:
    return _pwd.hash(senha)


def verificar_senha(senha_plana: str, senha_hash: str) -> bool:
    return _pwd.verify(senha_plana, senha_hash)