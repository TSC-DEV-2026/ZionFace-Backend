from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.face import router as face_router
from app.routes.user import router as auth_router
from app.service.face_service import warmup


def create_app() -> FastAPI:
    app = FastAPI(title="Reconhecimento Facial API", version="1.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(face_router)
    app.include_router(auth_router)

    @app.on_event("startup")
    def on_startup() -> None:
        warmup()

    return app


app = create_app()