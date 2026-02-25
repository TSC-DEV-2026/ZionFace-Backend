from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.service.face_service import get_face_service

router = APIRouter(prefix="/face", tags=["face"])

class VerifyPayload(BaseModel):
    img1_base64: str = Field(min_length=10)
    img2_base64: str = Field(min_length=10)

@router.get("/health")
def health():
    return {"ok": True}

@router.post("/verify")
def verify(payload: VerifyPayload):
    try:
        svc = get_face_service()
        return svc.verify_1to1(payload.img1_base64, payload.img2_base64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Erro interno no reconhecimento facial")