from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app.service.face_service import get_face_service
from app.service.face_validation import identify_1toN, save_reference_embedding, verify_1to1
from app.utils.face_index import face_index

router = APIRouter(prefix="/face", tags=["face"])


class VerifyPayload(BaseModel):
    img1_base64: str = Field(min_length=10)
    img2_base64: str = Field(min_length=10)


@router.get("/health")
def health():
    return {"ok": True}


@router.post("/verify/base64")
def verify_base64(payload: VerifyPayload):
    try:
        svc = get_face_service()
        return svc.verify_1to1(payload.img1_base64, payload.img2_base64)

    except RuntimeError as e:
        # Ex.: onnxruntime indisponível neste ambiente / modelo não encontrado
        msg = str(e).lower()
        if "onnxruntime" in msg or "modelo onnx" in msg:
            raise HTTPException(status_code=501, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    except ValueError as e:
        # Ex.: "Nenhum rosto detectado"
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        raise HTTPException(status_code=500, detail="Erro interno no reconhecimento facial")


@router.post("/enroll/{user_id}")
async def enroll_face(
    user_id: str,
    file: UploadFile = File(...),
    append: bool = Query(False),
):
    try:
        image_bytes = await file.read()
        result = save_reference_embedding(user_id=user_id, image_bytes=image_bytes, append=append)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no enroll: {str(e)}")


@router.post("/verify/{user_id}")
async def verify_face(
    user_id: str,
    file: UploadFile = File(...),
):
    try:
        selfie_bytes = await file.read()
        result = verify_1to1(user_id=user_id, selfie_bytes=selfie_bytes)
        return result.__dict__
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Referência do usuário não encontrada")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Falha na validação 1:1: {str(e)}")


@router.post("/identify")
async def identify_face(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=2, le=50),
):
    try:
        selfie_bytes = await file.read()
        result = identify_1toN(selfie_bytes=selfie_bytes, top_k=top_k)
        return {
            "identified": result.identified,
            "best_user_id": result.best_user_id,
            "distance": result.distance,
            "threshold": result.threshold,
            "top_k": [c.__dict__ for c in result.top_k],
            "reason": result.reason,
            "path": result.path,
            "model": result.model,
            "metric": result.metric,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Falha na identificação 1:N: {str(e)}")


@router.get("/index/status")
def index_status():
    st = face_index.status()
    return {
        "ready": st.ready,
        "total_users": st.total_users,
        "total_embeddings": st.total_embeddings,
        "index_type": st.index_type,
        "banco_fotos_dir": st.banco_fotos_dir,
    }


@router.post("/index/rebuild")
def index_rebuild():
    try:
        st = face_index.rebuild()
        return {
            "ready": st.ready,
            "total_users": st.total_users,
            "total_embeddings": st.total_embeddings,
            "index_type": st.index_type,
            "banco_fotos_dir": st.banco_fotos_dir,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))