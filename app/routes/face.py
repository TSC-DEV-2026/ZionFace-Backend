from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .face_service import warmup, save_reference_embedding, verify_1to1, identify_1toN
from .face_index import face_index

app = FastAPI(title="Reconhecimento Facial API", version="1.1.0")


@app.on_event("startup")
def on_startup() -> None:
    warmup()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/face/enroll/{user_id}")
async def enroll_face(
    user_id: str,
    file: UploadFile = File(...),
    append: bool = Query(False),
):
    try:
        image_bytes = await file.read()
        result = save_reference_embedding(user_id=user_id, image_bytes=image_bytes, append=append)
        return JSONResponse(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no enroll: {str(e)}")


@app.post("/face/verify/{user_id}")
async def verify_face(
    user_id: str,
    file: UploadFile = File(...),
):
    try:
        selfie_bytes = await file.read()
        result = verify_1to1(user_id=user_id, selfie_bytes=selfie_bytes)
        return JSONResponse(result.__dict__)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Referência do usuário não encontrada")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Falha na validação 1:1: {str(e)}")


@app.post("/face/identify")
async def identify_face(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=2, le=50),
):
    try:
        selfie_bytes = await file.read()
        result = identify_1toN(selfie_bytes=selfie_bytes, top_k=top_k)
        payload = {
            "identified": result.identified,
            "best_user_id": result.best_user_id,
            "distance": result.distance,
            "threshold": result.threshold,
            "top_k": [c.__dict__ for c in result.top_k],
            "reason": result.reason,
            "path": result.path,
            "model": result.model,
            "detector_fast": result.detector_fast,
            "detector_fallback": result.detector_fallback,
            "metric": result.metric,
        }
        return JSONResponse(payload)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Falha na identificação 1:N: {str(e)}")


@app.get("/face/index/status")
def index_status():
    st = face_index.status()
    return {
        "ready": st.ready,
        "total_users": st.total_users,
        "total_embeddings": st.total_embeddings,
        "index_type": st.index_type,
        "banco_fotos_dir": st.banco_fotos_dir,
    }


@app.post("/face/index/rebuild")
def index_rebuild():
    st = face_index.rebuild()
    return {
        "ready": st.ready,
        "total_users": st.total_users,
        "total_embeddings": st.total_embeddings,
        "index_type": st.index_type,
        "banco_fotos_dir": st.banco_fotos_dir,
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)