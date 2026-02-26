import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.settings import settings
from app.service.face_service import get_face_service
from app.utils.face_index import face_index


@dataclass
class VerifyResult:
    verified: bool
    distance: Optional[float]
    threshold: float
    best_reference_index: Optional[int]
    reason: str
    path: str
    model: str
    metric: str


@dataclass
class IdentifyCandidate:
    user_id: str
    distance: float
    ref_index: int


@dataclass
class IdentifyResult:
    identified: bool
    best_user_id: Optional[str]
    distance: Optional[float]
    threshold: float
    top_k: List[IdentifyCandidate]
    reason: str
    path: str
    model: str
    metric: str


_ref_cache: Dict[str, np.ndarray] = {}
_ref_meta_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.RLock()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # a e b já são normalizados (FaceService devolve L2-normalizado)
    sim = float(np.dot(a.reshape(-1).astype(np.float32), b.reshape(-1).astype(np.float32)))
    return 1.0 - sim


def _load_user_refs(user_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    with _cache_lock:
        if user_id in _ref_cache and user_id in _ref_meta_cache:
            return _ref_cache[user_id], _ref_meta_cache[user_id]

    ref_path = Path(settings.banco_fotos_dir) / str(user_id) / "ref.embedding.json"
    if not ref_path.exists():
        raise FileNotFoundError("Referência não encontrada")

    data = json.loads(ref_path.read_text(encoding="utf-8"))
    embeddings = data.get("embeddings") or []
    if not embeddings:
        raise FileNotFoundError("Referência vazia")

    mat = np.asarray(embeddings, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    if mat.shape[1] != 512:
        raise ValueError("Embeddings com dimensão inválida")

    # garante normalização (defensivo)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    mat = mat / norms

    with _cache_lock:
        _ref_cache[user_id] = mat
        _ref_meta_cache[user_id] = data

    return mat, data


def save_reference_embedding(user_id: str, image_bytes: bytes, append: bool = False) -> Dict[str, Any]:
    """Gera embedding para uma imagem de referência e salva em banco_fotos/<user_id>/ref.embedding.json."""

    svc = get_face_service()
    emb = svc.get_embedding_from_bytes(image_bytes)

    out_dir = Path(settings.banco_fotos_dir) / str(user_id)
    out_path = out_dir / "ref.embedding.json"

    payload: Dict[str, Any]
    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    else:
        payload = {}

    existing = payload.get("embeddings")
    if not isinstance(existing, list):
        existing = []

    if append and existing:
        existing.append(emb.tolist())
    else:
        existing = [emb.tolist()]

    payload.update(
        {
            "user_id": str(user_id),
            "model": settings.model_name,
            "metric": settings.distance_metric,
            "embeddings": existing,
        }
    )

    _atomic_write_json(out_path, payload)

    with _cache_lock:
        _ref_cache.pop(str(user_id), None)
        _ref_meta_cache.pop(str(user_id), None)

    # Índice 1:N incremental (se já estiver pronto)
    try:
        # ref_index correto é o último do arquivo
        face_index.add_embeddings(str(user_id), [existing[-1]])
    except Exception:
        pass

    return {
        "ok": True,
        "user_id": str(user_id),
        "saved_to": str(out_path),
        "append": bool(append),
        "total_refs_for_user": len(existing),
    }


def verify_1to1(user_id: str, selfie_bytes: bytes) -> VerifyResult:
    refs, _ = _load_user_refs(str(user_id))

    svc = get_face_service()
    emb = svc.get_embedding_from_bytes(selfie_bytes)

    dists = np.array([_cosine_distance(emb, r) for r in refs], dtype=np.float32)
    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])

    if best_dist <= settings.threshold_super_strict:
        return VerifyResult(
            verified=True,
            distance=best_dist,
            threshold=settings.threshold_super_strict,
            best_reference_index=best_idx,
            reason="Aprovado (super_strict)",
            path="arcface_onnx",
            model=settings.model_name,
            metric=settings.distance_metric,
        )

    if best_dist <= settings.threshold_strict:
        return VerifyResult(
            verified=True,
            distance=best_dist,
            threshold=settings.threshold_strict,
            best_reference_index=best_idx,
            reason="Aprovado (strict)",
            path="arcface_onnx",
            model=settings.model_name,
            metric=settings.distance_metric,
        )

    return VerifyResult(
        verified=False,
        distance=best_dist,
        threshold=settings.threshold_strict,
        best_reference_index=best_idx,
        reason="Reprovado (acima do threshold)",
        path="arcface_onnx",
        model=settings.model_name,
        metric=settings.distance_metric,
    )


def identify_1toN(selfie_bytes: bytes, top_k: Optional[int] = None) -> IdentifyResult:
    face_index.ensure_ready()

    k = int(top_k or settings.identify_top_k)
    if k < 2:
        k = 2

    svc = get_face_service()
    emb = svc.get_embedding_from_bytes(selfie_bytes)

    sims, idxs = face_index.search(emb, k)

    candidates: List[IdentifyCandidate] = []
    for sim, idx in zip(sims, idxs):
        if int(idx) < 0:
            continue
        meta = face_index.get_meta(int(idx))
        dist = 1.0 - float(sim)
        candidates.append(
            IdentifyCandidate(
                user_id=str(meta["user_id"]),
                distance=dist,
                ref_index=int(meta.get("ref_index", 0)),
            )
        )

    if not candidates:
        return IdentifyResult(
            identified=False,
            best_user_id=None,
            distance=None,
            threshold=settings.threshold_strict,
            top_k=[],
            reason="Sem candidatos no índice",
            path="ann_empty",
            model=settings.model_name,
            metric=settings.distance_metric,
        )

    candidates.sort(key=lambda x: x.distance)
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None

    margin_ok = True
    if second is not None:
        margin_ok = (second.distance - best.distance) >= settings.identify_margin

    if best.distance <= settings.threshold_super_strict and margin_ok:
        return IdentifyResult(
            identified=True,
            best_user_id=best.user_id,
            distance=best.distance,
            threshold=settings.threshold_super_strict,
            top_k=candidates,
            reason="Identificado (super_strict + margin) via ANN",
            path="ann_strict_margin",
            model=settings.model_name,
            metric=settings.distance_metric,
        )

    if best.distance <= settings.threshold_strict and margin_ok:
        return IdentifyResult(
            identified=True,
            best_user_id=best.user_id,
            distance=best.distance,
            threshold=settings.threshold_strict,
            top_k=candidates,
            reason="Identificado (strict + margin) via ANN",
            path="ann_strict_margin",
            model=settings.model_name,
            metric=settings.distance_metric,
        )

    return IdentifyResult(
        identified=False,
        best_user_id=None,
        distance=best.distance,
        threshold=settings.threshold_strict,
        top_k=candidates,
        reason="Desconhecido (falhou threshold/margin)",
        path="ann_reject",
        model=settings.model_name,
        metric=settings.distance_metric,
        )