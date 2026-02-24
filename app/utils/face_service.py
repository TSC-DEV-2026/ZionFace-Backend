import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from deepface import DeepFace

from .settings import settings
from .face_index import face_index


@dataclass
class VerifyResult:
    verified: bool
    distance: Optional[float]
    threshold: float
    best_reference_index: Optional[int]
    reason: str
    path: str
    model: str
    detector_fast: str
    detector_fallback: str
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
    detector_fast: str
    detector_fallback: str
    metric: str


_ref_cache: Dict[str, np.ndarray] = {}
_ref_meta_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.RLock()


def warmup() -> None:
    dummy = np.zeros((112, 112, 3), dtype=np.uint8)
    DeepFace.represent(
        img_path=dummy,
        model_name=settings.model_name,
        detector_backend="skip",
        enforce_detection=False,
        align=True,
        normalization="base",
    )


def _read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Não foi possível decodificar a imagem")
    return img


def _resize_if_needed(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    max_side = max(h, w)
    if max_side <= settings.max_side:
        return img_bgr
    scale = settings.max_side / float(max_side)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _extract_best_face(img_bgr: np.ndarray, detector: str) -> np.ndarray:
    faces = DeepFace.extract_faces(
        img_path=img_bgr,
        detector_backend=detector,
        enforce_detection=False,
        align=True,
        grayscale=False,
    )
    if not faces:
        raise ValueError("Nenhuma face detectada")

    best = None
    best_area = -1

    for f in faces:
        region = f.get("facial_area") or {}
        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        w = int(region.get("w", 0))
        h = int(region.get("h", 0))
        area = w * h
        if area > best_area:
            best_area = area
            best = (x, y, w, h)

    if best is None:
        raise ValueError("Falha ao selecionar a melhor face")

    x, y, w, h = best
    if w * h < settings.min_face_area:
        raise ValueError("Face muito pequena na imagem")

    pad = int(0.25 * max(w, h))
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, img_bgr.shape[1])
    y1 = min(y + h + pad, img_bgr.shape[0])

    face = img_bgr[y0:y1, x0:x1]
    if face.size == 0:
        raise ValueError("Recorte de face inválido")
    return face


def _represent_skip(face_bgr: np.ndarray) -> np.ndarray:
    reps = DeepFace.represent(
        img_path=face_bgr,
        model_name=settings.model_name,
        detector_backend="skip",
        enforce_detection=False,
        align=True,
        normalization="base",
    )
    if not reps or not isinstance(reps, list):
        raise ValueError("Falha ao gerar embedding")
    emb = reps[0].get("embedding")
    if emb is None:
        raise ValueError("Embedding ausente")
    v = np.asarray(emb, dtype=np.float32).reshape(-1)
    if v.shape[0] != 512:
        raise ValueError("Embedding com dimensão inválida")
    return v


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    sim = float(np.dot(a, b) / (na * nb))
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

    with _cache_lock:
        _ref_cache[user_id] = mat
        _ref_meta_cache[user_id] = data

    return mat, data


def save_reference_embedding(user_id: str, image_bytes: bytes, append: bool = False) -> Dict[str, Any]:
    img = _read_image_from_bytes(image_bytes)
    img = _resize_if_needed(img)

    # FAST-FIRST: tenta detector rápido no enroll
    if settings.use_face_crop:
        try:
            face = _extract_best_face(img, settings.detector_fast)  # opencv
            used_detector = settings.detector_fast
        except Exception:
            face = _extract_best_face(img, settings.detector_fallback)  # retinaface (fallback)
            used_detector = settings.detector_fallback
    else:
        face = img
        used_detector = "skip_crop"

    emb = _represent_skip(face)

    out_dir = Path(settings.banco_fotos_dir) / str(user_id)
    out_dir.mkdir(parents=True, exist_ok=True)
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
            "detector_fast": settings.detector_fast,
            "detector_fallback": settings.detector_fallback,
            "metric": settings.distance_metric,
            "max_side": settings.max_side,
            "use_face_crop": settings.use_face_crop,
            "enroll_detector_used": used_detector,
            "embeddings": existing,
        }
    )

    # Escrita mais rápida (sem indent) também ajuda um pouco
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with _cache_lock:
        _ref_cache.pop(str(user_id), None)
        _ref_meta_cache.pop(str(user_id), None)

    try:
        # adiciona incrementalmente o último embedding
        face_index.add_embeddings(str(user_id), [existing[-1]])
    except Exception:
        pass

    return {
        "ok": True,
        "user_id": str(user_id),
        "saved_to": str(out_path),
        "append": bool(append),
        "total_refs_for_user": len(existing),
        "enroll_detector_used": used_detector,
    }


def verify_1to1(user_id: str, selfie_bytes: bytes) -> VerifyResult:
    refs, _ = _load_user_refs(str(user_id))

    img = _read_image_from_bytes(selfie_bytes)
    img = _resize_if_needed(img)

    face_fast = _extract_best_face(img, settings.detector_fast) if settings.use_face_crop else img
    emb_fast = _represent_skip(face_fast)

    dists_fast = np.array([_cosine_distance(emb_fast, r) for r in refs], dtype=np.float32)
    best_idx_fast = int(np.argmin(dists_fast))
    best_dist_fast = float(dists_fast[best_idx_fast])

    if best_dist_fast <= settings.threshold_super_strict:
        return VerifyResult(
            verified=True,
            distance=best_dist_fast,
            threshold=settings.threshold_super_strict,
            best_reference_index=best_idx_fast,
            reason="Aprovado (super_strict) via detector rápido",
            path="fast_only",
            model=settings.model_name,
            detector_fast=settings.detector_fast,
            detector_fallback=settings.detector_fallback,
            metric=settings.distance_metric,
        )

    if best_dist_fast > settings.threshold_strict:
        if best_dist_fast > settings.threshold_loose:
            return VerifyResult(
                verified=False,
                distance=best_dist_fast,
                threshold=settings.threshold_strict,
                best_reference_index=best_idx_fast,
                reason="Reprovado (muito distante) sem fallback",
                path="fast_reject",
                model=settings.model_name,
                detector_fast=settings.detector_fast,
                detector_fallback=settings.detector_fallback,
                metric=settings.distance_metric,
            )

    face_fb = _extract_best_face(img, settings.detector_fallback) if settings.use_face_crop else img
    emb_fb = _represent_skip(face_fb)

    dists_fb = np.array([_cosine_distance(emb_fb, r) for r in refs], dtype=np.float32)
    best_idx_fb = int(np.argmin(dists_fb))
    best_dist_fb = float(dists_fb[best_idx_fb])

    if best_dist_fb <= settings.threshold_strict:
        return VerifyResult(
            verified=True,
            distance=best_dist_fb,
            threshold=settings.threshold_strict,
            best_reference_index=best_idx_fb,
            reason="Aprovado (strict) com fallback",
            path="fast+fallback_confirmed",
            model=settings.model_name,
            detector_fast=settings.detector_fast,
            detector_fallback=settings.detector_fallback,
            metric=settings.distance_metric,
        )

    return VerifyResult(
        verified=False,
        distance=best_dist_fb,
        threshold=settings.threshold_strict,
        best_reference_index=best_idx_fb,
        reason="Reprovado após fallback",
        path="fast+fallback_reject",
        model=settings.model_name,
        detector_fast=settings.detector_fast,
        detector_fallback=settings.detector_fallback,
        metric=settings.distance_metric,
    )


def identify_1toN(selfie_bytes: bytes, top_k: Optional[int] = None) -> IdentifyResult:
    face_index.ensure_ready()

    k = int(top_k or settings.identify_top_k)
    if k < 2:
        k = 2

    img = _read_image_from_bytes(selfie_bytes)
    img = _resize_if_needed(img)

    face_fast = _extract_best_face(img, settings.detector_fast) if settings.use_face_crop else img
    emb_fast = _represent_skip(face_fast)

    sims, idxs = face_index.search(emb_fast, k)
    candidates_fast: List[IdentifyCandidate] = []
    for sim, idx in zip(sims, idxs):
        if idx < 0:
            continue
        meta = face_index.get_meta(int(idx))
        dist = 1.0 - float(sim)
        candidates_fast.append(
            IdentifyCandidate(
                user_id=str(meta["user_id"]),
                distance=dist,
                ref_index=int(meta.get("ref_index", 0)),
            )
        )

    if not candidates_fast:
        return IdentifyResult(
            identified=False,
            best_user_id=None,
            distance=None,
            threshold=settings.threshold_strict,
            top_k=[],
            reason="Sem candidatos no índice",
            path="ann_empty",
            model=settings.model_name,
            detector_fast=settings.detector_fast,
            detector_fallback=settings.detector_fallback,
            metric=settings.distance_metric,
        )

    candidates_fast.sort(key=lambda x: x.distance)
    best = candidates_fast[0]
    second = candidates_fast[1] if len(candidates_fast) > 1 else None

    margin_ok = True
    if second is not None:
        margin_ok = (second.distance - best.distance) >= settings.identify_margin

    if best.distance <= settings.threshold_super_strict and margin_ok:
        return IdentifyResult(
            identified=True,
            best_user_id=best.user_id,
            distance=best.distance,
            threshold=settings.threshold_super_strict,
            top_k=candidates_fast,
            reason="Identificado (super_strict) via ANN",
            path="ann_fast_only",
            model=settings.model_name,
            detector_fast=settings.detector_fast,
            detector_fallback=settings.detector_fallback,
            metric=settings.distance_metric,
        )

    needs_confirm = (best.distance <= settings.threshold_loose) and (best.distance > settings.threshold_strict or not margin_ok)

    if not needs_confirm:
        if best.distance <= settings.threshold_strict and margin_ok:
            return IdentifyResult(
                identified=True,
                best_user_id=best.user_id,
                distance=best.distance,
                threshold=settings.threshold_strict,
                top_k=candidates_fast,
                reason="Identificado (strict + margin) via ANN",
                path="ann_strict_margin",
                model=settings.model_name,
                detector_fast=settings.detector_fast,
                detector_fallback=settings.detector_fallback,
                metric=settings.distance_metric,
            )

        return IdentifyResult(
            identified=False,
            best_user_id=None,
            distance=best.distance,
            threshold=settings.threshold_strict,
            top_k=candidates_fast,
            reason="Desconhecido (falhou threshold/margin) sem fallback",
            path="ann_reject",
            model=settings.model_name,
            detector_fast=settings.detector_fast,
            detector_fallback=settings.detector_fallback,
            metric=settings.distance_metric,
        )

    face_fb = _extract_best_face(img, settings.detector_fallback) if settings.use_face_crop else img
    emb_fb = _represent_skip(face_fb)

    best_fb: Optional[IdentifyCandidate] = None
    second_fb: Optional[IdentifyCandidate] = None

    rescored: List[IdentifyCandidate] = []
    for c in candidates_fast:
        try:
            refs, _ = _load_user_refs(c.user_id)
        except Exception:
            continue
        d = float(np.min([_cosine_distance(emb_fb, r) for r in refs]))
        rescored.append(IdentifyCandidate(user_id=c.user_id, distance=d, ref_index=c.ref_index))

    if rescored:
        rescored.sort(key=lambda x: x.distance)
        best_fb = rescored[0]
        second_fb = rescored[1] if len(rescored) > 1 else None

    if best_fb is None:
        return IdentifyResult(
            identified=False,
            best_user_id=None,
            distance=best.distance,
            threshold=settings.threshold_strict,
            top_k=candidates_fast,
            reason="Fallback sem candidatos válidos",
            path="ann+fallback_empty",
            model=settings.model_name,
            detector_fast=settings.detector_fast,
            detector_fallback=settings.detector_fallback,
            metric=settings.distance_metric,
        )

    margin_ok_fb = True
    if second_fb is not None:
        margin_ok_fb = (second_fb.distance - best_fb.distance) >= settings.identify_margin

    if best_fb.distance <= settings.threshold_strict and margin_ok_fb:
        return IdentifyResult(
            identified=True,
            best_user_id=best_fb.user_id,
            distance=best_fb.distance,
            threshold=settings.threshold_strict,
            top_k=rescored[:k],
            reason="Identificado (strict + margin) com confirmação fallback",
            path="ann+fallback_confirmed",
            model=settings.model_name,
            detector_fast=settings.detector_fast,
            detector_fallback=settings.detector_fallback,
            metric=settings.distance_metric,
        )

    return IdentifyResult(
        identified=False,
        best_user_id=None,
        distance=best_fb.distance,
        threshold=settings.threshold_strict,
        top_k=rescored[:k],
        reason="Desconhecido após fallback (threshold/margin)",
        path="ann+fallback_reject",
        model=settings.model_name,
        detector_fast=settings.detector_fast,
        detector_fallback=settings.detector_fallback,
        metric=settings.distance_metric,
    )
