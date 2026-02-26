import base64
import io
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2
from PIL import Image

try:
    import onnxruntime as ort
except Exception as e:
    ort = None
    _ORT_IMPORT_ERROR = str(e)


@dataclass
class FaceConfig:
    model_path: str = os.path.join("models", "arcface.onnx")
    input_size: Tuple[int, int] = (112, 112)  # ArcFace padrão
    verify_threshold: float = 0.35


class FaceService:
    def __init__(self, cfg: FaceConfig):
        self.cfg = cfg
        self.session: Optional[object] = None
        self.input_name: Optional[str] = None
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _ensure_loaded(self) -> None:
        if self.session is not None:
            return

        if ort is None:
            raise RuntimeError(f"onnxruntime indisponível neste ambiente: {_ORT_IMPORT_ERROR}")

        if not os.path.exists(self.cfg.model_path):
            raise RuntimeError(f"Modelo ONNX não encontrado: {self.cfg.model_path}")

        self.session = ort.InferenceSession(self.cfg.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def warmup(self) -> None:
        self._ensure_loaded()
        dummy = np.zeros((1, 3, self.cfg.input_size[1], self.cfg.input_size[0]), dtype=np.float32)
        _ = self.session.run(None, {self.input_name: dummy})

    @staticmethod
    def _b64_to_bgr(img_b64: str) -> np.ndarray:
        raw = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        rgb = np.array(img)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _detect_face(self, bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) == 0:
            raise ValueError("Nenhum rosto detectado")

        x, y, w, h = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]

        pad = int(0.15 * max(w, h))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(bgr.shape[1], x + w + pad)
        y1 = min(bgr.shape[0], y + h + pad)

        return bgr[y0:y1, x0:x1]

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        face = cv2.resize(face_bgr, self.cfg.input_size, interpolation=cv2.INTER_AREA)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        x = face.astype(np.float32)
        x = (x - 127.5) / 128.0
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)
        return x

    def get_embedding_from_b64(self, img_b64: str) -> np.ndarray:
        self._ensure_loaded()

        bgr = self._b64_to_bgr(img_b64)
        face = self._detect_face(bgr)
        inp = self._preprocess(face)

        out = self.session.run(None, {self.input_name: inp})[0]
        emb = out[0].astype(np.float32)

        norm = np.linalg.norm(emb) + 1e-12
        return emb / norm

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def verify_1to1(self, img1_b64: str, img2_b64: str) -> dict:
        e1 = self.get_embedding_from_b64(img1_b64)
        e2 = self.get_embedding_from_b64(img2_b64)

        score = self.cosine_similarity(e1, e2)
        match = score >= self.cfg.verify_threshold

        return {"match": match, "score": score, "threshold": self.cfg.verify_threshold}


_face_service: Optional[FaceService] = None


def get_face_service() -> FaceService:
    global _face_service
    if _face_service is None:
        _face_service = FaceService(FaceConfig())
    return _face_service


def warmup() -> None:
    get_face_service().warmup()