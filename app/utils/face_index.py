import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None
    _FAISS_IMPORT_ERROR = str(e)
import numpy as np

from app.core.settings import settings


@dataclass
class IndexStatus:
    ready: bool
    total_users: int
    total_embeddings: int
    index_type: str
    banco_fotos_dir: str


class FaceIndex:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._ready = False

        self._index: Optional[faiss.Index] = None
        self._meta: List[Dict[str, Any]] = []
        self._matrix: Optional[np.ndarray] = None

        self._total_users = 0
        self._total_embeddings = 0

    def status(self) -> IndexStatus:
        with self._lock:
            return IndexStatus(
                ready=self._ready,
                total_users=self._total_users,
                total_embeddings=self._total_embeddings,
                index_type="faiss_hnsw_ip",
                banco_fotos_dir=settings.banco_fotos_dir,
            )

    def ensure_ready(self) -> None:
        with self._lock:
            if self._ready:
                return
        self.rebuild()

    def rebuild(self) -> IndexStatus:
        if faiss is None:
            raise RuntimeError(f"faiss indisponível neste ambiente: {_FAISS_IMPORT_ERROR}")
        banco_dir = Path(settings.banco_fotos_dir)
        banco_dir.mkdir(parents=True, exist_ok=True)

        metas: List[Dict[str, Any]] = []
        vectors: List[np.ndarray] = []
        user_ids_set = set()

        for user_dir in banco_dir.iterdir():
            if not user_dir.is_dir():
                continue

            ref_path = user_dir / "ref.embedding.json"
            if not ref_path.exists():
                continue

            try:
                data = json.loads(ref_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            embeddings = data.get("embeddings") or []
            if not isinstance(embeddings, list) or len(embeddings) == 0:
                continue

            user_id = str(user_dir.name)
            user_ids_set.add(user_id)

            for i, emb in enumerate(embeddings):
                try:
                    v = np.asarray(emb, dtype=np.float32).reshape(-1)
                    if v.shape[0] != 512:
                        continue
                except Exception:
                    continue

                vectors.append(v)
                metas.append({"user_id": user_id, "ref_index": int(i), "path": str(ref_path)})

        if not vectors:
            with self._lock:
                self._index = None
                self._meta = []
                self._matrix = None
                self._ready = False
                self._total_users = 0
                self._total_embeddings = 0
                return self.status()

        matrix = np.vstack(vectors).astype(np.float32)
        faiss.normalize_L2(matrix)

        dim = matrix.shape[1]
        index = faiss.IndexHNSWFlat(dim, int(settings.hnsw_m), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = int(settings.hnsw_ef_search)
        index.hnsw.efConstruction = int(settings.hnsw_ef_construction)
        index.add(matrix)

        with self._lock:
            self._index = index
            self._meta = metas
            self._matrix = matrix
            self._ready = True
            self._total_users = len(user_ids_set)
            self._total_embeddings = len(metas)

        return self.status()

    def add_embeddings(self, user_id: str, embeddings: List[List[float]]) -> None:
        if not embeddings:
            return

        if faiss is None:
            return

        with self._lock:
            if not self._ready or self._index is None:
                return

            # garante ref_index monotônico por usuário (útil para debug / rastreio)
            current_max = -1
            for m in self._meta:
                if m.get("user_id") == str(user_id):
                    try:
                        current_max = max(current_max, int(m.get("ref_index", -1)))
                    except Exception:
                        pass

            new_vecs: List[np.ndarray] = []
            for i, emb in enumerate(embeddings):
                try:
                    v = np.asarray(emb, dtype=np.float32).reshape(-1)
                    if v.shape[0] != 512:
                        continue

                    new_vecs.append(v)
                    self._meta.append(
                        {
                            "user_id": str(user_id),
                            "ref_index": int(current_max + 1 + i),
                            "path": str(Path(settings.banco_fotos_dir) / str(user_id) / "ref.embedding.json"),
                        }
                    )
                except Exception:
                    continue

            if not new_vecs:
                return

            m = np.vstack(new_vecs).astype(np.float32)
            faiss.normalize_L2(m)
            self._index.add(m)

            self._total_embeddings = len(self._meta)
            self._total_users = len({x["user_id"] for x in self._meta})

    def search(self, embedding_512: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if faiss is None:
            raise RuntimeError(f"faiss indisponível neste ambiente: {_FAISS_IMPORT_ERROR}")
        with self._lock:
            if not self._ready or self._index is None:
                raise RuntimeError("Índice 1:N não está pronto")

            q = embedding_512.astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(q)

            sims, idxs = self._index.search(q, int(top_k))
            return sims.reshape(-1), idxs.reshape(-1)

    def get_meta(self, idx: int) -> Dict[str, Any]:
        with self._lock:
            return self._meta[idx]


face_index = FaceIndex()