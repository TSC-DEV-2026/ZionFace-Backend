from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    banco_fotos_dir: str = "banco_fotos"

    model_name: str = "ArcFace"
    distance_metric: str = "cosine"

    detector_fast: str = "opencv"
    detector_fallback: str = "retinaface"

    max_side: int = 640
    use_face_crop: bool = True
    min_face_area: int = 80 * 80

    threshold_super_strict: float = 0.33
    threshold_strict: float = 0.40
    threshold_loose: float = 0.52

    identify_top_k: int = 5
    identify_margin: float = 0.04
    hnsw_m: int = 32
    hnsw_ef_search: int = 64
    hnsw_ef_construction: int = 128

settings = Settings()
