from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Global in-memory cache so the SentenceTransformer loads ONCE per server process
_MODEL_CACHE: dict[str, object] = {}

# Romanized Malayalam (Manglish) -> English bridge so pretrained multilingual
# embeddings map code-mixed hospital feedback accurately alongside native scripts.
MANGLISH_MAP: dict[str, str] = {
    "valare": "very",
    "kooduthal": "too much long",
    "kooduthalaanu": "too much long",
    "thamasam": "delay waiting",
    "thamasichu": "delayed",
    "kaathirippu": "waiting",
    "aayirunnu": "was",
    "aanu": "is",
    "pakshe": "but",
    "pakshae": "but",
    "mosham": "very bad poor",
    "vrithiked": "dirty unclean",
    "vrithikedum": "dirty unclean",
    "vrithi": "clean",
    "vrithiyilla": "not clean dirty",
    "illa": "no not available",
    "illayirunnu": "was not available",
    "kittilla": "did not receive",
    "kittan": "to get",
    "kittiyilla": "not received",
    "paranjilla": "did not explain",
    "vyakthamayi": "clearly",
    "nallathaanu": "is good",
    "nallathu": "good",
    "nannayi": "well nicely",
    "marunnu": "medicine",
    "bhakshanam": "food",
    "thanutha": "cold",
    "sahayam": "help",
    "sahayavum": "help",
    "thannilla": "did not give",
    "vannilla": "did not come",
    "manikkoor": "hours",
    "oru": "one",
    "randu": "two",
    "perumaattam": "behaviour",
    "paricharanam": "care",
}

_MANGLISH_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(MANGLISH_MAP, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)


def normalize_multilingual_text(text: str) -> str:
    """Unicode-safe normalizer preserving Malayalam, Hindi, Arabic, and English."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).strip()
    if not s:
        return ""
    s = _MANGLISH_RE.sub(lambda m: MANGLISH_MAP[m.group(0).lower()], s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class MultilingualEmbedder(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer wrapping SentenceTransformers.
    Maps English, Malayalam, Manglish, Hindi, and Arabic into ONE shared 384-d space.
    Saves only `model_name` in .pkl files (keeping .pkl < 100 KB).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

    def _get_model(self):
        model = _MODEL_CACHE.get(self.model_name)
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name, device="cpu")
            _MODEL_CACHE[self.model_name] = model
        return model

    def fit(self, X: Iterable[str], y=None) -> "MultilingualEmbedder":
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        texts = [normalize_multilingual_text(x) for x in X]
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        model = self._get_model()
        embs = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(embs, dtype=np.float32)