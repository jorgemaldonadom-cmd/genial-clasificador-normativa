"""Definición, persistencia y uso del modelo (capa de alto nivel sobre `genial_clf`).

El modelo es un `Pipeline` de scikit-learn: TF-IDF (1-2 gramas) + Regresión Logística multiclase con clases ponderadas.
`Predictor` envuelve un modelo entrenado con todo lo que la aplicación necesita: predicción con probabilidades,
segunda candidata (top-2), términos que sustentan la decisión y la regla de gobernanza (enrutamiento a revisión humana).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils import ROOT, load_config, setup_logging

log = logging.getLogger(__name__)
MODELS_DIR = ROOT / "models"


def build_model(C: float = 1.0, max_features: int = 5000, ngram_max: int = 2, min_df: int = 1, class_weight: str | None = "balanced",
                sublinear_tf: bool = True, seed: int = 2026) -> Pipeline:
    """Construye el pipeline TF-IDF + Regresión Logística con los hiperparámetros indicados.

    Args:
        C: inverso de la fuerza de regularización L2 (menor = más regularización).
        max_features: tamaño máximo del vocabulario TF-IDF.
        ngram_max: n-grama máximo (1 = unigramas, 2 = unigramas y bigramas).
        min_df: frecuencia documental mínima de un término.
        class_weight: 'balanced' (pondera clases minoritarias) o None.
        sublinear_tf: usa 1 + log(tf) en lugar de tf.
        seed: semilla.
    """
    if C <= 0 or max_features < 50 or ngram_max not in (1, 2, 3) or min_df < 1:
        raise ValueError("Hiperparámetros fuera de rango: C > 0, max_features ≥ 50, ngram_max ∈ {1,2,3}, min_df ≥ 1")
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, ngram_max), max_features=max_features, min_df=min_df, sublinear_tf=sublinear_tf, lowercase=True)),
        ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=3000, random_state=seed)),
    ])


def model_params(pipe: Pipeline) -> dict:
    """Hiperparámetros relevantes de un pipeline entrenado (para la model card y la app)."""
    tf, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    return {"C": clf.C, "class_weight": clf.class_weight, "max_features": tf.max_features, "ngram_range": list(tf.ngram_range), "min_df": tf.min_df,
            "sublinear_tf": tf.sublinear_tf, "n_vocab": len(tf.vocabulary_) if hasattr(tf, "vocabulary_") else None}


def save_model(pipe: Pipeline, path: str | Path, metadata: dict | None = None) -> Path:
    """Guarda el pipeline con joblib y un JSON de metadatos al lado (fecha, hiperparámetros, métricas)."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    meta = {"guardado": datetime.now(timezone.utc).isoformat(timespec="seconds"), "clases": list(map(str, pipe.classes_)), "parametros": model_params(pipe), **(metadata or {})}
    Path(str(path).replace(".pkl", ".json")).write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log.info("modelo guardado en %s", path)
    return path


def load_model(path: str | Path = MODELS_DIR / "best_model.pkl") -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el modelo {path}. Ejecute `python src/train.py` o el cuaderno 03_modelado.ipynb.")
    return joblib.load(path)


def load_model_metadata(path: str | Path = MODELS_DIR / "best_model.pkl") -> dict:
    p = Path(str(path).replace(".pkl", ".json"))
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@dataclass
class Predictor:
    """Modelo entrenado + regla de gobernanza, listo para la aplicación."""

    pipe: Pipeline
    cfg: dict = field(default_factory=load_config)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_files(cls, path: str | Path = MODELS_DIR / "best_model.pkl") -> "Predictor":
        return cls(pipe=load_model(path), metadata=load_model_metadata(path))

    @property
    def classes(self) -> list[str]:
        return list(map(str, self.pipe.classes_))

    @property
    def threshold(self) -> float:
        """Umbral de confianza: el calibrado para este modelo (metadatos) o, si no existe, el de config.yaml."""
        return float(self.metadata.get("umbral_confianza", self.cfg["routing"]["confidence_threshold"]))

    def predict_one(self, text: str, country: str | None = None, language: str = "es", n_country: int | None = None) -> dict:
        """Clasifica una norma. Devuelve dimensión, probabilidades, top-2, decisión de enrutamiento y términos influyentes."""
        text = (text or "").strip()
        if len(text.split()) < 3:
            raise ValueError("El texto es demasiado corto: escriba al menos el título de la norma y una observación (≥ 3 palabras).")
        proba = self.pipe.predict_proba([text])[0]
        order = np.argsort(-proba)
        r = self.cfg["routing"]
        baja = float(proba[order[0]]) < self.threshold
        pais_sub = n_country is not None and n_country < r["mandatory_review_min_country_n"]
        idioma_min = language in r["mandatory_review_languages"]
        motivos = [m for m, ok in [("probabilidad por debajo del umbral de confianza", baja), ("país con pocas normas en el corpus", pais_sub), ("idioma distinto del español", idioma_min)] if ok]
        return {"texto": text, "dimension": self.classes[order[0]], "probabilidad": float(proba[order[0]]), "segunda_dimension": self.classes[order[1]],
                "probabilidad_segunda": float(proba[order[1]]), "probabilidades": {c: float(p) for c, p in zip(self.classes, proba)},
                "a_revision_humana": bool(baja or pais_sub or idioma_min), "motivos_revision": motivos, "terminos": self.top_terms(text, self.classes[order[0]])}

    def predict_many(self, texts: list[str], countries: list[str] | None = None, languages: list[str] | None = None) -> pd.DataFrame:
        rows = []
        for i, t in enumerate(texts):
            try:
                r = self.predict_one(t, country=countries[i] if countries else None, language=languages[i] if languages else "es")
                rows.append({"texto": t, "dimension": r["dimension"], "probabilidad": r["probabilidad"], "segunda_dimension": r["segunda_dimension"],
                             "probabilidad_segunda": r["probabilidad_segunda"], "a_revision_humana": r["a_revision_humana"], "error": ""})
            except ValueError as e:
                rows.append({"texto": t, "dimension": None, "probabilidad": np.nan, "segunda_dimension": None, "probabilidad_segunda": np.nan, "a_revision_humana": True, "error": str(e)})
        return pd.DataFrame(rows)

    def top_terms(self, text: str, dimension: str, k: int = 8) -> list[tuple[str, float]]:
        """Términos del texto que más empujan hacia la dimensión predicha (peso × TF-IDF), para explicar la decisión."""
        tf, clf = self.pipe.named_steps["tfidf"], self.pipe.named_steps["clf"]
        x = tf.transform([text]); idx = self.classes.index(dimension)
        contrib = x.toarray()[0] * clf.coef_[idx]
        vocab = tf.get_feature_names_out(); top = np.argsort(-contrib)[:k]
        return [(str(vocab[i]), float(contrib[i])) for i in top if contrib[i] > 0]


if __name__ == "__main__":  # demostración rápida
    setup_logging()
    p = Predictor.from_files()
    print(json.dumps(p.predict_one("Lineamientos para el uso ético de la inteligencia artificial generativa en la docencia universitaria. Establece principios de integridad académica y transparencia."), ensure_ascii=False, indent=1))
