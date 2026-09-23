"""Rama clásica: TF-IDF (1-2 gramas, 5 000 features) + Regresión Logística multiclase con class_weight balanceado.

Salidas por norma: dimensión predicha, segunda dimensión (top-2) y probabilidad máxima (para el umbral de confianza).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


def build_pipeline(cfg: dict) -> Pipeline:
    c = cfg["classic"]
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=tuple(c["ngram_range"]), max_features=c["max_features"],
                                  sublinear_tf=True, strip_accents=None, lowercase=True)),
        ("clf", LogisticRegression(C=c["C"], class_weight=c["class_weight"], max_iter=2000, random_state=cfg["seed"])),
    ])


def top2(proba: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-proba, axis=1)
    return classes[order[:, 0]], classes[order[:, 1]], proba[np.arange(len(proba)), order[:, 0]]


def cross_validate(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Validación cruzada estratificada sobre el conjunto de entrenamiento: predicciones fuera de muestra por norma.
    Si existe `grupo_url`, los duplicados por URL se mantienen en el mismo pliegue (StratifiedGroupKFold)."""
    pipe = build_pipeline(cfg)
    y = df[cfg["columns"]["target"]].to_numpy()
    k = cfg["classic"]["cv_folds"]
    if "grupo_url" in df.columns:
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg["seed"])
        proba = cross_val_predict(pipe, df["texto"], y, cv=cv, groups=df["grupo_url"].to_numpy(), method="predict_proba")
    else:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg["seed"])
        proba = cross_val_predict(pipe, df["texto"], y, cv=cv, method="predict_proba")
    classes = np.array(sorted(set(y)))
    p1, p2, conf = top2(proba, classes)
    return pd.DataFrame({"id_norma": df["id_norma"].to_numpy(), "y_true": y, "pred": p1, "pred_2": p2, "conf": conf})


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> tuple[Pipeline, pd.DataFrame]:
    """Entrena con el conjunto de entrenamiento (sin la prueba ni el pool few-shot) y predice el conjunto ciego."""
    pipe = build_pipeline(cfg).fit(train["texto"], train[cfg["columns"]["target"]])
    proba = pipe.predict_proba(test["texto"])
    p1, p2, conf = top2(proba, pipe.classes_)
    out = pd.DataFrame({"id_norma": test["id_norma"].to_numpy(), "pred": p1, "pred_2": p2, "conf": conf})
    return pipe, out


def top_terms(pipe: Pipeline, k: int = 15) -> pd.DataFrame:
    """Términos con mayor peso por dimensión (interpretabilidad)."""
    vocab = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coef = pipe.named_steps["clf"].coef_
    rows = []
    for i, c in enumerate(pipe.named_steps["clf"].classes_):
        idx = np.argsort(-coef[i])[:k]
        rows.append({"dimension": c, "terminos": ", ".join(vocab[idx])})
    return pd.DataFrame(rows)
