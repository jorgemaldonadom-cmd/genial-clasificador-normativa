"""Líneas base sin IA: clase mayoritaria y línea base de metadata sobre `Tipo de norma`.

La línea base de metadata cuantifica cuánta señal aporta `Tipo de norma` por sí sola y fundamenta su
exclusión como variable predictora (fuga de información: asignar el tipo a una norma nueva exige el mismo
trabajo experto que el proyecto busca evitar).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class MajorityClass:
    def fit(self, y: pd.Series) -> "MajorityClass":
        self.label_ = y.value_counts().idxmax()
        return self

    def predict(self, n: int) -> np.ndarray:
        return np.array([self.label_] * n)


class MetadataBaseline:
    """Predice la dimensión más frecuente para cada valor de una columna de metadata (tabla de contingencia)."""

    def __init__(self, column: str = "Tipo de norma"):
        self.column = column

    def fit(self, df: pd.DataFrame, target: str) -> "MetadataBaseline":
        tab = pd.crosstab(df[self.column], df[target])
        self.map_ = tab.idxmax(axis=1).to_dict()
        self.fallback_ = df[target].value_counts().idxmax()
        # probabilidad = frecuencia relativa de la clase modal dentro del tipo
        self.conf_ = (tab.max(axis=1) / tab.sum(axis=1)).to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.column].map(self.map_).fillna(self.fallback_).to_numpy()

    def predict_conf(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.column].map(self.conf_).fillna(0.0).to_numpy()


def arithmetic_floor(counts: dict[str, int]) -> dict[str, float]:
    """Exactitud y F1 macro de un predictor de clase mayoritaria, calculados sin entrenar nada."""
    n = sum(counts.values())
    maj = max(counts.values())
    acc = maj / n
    f1_maj = 2 * acc * 1.0 / (acc + 1.0)  # precisión = acc, recall = 1 en la clase mayoritaria
    return {"accuracy": acc, "f1_macro": f1_maj / len(counts)}
