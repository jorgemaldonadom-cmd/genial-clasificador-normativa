"""Operación: umbral de confianza, enrutamiento a revisión humana y agregación al Índice de Madurez Normativa.

Regla de gobernanza: una norma se enruta a revisión humana si la confianza es baja O si pertenece a un país o
idioma subrepresentado en el corpus, aunque el modelo esté seguro. Cada versión del IMN distingue qué normas
codificó un humano y cuáles el modelo, y publica el margen de error por dimensión.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def route(preds: pd.DataFrame, meta: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """preds: id_norma, pred, pred_2, conf · meta: id_norma, País, idioma."""
    r = cfg["routing"]
    df = preds.merge(meta[["id_norma", "País", "idioma"]], on="id_norma", how="left")
    n_by_country = meta["País"].value_counts()
    df["pais_subrepresentado"] = df["País"].map(n_by_country).fillna(0) < r["mandatory_review_min_country_n"]
    df["idioma_minoritario"] = df["idioma"].isin(r["mandatory_review_languages"])
    df["baja_confianza"] = df["conf"] < r["confidence_threshold"]
    df["a_revision"] = df["baja_confianza"] | df["pais_subrepresentado"] | df["idioma_minoritario"]
    df["origen_etiqueta"] = np.where(df["a_revision"], "humano (revisión)", "modelo")
    return df


def routing_curve(preds: pd.DataFrame, y_true: pd.Series, thresholds=np.linspace(0.3, 0.95, 14)) -> pd.DataFrame:
    """Curva enrutamiento-exactitud: para cada umbral, % enrutado y exactitud del tramo automático."""
    rows = []
    ok = (preds["pred"].to_numpy() == y_true.to_numpy())
    for t in thresholds:
        auto = preds["conf"].to_numpy() >= t
        rows.append({"umbral": round(float(t), 2), "pct_enrutado": float(1 - auto.mean()),
                     "exactitud_tramo_automatico": float(ok[auto].mean()) if auto.any() else float("nan")})
    return pd.DataFrame(rows)


def residual_workload(routed: pd.DataFrame, minutes_per_norm: float = 10.0) -> dict:
    """Carga humana residual en horas-persona bajo el supuesto declarado de minutos por norma."""
    n = len(routed)
    n_rev = int(routed["a_revision"].sum())
    return {"normas": n, "a_revision": n_rev, "pct_enrutado": n_rev / n if n else float("nan"),
            "horas_manual_total": n * minutes_per_norm / 60, "horas_residuales": n_rev * minutes_per_norm / 60,
            "horas_ahorradas": (n - n_rev) * minutes_per_norm / 60}


def imn_table(routed: pd.DataFrame, f1_by_class: dict[str, float], classes: list[str]) -> pd.DataFrame:
    """Conteo por país y dimensión con trazabilidad del origen y margen de error (1 − F1 de la clase)."""
    rows = []
    for pais, g in routed.groupby("País"):
        row = {"País": pais, "normas": len(g), "codificadas_por_modelo": int((g["origen_etiqueta"] == "modelo").sum())}
        for c in classes:
            row[c] = int((g["pred"] == c).sum())
            row[f"{c} (margen)"] = round((1 - f1_by_class.get(c, 0.0)) * row[c], 1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("normas", ascending=False)
