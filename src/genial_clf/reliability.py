"""Fiabilidad: kappa humano-humano y modelo-humano, adjudicación y tipificación de errores.

Protocolo (Decisión 3): el segundo codificador codifica A CIEGAS el conjunto de prueba, sin ver la etiqueta
original ni las predicciones; la regla de gold standard se fija por escrito ANTES de medir; las discrepancias
las adjudica un tercer codificador. El desempeño de cada rama se reporta relativo al acuerdo humano-humano.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ERROR_TYPES = ("error genuino", "2.ª dimensión defendible", "etiqueta original discutible")


def kappa(a, b, classes: list[str]) -> float:
    return float(cohen_kappa_score(a, b, labels=classes))


def landis_koch(k: float) -> str:
    """Interpretación convencional de kappa (Landis y Koch, 1977)."""
    for lim, name in ((0.0, "pobre"), (0.2, "leve"), (0.4, "aceptable"), (0.6, "moderado"), (0.8, "sustancial"), (1.01, "casi perfecto")):
        if k < lim:
            return name
    return "casi perfecto"


def adjudicate(original: pd.Series, second: pd.Series, third: pd.Series | None = None) -> pd.DataFrame:
    """Etiqueta de referencia consolidada: coincidencia -> etiqueta; discrepancia -> tercer codificador."""
    df = pd.DataFrame({"original": original, "segundo": second})
    df["discrepancia"] = df["original"] != df["segundo"]
    df["referencia"] = df["original"].where(~df["discrepancia"])
    if third is not None:
        df.loc[df["discrepancia"], "referencia"] = third[df["discrepancia"]]
    return df


def reliability_table(ref: pd.Series, original: pd.Series, second: pd.Series, preds: dict[str, pd.Series], classes: list[str]) -> pd.DataFrame:
    """κ humano-humano y κ de cada rama frente a la referencia consolidada, con interpretación."""
    rows = [{"comparación": "humano (original) – humano (2.º codificador)", "kappa": kappa(original, second, classes)}]
    for name, p in preds.items():
        rows.append({"comparación": f"{name} – referencia consolidada", "kappa": kappa(ref, p, classes)})
    out = pd.DataFrame(rows)
    out["interpretación"] = out["kappa"].map(landis_koch)
    return out


def error_typing_sheet(test: pd.DataFrame, ref: pd.Series, preds: pd.DataFrame) -> pd.DataFrame:
    """Hoja de trabajo para la tipificación cualitativa de errores (se completa a mano por el equipo)."""
    df = test[["id_norma", "País", "idioma", "texto"]].copy()
    df["referencia"] = ref.to_numpy()
    df["pred"] = preds["pred"].to_numpy()
    df["pred_2"] = preds["pred_2"].to_numpy()
    df = df[df["pred"] != df["referencia"]].copy()
    df["segunda_acierta"] = df["pred_2"] == df["referencia"]
    df["tipo_error"] = ""  # a completar: uno de ERROR_TYPES
    df["justificación"] = ""
    return df


def error_typing_summary(sheet: pd.DataFrame) -> pd.DataFrame:
    counts = sheet["tipo_error"].value_counts().reindex(ERROR_TYPES, fill_value=0)
    return pd.DataFrame({"tipo": counts.index, "n": counts.values, "proporción": counts.values / max(1, counts.sum())})
