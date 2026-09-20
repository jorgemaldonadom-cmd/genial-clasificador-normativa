"""Evaluación: F1 macro y por clase, matriz de confusión, top-2 accuracy, McNemar, IC de Wilson y bootstrap, kappa.

Toda métrica se reporta con su intervalo de confianza; toda comparación entre ramas, con su valor p
(diseño pareado sobre idéntico conjunto de prueba -> prueba de McNemar, no de proporciones independientes).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, norm
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support


def wilson_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Intervalo de Wilson (1927) para una proporción."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(1 - (1 - level) / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(y_true, y_pred, metric, iters: int = 2000, level: float = 0.95, seed: int = 2026) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    vals = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, n)
        vals[i] = metric(y_true[idx], y_pred[idx])
    a = (1 - level) / 2
    return (float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a)))


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Prueba de McNemar sobre predicciones pareadas (exacta si b+c < 25; con corrección de continuidad si no)."""
    y_true, a, b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    ok_a, ok_b = a == y_true, b == y_true
    n01 = int((ok_a & ~ok_b).sum())  # A acierta, B falla
    n10 = int((~ok_a & ok_b).sum())  # A falla, B acierta
    if n01 + n10 == 0:
        return {"n01": n01, "n10": n10, "statistic": 0.0, "p_value": 1.0, "method": "sin discordancias"}
    if n01 + n10 < 25:
        p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
        return {"n01": n01, "n10": n10, "statistic": float("nan"), "p_value": float(p), "method": "binomial exacta"}
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p = float(chi2.sf(stat, df=1))
    return {"n01": n01, "n10": n10, "statistic": float(stat), "p_value": p, "method": "chi2 con corrección de continuidad"}


def report(y_true, y_pred, classes: list[str], y_pred2=None, iters: int = 2000, level: float = 0.95) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    k = int((y_true == y_pred).sum())
    f1m = f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    out = {
        "n": n,
        "accuracy": acc,
        "accuracy_ci_wilson": wilson_ci(k, n, level),
        "f1_macro": f1m,
        "f1_macro_ci_bootstrap": bootstrap_ci(y_true, y_pred, lambda a, b: f1_score(a, b, average="macro", labels=classes, zero_division=0), iters, level),
        "kappa_vs_reference": cohen_kappa_score(y_true, y_pred, labels=classes),
        "per_class": pd.DataFrame({"clase": classes, "precision": p, "recall": r, "f1": f, "soporte": s}),
        "confusion": pd.DataFrame(confusion_matrix(y_true, y_pred, labels=classes), index=classes, columns=classes),
    }
    if y_pred2 is not None:
        y_pred2 = np.asarray(y_pred2)
        out["top2_accuracy"] = float(((y_pred == y_true) | (y_pred2 == y_true)).mean())
    return out


def by_group(df: pd.DataFrame, y_true: str, y_pred: str, group: str, classes: list[str]) -> pd.DataFrame:
    """Métricas desagregadas por idioma o por país (Decisión 4)."""
    rows = []
    for g, sub in df.groupby(group):
        rows.append({
            group: g, "n": len(sub),
            "accuracy": accuracy_score(sub[y_true], sub[y_pred]),
            "f1_macro": f1_score(sub[y_true], sub[y_pred], average="macro", labels=classes, zero_division=0),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def summarize(rep: dict, name: str) -> str:
    lo, hi = rep["accuracy_ci_wilson"]
    flo, fhi = rep["f1_macro_ci_bootstrap"]
    s = [f"## {name} (n = {rep['n']})",
         f"- Exactitud: {rep['accuracy']:.3f}  IC95 % Wilson [{lo:.3f}, {hi:.3f}]",
         f"- F1 macro: {rep['f1_macro']:.3f}  IC95 % bootstrap [{flo:.3f}, {fhi:.3f}]",
         f"- κ frente a la etiqueta de referencia: {rep['kappa_vs_reference']:.3f}"]
    if "top2_accuracy" in rep:
        s.append(f"- Top-2 accuracy: {rep['top2_accuracy']:.3f}")
    s.append("")
    s.append(rep["per_class"].to_markdown(index=False, floatfmt=".3f"))
    return "\n".join(s) + "\n"
