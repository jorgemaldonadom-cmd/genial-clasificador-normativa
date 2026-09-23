"""Lectura del instrumento de codificación ciega de GENIAL (Framework Dimensional, seis dimensiones D1-D6).

El instrumento NO recodifica la variable `Dominio` (la muestra como contexto, columna G): codifica actores (D1), tipo de IA
(D2), tipo de disposición (D3), finalidad educativa (D4), enfoque (D5) y madurez (D6), con etiquetas secundarias y un nivel
de confianza. Por eso:
  * define el conjunto de prueba a ciegas (las normas que el investigador leyó), pero
  * no permite calcular κ humano-humano sobre `Dominio`; para eso hace falta `coder2_labels.csv` (véase notebooks/README_experimento.md).

Las normas del instrumento se emparejan con la versión congelada del corpus por título + país (desempate por URL y por
observaciones; respaldo difuso dentro del país). Las que no aparecen en la versión congelada (incorporadas a GENIAL
después del corte) se devuelven aparte como «normas nuevas».
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from pathlib import Path

import pandas as pd

DIMS = ["D1", "D2", "D3", "D4"]
COLS_INSTR = ["ID", "D1_principal", "D1_secundarias", "D2_principal", "D2_secundarias", "D3_principal", "D3_secundarias",
              "D4_principal", "D4_secundarias", "D5_enfoque", "D5_secundaria_opcional", "D6_madurez", "Confianza", "Notas"]


def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s.lower().strip())


def _url(s) -> str:
    return str(s).strip().lower().rstrip("/")


def load_instrument(path: str | Path, df: pd.DataFrame, sheet: str = "Codificación") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (emparejadas, nuevas). `emparejadas` trae `id_norma` del corpus congelado + columnas del instrumento;
    `nuevas` son las filas del instrumento sin correspondencia en la versión congelada (con su propio texto y Dominio)."""
    c = pd.read_excel(path, sheet_name=sheet)
    kt, kp, ku = df["Título de la Norma"].map(_norm), df["País"].map(_norm), df["URL Oficial"].map(_url)
    ko = df["Observaciones"].map(_norm)
    ids, unmatched = [], []
    for _, r in c.iterrows():
        t, p, u, o = _norm(r["Título de la Norma"]), _norm(r["País"]), _url(r["URL Oficial"]), _norm(r["Observaciones"])
        cand = df[(kt == t) & (kp == p)]
        if len(cand) > 1 and (cand.index.map(lambda i: ku[i] == u)).any():
            cand = cand[cand.index.map(lambda i: ku[i] == u)]
        if len(cand) > 1 and (cand.index.map(lambda i: ko[i] == o)).any():
            cand = cand[cand.index.map(lambda i: ko[i] == o)]
        if len(cand) == 0:
            cand = df[(kp == p) & (ku == u) & (ku != "nan")]
        if len(cand) == 0:
            sub = df[kp == p]
            best = difflib.get_close_matches(t, kt[sub.index].tolist(), n=1, cutoff=0.85)
            if best:
                cand = sub[kt[sub.index] == best[0]]
        if len(cand) == 0:
            unmatched.append(r)
        else:
            ids.append(int(cand.iloc[0]["id_norma"]))
    matched_rows = c[~c["ID"].isin([r["ID"] for r in unmatched])].copy()
    matched_rows["id_norma"] = ids
    keep = [k for k in COLS_INSTR if k in matched_rows.columns]
    emparejadas = matched_rows[["id_norma"] + keep].rename(columns={"ID": "id_instrumento"})
    emparejadas = emparejadas.drop_duplicates("id_norma")
    for d in DIMS:
        emparejadas[f"{d}_n_secundarias"] = emparejadas[f"{d}_secundarias"].fillna("").map(lambda s: 0 if not str(s).strip() else len([x for x in str(s).split(",") if x.strip()]))
    emparejadas["multidimensional"] = (emparejadas["D3_n_secundarias"] + emparejadas["D4_n_secundarias"]) > 0
    nuevas = pd.DataFrame(unmatched).reset_index(drop=True) if unmatched else pd.DataFrame(columns=c.columns)
    return emparejadas, nuevas


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """V de Cramér con corrección de sesgo (Bergsma, 2013)."""
    import numpy as np
    from scipy.stats import chi2_contingency

    ct = pd.crosstab(a, b)
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.to_numpy().sum(); r, k = ct.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(1e-12, min(kc - 1, rc - 1))))
