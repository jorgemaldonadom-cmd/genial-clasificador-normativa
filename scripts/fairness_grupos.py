"""Auditoría de equidad por grupo (Semana 4: impacto social y responsabilidad).

Calcula, sobre las predicciones fuera de muestra de la validación cruzada por grupo (results/clasico_cv_predicciones.csv),
métricas de desempeño por subgrupo de normas —idioma del título, tamaño del país en el corpus, región, ámbito y
codificador (anonimizado)— con intervalos de Wilson, la razón frente al grupo de referencia y la proporción de normas
que la regla de gobernanza envía a revisión humana. El «sujeto» de la equidad aquí no es una persona sino una
jurisdicción: el riesgo es que países pequeños o no hispanohablantes reciban una codificación automática peor y, con
ella, un Índice de Madurez Normativa distorsionado.

Uso:  python scripts/fairness_grupos.py
Salidas: results/fairness/fairness_grupos.csv · results/fairness/fairness_resumen.json · results/figuras/12_fairness_grupos.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from genial_clf.evaluate import wilson_ci  # noqa: E402

cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
C5 = cfg["classes"]; R = cfg["routing"]
OUT = ROOT / "results/fairness"; OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "results/figuras"; FIG.mkdir(parents=True, exist_ok=True)

cv = pd.read_csv(ROOT / "results/clasico_cv_predicciones.csv")
corpus = pd.read_csv(ROOT / "data/processed/corpus_depurado.csv")
d = cv.merge(corpus[["id_norma", "Investigador", "n_palabras"]], on="id_norma")

CARIBE = {"Jamaica", "Barbados", "Antigua y Barbuda", "Bahamas", "Belice", "Guyana", "San Cristóbal y Nieves", "Trinidad y Tobago", "Haití", "Santa Lucía", "Surinam", "San Vicente y las Granadinas", "Puerto Rico", "República Dominicana", "Cuba"}
n_pais = d["País"].value_counts()
d["tamaño_pais"] = d["País"].map(lambda p: "≥ 50 normas" if n_pais[p] >= 50 else ("20-49 normas" if n_pais[p] >= 20 else "< 20 normas"))
d["región"] = d["País"].map(lambda p: "Caribe" if p in CARIBE else ("Brasil" if p == "Brasil" else "Hispanoamérica continental"))
codes = {k: f"Codificador {chr(65 + i)}" for i, k in enumerate(d["Investigador"].value_counts().index)}   # anonimización
d["codificador"] = d["Investigador"].map(codes)
d["acierto"] = d["pred"] == d["y_true"]; d["top2"] = d["acierto"] | (d["pred_2"] == d["y_true"])
d["enrutado"] = (d["conf"] < R["confidence_threshold"]) | (d["País"].map(n_pais) < R["mandatory_review_min_country_n"]) | d["idioma"].isin(R["mandatory_review_languages"])


def fila(nombre, grupo, sub, ref_acc):
    k, n = int(sub["acierto"].sum()), len(sub); lo, hi = wilson_ci(k, n)
    return {"dimensión": nombre, "grupo": grupo, "n": n, "exactitud": k / n, "IC95 inf": lo, "IC95 sup": hi,
            "F1 macro": f1_score(sub["y_true"], sub["pred"], average="macro", labels=C5, zero_division=0),
            "top-2": sub["top2"].mean(), "razón vs referencia": (k / n) / ref_acc if ref_acc else np.nan,
            "% a revisión humana (regla)": sub["enrutado"].mean(), "exactitud tramo automático": sub.loc[~sub["enrutado"], "acierto"].mean() if (~sub["enrutado"]).any() else np.nan}


DIMS = [("Idioma del título", "idioma", "es"), ("Tamaño del país en el corpus", "tamaño_pais", "≥ 50 normas"), ("Región", "región", "Hispanoamérica continental"), ("Ámbito", "Ámbito", "Institucional"), ("Codificador (anonimizado)", "codificador", None)]
filas = []
for nombre, col, ref in DIMS:
    grupos = d.groupby(col)
    if ref is None:
        ref = d[col].value_counts().index[0]
    ref_acc = grupos.get_group(ref)["acierto"].mean()
    for g, sub in grupos:
        if len(sub) >= 10:
            filas.append(fila(nombre, g, sub, ref_acc))
tabla = pd.DataFrame(filas)
tabla.to_csv(OUT / "fairness_grupos.csv", index=False)

# resumen: brechas máximas por dimensión (regla de los cuatro quintos como referencia orientativa)
resumen = {}
for nombre, _, _ in DIMS:
    t = tabla[tabla["dimensión"] == nombre]
    resumen[nombre] = {"grupos": int(len(t)), "exactitud_min": float(t["exactitud"].min()), "exactitud_max": float(t["exactitud"].max()),
                       "razón_min": float(t["razón vs referencia"].min()), "cumple_4/5": bool(t["razón vs referencia"].min() >= 0.8),
                       "grupo_peor": str(t.loc[t["exactitud"].idxmin(), "grupo"]), "grupo_mejor": str(t.loc[t["exactitud"].idxmax(), "grupo"])}
resumen["enrutado_global"] = float(d["enrutado"].mean()); resumen["exactitud_global"] = float(d["acierto"].mean())
json.dump(resumen, open(OUT / "fairness_resumen.json", "w"), ensure_ascii=False, indent=1)

# figura
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .35, "grid.linestyle": ":", "savefig.dpi": 300})
sel = tabla[tabla["dimensión"].isin(["Idioma del título", "Tamaño del país en el corpus", "Región", "Ámbito"])].copy()
sel["etiqueta"] = sel["dimensión"].str.replace(" en el corpus", "") + " · " + sel["grupo"].astype(str) + " (n=" + sel["n"].astype(str) + ")"
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
y = np.arange(len(sel))[::-1]
ax = axes[0]
ax.barh(y, sel["exactitud"], color="#7A1F3D", height=0.6, xerr=[sel["exactitud"] - sel["IC95 inf"], sel["IC95 sup"] - sel["exactitud"]], error_kw={"ecolor": "#6e6862", "capsize": 3, "lw": 1})
ax.axvline(d["acierto"].mean(), color="#1b1a19", ls="--", lw=1); ax.text(d["acierto"].mean() + 0.01, len(sel) - 0.4, f"global {d['acierto'].mean():.2f}", fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels(sel["etiqueta"], fontsize=8.5); ax.set_xlim(0, 1); ax.set_xlabel("exactitud (CV por grupo, IC95 Wilson)")
ax.set_title("Desempeño por subgrupo de normas", loc="left", fontweight="bold")
ax = axes[1]
ax.barh(y, sel["% a revisión humana (regla)"] * 100, color="#eb6834", height=0.6)
for yi, v in zip(y, sel["% a revisión humana (regla)"] * 100):
    ax.text(v + 1, yi, f"{v:.0f} %", va="center", fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlim(0, 115); ax.set_xlabel("% enviado a revisión humana (regla de gobernanza)")
ax.set_title("Salvaguarda: a quién protege la regla", loc="left", fontweight="bold")
fig.suptitle("Auditoría de equidad por jurisdicción · clasificador RL + TF-IDF · n = 813", x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "12_fairness_grupos.png", facecolor="white")
print(tabla.round(3).to_string()); print(json.dumps(resumen, ensure_ascii=False, indent=1))
