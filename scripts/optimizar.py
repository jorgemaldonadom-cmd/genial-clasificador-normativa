#!/usr/bin/env python3
"""Optimización de hiperparámetros con análisis de sensibilidad (Workshop S5).

Uso:  python scripts/optimizar.py [--trials 60] [--seed 2026]
Salidas: results/metrics/optimizacion/{trials.csv, importancia.csv, dependencia_parcial.csv, interacciones.csv, sensibilidad.csv, resumen.json}
         results/figures/13-17_*.png
El cuaderno notebooks/04_optimizacion.ipynb reproduce estos pasos de forma comentada y lee/escribe los mismos archivos.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils import load_config, setup_logging  # noqa: E402
from model import build_model  # noqa: E402
from genial_clf import optimize as O  # noqa: E402

ACC, TRAIN_C, VAL_C, MUTED, INK = "#7A1F3D", "#2a78d6", "#eb6834", "#6e6862", "#1b1a19"


def plots(trials: pd.DataFrame, imp: pd.DataFrame, pdp: pd.DataFrame, grid: pd.DataFrame, comp: pd.DataFrame, fig_dir: Path) -> None:
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .35, "grid.linestyle": ":", "savefig.dpi": 300, "savefig.bbox": "tight"})
    # 13 historia
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.scatter(trials["trial"], trials["val_f1"], s=18, color=VAL_C, label="F1 macro de validación por ensayo")
    ax.plot(trials["trial"], trials["val_f1"].cummax(), color=ACC, lw=2, label="mejor hasta el ensayo")
    ax.axhline(trials.loc[0, "val_f1"], color=INK, ls="--", lw=1); ax.text(trials["trial"].max(), trials.loc[0, "val_f1"] + 0.003, "configuración base", ha="right", fontsize=8.5)
    ax.set_xlabel("ensayo (Optuna, TPE)"); ax.set_ylabel("F1 macro (CV por grupo)"); ax.set_title("Historia de la optimización", loc="left", fontweight="bold"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(fig_dir / "13_optuna_historia.png", facecolor="white"); plt.close(fig)
    # 14 importancia
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(imp["hiperparámetro"][::-1], imp["importancia"][::-1], color=ACC, height=0.6)
    for i, v in enumerate(imp["importancia"][::-1]):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8.5)
    ax.set_xlim(0, max(0.6, imp["importancia"].max() + 0.1)); ax.set_xlabel("importancia (fANOVA sobre los ensayos)"); ax.set_title("Ranking de importancia de hiperparámetros", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(fig_dir / "14_importancia_hiperparametros.png", facecolor="white"); plt.close(fig)
    # 15 dependencia parcial
    keys = ["C", "max_features", "ngram_max", "min_df", "class_weight", "sublinear_tf"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, k in zip(axes.ravel(), keys):
        g = pdp[pdp["hiperparámetro"] == k]; x = np.arange(len(g)); lab = [str(v) for v in g["valor"]]
        ax.plot(x, g["train_f1"], "o-", color=TRAIN_C, label="entrenamiento"); ax.plot(x, g["val_f1"], "o-", color=VAL_C, label="validación")
        ax.fill_between(x, g["val_f1"] - g["val_std"], g["val_f1"] + g["val_std"], color=VAL_C, alpha=0.15)
        ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8.5); ax.set_ylim(0.4, 1.02); ax.set_title(k, fontweight="bold"); ax.set_ylabel("F1 macro")
    axes[0, 0].legend(loc="center right", fontsize=8.5)
    fig.suptitle("Dependencia parcial: F1 al variar un hiperparámetro con los demás en el óptimo (CV por grupo)", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(fig_dir / "15_dependencia_parcial.png", facecolor="white"); plt.close(fig)
    # 16 interacción
    piv = grid.pivot(index="C", columns="max_features", values="val_f1"); pivg = grid.pivot(index="C", columns="max_features", values="gap")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, p, title, cmap in [(axes[0], piv, "F1 macro de validación", "Reds"), (axes[1], pivg, "gap entrenamiento − validación", "Blues")]:
        im = ax.imshow(p.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(p.shape[1])); ax.set_xticklabels(p.columns); ax.set_yticks(range(p.shape[0])); ax.set_yticklabels(p.index)
        ax.set_xlabel("max_features"); ax.set_ylabel("C"); ax.set_title(title, loc="left", fontweight="bold"); ax.grid(False)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                ax.text(j, i, f"{p.values[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color="white" if p.values[i, j] > p.values.mean() else INK)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Interacción C × max_features (los demás hiperparámetros en el óptimo)", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(fig_dir / "16_interaccion_C_max_features.png", facecolor="white"); plt.close(fig)
    # 17 antes/después
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(comp)); w = 0.38
    ax.bar(x - w / 2, comp["train_f1"], w, color=TRAIN_C, label="F1 entrenamiento"); ax.bar(x + w / 2, comp["val_f1"], w, color=VAL_C, label="F1 validación")
    for i, r in comp.iterrows():
        ax.text(i - w / 2, r["train_f1"] + 0.01, f"{r['train_f1']:.2f}", ha="center", fontsize=8.5); ax.text(i + w / 2, r["val_f1"] + 0.01, f"{r['val_f1']:.2f}", ha="center", fontsize=8.5)
        ax.text(i, 0.03, f"gap {r['gap']:.2f}", ha="center", fontsize=8.5, color=ACC, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([c.replace(": ", ":\n", 1) for c in comp["configuración"]], fontsize=8.5); ax.set_ylim(0, 1.08); ax.set_ylabel("F1 macro (CV por grupo)")
    ax.set_title("Antes y después de la optimización", loc="left", fontweight="bold"); ax.legend(loc="upper right", ncol=2)
    fig.tight_layout(); fig.savefig(fig_dir / "17_antes_despues_optimizacion.png", facecolor="white"); plt.close(fig)


def main(n_trials: int, seed: int) -> dict:
    setup_logging()
    cfg = load_config(); out = ROOT / "results/metrics/optimizacion"; out.mkdir(parents=True, exist_ok=True); fig_dir = ROOT / "results/figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(Path(cfg["paths"]["splits_dir"]) / "train.csv")
    X, y, g = tr["texto"].to_numpy(), tr[cfg["columns"]["target"]].to_numpy(), tr["grupo_url"].to_numpy()
    t0 = time.time()
    study, trials = O.run_study(build_model, X, y, g, n_trials=n_trials, seed=seed)
    trials.to_csv(out / "trials.csv", index=False)
    best = O._typed({k: study.best_params[k] for k in O.SPACE})
    imp = O.importance(study); imp.to_csv(out / "importancia.csv", index=False)
    pdp = O.partial_dependence(build_model, best, X, y, g, seed=seed); pdp.to_csv(out / "dependencia_parcial.csv", index=False)
    grid = O.interaction_grid(build_model, best, X, y, g, seed=seed); grid.to_csv(out / "interacciones.csv", index=False)
    sens = O.sensitivity_table(pdp, best); sens.to_csv(out / "sensibilidad.csv", index=False)
    base_r = O.cv_score(build_model, O.BASELINE, X, y, g, seed=seed); best_r = O.cv_score(build_model, best, X, y, g, seed=seed)
    one_se = O.one_se_choice(trials, study.best_value, base_r["val_std"]); one_se_r = O.cv_score(build_model, one_se, X, y, g, seed=seed)
    comp = pd.DataFrame([{"configuración": "Base: C=1, 5 000 feat., 1-2 gramas", **base_r}, {"configuración": "Óptimo Optuna: " + ", ".join(f"{k}={v}" for k, v in best.items() if k in ("C", "max_features", "ngram_max")), **best_r},
                         {"configuración": "Regla 1-DE (menor gap): " + ", ".join(f"{k}={v}" for k, v in one_se.items() if k in ("C", "max_features", "ngram_max")), **one_se_r}])
    comp["configuración"] = comp["configuración"].str.replace("C=", "C=").str.replace(r"C=(\d+\.\d{3})\d*", r"C=\1", regex=True)
    comp.to_csv(out / "comparacion_antes_despues.csv", index=False)
    plots(trials, imp, pdp, grid, comp, fig_dir)
    resumen = {"n_trials": int(len(trials)), "segundos": round(time.time() - t0, 1), "mejor": best, "mejor_val_f1": float(study.best_value), "base": O.BASELINE, "base_cv": base_r, "mejor_cv": best_r,
               "regla_1se": one_se, "regla_1se_cv": one_se_r, "importancia": imp.to_dict("records"), "sensibilidad": sens.to_dict("records"), "espacio": {k: (list(v[1]) if v[0] == "cat" else {"log": True, "min": v[1], "max": v[2]}) for k, v in O.SPACE.items()}}
    json.dump(resumen, open(out / "resumen.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: resumen[k] for k in ["n_trials", "segundos", "mejor", "mejor_val_f1", "base_cv", "mejor_cv", "regla_1se", "regla_1se_cv"]}, ensure_ascii=False, indent=1, default=float))
    print(imp.to_string()); print(sens.round(3).to_string())
    return resumen


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=60); ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args(); main(a.trials, a.seed)
