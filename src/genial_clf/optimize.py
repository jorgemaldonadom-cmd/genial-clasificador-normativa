"""Optimización de hiperparámetros y análisis de sensibilidad (Workshop S5 / cuaderno 04).

  * `run_study`            búsqueda bayesiana (Optuna, TPE) del F1 macro en validación cruzada por grupo;
  * `importance`           importancia de cada hiperparámetro (fANOVA de Optuna) sobre los ensayos;
  * `partial_dependence`   dependencia parcial 1-D: F1 de validación al variar un hiperparámetro con los demás fijos en el óptimo;
  * `interaction_grid`     dependencia parcial 2-D (C × max_features) para ver interacciones;
  * `sensitivity_table`    resumen por hiperparámetro: rango explorado, mejor valor, variación del F1 (máx − mín) en su barrido.

La función objetivo evalúa el pipeline TF-IDF + Regresión Logística con `StratifiedGroupKFold` (duplicados por URL en el mismo
pliegue) sobre el conjunto de entrenamiento. El conjunto ciego no interviene.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

log = logging.getLogger(__name__)

SPACE = {  # espacio de búsqueda; se documenta en docs/optimizacion.md
    "C": ("log", 0.01, 30.0),
    "max_features": ("cat", [1000, 2000, 3000, 5000, 8000, 12000, 20000]),
    "ngram_max": ("cat", [1, 2]),
    "min_df": ("cat", [1, 2, 3]),
    "class_weight": ("cat", ["balanced", None]),
    "sublinear_tf": ("cat", [True, False]),
}
BASELINE = {"C": 1.0, "max_features": 5000, "ngram_max": 2, "min_df": 1, "class_weight": "balanced", "sublinear_tf": True}


def cv_score(build_fn, params: dict, X, y, groups, n_splits: int = 5, seed: int = 2026, return_folds: bool = False):
    """F1 macro medio (y por pliegue) en validación cruzada estratificada por grupo."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores, train_scores = [], []
    for tr, va in cv.split(X, y, groups):
        m = clone(build_fn(**params)).fit(X[tr], y[tr])
        scores.append(f1_score(y[va], m.predict(X[va]), average="macro"))
        train_scores.append(f1_score(y[tr], m.predict(X[tr]), average="macro"))
    out = {"val_f1": float(np.mean(scores)), "val_std": float(np.std(scores)), "train_f1": float(np.mean(train_scores)), "gap": float(np.mean(train_scores) - np.mean(scores))}
    return (out, scores) if return_folds else out


def _suggest(trial):
    p = {}
    for k, spec in SPACE.items():
        if spec[0] == "log":
            p[k] = trial.suggest_float(k, spec[1], spec[2], log=True)
        else:
            p[k] = trial.suggest_categorical(k, spec[1])
    return p


def run_study(build_fn, X, y, groups, n_trials: int = 60, seed: int = 2026, n_splits: int = 5, timeout: int | None = None):
    """Estudio Optuna (TPE) que maximiza el F1 macro de validación. Devuelve (study, DataFrame de ensayos)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        p = _suggest(trial); t0 = time.time()
        r = cv_score(build_fn, p, X, y, groups, n_splits=n_splits, seed=seed)
        trial.set_user_attr("train_f1", r["train_f1"]); trial.set_user_attr("gap", r["gap"]); trial.set_user_attr("val_std", r["val_std"]); trial.set_user_attr("segundos", round(time.time() - t0, 1))
        return r["val_f1"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=15), study_name="genial_rl_tfidf")
    study.enqueue_trial(BASELINE)  # el primer ensayo es la configuración base del proyecto
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    rows = []
    for t in study.trials:
        if t.value is None:
            continue
        rows.append({"trial": t.number, "val_f1": t.value, **t.params, **t.user_attrs})
    return study, pd.DataFrame(rows)


def importance(study) -> pd.DataFrame:
    """Ranking de importancia de hiperparámetros (fANOVA; si falla por pocos ensayos, usa el evaluador de media-descenso)."""
    import optuna
    try:
        imp = optuna.importance.get_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))
    except Exception:  # pragma: no cover
        imp = optuna.importance.get_param_importances(study, evaluator=optuna.importance.MeanDecreaseImpurityImportanceEvaluator(seed=0))
    return pd.DataFrame({"hiperparámetro": list(imp.keys()), "importancia": list(imp.values())}).sort_values("importancia", ascending=False).reset_index(drop=True)


def partial_dependence(build_fn, best: dict, X, y, groups, seed: int = 2026, n_splits: int = 5) -> pd.DataFrame:
    """Barrido 1-D de cada hiperparámetro con los demás fijos en el óptimo (dependencia parcial condicional)."""
    grids = {"C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], "max_features": [1000, 2000, 3000, 5000, 8000, 12000, 20000], "ngram_max": [1, 2],
             "min_df": [1, 2, 3], "class_weight": ["balanced", None], "sublinear_tf": [True, False]}
    rows = []
    for k, vals in grids.items():
        for v in vals:
            p = {**best, k: v}; r = cv_score(build_fn, p, X, y, groups, seed=seed, n_splits=n_splits)
            rows.append({"hiperparámetro": k, "valor": v, **r})
            log.info("PD %s=%s → val F1 %.3f (gap %.2f)", k, v, r["val_f1"], r["gap"])
    return pd.DataFrame(rows)


def interaction_grid(build_fn, best: dict, X, y, groups, Cs=(0.03, 0.1, 0.3, 1, 3, 10), feats=(1000, 2000, 5000, 12000), seed: int = 2026, n_splits: int = 5) -> pd.DataFrame:
    """Dependencia parcial 2-D C × max_features (los demás hiperparámetros en el óptimo)."""
    rows = []
    for C in Cs:
        for f in feats:
            r = cv_score(build_fn, {**best, "C": C, "max_features": f}, X, y, groups, seed=seed, n_splits=n_splits)
            rows.append({"C": C, "max_features": f, **r})
    return pd.DataFrame(rows)


def sensitivity_table(pd_df: pd.DataFrame, best: dict) -> pd.DataFrame:
    """Rango de variación del F1 de validación al barrer cada hiperparámetro (los demás fijos): medida simple de sensibilidad."""
    rows = []
    for k, g in pd_df.groupby("hiperparámetro"):
        rows.append({"hiperparámetro": k, "valores explorados": ", ".join(str(v) for v in g["valor"].tolist()), "valor óptimo": best[k],
                     "F1 val. mín": g["val_f1"].min(), "F1 val. máx": g["val_f1"].max(), "variación (pp)": (g["val_f1"].max() - g["val_f1"].min()) * 100,
                     "gap mín": g["gap"].min(), "gap máx": g["gap"].max()})
    return pd.DataFrame(rows).sort_values("variación (pp)", ascending=False).reset_index(drop=True)


def one_se_choice(trials: pd.DataFrame, best_val: float, se: float) -> dict:
    """Regla de una desviación estándar con criterio de menor sobreajuste: entre los ensayos cuyo F1 de validación está a
    ≤ 1 DE (de la configuración base) del máximo, el de menor gap entrenamiento−validación (menos varianza)."""
    cand = trials[trials["val_f1"] >= best_val - se].copy()
    r = cand.sort_values(["gap", "val_f1"], ascending=[True, False]).iloc[0]
    return _typed({k: r[k] for k in SPACE})


def _typed(p: dict) -> dict:
    out = dict(p)
    out["C"] = float(out["C"]); out["max_features"] = int(out["max_features"]); out["ngram_max"] = int(out["ngram_max"]); out["min_df"] = int(out["min_df"])
    out["sublinear_tf"] = bool(out["sublinear_tf"]) if not isinstance(out["sublinear_tf"], str) else out["sublinear_tf"] == "True"
    cw = out["class_weight"]; out["class_weight"] = None if (cw is None or (isinstance(cw, float) and np.isnan(cw)) or cw in ("None", "nan")) else cw
    return out
