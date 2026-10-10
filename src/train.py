#!/usr/bin/env python3
"""Entrenamiento del modelo y persistencia en models/.

Uso:
  python src/train.py                  # entrena model_v1 (configuración base) y best_model (configuración optimizada) con train.csv
  python src/train.py --prepare        # además prepara el corpus y las particiones desde data/raw/
  python src/train.py --config best    # solo el modelo optimizado

La configuración optimizada se lee de results/metrics/optimizacion/resumen.json (clave `regla_1se`, elegida por la regla de una
desviación estándar: entre los ensayos a ≤ 1 DE del máximo, el más regularizado); si no existe, usa la configuración regularizada
recomendada por el diagnóstico de la Semana 3.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from utils import ROOT, load_config, set_seed, setup_logging
from data_processing import load_splits, make_partitions, prepare_corpus
from model import MODELS_DIR, build_model, save_model
from genial_clf.optimize import BASELINE, cv_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

log = logging.getLogger("train")
FALLBACK_BEST = {"C": 0.1, "max_features": 2000, "ngram_max": 2, "min_df": 2, "class_weight": "balanced", "sublinear_tf": True}


def best_params() -> tuple[dict, str]:
    p = ROOT / "results/metrics/optimizacion/resumen.json"
    if p.exists():
        r = json.loads(p.read_text(encoding="utf-8")); cfg = r.get("regla_1se") or r["mejor"]
        return {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in cfg.items()}, "results/metrics/optimizacion/resumen.json (regla 1-DE)"
    return FALLBACK_BEST, "configuración regularizada por defecto (sin resultados de optimización)"


def calibrate_threshold(params: dict, X, y, g, cfg: dict) -> dict:
    """Umbral de confianza propio del modelo: el menor umbral con el que el tramo automático (probabilidad máxima ≥ umbral)
    alcanza la exactitud objetivo sobre predicciones fuera de muestra (CV por grupo), conservando una fracción mínima
    de normas en ese tramo. La regularización aplana las probabilidades, por lo que un umbral fijo no sirve para todos los modelos."""
    r = cfg["routing"]; cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg["seed"])
    pipe = build_model(**params, seed=cfg["seed"])
    proba = cross_val_predict(pipe, X, y, cv=cv, groups=g, method="predict_proba")
    classes = np.array(sorted(set(y))); pred = classes[proba.argmax(axis=1)]; conf = proba.max(axis=1); ok = pred == y
    curva = []
    for t in np.round(np.arange(0.20, 0.96, 0.01), 2):
        auto = conf >= t
        curva.append({"umbral": float(t), "fraccion_automatica": float(auto.mean()), "exactitud_automatica": float(ok[auto].mean()) if auto.any() else float("nan")})
    ok_rows = [c for c in curva if c["fraccion_automatica"] >= r["min_auto_fraction"] and c["exactitud_automatica"] >= r["target_auto_accuracy"]]
    elegido = ok_rows[0] if ok_rows else max((c for c in curva if c["fraccion_automatica"] >= r["min_auto_fraction"]), key=lambda c: c["exactitud_automatica"])
    return {"umbral_confianza": elegido["umbral"], "calibracion_umbral": {"criterio": f"exactitud del tramo automático ≥ {r['target_auto_accuracy']} con ≥ {r['min_auto_fraction']:.0%} de normas automáticas (CV por grupo, train)",
            "objetivo_alcanzado": bool(ok_rows), **elegido, "curva": [c for c in curva if round(c["umbral"] * 100) % 5 == 0]}}


def train_and_save(name: str, params: dict, train: pd.DataFrame, cfg: dict, origen: str) -> dict:
    X, y, g = train["texto"].to_numpy(), train[cfg["columns"]["target"]].to_numpy(), train["grupo_url"].to_numpy()
    cv = cv_score(build_model, params, X, y, g, seed=cfg["seed"])
    cal = calibrate_threshold(params, X, y, g, cfg)
    pipe = build_model(**params, seed=cfg["seed"]).fit(X, y)
    meta = {"nombre": name, "origen_parametros": origen, "n_entrenamiento": int(len(train)), "cv_por_grupo": cv, "f1_entrenamiento": float(f1_score(y, pipe.predict(X), average="macro")),
            **cal, "corpus_sha256": cfg["corpus_sha256"], "nota": "entrenado solo con train.csv; el conjunto ciego no interviene"}
    save_model(pipe, MODELS_DIR / f"{name}.pkl", meta)
    log.info("%s: CV val F1 %.3f (gap %.2f) · umbral de confianza %.2f (tramo automático %.0f %%, exactitud %.2f) · guardado", name, cv["val_f1"], cv["gap"],
             cal["umbral_confianza"], 100 * cal["calibracion_umbral"]["fraccion_automatica"], cal["calibracion_umbral"]["exactitud_automatica"])
    return meta


def main(prepare: bool = False, which: str = "both") -> None:
    setup_logging(logfile=ROOT / "results/metrics/train.log"); cfg = load_config(); set_seed(cfg["seed"])
    if prepare:
        df = prepare_corpus(cfg); make_partitions(df, cfg)
    train = load_splits(cfg)["train"]
    out = {}
    if which in ("both", "v1"):
        out["model_v1"] = train_and_save("model_v1", BASELINE, train, cfg, "configuración base del documento técnico")
    if which in ("both", "best"):
        params, origen = best_params(); out["best_model"] = train_and_save("best_model", params, train, cfg, origen)
    (MODELS_DIR / "entrenamiento.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--prepare", action="store_true"); ap.add_argument("--config", choices=["both", "v1", "best"], default="both")
    a = ap.parse_args(); main(a.prepare, a.config)
