#!/usr/bin/env python3
"""Evaluación final de los modelos guardados sobre el conjunto de prueba a ciegas.

Uso:  python src/evaluate.py [--models model_v1 best_model]
Lee las etiquetas ocultas del conjunto ciego (único lugar del pipeline final donde ocurre), evalúa cada modelo con
intervalos de confianza y McNemar frente a las líneas base, desagrega por idioma/región y escribe
results/metrics/evaluacion_final.json y results/metrics/evaluacion_final.csv.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from utils import ROOT, load_config, setup_logging
from data_processing import load_splits
from model import load_model, load_model_metadata, MODELS_DIR
from genial_clf import baselines, evaluate as E

log = logging.getLogger("evaluate")


def evaluate_models(names: list[str], cfg: dict | None = None, write: bool = True) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or load_config(); C5 = cfg["classes"]; t = cfg["columns"]["target"]
    sp = load_splits(cfg, with_hidden_labels=True); train, test, labels = sp["train"], sp["test"], sp["test_labels"]
    test = test.merge(labels, on="id_norma"); y = test[t].to_numpy()
    preds, rows, detalle = {}, [], {}
    maj = baselines.MajorityClass().fit(train[t]); preds["clase_mayoritaria"] = np.array(maj.predict(len(test)))
    for name in names:
        pipe = load_model(MODELS_DIR / f"{name}.pkl"); proba = pipe.predict_proba(test["texto"]); cls = np.asarray(pipe.classes_)
        order = np.argsort(-proba, axis=1); p1, p2 = cls[order[:, 0]], cls[order[:, 1]]; preds[name] = p1
        rep = E.report(y, p1, C5, p2, cfg["evaluation"]["bootstrap_iters"])
        mc = E.mcnemar(y, p1, preds["clase_mayoritaria"])
        rows.append({"modelo": name, "n": rep["n"], "exactitud": rep["accuracy"], "IC95 exactitud": rep["accuracy_ci_wilson"], "F1 macro": rep["f1_macro"], "IC95 F1 macro": rep["f1_macro_ci_bootstrap"],
                     "top-2": rep["top2_accuracy"], "kappa": rep["kappa_vs_reference"], "McNemar vs mayoritaria p": mc["p_value"], **{f"F1 {c}": float(v) for c, v in zip(C5, rep["per_class"]["f1"])}})
        d = test.assign(pred=p1, conf=proba.max(axis=1))
        detalle[name] = {"por_idioma": E.by_group(d, t, "pred", "idioma", C5).to_dict("records"), "confusion": rep["confusion"].to_dict(), "metadata": load_model_metadata(MODELS_DIR / f"{name}.pkl")}
    if len(names) == 2:
        mc = E.mcnemar(y, preds[names[0]], preds[names[1]]); detalle["mcnemar_entre_modelos"] = {"a": names[0], "b": names[1], **mc}
    tabla = pd.DataFrame(rows)
    if write:
        out = ROOT / "results/metrics"; out.mkdir(parents=True, exist_ok=True)
        tabla.to_csv(out / "evaluacion_final.csv", index=False)
        json.dump({"tabla": tabla.to_dict("records"), "detalle": detalle}, open(out / "evaluacion_final.json", "w"), ensure_ascii=False, indent=1, default=float)
    return tabla, detalle


if __name__ == "__main__":
    setup_logging(); ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=["model_v1", "best_model"]); a = ap.parse_args()
    tabla, det = evaluate_models(a.models); print(tabla.round(3).to_string())
    if "mcnemar_entre_modelos" in det:
        print("McNemar entre modelos:", det["mcnemar_entre_modelos"])
