#!/usr/bin/env python3
"""Ejecución del pipeline por etapas.

  python scripts/run_pipeline.py prepare      # E1: corpus depurado, idioma, duplicados, reporte de calidad
  python scripts/run_pipeline.py split        # E2: pool few-shot, prueba ciega, entrenamiento
  python scripts/run_pipeline.py baselines    # E3: clase mayoritaria y línea base de metadata
  python scripts/run_pipeline.py classic      # E4: rama clásica (validación cruzada + predicción del conjunto ciego)
  python scripts/run_pipeline.py llm [--dry-run]   # E5: rama LLM few-shot (requiere ANTHROPIC_API_KEY)
  python scripts/run_pipeline.py evaluate     # E5/E6: comparación pareada; abre test_labels_hidden.csv SOLO aquí
  python scripts/run_pipeline.py all --dry-run

Las etapas escriben en results/ y data/processed|splits/. Ninguna etapa lee las etiquetas del conjunto de
prueba antes de `evaluate`.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genial_clf import baselines, classic, data, evaluate, imn, llm, split  # noqa: E402
from genial_clf.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pipeline")


def stage_prepare(cfg):
    df = data.load_raw(cfg["paths"]["raw_corpus"], cfg["paths"]["raw_sheet"], cfg["corpus_sha256"])
    df = data.preprocess(df, cfg)
    Path(cfg["paths"]["processed"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg["paths"]["processed"], index=False)
    res = Path(cfg["paths"]["results_dir"]); res.mkdir(exist_ok=True)
    (res / "reporte_calidad.md").write_text(data.quality_report(df, cfg), encoding="utf-8")
    leaks = data.leakage_probe(df, cfg)
    leaks.to_csv(res / "fuga_lexica_candidatos.csv", index=False)
    log.info("candidatos a fuga léxica (etiqueta literal en el texto): %d", len(leaks))


def stage_split(cfg):
    df = pd.read_csv(cfg["paths"]["processed"])
    splits = split.make_splits(df, cfg)
    split.write_splits(splits, cfg["paths"]["splits_dir"], cfg)


def _load_splits(cfg):
    d = Path(cfg["paths"]["splits_dir"])
    return pd.read_csv(d / "train.csv"), pd.read_csv(d / "fewshot_examples.csv"), pd.read_csv(d / "test_blind.csv")


def stage_baselines(cfg):
    train, _, test = _load_splits(cfg)
    t = cfg["columns"]["target"]
    res = Path(cfg["paths"]["results_dir"])
    floor = baselines.arithmetic_floor(train[t].value_counts().to_dict())
    maj = baselines.MajorityClass().fit(train[t])
    meta = baselines.MetadataBaseline("Tipo de norma").fit(train, t)
    full = pd.read_csv(cfg["paths"]["processed"])[["id_norma", "Tipo de norma"]]
    test_m = test.merge(full, on="id_norma", how="left")
    pd.DataFrame({"id_norma": test["id_norma"], "pred": maj.predict(len(test)), "pred_2": maj.label_, "conf": 1.0}).to_csv(res / "pred_mayoritaria.csv", index=False)
    pd.DataFrame({"id_norma": test["id_norma"], "pred": meta.predict(test_m), "pred_2": maj.label_, "conf": meta.predict_conf(test_m)}).to_csv(res / "pred_metadata.csv", index=False)
    (res / "piso_aritmetico.json").write_text(json.dumps(floor, indent=1), encoding="utf-8")
    log.info("piso aritmético (entrenamiento): %s", floor)


def stage_classic(cfg):
    train, _, test = _load_splits(cfg)
    res = Path(cfg["paths"]["results_dir"])
    cv = classic.cross_validate(train, cfg)
    cv.to_csv(res / "clasico_cv_predicciones.csv", index=False)
    rep = evaluate.report(cv["y_true"], cv["pred"], cfg["classes"], cv["pred_2"], cfg["evaluation"]["bootstrap_iters"])
    (res / "clasico_cv_reporte.md").write_text(evaluate.summarize(rep, "Rama clásica · validación cruzada (entrenamiento)"), encoding="utf-8")
    pipe, pred = classic.fit_predict(train, test, cfg)
    pred.to_csv(res / "pred_clasico.csv", index=False)
    classic.top_terms(pipe).to_csv(res / "clasico_terminos_por_dimension.csv", index=False)
    log.info("rama clásica: predicciones del conjunto ciego guardadas")


def stage_llm(cfg, dry_run: bool):
    _, fewshot, test = _load_splits(cfg)
    res = Path(cfg["paths"]["results_dir"])
    preds = llm.classify_blind(test, fewshot, cfg, cfg["paths"]["prompt"], dry_run=dry_run)
    llm.save(preds, res / "pred_llm_runs.csv")
    (res / "llm_estabilidad.json").write_text(json.dumps(llm.stability(preds), indent=1), encoding="utf-8")
    # predicción consolidada = ejecución 1 (las demás sirven para medir estabilidad)
    preds[preds["run"] == 1][["id_norma", "pred", "pred_2"]].assign(conf=float("nan")).to_csv(res / "pred_llm.csv", index=False)


def stage_evaluate(cfg):
    """Único punto del pipeline que abre las etiquetas ocultas del conjunto de prueba."""
    res = Path(cfg["paths"]["results_dir"])
    labels = pd.read_csv(Path(cfg["paths"]["splits_dir"]) / "test_labels_hidden.csv")
    _, _, test = _load_splits(cfg)
    t = cfg["columns"]["target"]
    md = ["# Comparación pareada sobre el conjunto de prueba ciego", ""]
    preds = {}
    for name in ("mayoritaria", "metadata", "clasico", "llm"):
        p = res / f"pred_{name}.csv"
        if not p.exists():
            continue
        df = labels.merge(pd.read_csv(p), on="id_norma").merge(test[["id_norma", "idioma", "País"]], on="id_norma")
        df["pred"] = df["pred"].fillna("__NA__")
        preds[name] = df
        rep = evaluate.report(df[t], df["pred"], cfg["classes"], df.get("pred_2"), cfg["evaluation"]["bootstrap_iters"])
        md.append(evaluate.summarize(rep, name))
        md.append(evaluate.by_group(df, t, "pred", "idioma", cfg["classes"]).to_markdown(index=False, floatfmt=".3f"))
        md.append("")
    if "clasico" in preds and "llm" in preds:
        a, b = preds["clasico"], preds["llm"]
        m = evaluate.mcnemar(a[t], a["pred"], b["pred"])
        md.append(f"## McNemar clásico vs. LLM\n\n{json.dumps(m, ensure_ascii=False, indent=1)}\n")
    if "clasico" in preds:
        a = preds["clasico"]
        curve = imn.routing_curve(a, a[t])
        curve.to_csv(res / "curva_enrutamiento.csv", index=False)
        md.append("## Curva enrutamiento-exactitud (rama clásica)\n\n" + curve.to_markdown(index=False, floatfmt=".3f"))
    (res / "comparacion_pareada.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("reporte escrito en results/comparacion_pareada.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prepare", "split", "baselines", "classic", "llm", "evaluate", "all"])
    ap.add_argument("--dry-run", action="store_true", help="la rama LLM no llama a la API (respuesta simulada)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    stages = {"prepare": stage_prepare, "split": stage_split, "baselines": stage_baselines, "classic": stage_classic,
              "llm": lambda c: stage_llm(c, args.dry_run), "evaluate": stage_evaluate}
    for s in (list(stages) if args.stage == "all" else [args.stage]):
        log.info("=== etapa: %s ===", s)
        stages[s](cfg)


if __name__ == "__main__":
    main()
