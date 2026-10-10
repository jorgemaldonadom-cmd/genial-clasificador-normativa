"""Procesamiento de datos: del archivo consolidado de GENIAL a los conjuntos de entrenamiento y prueba.

Capa de alto nivel sobre `genial_clf.data`, `genial_clf.split` y `genial_clf.coder`:
  prepare_corpus()         carga + verificación de hash + limpieza + idioma + duplicados → data/processed/corpus_depurado.csv
  make_partitions()        entrenamiento / pool few-shot / prueba ciega (por instrumento del investigador si existe) → data/splits/
  load_processed()         lee el corpus depurado
  load_splits()            lee train / fewshot / test (y las etiquetas ocultas solo si se pide explícitamente)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from utils import ROOT, load_config, set_seed

from genial_clf import coder, data, split  # noqa: E402

log = logging.getLogger(__name__)


def prepare_corpus(cfg: dict | None = None, write: bool = True) -> pd.DataFrame:
    """Lee el corpus autorizado, verifica el hash SHA-256 y aplica el preprocesamiento del proyecto."""
    cfg = cfg or load_config()
    raw = Path(cfg["paths"]["raw_corpus"])
    if not raw.exists():
        raise FileNotFoundError(f"No se encuentra el corpus autorizado en {raw}. Cópielo allí (véase data/README.md).")
    df = data.load_raw(raw, cfg["paths"]["raw_sheet"], cfg["corpus_sha256"])
    df = data.preprocess(df, cfg)
    if write:
        out = Path(cfg["paths"]["processed"]); out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False); log.info("corpus depurado escrito en %s (%d filas)", out, len(df))
        rep = Path(cfg["paths"]["results_dir"]) / "metrics" / "reporte_calidad.md"; rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(data.quality_report(df, cfg), encoding="utf-8")
    return df


def load_processed(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    p = Path(cfg["paths"]["processed"])
    if not p.exists():
        raise FileNotFoundError(f"No existe {p}: ejecute prepare_corpus() primero (cuaderno 02 o `python src/train.py --prepare`).")
    return pd.read_csv(p)


def partitions_from_ids(df: pd.DataFrame, test_ids: set[int], cfg: dict) -> dict[str, pd.DataFrame]:
    """Prueba = normas indicadas + sus duplicados por URL; pool few-shot (k por clase) y entrenamiento con el resto."""
    rng = np.random.default_rng(cfg["seed"]); target = cfg["columns"]["target"]
    grupos_test = set(df.loc[df["id_norma"].isin(test_ids), "grupo_url"])
    test = df[df["grupo_url"].isin(grupos_test)].copy(); resto = df[~df["grupo_url"].isin(grupos_test)].copy()
    fs_ids: list[int] = []
    for c in cfg["classes"]:
        pool = resto[(resto[target] == c) & resto["n_palabras"].between(25, 90)]["id_norma"].to_numpy()
        fs_ids += list(rng.choice(pool, cfg["split"]["fewshot_per_class"], replace=False))
    fewshot = resto[resto["id_norma"].isin(fs_ids)].copy(); train = resto[~resto["id_norma"].isin(fs_ids)].copy()
    assert not set(train["grupo_url"]) & set(test["grupo_url"])
    return {"train": train, "fewshot": fewshot, "test": test}


def make_partitions(df: pd.DataFrame | None = None, cfg: dict | None = None, write: bool = True) -> tuple[dict[str, pd.DataFrame], str]:
    """Particiona el corpus. Si existe el instrumento de codificación ciega del investigador (data/raw/codificacion_investigador_A.xlsx),
    el conjunto de prueba son exactamente sus normas; si no, muestreo estratificado por dimensión e idioma."""
    cfg = cfg or load_config(); set_seed(cfg["seed"])
    df = df if df is not None else load_processed(cfg)
    instr_path = ROOT / "data/raw/codificacion_investigador_A.xlsx"
    if instr_path.exists():
        instr, nuevas = coder.load_instrument(instr_path, df)
        splits = partitions_from_ids(df, set(instr["id_norma"]), cfg)
        origen = f"instrumento de codificación ciega del investigador A ({len(instr)} normas en la versión congelada; {len(nuevas)} nuevas fuera de ella)"
        splits["instrumento"], splits["normas_nuevas"] = instr, nuevas
    else:
        splits = split.make_splits(df, cfg); origen = "muestreo estratificado por dimensión e idioma (sin instrumento del investigador)"
    if write:
        split.write_splits({k: splits[k] for k in ("train", "fewshot", "test")}, cfg["paths"]["splits_dir"], cfg)
    log.info("particiones: %s · train %d · few-shot %d · test %d", origen, len(splits["train"]), len(splits["fewshot"]), len(splits["test"]))
    return splits, origen


def load_splits(cfg: dict | None = None, with_hidden_labels: bool = False) -> dict[str, pd.DataFrame]:
    """Lee las particiones escritas. Las etiquetas del conjunto ciego solo se cargan con `with_hidden_labels=True`
    (regla de ceguera: hacerlo únicamente en la evaluación final)."""
    cfg = cfg or load_config(); d = Path(cfg["paths"]["splits_dir"])
    out = {k: pd.read_csv(d / f"{f}.csv") for k, f in [("train", "train"), ("fewshot", "fewshot_examples"), ("test", "test_blind")]}
    if with_hidden_labels:
        out["test_labels"] = pd.read_csv(d / "test_labels_hidden.csv")
    return out


def describe(df: pd.DataFrame, cfg: dict | None = None) -> dict:
    """Resumen descriptivo del corpus depurado para cuadernos y documentación."""
    cfg = cfg or load_config(); t = cfg["columns"]["target"]
    vc = df[t].value_counts()
    return {"n": int(len(df)), "clases": vc.to_dict(), "desbalance": round(float(vc.max() / vc.min()), 2), "idioma": df["idioma"].value_counts().to_dict(),
            "paises": int(df["País"].nunique()), "palabras_mediana": float(df["n_palabras"].median()), "palabras_media": round(float(df["n_palabras"].mean()), 1),
            "grupos_url_repetidos": int((df.groupby("grupo_url").size() > 1).sum())}
