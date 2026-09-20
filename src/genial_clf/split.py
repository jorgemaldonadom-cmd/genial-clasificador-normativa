"""Muestreo del experimento: pool few-shot, conjunto de prueba ciego y conjunto de entrenamiento.

Reglas:
  * estratificación por dimensión e idioma (Decisiones 2 y 4);
  * los duplicados por URL caen siempre en la misma partición (sin fuga entrenamiento/prueba);
  * el pool few-shot no forma parte ni del entrenamiento ni de la prueba;
  * las etiquetas del conjunto de prueba se guardan en un archivo separado (test_labels_hidden.csv).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _stratum(df: pd.DataFrame, target: str) -> pd.Series:
    return df[target].astype(str) + "|" + df["idioma"].astype(str)


def make_splits(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg["seed"])
    target = cfg["columns"]["target"]
    classes = cfg["classes"]
    n_test, k_shot = cfg["split"]["test_size"], cfg["split"]["fewshot_per_class"]
    min_non_es = cfg["split"]["min_non_spanish_test"]

    # un representante por grupo de URL para muestrear a nivel de grupo
    groups = df.groupby("grupo_url").agg(n=("id_norma", "size")).reset_index()
    single = df[df["grupo_url"].isin(groups.loc[groups["n"] == 1, "grupo_url"])].copy()
    multi = df[~df["id_norma"].isin(single["id_norma"])].copy()
    log.info("normas únicas por URL: %d · en grupos duplicados: %d (van a entrenamiento)", len(single), len(multi))

    # --- pool few-shot: k por dimensión, priorizando textos de longitud media y cubriendo idiomas ---
    fs_ids: list[int] = []
    for c in classes:
        sub = single[(single[target] == c) & single["n_palabras"].between(25, 90)]
        if len(sub) < k_shot:
            sub = single[single[target] == c]
        fs_ids += list(rng.choice(sub["id_norma"].to_numpy(), size=k_shot, replace=False))
    fewshot = single[single["id_norma"].isin(fs_ids)].copy()
    rest = single[~single["id_norma"].isin(fs_ids)].copy()

    # --- conjunto de prueba: asignación proporcional por estrato dimensión×idioma, con mínimo de no hispanohablantes ---
    rest["estrato"] = _stratum(rest, target)
    alloc = (rest["estrato"].value_counts(normalize=True) * n_test).round().astype(int)
    non_es = rest[rest["idioma"] != "es"]
    test_ids: list[int] = []
    for est, k in alloc.items():
        pool = rest[rest["estrato"] == est]["id_norma"].to_numpy()
        k = min(k, len(pool))
        if k > 0:
            test_ids += list(rng.choice(pool, size=k, replace=False))
    # garantizar presencia mínima de normas no hispanohablantes en prueba
    n_non_es = int(rest[rest["id_norma"].isin(test_ids) & (rest["idioma"] != "es")].shape[0])
    if n_non_es < min_non_es:
        extra_pool = non_es[~non_es["id_norma"].isin(test_ids)]["id_norma"].to_numpy()
        add = min(min_non_es - n_non_es, len(extra_pool))
        if add > 0:
            test_ids += list(rng.choice(extra_pool, size=add, replace=False))
        log.warning("solo %d normas no hispanohablantes disponibles en prueba (mínimo deseado %d)", n_non_es + add, min_non_es)
    test = rest[rest["id_norma"].isin(test_ids)].copy()
    train = pd.concat([rest[~rest["id_norma"].isin(test_ids)], multi]).copy()

    assert not set(train["id_norma"]) & set(test["id_norma"])
    assert not set(fewshot["id_norma"]) & (set(test["id_norma"]) | set(train["id_norma"]))
    assert not set(train["grupo_url"]) & set(test["grupo_url"]), "fuga: mismo grupo de URL en entrenamiento y prueba"
    log.info("particiones: entrenamiento %d · few-shot %d · prueba %d", len(train), len(fewshot), len(test))
    return {"train": train, "fewshot": fewshot, "test": test}


def write_splits(splits: dict[str, pd.DataFrame], out_dir: str | Path, cfg: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = cfg["columns"]["target"]
    public_cols = ["id_norma", "País", "idioma", "texto"]
    splits["train"].to_csv(out / "train.csv", index=False)
    splits["fewshot"][public_cols + [target]].to_csv(out / "fewshot_examples.csv", index=False)
    splits["test"][public_cols].to_csv(out / "test_blind.csv", index=False)               # SIN etiqueta
    splits["test"][["id_norma", target]].to_csv(out / "test_labels_hidden.csv", index=False)  # se abre al final
    log.info("archivos escritos en %s", out)
