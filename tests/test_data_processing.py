"""Pruebas de procesamiento de datos (sobre la muestra pública; no requieren el corpus autorizado)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import build_norm_text, clean_text, load_config, validate_norm_text  # noqa: E402
from data_processing import describe, partitions_from_ids  # noqa: E402
from genial_clf import data  # noqa: E402


@pytest.fixture(scope="module")
def corpus():
    cfg = load_config()
    return data.preprocess(pd.read_csv(ROOT / "data/samples/sample_corpus_40.csv"), cfg), cfg


def test_clean_text_normaliza_espacios_y_control():
    assert clean_text("  Ley   de\tIA\x00 2025 ") == "Ley de IA 2025"
    assert clean_text(None) == ""


def test_validate_norm_text():
    assert validate_norm_text("")[0] is False
    assert validate_norm_text("Ley IA")[0] is False
    assert validate_norm_text("Ley de inteligencia artificial para universidades")[0] is True
    assert validate_norm_text("x " * 3000)[0] is False


def test_build_norm_text():
    assert build_norm_text("Título", "obs") == "Título. obs"
    assert build_norm_text("Título", "") == "Título"


def test_preprocess_sample(corpus):
    df, cfg = corpus
    assert len(df) == 40 and {"texto", "idioma", "grupo_url", "id_norma"} <= set(df.columns)
    assert set(df[cfg["columns"]["target"]].unique()) <= set(cfg["classes"])


def test_partitions_sin_fuga(corpus):
    df, cfg = corpus
    cfg = {**cfg, "split": {**cfg["split"], "fewshot_per_class": 1}}
    sp = partitions_from_ids(df, set(df["id_norma"].head(8)), cfg)
    assert not set(sp["train"]["grupo_url"]) & set(sp["test"]["grupo_url"])
    assert not set(sp["fewshot"]["id_norma"]) & (set(sp["train"]["id_norma"]) | set(sp["test"]["id_norma"]))
    assert len(sp["train"]) + len(sp["fewshot"]) + len(sp["test"]) == 40


def test_describe(corpus):
    df, cfg = corpus
    d = describe(df, cfg)
    assert d["n"] == 40 and d["desbalance"] >= 1 and d["paises"] > 1
