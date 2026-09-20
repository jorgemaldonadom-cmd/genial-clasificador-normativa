"""Pruebas unitarias del pipeline sobre la muestra pública (data/samples/sample_corpus_40.csv).

Ejecutar: python -m pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genial_clf import baselines, classic, data, evaluate, imn, llm, split  # noqa: E402
from genial_clf.config import load_config  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["split"]["test_size"] = 10
    c["split"]["fewshot_per_class"] = 1
    c["split"]["min_non_spanish_test"] = 0
    c["classic"]["cv_folds"] = 2
    c["evaluation"]["bootstrap_iters"] = 50
    return c


@pytest.fixture(scope="module")
def corpus(cfg):
    df = pd.read_csv(ROOT / "data/samples/sample_corpus_40.csv")
    return data.preprocess(df, cfg)


def test_preprocess_columns(corpus, cfg):
    assert {"texto", "idioma", "grupo_url", "id_norma"} <= set(corpus.columns)
    assert corpus[cfg["columns"]["target"]].isna().sum() == 0
    assert set(corpus["idioma"]) <= {"es", "pt", "en", "fr"}


def test_tipo_norma_normalizado():
    s = pd.Series(["Estrategia Nacional", "Estrategia nacional", " Lineamiento  Académico "])
    assert data.normalize_tipo_norma(s).nunique() == 2


def test_splits_sin_fuga(corpus, cfg):
    sp = split.make_splits(corpus, cfg)
    ids = lambda k: set(sp[k]["id_norma"])
    assert not ids("train") & ids("test")
    assert not ids("fewshot") & (ids("train") | ids("test"))
    assert not set(sp["train"]["grupo_url"]) & set(sp["test"]["grupo_url"])
    assert len(sp["fewshot"]) == 5 * cfg["split"]["fewshot_per_class"]


def test_piso_aritmetico_corpus_real():
    floor = baselines.arithmetic_floor({"Pedagógico": 294, "Ética": 265, "Administrativo": 120, "Protección de Datos": 71, "Técnico": 63})
    assert abs(floor["accuracy"] - 0.362) < 0.001
    assert abs(floor["f1_macro"] - 0.106) < 0.001


def test_clasico_entrena_y_predice(corpus, cfg):
    sp = split.make_splits(corpus, cfg)
    pipe, pred = classic.fit_predict(sp["train"], sp["test"], cfg)
    assert len(pred) == len(sp["test"])
    assert set(pred["pred"]) <= set(cfg["classes"])
    assert (pred["conf"].between(0, 1)).all()
    assert (pred["pred"] != pred["pred_2"]).all()


def test_wilson_y_mcnemar():
    lo, hi = evaluate.wilson_ci(70, 100)
    assert 0.60 < lo < 0.70 < hi < 0.79
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    a = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 1])
    b = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    m = evaluate.mcnemar(y, a, b)
    assert m["n01"] + m["n10"] == 4 and 0 <= m["p_value"] <= 1


def test_parse_respuesta_llm(cfg):
    p1, p2 = llm.parse_response("Primera: Protección de datos\nSegunda: ética.", cfg["classes"])
    assert p1 == "Protección de Datos" and p2 == "Ética"
    assert llm.parse_response("no sé", cfg["classes"]) == (None, None)


def test_enrutamiento_obligatorio(cfg):
    preds = pd.DataFrame({"id_norma": [1, 2, 3], "pred": ["Ética"] * 3, "pred_2": ["Pedagógico"] * 3, "conf": [0.95, 0.95, 0.40]})
    # 25 normas de México (bien representado) y 1 de Cuba (subrepresentado)
    meta = pd.DataFrame({"id_norma": [1, 2, 3] + list(range(100, 123)),
                         "País": ["México", "Cuba", "México"] + ["México"] * 23,
                         "idioma": ["es"] * 26})
    routed = imn.route(preds, meta, cfg).set_index("id_norma")
    assert not routed.loc[1, "a_revision"]      # confianza alta, país bien representado
    assert routed.loc[2, "a_revision"]          # Cuba: país subrepresentado -> revisión aunque la confianza sea alta
    assert routed.loc[3, "a_revision"]          # baja confianza
