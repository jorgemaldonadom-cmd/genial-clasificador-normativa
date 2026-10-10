"""Pruebas del modelo: construcción, entrenamiento rápido, persistencia y Predictor."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import load_config  # noqa: E402
from model import Predictor, build_model, load_model, model_params, save_model  # noqa: E402
from genial_clf import data  # noqa: E402
from genial_clf.optimize import BASELINE, cv_score  # noqa: E402


@pytest.fixture(scope="module")
def entrenado(tmp_path_factory):
    cfg = load_config()
    df = data.preprocess(pd.read_csv(ROOT / "data/samples/sample_corpus_40.csv"), cfg)
    pipe = build_model(C=1.0, max_features=500).fit(df["texto"], df["Dominio"])
    path = tmp_path_factory.mktemp("models") / "m.pkl"
    save_model(pipe, path, {"prueba": True})
    return pipe, path, df, cfg


def test_build_model_valida_hiperparametros():
    with pytest.raises(ValueError):
        build_model(C=0)
    with pytest.raises(ValueError):
        build_model(ngram_max=5)


def test_guardar_y_cargar(entrenado):
    pipe, path, df, _ = entrenado
    cargado = load_model(path)
    assert list(cargado.classes_) == list(pipe.classes_)
    assert (cargado.predict(df["texto"].head(5)) == pipe.predict(df["texto"].head(5))).all()
    assert model_params(cargado)["max_features"] == 500 and path.with_suffix(".json").exists()


def test_load_model_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "no.pkl")


def test_predictor(entrenado):
    pipe, path, df, cfg = entrenado
    p = Predictor(pipe=pipe, cfg=cfg)
    r = p.predict_one("Lineamientos para el uso ético de la inteligencia artificial generativa en la docencia universitaria")
    assert r["dimension"] in cfg["classes"] and 0 <= r["probabilidad"] <= 1 and r["segunda_dimension"] != r["dimension"]
    assert abs(sum(r["probabilidades"].values()) - 1) < 1e-6
    with pytest.raises(ValueError):
        p.predict_one("Ley")
    r2 = p.predict_one("Guidance for generative AI in education and research", language="en")
    assert r2["a_revision_humana"] is True and "idioma" in " ".join(r2["motivos_revision"])
    lote = p.predict_many(["Política de datos personales de la universidad y su tratamiento automatizado", "x"])
    assert len(lote) == 2 and lote["error"].iloc[1] != "" and lote["dimension"].iloc[0] in cfg["classes"]


def test_cv_score_devuelve_metricas(entrenado):
    _, _, df, _ = entrenado
    r = cv_score(build_model, {**BASELINE, "max_features": 500}, df["texto"].to_numpy(), df["Dominio"].to_numpy(), df["grupo_url"].to_numpy(), n_splits=2)
    assert {"val_f1", "train_f1", "gap", "val_std"} <= set(r) and 0 <= r["val_f1"] <= 1
