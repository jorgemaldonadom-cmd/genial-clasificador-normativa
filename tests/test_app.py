"""Pruebas de la aplicación Streamlit con streamlit.testing (sin navegador).

Requiere models/best_model.pkl; si no existe, las pruebas se omiten (skip) en lugar de fallar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODEL = ROOT / "models/best_model.pkl"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="models/best_model.pkl no existe: ejecute python src/train.py")


@pytest.fixture
def app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=60)
    at.run()
    return at


def test_app_carga_sin_errores(app):
    assert not app.exception
    assert any("Clasificador" in t.value for t in app.title)


def test_app_clasifica_una_norma(app):
    app.text_input[0].set_value("Lineamientos para el uso de inteligencia artificial generativa en la docencia universitaria")
    app.text_area[0].set_value("Orienta a los docentes sobre el diseño de actividades y evaluaciones con herramientas de IA generativa.")
    app.button[0].click().run()
    assert not app.exception
    assert any("Dimensión propuesta" in m.value for m in app.markdown)
    assert any("Probabilidad" in str(m.label) for m in app.metric)


def test_app_valida_entrada_corta(app):
    app.text_input[0].set_value("Ley")
    app.text_area[0].set_value("")
    app.button[0].click().run()
    assert not app.exception
    assert any("demasiado corto" in e.value for e in app.error)
