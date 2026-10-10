"""Funciones auxiliares compartidas: rutas, configuración, logging, semillas y validación de entradas."""
from __future__ import annotations

import logging
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:  # permite `import genial_clf` desde scripts, cuadernos y la app sin instalar el paquete
    sys.path.insert(0, str(SRC))

from genial_clf.config import load_config, sha256_of  # noqa: E402  (re-exportados)

CLASSES = ["Pedagógico", "Ética", "Administrativo", "Protección de Datos", "Técnico"]
LANG_NAMES = {"es": "español", "pt": "portugués", "en": "inglés", "fr": "francés"}


def setup_logging(level: int = logging.INFO, logfile: str | Path | None = None) -> None:
    """Configura logging a consola (y opcionalmente a archivo) con formato uniforme."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(level=level, format="%(asctime)s · %(levelname)s · %(name)s · %(message)s", handlers=handlers, force=True)


def set_seed(seed: int = 2026) -> None:
    random.seed(seed); np.random.seed(seed)


def clean_text(text: str) -> str:
    """Normaliza espacios y elimina caracteres de control; no altera acentos ni mayúsculas (lo hace el vectorizador)."""
    if text is None:
        return ""
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def validate_norm_text(text: str, min_words: int = 3, max_chars: int = 5000) -> tuple[bool, str]:
    """Valida la entrada de la aplicación. Devuelve (ok, mensaje)."""
    t = clean_text(text)
    if not t:
        return False, "Escriba el título de la norma y, si es posible, una observación breve."
    if len(t.split()) < min_words:
        return False, f"El texto es demasiado corto ({len(t.split())} palabras); se necesitan al menos {min_words}."
    if len(t) > max_chars:
        return False, f"El texto supera los {max_chars} caracteres; resuma la norma (título + observaciones)."
    return True, ""


def build_norm_text(title: str, observations: str = "") -> str:
    """Construye el campo de entrada del modelo tal como en el corpus: «título. observaciones»."""
    title, observations = clean_text(title), clean_text(observations)
    return f"{title}. {observations}".strip(". ").strip() if observations else title
