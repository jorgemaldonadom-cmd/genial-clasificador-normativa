"""Carga, verificación, limpieza y enriquecimiento del corpus GENIAL.

Decisiones que implementa (véase docs/decisiones_metodologicas_v1.md):
  * el archivo se congela: si el hash SHA-256 no coincide con config.yaml, el pipeline se detiene;
  * el único valor faltante de `Dominio` se corrige con criterio experto documentado (no por imputación);
  * `Tipo de norma` se normaliza (capitalización) pero NUNCA entra como variable predictora;
  * se añade la variable `idioma` detectada sobre el TÍTULO (las observaciones están en español; hallazgo del EDA), con el país como respaldo;
  * se marcan duplicados por URL oficial para que caigan en la misma partición;
  * el texto de entrada del modelo es `Título de la Norma` + `Observaciones`.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from .config import sha256_of

log = logging.getLogger(__name__)

# Corrección experta del único registro sin dimensión: política de privacidad universitaria (UNESUM, Ecuador).
EXPERT_FIXES = {
    ("Ecuador", "Políticas de Privacidad"): "Protección de Datos",
}

# Países del corpus cuyo idioma oficial no es el español (respaldo cuando el texto es demasiado corto).
COUNTRY_LANG = {
    "Brasil": "pt",
    "Jamaica": "en", "Barbados": "en", "Antigua y Barbuda": "en", "Bahamas": "en", "Belice": "en",
    "Guyana": "en", "San Cristóbal y Nieves": "en", "Trinidad y Tobago": "en",
    "Haití": "fr",
}


def load_raw(path: str | Path, sheet: str, expected_sha256: str | None = None) -> pd.DataFrame:
    """Carga el archivo consolidado y verifica que sea la versión congelada."""
    if expected_sha256:
        actual = sha256_of(path)
        if actual != expected_sha256:
            raise ValueError(
                f"El corpus no coincide con la versión congelada.\n  esperado: {expected_sha256}\n  actual:   {actual}"
            )
    df = pd.read_excel(path, sheet_name=sheet)
    log.info("corpus cargado: %d filas × %d columnas", *df.shape)
    return df


def normalize_tipo_norma(s: pd.Series) -> pd.Series:
    """Unifica categorías duplicadas por capitalización ('Estrategia Nacional' / 'Estrategia nacional')."""
    return s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.capitalize()


def apply_expert_fixes(df: pd.DataFrame, target: str, country: str) -> pd.DataFrame:
    df = df.copy()
    for (pais, titulo), label in EXPERT_FIXES.items():
        mask = df[target].isna() & (df[country] == pais) & (df["Título de la Norma"].astype(str).str.strip() == titulo)
        if mask.any():
            df.loc[mask, target] = label
            log.info("corrección experta aplicada: %s / %s -> %s (%d fila)", pais, titulo, label, int(mask.sum()))
    return df


def build_text(df: pd.DataFrame, parts: list[str]) -> pd.Series:
    cols = [df[p].fillna("").astype(str).str.strip() for p in parts]
    text = cols[0]
    for c in cols[1:]:
        text = text.str.cat(c, sep=". ")
    return text.str.replace(r"\s+", " ", regex=True).str.strip()


def detect_language(title: pd.Series, country: pd.Series) -> pd.Series:
    """Idioma de la norma = idioma del TÍTULO (texto original). Hallazgo del EDA (Semana 2): las
    `Observaciones` están redactadas en español por los codificadores casi siempre, incluso para Brasil
    y el Caribe anglófono, así que detectar sobre el texto completo devolvería 'es' para casi todo.
    Respaldo: idioma esperado por país cuando el título es demasiado corto o ambiguo."""
    try:
        from lingua import Language, LanguageDetectorBuilder

        det = (
            LanguageDetectorBuilder.from_languages(Language.SPANISH, Language.PORTUGUESE, Language.ENGLISH, Language.FRENCH)
            .with_minimum_relative_distance(0.25)
            .build()
        )
        code = {Language.SPANISH: "es", Language.PORTUGUESE: "pt", Language.ENGLISH: "en", Language.FRENCH: "fr"}

        def one(t: str, pais: str) -> str:
            t = str(t)
            lang = det.detect_language_of(t) if len(t) >= 20 else None
            if lang is None:
                return COUNTRY_LANG.get(pais, "es")
            return code[lang]

        return pd.Series([one(t, p) for t, p in zip(title, country)], index=title.index)
    except ImportError:  # pragma: no cover
        log.warning("lingua no disponible: idioma asignado solo por país")
        return country.map(COUNTRY_LANG).fillna("es")


def mark_duplicates(df: pd.DataFrame, url_col: str) -> pd.Series:
    """Identificador de grupo de duplicados por URL (misma URL => mismo grupo)."""
    url = df[url_col].astype(str).str.strip().str.lower().str.rstrip("/")
    return url.factorize()[0]


def preprocess(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cols, split = cfg["columns"], cfg["split"]
    df = apply_expert_fixes(df, cols["target"], cols["country"])
    if df[cols["target"]].isna().any():
        raise ValueError("Quedan valores faltantes en la variable objetivo sin corrección documentada")
    unknown = set(df[cols["target"]].unique()) - set(cfg["classes"])
    if unknown:
        raise ValueError(f"Clases no previstas en la variable objetivo: {unknown}")

    df["Tipo de norma"] = normalize_tipo_norma(df["Tipo de norma"])
    df["texto"] = build_text(df, cols["text_parts"])
    df["n_palabras"] = df["texto"].str.split().str.len()
    df["idioma"] = detect_language(df["Título de la Norma"], df[cols["country"]])
    df["grupo_url"] = mark_duplicates(df, cols["url"])
    n_dup = int((df.groupby("grupo_url").size() > 1).sum())
    log.info("grupos de URL repetida: %d", n_dup)
    df["id_norma"] = range(1, len(df) + 1)
    return df


def quality_report(df: pd.DataFrame, cfg: dict) -> str:
    """Reporte de calidad en Markdown (E1 del documento de alcance)."""
    t = cfg["columns"]["target"]
    lines = ["# Reporte de calidad del corpus", ""]
    lines.append(f"- Filas: {len(df)} · Columnas originales: 13 · Texto medio: {df['n_palabras'].mean():.1f} palabras (mediana {df['n_palabras'].median():.0f})")
    lines.append("- Distribución de la variable objetivo:")
    for k, v in df[t].value_counts().items():
        lines.append(f"  - {k}: {v} ({v / len(df) * 100:.1f} %)")
    vc = df[t].value_counts()
    lines.append(f"- Razón de desbalance: {vc.max() / vc.min():.1f}:1")
    lines.append("- Idioma (detección automática con respaldo por país):")
    for k, v in df["idioma"].value_counts().items():
        lines.append(f"  - {k}: {v}")
    top = df[cfg["columns"]["country"]].value_counts()
    lines.append(f"- Países: {top.size} · los 5 primeros concentran {top.head(5).sum() / len(df) * 100:.0f} %")
    lines.append(f"- Grupos de URL repetida: {int((df.groupby('grupo_url').size() > 1).sum())}")
    lines.append(f"- Categorías de `Tipo de norma` tras normalizar: {df['Tipo de norma'].nunique()}")
    lines.append("- Variables excluidas como predictoras por diseño: " + ", ".join(cfg["columns"]["leakage_excluded"]))
    return "\n".join(lines) + "\n"


def leakage_probe(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Busca la etiqueta escrita literalmente dentro del texto (fuga léxica). Devuelve los casos."""
    t = cfg["columns"]["target"]
    pat = {c: re.compile(re.escape(c.split()[0].lower())) for c in cfg["classes"]}
    hits = []
    for _, r in df.iterrows():
        txt = r["texto"].lower()
        if pat[r[t]].search(txt):
            hits.append(r["id_norma"])
    return df[df["id_norma"].isin(hits)][["id_norma", t, "texto"]]
