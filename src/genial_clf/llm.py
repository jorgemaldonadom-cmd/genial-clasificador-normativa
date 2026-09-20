"""Rama LLM few-shot: construcción del prompt, llamada a la API, parseo de primera y segunda dimensión, repeticiones.

Política de idioma (Decisión 4): definiciones en español; la norma se presenta en su idioma original, sin traducción.
Cada ejecución registra la versión exacta del modelo y la temperatura para que el resultado sea auditable.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def load_prompt_template(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_prompt(template: str, examples_block: str, titulo: str, observaciones: str) -> str:
    return (template.replace("{ejemplos}", examples_block)
                    .replace("{titulo}", titulo.strip())
                    .replace("{observaciones}", observaciones.strip()))


_LINE = re.compile(r"^(Primera|Segunda)\s*:\s*(.+?)\s*$", re.I | re.M)


def parse_response(text: str, classes: list[str]) -> tuple[str | None, str | None]:
    """Extrae «Primera: X» y «Segunda: Y»; normaliza a los nombres canónicos de las clases."""
    found = {m.group(1).lower(): m.group(2) for m in _LINE.finditer(text)}
    norm = {c.lower(): c for c in classes}

    def canon(v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip().strip(".").lower()
        for k, c in norm.items():
            if v.startswith(k[:6]):
                return c
        return None

    return canon(found.get("primera")), canon(found.get("segunda"))


class AnthropicClassifier:
    """Cliente mínimo. Requiere ANTHROPIC_API_KEY en el entorno. Con `dry_run=True` no llama a la API."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 40, dry_run: bool = False):
        self.model, self.temperature, self.max_tokens, self.dry_run = model, temperature, max_tokens, dry_run
        self._client = None
        if not dry_run:
            import anthropic  # importación diferida para que el resto del pipeline no dependa del SDK

            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def complete(self, system: str, user: str) -> str:
        if self.dry_run:
            return "Primera: Pedagógico\nSegunda: Ética"
        for attempt in range(4):
            try:
                msg = self._client.messages.create(
                    model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
                    system=system, messages=[{"role": "user", "content": user}],
                )
                return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            except Exception as e:  # reintento con espera exponencial ante límites de cuota
                wait = 2 ** attempt
                log.warning("error en la API (%s); reintento en %ds", e, wait)
                time.sleep(wait)
        raise RuntimeError("la API no respondió tras 4 intentos")


def classify_blind(test: pd.DataFrame, fewshot: pd.DataFrame, cfg: dict, template_path: str | Path,
                   repeats: int | None = None, dry_run: bool = False) -> pd.DataFrame:
    """Clasifica el conjunto ciego `repeats` veces y devuelve una fila por (id_norma, ejecución)."""
    classes = cfg["classes"]
    target = cfg["columns"]["target"]
    template = load_prompt_template(template_path)
    system, user_tmpl = template.split("=== USUARIO ===", 1)
    ex_block = "\n".join(f"[Ejemplo {i}] Texto: «{t}» → Dimensión: {d}"
                         for i, (t, d) in enumerate(zip(fewshot["texto"], fewshot[target]), start=1))
    clf = AnthropicClassifier(cfg["llm"]["model"], cfg["llm"]["temperature"], cfg["llm"]["max_tokens"], dry_run)
    rows = []
    for run in range(1, (repeats or cfg["llm"]["repeats"]) + 1):
        for r in test.itertuples(index=False):
            titulo, obs = r.texto.split(". ", 1) if ". " in r.texto else (r.texto, "")
            prompt = build_prompt(user_tmpl, ex_block, titulo, obs)
            raw = clf.complete(system.strip(), prompt)
            p1, p2 = parse_response(raw, classes)
            rows.append({"id_norma": r.id_norma, "run": run, "pred": p1, "pred_2": p2, "raw": raw.strip(),
                         "model": cfg["llm"]["model"], "temperature": cfg["llm"]["temperature"]})
    out = pd.DataFrame(rows)
    n_bad = int(out["pred"].isna().sum())
    if n_bad:
        log.warning("%d respuestas no parseables (se registran como NA y cuentan como error)", n_bad)
    return out


def stability(preds: pd.DataFrame) -> dict:
    """Proporción de normas con la misma primera dimensión en todas las ejecuciones."""
    agree = preds.groupby("id_norma")["pred"].nunique() == 1
    return {"runs": int(preds["run"].nunique()), "stable_fraction": float(agree.mean())}


def save(preds: pd.DataFrame, path: str | Path) -> None:
    preds.to_csv(path, index=False)
    meta = {"model": preds["model"].iloc[0], "temperature": float(preds["temperature"].iloc[0]),
            "runs": int(preds["run"].nunique()), "n": int(preds["id_norma"].nunique())}
    Path(str(path) + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
