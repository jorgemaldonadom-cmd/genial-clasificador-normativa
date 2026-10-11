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


class GeminiClassifier:
    """Cliente mínimo para Google Gemini (SDK `google-genai`). Requiere GEMINI_API_KEY en el entorno."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 40, dry_run: bool = False):
        self.model, self.temperature, self.max_tokens, self.dry_run = model, temperature, max_tokens, dry_run
        self._client = None
        if not dry_run:
            from google import genai  # importación diferida

            logging.getLogger("google_genai").setLevel(logging.WARNING); logging.getLogger("httpx").setLevel(logging.WARNING)
            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            if model == "auto":
                self.model = self.pick_model()

    @staticmethod
    def available_models(client) -> list[str]:
        """Modelos de la cuenta que admiten generateContent (los nombres cambian con el tiempo: 404 si el modelo se retiró)."""
        out = []
        for m in client.models.list():
            acts = getattr(m, "supported_actions", None) or []
            if not acts or "generateContent" in acts:
                out.append(m.name.replace("models/", ""))
        return out

    def pick_model(self) -> str:
        """Elige el modelo «flash» de texto más reciente disponible para la clave (excluye lite, imagen, audio, embeddings, previews)."""
        names = self.available_models(self._client)
        excl = ("lite", "image", "tts", "audio", "live", "embedding", "preview", "exp", "thinking", "native", "robotics", "computer")
        cand = [n for n in names if "flash" in n and not any(x in n for x in excl)] or [n for n in names if n.startswith("gemini") and not any(x in n for x in excl)]
        if not cand:
            raise RuntimeError(f"ningún modelo Gemini de texto disponible para esta clave; modelos vistos: {names[:20]}")
        cand.sort(key=lambda n: tuple(int(x) if x.isdigit() else 0 for x in re.findall(r"\d+", n)), reverse=True)
        log.info("modelos Gemini disponibles: %s → elegido %s", ", ".join(cand[:8]), cand[0])
        return cand[0]

    def complete(self, system: str, user: str) -> str:
        if self.dry_run:
            return "Primera: Pedagógico\nSegunda: Ética"
        from google.genai import types

        for attempt in range(4):
            try:
                r = self._client.models.generate_content(
                    model=self.model, contents=user,
                    config=types.GenerateContentConfig(system_instruction=system, temperature=self.temperature, max_output_tokens=max(self.max_tokens, 64)),
                )
                return r.text or ""
            except Exception as e:  # reintento con espera exponencial ante límites de cuota
                if "404" in str(e) or "NOT_FOUND" in str(e):
                    raise RuntimeError(f"el modelo {self.model} no está disponible para esta clave; use model_gemini: auto en config.yaml (modelos disponibles: {self.available_models(self._client)[:10]})") from e
                wait = 2 ** attempt
                log.warning("error en la API (%s); reintento en %ds", e, wait)
                time.sleep(wait)
        raise RuntimeError("la API no respondió tras 4 intentos")


PROVIDERS = {"anthropic": (AnthropicClassifier, "ANTHROPIC_API_KEY"), "gemini": (GeminiClassifier, "GEMINI_API_KEY")}


def make_classifier(cfg: dict, dry_run: bool = False):
    """Instancia el cliente según `llm.provider` de config.yaml (anthropic | gemini)."""
    provider = cfg["llm"].get("provider", "anthropic")
    if provider not in PROVIDERS:
        raise ValueError(f"proveedor LLM no soportado: {provider} (use anthropic o gemini)")
    cls, _ = PROVIDERS[provider]
    model = cfg["llm"]["model"] if provider == "anthropic" else cfg["llm"].get("model_gemini", cfg["llm"]["model"])
    return cls(model, cfg["llm"]["temperature"], cfg["llm"]["max_tokens"], dry_run)


def classify_blind(test: pd.DataFrame, fewshot: pd.DataFrame, cfg: dict, template_path: str | Path,
                   repeats: int | None = None, dry_run: bool = False) -> pd.DataFrame:
    """Clasifica el conjunto ciego `repeats` veces y devuelve una fila por (id_norma, ejecución)."""
    classes = cfg["classes"]
    target = cfg["columns"]["target"]
    template = load_prompt_template(template_path)
    system, user_tmpl = template.split("=== USUARIO ===", 1)
    ex_block = "\n".join(f"[Ejemplo {i}] Texto: «{t}» → Dimensión: {d}"
                         for i, (t, d) in enumerate(zip(fewshot["texto"], fewshot[target]), start=1))
    clf = make_classifier(cfg, dry_run)
    rows = []
    for run in range(1, (repeats or cfg["llm"]["repeats"]) + 1):
        for r in test.itertuples(index=False):
            titulo, obs = r.texto.split(". ", 1) if ". " in r.texto else (r.texto, "")
            prompt = build_prompt(user_tmpl, ex_block, titulo, obs)
            raw = clf.complete(system.strip(), prompt)
            p1, p2 = parse_response(raw, classes)
            rows.append({"id_norma": r.id_norma, "run": run, "pred": p1, "pred_2": p2, "raw": raw.strip(),
                         "model": clf.model, "provider": cfg["llm"].get("provider", "anthropic"), "temperature": cfg["llm"]["temperature"]})
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
