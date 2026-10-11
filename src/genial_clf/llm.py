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
    """Extrae la primera y la segunda dimensión. Acepta el formato pedido («Primera: X» / «Segunda: Y») y, como
    tolerancia, respuestas sin etiqueta: las dos primeras líneas (o fragmentos) que nombren una clase, en orden."""
    if not isinstance(text, str) or not text.strip():
        return None, None
    norm = {c.lower(): c for c in classes}

    def canon(v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip().strip(".").strip("*").strip().lower()
        for k, c in norm.items():
            if v.startswith(k[:6]):
                return c
        return None

    found = {m.group(1).lower(): m.group(2) for m in _LINE.finditer(text)}
    p1, p2 = canon(found.get("primera")), canon(found.get("segunda"))
    if p1 is None:  # sin etiquetas: buscar nombres de clase en orden de aparición
        hits = []
        for frag in re.split(r"[\n,;/]+", text):
            c = canon(re.sub(r"^\s*(primera|segunda|1|2)\s*[:.)-]\s*", "", frag, flags=re.I))
            if c and c not in hits:
                hits.append(c)
        p1 = hits[0] if hits else None
        p2 = hits[1] if len(hits) > 1 else p2
    return p1, p2


class QuotaExhausted(RuntimeError):
    """La API rechaza las llamadas por cuota (429 repetido): detener sin perder el progreso guardado."""


class AnthropicClassifier:
    """Cliente mínimo. Requiere ANTHROPIC_API_KEY en el entorno. Con `dry_run=True` no llama a la API."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 40, dry_run: bool = False):
        self.model, self.temperature, self.max_tokens, self.dry_run = model, temperature, max_tokens, dry_run
        self._client = None
        if not dry_run:
            import anthropic  # importación diferida para que el resto del pipeline no dependa del SDK

            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    _temperature_ok: bool | None = None   # None = sin probar; False = el modelo/SDK no admite el parámetro

    def complete(self, system: str, user: str) -> str:
        """Llama a la API. La temperatura se envía como parámetro extra (`extra_body`): los modelos recientes y el SDK ≥ 1.x
        ya no la admiten; si la API la rechaza, se omite para el resto de la ejecución y queda registrado en el log."""
        if self.dry_run:
            return "Primera: Pedagógico\nSegunda: Ética"
        n429 = 0
        for attempt in range(8):
            kw = dict(model=self.model, max_tokens=self.max_tokens, system=system, messages=[{"role": "user", "content": user}])
            if self.temperature is not None and self._temperature_ok is not False:
                kw["extra_body"] = {"temperature": self.temperature}
            try:
                msg = self._client.messages.create(**kw)
                if self._temperature_ok is None:
                    self._temperature_ok = "extra_body" in kw
                return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            except Exception as e:
                m = str(e)
                if "temperature" in m and self._temperature_ok is not False:
                    self._temperature_ok = False
                    log.warning("el modelo %s no admite `temperature`; se continúa sin fijarla (la estabilidad se mide con las repeticiones)", self.model)
                    continue
                if "429" in m or "rate_limit" in m:
                    n429 += 1
                    if n429 >= 4:
                        raise QuotaExhausted(f"límite de la API alcanzado para {self.model}; el progreso queda guardado y la celda puede reanudarse") from e
                    wait = 30
                elif "401" in m or "authentication" in m.lower() or "404" in m or "not_found" in m.lower():
                    raise RuntimeError(f"la API rechazó la llamada ({m[:200]}): revise la clave y el nombre del modelo en config.yaml") from e
                else:
                    wait = min(2 ** attempt, 60)
                log.warning("error en la API (%s); reintento en %ds", m[:160], wait)
                time.sleep(wait)
        raise RuntimeError("la API no respondió tras 8 intentos")


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

        n429 = 0
        for attempt in range(8):
            try:
                r = self._client.models.generate_content(
                    model=self.model, contents=user,
                    config=types.GenerateContentConfig(system_instruction=system, temperature=self.temperature, max_output_tokens=max(self.max_tokens, 64)),
                )
                return r.text or ""
            except Exception as e:
                msg = str(e)
                if "404" in msg or "NOT_FOUND" in msg:
                    raise RuntimeError(f"el modelo {self.model} no está disponible para esta clave; use model_gemini: auto en config.yaml (modelos disponibles: {self.available_models(self._client)[:10]})") from e
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:   # límite por minuto (espera) o cuota diaria agotada (abandonar con progreso guardado)
                    n429 += 1
                    if n429 >= 3 or "per_day" in msg.lower() or "daily" in msg.lower():
                        raise QuotaExhausted(f"cuota de la API agotada para {self.model}; el progreso queda guardado y la celda puede reanudarse más tarde") from e
                    wait = 65
                else:                                               # 503 u otros: espera exponencial
                    wait = min(5 * 2 ** attempt, 120)
                log.warning("error en la API (%s); reintento en %ds", msg[:160], wait)
                time.sleep(wait)
        raise RuntimeError("la API no respondió tras 8 intentos")


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
                   repeats: int | None = None, dry_run: bool = False, checkpoint: str | Path | None = None,
                   progress: bool = True) -> pd.DataFrame:
    """Clasifica el conjunto ciego `repeats` veces y devuelve una fila por (id_norma, ejecución).

    Con `checkpoint` (CSV) guarda cada respuesta al instante y, si se vuelve a llamar, **reanuda** desde donde quedó:
    pensado para cuotas gratuitas que se agotan a mitad de la ejecución. `llm.min_interval_s` en config.yaml espacia las
    llamadas para no superar el límite por minuto."""
    classes = cfg["classes"]
    target = cfg["columns"]["target"]
    template = load_prompt_template(template_path)
    system, user_tmpl = template.split("=== USUARIO ===", 1)
    ex_block = "\n".join(f"[Ejemplo {i}] Texto: «{t}» → Dimensión: {d}"
                         for i, (t, d) in enumerate(zip(fewshot["texto"], fewshot[target]), start=1))
    clf = make_classifier(cfg, dry_run)
    min_interval = 0 if dry_run else float(cfg["llm"].get("min_interval_s", 0))
    cols = ["id_norma", "run", "pred", "pred_2", "raw", "model", "provider", "temperature"]
    rows: list[dict] = []
    done: set[tuple[int, int]] = set()
    if checkpoint and Path(checkpoint).exists():
        prev = pd.read_csv(checkpoint, keep_default_na=False)
        prev = prev[prev["raw"].astype(str).str.strip() != ""]          # las respuestas vacías se vuelven a consultar
        prev[["pred", "pred_2"]] = [list(parse_response(t, classes)) for t in prev["raw"].astype(str)]   # re-parseo con el parser vigente
        rows = prev.to_dict("records"); done = {(int(a), int(b)) for a, b in zip(prev["id_norma"], prev["run"])}
        log.info("reanudando: %d respuestas ya guardadas en %s (%d sin parsear)", len(rows), checkpoint, int(prev["pred"].isna().sum() if len(prev) else 0))
    total = len(test) * (repeats or cfg["llm"]["repeats"]); t_last = 0.0
    try:
        for run in range(1, (repeats or cfg["llm"]["repeats"]) + 1):
            for r in test.itertuples(index=False):
                if (int(r.id_norma), run) in done:
                    continue
                titulo, obs = r.texto.split(". ", 1) if ". " in r.texto else (r.texto, "")
                prompt = build_prompt(user_tmpl, ex_block, titulo, obs)
                wait = min_interval - (time.time() - t_last)
                if wait > 0:
                    time.sleep(wait)
                raw = clf.complete(system.strip(), prompt); t_last = time.time()
                p1, p2 = parse_response(raw, classes)
                temp = cfg["llm"]["temperature"] if getattr(clf, "_temperature_ok", True) is not False else "n/a"
                rows.append({"id_norma": r.id_norma, "run": run, "pred": p1, "pred_2": p2, "raw": raw.strip(),
                             "model": clf.model, "provider": cfg["llm"].get("provider", "anthropic"), "temperature": temp})
                done.add((int(r.id_norma), run))
                if checkpoint:
                    pd.DataFrame(rows, columns=cols).to_csv(checkpoint, index=False)
                if progress and len(rows) % 10 == 0:
                    print(f"  {len(rows)}/{total} respuestas", end="\r", flush=True)
    except QuotaExhausted as e:
        log.warning("%s (%d/%d respuestas guardadas)", e, len(rows), total)
        if not rows:
            raise
    out = pd.DataFrame(rows, columns=cols)
    out.attrs["completo"] = len(out) == total; out.attrs["total"] = total
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
    t0 = preds["temperature"].iloc[0]
    meta = {"model": preds["model"].iloc[0], "temperature": (None if str(t0) in ("n/a", "nan") else float(t0)),
            "runs": int(preds["run"].nunique()), "n": int(preds["id_norma"].nunique())}
    Path(str(path) + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
