#!/usr/bin/env python3
"""Genera los cuadernos del proyecto final (nbformat):

    00_ejecutar_todo.ipynb      cuaderno maestro paso a paso (preparar → entrenar → optimizar → evaluar → auditar → probar → app)
    02_preprocesamiento.ipynb   del archivo consolidado a las particiones
    03_modelado.ipynb           líneas base, modelo TF-IDF + RL, validación cruzada por grupo, modelo v1
    04_optimizacion.ipynb       Optuna + análisis de sensibilidad + regla 1-DE + modelo optimizado (Workshop S5)
    05_evaluacion.ipynb         prueba ciega, enrutamiento, equidad, rama LLM opcional

Uso (desde la raíz del repositorio):
    python notebooks/make_final_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/0{2,3,4,5}_*.ipynb notebooks/00_ejecutar_todo.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def write(name: str, cells: list) -> None:
    nb = nbf.v4.new_notebook(); nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
    nbf.write(nb, OUT / name); print("escrito", name)


HEADER = r'''import json, os, subprocess, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown
warnings.filterwarnings("ignore")
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
ROOT = Path.cwd() if (Path.cwd() / "config.yaml").exists() else Path.cwd().parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT / "src"))
from utils import load_config, setup_logging, set_seed, CLASSES
cfg = load_config(); set_seed(cfg["seed"]); setup_logging()
RES = ROOT / "results/metrics"; FIG = ROOT / "results/figures"; RES.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
print("raíz del repositorio:", ROOT)'''

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# 00 · cuaderno maestro
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
C = []
C.append(md(r"""# 00 · Ejecutar todo el proyecto, paso a paso

**Clasificación automática de normativa sobre IA en educación superior de ALC · Proyecto Integrador · UEES · Grupo 9**

Este cuaderno reproduce el proyecto de principio a fin. Cada paso es una celda independiente que **omite el trabajo ya hecho** (si el artefacto existe no lo recalcula), de modo que puede ejecutarse completo (*Run All*) o celda por celda. Al terminar, el repositorio contiene las particiones, los dos modelos, la optimización, la evaluación final, la auditoría de equidad y las pruebas; la última celda lanza la aplicación web.

| Paso | Qué hace | Tiempo aprox. | Artefactos |
|---|---|---|---|
| 0 | Verifica entorno, dependencias y archivos de datos | segundos | — |
| 1 | Prepara el corpus (hash, limpieza, idioma, duplicados) y las particiones | 10 s | `data/processed/`, `data/splits/` |
| 2 | Optimiza hiperparámetros (Optuna, 60 ensayos) — *opcional, se reutiliza si ya existe* | 6 min | `results/metrics/optimizacion/`, figuras 13-17 |
| 3 | Entrena `model_v1` (base) y `best_model` (optimizado) | 20 s | `models/*.pkl` |
| 4 | Evalúa ambos modelos en el conjunto ciego | 30 s | `results/metrics/evaluacion_final.*` |
| 5 | Auditoría de equidad por jurisdicción | 20 s | `results/metrics/fairness/`, figura 12 |
| 6 | Pruebas automatizadas (`pytest`) | 30 s | — |
| 7 | Rama LLM *few-shot* — *opcional, requiere clave de API* | 10 min | `results/metrics/pred_llm*.csv` |
| 8 | Lanza la aplicación Streamlit | — | http://localhost:8501 |

**Requisitos previos.** Python ≥ 3.11; el archivo autorizado `consolidado_limpio.xlsx` en `data/raw/` y, si se dispone de él, `codificacion_investigador_A.xlsx` en la misma carpeta (sin él, la partición ciega se hace por muestreo estratificado). Ninguno de los dos se versiona.

> Abra este cuaderno desde la carpeta `notebooks/` del repositorio (JupyterLab, VS Code o Jupyter Notebook): la primera celda localiza la raíz automáticamente."""))
C.append(md("## Paso 0 · Entorno\n\nLocaliza la raíz del repositorio, comprueba la versión de Python y las dependencias. Si falta alguna, cambie `INSTALAR = True` y vuelva a ejecutar la celda (instala `requirements.txt` con el `pip` del propio kernel)."))
C.append(code(r'''INSTALAR = False          # True → pip install -r requirements.txt desde este kernel
import importlib, sys, os, subprocess, json, time
import numpy as np, pandas as pd
from IPython.display import display, Image
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / "config.yaml").exists() else Path.cwd().parent
assert (ROOT / "config.yaml").exists(), "No encuentro config.yaml: abra el cuaderno desde la carpeta notebooks/ del repositorio"
os.chdir(ROOT); sys.path.insert(0, str(ROOT / "src"))
print("raíz:", ROOT); print("python:", sys.version.split()[0], "·", sys.executable)
assert sys.version_info >= (3, 11), "Se requiere Python 3.11 o superior"
if INSTALAR:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True); print("dependencias instaladas")
faltan = []
for mod, pip_name in [("pandas", "pandas"), ("numpy", "numpy"), ("sklearn", "scikit-learn"), ("joblib", "joblib"), ("yaml", "PyYAML"), ("openpyxl", "openpyxl"),
                      ("lingua", "lingua-language-detector"), ("matplotlib", "matplotlib"), ("optuna", "optuna"), ("streamlit", "streamlit"), ("plotly", "plotly"), ("pytest", "pytest")]:
    try:
        m = importlib.import_module(mod); print(f"  ✓ {pip_name:28s} {getattr(m, '__version__', '')}")
    except ImportError:
        faltan.append(pip_name); print(f"  ✗ {pip_name:28s} FALTA")
print("\nTodo listo." if not faltan else f"\nFaltan: {faltan} → ponga INSTALAR = True y vuelva a ejecutar esta celda.")'''))
C.append(md("### Archivos de datos\n\nEl pipeline se detiene si el hash SHA-256 del corpus no coincide con el congelado en `config.yaml` (garantía de que todos trabajamos sobre la misma versión)."))
C.append(code(r'''from utils import load_config, sha256_of, setup_logging, set_seed
cfg = load_config(); set_seed(cfg["seed"]); setup_logging()
RES = ROOT / "results/metrics"; FIG = ROOT / "results/figures"
raw = ROOT / cfg["paths"]["raw_corpus"]; instr = ROOT / "data/raw/codificacion_investigador_A.xlsx"
print("corpus autorizado      :", raw, "→", "existe" if raw.exists() else "NO EXISTE (cópielo en data/raw/)")
if raw.exists():
    h = sha256_of(raw); print("  SHA-256:", h[:16] + "…", "✓ coincide" if h == cfg["corpus_sha256"] else "✗ NO coincide con config.yaml")
print("instrumento del investigador:", instr.name, "→", "existe (la prueba ciega serán sus normas)" if instr.exists() else "no existe (prueba ciega por muestreo estratificado)")
FLAGS = dict(REHACER_PARTICIONES=False,   # True → vuelve a generar data/processed y data/splits aunque existan
             EJECUTAR_OPTUNA=False,       # True → corre los 60 ensayos aunque exista results/metrics/optimizacion/resumen.json (~6 min)
             N_TRIALS=60,
             EJECUTAR_LLM=False)          # True → rama LLM few-shot (necesita clave de API; ~10 min y coste de API)
FLAGS'''))
C.append(md("## Paso 1 · Preparar el corpus y las particiones\n\n`prepare_corpus()` carga el Excel, verifica el hash, normaliza, construye el texto (`título. observaciones`), detecta el idioma, marca duplicados por URL y escribe `data/processed/corpus_depurado.csv` más `results/metrics/reporte_calidad.md`. `make_partitions()` separa entrenamiento (673), pool *few-shot* (15) y prueba ciega (125, las normas del instrumento del investigador con sus duplicados). Las etiquetas de la prueba quedan en `test_labels_hidden.csv` y **solo se leen en el paso 4**."))
C.append(code(r'''from data_processing import prepare_corpus, make_partitions, load_splits, load_processed, describe
splits_dir = ROOT / cfg["paths"]["splits_dir"]
if FLAGS["REHACER_PARTICIONES"] or not (splits_dir / "train.csv").exists():
    df = prepare_corpus(cfg); splits, origen = make_partitions(df, cfg); print("particiones generadas ·", origen)
else:
    print("particiones ya existentes en", splits_dir, "(REHACER_PARTICIONES = True para regenerarlas)")
sp = load_splits(cfg); corpus = load_processed(cfg)
print({k: len(v) for k, v in sp.items()}); print(json.dumps(describe(corpus, cfg), ensure_ascii=False, indent=1))'''))
C.append(md("## Paso 2 · Optimización de hiperparámetros (opcional)\n\nBúsqueda bayesiana (Optuna, TPE, 60 ensayos) del F1 macro en validación cruzada por grupo sobre el conjunto de entrenamiento, seguida de importancia fANOVA, dependencia parcial, interacción `C × max_features`, tabla de sensibilidad y comparación antes/después. Tarda unos seis minutos; si `results/metrics/optimizacion/resumen.json` ya existe se reutiliza. El detalle comentado está en `04_optimizacion.ipynb`."))
C.append(code(r'''resumen_opt = ROOT / "results/metrics/optimizacion/resumen.json"
if FLAGS["EJECUTAR_OPTUNA"] or not resumen_opt.exists():
    t0 = time.time(); r = subprocess.run([sys.executable, "scripts/optimizar.py", "--trials", str(FLAGS["N_TRIALS"])], capture_output=True, text=True)
    print(r.stdout[-3000:]); print(r.stderr[-1500:] if r.returncode else f"optimización terminada en {time.time()-t0:.0f} s")
else:
    print("optimización ya existente (EJECUTAR_OPTUNA = True para repetirla)")
r = json.loads(resumen_opt.read_text(encoding="utf-8"))
print("configuración base      :", r["base"], "→ val F1 %.3f · gap %.2f" % (r["base_cv"]["val_f1"], r["base_cv"]["gap"]))
print("mejor ensayo Optuna     :", r["mejor"], "→ val F1 %.3f · gap %.2f" % (r["mejor_cv"]["val_f1"], r["mejor_cv"]["gap"]))
print("regla 1-DE (elegida)    :", r["regla_1se"], "→ val F1 %.3f · gap %.2f" % (r["regla_1se_cv"]["val_f1"], r["regla_1se_cv"]["gap"]))'''))
C.append(md("## Paso 3 · Entrenar y guardar los modelos\n\n`src/train.py` entrena con `train.csv` el modelo base (`model_v1.pkl`) y el optimizado (`best_model.pkl`, hiperparámetros de la regla 1-DE), cada uno con su `.json` de metadatos (hash del corpus, métricas de validación cruzada, fecha) y con su **umbral de confianza calibrado** (menor umbral que da ≥ 85 % de exactitud en el tramo automático sobre predicciones fuera de muestra)."))
C.append(code(r'''import train as T
T.main(prepare=False, which="both")
print(json.dumps(json.loads((ROOT / "models/entrenamiento.json").read_text(encoding="utf-8")), ensure_ascii=False, indent=1)[:1800])'''))
C.append(md("## Paso 4 · Evaluación final en el conjunto ciego\n\nÚnico punto del flujo donde se leen las etiquetas ocultas. Exactitud e IC de Wilson, F1 macro e IC *bootstrap* (2 000 remuestreos), top-2, κ frente a la etiqueta del codificador, McNemar frente a la clase mayoritaria y entre los dos modelos."))
C.append(code(r'''import evaluate as EV
tabla, detalle = EV.evaluate_models(["model_v1", "best_model"], cfg)
display(tabla.round(3))
print("McNemar model_v1 vs best_model:", {k: round(v, 3) if isinstance(v, float) else v for k, v in detalle["mcnemar_entre_modelos"].items()})'''))
C.append(md("## Paso 5 · Auditoría de equidad por jurisdicción\n\nDesempeño por idioma, tamaño del país, región, ámbito y codificador (anonimizado) sobre las predicciones fuera de muestra de la validación cruzada por grupo (n = 813), con IC de Wilson, razón frente al grupo de referencia (regla de los cuatro quintos) y proporción que la regla de gobernanza envía a revisión humana. Si las predicciones de validación cruzada no existen, se generan primero."))
C.append(code(r'''cvp = ROOT / "results/metrics/clasico_cv_predicciones.csv"
if not cvp.exists():
    from genial_clf import classic
    corpus = load_processed(cfg); p = classic.cross_validate(corpus, cfg)
    p.merge(corpus[["id_norma", "idioma", "País", "Ámbito", "Vínculo con Educación"]], on="id_norma").to_csv(cvp, index=False); print("predicciones de validación cruzada generadas")
r = subprocess.run([sys.executable, "scripts/fairness_grupos.py"], capture_output=True, text=True); print(r.stderr[-1200:] if r.returncode else "auditoría ejecutada")
fair = pd.read_csv(ROOT / "results/metrics/fairness/fairness_grupos.csv")
display(fair[fair["dimensión"].isin(["Idioma del título", "Región"])][["dimensión", "grupo", "n", "exactitud", "IC95 inf", "IC95 sup", "razón vs referencia", "% a revisión humana (regla)"]].round(3))
display(Image(filename=str(ROOT / "results/figures/12_fairness_grupos.png"), width=900))'''))
C.append(md("## Paso 6 · Pruebas automatizadas\n\n`tests/test_data_processing.py` (limpieza, texto, validación), `tests/test_model.py` (construcción, guardado, predicción, enrutamiento) y `tests/test_app.py` (la aplicación carga y responde, con `streamlit.testing`)."))
C.append(code(r'''r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_data_processing.py", "tests/test_model.py", "tests/test_app.py"], capture_output=True, text=True)
print(r.stdout[-2500:]); print(r.stderr[-800:] if r.returncode else "")'''))
C.append(md("## Paso 7 · Rama LLM *few-shot* (opcional)\n\nClasifica el conjunto ciego con un modelo de lenguaje a temperatura 0, tres veces, con el *prompt* versionado en `prompts/clasificador_v0.1.md` y los 15 ejemplos del pool. Funciona con **Gemini** (`GEMINI_API_KEY`, gratuito con límites) o con **Anthropic** (`ANTHROPIC_API_KEY`). **Dónde va la clave:** la celda la busca, en este orden, en la variable de entorno, en `data/raw/gemini_key.txt` / `data/raw/anthropic_key.txt` (una sola línea con la clave) o en `.env` (`GEMINI_API_KEY=…`). Los tres lugares están fuera del control de versiones; **nunca pegue la clave en el cuaderno**. Ponga `FLAGS[\"EJECUTAR_LLM\"] = True` para ejecutarla (≈ 375 llamadas; con Gemini Flash, unos 10 minutos y dentro de la cuota gratuita)."))
C.append(md("### 7a · Guardar la clave de API sin escribirla en el cuaderno\n\nEjecute esta celda una sola vez: pide la clave con un cuadro oculto (`getpass`, no queda en la salida) y la guarda en `data/raw/gemini_key.txt`, que está en `.gitignore`. Si el archivo ya existe, no hace nada. Para Anthropic, cambie `PROVEEDOR = \"anthropic\"`."))
C.append(code(r'''PROVEEDOR = "gemini"            # "gemini" | "anthropic"
archivo_clave = ROOT / "data/raw" / f"{PROVEEDOR}_key.txt"
if FLAGS["EJECUTAR_LLM"] and not archivo_clave.exists():
    from getpass import getpass
    clave = getpass(f"Pegue la clave de {PROVEEDOR} y pulse Enter (no se muestra): ").strip()
    if clave:
        archivo_clave.write_text(clave + "\n", encoding="utf-8"); print("clave guardada en", archivo_clave.relative_to(ROOT), "(fuera del control de versiones)")
    else:
        print("no se guardó ninguna clave")
else:
    print("archivo de clave:", "existe" if archivo_clave.exists() else "no existe (ponga FLAGS['EJECUTAR_LLM'] = True y vuelva a ejecutar para introducirla)")'''))
C.append(code(r'''def _buscar_api_key(var, archivo):
    k = os.environ.get(var, "").strip()
    if k: return k, "variable de entorno"
    for f, parse in [(ROOT / "data/raw" / archivo, lambda s: s.strip().splitlines()[0].strip() if s.strip() else ""),
                     (ROOT / ".env", lambda s: next((l.split("=", 1)[1].strip().strip('"').strip("'") for l in s.splitlines() if l.strip().startswith(var)), ""))]:
        if f.exists():
            k = parse(f.read_text(encoding="utf-8"))
            if k: return k, f.name
    return None, None

if FLAGS["EJECUTAR_LLM"]:
    proveedor = None
    for prov, var, archivo in [("gemini", "GEMINI_API_KEY", "gemini_key.txt"), ("anthropic", "ANTHROPIC_API_KEY", "anthropic_key.txt")]:
        key, origen = _buscar_api_key(var, archivo)
        if key:
            os.environ[var] = key; proveedor = prov; break
    if not proveedor:
        print("Sin clave de API: cree data/raw/gemini_key.txt (o anthropic_key.txt) con la clave en una sola línea, o defina GEMINI_API_KEY / ANTHROPIC_API_KEY.")
    else:
        from genial_clf import llm
        cfg_llm = {**cfg, "llm": {**cfg["llm"], "provider": proveedor}}
        modelo = cfg_llm["llm"]["model_gemini"] if proveedor == "gemini" else cfg_llm["llm"]["model"]
        print(f"proveedor: {proveedor} · modelo: {modelo} · clave leída de {origen} · {len(sp['test'])} normas × {cfg['llm']['repeats']} ejecuciones")
        preds_llm = llm.classify_blind(sp["test"], sp["fewshot"], cfg_llm, cfg["paths"]["prompt"]); llm.save(preds_llm, RES / "pred_llm_runs.csv")
        preds_llm[preds_llm["run"] == 1][["id_norma", "pred", "pred_2"]].assign(conf=np.nan).to_csv(RES / "pred_llm.csv", index=False)
        print("rama LLM ejecutada:", llm.stability(preds_llm), "· sin respuesta parseable:", int(preds_llm["pred"].isna().sum()))
else:
    print("rama LLM omitida (FLAGS['EJECUTAR_LLM'] = False)." + (" Hay predicciones guardadas en results/metrics/pred_llm.csv." if (RES / "pred_llm.csv").exists() else ""))'''))
C.append(md("## Paso 8 · Lanzar la aplicación web\n\nLa celda siguiente arranca Streamlit en segundo plano y abre http://localhost:8501 en el navegador. La aplicación carga `models/best_model.pkl`. Para detenerla ejecute la celda que sigue (o cierre el kernel)."))
C.append(code(r'''import webbrowser
app_proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/app.py", "--server.headless", "true", "--server.port", "8501", "--browser.gatherUsageStats", "false"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(4); print("aplicación en http://localhost:8501  (PID", app_proc.pid, ")")
if not os.environ.get("CI") and not os.environ.get("NBCONVERT"):
    webbrowser.open("http://localhost:8501")'''))
C.append(code(r'''# Detener la aplicación
if "app_proc" in globals() and app_proc.poll() is None:
    app_proc.terminate(); print("aplicación detenida")'''))
C.append(md("## Resumen de artefactos generados"))
C.append(code(r'''arte = ["data/processed/corpus_depurado.csv", "data/splits/train.csv", "data/splits/test_blind.csv", "data/splits/test_labels_hidden.csv",
        "results/metrics/optimizacion/resumen.json", "models/model_v1.pkl", "models/best_model.pkl", "results/metrics/evaluacion_final.csv",
        "results/metrics/fairness/fairness_grupos.csv", "results/figures/12_fairness_grupos.png", "results/figures/17_antes_despues_optimizacion.png", "results/metrics/pred_llm.csv"]
for a in arte:
    p = ROOT / a; print(("✓" if p.exists() else "·"), a, f"({p.stat().st_size/1024:.0f} KB)" if p.exists() else "(no generado)")'''))
write("00_ejecutar_todo.ipynb", C)

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# 02 · preprocesamiento
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
C = []
C.append(md(r"""# 02 · Preprocesamiento: del archivo consolidado a las particiones

Este cuaderno documenta y ejecuta la capa de datos del proyecto (`src/data_processing.py` sobre `genial_clf.data`, `genial_clf.split` y `genial_clf.coder`). La exploración descriptiva está en `01_exploracion.ipynb`; aquí interesa **qué transformaciones sufre el corpus antes de llegar al modelo y por qué**.

1. Carga con verificación de integridad (SHA-256).
2. Normalización de categorías y correcciones acordadas con el equipo experto.
3. Construcción del texto de entrada (`título. observaciones`) y limpieza.
4. Detección de idioma y marcado de duplicados por URL.
5. Exclusión de columnas que filtrarían la etiqueta (`Investigador`, `Tipo de norma`, `Fecha`).
6. Partición entrenamiento / pool *few-shot* / prueba ciega, con las etiquetas de la prueba ocultas."""))
C.append(code(HEADER))
C.append(md("## 1. Carga y verificación de integridad\n\nEl corpus autorizado (813 normas × 13 columnas) no se versiona. El hash fijado en `config.yaml` evita trabajar sin saberlo sobre una versión distinta del Excel."))
C.append(code(r'''from utils import sha256_of
from genial_clf import data as D
raw = ROOT / cfg["paths"]["raw_corpus"]
h = sha256_of(raw); print("SHA-256:", h, "\ncoincide con config.yaml:", h == cfg["corpus_sha256"])
bruto = D.load_raw(raw, cfg["paths"]["raw_sheet"], cfg["corpus_sha256"])
print("filas × columnas:", bruto.shape); print("columnas:", list(bruto.columns))'''))
C.append(md("## 2-4. Preprocesamiento\n\n`preprocess()` aplica en orden: normalización de `Tipo de norma`, correcciones expertas (etiquetas revisadas con el equipo GENIAL), construcción de `texto`, conteo de palabras, detección de idioma (biblioteca *lingua*, con el país como apoyo cuando el título es ambiguo) y `grupo_url` (normas que comparten URL oficial se tratan como una sola unidad al particionar)."))
C.append(code(r'''df = D.preprocess(bruto.copy(), cfg)
t = cfg["columns"]["target"]
print("columnas añadidas:", [c for c in df.columns if c not in bruto.columns])
print("\nDistribución de la variable objetivo:"); display(df[t].value_counts().to_frame("n").assign(pct=lambda x: (100 * x["n"] / len(df)).round(1)))
print("idioma del título:", df["idioma"].value_counts().to_dict())
print("grupos de URL con más de una norma:", int((df.groupby("grupo_url").size() > 1).sum()), "· normas implicadas:", int(df.groupby("grupo_url")["id_norma"].transform("size").gt(1).sum()))
print("longitud del texto (palabras): mediana %.0f · media %.1f · p10 %.0f · p90 %.0f" % (df["n_palabras"].median(), df["n_palabras"].mean(), df["n_palabras"].quantile(.1), df["n_palabras"].quantile(.9)))'''))
C.append(md("### Ejemplo de texto de entrada\n\nEl modelo solo ve `texto`. Se muestran títulos (documentos públicos) con las observaciones truncadas."))
C.append(code(r'''ej = df.sample(3, random_state=cfg["seed"])[["id_norma", "País", t, "idioma", "n_palabras", "texto"]].copy()
ej["texto"] = ej["texto"].str.slice(0, 160) + "…"; display(ej)'''))
C.append(md("## 5. Variables excluidas por fuga de información\n\n`Investigador` (quién codificó) tiene V de Cramér 0.99 con `País`: cada investigador cubre países concretos, de modo que la columna equivale a conocer el país y, además, es un dato personal. `Tipo de norma` y `Fecha de Publicación` no describen el contenido. Ninguna entra al modelo; `Investigador` además se anonimiza (Codificador A-J) en cualquier artefacto del repositorio."))
C.append(code(r'''print("excluidas del modelo:", cfg["columns"]["leakage_excluded"])
probe = D.leakage_probe(df, cfg)
print(f"fuga léxica (la etiqueta aparece literalmente en el texto): {len(probe)} normas de {len(df)} ({100*len(probe)/len(df):.1f} %) → se conservan; es vocabulario legítimo de la norma, no una columna filtrada")
display(probe.assign(texto=probe["texto"].str.slice(0, 110) + "…").head(5))'''))
C.append(md("## 6. Particiones\n\nSi existe el instrumento de codificación ciega del investigador (`data/raw/codificacion_investigador_A.xlsx`), la **prueba ciega son exactamente sus normas** (119 emparejadas por título/URL + sus duplicados → 125) y las 23 normas nuevas del Caribe que no están en la versión congelada se guardan aparte. El resto se divide en pool *few-shot* (3 por clase, entre 25 y 90 palabras) y entrenamiento. Entrenamiento y prueba no comparten ningún `grupo_url`."))
C.append(code(r'''from data_processing import make_partitions, load_splits
splits, origen = make_partitions(df, cfg); print(origen)
for k in ("train", "fewshot", "test"):
    print(f"{k:8s} {len(splits[k]):4d}", splits[k][t].value_counts().to_dict() if k != "test" else "(etiquetas ocultas en test_labels_hidden.csv)")
assert not set(splits["train"]["grupo_url"]) & set(splits["test"]["grupo_url"]), "fuga por URL entre entrenamiento y prueba"
sp = load_splits(cfg); print("\narchivos escritos en", cfg["paths"]["splits_dir"], "→", {k: list(v.columns)[:6] for k, v in sp.items()})
print("idioma en prueba:", sp["test"]["idioma"].value_counts().to_dict())'''))
C.append(md("## Comprobaciones de calidad\n\n`results/metrics/reporte_calidad.md` recoge faltantes, duplicados y distribución final; es el informe que acompaña a `data/README.md`."))
C.append(code(r'''print((RES / "reporte_calidad.md").read_text(encoding="utf-8")[:2500])'''))
write("02_preprocesamiento.ipynb", C)

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# 03 · modelado
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
C = []
C.append(md(r"""# 03 · Modelado: líneas base y clasificador TF-IDF + Regresión Logística

Objetivo: construir el modelo base (`model_v1`) y medir su desempeño **sin tocar el conjunto ciego**. Toda la validación es cruzada, estratificada por dimensión y **por grupo de URL** (`StratifiedGroupKFold`): las normas que comparten URL caen en el mismo pliegue, para que el modelo nunca sea evaluado sobre un duplicado de algo que vio en entrenamiento.

Contenido: (1) pisos de referencia; (2) el pipeline y sus decisiones; (3) validación cruzada por grupo con intervalos; (4) desempeño por clase y matriz de confusión; (5) ablaciones (qué aporta cada decisión) y comparación con otras técnicas; (6) interpretabilidad; (7) entrenamiento y guardado de `model_v1`."""))
C.append(code(HEADER))
C.append(code(r'''from data_processing import load_splits
from model import build_model, model_params
from genial_clf import baselines, classic
from genial_clf.optimize import cv_score, BASELINE
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
sp = load_splits(cfg); train = sp["train"]; t = cfg["columns"]["target"]
X, y, g = train["texto"].to_numpy(), train[t].to_numpy(), train["grupo_url"].to_numpy()
print("entrenamiento:", len(train), "normas ·", train[t].value_counts().to_dict())'''))
C.append(md("## 1. Pisos de referencia\n\nAntes de cualquier modelo: ¿qué exactitud da adivinar siempre la clase mayoritaria y qué da usar solo un metadato (el tipo de norma: ley, resolución, guía…) sin leer el texto? Todo lo que el clasificador no supere con holgura no justifica su coste."))
C.append(code(r'''piso = baselines.arithmetic_floor(train[t].value_counts().to_dict()); print("clase mayoritaria:", {k: round(v, 3) for k, v in piso.items()})
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg["seed"])
acc_m, f1_m = [], []
for tr, va in cv.split(X, y, g):
    mb = baselines.MetadataBaseline().fit(train.iloc[tr], t); p = mb.predict(train.iloc[va])
    acc_m.append(accuracy_score(y[va], p)); f1_m.append(f1_score(y[va], p, average="macro"))
print("metadatos (sin texto): exactitud %.3f ± %.3f · F1 macro %.3f ± %.3f" % (np.mean(acc_m), np.std(acc_m), np.mean(f1_m), np.std(f1_m)))'''))
C.append(md("## 2. El pipeline\n\n`build_model()` (en `src/model.py`) devuelve `TfidfVectorizer → LogisticRegression`. Decisiones y motivo:\n\n| Decisión | Valor | Por qué |\n|---|---|---|\n| n-gramas | 1-2 | los bigramas capturan «protección datos», «inteligencia artificial», «uso ético» |\n| `max_features` | 5 000 | más *features* que ejemplos (673) → se limita el vocabulario a lo frecuente |\n| `sublinear_tf` | sí | amortigua títulos con términos repetidos |\n| `class_weight` | `balanced` | desbalance 4.7:1; sin él, Técnico y Administrativo casi no se predicen |\n| `C` | 1.0 | valor por defecto, revisado en el cuaderno 04 |\n| solver | `lbfgs`, multinomial | problema pequeño, convexo |"))
C.append(code(r'''pipe = build_model(**BASELINE, seed=cfg["seed"]); print(pipe); print(model_params(pipe))'''))
C.append(md("## 3. Validación cruzada por grupo\n\nCinco pliegues; se reporta F1 macro de validación y de entrenamiento por pliegue. El *gap* entre ambos anticipa el diagnóstico de sobreajuste de la Semana 3 (`anexos/overfitting_analysis.ipynb`)."))
C.append(code(r'''r, folds = cv_score(build_model, BASELINE, X, y, g, seed=cfg["seed"], return_folds=True)
print("F1 macro validación por pliegue:", [round(f, 3) for f in folds])
print("media %.3f ± %.3f · entrenamiento %.3f · gap %.2f" % (r["val_f1"], r["val_std"], r["train_f1"], r["gap"]))
pred_cv = classic.cross_validate(train, cfg)          # predicciones fuera de muestra, top-2 y confianza
print("exactitud %.3f · F1 macro %.3f · top-2 %.3f" % (accuracy_score(pred_cv["y_true"], pred_cv["pred"]), f1_score(pred_cv["y_true"], pred_cv["pred"], average="macro"), ((pred_cv["pred"] == pred_cv["y_true"]) | (pred_cv["pred_2"] == pred_cv["y_true"])).mean()))'''))
C.append(md("## 4. Desempeño por clase\n\nLas clases pequeñas (Técnico 63 normas en el corpus, Administrativo 120) son las difíciles: comparten vocabulario con Pedagógico y Ética. Protección de Datos, en cambio, tiene léxico propio y se reconoce bien pese a ser minoritaria."))
C.append(code(r'''print(classification_report(pred_cv["y_true"], pred_cv["pred"], labels=CLASSES, digits=3))
cm = confusion_matrix(pred_cv["y_true"], pred_cv["pred"], labels=CLASSES)
fig, ax = plt.subplots(figsize=(6.2, 5)); im = ax.imshow(cm, cmap="Reds")
ax.set_xticks(range(5)); ax.set_xticklabels([c[:12] for c in CLASSES], rotation=35, ha="right"); ax.set_yticks(range(5)); ax.set_yticklabels(CLASSES)
for i in range(5):
    for j in range(5):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("predicción"); ax.set_ylabel("etiqueta del codificador"); ax.set_title("Matriz de confusión · CV por grupo (train, n = %d)" % len(train), loc="left", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "18_confusion_cv_train.png", dpi=300, facecolor="white"); plt.show()'''))
C.append(md("## 5. Ablaciones y otras técnicas\n\nResultados del experimento del Sprint 3 sobre el corpus completo (validación cruzada por grupo, n = 813; `anexos/experimento_sprint3.ipynb`): qué cambia al quitar la ponderación de clases, los bigramas o reducir el vocabulario, y cómo se comparan SVM lineal y las líneas base. La conclusión que fija el modelo base: **la ponderación de clases es la decisión que más importa** (F1 macro 0.587 vs 0.475 sin ella); el resto mueve el F1 en el margen del ruido entre pliegues."))
C.append(code(r'''display(pd.read_csv(RES / "ablaciones_cv.csv").round(3))
cp = pd.read_csv(RES / "comparacion_pareada.csv"); display(cp[["rama", "n", "exactitud", "IC95 exactitud", "F1 macro", "IC95 F1 macro", "top-2", "κ vs etiqueta original"]].round(3))
print("(comparación pareada sobre el conjunto ciego, Sprint 3; McNemar en results/metrics/mcnemar.csv)")'''))
C.append(md("## 6. Interpretabilidad: términos con mayor peso por dimensión\n\nCoeficientes de la regresión logística multinomial sobre el vocabulario TF-IDF. Sirven para comprobar que el modelo aprende léxico de la dimensión y no artefactos (nombres de países, de instituciones o fechas)."))
C.append(code(r'''pipe.fit(X, y)
tt = classic.top_terms(pipe, k=12) if "tfidf" in pipe.named_steps else None
if tt is None:
    vocab = np.array(pipe.named_steps[list(pipe.named_steps)[0]].get_feature_names_out()); coef = pipe.named_steps[list(pipe.named_steps)[-1]].coef_
    tt = pd.DataFrame({c: vocab[np.argsort(-coef[i])[:12]] for i, c in enumerate(pipe.classes_)})
display(tt)'''))
C.append(md("## 7. Entrenar y guardar `model_v1`\n\n`src/train.py` entrena con todo `train.csv`, registra la validación cruzada en los metadatos y guarda `models/model_v1.pkl` + `models/model_v1.json`. El modelo optimizado (`best_model`) se entrena en el cuaderno 04 y ambos se evalúan a ciegas en el 05."))
C.append(code(r'''import train as T
meta = T.train_and_save("model_v1", BASELINE, train, cfg, "configuración base (cuaderno 03)")
print(json.dumps(meta, ensure_ascii=False, indent=1, default=float))'''))
write("03_modelado.ipynb", C)

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# 04 · optimización
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
C = []
C.append(md(r"""# 04 · Optimización de hiperparámetros y análisis de sensibilidad

Workshop S5. Pregunta: **¿cuánto mejora el modelo base si se buscan sistemáticamente los hiperparámetros, y a cuáles es realmente sensible?** La respuesta condiciona la elección de `best_model`.

Método (`src/genial_clf/optimize.py`, ejecutado por `scripts/optimizar.py`):

1. **Búsqueda bayesiana** con Optuna (muestreador TPE, 15 ensayos iniciales aleatorios, 60 en total). Función objetivo: F1 macro en validación cruzada estratificada por grupo de URL (5 pliegues) sobre el conjunto de entrenamiento. El primer ensayo es la configuración base, para que la historia parta de ella.
2. **Ranking de importancia** de hiperparámetros (fANOVA sobre los ensayos).
3. **Dependencia parcial 1-D**: barrido de cada hiperparámetro con los demás fijos en el óptimo.
4. **Interacción** `C × max_features` (dependencia parcial 2-D).
5. **Tabla de sensibilidad**: rango explorado, valor óptimo y variación del F1 (máx − mín) en el barrido.
6. **Antes / después** con la configuración base, el óptimo de Optuna y la **regla de una desviación estándar** (entre los ensayos a ≤ 1 DE del máximo, el de menor *gap* entrenamiento − validación), que es la que se adopta.

Espacio de búsqueda:

| Hiperparámetro | Rango | Tipo |
|---|---|---|
| `C` (inverso de la regularización L2) | 0.01 – 30 | continuo, escala log |
| `max_features` | {1 000, 2 000, 3 000, 5 000, 8 000, 12 000, 20 000} | categórico |
| `ngram_max` | {1, 2} | categórico |
| `min_df` | {1, 2, 3} | categórico |
| `class_weight` | {balanced, None} | categórico |
| `sublinear_tf` | {True, False} | categórico |

El conjunto ciego **no interviene** en ningún paso."""))
C.append(code(HEADER))
C.append(md("## 1. Ejecutar o reutilizar la búsqueda\n\nLa búsqueda completa (60 ensayos + barridos ≈ 200 ajustes de modelo) tarda unos seis minutos. Si `results/metrics/optimizacion/resumen.json` existe se reutiliza; cambie `EJECUTAR = True` para repetirla."))
C.append(code(r'''EJECUTAR = False; N_TRIALS = 60
OPT = RES / "optimizacion"
if EJECUTAR or not (OPT / "resumen.json").exists():
    sys.argv = ["optimizar"]; import importlib.util
    spec = importlib.util.spec_from_file_location("optimizar", ROOT / "scripts/optimizar.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    resumen = m.main(N_TRIALS, cfg["seed"])
else:
    resumen = json.loads((OPT / "resumen.json").read_text(encoding="utf-8")); print("resultados reutilizados de", OPT)
trials = pd.read_csv(OPT / "trials.csv"); print(resumen["n_trials"], "ensayos ·", resumen["segundos"], "s")'''))
C.append(md("## 2. Historia de la búsqueda\n\nCada punto es un ensayo; la línea, el mejor F1 hasta ese momento; la discontinua, la configuración base. Lo primero que se observa es que **la nube es estrecha**: más de la mitad de los ensayos quedan en ±2 pp del valor base."))
C.append(code(r'''display(Image(filename=str(FIG / "13_optuna_historia.png"), width=820))
mejor = trials.sort_values("val_f1", ascending=False).head(8)[["trial", "val_f1", "train_f1", "gap", "C", "max_features", "ngram_max", "min_df", "class_weight", "sublinear_tf"]]
display(mejor.round(3)); print("base (ensayo 0): val F1 %.3f · gap %.2f" % (trials.loc[0, "val_f1"], trials.loc[0, "gap"]))'''))
C.append(md("## 3. Ranking de importancia (fANOVA)\n\nQué fracción de la varianza del F1 entre ensayos explica cada hiperparámetro. **`class_weight` domina** (≈ 0.77): es la decisión que separa los ensayos buenos de los malos. `C` queda segundo; el resto es casi irrelevante dentro del rango explorado."))
C.append(code(r'''display(pd.read_csv(OPT / "importancia.csv").round(3)); display(Image(filename=str(FIG / "14_importancia_hiperparametros.png"), width=700))'''))
C.append(md("## 4. Dependencia parcial y tabla de sensibilidad\n\nBarrido de cada hiperparámetro con los demás en el óptimo. La banda es ± 1 DE entre pliegues. Lectura: la curva de validación es **plana** en casi todos los paneles (la variación máx − mín no supera 3 pp), mientras la de entrenamiento sube con `C` y con el vocabulario: el modelo puede memorizar más, pero no generaliza más."))
C.append(code(r'''display(Image(filename=str(FIG / "15_dependencia_parcial.png"), width=1000))
sens = pd.read_csv(OPT / "sensibilidad.csv"); display(sens.round(3))'''))
C.append(md("## 5. Interacción `C × max_features`\n\nMapa de calor del F1 de validación (izquierda) y del *gap* (derecha). El F1 es casi constante sobre toda la rejilla; el *gap*, en cambio, crece con ambos ejes. De ahí la regla de decisión: **a igual F1, la esquina de menor `C` y menor vocabulario**."))
C.append(code(r'''display(Image(filename=str(FIG / "16_interaccion_C_max_features.png"), width=1000))
display(pd.read_csv(OPT / "interacciones.csv").pivot(index="C", columns="max_features", values="val_f1").round(3))'''))
C.append(md("## 6. Antes y después: cómo se eligió `best_model`\n\nTres configuraciones evaluadas con la misma validación cruzada:\n\n* **Base** (`C = 1`, 5 000 *features*): la del documento técnico.\n* **Óptimo de Optuna**: el ensayo con mayor F1 de validación. Gana décimas, con más sobreajuste.\n* **Regla 1-DE** (la adoptada): entre los ensayos cuyo F1 está a menos de una desviación estándar del máximo, el de menor *gap*. Mismo F1 dentro del ruido, 12 pp menos de sobreajuste y menor varianza entre pliegues.\n\nConclusión honesta del *workshop*: **la optimización no mejora el F1 de validación de forma apreciable** (el error restante es de etiqueta y de tamaño de muestra, como mostró la Semana 3), pero sí permite elegir un modelo igual de bueno y más estable. Esa es la configuración que se guarda como `best_model`."))
C.append(code(r'''comp = pd.read_csv(OPT / "comparacion_antes_despues.csv"); display(comp.round(3)); display(Image(filename=str(FIG / "17_antes_despues_optimizacion.png"), width=820))
print("regla 1-DE →", resumen["regla_1se"])'''))
C.append(md("## 7. Entrenar y guardar `best_model`\n\n`src/train.py` lee `regla_1se` de `resumen.json`, entrena con todo `train.csv` y guarda `models/best_model.pkl` + metadatos. Es el modelo que sirve la aplicación."))
C.append(code(r'''import train as T
from data_processing import load_splits
params, origen = T.best_params(); meta = T.train_and_save("best_model", params, load_splits(cfg)["train"], cfg, origen)
print(json.dumps(meta, ensure_ascii=False, indent=1, default=float))'''))
write("04_optimizacion.ipynb", C)

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# 05 · evaluación
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
C = []
C.append(md(r"""# 05 · Evaluación final: prueba ciega, enrutamiento, equidad y rama LLM

Único cuaderno que lee las etiquetas del conjunto ciego (`data/splits/test_labels_hidden.csv`). Evalúa `model_v1` y `best_model` sobre las 125 normas codificadas a ciegas por el investigador del proyecto GENIAL, con intervalos de confianza y pruebas pareadas; después simula la **operación** (regla de gobernanza con umbral de confianza y revisión humana obligatoria), resume la **auditoría de equidad por jurisdicción** y deja lista la **rama LLM** para quien disponga de clave de API.

Lo que se reporta y por qué:

| Métrica | Para qué |
|---|---|
| Exactitud + IC Wilson 95 % | lectura directa; el IC recuerda que n = 125 |
| F1 macro + IC *bootstrap* (2 000) | métrica principal: trata igual a las cinco clases |
| Top-2 | mide si la segunda opción rescata los errores (multi-dimensionalidad) |
| κ de Cohen vs codificador | compara con el acuerdo humano, no con el azar |
| McNemar | diferencia **pareada** entre modelos sobre las mismas normas |"""))
C.append(code(HEADER))
C.append(md("## 1. Los dos modelos sobre el conjunto ciego"))
C.append(code(r'''import evaluate as EV
tabla, detalle = EV.evaluate_models(["model_v1", "best_model"], cfg)
display(tabla.round(3))
mc = detalle["mcnemar_entre_modelos"]; print("McNemar model_v1 vs best_model: p = %.3f → %s" % (mc["p_value"], "diferencia no significativa" if mc["p_value"] > 0.05 else "diferencia significativa"))'''))
C.append(md("Lectura: ambos modelos están claramente por encima del piso (clase mayoritaria 0.36, metadatos 0.54, McNemar p < 0.05 en ambos casos) y la diferencia entre ellos no es significativa. `best_model` gana en F1 macro por una mejor cobertura de las clases pequeñas (Técnico, Administrativo), a costa de algo de exactitud global; es el compromiso que buscaba la regla 1-DE."))
C.append(md("## 2. Matriz de confusión y desempeño por idioma (`best_model`)"))
C.append(code(r'''from sklearn.metrics import confusion_matrix
d = detalle["best_model"]; cm = pd.DataFrame(d["confusion"]); display(cm)
por_idioma = pd.DataFrame(d["por_idioma"]); display(por_idioma.round(3))
print("Las normas en inglés y portugués (minoría en entrenamiento) se clasifican peor: por eso la regla de gobernanza las envía siempre a revisión humana.")'''))
C.append(md("## 3. ¿Qué dicen los errores?\n\nDel Sprint 3 (mismo conjunto, modelo base): de 39 errores, 23 se rescatan con la segunda dimensión y 35 de 37 revisados eran normas **multidimensionales** (el codificador también dudó: exactitud 0.76 cuando su confianza fue *Alta* frente a 0.41 cuando fue *Media*). El error del modelo es en gran parte el error del instrumento: una norma con una sola etiqueta cuando trata dos temas."))
C.append(code(r'''s = json.loads((RES / "sintesis_resultados.json").read_text(encoding="utf-8"))
print(json.dumps(s.get("instrumento_investigador", {}), ensure_ascii=False, indent=1)[:1500])'''))
C.append(md("## 4. Operación: enrutamiento con revisión humana\n\nRegla de gobernanza (`config.yaml → routing`): va a revisión humana toda norma con probabilidad máxima por debajo del **umbral de confianza del modelo**, de país con menos de 20 normas en el corpus o en inglés, francés o portugués. El umbral se calibra en `src/train.py` para cada modelo sobre predicciones fuera de muestra del entrenamiento (menor umbral con ≥ 85 % de exactitud en el tramo automático y ≥ 10 % de normas en él): 0.50 para `model_v1` y 0.29 para `best_model`, cuya regularización aplana las probabilidades. Se simula sobre el conjunto ciego con `best_model` y se calcula la carga humana residual (10 min por norma, supuesto declarado)."))
C.append(code(r'''from model import Predictor, MODELS_DIR
from data_processing import load_splits
from genial_clf import imn
sp = load_splits(cfg, with_hidden_labels=True); test = sp["test"].merge(sp["test_labels"], on="id_norma")
P = Predictor.from_files(MODELS_DIR / "best_model.pkl")
proba = P.pipe.predict_proba(test["texto"]); cls = np.asarray(P.pipe.classes_); order = np.argsort(-proba, axis=1)
preds = pd.DataFrame({"id_norma": test["id_norma"], "pred": cls[order[:, 0]], "pred_2": cls[order[:, 1]], "conf": proba.max(axis=1)})
corpus = pd.read_csv(cfg["paths"]["processed"])
cfg_m = {**cfg, "routing": {**cfg["routing"], "confidence_threshold": P.threshold}}; print("umbral de confianza de best_model:", P.threshold, "·", P.metadata["calibracion_umbral"]["criterio"])
routed = imn.route(preds, corpus[["id_norma", "País", "idioma"]], cfg_m).merge(test[["id_norma", cfg["columns"]["target"]]], on="id_norma")
auto = routed[~routed["a_revision"]]; ok_auto = (auto["pred"] == auto[cfg["columns"]["target"]]).mean()
print("enrutado a revisión: %.1f %% · exactitud del tramo automático: %.3f (n = %d)" % (100 * routed["a_revision"].mean(), ok_auto, len(auto)))
print("motivos:", {k: int(routed[k].sum()) for k in ["baja_confianza", "pais_subrepresentado", "idioma_minoritario"]})
print(imn.residual_workload(routed))
curva = imn.routing_curve(preds, test[cfg["columns"]["target"]]); display(curva.round(3).T)'''))
C.append(md("Con n = 125 y una prueba ciega que sobrerrepresenta países pequeños e idiomas minoritarios (así fue diseñado el instrumento), la regla envía a revisión la mayor parte de las normas: el modelo opera como **codificación asistida** (propone, con top-2 y términos), no como sustituto. Sobre el corpus completo (n = 813, Sprint 3, `model_v1`) el tramo automático alcanzó 88.9 % de exactitud y ahorró 21 de 136 horas."))
C.append(md("## 5. Auditoría de equidad por jurisdicción\n\nResumen de `scripts/fairness_grupos.py` (Semana 4): desempeño por idioma, tamaño del país, región, ámbito y codificador anonimizado, sobre las predicciones fuera de muestra de la validación cruzada por grupo (n = 813). Se usa la regla de los cuatro quintos como referencia orientativa."))
C.append(code(r'''fair = pd.read_csv(RES / "fairness/fairness_grupos.csv"); fr = json.loads((RES / "fairness/fairness_resumen.json").read_text(encoding="utf-8"))
display(fair[["dimensión", "grupo", "n", "exactitud", "IC95 inf", "IC95 sup", "razón vs referencia", "% a revisión humana (regla)", "exactitud tramo automático"]].round(3))
for k, v in fr.items():
    if isinstance(v, dict): print(f"{k:30s} peor: {v['grupo_peor']} ({v['exactitud_min']:.2f}) · mejor: {v['grupo_mejor']} ({v['exactitud_max']:.2f}) · razón mín {v['razón_min']:.2f} · 4/5: {'cumple' if v['cumple_4/5'] else 'NO cumple'}")
display(Image(filename=str(FIG / "12_fairness_grupos.png"), width=950))'''))
C.append(md("La brecha es real (es 0.66 · pt 0.47 · en 0.36) y no se corrige con el modelo: se **contiene** con la regla de gobernanza, que envía a revisión el 100 % de las normas en esos idiomas y de los países pequeños. El coste es que la salvaguarda se paga en horas humanas justo donde el corpus es más pobre; eso es lo que la sección de impacto social documenta como límite."))
C.append(md("## 6. Rama LLM *few-shot* (opcional)\n\nSi existe `results/metrics/pred_llm.csv` (generado por el paso 7 del cuaderno 00 o por `anexos/experimento_sprint3.ipynb`) se compara aquí con los modelos clásicos sobre las mismas normas. Si no, se indica cómo obtenerlo. La clave de API (Gemini o Anthropic) se busca en la variable de entorno, en `data/raw/gemini_key.txt` / `anthropic_key.txt` o en `.env`; nunca se escribe en el cuaderno."))
C.append(code(r'''from genial_clf import evaluate as E
p = RES / "pred_llm.csv"
if p.exists():
    dl = test[["id_norma", cfg["columns"]["target"]]].merge(pd.read_csv(p), on="id_norma"); dl["pred"] = dl["pred"].fillna("__sin respuesta__")
    rep = E.report(dl[cfg["columns"]["target"]], dl["pred"], CLASSES, dl["pred_2"], cfg["evaluation"]["bootstrap_iters"])
    print("LLM few-shot: exactitud %.3f %s · F1 macro %.3f %s · top-2 %.3f · κ %.3f" % (rep["accuracy"], rep["accuracy_ci_wilson"], rep["f1_macro"], rep["f1_macro_ci_bootstrap"], rep["top2_accuracy"], rep["kappa_vs_reference"]))
    print("McNemar LLM vs best_model:", E.mcnemar(dl[cfg["columns"]["target"]], dl["pred"], dl.merge(preds, on="id_norma", suffixes=("", "_rl"))["pred_rl"]))
else:
    print("Sin predicciones LLM. Para obtenerlas: cuaderno 00, paso 7 (FLAGS['EJECUTAR_LLM'] = True) con la clave de API en data/raw/anthropic_key.txt.")'''))
C.append(md("## 7. Conclusiones\n\n1. El clasificador supera con holgura las líneas base (p < 0.05) y alcanza κ ≈ 0.55 frente al codificador: acuerdo *moderado*, del orden del que cabe esperar entre dos codificadores humanos en una tarea donde más de un tercio de los errores son normas multidimensionales.\n2. La optimización no mueve el F1 de validación; permite elegir un modelo más regularizado y estable (`best_model`), cuyo F1 macro ciego es 0.631 [0.530, 0.724].\n3. El sistema se despliega como **codificación asistida con revisión humana obligatoria** para idiomas minoritarios y países pequeños; esa regla es la respuesta a la brecha de equidad detectada, no un adorno.\n4. Límites: n = 125 en prueba (intervalos anchos), una sola etiqueta por norma, κ humano-humano sobre `Dominio` pendiente de una segunda codificación."))
write("05_evaluacion.ipynb", C)
