# Cómo ejecutar `02_experimento.ipynb`

El cuaderno reproduce el experimento completo y regenera `results/` y `results/figuras/`. Tarda menos de dos minutos sin la rama LLM.

## 1. Requisitos

```bash
git clone https://github.com/jorgemaldonadom-cmd/genial-clasificador-normativa.git
cd genial-clasificador-normativa
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
```

Copiar el corpus autorizado a `data/raw/consolidado_limpio.xlsx` (el cuaderno verifica el hash SHA-256; si no coincide, se detiene).

## 2. Codificación del investigador de GENIAL (define el conjunto de prueba)

Hay dos archivos posibles en `data/raw/` (ninguno se versiona):

**a) `codificacion_investigador_A.xlsx`** — el instrumento de codificación ciega del Framework Dimensional GENIAL (hoja `Codificación`: 142 normas, dimensiones D1-D6 con etiquetas secundarias, `Confianza`, `Notas`). El cuaderno empareja sus normas con la versión congelada del corpus (título + país; desempate por URL y observaciones) y **esas normas son el conjunto de prueba a ciegas**; las que no están en la versión congelada se evalúan aparte como «normas nuevas». Este instrumento **no recodifica `Dominio`**, así que no permite κ humano-humano sobre la dimensión; sí permite cruzar los errores del modelo con la confianza humana y con la multidimensionalidad (sección 6.2).

**b) `coder2_labels.csv`** — recodificación ciega de `Dominio` sobre las mismas normas (UTF-8, separado por comas):

| columna | obligatoria | contenido |
|---|---|---|
| `id_norma` **o** `Título de la Norma` | sí | identificador de la norma (el `id_norma` de `data/processed/corpus_depurado.csv`) o el título literal |
| `Dominio_coder2` | sí | dimensión asignada por el segundo codificador (una de las cinco) |
| `Dominio_alt_coder2` | no | segunda dimensión que el codificador consideró defendible |

El cuaderno deja lista la plantilla con las normas del conjunto de prueba en `results/coder2_labels_plantilla.csv`: el codificador llena `Dominio_coder2` (≈ 20-30 minutos para 125 normas), se copia a `data/raw/coder2_labels.csv` y, al re-ejecutar, se calculan κ humano-humano, κ modelo-humano por rama y la matriz de confusión entre codificadores.

## 3. Rama LLM (ejecución local)

La clave de API **no se pega en el cuaderno** (el cuaderno se versiona). La sección 5 la busca, en este orden:

1. la variable de entorno `ANTHROPIC_API_KEY`;
2. el archivo `data/raw/anthropic_key.txt`, con la clave como única línea (opción recomendada sin terminal: crear el archivo con TextEdit en formato texto plano);
3. el archivo `.env` en la raíz del repositorio con la línea `ANTHROPIC_API_KEY=sk-ant-...`.

Los tres lugares están fuera del control de versiones (`.gitignore`). Con la clave en su sitio, abrir `notebooks/02_experimento.ipynb` en Jupyter o VS Code y ejecutar *Kernel → Restart & Run All*; o desde terminal:

```bash
export ANTHROPIC_API_KEY="..."        # opcional si ya existe data/raw/anthropic_key.txt
jupyter nbconvert --to notebook --execute notebooks/02_experimento.ipynb --inplace --ExecutePreprocessor.timeout=1800
```

La sección 5 detecta la clave, ejecuta tres pasadas a temperatura 0 con `prompts/clasificador_v0.1.md`, guarda `results/pred_llm.csv`, `results/pred_llm_runs.csv` y `results/pred_llm_runs.csv.meta.json` (modelo y versión exacta), y la sección 6 incorpora la rama a la comparación pareada y a McNemar. Costo estimado: < USD 2 por ejecución completa del conjunto de prueba (≈ 130 normas × 3 pasadas). Modelo y proveedor se cambian en `config.yaml` (`llm.model`, `llm.provider`).

Si ya existe `results/pred_llm.csv` de una ejecución anterior y no hay clave en el entorno, el cuaderno reutiliza ese archivo.

## 4. Salidas

| archivo | contenido |
|---|---|
| `results/sintesis_resultados.json` | todas las cifras citadas en informe, presentación y guion |
| `results/ablaciones_cv.csv` | seis variantes de la rama clásica en validación cruzada |
| `results/comparacion_pareada.csv` · `mcnemar.csv` | ramas sobre el conjunto ciego con IC95 y pruebas pareadas |
| `results/fiabilidad.csv` · `confusion_humano_humano.csv` | solo con `coder2_labels.csv` |
| `results/dominio_x_D4_finalidad.csv` · `pred_normas_nuevas.csv` · `coder2_labels_plantilla.csv` | sección 6.2 (instrumento del investigador) |
| `results/hoja_tipificacion_errores_clasico.csv` | hoja de trabajo para tipificar cada error (multi-etiqueta) |
| `results/curva_enrutamiento.csv` · `imn_propuesta.csv` | operación: umbral, carga residual, IMN por país con margen |
| `results/figuras/10_cv_confusion_f1.png` · `11_curva_enrutamiento.png` | figuras del informe |
| `notebooks/02_experimento.html` | exportación estática del cuaderno ejecutado |

`results/` está en `.gitignore` salvo los agregados sin texto de normas (síntesis, ablaciones, comparación, curva, IMN, figuras), que sí se versionan.

## 5. Regenerar el cuaderno desde su generador

El cuaderno se construye con `notebooks/make_results_notebook.py` (para que texto y código vivan en un solo archivo revisable):

```bash
python notebooks/make_results_notebook.py
jupyter nbconvert --to notebook --execute notebooks/02_experimento.ipynb --inplace
jupyter nbconvert --to html notebooks/02_experimento.ipynb
```
