# Arquitectura del sistema

## 1. Vista general

```
                 ┌──────────────────────────────┐
  data/raw/      │  src/data_processing.py       │   data/processed/corpus_depurado.csv
  consolidado ──▶│  hash · limpieza · idioma ·   │──▶ data/splits/{train, fewshot_examples,
  _limpio.xlsx   │  duplicados · particiones     │        test_blind, test_labels_hidden}.csv
  (no versionado)└──────────────────────────────┘
                                │ train.csv
                                ▼
   ┌──────────────────────────────────────────────┐    ┌────────────────────────────────┐
   │  src/model.py  build_model()                 │    │  scripts/optimizar.py           │
   │  TfidfVectorizer(1-2 g) → LogisticRegression │◀───│  Optuna TPE + sensibilidad      │
   │  src/train.py  CV por grupo + calibración    │    │  → results/metrics/optimizacion │
   │  del umbral → models/{model_v1, best_model}  │    └────────────────────────────────┘
   └──────────────────────────────────────────────┘
                                │ best_model.pkl + .json
                ┌───────────────┴────────────────┐
                ▼                                ▼
   ┌────────────────────────┐       ┌─────────────────────────────────┐
   │  src/evaluate.py        │       │  app/app.py (Streamlit)          │
   │  prueba ciega, IC,      │       │  Predictor.predict_one / _many   │
   │  McNemar, por idioma    │       │  regla de gobernanza, Plotly,    │
   │  → results/metrics/     │       │  lote CSV, historial, es/en      │
   └────────────────────────┘       └─────────────────────────────────┘
                                                 │
                                                 ▼
                                   codificador humano (revisión) ──▶ IMN (GENIAL)
```

## 2. Componentes

| Capa | Módulo | Responsabilidad |
|---|---|---|
| Configuración | `config.yaml`, `src/genial_clf/config.py` | semilla, rutas, hash del corpus, columnas, clases, hiperparámetros base, regla de gobernanza |
| Datos | `src/data_processing.py` → `genial_clf.data`, `.split`, `.coder` | carga verificada, preprocesamiento, idioma, duplicados, particiones (por instrumento o estratificadas) |
| Modelo | `src/model.py` | `build_model`, `save_model`/`load_model` (joblib + JSON de metadatos), `Predictor` (predicción, top-2, términos influyentes, regla de gobernanza) |
| Entrenamiento | `src/train.py` | validación cruzada por grupo, **calibración del umbral de confianza por modelo**, persistencia en `models/` |
| Optimización | `src/genial_clf/optimize.py`, `scripts/optimizar.py` | estudio Optuna, importancia fANOVA, dependencia parcial, interacciones, sensibilidad, regla 1-DE |
| Evaluación | `src/evaluate.py` → `genial_clf.evaluate`, `.baselines`, `.reliability`, `.imn` | métricas con IC, McNemar, por idioma, enrutamiento, carga humana residual |
| Equidad | `scripts/fairness_grupos.py` | desempeño por idioma, país, región, ámbito y codificador anonimizado |
| LLM | `src/genial_clf/llm.py`, `prompts/clasificador_v0.1.md` | rama *few-shot* (Anthropic o Gemini) con clave fuera del repositorio, guardado incremental y reanudación |
| Aplicación | `app/app.py` | interfaz Streamlit (una norma, lote CSV, acerca del modelo, historial) |
| Pruebas | `tests/` | 14 pruebas (datos, modelo, aplicación con `streamlit.testing`) |
| Utilidades | `src/utils.py` | logging, semillas, validación de texto, hash |

## 3. Decisiones de diseño

1. **Modelo lineal sobre TF-IDF, no una red ni un transformador ajustado.** 673 ejemplos de entrenamiento y textos de 44 palabras de mediana: en la comparación de ocho técnicas (Semana 2) ningún método no lineal superó a la regresión logística ponderada dentro del ruido entre pliegues, y el modelo lineal es interpretable (coeficientes por término), rápido (< 2 s de entrenamiento) y pesa 160 KB.
2. **Validación cruzada por grupo de URL.** Las normas que comparten URL se mantienen en el mismo pliegue; de lo contrario el modelo se evaluaría sobre duplicados de su entrenamiento.
3. **El conjunto ciego es exógeno.** Lo definió el instrumento de un investigador del consorcio, no un muestreo nuestro; sus etiquetas se leen en un único módulo.
4. **Umbral de confianza calibrado por modelo.** La regularización aplana las probabilidades: un umbral fijo (0.50) deja el modelo optimizado sin tramo automático. `src/train.py` calcula, sobre predicciones fuera de muestra, el menor umbral con ≥ 85 % de exactitud automática y ≥ 10 % de cobertura, y lo guarda en los metadatos del modelo; `Predictor` lo lee de ahí (0.50 para `model_v1`, 0.29 para `best_model`).
5. **Regla de gobernanza en configuración, no en código.** Umbral, mínimo de normas por país e idiomas de revisión obligatoria viven en `config.yaml` y se muestran en la aplicación.
6. **Separación datos / código / resultados.** Nada personal ni no autorizado se versiona: el corpus y el instrumento quedan en `data/raw/` (ignorado); los resultados que contienen texto de normas (`pred_*.csv`, predicciones de validación cruzada) también.

## 4. Flujo de una predicción en la aplicación

1. El usuario introduce título y observaciones (o elige un ejemplo), país e idioma; `validate_norm_text` rechaza textos de menos de 3 palabras o más de 5 000 caracteres.
2. `Predictor.predict_one` vectoriza, obtiene probabilidades, primera y segunda dimensión y términos que más pesan (coeficiente × TF-IDF).
3. Se aplica la regla de gobernanza: probabilidad < umbral del modelo, país con < 20 normas en el corpus o idioma en {en, fr, pt} → **revisión humana**, con los motivos.
4. La interfaz muestra la dimensión propuesta, la segunda candidata, el gráfico de probabilidades, el aviso de revisión y los términos; la predicción se añade al historial de sesión y puede exportarse en lote como CSV.

## 5. Despliegue

* Local: `streamlit run app/app.py` (o la última celda de `notebooks/00_ejecutar_todo.ipynb`).
* Nube: Streamlit Community Cloud apuntando a `app/app.py` con `app/requirements.txt`; los modelos (`models/*.pkl`) están versionados, de modo que el despliegue no necesita el corpus.
* Pruebas en integración continua: `.github/workflows/tests.yml` ejecuta `pytest` sobre la muestra pública y el modelo versionado.

## 6. Reproducibilidad

Semilla global 2026; hash del corpus verificado; versiones fijadas en `requirements.txt`; metadatos JSON junto a cada modelo (hiperparámetros, métricas de validación cruzada, umbral calibrado, versión de scikit-learn, fecha); cuadernos ejecutados con sus salidas; `notebooks/00_ejecutar_todo.ipynb` regenera todo en menos de diez minutos (sin Optuna, dos).
