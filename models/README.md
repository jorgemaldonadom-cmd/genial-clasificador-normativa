# Modelos entrenados

Los modelos se generan con `python src/train.py` (o con el cuaderno `notebooks/00_ejecutar_todo.ipynb`) a partir de
`data/splits/train.csv` (673 normas). **El conjunto ciego (125 normas) no interviene en el entrenamiento ni en la
selección de hiperparámetros**; sus etiquetas solo se leen en `src/evaluate.py` / cuaderno 05.

| Archivo | Configuración | Origen de los hiperparámetros | F1 macro CV por grupo (train) | gap train − val | F1 macro ciego (n = 125) |
|---|---|---|---:|---:|---:|
| `model_v1.pkl` | TF-IDF 1-2 gramas, 5 000 *features*, `min_df = 1`, `sublinear_tf`, RL `C = 1`, `class_weight = balanced` | configuración base del documento técnico | 0.547 ± 0.03 | 0.40 | 0.617 [0.505, 0.712] |
| `best_model.pkl` | TF-IDF 1-2 gramas, 2 000 *features*, `min_df = 1`, `sublinear_tf`, RL `C = 0.157`, `class_weight = balanced` | Optuna (60 ensayos, TPE) + regla de una desviación estándar con criterio de menor *gap* (`results/metrics/optimizacion/resumen.json`, clave `regla_1se`) | 0.540 ± 0.02 | 0.27 | 0.631 [0.530, 0.724] |

Cada `.pkl` (pipeline completo de scikit-learn serializado con `joblib`) lleva un archivo `.json` con la misma raíz
(metadatos: hiperparámetros, fecha, hash SHA-256 del corpus, métricas de validación cruzada, versión de scikit-learn).
`entrenamiento.json` resume la última ejecución de `src/train.py`.

`best_model.pkl` es el modelo que sirve la aplicación (`app/app.py`). La diferencia entre ambos modelos en el conjunto
ciego **no es significativa** (McNemar p = 0.75); se eligió el regularizado por su menor varianza entre pliegues y su
menor sobreajuste, no por su F1 puntual (véase `docs/optimizacion.md`).

Para cargarlo desde Python:

```python
from src.model import Predictor
p = Predictor.from_files("models/best_model.pkl")
p.predict_one("Lineamientos para el uso de IA generativa en la docencia", country="Ecuador", language="es")
```

Los modelos pesan menos de 1 MB cada uno y se versionan en el repositorio; se regeneran de forma determinista
(`seed = 2026`) con el corpus autorizado.
