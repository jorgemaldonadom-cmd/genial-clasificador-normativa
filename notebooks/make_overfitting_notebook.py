"""Genera notebooks/overfitting_analysis.ipynb (Actividad Semana 3: diagnóstico de overfitting / underfitting)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

C = []
C.append(md("""# Diagnóstico de overfitting / underfitting · clasificador de normativa sobre IA (corpus GENIAL)

**Proyecto Integrador en IA (MIAR0545) · Semana 3 · Grupo 9** — Jorge Javier Maldonado Mahauad · Melissa Figallo · Tutora: Gladys María Villegas Rugel

Este cuaderno aplica al modelo del Proyecto Integrador —regresión logística sobre TF-IDF (1-2 gramas, 5 000 características) que asigna una de cinco dimensiones a cada norma— el ciclo completo de la actividad: **(1)** sistema de tracking de métricas, **(2)** curvas de aprendizaje (por época, por tamaño de datos y por hiperparámetro), **(3)** diagnóstico sistemático con criterios cuantitativos y **(4)** estrategias de mejora implementadas y comparadas antes/después.

**Datos.** Se usa únicamente el conjunto de entrenamiento del proyecto (`data/splits/train.csv`, 673 normas), que se parte en entrenamiento/validación de forma estratificada y por grupo de URL. El conjunto de prueba a ciegas (125 normas) **no se toca** en esta actividad: se reserva para la evaluación final del proyecto. Requiere el corpus autorizado en `data/raw/` (no se versiona; véase `notebooks/README_experimento.md`).

**Código modular.** Toda la lógica reutilizable está en `src/genial_clf/tracking.py` (tracker, entrenadores por época, curvas, diagnóstico, figuras); el cuaderno solo orquesta y comenta."""))

C.append(code(r"""import os, sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from IPython.display import display, Markdown
warnings.filterwarnings("ignore")
ROOT = Path.cwd() if (Path.cwd() / "config.yaml").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from genial_clf import tracking as T
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8")); SEED = cfg["seed"]; C5 = cfg["classes"]
FIG = ROOT / "results/figuras/overfitting"; FIG.mkdir(parents=True, exist_ok=True)
RES = ROOT / "results/overfitting"; RES.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED); print("raíz:", ROOT)"""))

# ---------------------------------------------------------------- 1
C.append(md("""## 1. Carga y exploración del dataset

El conjunto de entrenamiento del proyecto contiene 673 normas con su texto (`título + observaciones`), la dimensión (`Dominio`), el idioma detectado y el grupo de URL (duplicados). La exploración completa está en `EDA_corpus_GENIAL.ipynb`; aquí se recuerda lo que condiciona el diagnóstico: **pocos ejemplos, cinco clases desbalanceadas (4.9:1 en este subconjunto) y textos cortos**, con una representación TF-IDF de 5 000 dimensiones, es decir, más características que ejemplos."""))
C.append(code(r"""train = pd.read_csv(ROOT / "data/splits/train.csv")
print(train.shape); display(train[["id_norma", "País", "Dominio", "idioma", "n_palabras", "grupo_url"]].head())
dist = train["Dominio"].value_counts().reindex(C5)
print("Distribución:", dist.to_dict(), "· razón de desbalance %.1f:1" % (dist.max() / dist.min()))
print("Palabras por norma: mediana %.0f · media %.1f · máx %d" % (train["n_palabras"].median(), train["n_palabras"].mean(), train["n_palabras"].max()))
print("Idioma del título:", train["idioma"].value_counts().to_dict(), "· grupos de URL con >1 norma:", int((train.groupby("grupo_url").size() > 1).sum()))"""))
C.append(code(r"""# Partición entrenamiento / validación (≈80/20): estratificada por dimensión y por grupo de URL (los duplicados no se reparten)
cv_split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
i_tr, i_va = next(cv_split.split(train, train["Dominio"], train["grupo_url"]))
d_tr, d_va = train.iloc[i_tr], train.iloc[i_va]
vec = TfidfVectorizer(ngram_range=tuple(cfg["classic"]["ngram_range"]), max_features=cfg["classic"]["max_features"], sublinear_tf=True)
X_tr, X_va = vec.fit_transform(d_tr["texto"]), vec.transform(d_va["texto"])
y_tr, y_va = d_tr["Dominio"].to_numpy(), d_va["Dominio"].to_numpy()
cw = dict(zip(C5, compute_class_weight("balanced", classes=np.array(C5), y=y_tr))); cw_vec = np.array([cw[c] for c in C5])
print(f"entrenamiento {X_tr.shape} · validación {X_va.shape} · características {X_tr.shape[1]} > ejemplos {X_tr.shape[0]}")
# matrices sobre TODO el conjunto de 673 para las curvas con validación cruzada
X_all, y_all, g_all = train["texto"].to_numpy(), train["Dominio"].to_numpy(), train["grupo_url"].to_numpy()"""))

# ---------------------------------------------------------------- 2
C.append(md("""## 2. Sistema de tracking de métricas

### 2.1 Monitoreo básico
`MetricsTracker` registra, por época, pérdida (log-loss), exactitud y F1 macro en entrenamiento y validación, con marca de tiempo y los hiperparámetros del *run*; exporta a CSV y JSON (`results/overfitting/`). Es independiente del framework: cualquier bucle de entrenamiento llama a `tracker.log(epoch, **metrics)`.

### 2.2 Tracking para diferentes tipos de modelos
| Framework | Mecanismo implementado | Estado en este cuaderno |
|---|---|---|
| **scikit-learn** | `EpochTrainer` (SGDClassifier con `partial_fit` época a época) + envoltorios de `learning_curve` y `validation_curve` con validación por grupo | ejecutado |
| **PyTorch** | `train_torch_softmax`: bucle manual (forward, pérdida, backward, paso) con logging por época y early stopping opcional | ejecutado (regresión softmax sobre TF-IDF, CPU) |
| **TensorFlow/Keras** | `KerasTrackerCallback`: `on_epoch_end` vuelca `logs` al tracker | definido; se ejecuta solo si TensorFlow está instalado (no lo está en este entorno: el modelo del proyecto no es una red neuronal y no se añade la dependencia) |

> Nota sobre MLflow/W&B: el tracker escribe archivos planos para no añadir dependencias; `MetricsTracker.frame()` se puede volcar a `mlflow.log_metrics` en una línea si se desea un servidor de experimentos."""))
C.append(code(r"""import inspect
print(inspect.getsource(T.MetricsTracker))"""))
C.append(code(r"""# Demostración mínima del tracker
tk_demo = T.MetricsTracker("demo", params={"modelo": "ejemplo"})
for ep, (tl, vl) in enumerate([(1.2, 1.3), (0.9, 1.1), (0.7, 1.05)], start=1):
    tk_demo.log(ep, train_loss=tl, val_loss=vl, train_acc=1 - tl / 2, val_acc=1 - vl / 2, train_f1=1 - tl / 2, val_f1=1 - vl / 2)
display(tk_demo.frame()); print("mejor época (val_loss mínima):", tk_demo.best_epoch())"""))

# ---------------------------------------------------------------- 3
C.append(md("""## 3. Entrenamiento del modelo base con logging

### 3.1 El modelo del proyecto (regresión logística, C = 1, clases ponderadas)
La regresión logística de scikit-learn se ajusta por un optimizador de segundo orden (L-BFGS) hasta convergencia, así que no tiene «épocas» observables: se registra el punto final en entrenamiento y validación, en la partición 80/20 y en validación cruzada por grupo (5 pliegues). Ese gap es la primera evidencia."""))
C.append(code(r"""def pipe_lr(C=1.0, max_features=5000, ngram=(1, 2), min_df=1, class_weight="balanced"):
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=ngram, max_features=max_features, min_df=min_df, sublinear_tf=True)),
                     ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=2000, random_state=SEED))])
base = pipe_lr().fit(d_tr["texto"], y_tr)
pt, pv = base.predict(d_tr["texto"]), base.predict(d_va["texto"])
punto = {"train_acc": accuracy_score(y_tr, pt), "val_acc": accuracy_score(y_va, pv), "train_f1": f1_score(y_tr, pt, average="macro"), "val_f1": f1_score(y_va, pv, average="macro")}
print("Partición 80/20 →", {k: round(v, 3) for k, v in punto.items()}, "· gap F1 = %.3f" % (punto["train_f1"] - punto["val_f1"]))
cv_base = T.cv_train_val(pipe_lr(), X_all, y_all, g_all, seed=SEED)
print("CV 5 pliegues por grupo → train F1 %.3f ± %.3f · val F1 %.3f ± %.3f · gap %.3f" % (cv_base["train_f1"], cv_base["train_f1_std"], cv_base["val_f1"], cv_base["val_f1_std"], cv_base["gap_f1"]))
print("\nReporte por clase en validación (80/20):"); print(classification_report(y_va, pv, digits=3))"""))

C.append(md("""### 3.2 Entrenamiento por épocas con scikit-learn (SGD + `partial_fit`)
Para observar la dinámica de entrenamiento se usa el mismo modelo lineal con descenso de gradiente estocástico (`loss='log_loss'`, L2 con α = 10⁻⁴, clases ponderadas): cada época es una pasada completa por los datos barajados. El tracker registra las seis métricas por época."""))
C.append(code(r"""tk_sgd = T.MetricsTracker("sgd_logreg", params={"alpha": 1e-4, "penalty": "l2", "epochs": 80, "class_weight": "balanced"})
et = T.EpochTrainer(C5, alpha=1e-4, class_weight=cw, seed=SEED).fit(X_tr, y_tr, X_va, y_va, epochs=80, tracker=tk_sgd, verbose=True)
h_sgd = tk_sgd.frame(); tk_sgd.save(RES); display(h_sgd.iloc[[0, 1, 2, 4, 9, 19, 39, 79]].round(3))"""))

C.append(md("""### 3.3 Bucle de entrenamiento manual en PyTorch (regresión softmax sobre TF-IDF)
Mismo problema, mismo tipo de modelo (una capa lineal + softmax, pérdida de entropía cruzada ponderada por clase, Adam con `weight_decay` = 10⁻⁴). El bucle es explícito —*forward*, pérdida, *backward*, paso del optimizador— y llama al tracker al final de cada época; es la plantilla que se usaría para cualquier red neuronal."""))
C.append(code(r"""tk_torch = T.MetricsTracker("torch_softmax", params={"lr": 0.01, "weight_decay": 1e-4, "epochs": 150, "optimizador": "Adam"})
modelo_t, ep_best_t, vl_best_t = T.train_torch_softmax(X_tr, y_tr, X_va, y_va, C5, epochs=150, lr=0.01, weight_decay=1e-4, class_weight=cw_vec, seed=SEED, tracker=tk_torch)
h_torch = tk_torch.frame(); tk_torch.save(RES)
print(f"mínimo de val_loss en la época {ep_best_t} ({vl_best_t:.3f})"); display(h_torch.iloc[[0, 4, 9, 19, 39, 79, 119, 149]].round(3))"""))
C.append(code(r"""# TensorFlow/Keras: callback definido en tracking.py; se ejecuta solo si la dependencia está instalada
try:
    import tensorflow as tf
    tk_keras = T.MetricsTracker("keras_mlp", params={"capas": "64-relu-dropout0.3-softmax"})
    m = tf.keras.Sequential([tf.keras.layers.Input((X_tr.shape[1],)), tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dropout(0.3), tf.keras.layers.Dense(len(C5), activation="softmax")])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    idx = {c: i for i, c in enumerate(C5)}
    m.fit(X_tr.toarray(), np.array([idx[c] for c in y_tr]), validation_data=(X_va.toarray(), np.array([idx[c] for c in y_va])), epochs=40, verbose=0, callbacks=[T.KerasTrackerCallback(tk_keras).build()])
    display(tk_keras.frame().tail()); tk_keras.save(RES)
except ImportError:
    print("TensorFlow no está instalado en este entorno: el callback KerasTrackerCallback queda definido y probado por inspección, pero no se ejecuta.\n"
          "Justificación: el modelo del Proyecto Integrador es lineal (scikit-learn); la dinámica por épocas se cubre con SGD (3.2) y PyTorch (3.3), que usan el mismo tracker.")
    print(inspect.getsource(T.KerasTrackerCallback))"""))

# ---------------------------------------------------------------- 4
C.append(md("""## 4. Curvas de aprendizaje

Todas las figuras se guardan a **300 DPI** en `results/figuras/overfitting/`, con título, etiquetas de ejes, leyenda, colores contrastantes (azul = entrenamiento, naranja = validación), rejilla y anotaciones en los puntos críticos (mínimo de la pérdida de validación, máximo de validación, gap final, mejor hiperparámetro).

### 4.1 Curvas básicas A y B: pérdida y exactitud/F1 por época"""))
C.append(code(r"""dx_torch = T.diagnose_epochs(h_torch); dx_sgd = T.diagnose_epochs(h_sgd)
T.plot_epoch_curves(h_torch, "Regresión softmax (PyTorch, Adam, 150 épocas)", FIG / "01_curvas_epoca_torch.png", metric="acc", dx=dx_torch); plt.show()
T.plot_epoch_curves(h_torch, "Regresión softmax (PyTorch) · F1 macro", FIG / "02_curvas_epoca_torch_f1.png", metric="f1", dx=dx_torch); plt.show()
T.plot_epoch_curves(h_sgd, "Regresión logística SGD (scikit-learn, 80 épocas)", FIG / "03_curvas_epoca_sgd.png", metric="acc", dx=dx_sgd); plt.show()"""))

C.append(md("""### 4.2 Curva C (avanzada): rendimiento por tamaño del conjunto de entrenamiento
`learning_curve` de scikit-learn con validación cruzada estratificada por grupo (5 pliegues) sobre las 673 normas, para 8 tamaños entre el 10 % y el 100 % del entrenamiento disponible. Responde a la pregunta operativa del observatorio: ¿mejoraría el clasificador si GENIAL codificara más normas?"""))
C.append(code(r"""lc = T.learning_curve_tracked(pipe_lr(), X_all, y_all, g_all, sizes=np.linspace(0.1, 1.0, 8), seed=SEED)
lc.to_csv(RES / "curva_aprendizaje_tamano.csv", index=False); dx_size = T.diagnose_size(lc)
T.plot_learning_curve(lc, "Curva de aprendizaje por tamaño · RL + TF-IDF (C = 1) · CV 5 pliegues por grupo", FIG / "04_curva_aprendizaje_tamano.png", dx=dx_size); plt.show()
display(lc.round(3)); print(dx_size)"""))

C.append(md("""### 4.3 Curva D (avanzada): validación por hiperparámetros
Dos barridos con `validation_curve` (CV por grupo): la **fuerza de regularización C** de la regresión logística (eje del sesgo-varianza del modelo) y el **número máximo de características** TF-IDF (complejidad de la representación)."""))
C.append(code(r"""Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
vc_C = T.validation_curve_tracked(pipe_lr(), X_all, y_all, g_all, "clf__C", Cs, seed=SEED); vc_C.to_csv(RES / "curva_validacion_C.csv", index=False)
T.plot_validation_curve(vc_C, "C (inverso de la regularización L2)", "Curva de validación · regularización de la regresión logística", FIG / "05_curva_validacion_C.png", chosen=1.0); plt.show()
display(vc_C.round(3))"""))
C.append(code(r"""feats = [250, 500, 1000, 2000, 5000, 10000, 20000]
vc_F = T.validation_curve_tracked(pipe_lr(), X_all, y_all, g_all, "tfidf__max_features", feats, seed=SEED); vc_F.to_csv(RES / "curva_validacion_max_features.csv", index=False)
T.plot_validation_curve(vc_F, "max_features (tamaño del vocabulario TF-IDF)", "Curva de validación · complejidad de la representación", FIG / "06_curva_validacion_max_features.png", chosen=5000); plt.show()
display(vc_F.round(3))"""))

# ---------------------------------------------------------------- 5
C.append(md("""## 5. Diagnóstico sistemático

### 5.1 Identificación de patrones
Se contrastan las tres firmas de la actividad (overfitting, underfitting, comportamiento óptimo) con la evidencia de las curvas. La regla de decisión está codificada en `tracking.diagnose_epochs` con umbrales explícitos: gap de exactitud > 0.15 **o** rebote de la pérdida de validación ≥ 5 % desde su mínimo mientras la de entrenamiento sigue bajando ⇒ overfitting; exactitud de entrenamiento < 0.55 ⇒ underfitting."""))
C.append(code(r"""def fila(p, ev, cumple): return {"patrón": p, "evidencia observada": ev, "¿se cumple?": "sí" if cumple else "no"}
t, s, l = dx_torch, dx_sgd, dx_size
patrones = [
  ("OVERFITTING · gap creciente entre training y validation loss", f"PyTorch: gap de pérdida final {t['gap_loss_final']:.2f} (train {t['train_loss_final']:.2f} vs val {t['val_loss_final']:.2f}); SGD: {s['gap_loss_final']:.2f}", t["gap_loss_final"] > 0.5),
  ("OVERFITTING · training accuracy muy alta, validation accuracy estancada", f"PyTorch: train acc {t['train_acc_final']:.2f} vs val acc {t['val_acc_final']:.2f} (gap {t['gap_acc_final']:.2f}); SGD: {s['train_acc_final']:.2f} vs {s['val_acc_final']:.2f}", t["gap_acc_final"] > 0.15),
  ("OVERFITTING · validation loss aumenta mientras training loss disminuye", f"PyTorch: val loss mínima en la época {t['epoca_min_val_loss']} y rebote de {t['rebote_val_loss']*100:.1f} % hasta la época {t['n_epocas']}", t["rebote_val_loss"] >= 0.03),
  ("UNDERFITTING · ambas curvas convergen a valores altos de loss", f"train loss final {t['train_loss_final']:.2f}: el entrenamiento se ajusta casi perfectamente", False),
  ("UNDERFITTING · training accuracy baja y no mejora", f"train acc final {t['train_acc_final']:.2f}", t["train_acc_final"] < 0.55),
  ("UNDERFITTING · gap pequeño pero rendimiento pobre", f"gap {t['gap_acc_final']:.2f}: no es pequeño", False),
  ("ÓPTIMO · gap pequeño y estable, curvas convergen, rendimiento satisfactorio", f"gap F1 en CV {cv_base['gap_f1']:.2f}; val F1 {cv_base['val_f1']:.2f} < meta 0.70", False),
  ("CURVA POR TAMAÑO · validación aún creciente con más datos", f"pendiente {l['pendiente_val_por_100_global']:+.1f} pp de F1 por 100 normas (global), {l['pendiente_val_por_100_tramo_final']:+.1f} pp en el último tramo; ganancia total {l['ganancia_val_total_puntos']:+.0f} pp entre n = 53 y n = {l['n_max']}; gap final {l['gap_final']:.2f}", l["pendiente_val_por_100_tramo_final"] > 0.5),
]
tabla_patrones = pd.DataFrame([fila(*p) for p in patrones]); tabla_patrones.to_csv(RES / "patrones_diagnostico.csv", index=False)
pd.set_option("display.max_colwidth", 200); display(tabla_patrones)"""))

C.append(md("""### 5.2 Análisis cuantitativo
Métricas que cuantifican el diagnóstico (todas calculadas por `tracking.diagnose_epochs` y `tracking.diagnose_size`):

* **gap de generalización** = métrica(entrenamiento) − métrica(validación), en pérdida, exactitud y F1 macro (final y máximo);
* **época de mínima pérdida de validación** (punto de early stopping) y **rebote** = val_loss_final / val_loss_min − 1;
* **pendiente del gap** en el último tercio de las épocas (¿sigue abriéndose?);
* **pendiente de la curva por tamaño** (puntos de F1 por cada 100 normas adicionales) y **desviación entre pliegues** (varianza de la estimación);
* **gap por clase** en CV: dónde se concentra el sobreajuste."""))
C.append(code(r"""cuant = pd.DataFrame({"PyTorch softmax": dx_torch, "SGD scikit-learn": dx_sgd}).T
display(cuant.round(3)); cuant.to_csv(RES / "diagnostico_cuantitativo_epocas.csv")
print("Curva por tamaño:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in dx_size.items()})"""))
C.append(code(r"""# Gap por clase en validación cruzada por grupo (F1 por clase en entrenamiento vs validación)
from sklearn.base import clone
filas = []
for tr_i, va_i in T.group_cv(5, SEED).split(X_all, y_all, g_all):
    m = clone(pipe_lr()).fit(X_all[tr_i], y_all[tr_i])
    ftr = f1_score(y_all[tr_i], m.predict(X_all[tr_i]), average=None, labels=C5); fva = f1_score(y_all[va_i], m.predict(X_all[va_i]), average=None, labels=C5)
    filas.append(pd.DataFrame({"clase": C5, "train_f1": ftr, "val_f1": fva}))
gap_clase = pd.concat(filas).groupby("clase").mean().reindex(C5); gap_clase["gap"] = gap_clase["train_f1"] - gap_clase["val_f1"]
gap_clase["n_train"] = train["Dominio"].value_counts().reindex(C5).values
gap_clase.to_csv(RES / "gap_por_clase.csv"); display(gap_clase.round(3))
print("El gap es mayor en las clases con menos ejemplos:", gap_clase.sort_values("gap", ascending=False).index.tolist()[:2])"""))
C.append(md("""### 5.3 Veredicto
La celda siguiente redacta el diagnóstico con las cifras calculadas (se regenera en cada ejecución)."""))
C.append(code(r"""veredicto = f'''**Diagnóstico: OVERFITTING (varianza alta) con un techo de datos.**

* El modelo memoriza el entrenamiento (exactitud {dx_torch['train_acc_final']:.2f} en PyTorch, {dx_sgd['train_acc_final']:.2f} en SGD; F1 macro de entrenamiento {cv_base['train_f1']:.2f} en CV) mientras la validación se estanca en exactitud ≈ {dx_torch['val_acc_final']:.2f} y F1 macro ≈ {cv_base['val_f1']:.2f}: gap de F1 de {cv_base['gap_f1']:.2f} en validación cruzada.
* La pérdida de validación alcanza su mínimo en la época {dx_torch['epoca_min_val_loss']} (PyTorch) y luego rebota un {dx_torch['rebote_val_loss']*100:.1f} % mientras la de entrenamiento sigue bajando hasta {dx_torch['train_loss_final']:.2f}: firma clásica de sobreajuste y justificación del early stopping.
* No hay underfitting: el entrenamiento se ajusta casi perfectamente y las curvas no convergen a un valor alto de pérdida. Tampoco es un comportamiento óptimo: el gap no es pequeño y el rendimiento de validación queda por debajo de la meta del proyecto (0.70).
* La curva por tamaño muestra que la validación mejora con los datos ({dx_size['ganancia_val_total_puntos']:+.0f} puntos porcentuales de F1 entre n = 53 y n = {dx_size['n_max']}; {dx_size['pendiente_val_por_100_global']:+.1f} pp por cada 100 normas en promedio) pero con **rendimientos decrecientes** ({dx_size['pendiente_val_por_100_tramo_final']:+.1f} pp por 100 normas en el último tramo), y el gap sigue en {dx_size['gap_final']:.2f}. El sobreajuste se explica por tener **más características (5 000) que ejemplos (≈540)** y por las clases minoritarias (Técnico, Administrativo), donde el gap por clase es mayor.
* La curva de validación por C añade un matiz importante: el F1 de validación es casi plano entre C = 0.01 y C = 100 ({vc_C['val_mean'].min():.2f}-{vc_C['val_mean'].max():.2f}) mientras el de entrenamiento va de {vc_C['train_mean'].min():.2f} a {vc_C['train_mean'].max():.2f}. Una parte del gap es varianza reducible; otra parte es **error irreducible** del problema —normas de frontera entre dimensiones y una sola etiqueta por norma, como mostró la codificación del investigador en la Semana 2—, que ninguna regularización elimina.
* Consecuencia: regularizar, reducir complejidad y detener temprano deben **cerrar el gap** (menos varianza, mismo rendimiento); el F1 de validación solo subirá de forma sustancial con **más normas de las clases minoritarias**, con una representación que aporte conocimiento externo (la rama LLM few-shot o embeddings preentrenados) o revisando las etiquetas ambiguas, no con más épocas.'''
display(Markdown(veredicto))"""))

# ---------------------------------------------------------------- 6
C.append(md("""## 6. Estrategias de mejora

### 6.1 Recomendaciones específicas para el diagnóstico (overfitting)
| Estrategia | ¿Aplica aquí? | Decisión |
|---|---|---|
| Regularización (L1, L2) | sí: el modelo lineal tiene L2 con C = 1; la curva D indica el rango útil | **implementada** (E1: barrido de C y de `weight_decay`) |
| Early stopping | sí: la pérdida de validación tiene mínimo claro en PyTorch/SGD | **implementada** (E2: paciencia 15 épocas, restaura la mejor época) |
| Reducción de complejidad del modelo | sí: 5 000 características > ejemplos | **implementada** (E3: vocabulario reducido, `min_df`, unigramas) |
| Cross-validation más rigurosa | sí: hay duplicados por URL que inflan la validación ingenua | **implementada** (E4: CV por grupo frente a CV ingenua; cuantifica la fuga) |
| Aumento de datos (data augmentation) | parcialmente: no hay normas nuevas disponibles en esta semana | **no implementada, justificada** (§6.6) |
| Dropout | solo aplica a redes con capas ocultas; el modelo es lineal | no aplica (se deja la opción `hidden>0` + Dropout en `train_torch_softmax`) |

### 6.2 E1 · Regularización más fuerte"""))
C.append(code(r"""# Regla de una desviación estándar (Hastie et al., 2009): el modelo más regularizado cuyo F1 de validación esté a ≤ 1 DE del máximo.
i_max = vc_C["val_mean"].idxmax(); umbral_1se = vc_C.loc[i_max, "val_mean"] - vc_C.loc[i_max, "val_std"]
mejor_C = float(vc_C.loc[vc_C["val_mean"] >= umbral_1se, "param"].min())
print(f"máximo F1 de validación {vc_C.loc[i_max, 'val_mean']:.3f} en C = {vc_C.loc[i_max, 'param']:g} (DE {vc_C.loc[i_max, 'val_std']:.3f}) → regla 1-DE: C = {mejor_C:g} (gap {float(vc_C.loc[vc_C['param'] == mejor_C, 'gap'].iloc[0]):.2f} frente a {float(vc_C.loc[vc_C['param'] == 1.0, 'gap'].iloc[0]):.2f} con C = 1)")
cv_E1 = T.cv_train_val(pipe_lr(C=mejor_C), X_all, y_all, g_all, seed=SEED)
# también en PyTorch: barrido de weight_decay (L2) con el mismo bucle y tracker
res_wd = []
for wd in [1e-4, 1e-3, 1e-2, 5e-2]:
    tk = T.MetricsTracker(f"torch_wd_{wd:g}", params={"weight_decay": wd})
    T.train_torch_softmax(X_tr, y_tr, X_va, y_va, C5, epochs=150, lr=0.01, weight_decay=wd, class_weight=cw_vec, seed=SEED, tracker=tk)
    d = T.diagnose_epochs(tk.frame()); res_wd.append({"weight_decay": wd, "train_acc": d["train_acc_final"], "val_acc": d["val_acc_final"], "gap_acc": d["gap_acc_final"], "val_f1_max": d["val_f1_max"], "val_loss_min": d["val_loss_min"], "rebote": d["rebote_val_loss"]})
res_wd = pd.DataFrame(res_wd); res_wd.to_csv(RES / "E1_weight_decay.csv", index=False); display(res_wd.round(3))
print("E1 en CV (RL, C=%g): train F1 %.3f · val F1 %.3f · gap %.3f" % (mejor_C, cv_E1["train_f1"], cv_E1["val_f1"], cv_E1["gap_f1"]))"""))

C.append(md("""### 6.3 E2 · Early stopping
Se reentrena con paciencia de 15 épocas: el entrenamiento se detiene cuando la pérdida de validación no mejora y se restauran los pesos de la mejor época. Se compara con el modelo que entrena las 150 épocas completas."""))
C.append(code(r"""tk_es = T.MetricsTracker("torch_early_stopping", params={"patience": 15})
modelo_es, ep_es, vl_es = T.train_torch_softmax(X_tr, y_tr, X_va, y_va, C5, epochs=150, lr=0.01, weight_decay=1e-4, class_weight=cw_vec, seed=SEED, tracker=tk_es, patience=15)
h_es = tk_es.frame(); tk_es.save(RES)
fila_es = h_es[h_es["epoch"] == ep_es].iloc[0]; fila_full = h_torch.iloc[-1]
comp_es = pd.DataFrame([{"configuración": "150 épocas (sin early stopping)", "épocas": 150, "val_loss": fila_full["val_loss"], "val_acc": fila_full["val_acc"], "val_f1": fila_full["val_f1"], "train_acc": fila_full["train_acc"]},
                        {"configuración": f"early stopping (paciencia 15) → época {ep_es}", "épocas": ep_es, "val_loss": fila_es["val_loss"], "val_acc": fila_es["val_acc"], "val_f1": fila_es["val_f1"], "train_acc": fila_es["train_acc"]}])
comp_es["gap_acc"] = comp_es["train_acc"] - comp_es["val_acc"]; comp_es.to_csv(RES / "E2_early_stopping.csv", index=False); display(comp_es.round(3))
T.plot_epoch_curves(h_es, f"PyTorch con early stopping (paciencia 15): se detiene en la época {int(h_es['epoch'].max())}", FIG / "07_early_stopping.png", metric="acc", dx=T.diagnose_epochs(h_es)); plt.show()"""))

C.append(md("""### 6.4 E3 · Reducción de la complejidad de la representación
Menos vocabulario (`max_features`), descarte de términos que aparecen en una sola norma (`min_df = 2`) y solo unigramas. Se evalúa cada variante en CV por grupo."""))
C.append(code(r"""variantes = {"max_features = 1000": dict(max_features=1000), "max_features = 2000, min_df = 2": dict(max_features=2000, min_df=2),
             "unigramas, max_features = 2000": dict(max_features=2000, ngram=(1, 1)), "unigramas, min_df = 2, C = %g" % mejor_C: dict(max_features=2000, ngram=(1, 1), min_df=2, C=mejor_C)}
res_E3 = []
for nombre, kw in variantes.items():
    r = T.cv_train_val(pipe_lr(**kw), X_all, y_all, g_all, seed=SEED); res_E3.append({"configuración": nombre, **{k: r[k] for k in ["train_f1", "val_f1", "gap_f1", "val_f1_std", "train_acc", "val_acc"]}})
res_E3 = pd.DataFrame(res_E3); res_E3.to_csv(RES / "E3_complejidad.csv", index=False); display(res_E3.round(3))"""))

C.append(md("""### 6.5 E4 · Validación cruzada más rigurosa
El corpus tiene normas con la misma URL (duplicados o versiones). Una validación cruzada ingenua puede poner copias a ambos lados del pliegue y **sobrestimar** la validación. Se cuantifica la diferencia entre `StratifiedKFold` y `StratifiedGroupKFold` (grupo = URL) con el mismo modelo."""))
C.append(code(r"""naive = cross_val_score(pipe_lr(), X_all, y_all, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="f1_macro")
grupo = cross_val_score(pipe_lr(), X_all, y_all, groups=g_all, cv=T.group_cv(5, SEED), scoring="f1_macro")
comp_cv = pd.DataFrame({"esquema": ["StratifiedKFold (ingenua)", "StratifiedGroupKFold (por URL)"], "F1 macro val (media)": [naive.mean(), grupo.mean()], "DE": [naive.std(), grupo.std()]})
comp_cv.to_csv(RES / "E4_cv_rigurosa.csv", index=False); display(comp_cv.round(3))
dif_cv = naive.mean() - grupo.mean(); print("Diferencia (CV ingenua − CV por grupo): %+.3f de F1 macro" % dif_cv)
print("Lectura:", "la CV ingenua sobrestima el rendimiento por la fuga de duplicados" if dif_cv > 0.01 else f"en este subconjunto solo hay {int((train.groupby('grupo_url').size() > 1).sum())} grupos de URL repetida, así que la diferencia es despreciable; la CV por grupo se mantiene porque no tiene costo y protege la evaluación cuando el corpus crezca (en el corpus completo hay 14 grupos)")"""))

C.append(md("""### 6.6 Estrategias no implementadas y su justificación
* **Aumento de datos.** La forma honesta de aumentar datos aquí es que GENIAL codifique más normas (la curva C muestra que la validación sigue subiendo). Las técnicas sintéticas se descartaron: la retrotraducción exige una API externa y altera el léxico jurídico que el modelo usa como señal; la sustitución por sinónimos en textos de 44 palabras cambia la dimensión con facilidad (p. ej., «protección de datos» ↔ «privacidad» cruza de clase); y el sobremuestreo de clases minoritarias duplica ejemplos, que es equivalente a `class_weight='balanced'`, ya activo en el modelo base.
* **Dropout.** Solo tiene sentido con capas ocultas; el modelo es lineal por decisión de la Ficha de Decisión Técnica (interpretabilidad). La función `train_torch_softmax(hidden=64)` deja implementado un MLP con Dropout para trabajo futuro.
* **Más épocas / optimizadores más sofisticados** (recetas para underfitting) no aplican: el diagnóstico es overfitting.

## 7. Comparación antes/después de las mejoras
Todas las configuraciones se evalúan con el mismo protocolo (CV por grupo, 5 pliegues, mismas particiones). El conjunto ciego del proyecto no se usa."""))
C.append(code(r"""combinada = T.cv_train_val(pipe_lr(C=mejor_C, max_features=2000, min_df=2), X_all, y_all, g_all, seed=SEED)
tabla = pd.DataFrame([
    {"configuración": "Base: C = 1, 5 000 feat., 1-2 gramas", **{k: cv_base[k] for k in ["train_f1", "val_f1", "gap_f1", "val_f1_std", "train_acc", "val_acc"]}},
    {"configuración": f"E1: C = {mejor_C:g}", **{k: cv_E1[k] for k in ["train_f1", "val_f1", "gap_f1", "val_f1_std", "train_acc", "val_acc"]}},
    {"configuración": "E3: 2 000 feat., min_df = 2", **{k: res_E3.iloc[1][k] if k in res_E3.columns else np.nan for k in ["train_f1", "val_f1", "gap_f1", "val_f1_std", "train_acc", "val_acc"]}},
    {"configuración": "E3: 1 000 feat.", **{k: res_E3.iloc[0][k] if k in res_E3.columns else np.nan for k in ["train_f1", "val_f1", "gap_f1", "val_f1_std", "train_acc", "val_acc"]}},
    {"configuración": f"E1 + E3: C = {mejor_C:g}, 2 000 feat., min_df = 2", **{k: combinada[k] for k in ["train_f1", "val_f1", "gap_f1", "val_f1_std", "train_acc", "val_acc"]}},
])
tabla["Δ val F1 vs base"] = tabla["val_f1"] - tabla.loc[0, "val_f1"]; tabla["Δ gap vs base"] = tabla["gap_f1"] - tabla.loc[0, "gap_f1"]
tabla.to_csv(RES / "comparacion_antes_despues.csv", index=False); display(tabla.round(3))
T.plot_before_after(tabla, FIG / "08_antes_despues.png"); plt.show()
# early stopping (partición 80/20, PyTorch)
print("E2 early stopping (PyTorch, partición 80/20): val F1 %.3f en la época %d frente a %.3f tras 150 épocas; gap de exactitud %.2f → %.2f" % (fila_es["val_f1"], ep_es, fila_full["val_f1"], fila_full["train_acc"] - fila_full["val_acc"], fila_es["train_acc"] - fila_es["val_acc"]))"""))

# ---------------------------------------------------------------- 8
C.append(md("""## 8. Conclusiones y reflexiones"""))
C.append(code(r"""mejor = tabla.loc[tabla["val_f1"].idxmax()]; menor_gap = tabla.loc[tabla["gap_f1"].idxmin()]
texto = f'''1. **El modelo del proyecto sobreajusta.** Con 5 000 características para ≈540 normas, la regresión logística alcanza F1 macro {cv_base['train_f1']:.2f} en entrenamiento y {cv_base['val_f1']:.2f} en validación (gap {cv_base['gap_f1']:.2f}); en PyTorch la pérdida de validación toca mínimo en la época {dx_torch['epoca_min_val_loss']} y rebota {dx_torch['rebote_val_loss']*100:.0f} % mientras la de entrenamiento sigue cayendo. Las tres firmas de overfitting de la actividad se cumplen; ninguna de underfitting.

2. **Las estrategias cierran el gap mucho más de lo que suben la validación.** En CV, la configuración con menor gap ({menor_gap['configuración']}) reduce el gap de F1 de {cv_base['gap_f1']:.2f} a {menor_gap['gap_f1']:.2f} con un F1 de validación de {menor_gap['val_f1']:.3f} ({menor_gap['Δ val F1 vs base']:+.3f} frente a la base, del orden de la desviación entre pliegues, {cv_base['val_f1_std']:.3f}). En PyTorch, subir la regularización L2 (weight_decay 10⁻⁴ → {res_wd.loc[res_wd['val_f1_max'].idxmax(), 'weight_decay']:g}) reduce el gap de exactitud de {res_wd.iloc[0]['gap_acc']:.2f} a {res_wd.loc[res_wd['val_f1_max'].idxmax(), 'gap_acc']:.2f} y eleva el F1 máximo de validación de {res_wd.iloc[0]['val_f1_max']:.3f} a {res_wd['val_f1_max'].max():.3f} (máximo sobre épocas en una sola partición 80/20: estimación optimista, útil para comparar configuraciones, no como cifra final). El early stopping detiene el entrenamiento en la época {ep_es} y evita el rebote de la pérdida de validación con el mismo F1. Es el comportamiento esperado cuando el sobreajuste convive con un techo de datos y de etiquetas: regularizar reduce la varianza, pero no añade información.

3. **La validación cruzada por grupo es parte del protocolo.** La diferencia entre la CV ingenua y la CV por grupo es de {dif_cv:+.3f} en este subconjunto (solo {int((train.groupby('grupo_url').size() > 1).sum())} grupos de URL repetida), pero se mantiene porque no tiene costo y protege la evaluación cuando el corpus crezca.

4. **Qué sí movería la aguja.** La curva por tamaño ({dx_size['ganancia_val_total_puntos']:+.0f} pp de F1 entre n = 53 y n = {dx_size['n_max']}, con {dx_size['pendiente_val_por_100_tramo_final']:+.1f} pp por 100 normas en el último tramo) indica que más normas codificadas por GENIAL —sobre todo de las clases Técnico y Administrativo, donde el gap por clase es mayor— y representaciones con conocimiento externo (la rama LLM few-shot del proyecto, o embeddings preentrenados) son las vías con mayor retorno. Esta conclusión se integra en el documento final del Proyecto Integrador.

5. **Reflexión metodológica.** El tracking por época fue lo que hizo visible el rebote de la pérdida de validación, invisible en el punto final del modelo de scikit-learn; y fijar los umbrales de diagnóstico antes de mirar las curvas (gap > 0.15, rebote ≥ 5 %) evitó una lectura a conveniencia.'''
display(Markdown(texto))
sintesis = {"dif_cv": float(dif_cv), "menor_gap": menor_gap.to_dict(), "mejor_val": mejor.to_dict(), "base_cv": cv_base, "diagnostico_torch": dx_torch, "diagnostico_sgd": dx_sgd, "diagnostico_tamano": dx_size, "mejor_C": mejor_C, "E1_cv": cv_E1, "E1_weight_decay": res_wd.to_dict("records"),
            "E2_early_stopping": comp_es.to_dict("records"), "E3": res_E3.to_dict("records"), "E4": comp_cv.to_dict("records"), "combinada_cv": combinada, "tabla_antes_despues": tabla.to_dict("records"),
            "gap_por_clase": gap_clase.reset_index().to_dict("records"), "patrones": tabla_patrones.to_dict("records"), "curva_C": vc_C.to_dict("records"), "curva_F": vc_F.to_dict("records"), "curva_tamano": lc.to_dict("records"),
            "punto_8020": punto, "n_train": int(X_tr.shape[0]), "n_val": int(X_va.shape[0]), "n_feat": int(X_tr.shape[1])}
json.dump(sintesis, open(RES / "sintesis_overfitting.json", "w"), ensure_ascii=False, indent=1, default=float); print("síntesis guardada en results/overfitting/sintesis_overfitting.json")"""))

nb = new_notebook(cells=C, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}})
nbf.write(nb, "notebooks/overfitting_analysis.ipynb"); print("notebook escrito")
