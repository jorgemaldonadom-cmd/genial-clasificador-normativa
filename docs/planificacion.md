# Planificación del proyecto

**Clasificación automática de normativa sobre IA en educación superior de ALC: hacia un Índice de Madurez Normativa (IMN) escalable**
Proyecto Integrador (MIAR0545) · Maestría en Inteligencia Artificial · UEES · Grupo 9 · Jorge Javier Maldonado Mahauad y Melissa Figallo · Tutora: Gladys María Villegas Rugel

## 1. Problema y objetivo

El observatorio regional del proyecto **GENIAL** (Erasmus+) codifica a mano cada norma sobre inteligencia artificial en educación superior en una de cinco dimensiones (Pedagógico, Ética, Administrativo, Protección de Datos, Técnico). Ese conteo alimenta el Índice de Madurez Normativa por país. La codificación manual (≈ 10 min por norma) no escala al ritmo al que aparecen normas, depende de un solo codificador por norma y no tiene fiabilidad intercodificador medida.

**Objetivo general.** Construir y evaluar un clasificador automático de la dimensión a partir del texto de la norma (título + observaciones) e integrarlo en un flujo de *codificación asistida* con revisión humana obligatoria allí donde el modelo no es fiable.

**Objetivos específicos.**

1. Preparar y auditar el corpus GENIAL (813 normas, 28 jurisdicciones) con una partición ciega independiente.
2. Comparar una rama clásica (TF-IDF + Regresión Logística) con líneas base y con una rama LLM *few-shot* (Claude Sonnet 5.5, 15 ejemplos), sobre el mismo conjunto ciego.
3. Diagnosticar sobreajuste, optimizar hiperparámetros con análisis de sensibilidad y elegir el modelo de producción con un criterio explícito.
4. Auditar la equidad por jurisdicción y fijar la regla de gobernanza (umbral de confianza, revisión obligatoria).
5. Entregar una aplicación web funcional, un repositorio reproducible y la documentación técnica y ética.

## 2. Alcance

| Dentro | Fuera |
|---|---|
| Clasificación en 5 dimensiones a partir de título + observaciones | Clasificación multi-etiqueta (una norma, varias dimensiones) |
| Modelo clásico interpretable, umbral de confianza, enrutamiento a revisión humana | Sustitución del codificador humano |
| Evaluación ciega contra la codificación de un investigador del consorcio | Cálculo oficial del IMN (lo publica GENIAL) |
| Auditoría de equidad por idioma, país, región y codificador | Modelos multilingües de gran tamaño ajustados (*fine-tuning*) |
| Aplicación Streamlit para una norma y por lotes | Integración con el sistema de información de GENIAL |

## 3. Metodología de trabajo

Ciclo de siete semanas con entregables semanales evaluados, organizados como *sprints*:

| Semana | Entregable | Estado |
|---|---|---|
| 1 | Canvas SMART, análisis de datasets, stakeholders, estado del arte, KPIs, cronograma, pitch de 3 min | entregado |
| 2 | EDA completo, análisis comparativo de ocho técnicas, Ficha de Decisión Técnica y Arquitectura | entregado |
| 3 | Experimento (preparación, líneas base, rama clásica, prueba ciega) + diagnóstico de *overfitting/underfitting* | entregado |
| 4 | Workshop de impacto social y responsabilidad: auditoría de equidad, *model card*, *datasheet* | entregado |
| 5 | Workshop de optimización de hiperparámetros con análisis de sensibilidad | entregado (cuaderno 04, `docs/optimizacion.md`) |
| 6 | Aplicación web, pruebas, repositorio final, documentación | entregado |
| 7 | Vídeo *pitch* (5 min), vídeo de preguntas, presentación final | en curso |

Herramientas: Python 3.11, scikit-learn, Optuna, Streamlit, pytest, Git/GitHub (ramas `main` y `dev`, *tags* por versión), cuadernos Jupyter ejecutados y versionados con sus salidas.

## 4. Roles

| Persona | Rol | Responsabilidades |
|---|---|---|
| Jorge Javier Maldonado Mahauad | Dirección científica, ingeniería de datos y modelado | Acceso y gobernanza del corpus GENIAL, preprocesamiento, modelo clásico, optimización, evaluación, repositorio, aplicación |
| Melissa Figallo | Análisis, documentación y validación | Revisión del instrumento de codificación, análisis de errores, documentación ética, pruebas de la aplicación, vídeo |
| Investigador A (GENIAL) | Codificador ciego | Codificación independiente de 142 normas con el Framework Dimensional D1-D6 (conjunto de prueba) |
| Tutora | Seguimiento | Revisión semanal de entregables |

## 5. KPIs y criterios de éxito

| KPI | Meta planificada | Resultado |
|---|---|---|
| F1 macro en prueba ciega | ≥ 0.60 | **0.631** [0.530, 0.724] (`best_model`) |
| Exactitud en prueba ciega | ≥ 0.65 | **0.672** [0.586, 0.748] |
| Superar líneas base (McNemar) | p < 0.05 vs mayoritaria y metadatos | **p < 0.001** y **p = 0.012** |
| κ modelo-codificador | ≥ 0.40 (moderado) | **0.54** |
| Exactitud del tramo automático | ≥ 0.85 | **0.85-0.89** (CV por grupo) |
| Aplicación web funcional con pruebas | ≥ 3 pruebas automatizadas | **14 pruebas** (datos, modelo, app) |
| Documentación ética completa | 8 secciones | `docs/consideraciones_eticas.md` |

## 6. Riesgos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación aplicada |
|---|---|---|---|
| Corpus pequeño y desbalanceado (4.7:1) | alta | alto | `class_weight = balanced`, validación cruzada por grupo, intervalos de confianza en todo |
| Fuga de información por duplicados de URL o por la columna `Investigador` | media | alto | `StratifiedGroupKFold` por `grupo_url`; columnas de fuga excluidas por diseño |
| Etiqueta única en normas multidimensionales | alta | medio | top-2 y tipificación de errores; recomendación de multi-etiqueta a GENIAL |
| Peor desempeño en idiomas y países minoritarios | alta | alto | auditoría de equidad; revisión humana obligatoria para en/fr/pt y países < 20 normas |
| Coste y disponibilidad de la API para la rama LLM | media | bajo | clave fuera del repositorio; guardado incremental y reanudación; ejecutada con Claude Sonnet 5.5 (≈ 2 USD) |
| Sobreajuste del modelo (gap 0.40) | alta | medio | diagnóstico Semana 3, regularización y regla 1-DE en la optimización |
| Datos no publicables | — | alto | corpus e instrumento fuera del repositorio; solo hash, agregados y muestra anonimizada |

## 7. Cronograma (Gantt simplificado)

```
Semana        1   2   3   4   5   6   7
Planificación ███
EDA / datos       ███
Modelado              ███ ███
Optimización                  ███
Ética / equidad           ███
App + pruebas                 ███ ███
Documentación                     ███ ███
Vídeos / cierre                       ███
```

Cierre del proyecto: 25 de octubre de 2026.
