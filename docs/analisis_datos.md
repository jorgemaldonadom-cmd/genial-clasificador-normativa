# Análisis de datos

Resumen del análisis exploratorio (`notebooks/01_exploracion.ipynb`, informe completo en `results/reports/semana2/Grupo_9_S2_EDA_Corpus_GENIAL.pdf`) y del preprocesamiento (`notebooks/02_preprocesamiento.ipynb`). La ficha de datos completa está en `docs/datasheet_corpus.md`.

## 1. Fuente

Corpus consolidado del observatorio de normativa sobre IA en educación superior del proyecto **GENIAL (Erasmus+)**: **813 normas × 13 columnas**, 27 países de América Latina y el Caribe (28 jurisdicciones con Puerto Rico), codificadas entre 2024 y 2026 por diez investigadores del consorcio. Versión congelada verificada por hash SHA-256 (`f9797147…3f33`, en `config.yaml`). El archivo no se versiona; `data/samples/sample_corpus_40.csv` contiene 40 normas (8 por clase) para pruebas y demostración.

Columnas: `País`, `Título de la Norma`, `Tipo de norma`, `Ámbito`, `Vínculo con Educación`, `Dominio` (variable objetivo), `Fecha de Publicación`, `URL Oficial`, `Observaciones`, `Investigador`, y tres columnas auxiliares del observatorio.

## 2. Variable objetivo

| Dimensión | n | % |
|---|---:|---:|
| Pedagógico | 294 | 36.2 |
| Ética | 265 | 32.6 |
| Administrativo | 120 | 14.8 |
| Protección de Datos | 71 | 8.7 |
| Técnico | 63 | 7.7 |

Razón de desbalance **4.7 : 1**. Piso aritmético (clase mayoritaria): exactitud 0.362, F1 macro 0.106. Las dos clases pequeñas (Técnico, Administrativo) son también las de léxico menos específico: comparten vocabulario con Pedagógico y Ética. Protección de Datos, en cambio, tiene vocabulario propio («datos personales», «privacidad», «titular»).

## 3. Calidad

* **Faltantes**: 6 celdas en 813 × 13 (0.06 %). Ninguna en `Título`, `Dominio` ni `País`.
* **Duplicados**: 14 grupos de normas que comparten `URL Oficial` (8 de ellos con dimensiones distintas: la misma norma codificada dos veces de forma diferente). Se conservan, pero se tratan como una unidad (`grupo_url`) en toda partición y validación cruzada.
* **Fechas**: solo el 58 % interpretables como fecha; la columna se excluye.
* **`Tipo de norma`**: 14 valores crudos → 11 normalizados (mayúsculas, tildes, sinónimos).
* **Correcciones expertas**: un pequeño conjunto de etiquetas revisadas con el equipo GENIAL se aplica en `genial_clf.data.apply_expert_fixes` (documentadas en el código).

## 4. Texto de entrada

El modelo ve `texto = título + ". " + observaciones`. Longitud: mediana **44 palabras**, media 50.9, mínimo 6, máximo 274; 26 atípicos por IQR y 10 textos con menos de 12 palabras. Vocabulario de 2 421 términos distintos tras limpieza. Es un problema de **texto corto**: más *features* TF-IDF que ejemplos, lo que anticipa el sobreajuste diagnosticado en la Semana 3.

**Idioma** (detección con *lingua* sobre el título, con el país como apoyo): español 712, portugués 76, inglés 25. Las observaciones están casi siempre en español (798) aunque el título no lo esté: los codificadores resumen en español.

## 5. Asociaciones y fuga de información

| Par | V de Cramér | Decisión |
|---|---:|---|
| `País` ↔ `Investigador` | **0.99** | cada investigador cubre países concretos: `Investigador` equivale a conocer el país y es dato personal → **excluida y anonimizada** |
| `Tipo de norma` ↔ `Dominio` | 0.36 | describe la forma jurídica, no el contenido → excluida como predictora (sirve de **línea base de metadatos**) |
| `Ámbito` ↔ `Dominio` | 0.24 | se usa solo en la auditoría de equidad |

**Fuga léxica**: la palabra clave de la dimensión aparece en el texto en el 62.6 % de las normas de Ética y el 73.2 % de las de Protección de Datos, pero también fuera de su clase (22.8 % y 6.9 %). Es vocabulario legítimo de la norma, no una columna filtrada: se conserva, y se verifica que los términos de mayor peso del modelo son léxico de la dimensión y no nombres de países ni fechas (`notebooks/03_modelado.ipynb`, §6).

## 6. Concentración geográfica

Cinco países concentran el 67 % de las normas (México 179, Chile 105, Argentina 100, Ecuador 81, Brasil 81); 146 normas (18 %) proceden de países con menos de 20 normas, en su mayoría del Caribe anglófono. Esta asimetría es la raíz de la brecha de equidad por jurisdicción (`docs/consideraciones_eticas.md`).

## 7. Particiones

| Conjunto | n | Origen |
|---|---:|---|
| Entrenamiento | 673 | resto del corpus |
| Pool *few-shot* | 15 | 3 por clase, textos de 25-90 palabras (para la rama LLM) |
| Prueba ciega | 125 | las normas del instrumento de codificación ciega del investigador A (119 emparejadas por título/URL + sus duplicados) |
| Normas nuevas | 23 | normas del instrumento que no están en la versión congelada (Caribe); solo para evaluación exploratoria |

Entrenamiento y prueba no comparten ningún `grupo_url`. Las etiquetas de la prueba se guardan en `data/splits/test_labels_hidden.csv` y **solo se leen en `src/evaluate.py` / cuaderno 05**. En la prueba hay 112 normas en español, 12 en portugués y 1 en inglés; por diseño del instrumento sobrerrepresenta países pequeños.

## 8. Figuras

`notebooks/figuras/01_faltantes.png` … `09_fuga_lexica.png` (EDA); `results/figures/18_confusion_cv_train.png` (modelado); `results/figures/12_fairness_grupos.png` (equidad).
