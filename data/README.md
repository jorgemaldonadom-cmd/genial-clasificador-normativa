# Datos

## Procedencia

Corpus consolidado del observatorio de normativa sobre IA en educación superior del proyecto **GENIAL (Erasmus+)**:
813 normas de 27 países de América Latina y el Caribe (28 jurisdicciones incluyendo Puerto Rico), codificadas por
diez investigadores del consorcio en cinco dimensiones. Construido entre 2024 y 2026. No publicado. Uso autorizado
por escrito al equipo del proyecto (Sprint 1). Condiciones en `../DATA_LICENSE.md`.

## Versión congelada

| Archivo | Hoja | Filas × columnas | SHA-256 |
|---|---|---|---|
| `consolidado_limpio.xlsx` | `Datos limpios` | 813 × 13 | `f9797147e7026f796d453fa84d09c4bf32c48ae75c3744912806a72db3ff7f33` |

Colóquelo en `data/raw/`. El pipeline (`scripts/run_pipeline.py prepare`) verifica el hash y se detiene si no coincide.

## Distribución de la variable objetivo (`Dominio`)

| Dimensión | n | % |
|---|---:|---:|
| Pedagógico | 294 | 36.2 |
| Ética | 265 | 32.6 |
| Administrativo | 120 | 14.8 |
| Protección de Datos | 71 | 8.7 |
| Técnico | 63 | 7.7 |

Razón de desbalance 4.7:1. Piso aritmético (clase mayoritaria): 36.2 % de exactitud, 0.106 de F1 macro.
Cinco países concentran el 67 % de las normas (México 179, Chile 105, Argentina 100, Ecuador 81, Brasil 81).

## Diccionario de datos

| Columna | Tipo | Contenido | Uso en el proyecto |
|---|---|---|---|
| Investigador | categórica (10) | investigador que codificó la norma | análisis de sesgo de codificador (agregado); **excluida** como predictora; anonimizada en la muestra |
| País | categórica (28) | país o jurisdicción | estratificación, desagregación, regla de revisión obligatoria |
| Título de la Norma | texto | título oficial o descriptivo | **texto de entrada** |
| Tipo de norma | categórica (14 → 11 normalizadas) | Ley, Decreto, Estrategia Nacional, Reglamento, Resolución rectoral, Lineamiento Académico, Guía ética, Guía Pedagógica, Libro blanco, Otro… | línea base de metadata; **excluida** como predictora (fuga) |
| Estado | categórica (3) | Vigente / En proyecto-borrador / Derogada | caracterización |
| Organismo Emisor/Universidad | texto | entidad emisora | caracterización |
| Dominio | categórica (5) | dimensión GENIAL | **variable objetivo** |
| Vínculo con Educación | categórica (2) | Directo / Indirecto | caracterización, análisis de errores |
| Dedicación del Texto | categórica (5) | Articulado completo / Sección-Capítulo / Mención breve / Otro | caracterización |
| Fecha de Publicación | texto heterogéneo | fechas en formatos mixtos, ≈25 % «N/D», «s/f» | **excluida**; heterogeneidad documentada |
| URL Oficial | URL | fuente pública de la norma | deduplicación antes de particionar; trazabilidad |
| Observaciones | texto | resumen del contenido redactado por el codificador | **texto de entrada** |
| Ámbito | categórica (2) | Institucional / Nacional | caracterización, análisis de errores |

Variables derivadas por el pipeline: `texto` (Título + Observaciones), `n_palabras`, `idioma` (es/pt/en/fr), `grupo_url`, `id_norma`.

## Muestra pública

`samples/sample_corpus_40.csv`: 40 normas reales (8 por dimensión) con las 13 columnas, `Investigador` anonimizado,
seleccionadas con semilla fija. Sirve para ejecutar las pruebas unitarias y entender el esquema; **no** es
representativa de la distribución de clases del corpus (está balanceada a propósito).

`samples/fewshot_examples_template.csv`: esquema del pool few-shot que produce `split.py`.
