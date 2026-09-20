# Condiciones de uso de los datos

## Muestra pública (`data/samples/`)

`sample_corpus_40.csv` contiene 40 normas reales del corpus consolidado del proyecto GENIAL (Erasmus+), ocho por
dimensión, con sus 13 columnas. La columna `Investigador` está anonimizada (códigos C01-C10). Las normas son
documentos públicos y la columna `URL Oficial` remite a su fuente; la codificación en dimensiones es obra intelectual
del equipo investigador de GENIAL.

Se autoriza su uso con fines académicos y de reproducibilidad, con la condición de **atribuir al proyecto GENIAL
(Erasmus+)** en toda publicación derivada. No se autoriza su uso comercial ni su presentación como corpus completo.

## Corpus completo (813 normas)

El archivo `consolidado_limpio.xlsx` **no se redistribuye**. Su uso en este trabajo fue autorizado por escrito por la
coordinación del consorcio GENIAL al equipo del proyecto, con las condiciones de atribución y no redistribución.
El repositorio publica únicamente su hash SHA-256 (véase `config.yaml` y `data/README.md`) y estadísticas agregadas.
Quien desee acceder al corpus debe solicitarlo a la coordinación del proyecto GENIAL.

## Archivos derivados

Los archivos que el pipeline escribe en `data/processed/`, `data/splits/` y `results/` derivan del corpus completo
y quedan excluidos del control de versiones (`.gitignore`). No deben publicarse sin la autorización del consorcio.
