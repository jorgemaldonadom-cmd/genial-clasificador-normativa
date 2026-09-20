# Protocolo de codificación ciega (Sprint 3, Semana 6)

**Objetivo.** Medir el acuerdo humano-humano (κ de Cohen) sobre el conjunto de prueba, para que el desempeño de
ambas ramas se interprete contra la fiabilidad del propio estándar de referencia.

## Participantes
- **Segundo codificador**: investigador del consorcio GENIAL distinto del que codificó originalmente cada norma.
- **Tercer codificador**: adjudica las discrepancias entre la etiqueta original y la del segundo codificador.
- **Equipo del proyecto**: prepara el cuadernillo, custodia las etiquetas y calcula κ. No codifica.

## Regla de gold standard (fijada ANTES de medir)
1. Si la etiqueta original y la del segundo codificador coinciden, esa es la etiqueta de referencia.
2. Si discrepan, el tercer codificador decide entre las dos (no puede proponer una tercera dimensión).
3. Se registra el número y la lista de casos adjudicados; forman parte del reporte.

## Ceguera
- El cuadernillo (`test_blind.csv`) contiene únicamente id, país, idioma y texto. **No** contiene la etiqueta
  original ni ninguna predicción de los modelos.
- Los archivos `pred_clasico.csv` y `pred_llm.csv` no se comparten con los codificadores hasta cerrar la codificación.
- El segundo codificador no consulta al codificador original durante el ejercicio.

## Instrucciones al codificador
- Asigne **una** dimensión por norma según las definiciones del framework GENIAL (las mismas del prompt,
  `prompts/clasificador_v0.1.md`).
- Si duda entre dos, anote la segunda en la columna `Dimensión alternativa` y marque `Duda = 1`. Esa información
  alimenta el diagnóstico de multi-etiqueta, pero la columna `Dimensión` debe tener un único valor.
- Codifique la norma en su idioma original; no la traduzca.
- Tiempo estimado: 100-150 normas, ≈ 5 minutos por norma (una a dos jornadas).

## Salidas
- `data/splits/coder2_labels.csv`: `id_norma, Dimensión, Dimensión alternativa, Duda`.
- `data/splits/coder3_adjudication.csv`: `id_norma, Dimensión` para los casos discrepantes.
- `results/fiabilidad.md`: κ humano-humano, κ de cada rama frente a la referencia consolidada, interpretación
  (Landis y Koch), casos adjudicados y hoja de tipificación de errores.
