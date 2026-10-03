# Datasheet · Corpus GENIAL de normativa sobre IA en educación superior (versión congelada, septiembre de 2026)

Formato basado en Gebru et al. (2021). Complementa `data/README.md` y `DATA_LICENSE.md`.

## Motivación
- **¿Para qué se creó?** Para el observatorio regional del proyecto GENIAL (Erasmus+): inventariar y caracterizar la normativa sobre inteligencia artificial en la educación superior de América Latina y el Caribe y alimentar un Índice de Madurez Normativa (IMN) por país.
- **¿Quién lo creó y financió?** Diez investigadores del consorcio GENIAL, coordinados desde la Universidad de Cuenca para Ecuador y desde la Université de Toulouse a nivel general; financiación Erasmus+ (Unión Europea).

## Composición
- 813 registros; cada uno es un instrumento normativo (ley, estrategia nacional, reglamento, política institucional, guía) de 28 jurisdicciones (27 países y Puerto Rico). Cinco países concentran el 66 % (México 179, Chile 105, Argentina 100, Brasil 81, Ecuador 81); 12 jurisdicciones tienen menos de 20 normas.
- 13 columnas: Investigador, País, Título de la Norma, Tipo de norma, Estado, Organismo Emisor/Universidad, Dominio (variable objetivo: Pedagógico 294, Ética 265, Administrativo 120, Protección de Datos 71, Técnico 63), Vínculo con Educación, Dedicación del Texto, Ámbito, Fecha de Publicación, Observaciones, URL Oficial.
- **Datos personales:** la columna `Investigador` contiene el nombre del codificador (dato personal de 10 personas identificables); el resto describe documentos públicos. No hay datos de estudiantes ni de docentes.
- **Errores y ruido conocidos:** 1 valor faltante de Dominio (corregido con criterio experto documentado); 14 grupos de URL repetida (30 normas, 8 con dimensiones distintas); 42 % de fechas no interpretables; `Tipo de norma` con 14 variantes que se normalizan a 11; textos muy cortos (10 con menos de 12 palabras).
- **Etiqueta:** una sola dimensión por norma asignada por un único codificador; fiabilidad intercodificador no medida (κ humano-humano pendiente). El 84 % de las normas releídas por un segundo investigador recibieron etiquetas secundarias: la etiqueta única es una simplificación.

## Proceso de recolección
- Búsqueda documental por país (fuentes oficiales, portales legislativos, sitios institucionales), 2025-2026. Cada norma fue leída y codificada por un investigador; las `Observaciones` son resúmenes redactados por el codificador en español, incluso para normas en portugués o inglés. Cada país fue cubierto por un solo investigador: codificador y país están confundidos (V de Cramér 0.99).
- La versión usada en este proyecto está congelada con hash SHA-256 `f9797147e7026f796d453fa84d09c4bf32c48ae75c3744912806a72db3ff7f33`; GENIAL ha incorporado normas posteriores (p. ej., 23 del Caribe) que no están en ella.

## Preprocesamiento
- Normalización de `Tipo de norma`, texto = título + observaciones, detección de idioma sobre el título, marcado de duplicados por URL, identificador `id_norma`. `Tipo de norma`, `Investigador` y `Fecha` se excluyen como predictores por diseño.

## Usos
- Uso actual: entrenar y evaluar el clasificador de dimensión y el flujo asistido del observatorio; análisis exploratorio; auditoría de equidad por jurisdicción.
- **Usos que deben evitarse:** inferir calidad o rendimiento de los codificadores a partir del modelo (el efecto codificador está confundido con el país); publicar el corpus completo; usar el conteo por dimensión como medida de «calidad» de la política pública de un país sin contexto.

## Distribución
- El corpus completo **no se redistribuye** (autorización del consorcio GENIAL limitada al equipo). Se publican: el hash, las estadísticas agregadas, una muestra anonimizada de 40 normas (`data/samples/`) y los resultados agregados sin texto de normas. Licencia de la muestra: uso académico con atribución.

## Mantenimiento
- Responsable de la versión congelada: coordinación Ecuador de GENIAL (J. Maldonado). Las actualizaciones del corpus generan una nueva versión con nuevo hash, reentrenamiento y nueva auditoría por grupo; los codificadores pueden solicitar la corrección o eliminación de su nombre en cualquier copia.
