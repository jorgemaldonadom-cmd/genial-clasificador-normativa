# Decisiones metodológicas del Proyecto Integrador — v1.0

**Proyecto:** Clasificación automática de normativa sobre IA en educación superior de ALC (corpus GENIAL)
**Equipo:** Jorge Maldonado · Melissa Figallo
**Origen:** retroalimentación docente al foro de la Semana 1
**Fecha:** 16 de septiembre de 2026 (revisión de redacción: 20 de septiembre de 2026)
**Estado:** documento vinculante — rige la matriz SMART y las Unidades 2, 3 y 4

---

## 0. El comentario y su lectura

La retroalimentación valida la elección del Problema 1 y reubica el riesgo:

> "aquí el riesgo no está en conseguir datos sino en el diseño metodológico fino (multi-etiqueta vs. etiqueta única, balance de clases, tratamiento del desacuerdo intercodificador, idioma). Sugeriría que esos cuatro puntos queden resueltos explícitamente en la matriz SMART antes de iniciar el desarrollo."

Los cuatro puntos no son observaciones sueltas: los cuatro atacan el mismo punto de apoyo del proyecto —**la validez de la etiqueta de referencia y de la métrica que se reporta contra ella**. Si el *gold standard* es un artefacto de cómo codifica un equipo humano, y no una propiedad de la norma, entonces ningún valor de F1 macro que obtengamos significará nada sin un referente.

**Principio rector que se adopta:** ninguna métrica de desempeño se reporta sin declarar (a) contra qué etiqueta de referencia se mide, (b) cuál es la fiabilidad de esa etiqueta, y (c) sobre qué distribución de datos. Tres preguntas distintas exigen tres diseños distintos, y no se mezclan en un solo número.

---

## Decisión 1 — Etiqueta única vs. multi-etiqueta

**El punto.** El caso de negocio afirma que `Dominio` es *single-label* porque no se hallaron separadores múltiples en ningún registro. Esa es una verificación **operativa** del archivo, no una verificación **conceptual** del objeto. Una ley nacional de IA regula simultáneamente ética, protección de datos y aspectos administrativos; que el codificador haya escrito un solo valor refleja una restricción del instrumento de codificación de GENIAL, no la naturaleza de la norma.

**Consecuencia que hay que declarar:** si el objeto es intrínsecamente multi-etiqueta y el *gold standard* fuerza una sola, entonces una fracción del "error" del modelo consiste en elegir una dimensión secundaria legítima. La frontera Administrativo–Ética, discutible dentro del propio framework, es la candidata más probable a este fenómeno.

**Decisión.** Se mantiene la formulación *single-label* para el experimento principal —es lo único que el corpus etiquetado soporta— y se incorpora un **diagnóstico de multi-etiquetado latente** como análisis complementario, no como cambio de tarea.

**Evidencia requerida:**

1. Registrar el **top-2** de predicciones en ambas ramas (`predict_proba` en la clásica; solicitud explícita de primera y segunda dimensión en el *prompt* del LLM) y reportar *top-2 accuracy* junto a la exactitud simple.
2. Revisión cualitativa de los casos mal clasificados del conjunto de prueba: clasificar cada error como (a) error genuino, (b) dimensión secundaria defendible, (c) etiqueta original discutible. Reportar los tres conteos.
3. Si *top-2 accuracy* supera de forma clara a la exactitud simple, se documenta como hallazgo: **la tarea real de GENIAL no es asignar una dimensión, es proponer una lista corta ordenada para validación humana** — lo que además modifica el diseño de la solución asistida en un sentido más útil para el equipo.

**Dónde se declara:** matriz SMART, criterio *Específico* (definición exacta de la tarea) y *Medible* (métrica primaria + métrica complementaria). Unidad 2, sección 2.3.3. Unidad 3, análisis de errores.

---

## Decisión 2 — Balance de clases y validez de la comparación

**El punto.** Hay dos problemas distintos bajo la misma etiqueta y conviene no confundirlos.

*(a) Distribución.* El corpus tiene razón de desbalance 4.7:1 (Pedagógico 294 → Técnico 63). Un conjunto de prueba balanceado por construcción (igual número por dimensión) es correcto para comparar técnicas de forma limpia, pero **no estima el desempeño operativo** sobre normas entrantes, que llegan con la distribución natural. Hay que marcar esa distinción desde el diseño y no reportar un solo número como si respondiera a ambas preguntas.

*(b) Potencia estadística.* Con un conjunto de prueba pequeño (por ejemplo, n = 50), el intervalo de confianza de una exactitud cercana a 0.65 ronda ±0.13, de modo que una diferencia de varios puntos de F1 macro entre técnicas **puede no ser estadísticamente significativa**. Es el punto más atacable de cualquier comparación de este tipo y conviene resolverlo desde el diseño, antes de que lo señale un tribunal.

**Decisión.** Se separan explícitamente dos preguntas de evaluación, cada una con su diseño:

| Pregunta | Diseño | Métrica | Uso |
|---|---|---|---|
| ¿Qué técnica clasifica mejor? | Conjunto de prueba balanceado, pareado | F1 macro + prueba de McNemar | Decisión de arquitectura |
| ¿Qué rendimiento tendría en operación? | Validación cruzada sobre distribución natural | F1 macro, F1 por clase, F1 *weighted* | Estimación para el flujo GENIAL |

**Evidencia requerida:**

1. **Prueba de McNemar** sobre las predicciones pareadas de ambos modelos en el mismo conjunto de prueba (diseño pareado → McNemar es la prueba correcta, no una prueba de proporciones independientes). Reportar el valor *p*.
2. Intervalos de confianza (Wilson) para exactitud y *bootstrap* para F1 macro.
3. **Ampliar el conjunto de prueba** a 100–150 casos. La inferencia del LLM tiene costo marginal bajo y entrenar el modelo clásico excluyendo esos casos es trivial; no hay razón para un conjunto de prueba menor.
4. Mantener `class_weight='balanced'` y documentarlo como decisión, comparándolo contra la línea base sin ponderación.

**Reencuadre del resultado principal.** Si McNemar no alcanza significancia, la afirmación honesta y más fuerte no es "el LLM supera al modelo clásico", sino: **el LLM alcanza desempeño equivalente o superior usando ~50 veces menos ejemplos etiquetados**. Ese es un resultado sobre *eficiencia de datos*, que es defendible con n pequeño y que además es el que le importa a GENIAL, porque implica que el observatorio no necesita invertir en etiquetar miles de normas para automatizar.

**Dónde se declara:** matriz SMART, criterio *Medible*. Unidad 3, secciones 3.1 y 3.3.

---

## Decisión 3 — Desacuerdo intercodificador y techo de desempeño

**El punto.** El caso de negocio ya admite la limitación: cada norma tiene la codificación de **un solo investigador**, sin doble codificación ni consenso. El modelo aprende a imitar a un codificador individual. La consecuencia es más severa de lo que el documento reconoce: **ningún clasificador puede superar la fiabilidad de su propia etiqueta de referencia**. Sin medir el acuerdo humano–humano, cualquier F1 macro carece de referente: no sabremos si es mediocre o si está en el techo.

**Esta es la adición de mayor rendimiento del proyecto entero**, y GENIAL ya tiene el instrumento construido (protocolo de fiabilidad intercodificador v1.0, con κ de Cohen, κ de Fleiss y α de Krippendorff).

**Decisión.** Se incorpora un **ejercicio de doble codificación humana** sobre el mismo conjunto de prueba, y el desempeño del modelo se reporta **relativo al acuerdo humano**, no en abstracto.

**Evidencia requerida:**

1. Un segundo codificador —investigador de GENIAL distinto del original— codifica a ciegas el conjunto de prueba completo. Calcular **κ de Cohen humano–humano**.
2. Calcular **κ de Cohen modelo–humano** con la misma métrica, para ambas ramas. La comparación de los tres valores es el resultado interpretable: el modelo se posiciona como "segundo codificador" dentro del mismo marco de fiabilidad que GENIAL ya usa para sus auditorías.
3. **Regla de *gold standard* fijada antes de medir** (no después): ante discrepancia entre codificadores, la etiqueta de referencia se resuelve por adjudicación de un tercer codificador. Documentar la regla y el número de casos adjudicados.
4. En el análisis de errores, separar **error del modelo** de **error de la etiqueta original**. Reportar cuántos de los casos "mal clasificados" son en realidad casos donde el modelo coincide con el segundo codificador y no con el primero. Ese conteo es un resultado legítimo y probablemente el más interesante del trabajo.

**Costo:** entre 60 y 100 normas de codificación humana adicional. Es el único elemento de este documento que exige gestión con terceros, y por eso se adelanta al cronograma.

**Dónde se declara:** matriz SMART, criterios *Medible* (referente de la métrica) y *Alcanzable* (dependencia de un segundo codificador). Unidad 2, supuestos. Unidad 3, secciones 3.1 y 3.4. Unidad 4, trazabilidad.

---

## Decisión 4 — Idioma del corpus

**El punto.** El corpus cubre 27 países de América Latina **y el Caribe**. No es un corpus monolingüe: el Caribe anglófono (Trinidad y Tobago, San Cristóbal y Nieves, Jamaica, Barbados, Guyana, Belice) aporta normas en inglés, Brasil en portugués, y Haití en francés. El caso de negocio no registra la variable idioma en ninguna parte.

**Consecuencia técnica, y es grande.** TF-IDF opera sobre formas superficiales: una norma en inglés **no comparte vocabulario** con el conjunto de entrenamiento en español. Para la rama clásica, las normas no hispanohablantes son prácticamente inclasificables, salvo que haya suficientes ejemplos de cada idioma en entrenamiento —y no los hay. El LLM, en cambio, es nativamente multilingüe.

**La hipótesis que esto abre es la más valiosa del proyecto:** buena parte de la ventaja del LLM puede no deberse a "los LLM son mejores", sino a que **el LLM transfiere entre idiomas y TF-IDF no**. Es una explicación mucho más precisa, es contrastable con los datos que ya existen, y justifica la elección arquitectónica para un observatorio regional por una razón estructural y no por una diferencia de puntos porcentuales.

**Decisión.** El idioma se incorpora como variable de análisis de primera clase, no como nota al pie.

**Evidencia requerida:**

1. Añadir la variable `idioma` al corpus (derivada de país + detección automática sobre el texto, con verificación manual de los casos no hispanohablantes, que son pocos). Reportar la distribución en la caracterización del corpus.
2. **Desagregar todas las métricas por idioma.** Si la ventaja del LLM se concentra en las normas no hispanohablantes, se reporta como hallazgo principal, con implicación directa de diseño.
3. Garantizar presencia de normas no hispanohablantes en el conjunto de prueba ampliado (muestreo estratificado por dimensión **e** idioma).
4. Declarar la política de idioma del *prompt*: definiciones de las dimensiones GENIAL en español, textos de las normas en su idioma original, sin traducción previa. Documentar la decisión y su justificación.

**Vínculo con el sesgo ya identificado.** El sesgo por país y el sesgo por idioma **son el mismo sesgo**: los países subrepresentados del corpus (Trinidad y Tobago = 1, San Cristóbal y Nieves = 2) son precisamente los del Caribe anglófono. Esto refuerza —y ahora fundamenta técnicamente— la recomendación de gobernanza de la Unidad 4: revisión humana obligatoria para normas de países subrepresentados, que ahora se enuncia como **revisión humana obligatoria para normas en idioma minoritario del corpus**.

**Dónde se declara:** matriz SMART, criterios *Específico* (alcance del corpus) y *Alcanzable* (viabilidad multilingüe). Unidad 2, secciones 2.3.1 y 2.3.4. Unidad 3, sección 3.2.3. Unidad 4, sección 4.2.

---

## Impacto en la matriz SMART

Los cuatro puntos se concentran en dos criterios, y conviene que la justificación escrita de esos criterios los cite de forma explícita:

| Criterio | Peso | Qué debe resolver en su justificación |
|---|---|---|
| Específico | 20 % | Tarea *single-label* con diagnóstico multi-etiqueta (D1); alcance multilingüe declarado (D4) |
| **Medible** | **20 %** | **Doble métrica balanceada/natural + McNemar e IC (D2); κ humano–humano como referente (D3); métricas desagregadas por idioma (D4)** |
| **Alcanzable** | **25 %** | **Dependencia del segundo codificador y su costo (D3); viabilidad de la rama clásica en corpus multilingüe (D4)** |
| Relevante | 20 % | El diagnóstico del framework (D1, D3) es aporte a GENIAL, no solo al curso |
| Temporal | 15 % | La doble codificación se adelanta por ser la única dependencia externa |

---

## Ajustes al cronograma

| Fase | Sem. | Adición derivada de este documento |
|---|---|---|
| 1. Definición | 1–2 | **Gestionar desde ahora** el segundo codificador con el equipo GENIAL (D3) — única dependencia externa |
| 2. Preparación de datos | 3 | Variable `idioma` y su distribución (D4); muestreo estratificado por dimensión e idioma (D2, D4) |
| 3. Técnica clásica | 4 | `predict_proba` para top-2 (D1); línea base sin ponderación de clases (D2) |
| 4. Técnica LLM | 5 | *Prompt* con primera y segunda dimensión (D1); política de idioma documentada (D4) |
| 5. Evaluación y cierre | 6–7 | McNemar e IC (D2); κ triple humano–humano / modelo–humano (D3); métricas por idioma (D4); tipificación cualitativa de errores (D1, D3) |

---

## Riesgo residual declarado

Si el segundo codificador no se consigue dentro de la ventana del curso, la Decisión 3 no puede ejecutarse. Plan de contingencia: doble codificación por parte del propio equipo del proyecto sobre una submuestra reducida (30 normas), reportada explícitamente como **acuerdo intraequipo y no como fiabilidad intercodificador del proyecto GENIAL**, con la limitación declarada en la Unidad 3 y el ejercicio completo trasladado a trabajo futuro.
