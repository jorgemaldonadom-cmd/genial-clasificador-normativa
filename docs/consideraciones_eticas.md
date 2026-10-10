# Consideraciones éticas

Este documento sintetiza el workshop de impacto social y responsabilidad (Semana 4; informe completo en `results/reports/semana4/Grupo_9_S4_Impacto_Social_y_Responsabilidad.pdf`), la *model card* (`docs/model_card.md`), la *datasheet* del corpus (`docs/datasheet_corpus.md`) y la auditoría de equidad (`scripts/fairness_grupos.py`, `results/metrics/fairness/`). El sistema clasifica **normas** (documentos públicos), no personas; aun así, sus salidas afectan a jurisdicciones, a codificadores y a la credibilidad de un índice regional.

## 1. Sesgos identificados en los datos y el modelo

| Sesgo | Origen | Evidencia | Efecto |
|---|---|---|---|
| **Representación geográfica** | 5 países concentran el 67 % de las normas; 146 normas (18 %) vienen de países con < 20 normas, sobre todo del Caribe | exactitud (CV por grupo): Hispanoamérica continental 0.67 · Brasil 0.47 · Caribe 0.38 | el modelo aprende el léxico normativo de México, Chile, Argentina y Ecuador |
| **Idioma** | 712 normas en español, 76 en portugués, 25 en inglés | exactitud es 0.66 · pt 0.47 · en 0.36 | peor cobertura justo donde el corpus es más pobre |
| **Desbalance de clases** | Pedagógico 36 % vs Técnico 7.7 % | F1 por clase: Ped 0.71 · Ética 0.67 · Adm 0.41 · PD 0.80 · Téc 0.35 (CV) | sin ponderación, Técnico casi no se predice (F1 0.03) |
| **Etiqueta única en normas multidimensionales** | el instrumento asigna una dimensión aunque la norma trate varias | 35 de 37 errores ciegos revisados eran normas multidimensionales; exactitud 0.76 cuando el codificador tuvo confianza *Alta* vs 0.41 con *Media* | parte del «error del modelo» es desacuerdo legítimo |
| **Codificador** | diez codificadores con criterios no idénticos; V de Cramér País ↔ Investigador 0.99 | exactitud por codificador anonimizado entre 0.29 y 0.71 | el modelo hereda la heterogeneidad de criterio; el κ humano-humano está pendiente |

Mitigaciones aplicadas: `class_weight = balanced`; validación cruzada por grupo de URL; exclusión de `Investigador`, `Tipo de norma` y `Fecha`; top-2 como salida; conjunto ciego exógeno; revisión humana obligatoria (sección 2).

## 2. Equidad (*fairness*)

**Sujeto de la equidad**: la jurisdicción. El riesgo no es un trato desigual a personas sino que países pequeños o no hispanohablantes reciban una codificación automática peor y, con ella, un Índice de Madurez Normativa distorsionado.

Auditoría por subgrupo (predicciones fuera de muestra, n = 813, IC de Wilson, razón frente al grupo de referencia; regla de los cuatro quintos como referencia orientativa):

| Dimensión | Grupo peor (exactitud) | Grupo mejor | Razón mín. | ¿Cumple 4/5? |
|---|---|---|---:|---|
| Idioma del título | en (0.36) | es (0.66) | 0.54 | **no** |
| Región | Caribe (0.38) | Hispanoamérica (0.67) | 0.56 | **no** |
| Tamaño del país | < 20 normas (0.56) | 20-49 normas (0.66) | 0.86 | sí |
| Ámbito | Nacional (0.55) | Institucional (0.67) | 0.82 | sí |
| Codificador (anonimizado) | Codificador I (0.29) | Codificador B (0.71) | 0.43 | **no** |

La brecha **no se corrige con el modelo** (no hay datos suficientes en esos grupos); se **contiene** con la regla de gobernanza: va a revisión humana toda norma en inglés, francés o portugués, de país con menos de 20 normas en el corpus, o con probabilidad por debajo del umbral calibrado del modelo. Con ella, el 100 % de las normas del Caribe, de Brasil y de los países pequeños llevan respaldo humano. El coste de la salvaguarda se paga en horas humanas precisamente donde el corpus es más pobre; es un límite declarado, no un efecto colateral.

Compromiso: repetir la auditoría con cada versión del modelo o del corpus y **no publicar ningún valor del IMN** cuyo grupo no pase la revisión humana obligatoria.

## 3. Privacidad y protección de datos

* El corpus contiene el **nombre del codificador** de cada norma (dato personal identificable). Se excluye del modelo, se anonimiza (Codificador A-J) en todo artefacto del repositorio y no se envía a ningún proveedor de LLM.
* El corpus consolidado y el instrumento de codificación ciega **no se versionan**: en el repositorio solo hay su hash SHA-256, agregados, los modelos entrenados y una muestra de 40 normas (documentos públicos) sin la columna de codificador. Las predicciones que contienen texto de normas (`pred_*.csv`, predicciones de validación cruzada) están en `.gitignore`.
* Las claves de API se leen de variables de entorno o de archivos ignorados por Git (`data/raw/anthropic_key.txt`, `.env`); los cuadernos nunca las contienen.
* Marco normativo: no hay decisiones automatizadas sobre personas (art. 22 RGPD no aplica); el tratamiento de los nombres de codificadores en un consorcio multinacional debe documentarse bajo RGPD (coordinación en Francia), LOPDP (Ecuador) y LGPD (Brasil): principios de finalidad, minimización y seguridad. Es el único punto que requiere confirmación jurídica antes de operar el flujo en el observatorio.
* La aplicación no almacena las normas introducidas: el historial vive en la sesión del navegador y desaparece al cerrarla.

## 4. Transparencia y explicabilidad

* Modelo lineal: cada predicción se explica con los **términos que más pesan** (coeficiente × TF-IDF), visibles en la aplicación.
* Se muestran siempre la **probabilidad**, la **segunda dimensión candidata** y los **motivos** por los que una norma va a revisión humana.
* `docs/model_card.md` (Mitchell et al., 2019) y `docs/datasheet_corpus.md` (Gebru et al., 2021) documentan uso previsto, factores, métricas desagregadas y límites; los metadatos de cada modelo (`models/*.json`) registran hiperparámetros, hash del corpus, validación cruzada y umbral calibrado.
* Todas las cifras del proyecto se reportan con intervalos de confianza y pruebas pareadas, incluidas las que no alcanzan la meta.

## 5. Impacto social

**Positivo.** Un observatorio sostenible: el modelo reduce el espacio de decisión de cinco a dos opciones en el 85-87 % de las normas (top-2) y permite estimar la carga residual de cada actualización (hoy, 21 de 136 horas-persona por pasada sobre el corpus completo). Beneficiarios: codificadores y coordinación de GENIAL; a mediano plazo, los países, porque un índice que se actualiza orienta política pública y cooperación.

**Riesgos.** (a) Un IMN calculado con etiquetas automáticas sin respaldo humano subestimaría la madurez de las jurisdicciones pequeñas y no hispanohablantes; (b) uso del modelo o de sus métricas para evaluar a codificadores individuales; (c) lectura de la dimensión asignada como juicio sobre la calidad de la norma, cuando solo describe su tema principal. Las tres se tratan con la regla de gobernanza, el anonimato y la documentación de uso previsto.

**Partes interesadas.** Codificadores de GENIAL (usuarios directos), coordinación del observatorio, ministerios y universidades de la región (usuarios del IMN), comunidad académica (reproducibilidad), financiador (Erasmus+).

## 6. Responsabilidad y rendición de cuentas

* **Decisión final humana.** El sistema es de *codificación asistida*: propone, no decide; el codificador conserva la etiqueta final y el tramo automático solo se contempla para países bien representados y con auditoría vigente.
* **Trazabilidad.** Semilla, hash del corpus, versión de cada modelo y umbral calibrado se registran; los cuadernos se versionan ejecutados.
* **Responsables.** Los autores (Grupo 9) del prototipo y su documentación; la coordinación de GENIAL de cualquier uso operativo y de la base jurídica del tratamiento de datos.
* **Declaración de compromiso** (Semana 4): publicar la auditoría de equidad con cada versión; acompañar cada cifra del IMN del porcentaje de normas con respaldo humano y su margen de incertidumbre; no evaluar codificadores con el modelo; congelar y corregir con registro público ante un daño identificado; mantener transparencia sobre los umbrales no alcanzados.

## 7. Uso dual y mal uso

El clasificador tiene un potencial de daño bajo: no produce contenido, no decide sobre personas y su dominio es documental. Los usos indebidos plausibles son de **sobreconfianza**, no de arma: (a) publicar rankings de países con etiquetas no revisadas; (b) reutilizarlo en otros dominios (salud, justicia) o idiomas sin reentrenar ni auditar; (c) presentar la dimensión como evaluación de calidad normativa. Clasificación de riesgo bajo el AI Act de la UE: **riesgo mínimo** (no está en los casos del Anexo III); se aplican de todos modos las obligaciones de transparencia por buena práctica. Salvaguardas: uso previsto explícito en la *model card*, licencia MIT con descargo de garantía, y la regla de gobernanza en la configuración por defecto de la aplicación.

## 8. Limitaciones

1. **Tamaño y origen de la prueba ciega**: n = 125, intervalos de ± 9 pp; sobrerrepresenta países pequeños por diseño del instrumento.
2. **Etiqueta única**: la tarea real es multi-etiqueta; recomendamos a GENIAL registrar dimensión principal y secundaria.
3. **κ humano-humano sobre `Dominio` pendiente**: el instrumento del investigador codifica D1-D6; falta una segunda codificación ciega de la misma dimensión para fijar el techo humano.
4. **Rama LLM no ejecutada en esta versión** (clave de API no disponible en el entorno de entrega); el código y el *prompt* versionado permiten ejecutarla.
5. **Umbral y cobertura**: con ≥ 85 % de exactitud automática solo el 14-15 % de las normas queda en el tramo automático; el ahorro de horas es modesto hasta que mejore el corpus.
6. **Deriva**: la normativa sobre IA cambia rápido; el modelo debe reentrenarse y reauditarse con cada actualización del corpus.
