# Model Card · Clasificador de dimensión normativa GENIAL (RL + TF-IDF) · v0.3 (octubre de 2026)

Formato basado en Mitchell et al. (2019). Cifras tomadas de `results/metrics/sintesis_resultados.json`, `results/metrics/overfitting/sintesis_overfitting.json` y `results/metrics/fairness/fairness_grupos.csv`.

## 1. Detalles del modelo
- **Desarrolladores:** Jorge Javier Maldonado Mahauad y Melissa Figallo (Grupo 9, Maestría en IA, UEES), en colaboración con el proyecto GENIAL (Erasmus+). Tutora académica: Gladys María Villegas Rugel.
- **Tipo:** clasificador lineal multiclase (regresión logística, L2, `class_weight='balanced'`) sobre TF-IDF de 1-2 gramas (5 000 características; configuración regularizada recomendada tras la Semana 3: C ≤ 0.1, 2 000 características, `min_df = 2`). Rama de comparación: LLM *few-shot* (15 ejemplos, temperatura 0), pendiente de ejecución.
- **Versión / fecha:** v0.3, octubre de 2026. Repositorio: `github.com/jorgemaldonadom-cmd/genial-clasificador-normativa`. Licencia del código: MIT.
- **Entrada:** texto UTF-8 «título de la norma. observaciones del codificador» (mediana 44 palabras). **Salida:** dimensión predicha, segunda dimensión (top-2) y probabilidad máxima.

## 2. Uso previsto
- **Uso principal:** asistir a los codificadores del observatorio GENIAL proponiendo dos dimensiones candidatas con su probabilidad para cada norma sobre IA en educación superior de América Latina y el Caribe. La decisión final es humana (codificación asistida).
- **Usuarios previstos:** investigadores codificadores de GENIAL; coordinación del observatorio para estimar carga de trabajo.
- **Usos fuera de alcance:** publicar valores del Índice de Madurez Normativa (IMN) calculados solo con etiquetas automáticas; clasificar normas de otros dominios o idiomas sin reentrenar; tomar decisiones sobre personas; comparar o sancionar países sin revisión humana ni margen de incertidumbre.

## 3. Factores relevantes
- **Idioma del título** (es / pt / en), **región** (Hispanoamérica continental / Brasil / Caribe), **tamaño del país en el corpus**, **ámbito** (nacional / institucional), **dimensión** (cinco clases desbalanceadas 4.7:1), **codificador original** (confundido con el país: V de Cramér 0.99).

## 4. Métricas
- F1 macro (principal, por el desbalance), exactitud, top-2, κ de Cohen frente a la etiqueta de referencia; intervalos de confianza de Wilson y *bootstrap*; McNemar frente a líneas base; desagregación por factor.

## 5. Datos de evaluación
- Validación cruzada de 5 pliegues estratificada por dimensión y por grupo de URL (n = 813). Conjunto ciego: las 125 normas leídas por un investigador de GENIAL con el instrumento de codificación ciega; etiquetas leídas una sola vez.

## 6. Datos de entrenamiento
- Corpus consolidado GENIAL, versión congelada (SHA-256 `f9797147…3f33`), 813 normas de 28 jurisdicciones; véase `docs/datasheet_corpus.md`.

## 7. Resultados cuantitativos
| Métrica | CV por grupo (n = 813) | Ciego (n = 125) |
|---|---|---|
| F1 macro | 0.587 [0.549, 0.623] | 0.617 [0.505, 0.712] |
| Exactitud | 0.635 [0.601, 0.667] | 0.688 [0.602, 0.763] |
| Top-2 | 0.852 | 0.872 |
| κ vs referencia | 0.50 | 0.56 |
| F1 por clase | Ped 0.71 · Ética 0.67 · Adm 0.41 · PD 0.80 · Téc 0.35 | — |

Desagregado (exactitud, CV): es 0.66 · pt 0.47 · en 0.36; Hispanoamérica continental 0.67 · Brasil 0.47 · Caribe 0.38; institucional 0.67 · nacional 0.55; países < 20 normas 0.56. Normas nuevas del Caribe (n = 23, fuera de la versión congelada): 0.39.

## 8. Consideraciones éticas
- El modelo rinde peor en normas en portugués e inglés y en el Caribe: sin salvaguardas, el IMN de esas jurisdicciones quedaría sistemáticamente distorsionado. Regla de gobernanza: revisión humana obligatoria para países con < 20 normas e idiomas distintos del español (incluido el portugués desde v0.3) y para toda predicción con probabilidad < 0.50.
- Los errores se concentran en normas que el codificador humano marcó como multidimensionales o de confianza media: la etiqueta única es una simplificación del framework y el modelo la hereda.
- El corpus contiene el nombre del codificador de cada norma (dato personal): se excluye del modelo, se anonimiza en toda publicación y no se envía al proveedor de LLM.

## 9. Advertencias y recomendaciones
- No usar como sustituto del codificador: la automatización con ≥ 85 % de exactitud exigiría revisar ~84 % de las normas.
- Reentrenar y volver a auditar por grupo en cada actualización del corpus; publicar esta tarjeta actualizada con cada versión.
- Toda cifra del IMN debe acompañarse del porcentaje de normas con respaldo humano y de un margen de incertidumbre.
