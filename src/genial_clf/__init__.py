"""genial_clf — clasificación automática de normativa sobre IA en educación superior (corpus GENIAL).

Módulos:
  data        carga, verificación de hash, limpieza, idioma, deduplicación, campo de texto
  split       muestreo estratificado por dimensión e idioma; archivos ciegos
  baselines   clase mayoritaria y línea base de metadata
  classic     TF-IDF + Regresión Logística, validación cruzada, top-2
  llm         prompt few-shot, llamada a la API, parseo de top-2, repeticiones
  evaluate    F1, matriz de confusión, McNemar, IC de Wilson y bootstrap, kappa
  reliability kappa humano-humano, adjudicación, tipificación de errores
  imn         umbral de confianza, enrutamiento y agregación al IMN con trazabilidad
"""
__version__ = "0.1.0"
