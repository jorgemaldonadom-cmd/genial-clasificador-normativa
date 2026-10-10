# Historial de versiones

## v1.0.0 — octubre de 2026 (proyecto final)
- Estructura final del repositorio: `src/` (data_processing, model, train, evaluate, utils), `models/`, `app/`, `tests/`, `results/{figures,metrics,reports}`, `docs/`.
- Optimización de hiperparámetros con Optuna y análisis de sensibilidad (Workshop S5); `best_model` elegido por la regla 1-DE con menor sobreajuste.
- Umbral de confianza calibrado por modelo (0.50 `model_v1`, 0.29 `best_model`).
- Aplicación Streamlit (una norma, lote CSV, acerca del modelo, historial; es/en) con 14 pruebas automatizadas e integración continua.
- Cuadernos 00 (maestro paso a paso) y 01-05 ejecutados; documentación `docs/*.md` de seis apartados y README de catorce secciones.

## v0.3 — 3 de octubre de 2026 (Semana 4)
- Workshop de impacto social y responsabilidad; auditoría de equidad por jurisdicción; model card y datasheet; `pt` añadido a los idiomas de revisión obligatoria.

## v0.2 — septiembre de 2026 (Semana 3 / Sprint 3)
- Experimento completo: preparación, líneas base, rama clásica, prueba ciega con el instrumento del investigador A, especificación operativa.
- Diagnóstico de overfitting/underfitting (curvas por época en PyTorch y SGD, curvas de validación, estrategias de mitigación).

## v0.1 — septiembre de 2026 (Sprint 1-2)
- Planificación, EDA, comparación de ocho técnicas, Ficha de Decisión Técnica, documento técnico, repositorio inicial.
