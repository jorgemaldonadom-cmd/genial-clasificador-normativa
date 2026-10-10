"""Aplicación Streamlit: clasificador asistido de normativa sobre IA en educación superior (corpus GENIAL).

Ejecutar desde la raíz del repositorio:  streamlit run app/app.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils import LANG_NAMES, build_norm_text, load_config, validate_norm_text  # noqa: E402
from model import MODELS_DIR, Predictor  # noqa: E402

CAT = {"Pedagógico": "#2a78d6", "Ética": "#eb6834", "Administrativo": "#1baf7a", "Protección de Datos": "#eda100", "Técnico": "#e87ba4"}
EJEMPLOS = {
    "Guía de IA generativa para docentes (Ecuador)": ("Lineamientos para el uso de inteligencia artificial generativa en la docencia universitaria", "Orienta a los docentes sobre el diseño de actividades y evaluaciones con herramientas de IA generativa, con criterios de integridad académica y citación."),
    "Ley de protección de datos (Chile)": ("Ley 21.719 que regula la protección y el tratamiento de los datos personales", "Establece principios, derechos de los titulares y obligaciones de los responsables del tratamiento, incluido el tratamiento automatizado de datos."),
    "Estrategia nacional de IA (Brasil, portugués)": ("Estratégia Brasileira de Inteligência Artificial (EBIA)", "Eixo educação: formação de talentos, capacitação docente e diretrizes para o uso de IA no ensino superior."),
    "Reglamento de comité de ética (Colombia)": ("Reglamento del Comité de Ética en Investigación para proyectos con inteligencia artificial", "Define la composición del comité, el procedimiento de evaluación ética de proyectos y los principios de responsabilidad y no maleficencia."),
    "Plan de infraestructura tecnológica (México)": ("Programa institucional de infraestructura de cómputo y centros de datos para inteligencia artificial", "Lineamientos técnicos para la adquisición de servidores, GPU y servicios en la nube, estándares de interoperabilidad y seguridad de la información."),
}
TXT = {
    "es": {"titulo": "Clasificador asistido de normativa sobre IA en educación superior", "sub": "Proyecto GENIAL (Erasmus+) · Proyecto Integrador UEES · Grupo 9", "tab1": "Clasificar una norma", "tab2": "Lote (CSV)", "tab3": "Acerca del modelo", "tab4": "Historial",
           "t_norma": "Título de la norma", "obs": "Observaciones (resumen del contenido)", "pais": "País", "idioma": "Idioma del título", "btn": "Clasificar", "ej": "Probar con un ejemplo", "res": "Dimensión propuesta", "alt": "Segunda candidata",
           "rev": "Esta norma debe pasar por revisión humana", "auto": "Predicción de confianza suficiente para el tramo automático (la decisión final sigue siendo del codificador)", "motivos": "Motivos", "terms": "Términos que más pesan en la decisión", "prob": "Probabilidad por dimensión",
           "proc": "Clasificando…", "csv_help": "CSV con columnas `titulo` y `observaciones` (opcional `pais`, `idioma`). Máximo 500 filas.", "descargar": "Descargar resultados (CSV)", "hist_vacio": "Todavía no hay predicciones en esta sesión.", "limpiar": "Limpiar historial"},
    "en": {"titulo": "Assisted classifier for AI regulation in higher education", "sub": "GENIAL project (Erasmus+) · UEES Capstone · Group 9", "tab1": "Classify one norm", "tab2": "Batch (CSV)", "tab3": "About the model", "tab4": "History",
           "t_norma": "Norm title", "obs": "Observations (content summary)", "pais": "Country", "idioma": "Title language", "btn": "Classify", "ej": "Try an example", "res": "Proposed dimension", "alt": "Second candidate",
           "rev": "This norm must go through human review", "auto": "Confidence high enough for the automatic tier (the coder still makes the final call)", "motivos": "Reasons", "terms": "Terms that weigh most in the decision", "prob": "Probability per dimension",
           "proc": "Classifying…", "csv_help": "CSV with columns `titulo` and `observaciones` (optional `pais`, `idioma`). Max 500 rows.", "descargar": "Download results (CSV)", "hist_vacio": "No predictions in this session yet.", "limpiar": "Clear history"},
}


@st.cache_resource(show_spinner=False)
def cargar_predictor() -> Predictor:
    return Predictor.from_files(MODELS_DIR / "best_model.pkl")


@st.cache_data(show_spinner=False)
def cargar_metricas() -> dict:
    p = ROOT / "results/metrics/evaluacion_final.json"
    s = ROOT / "results/metrics/sintesis_resultados.json"
    out = {}
    if p.exists():
        out["final"] = json.loads(p.read_text(encoding="utf-8"))
    if s.exists():
        out["sintesis"] = json.loads(s.read_text(encoding="utf-8"))
    return out


@st.cache_data(show_spinner=False)
def paises_corpus() -> dict:
    p = ROOT / "data/processed/corpus_depurado.csv"
    if p.exists():
        return pd.read_csv(p)["País"].value_counts().to_dict()
    return {}


def grafico_probabilidades(probs: dict, t: dict):
    df = pd.DataFrame({"dimensión": list(probs.keys()), "probabilidad": list(probs.values())}).sort_values("probabilidad")
    fig = px.bar(df, x="probabilidad", y="dimensión", orientation="h", color="dimensión", color_discrete_map=CAT, text=df["probabilidad"].map(lambda v: f"{v:.0%}"), range_x=[0, 1])
    fig.update_layout(showlegend=False, height=260, margin=dict(l=10, r=10, t=30, b=10), title=t["prob"], xaxis_tickformat=".0%")
    return fig


def mostrar_resultado(r: dict, t: dict):
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"### {t['res']}: <span style='color:{CAT[r['dimension']]}'>{r['dimension']}</span>", unsafe_allow_html=True)
        st.metric("Probabilidad", f"{r['probabilidad']:.0%}")
        st.markdown(f"**{t['alt']}:** {r['segunda_dimension']} ({r['probabilidad_segunda']:.0%})")
        if r["a_revision_humana"]:
            st.warning(f"⚠️ {t['rev']}: " + "; ".join(r["motivos_revision"]))
        else:
            st.success("✅ " + t["auto"])
        if r["terminos"]:
            st.caption(t["terms"] + ": " + ", ".join(f"«{w}»" for w, _ in r["terminos"][:6]))
    with c2:
        st.plotly_chart(grafico_probabilidades(r["probabilidades"], t), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Clasificador GENIAL", page_icon="📚", layout="wide")
    lang = st.sidebar.radio("Idioma / Language", ["es", "en"], horizontal=True, format_func=lambda x: {"es": "Español", "en": "English"}[x])
    t = TXT[lang]
    st.title(t["titulo"]); st.caption(t["sub"])
    if "historial" not in st.session_state:
        st.session_state["historial"] = []
    try:
        predictor = cargar_predictor()
    except FileNotFoundError as e:
        st.error(str(e)); st.stop()
    cfg = load_config(); n_pais = paises_corpus()
    st.sidebar.markdown("**Regla de gobernanza**")
    st.sidebar.markdown(f"- Revisión humana si la probabilidad es < {predictor.threshold:.2f} (umbral calibrado para este modelo)\n- Siempre para países con < {cfg['routing']['mandatory_review_min_country_n']} normas en el corpus\n- Siempre para títulos en {', '.join(LANG_NAMES.get(l, l) for l in cfg['routing']['mandatory_review_languages'])}")
    st.sidebar.markdown("[Repositorio en GitHub](https://github.com/jorgemaldonadom-cmd/genial-clasificador-normativa)")

    tab1, tab2, tab3, tab4 = st.tabs([t["tab1"], t["tab2"], t["tab3"], t["tab4"]])

    with tab1:
        ej = st.selectbox(t["ej"], ["—"] + list(EJEMPLOS.keys()))
        titulo_def, obs_def = EJEMPLOS.get(ej, ("", ""))
        with st.form("form_norma"):
            titulo = st.text_input(t["t_norma"], value=titulo_def, max_chars=500)
            obs = st.text_area(t["obs"], value=obs_def, height=120, max_chars=4000)
            c1, c2 = st.columns(2)
            pais = c1.selectbox(t["pais"], ["(no indicado)"] + sorted(n_pais.keys()))
            idioma = c2.selectbox(t["idioma"], ["es", "pt", "en", "fr"], format_func=lambda x: LANG_NAMES[x])
            enviado = st.form_submit_button(t["btn"], type="primary")
        if enviado:
            texto = build_norm_text(titulo, obs); ok, msg = validate_norm_text(texto)
            if not ok:
                st.error(msg)
            else:
                with st.spinner(t["proc"]):
                    try:
                        r = predictor.predict_one(texto, country=None if pais == "(no indicado)" else pais, language=idioma, n_country=n_pais.get(pais))
                    except Exception as e:  # noqa: BLE001
                        st.error(f"No se pudo clasificar la norma: {e}"); r = None
                if r:
                    mostrar_resultado(r, t)
                    st.session_state["historial"].append({"título": titulo[:80], "dimensión": r["dimension"], "probabilidad": round(r["probabilidad"], 3), "segunda": r["segunda_dimension"], "revisión humana": r["a_revision_humana"]})

    with tab2:
        st.markdown(t["csv_help"])
        plantilla = pd.DataFrame([{"titulo": v[0], "observaciones": v[1], "pais": "", "idioma": "es"} for v in list(EJEMPLOS.values())[:3]])
        st.download_button("Descargar plantilla CSV", plantilla.to_csv(index=False).encode("utf-8"), "plantilla_normas.csv", "text/csv")
        archivo = st.file_uploader("CSV", type=["csv"])
        if archivo is not None:
            try:
                df = pd.read_csv(archivo)
                if "titulo" not in df.columns:
                    st.error("El CSV debe tener una columna `titulo` (y opcionalmente `observaciones`, `pais`, `idioma`).")
                elif len(df) > 500:
                    st.error("Máximo 500 filas por lote.")
                else:
                    with st.spinner(t["proc"]):
                        textos = [build_norm_text(a, b) for a, b in zip(df["titulo"].fillna(""), df.get("observaciones", pd.Series([""] * len(df))).fillna(""))]
                        idiomas = df["idioma"].fillna("es").tolist() if "idioma" in df.columns else None
                        res = predictor.predict_many(textos, languages=idiomas)
                    salida = pd.concat([df.reset_index(drop=True), res.drop(columns=["texto"])], axis=1)
                    st.dataframe(salida, use_container_width=True)
                    st.markdown(f"**{len(salida)}** normas · a revisión humana: **{int(salida['a_revision_humana'].sum())}** · con error de entrada: **{int((salida['error'] != '').sum())}**")
                    st.plotly_chart(px.histogram(salida.dropna(subset=["dimension"]), x="dimension", color="dimension", color_discrete_map=CAT, title="Distribución de dimensiones propuestas").update_layout(showlegend=False, height=300), use_container_width=True)
                    st.download_button(t["descargar"], salida.to_csv(index=False).encode("utf-8"), "predicciones.csv", "text/csv")
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo leer el archivo: {e}")

    with tab3:
        m = cargar_metricas(); meta = predictor.metadata
        st.markdown("""**Qué hace.** Propone la dimensión del framework GENIAL (Pedagógico, Ética, Administrativo, Protección de Datos, Técnico) para una norma sobre inteligencia artificial en educación superior, a partir de su título y un resumen. Entrega dos candidatas con probabilidad y aplica una regla de gobernanza que marca qué normas deben ir a revisión humana. **No sustituye al codificador**: es una herramienta de codificación asistida.""")
        st.markdown(f"**Modelo.** Regresión logística sobre TF-IDF (n-gramas). Hiperparámetros: `{json.dumps(meta.get('parametros', {}), ensure_ascii=False)}`. Entrenado con {meta.get('n_entrenamiento', '—')} normas; versión del corpus `{str(meta.get('corpus_sha256', ''))[:12]}…`.")
        if "cv_por_grupo" in meta:
            cv = meta["cv_por_grupo"]; st.markdown(f"**Validación cruzada por grupo (entrenamiento):** F1 macro {cv['val_f1']:.3f} ± {cv['val_std']:.3f} · gap entrenamiento−validación {cv['gap']:.2f}")
        if "final" in m:
            st.markdown("**Evaluación final sobre el conjunto ciego (normas leídas por un investigador de GENIAL):**")
            st.dataframe(pd.DataFrame(m["final"]["tabla"]).round(3), use_container_width=True)
        st.markdown("""**Limitaciones y advertencias.** Rinde peor en normas en portugués e inglés y en el Caribe (exactitud 0.38 frente a 0.67 en Hispanoamérica continental); las clases Técnico y Administrativo son las más difíciles; una parte de los errores corresponde a normas que tocan dos dimensiones. No usar para publicar índices sin revisión humana, para decisiones sobre personas ni para evaluar a codificadores. Documentación: `docs/model_card.md`, `docs/datasheet_corpus.md`, `docs/consideraciones_eticas.md`.""")

    with tab4:
        if st.session_state["historial"]:
            st.dataframe(pd.DataFrame(st.session_state["historial"]), use_container_width=True)
            if st.button(t["limpiar"]):
                st.session_state["historial"] = []; st.rerun()
        else:
            st.info(t["hist_vacio"])


if __name__ == "__main__":
    main()
