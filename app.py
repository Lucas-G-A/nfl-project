import json
from pathlib import Path

import pandas as pd
import streamlit as st


# -----------------------------
# Configuración de la página
# -----------------------------
st.set_page_config(
    page_title="NFL – Análisis y Predicción",
    layout="wide"
)

st.title("NFL – Análisis y Predicción de Partidos")
st.caption(
    "Aplicación interactiva basada en Elo y métricas de eficiencia "
    "(sin machine learning)"
)

ROOT = Path(__file__).parent
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"
PRED_CSV_DEFAULT = ROOT / "notebooks" / "predictions_this_weekend.csv"
PRED_CSV_ALT = DATA_DIR / "predictions_this_weekend.csv"
ELO_JSON = ROOT / "elo_ratings.json"
LATEST_JSON = ROOT / "latest_team_stats.json"



# -----------------------------
# Funciones auxiliares
# -----------------------------
def cargar_predicciones() -> pd.DataFrame:
    if PRED_CSV_ALT.exists():
        df = pd.read_csv(PRED_CSV_ALT)
        fuente = str(PRED_CSV_ALT)
    elif PRED_CSV_DEFAULT.exists():
        df = pd.read_csv(PRED_CSV_DEFAULT)
        fuente = str(PRED_CSV_DEFAULT)
    else:
        st.error(
            "No se encontró el archivo de predicciones.\n\n"
            "Genera primero `predictions_this_weekend.csv` desde el notebook."
        )
        st.stop()

    columnas_requeridas = {"home_team", "away_team", "home_win_prob"}
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        st.error(f"Faltan columnas en el CSV: {sorted(faltantes)}")
        st.stop()

    df = df.copy()
    if "kickoff_mx" in df.columns:
        df["etiqueta"] = (
            df["away_team"] + " @ " + df["home_team"]
            + " — " + df["kickoff_mx"].astype(str)
        )
    elif "gameday" in df.columns:
        df["etiqueta"] = (
            df["away_team"] + " @ " + df["home_team"]
            + " — " + df["gameday"].astype(str)
        )
    else:
        df["etiqueta"] = df["away_team"] + " @ " + df["home_team"]

    st.sidebar.caption(f"Predicciones cargadas desde: `{fuente}`")
    return df


def cargar_elo() -> dict:
    if not ELO_JSON.exists():
        st.error(
            "No se encontró `elo_ratings.json` en la raíz del proyecto.\n\n"
            "Exporta los ratings desde el notebook de predicciones."
        )
        st.stop()
    return json.loads(ELO_JSON.read_text())

def cargar_latest() -> pd.DataFrame:
    if not LATEST_JSON.exists():
        st.error(
            "No se encontró `latest_team_stats.json` en la raíz.\n\n"
            "Genéralo desde `predicciones.ipynb` (export de latest rolling stats)."
        )
        st.stop()
    data = json.loads(LATEST_JSON.read_text())
    df = pd.DataFrame(data)
    required = {"team","off_ypp","def_ypp","to_margin"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"`latest_team_stats.json` no tiene columnas: {sorted(missing)}")
        st.stop()
    return df.set_index("team")



def probabilidad_elo(elo_local: float, elo_visita: float) -> float:
    return 1 / (1 + 10 ** ((elo_visita - elo_local) / 400))


def mostrar_metricas(equipo_local: str, equipo_visita: str, p_local: float):
    col1, col2 = st.columns(2)
    col1.metric(
        f"Probabilidad de victoria – {equipo_local} (LOCAL)",
        f"{p_local:.1%}"
    )
    col2.metric(
        f"Probabilidad de victoria – {equipo_visita} (VISITA)",
        f"{(1 - p_local):.1%}"
    )

    st.progress(float(p_local))

    favorito = equipo_local if p_local >= 0.5 else equipo_visita
    p_fav = p_local if p_local >= 0.5 else (1 - p_local)
    st.write(f"**Favorito según el modelo:** {favorito} ({p_fav:.1%})")


# -----------------------------
# Navegación
# -----------------------------
pagina = st.sidebar.radio(
    "Navegación",
    [
        "Inicio",
        "Análisis",
        "Partidos de esta semana",
        "Partido hipotético",
        "Contrafactual",
        "Análisis de errores"
    ]
)
if pagina == "Inicio":
    st.header("Descripción del proyecto")

    st.markdown("""
    **Autor:** Lucas García y Alonso Zamanillo
    **Repositorio:** https://github.com/Lucas-G-A/nfl-project 

    Este proyecto analiza el desempeño de equipos de la **NFL** y construye
    un **modelo de predicción de partidos** utilizando métricas estadísticas
    y un sistema de ratings **Elo**, sin utilizar machine learning.

    El objetivo es:
    - Identificar qué factores explican el éxito de un equipo
    - Visualizar estos factores de forma clara
    - Predecir partidos reales y escenarios hipotéticos
    """)

    st.header("🧠 Metodología")

    st.markdown("""
    El proyecto se divide en **dos partes principales**:

    ### 1. Análisis de datos
    A partir de datos oficiales de la NFL se construyen métricas como:
    - Diferencial de entregas de balón (turnover margin)
    - Eficiencia ofensiva (yardas por jugada)
    - Eficiencia defensiva (yardas permitidas)
    - Consistencia en anotación
    - Balance ofensivo–defensivo

    Estas métricas se visualizan en **6 gráficas** que resumen los hallazgos.

    ### 2. Predicción de partidos
    Se utiliza un sistema **Elo** ajustado por:
    - Resultados históricos
    - Ventaja de local
    - (Previamente) métricas de eficiencia calculadas

    El modelo genera:
    - Probabilidades de victoria para los partidos de la semana
    - Predicciones para enfrentamientos hipotéticos
    """)

    st.header("🛠️ Tecnologías utilizadas")

    st.markdown("""
    - **Python**
    - **pandas / numpy** – manipulación de datos
    - **matplotlib** – visualización
    - **nfl_data_py** – datos oficiales de la NFL (schedules y play-by-play)
    - **Streamlit** – interfaz interactiva
    - **Docker** – reproducibilidad del entorno
    """)

    st.header("📦 Estructura del proyecto")

    st.markdown("""
    - `notebooks/`  
      Análisis exploratorio, generación de métricas y predicciones.

    - `figures/`  
      Gráficas finales exportadas como imágenes.

    - `data/`  
      Archivos CSV con predicciones semanales.

    - `elo_ratings.json`  
      Ratings Elo finales para predicciones hipotéticas.

    - `app.py`  
      Aplicación Streamlit (interfaz).

    - `requirements.txt`  
      Dependencias del proyecto.

    - `Dockerfile`  
      Definición del entorno reproducible.
    """)

    st.header("🚀 Cómo funciona la aplicación")

    st.markdown("""
    1. Las predicciones y gráficas se generan **offline** en notebooks.
    2. Los resultados se guardan como archivos (`CSV`, `PNG`, `JSON`).
    3. La aplicación Streamlit **consume estos archivos**, sin recalcular datos.
    4. Esto garantiza velocidad, estabilidad y reproducibilidad.
    """)

    st.info(
        "Este enfoque replica cómo funcionan sistemas reales: "
        "cálculo pesado separado de la capa de presentación."
    )


# -----------------------------
# Página 1: Análisis
# -----------------------------
if pagina == "Análisis":
    st.header("Resultados del análisis")

    if not FIG_DIR.exists():
        st.warning(
            "No se encontró la carpeta `figures/`.\n"
            "Exporta las gráficas como PNG desde los notebooks."
        )
        st.stop()

    imagenes = sorted(FIG_DIR.glob("*.png"))
    if not imagenes:
        st.warning("No hay imágenes PNG dentro de `figures/`.")
        st.stop()

    for img in imagenes:
        st.subheader(img.stem.replace("_", " ").title())
        st.image(str(img), use_container_width=True)


# -----------------------------
# Página 2: Partidos de esta semana
# -----------------------------
elif pagina == "Partidos de esta semana":
    st.header("Predicción de partidos de esta semana")

    df = cargar_predicciones()

    seleccion = st.selectbox("Selecciona un partido", df["etiqueta"])
    fila = df[df["etiqueta"] == seleccion].iloc[0]

    local = fila["home_team"]
    visita = fila["away_team"]
    p_local = float(fila["home_win_prob"])

    st.subheader(f"{visita} @ {local}")
    mostrar_metricas(local, visita, p_local)

    if "kickoff_mx" in fila.index:
        st.write(f"**Inicio del partido (hora CDMX):** {fila['kickoff_mx']}")
    elif "gameday" in fila.index:
        st.write(f"**Fecha del partido:** {fila['gameday']}")


# -----------------------------
# Página 3: Partido hipotético
# -----------------------------
elif pagina == "Partido hipotético":
    st.header("Predicción de partido hipotético (Elo)")

    elo = cargar_elo()
    equipos = sorted(elo.keys())

    col1, col2 = st.columns(2)
    with col1:
        equipo_local = st.selectbox("Equipo local", equipos)
    with col2:
        equipo_visita = st.selectbox(
            "Equipo visitante",
            equipos,
            index=min(1, len(equipos) - 1)
        )

    neutral = st.checkbox(
        "Sede neutral (sin ventaja de local)",
        value=False
    )

    VENTAJA_LOCAL = 55  # puntos Elo

    if st.button("Calcular probabilidad"):
        elo_local = float(elo.get(equipo_local, 1500.0))
        elo_visita = float(elo.get(equipo_visita, 1500.0))

        elo_local_ajustado = elo_local + (0 if neutral else VENTAJA_LOCAL)
        p_local = probabilidad_elo(elo_local_ajustado, elo_visita)

        if neutral:
            st.subheader(f"{equipo_visita} vs {equipo_local} (sede neutral)")
        else:
            st.subheader(f"{equipo_visita} @ {equipo_local}")

        mostrar_metricas(equipo_local, equipo_visita, p_local)

        st.caption(
            "Este cálculo utiliza únicamente el sistema Elo. "
            "Las predicciones semanales incorporan además métricas "
            "de eficiencia calculadas previamente."
        )

elif pagina == "Contrafactual":
    st.header("Contrafactual: impacto de pequeños cambios")

    st.markdown("""
    Esta sección responde preguntas tipo:
    - *¿Qué pasa si el equipo local comete 1 turnover menos?*
    - *¿Qué pasa si su ofensiva mejora 0.3 yardas por jugada?*

    **No es machine learning**: es un análisis explicable que ajusta la probabilidad base del partido.
    """)

    # Cargamos insumos
    df_pred = cargar_predicciones()
    elo = cargar_elo()
    latest_df = cargar_latest()

    # Selector de partido (de esta semana)
    seleccion = st.selectbox("Selecciona un partido de esta semana", df_pred["etiqueta"])
    fila = df_pred[df_pred["etiqueta"] == seleccion].iloc[0]

    home = fila["home_team"]
    away = fila["away_team"]

    # Prob base desde tu CSV (ya incluye tu modelo semanal)
    p_base_home = float(fila["home_win_prob"])

    st.subheader(f"{away} @ {home}")
    st.write(f"**Probabilidad base (LOCAL gana):** {p_base_home:.1%}")

    # Sliders contrafactuales
    st.markdown("### Ajustes hipotéticos")

    col1, col2, col3 = st.columns(3)
    with col1:
        delta_to = st.slider("Turnover margin del LOCAL (cambio)", -3, 3, 0, 1)
    with col2:
        delta_off = st.slider("Off YPP del LOCAL (cambio)", -1.0, 1.0, 0.0, 0.1)
    with col3:
        delta_def = st.slider("Def YPP del LOCAL (cambio)", -1.0, 1.0, 0.0, 0.1)

    st.caption("Nota: Def YPP menor es mejor; por eso un cambio negativo suele ayudar al equipo local.")

    # --- Modelo explicable de ajuste (sin ML) ---
    # Convertimos cambios en “puntos Elo” con pesos razonables y luego a prob.
    HFA = 55

    # Pesos heurísticos (defendibles). Puedes afinarlos.
    W_TO  = 45    # 1 turnover ~ 45 Elo pts (aprox 8-12 pp según matchup)
    W_OFF = 120   # +1.0 ypp es enorme; por eso normalmente usarás +0.1/+0.3
    W_DEF = 120

    # Tomamos Elo actual
    elo_home = float(elo.get(home, 1500.0))
    elo_away = float(elo.get(away, 1500.0))

    # Ajuste Elo por contrafactual
    # Defensa: si delta_def es negativo, mejora al local => suma Elo (por eso restamos)
    adj_elo = (W_TO * delta_to) + (W_OFF * delta_off) + (W_DEF * (-delta_def))

    # Probabilidad contrafactual usando Elo (rápido y estable)
    p_cf_home = probabilidad_elo((elo_home + HFA + adj_elo), elo_away)

    st.markdown("### Resultado")
    colA, colB = st.columns(2)
    colA.metric("Probabilidad base (LOCAL)", f"{p_base_home:.1%}")
    colB.metric("Probabilidad contrafactual (LOCAL)", f"{p_cf_home:.1%}")

    delta_pp = (p_cf_home - p_base_home) * 100
    st.write(f"**Cambio estimado:** {delta_pp:+.1f} puntos porcentuales")

    st.markdown("""
    **Interpretación:**  
    Este cálculo muestra sensibilidad del partido a cambios pequeños y plausibles.  
    No afirma causalidad perfecta, pero ayuda a entender *qué tanto “pesa”* cada factor.
    """)

elif pagina == "Análisis de errores":
    st.header("Análisis de errores: ¿cuándo falla el modelo?")

    path = ROOT / "data" / "backtest_recent_errors.csv"
    if not path.exists():
        st.error(
            "No se encontró `data/backtest_recent_errors.csv`.\n\n"
            "Genéralo desde `predicciones.ipynb` (export del backtest) y vuelve a hacer push."
        )
        st.stop()

    df = pd.read_csv(path)

    st.markdown("""
    Esta sección muestra los partidos donde el modelo estuvo más equivocado.
    Es útil para entender **limitaciones**, **varianza** y **contextos difíciles** (por ejemplo, juegos cerrados o sorpresas).
    """)

    # Métricas globales
    df["pred_home_win"] = df["pred_home_win_prob"] >= 0.5
    accuracy = (df["pred_home_win"].astype(int) == df["actual_home_win"]).mean()
    brier = ((df["pred_home_win_prob"] - df["actual_home_win"]) ** 2).mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy (rango evaluado)", f"{accuracy:.1%}")
    col2.metric("Brier score", f"{brier:.3f}")
    col3.metric("Partidos evaluados", str(len(df)))

    st.divider()

    # Filtro por semana
    weeks = sorted(df["week_num"].unique())
    wmin, wmax = st.select_slider(
        "Filtrar por semanas",
        options=weeks,
        value=(weeks[0], weeks[-1])
    )

    view = df[(df["week_num"] >= wmin) & (df["week_num"] <= wmax)].copy()
    view = view.sort_values("abs_error", ascending=False)

    # Tabla de peores errores
    st.subheader("Top errores (más grandes primero)")
    show_n = st.slider("Cuántos mostrar", 5, 25, 10)

    table = view.head(show_n).copy()

    # Formato amigable
    table["matchup"] = table["away_team"] + " @ " + table["home_team"]
    table["pred_local"] = (table["pred_home_win_prob"] * 100).round(1).astype(str) + "%"
    table["resultado"] = table["away_score"].astype(int).astype(str) + "–" + table["home_score"].astype(int).astype(str)
    table["ganó_local"] = table["actual_home_win"].map({1: "Sí", 0: "No"})
    table["error"] = (table["abs_error"] * 100).round(1).astype(str) + " pp"

    st.dataframe(
        table[["week_num", "matchup", "pred_local", "ganó_local", "resultado", "error"]],
        use_container_width=True
    )

    st.divider()

    # Detalle de un partido
    st.subheader("Detalle de un partido")
    choice = st.selectbox("Selecciona un partido", table["matchup"].tolist())
    r = table[table["matchup"] == choice].iloc[0]

    st.write(f"**Semana:** {int(r['week_num'])}")
    st.write(f"**Partido:** {r['matchup']}")
    st.write(f"**Probabilidad predicha (LOCAL):** {r['pred_local']}")
    st.write(f"**Resultado final (VISITA–LOCAL):** {r['resultado']}")
    st.write(f"**¿Ganó el local?:** {r['ganó_local']}")
    st.write(f"**Error absoluto:** {r['error']}")

    st.info(
        "Interpretación típica: partidos con alta incertidumbre real (lesiones, turnovers raros, "
        "juegos divisionales, o finales cerrados) pueden romper predicciones basadas en ratings/eficiencia."
    )
