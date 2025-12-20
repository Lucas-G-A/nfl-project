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
        "Análisis (6 gráficas)",
        "Partidos de esta semana",
        "Partido hipotético"
    ]
)


# -----------------------------
# Página 1: Análisis
# -----------------------------
if pagina == "Análisis (6 gráficas)":
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
