import json
from pathlib import Path

import pandas as pd
import streamlit as st


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="NFL Project", layout="wide")
st.title("NFL Analysis & Match Predictor")
st.caption("Interactive app: 5 graphs + this week's win probabilities + hypothetical matchups (Elo-based)")

ROOT = Path(__file__).parent
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"
PRED_CSV_DEFAULT = ROOT / "notebooks" / "predictions_this_weekend.csv"  # your current location
PRED_CSV_ALT = DATA_DIR / "predictions_this_weekend.csv"               # recommended location
ELO_JSON = ROOT / "elo_ratings.json"


# -----------------------------
# Helpers
# -----------------------------
def load_predictions() -> pd.DataFrame:
    if PRED_CSV_ALT.exists():
        df = pd.read_csv(PRED_CSV_ALT)
        source = str(PRED_CSV_ALT)
    elif PRED_CSV_DEFAULT.exists():
        df = pd.read_csv(PRED_CSV_DEFAULT)
        source = str(PRED_CSV_DEFAULT)
    else:
        st.error("Missing predictions CSV. Expected one of:\n"
                 f"- {PRED_CSV_ALT}\n"
                 f"- {PRED_CSV_DEFAULT}\n\n"
                 "Generate it from your notebook first.")
        st.stop()

    required = {"home_team", "away_team", "home_win_prob"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Predictions CSV is missing columns: {sorted(missing)}")
        st.stop()

    # Nice label for selection
    df = df.copy()
    if "kickoff_mx" in df.columns:
        df["matchup_label"] = df["away_team"] + " @ " + df["home_team"] + " — " + df["kickoff_mx"].astype(str)
    elif "gameday" in df.columns and "gametime" in df.columns:
        df["matchup_label"] = df["away_team"] + " @ " + df["home_team"] + " — " + df["gameday"].astype(str) + " " + df["gametime"].astype(str)
    else:
        df["matchup_label"] = df["away_team"] + " @ " + df["home_team"]

    st.sidebar.caption(f"Predictions loaded from: `{source}`")
    return df


def load_elo() -> dict:
    if not ELO_JSON.exists():
        st.error(
            f"Missing `{ELO_JSON.name}` at project root.\n\n"
            "Export it from your predictions notebook (instructions below)."
        )
        st.stop()
    return json.loads(ELO_JSON.read_text())


def win_prob_from_elo(elo_home: float, elo_away: float) -> float:
    # standard Elo logistic transform
    return 1 / (1 + 10 ** ((elo_away - elo_home) / 400))


def show_metric_block(home: str, away: str, p_home: float):
    col1, col2 = st.columns(2)
    col1.metric(f"{home} win probability (HOME)", f"{p_home:.1%}")
    col2.metric(f"{away} win probability (AWAY)", f"{(1 - p_home):.1%}")

    # Visual bar is home probability
    st.progress(float(p_home))

    fav = home if p_home >= 0.5 else away
    fav_p = p_home if p_home >= 0.5 else (1 - p_home)
    st.write(f"**Favorite:** {fav} ({fav_p:.1%})")


# -----------------------------
# Navigation
# -----------------------------
page = st.sidebar.radio("Page", ["Analysis (5 graphs)", "This Week", "Hypothetical Matchup"])


# -----------------------------
# Page 1: Analysis
# -----------------------------
if page == "Analysis (5 graphs)":
    st.header("Analysis Results (5 Graphs)")

    if not FIG_DIR.exists():
        st.warning(f"Missing folder: `{FIG_DIR}`. Create it and export your 6 graphs as PNGs there.")
        st.stop()

    # Display all PNGs in figures/, sorted
    pngs = sorted(FIG_DIR.glob("*.png"))
    if not pngs:
        st.warning(f"No PNG files found in `{FIG_DIR}`. Export your graphs as .png first.")
        st.stop()

    # Optional: If you want a specific order, rename files like 01_*, 02_* etc.
    for p in pngs:
        st.subheader(p.stem.replace("_", " ").title())
        st.image(str(p), use_container_width=True)


# -----------------------------
# Page 2: This Week
# -----------------------------
elif page == "This Week":
    st.header("This Week's Games")

    df = load_predictions()

    # Select game
    selection = st.selectbox("Select a game", df["matchup_label"])
    row = df[df["matchup_label"] == selection].iloc[0]

    home = row["home_team"]
    away = row["away_team"]
    p_home = float(row["home_win_prob"])

    st.subheader(f"{away} @ {home}")
    show_metric_block(home, away, p_home)

    # Show kickoff if available
    if "kickoff_mx" in row.index:
        st.write(f"**Kickoff (Mexico City):** {row['kickoff_mx']}")
    elif "gameday" in row.index:
        st.write(f"**Game day:** {row['gameday']} {row.get('gametime', '')}".strip())


# -----------------------------
# Page 3: Hypothetical Matchup
# -----------------------------
elif page == "Hypothetical Matchup":
    st.header("Hypothetical Matchup (Elo)")

    elo = load_elo()
    teams = sorted(elo.keys())

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", teams)
    with col2:
        away_team = st.selectbox("Away Team", teams, index=min(1, len(teams) - 1))

    neutral = st.checkbox("Neutral site (no home-field advantage)", value=False)

    HFA = 55  # Elo points; simple and explainable

    if st.button("Predict"):
        elo_home = float(elo.get(home_team, 1500.0))
        elo_away = float(elo.get(away_team, 1500.0))

        elo_home_adj = elo_home + (0 if neutral else HFA)
        p_home = win_prob_from_elo(elo_home_adj, elo_away)

        st.subheader(f"{away_team} @ {home_team}" if not neutral else f"{away_team} vs {home_team} (Neutral)")
        show_metric_block(home_team, away_team, p_home)

        st.caption("Hypothetical uses Elo only (fast, stable for demos). Your weekly predictions page uses Elo+efficiency outputs from the notebook.")
