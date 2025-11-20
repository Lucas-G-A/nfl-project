import pandas as pd
import nfl_data_py as nfl


def load_pbp(seasons):
    """
    Load play-by-play data for given seasons.
    """
    # This is the core function from the package:
    # nfl.import_pbp_data(years, columns=None, downcast=True, cache=False, alt_path=None)
    pbp = nfl.import_pbp_data(seasons)  # full set of columns
    # Keep only regular season offensive plays (rush or pass)
    pbp = pbp[
        ((pbp["pass"] == 1) | (pbp["rush"] == 1))
        & (pbp["season_type"] == "REG")
    ]
    return pbp


def team_offense_summary(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple offensive efficiency table by team.
    """
    grouped = (
        pbp
        .groupby("posteam")
        .agg(
            plays=("play_id", "count"),
            total_yards=("yards_gained", "sum"),
            yards_per_play=("yards_gained", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    # Sort: best offenses first
    grouped = grouped.sort_values("yards_per_play", ascending=False)
    return grouped


def main():
    seasons = [2025]  # you can change this to [2022, 2023], etc.
    print(f"Loading play-by-play data for seasons: {seasons} ...")
    pbp = load_pbp(seasons)

    print(f"PBP shape: {pbp.shape}")
    print("Columns example:", list(pbp.columns[:15]))

    summary = team_offense_summary(pbp)

    print("\nTop 10 offenses by yards per play:")
    print(summary.head(10).to_string(index=False))

    # Save to CSV so you have an artifact for the report
    summary.to_csv("team_offense_summary.csv", index=False)
    print("\nSaved team_offense_summary.csv")


if __name__ == "__main__":
    main()
