from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def convert_movies() -> None:
    df = pd.read_csv(
        DATA_DIR / "movies.dat",
        sep="::",
        engine="python",
        header=None,
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    df.to_parquet(DATA_DIR / "movies.parquet", index=False)


def convert_ratings() -> None:
    df = pd.read_csv(
        DATA_DIR / "ratings.dat",
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    df.to_parquet(DATA_DIR / "ratings.parquet", index=False)


def convert_users() -> None:
    df = pd.read_csv(
        DATA_DIR / "users.dat",
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"],
    )
    df.to_parquet(DATA_DIR / "users.parquet", index=False)


def main() -> None:
    convert_movies()
    convert_ratings()
    convert_users()
    print("Converted .dat files to parquet in", DATA_DIR)


if __name__ == "__main__":
    main()
