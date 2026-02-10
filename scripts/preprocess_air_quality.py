#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def preprocess_air_quality(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv, sep=";", decimal=",")

    # Drop empty trailing columns such as "Unnamed: 15/16".
    unnamed_cols = [column for column in df.columns if str(column).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    if "Date" not in df.columns or "Time" not in df.columns:
        raise ValueError("Input file must contain 'Date' and 'Time' columns.")

    # Parse date/time and place it as the first column.
    dt_str = df["Date"].astype(str) + " " + df["Time"].astype(str).str.replace(".", ":", regex=False)
    df["date"] = pd.to_datetime(dt_str, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df = df.drop(columns=["Date", "Time"])
    df = df.sort_values("date").reset_index(drop=True)

    # Drop NMHC due to extreme missing ratio.
    if "NMHC(GT)" in df.columns:
        df = df.drop(columns=["NMHC(GT)"])

    # Convert sentinel -200 to missing, then fill by time interpolation + edge fills.
    value_columns = [column for column in df.columns if column != "date"]
    df[value_columns] = df[value_columns].replace(-200, pd.NA).apply(pd.to_numeric, errors="coerce")

    df = df.set_index("date")
    df[value_columns] = df[value_columns].interpolate(method="time").ffill().bfill()
    df = df.reset_index()

    # Ensure final output has no missing values.
    missing_total = int(df.isna().sum().sum())
    if missing_total != 0:
        raise ValueError(f"Preprocessing finished with remaining missing values: {missing_total}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"saved: {output_csv}")
    print(f"rows={len(df)}, columns={len(df.columns)}")
    print(f"value_columns={len(value_columns)}")
    print("missing_after=0")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess AirQualityUCI into model-ready CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/air_quality/AirQualityUCI.csv"),
        help="Path to raw AirQualityUCI CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/air_quality/air_quality.csv"),
        help="Path to write preprocessed CSV",
    )
    args = parser.parse_args()

    preprocess_air_quality(args.input, args.output)


if __name__ == "__main__":
    main()
