#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_dataset(csv_path: Path, output_path: Path) -> None:
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError(f"'date' column not found in {csv_path}")

    df["date"] = pd.to_datetime(df["date"])
    value_columns = [column for column in df.columns if column != "date"]
    if not value_columns:
        raise ValueError(f"No value columns found in {csv_path}")

    rows = len(value_columns)
    fig, axes = plt.subplots(rows, 1, figsize=(16, 2.2 * rows), sharex=True)
    if rows == 1:
        axes = [axes]

    for axis, column in zip(axes, value_columns):
        axis.plot(df["date"], df[column], linewidth=0.8)
        axis.set_ylabel(column)
        axis.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    axes[-1].set_xlabel("date")
    fig.suptitle(f"{csv_path.stem} variable time series", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all variables for selected datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ETTh1", "ETTh2", "AirQuality", "AppliancesEnergy"],
        help="Dataset names to plot (e.g., ETTh1 ETTh2 ETTm1 ETTm2 Exchange Weather)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("documents/analysis/figures"),
        help="Directory to write output PNG files",
    )
    args = parser.parse_args()

    dataset_to_csv = {
        "ETTh1": Path("dataset/ETT-small/ETTh1.csv"),
        "ETTh2": Path("dataset/ETT-small/ETTh2.csv"),
        "ETTm1": Path("dataset/ETT-small/ETTm1.csv"),
        "ETTm2": Path("dataset/ETT-small/ETTm2.csv"),
        "Exchange": Path("dataset/exchange_rate/exchange_rate.csv"),
        "Weather": Path("dataset/weather/weather.csv"),
        "AirQuality": Path("dataset/air_quality/air_quality.csv"),
        "AppliancesEnergy": Path("dataset/appliances_energy/energydata_complete.csv")
    }

    for dataset in args.datasets:
        if dataset not in dataset_to_csv:
            raise ValueError(
                f"Unsupported dataset '{dataset}'. "
                f"Supported: {sorted(dataset_to_csv.keys())}"
            )
        csv_path = dataset_to_csv[dataset]
        out_path = args.out_dir / f"{dataset}_variables.png"
        plot_dataset(csv_path, out_path)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
