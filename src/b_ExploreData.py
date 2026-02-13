"""
Loads a consolidated dataset (e.g., Consolidated.csv or Consolidated_sparse.csv)
and launches the interactive explore_data browser app.
"""
import pandas as pd
from pathlib import Path
from utils.preprocessing import explore_data, prepare_tsne_plot

## Configuration Parameters
# Path to the consolidated dataset to visualize
input_file = "../data/output/regression/Consolidated_sparse.csv"  # Examples: ../data/output/regression/Consolidated.csv


if __name__ == '__main__':
    data_path = Path(input_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Input file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Ensure the timestamp column is present and parsed
    if "TIMESTAMP" in df.columns:
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    elif "Time" in df.columns:
        df.rename(columns={"Time": "TIMESTAMP"}, inplace=True)
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    else:
        raise ValueError("Input file must contain a 'TIMESTAMP' or 'Time' column.")
    
    x = prepare_tsne_plot(df)

    # Launch interactive exploration app
    # explore_data(df)
