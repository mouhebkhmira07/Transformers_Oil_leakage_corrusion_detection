import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from matplotlib import pyplot as plt





def snapshot(df: pd.DataFrame, step_name: str) -> None:
    """Print dataset snapshot to monitor changes."""
    print(f"\n--- {step_name} ---")
    print(f"Shape: {df.shape}")
    print("Missing values:\n", df.isna().sum())
    print("First 5 rows:\n", df.head())
    


def diff_rows(df_old: pd.DataFrame, df_new: pd.DataFrame, step_name: str) -> None:
    """Show rows removed in a step."""
    removed = df_old[~df_old.index.isin(df_new.index)]
    print(f"\nRows removed in {step_name}: {len(removed)}")
    if not removed.empty:
        print(removed.head())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    df = df.drop_duplicates()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "harvest_date" in df.columns:
        df["harvest_date"] = pd.to_datetime(df["harvest_date"], errors="coerce")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df[col] = df[col].replace([-999, -1e9, 9999, 99999], pd.NA)

    if "temperature" in df.columns:
        df.loc[df["temperature"] > 70, "temperature"] = pd.NA
        df.loc[df["temperature"] < -10, "temperature"] = pd.NA

    if "humidity" in df.columns:
        df.loc[df["humidity"] < 0, "humidity"] = pd.NA
        df.loc[df["humidity"] > 100, "humidity"] = pd.NA

    if "soil_moisture" in df.columns:
        df.loc[df["soil_moisture"] < 0, "soil_moisture"] = pd.NA
        df.loc[df["soil_moisture"] > 100, "soil_moisture"] = pd.NA

    if "ph" in df.columns:
        df.loc[(df["ph"] < 0) | (df["ph"] > 14), "ph"] = pd.NA

    snapshot(df, "After replacing invalid sensor values")
    
    old_df = df.copy()
    df = df.dropna(thresh=int(len(df.columns) * 0.6))
    diff_rows(old_df, df, "Dropping rows with too many missing values")
    snapshot(df, "After dropping rows with too many NaNs")

    # recompute numeric columns after dropping rows
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower, upper=upper)
    snapshot(df, "After outlier smoothing")

    df = df.reset_index(drop=True)
    snapshot(df, "Final dataset after cleaning")

    return df


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Loading raw Smart Farming IoT dataset...")
    df = pd.read_csv(input_filepath)
    logger.info(f"Raw dataset shape: {df.shape}")
    df = clean_data(df)
    logger.info(f"Processed dataset shape: {df.shape}")
    out_path = Path(output_filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved cleaned dataset to: {output_filepath}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Load environment variables if any
    load_dotenv(find_dotenv())

    main()




