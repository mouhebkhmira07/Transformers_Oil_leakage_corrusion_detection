import click
import logging
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

def safe_div(a, b):
    """Avoid division by zero."""
    return a / (b + 1e-6)

# ============================================================
# 1. TIME-BASED FEATURES
# ============================================================

def add_time_features(df):
    if "timestamp" in df.columns:
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month
        df["day_of_year"] = df["timestamp"].dt.dayofyear
        df["week"] = df["timestamp"].dt.isocalendar().week.astype(int)
        df["season"] = df["month"].map(
            {12:"Winter",1:"Winter",2:"Winter",
            3:"Spring",4:"Spring",5:"Spring",
            6:"Summer",7:"Summer",8:"Summer",
            9:"Autumn",10:"Autumn",11:"Autumn"}
        )
    return df


# ============================================================
# 2. AGRONOMIC FEATURES (your original work)
# ============================================================

def add_water_stress(df):
    if "soil_moisture_%" in df.columns:
        optimal_moisture = df["soil_moisture_%"].median()
        df["water_stress"] = abs(df["soil_moisture_%"] - optimal_moisture)
    return df

def add_temperature_stress(df):
    if "temperature_c" in df.columns:
        optimal_temp = df["temperature_c"].median()
        df["temp_stress"] = abs(df["temperature_c"] - optimal_temp)
    return df

def add_ndvi_stress(df):
    if "ndvi_index" in df.columns:
        df["ndvi_stress"] = 1 - df["ndvi_index"]
    return df

def add_rainfall_features(df):
    if "rainfall_mm" in df.columns:
        df["rainfall_7d"] = df["rainfall_mm"].rolling(7, min_periods=1).sum()
        df["rainfall_30d"] = df["rainfall_mm"].rolling(30, min_periods=1).sum()
    return df

def add_sunlight_features(df):
    if "sunlight_hours" in df.columns:
        df["sunlight_intensity"] = safe_div(df["sunlight_hours"], df["day_of_year"])
    return df

def add_growth_features(df):
    if "sowing_date" in df.columns and "timestamp" in df.columns:
        df["days_since_sowing"] = (df["timestamp"] - df["sowing_date"]).dt.days
        df["growth_progress"] = safe_div(df["days_since_sowing"], df.get("total_days", 1))
    return df

def add_gdd(df):
    if "temperature_c" in df.columns:
        base_temp = 10
        df["GDD"] = np.maximum(0, df["temperature_c"] - base_temp)
        df["cumulative_GDD"] = df["GDD"].cumsum()
    return df

def add_disease_risk(df):
    if all(col in df.columns for col in ["humidity_%", "temperature_c", "rainfall_mm"]):

        df["humidity_risk"] = df["humidity_%"] / 100.0
        df["temp_risk"] = np.exp(-((df["temperature_c"] - 25) ** 2) / 50)
        df["rain_risk"] = (df["rainfall_mm"] > 2).astype(int)

        df["disease_risk"] = (
            df["humidity_risk"] * df["temp_risk"] * (1 + df["rain_risk"])
        )

    return df

def add_location_features(df):
    if "latitude" in df.columns and "longitude" in df.columns:
        df["geo_cluster"] = (
            (df["latitude"] // 1).astype(int).astype(str) + "_" +
            (df["longitude"] // 1).astype(int).astype(str)
        )
    return df


# ============================================================
# 3. NEW FEATURES — Scaling + Anomaly Flags
# ============================================================

def add_scaled_features(df):
    """Add Min-Max scaled version of all numeric sensor columns."""
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    new_cols = {}

    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()

        if col_max - col_min == 0:
            new_cols[col + "_scaled"] = 0
        else:
            new_cols[col + "_scaled"] = (df[col] - col_min) / (col_max - col_min)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    
    return df


def add_anomaly_flags(df):
    """Simple rule-based anomaly flags for AIOps & anomaly detection."""
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    new_cols = {}

    for col in numeric_cols:
        # Rule 1: NaN or missing
        flag = df[col].isna().astype(int)

        # Rule 2: Sudden jumps (high first-derivative)
        # Fill NA in diff to avoid propagating NaNs to flags unnecessarily, or handle as 0
        jump_flag = (abs(df[col].diff().fillna(0)) > df[col].std() * 3).astype(int)
        flag = flag | jump_flag

        # Rule 3: Extreme values (beyond 1st & 99th percentile)
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        extreme_flag = ((df[col] < q1) | (df[col] > q99)).astype(int)
        flag = flag | extreme_flag
        
        new_cols[col + "_anomaly_flag"] = flag

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ============================================================
# 4. MASTER FUNCTION — Apply all features
# ============================================================

def add_all_agriculture_features(df):
    """
    Apply feature engineering based on available columns.
    For Crop_recommendation dataset: N, P, K, temperature, humidity, ph, rainfall, label
    """
    # Only apply features if the required columns exist
    df = add_time_features(df)
    df = add_water_stress(df)
    df = add_temperature_stress(df)
    df = add_ndvi_stress(df)
    df = add_rainfall_features(df)
    df = add_sunlight_features(df)
    df = add_growth_features(df)
    df = add_gdd(df)
    df = add_disease_risk(df)
    df = add_location_features(df)

    # These work with any numeric columns
    df = add_scaled_features(df)
    df = add_anomaly_flags(df)

    df = df.reset_index(drop=True)
    return df

def stratified_split(df, target_col="label", test_size=0.2):
    """
    Splits data while maintaining the same percentage of classes (e.g., Crops)
    in both Train and Test sets. Essential for small datasets.
    """
    logger = logging.getLogger(__name__)
    
    # We need to extract the indices for the split
    # We default to 'crop_type' to ensure we test on ALL crops, not just some.
    if target_col not in df.columns:
        # Fallback if the target column doesn't exist
        logger.warning(f"{target_col} not found. Falling back to simple random split.")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        return train_df, test_df

    # Initialize the splitter
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    
    for train_index, test_index in splitter.split(df, df[target_col]):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
    
    return train_df, test_df


@click.command()
@click.argument('cleaned_filepath', type=click.Path(exists=True))
@click.argument('features_output_dir', type=click.Path())
def main(cleaned_filepath, features_output_dir):
    logger = logging.getLogger(__name__)
    logger.info('Computing features...')

    df = pd.read_csv(cleaned_filepath)

    # Ensure datetime columns are parsed
    for col in ['timestamp', 'sowing_date', 'harvest_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Apply all features
    df = add_all_agriculture_features(df)

    logger.info("Splitting data using Stratified Strategy on 'label'...")
    
    train_df, test_df = stratified_split(df, target_col="label", test_size=0.2)

    # Save to csv
    output_path = Path(features_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_csv_path = output_path / "train.csv"
    test_csv_path = output_path / "test.csv"
    
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    logger.info(f"Saved train set to {train_csv_path}")
    logger.info(f"Saved test set to {test_csv_path}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
