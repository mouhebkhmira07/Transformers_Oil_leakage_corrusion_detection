import click
import logging
import pandas as pd
from pathlib import Path
import numpy as np

def safe_div(a, b):
    """Avoid division by zero."""
    return a / (b + 1e-6)

# -------------------------------------------------------------
# 1. TIME-BASED FEATURES
# -------------------------------------------------------------

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


# -------------------------------------------------------------
# 2. AGRONOMIC STRESS INDICATORS
# -------------------------------------------------------------

def add_water_stress(df):
    """Compute water stress based on soil moisture deviations."""
    if "soil_moisture_%" in df.columns:
        optimal_moisture = df["soil_moisture_%"].median()
        df["water_stress"] = abs(df["soil_moisture_%"] - optimal_moisture)
    return df


def add_temperature_stress(df):
    """Compute heat/chill stress based on deviation from median."""
    if "temperature_c" in df.columns:
        optimal_temp = df["temperature_c"].median()
        df["temp_stress"] = abs(df["temperature_c"] - optimal_temp)
    return df


def add_ndvi_stress(df):
    """NDVI stress = vegetation weakness."""
    if "ndvi_index" in df.columns:
        df["ndvi_stress"] = 1 - df["ndvi_index"]
    return df


# -------------------------------------------------------------
# 3. WEATHER / ENVIRONMENT FEATURES
# -------------------------------------------------------------

def add_rainfall_features(df):
    if "rainfall_mm" in df.columns:
        df["rainfall_7d"] = df["rainfall_mm"].rolling(7, min_periods=1).sum()
        df["rainfall_30d"] = df["rainfall_mm"].rolling(30, min_periods=1).sum()
    return df


def add_sunlight_features(df):
    if "sunlight_hours" in df.columns:
        df["sunlight_intensity"] = safe_div(df["sunlight_hours"], df["day_of_year"])
    return df


# -------------------------------------------------------------
# 4. CROP GROWTH FEATURES
# -------------------------------------------------------------

def add_growth_features(df):
    if "sowing_date" in df.columns and "timestamp" in df.columns:
        df["days_since_sowing"] = (df["timestamp"] - df["sowing_date"]).dt.days
        df["growth_progress"] = safe_div(df["days_since_sowing"], df["total_days"])
    return df


def add_gdd(df):
    """
    Growing Degree Days (thermal time).
    Base temp = 10°C (generic model).
    """
    if "temperature_c" in df.columns:
        base_temp = 10
        df["GDD"] = np.maximum(0, df["temperature_c"] - base_temp)
        df["cumulative_GDD"] = df["GDD"].cumsum()
    return df


# -------------------------------------------------------------
# 5. DISEASE RISK FEATURES
# -------------------------------------------------------------

def add_disease_risk(df):
    """
    Actual agronomic model:
    Disease risk increases when humidity high AND temperature moderate AND rainfall > 0.
    """
    if all(col in df.columns for col in ["humidity_%", "temperature_c", "rainfall_mm"]):

        df["humidity_risk"] = df["humidity_%"] / 100.0
        df["temp_risk"] = np.exp(-((df["temperature_c"] - 25) ** 2) / 50) # ideal pathogen temp ~25°C
        df["rain_risk"] = (df["rainfall_mm"] > 2).astype(int)

        df["disease_risk"] = (
            df["humidity_risk"] * df["temp_risk"] * (1 + df["rain_risk"])
        )

    return df


# -------------------------------------------------------------
# 6. LOCATION-BASED FEATURES
# -------------------------------------------------------------

def add_location_features(df):
    if "latitude" in df.columns and "longitude" in df.columns:
        df["geo_cluster"] = (
            (df["latitude"] // 1).astype(int).astype(str) + "_" +
            (df["longitude"] // 1).astype(int).astype(str)
        )
    return df


# -------------------------------------------------------------
# 7. MASTER FUNCTION — APPLY ALL FEATURES
# -------------------------------------------------------------

def add_all_agriculture_features(df):

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

    df = df.reset_index(drop=True)
    return df

@click.command()
@click.argument('cleaned_filepath', type=click.Path(exists=True))
@click.argument('features_output_filepath', type=click.Path())
def main(cleaned_filepath, features_output_filepath):
    logger = logging.getLogger(__name__)
    logger.info('Computing features...')

    df = pd.read_csv(cleaned_filepath)
    
    # Ensure datetime columns are parsed
    for col in ['timestamp', 'sowing_date', 'harvest_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Apply all features
    df = add_all_agriculture_features(df)
    
    # Save
    output_path = Path(features_output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Features saved to {features_output_filepath}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()