import click
import pandas as pd
import joblib
from pathlib import Path

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('model_path', type=click.Path(exists=True))
def main(input_filepath, output_filepath, model_path):
    """
    Runs model inference on new, UNLABELED data.
    """
    # 1. Load New Data (e.g., today's sensor readings)
    df = pd.read_csv(input_filepath)
    
    # 2. Load the Trained Model
    model = joblib.load(model_path)
    
    # 3. Ensure columns match what the model expects
    # (In a real pipeline, you'd check this strictly)
    features = ['soil_moisture_%', 'temperature_c', 'humidity_%', 'ndvi_index']
    
    # Check if we need to drop columns or just select specific ones
    # Note: We do NOT look for 'crop_disease_status' here because it doesn't exist!
    X = df[features]
    
    # 4. Predict
    print(f"Predicting for {len(df)} rows...")
    predictions = model.predict(X)
    
    # 5. Save Results
    # We create a new file that has the IDs and the Predictions
    output_df = df[['sensor_id', 'timestamp']].copy() # Keep IDs
    output_df['predicted_status'] = predictions
    
    output_df.to_csv(output_filepath, index=False)
    print(f"Predictions saved to {output_filepath}")

if __name__ == '__main__':
    main()