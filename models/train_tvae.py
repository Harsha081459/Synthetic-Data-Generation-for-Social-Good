import pandas as pd
import argparse
import os
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata

def train_and_sample(dataset_name, data_path, output_dir):
    print(f"Loading {dataset_name} from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Shape: {df.shape}")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    print("Training TVAE (300 epochs)...")
    synthesizer = TVAESynthesizer(metadata, epochs=300)
    synthesizer.fit(df)

    print(f"Sampling {len(df)} synthetic rows...")
    synthetic_data = synthesizer.sample(num_rows=len(df))

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"tvae_{dataset_name}.csv")
    synthetic_data.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    model_path = os.path.join(output_dir, f"tvae_{dataset_name}_model.pkl")
    synthesizer.save(filepath=model_path)
    print(f"Model saved: {model_path}")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TVAE on a clean CSV and generate synthetic data")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. synthea, diabetes, framingham)")
    parser.add_argument("--input",   required=True, help="Path to cleaned real CSV")
    parser.add_argument("--outdir",  default="data/synthetic", help="Directory to save outputs")
    args = parser.parse_args()
    train_and_sample(args.dataset, args.input, args.outdir)
