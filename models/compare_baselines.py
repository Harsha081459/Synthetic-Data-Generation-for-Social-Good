import pandas as pd
import argparse
import os
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

def compare_baselines(data_path, output_dir):
    print(f"Loading data from {data_path}...")
    real_data = pd.read_csv(data_path)
    print(f"  Shape: {real_data.shape}")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Training CTGAN (300 epochs) ---")
    ctgan = CTGANSynthesizer(metadata, epochs=300)
    ctgan.fit(real_data)
    syn_ctgan = ctgan.sample(num_rows=len(real_data))
    ctgan_path = os.path.join(output_dir, "ctgan_synthetic.csv")
    syn_ctgan.to_csv(ctgan_path, index=False)
    print(f"Saved: {ctgan_path}")

    print("\n--- Training TVAE (300 epochs) ---")
    tvae = TVAESynthesizer(metadata, epochs=300)
    tvae.fit(real_data)
    syn_tvae = tvae.sample(num_rows=len(real_data))
    tvae_path = os.path.join(output_dir, "tvae_synthetic.csv")
    syn_tvae.to_csv(tvae_path, index=False)
    print(f"Saved: {tvae_path}")

    print("\n--- SDV Quality Evaluation ---")
    print("CTGAN:")
    evaluate_quality(real_data, syn_ctgan, metadata)
    print("\nTVAE:")
    evaluate_quality(real_data, syn_tvae, metadata)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to real data CSV")
    parser.add_argument("--outdir", default="data/synthetic", help="Output directory")
    args = parser.parse_args()
    compare_baselines(args.input, args.outdir)
