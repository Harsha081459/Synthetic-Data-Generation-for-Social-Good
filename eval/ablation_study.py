"""
SynthoGen AI — DP-TVAE Ablation Study
=======================================
Trains the DP-TVAE model at multiple epsilon budgets and evaluates each one,
producing a JSON file with the real Privacy-Utility tradeoff curve.

Usage (run on GPU server):
  python eval/ablation_study.py --input data/processed/diabetes_mcdd_clean.csv \\
                                --target Diabetes_Target \\
                                --output eval/ablation_results.json
"""

import argparse
import json
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("ablation")


def parse_args():
    p = argparse.ArgumentParser(description="Run DP-TVAE Ablation Study")
    p.add_argument("--input", type=str, required=True, help="Path to real CSV")
    p.add_argument("--target", type=str, required=True, help="Target column")
    p.add_argument("--output", type=str, default="eval/ablation_results.json")
    p.add_argument("--epochs", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()

    epsilons = [1.0, 2.0, 5.0, 10.0, 9999.0]  # 9999 = effectively no DP
    epsilon_labels = ["1.0", "2.0", "5.0", "10.0", "inf"]
    results = []

    for eps, label in zip(epsilons, epsilon_labels):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training DP-TVAE with epsilon = {label}")
        logger.info(f"{'='*60}")

        synth_path = f"data/synthetic/dp_tvae_eps{label}.csv"

        # Step 1: Train DP-TVAE and generate synthetic data
        train_cmd = [
            sys.executable, "models/dp_tvae.py",
            "--dataset", "diabetes",
            "--input", args.input,
            "--output", synth_path,
            "--epsilon", str(eps),
            "--epochs", str(args.epochs),
        ]

        logger.info(f"Running: {' '.join(train_cmd)}")
        result = subprocess.run(train_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Training failed for epsilon={label}: {result.stderr}")
            continue
        logger.info(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

        # Step 2: Evaluate the generated synthetic data
        report_path = f"eval/ablation_eps{label}.json"
        eval_cmd = [
            sys.executable, "eval/evaluate.py",
            "--real", args.input,
            "--synth", synth_path,
            "--target", args.target,
            "--output", report_path,
        ]

        logger.info(f"Running evaluation...")
        result = subprocess.run(eval_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Evaluation failed for epsilon={label}: {result.stderr}")
            continue
        logger.info(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

        # Step 3: Read the report and extract key metrics
        try:
            with open(report_path, "r") as f:
                report = json.load(f)

            entry = {
                "epsilon": label,
                "tstr_accuracy": report.get("utility", {}).get("tstr_accuracy"),
                "tstr_f1": report.get("utility", {}).get("tstr_f1"),
                "utility_loss_gap": report.get("utility", {}).get("utility_loss_gap"),
                "avg_dcr": report.get("privacy_dcr", {}).get("avg_dcr"),
                "exact_matches": report.get("privacy_dcr", {}).get("exact_match_count"),
                "corr_mae": report.get("fidelity_correlation", {}).get("corr_matrix_mae"),
            }
            results.append(entry)
            logger.info(f"✅ Epsilon {label}: Acc={entry['tstr_accuracy']}, DCR={entry['avg_dcr']}")
        except Exception as e:
            logger.error(f"Failed to read report for epsilon={label}: {e}")

    # Save combined ablation results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"\n✅ Ablation study complete. Results saved to {args.output}")
    logger.info(f"\nSummary:")
    for r in results:
        logger.info(f"  ε={r['epsilon']:>5s} -> Accuracy: {r['tstr_accuracy']}, DCR: {r['avg_dcr']}")


if __name__ == "__main__":
    main()
