"""Score VLM perturbation-detection responses against ground truth.

Reads the raw responses run_benchmark.py writes to <output_dir>/<model>/<entry>.json, joins them
against the ground truth stored alongside each dataset entry by create_perturbation_dataset.py
(<dataset_dir>/<entry>/{labels,perturbations}.json), and reports per-class accuracy, F1 and AUROC
for every model found in <output_dir>.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from src.create_perturbation_dataset import DATASET_DIR
from src.run_benchmark import OUTPUT_DIR


def _digit_to_label(entry_dir: Path) -> dict[int, str]:
    with open(entry_dir / "labels.json") as f:
        legend_entries = json.load(f)
    return {digit: label for entry in legend_entries for label, digit in entry.items()}


def load_predictions(output_dir: Path, dataset_dir: Path) -> pd.DataFrame:
    rows = []
    for model_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        model = model_dir.name
        for response_path in sorted(model_dir.glob("*.json")):
            entry_dir = dataset_dir / response_path.stem
            if not entry_dir.is_dir():
                print(f"skipping {response_path.name}: no dataset entry at {entry_dir}")
                continue

            digit_to_label = _digit_to_label(entry_dir)
            with open(entry_dir / "perturbations.json") as f:
                ground_truth = json.load(f)
            with open(response_path) as f:
                response = json.load(f)

            for prediction in response["predictions"]:
                object_label = digit_to_label.get(prediction["object_number"])
                if object_label is None:
                    # the model referenced a digit that isn't in our legend -- can't score it
                    continue

                spec = ground_truth.get(object_label)
                rows.append(
                    {
                        "model": model,
                        "entry": entry_dir.name,
                        "object_label": object_label,
                        "predicted_type": prediction["perturbation_type"],
                        "true_type": spec["perturbation_type"] if spec else "none",
                    }
                )

    df = pd.DataFrame(rows)
    # The prompt's vocabulary calls this perturbation "translation" while perturbations.json
    # calls it "placement" -- normalize onto the ground-truth spelling.
    df["predicted_type"] = df["predicted_type"].replace("translation", "placement")
    return df


def per_class_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in predictions_df.groupby("model"):
        classes = sorted(set(group["true_type"]) | set(group["predicted_type"]))
        y_true_bin = label_binarize(group["true_type"], classes=classes)
        y_pred_bin = label_binarize(group["predicted_type"], classes=classes)

        for i, cls in enumerate(classes):
            y_true_c, y_pred_c = y_true_bin[:, i], y_pred_bin[:, i]
            # The raw responses only carry a hard predicted label, no confidence score, so AUROC
            # collapses to the single (FPR, TPR) point implied by that hard call -- equivalent to
            # balanced accuracy for this class, not a curve swept over a threshold.
            auroc = roc_auc_score(y_true_c, y_pred_c) if y_true_c.min() != y_true_c.max() else float("nan")
            rows.append(
                {
                    "model": model,
                    "class": cls,
                    "support": int(y_true_c.sum()),
                    "accuracy": (y_true_c == y_pred_c).mean(),
                    "f1": f1_score(y_true_c, y_pred_c, zero_division=0),
                    "auroc": auroc,
                }
            )

    return pd.DataFrame(rows).set_index(["model", "class"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory of <model>/<entry>.json responses to score (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Directory of <entry>/{labels,perturbations}.json ground truth (default: %(default)s)",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optionally write the per-class metrics to this CSV path")
    args = parser.parse_args()

    predictions_df = load_predictions(args.output_dir, args.dataset_dir)
    print(f"{len(predictions_df)} scored predictions across {predictions_df['model'].nunique()} model(s)\n")

    metrics_df = per_class_metrics(predictions_df).round(3)
    with pd.option_context("display.max_rows", None):
        print(metrics_df)

    if args.csv is not None:
        metrics_df.to_csv(args.csv)
        print(f"\nWrote per-class metrics to {args.csv}")


if __name__ == "__main__":
    main()
