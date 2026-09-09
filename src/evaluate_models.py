"""Score VLM perturbation-detection responses against ground truth.

Reads the raw responses run_benchmark.py writes to <output_dir>/<model>/<entry>.json, joins them
against the ground truth stored alongside each dataset entry by create_perturbation_dataset.py
(<dataset_dir>/<entry>/{labels,perturbations}.json), and reports one headline table -- macro_f1,
auroc, accuracy (see macro_metrics' docstring for exactly what each one measures and why) -- for
every model found in <output_dir>, plus two baselines (see add_baselines): always-<majority-class>
and random_choice, so raw class imbalance and pure chance can't be mistaken for model skill.

With --cross-view, <entry>_view<n>.json responses (written by run_benchmark.py --cross-view) are
scored too, joined against the same <entry>/perturbations.json ground truth as their base entry
-- the perturbation spec doesn't change across camera views, only what's visible from each one.
Same-view and cross-view numbers are never pooled into a single score -- judging from the camera
the perturbation was authored from is an easier task than judging from one that's never seen the
"before" state -- but both show up in the same table, as same/cross columns nested under each
metric, so the gap between them is easy to read at a glance.

With --sample <n> --seed <seed>, scoring is restricted to <n> dataset entries drawn via
random.Random(seed).sample() over dataset_dir's entries -- the exact selection logic
optimize_rendering.py uses for its --n-entries/--seed. Passing the same <n>/<seed> here scores
only the entries a study was actually tuned against, so the resulting macro_f1 is comparable to
the trial values in that study instead of being diluted by entries the tuning never saw.
"""

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

PERTURBATION_CLASSES = ["count", "scale", "rotation", "placement"]
RANDOM_BASELINE_SEED = 0

_CROSS_VIEW_RESPONSE = re.compile(r"^(?P<entry>.+)_view\d+$")


def _digit_to_label(entry_dir: Path) -> dict[int, str]:
    with open(entry_dir / "labels.json") as f:
        legend_entries = json.load(f)
    return {digit: label for entry in legend_entries for label, digit in entry.items()}


def sample_entry_names(dataset_dir: Path, n: int, seed: int) -> set[str]:
    """The exact entry-selection logic optimize_rendering.py uses for its --n-entries/--seed --
    duplicated here (rather than imported) so evaluate_models.py doesn't have to pull in Optuna
    just to reproduce a plain random.sample() call. Must stay byte-for-byte identical to that
    script's own selection or --sample stops matching the entries a study was tuned against.
    """
    entry_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    return {p.name for p in random.Random(seed).sample(entry_dirs, k=min(n, len(entry_dirs)))}


def load_predictions(
    output_dir: Path,
    dataset_dir: Path,
    include_cross_view: bool = False,
    entry_names: set[str] | None = None,
) -> pd.DataFrame:
    rows = []
    for model_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        model = model_dir.name
        for response_path in sorted(model_dir.glob("*.json")):
            cross_view_match = _CROSS_VIEW_RESPONSE.match(response_path.stem)
            if cross_view_match and not include_cross_view:
                continue
            entry_name = cross_view_match.group("entry") if cross_view_match else response_path.stem
            if entry_names is not None and entry_name not in entry_names:
                continue

            entry_dir = dataset_dir / entry_name
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
                        # The response file's own stem, not entry_dir.name: several cross-view
                        # responses share one entry_dir, and collapsing them onto the same
                        # "entry" would make add_baselines' drop_duplicates() treat their ground
                        # truth as a single repeated row instead of one per scored view.
                        "entry": response_path.stem,
                        "view": "cross" if cross_view_match else "same",
                        "object_label": object_label,
                        "predicted_type": prediction["perturbation_type"],
                        "true_type": spec["perturbation_type"] if spec else "none",
                        # Only present in responses scored under the newer prompt; NaN (rather
                        # than a KeyError) for anything scored before "confidence" existed.
                        "confidence": prediction.get("confidence"),
                    }
                )

    if not rows:
        # pd.DataFrame([]) has zero columns, so every column access below would raise KeyError
        # instead of just describing "nothing was scorable" -- a real outcome (e.g. every VLM
        # call for a run failed) that callers like optimize_rendering.py explicitly check
        # predictions_df.empty for, and need back as an empty-but-correctly-shaped frame.
        return pd.DataFrame(columns=["model", "entry", "view", "object_label", "predicted_type", "true_type", "confidence"])

    df = pd.DataFrame(rows)
    # The prompt's vocabulary calls these perturbations "translation" and "removed" while
    # perturbations.json calls them "placement" and "count" -- normalize onto the ground-truth
    # spelling.
    df["predicted_type"] = df["predicted_type"].replace({"translation": "placement", "removed": "count"})
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    return df


def _macro_f1(predictions_df: pd.DataFrame) -> pd.Series:
    # Mean F1 over the four perturbation classes (count/scale/rotation/placement) only. Each is
    # computed one-vs-rest across every row -- a "none" prediction on a genuinely-perturbed
    # object still counts as a false negative there, and a spurious perturbation-type prediction
    # on an untouched object still counts as a false positive -- "none" itself is left out of the
    # average since it isn't a perturbation type.
    rows = {}
    for (model, view), group in predictions_df.groupby(["model", "view"]):
        f1s = [
            f1_score(group["true_type"] == cls, group["predicted_type"] == cls, zero_division=0)
            for cls in PERTURBATION_CLASSES
        ]
        rows[(model, view)] = np.mean(f1s)
    return pd.Series(rows, name="macro_f1").rename_axis(["model", "view"])


def _accuracy(predictions_df: pd.DataFrame) -> pd.Series:
    return (
        (predictions_df["predicted_type"] == predictions_df["true_type"])
        .groupby([predictions_df["model"], predictions_df["view"]])
        .mean()
        .rename("accuracy")
    )


def add_baselines(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Append two trivial baselines as extra "model" rows, so every other row's numbers have
    something to clear before they mean anything:

    - always_<majority>: always predicts the majority true class. On this dataset "none" is
      ~75% of all objects, so this scores ~0.75 raw accuracy while identifying zero
      perturbations.
    - random_choice: predicts uniformly at random among the five classes (none plus the four
      perturbation types), seeded via RANDOM_BASELINE_SEED so re-running the report doesn't
      change its numbers.

    Ground truth is deduplicated across models first so neither baseline's support is skewed by
    any one model's occasional missing/failed predictions.

    confidence is set to 1.0 for both: neither is a genuine calibrated probability, and inventing
    a lower number would be worse than omitting one. For random_choice this also keeps
    _binary_auroc's ranking score independent of the true label, which is exactly what pins its
    AUROC to the mathematically-correct ~0.5 for an uninformative baseline instead of leaving it
    undefined.
    """
    ground_truth = predictions_df[["entry", "object_label", "true_type", "view"]].drop_duplicates()

    majority_class = predictions_df["true_type"].value_counts().idxmax()
    majority_df = ground_truth.copy()
    majority_df["model"] = f"always_{majority_class}"
    majority_df["predicted_type"] = majority_class
    majority_df["confidence"] = 1.0

    rng = np.random.default_rng(RANDOM_BASELINE_SEED)
    random_df = ground_truth.copy()
    random_df["model"] = "random_choice"
    random_df["predicted_type"] = rng.choice(["none", *PERTURBATION_CLASSES], size=len(random_df))
    random_df["confidence"] = 1.0

    return pd.concat([predictions_df, majority_df, random_df], ignore_index=True)


def _binary_auroc(predictions_df: pd.DataFrame) -> pd.Series:
    """AUROC for the binary "was this object perturbed at all" question (true_type != "none"),
    over every row. There's no direct probability of that; the ranking score is the model's own
    confidence in whichever class it predicted, reinterpreted as evidence for "perturbed" -- its
    stated confidence when it predicted a perturbation type, or 1 - confidence when it predicted
    "none" (so a confident "none" call counts as strong evidence *against* "perturbed"). NaN for
    a model with no usable confidence values at all (rows lacking one are dropped first, and if
    that leaves fewer than two true classes there's nothing left to rank).
    """
    is_perturbed = predictions_df["true_type"] != "none"
    is_none_prediction = predictions_df["predicted_type"] == "none"
    score = predictions_df["confidence"].where(~is_none_prediction, 1 - predictions_df["confidence"])

    result = {}
    for (model, view), index in predictions_df.groupby(["model", "view"]).groups.items():
        valid = score.loc[index].notna()
        y_true, y_score = is_perturbed.loc[index][valid], score.loc[index][valid]
        result[(model, view)] = roc_auc_score(y_true, y_score) if valid.any() and y_true.nunique() == 2 else float("nan")

    return pd.Series(result, name="auroc").rename_axis(["model", "view"])


def macro_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Three headline numbers per (model, view) row (plus the always-majority-class and
    random_choice baselines from add_baselines, for scale):

    - macro_f1: see _macro_f1.
    - auroc: see _binary_auroc.
    - accuracy: raw fraction of all rows correct. Dominated by whichever class is most common
      in the dataset (75% "none" here) -- only meaningful read next to the baseline rows.
    """
    return pd.concat([_macro_f1(predictions_df), _binary_auroc(predictions_df), _accuracy(predictions_df)], axis=1)


def _print_report(predictions_df: pd.DataFrame, csv_path: Path | None) -> None:
    if predictions_df.empty:
        print("No scored predictions.\n")
        return

    views = predictions_df["view"].unique()
    print(f"{len(predictions_df)} scored predictions across {predictions_df['model'].nunique()} model(s)\n")
    print(
        "  macro_f1: mean F1 over count/scale/rotation/placement, none excluded\n"
        "  auroc: binary perturbed-vs-not, ranked by confidence (1 - confidence when the model predicted \"none\")\n"
        "  accuracy: raw fraction of all rows correct\n"
    )

    predictions_df = add_baselines(predictions_df)
    metrics_df = macro_metrics(predictions_df)

    if len(views) > 1:
        # One table, not two: same/cross columns nested under each metric (module docstring
        # explains why the two are never averaged together into one number).
        metrics_df = metrics_df.unstack("view")
        metrics_df = metrics_df.reindex(columns=["macro_f1", "auroc", "accuracy"], level=0)
        metrics_df = metrics_df.reindex(columns=["same", "cross"], level=1)
    else:
        metrics_df = metrics_df.droplevel("view")

    print(metrics_df.round(3))

    if csv_path is not None:
        metrics_df.round(3).to_csv(csv_path)
        print(f"\nWrote metrics to {csv_path}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory of <model>/<entry>.json responses to score (default: %(default)s)",
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Directory of <entry>/{labels,perturbations}.json ground truth (default: %(default)s)",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optionally write the headline metrics to this CSV path")
    parser.add_argument(
        "--cross-view",
        action="store_true",
        help=(
            "Also include <entry>_view<n>.json cross-view responses (from run_benchmark.py --cross-view), "
            "shown alongside same-view performance in the same table"
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "Score only this many dataset entries, chosen the same way optimize_rendering.py's "
            "--n-entries does -- pass the same value (and --seed) used there to score exactly "
            "the entries a study was tuned against, instead of the whole dataset"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for --sample, matching optimize_rendering.py's --seed default (default: %(default)s)",
    )
    args = parser.parse_args()

    entry_names = None
    if args.sample is not None:
        entry_names = sample_entry_names(args.dataset_dir, args.sample, args.seed)
        print(f"Restricting to {len(entry_names)} entries sampled with --sample {args.sample} --seed {args.seed}\n")

    predictions_df = load_predictions(
        args.output_dir, args.dataset_dir, include_cross_view=args.cross_view, entry_names=entry_names
    )
    _print_report(predictions_df, args.csv)


if __name__ == "__main__":
    main()
