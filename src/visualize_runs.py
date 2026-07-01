from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re

RUNS_DIR = Path("./runs_kimi")
DATASET_DIR = Path("./dataset")
OUTPUT = Path("./assets/runs_grid_kimi2.7.png")


def step_number(p: Path) -> int:
    m = re.search(r"color_(\d+)\.png$", p.name)
    return int(m.group(1)) if m else -1


def main():
    runs = sorted(d for d in RUNS_DIR.iterdir() if d.is_dir())

    # Determine max number of steps across all runs
    max_steps = max(
        len(list(run.glob("color_*.png"))) for run in runs
    )
    n_cols = 1 + max_steps  # reference + steps

    fig, axes = plt.subplots(
        len(runs), n_cols,
        figsize=(3 * n_cols, 3 * len(runs)),
        squeeze=False,
    )
    fig.subplots_adjust(wspace=0.02, hspace=0.1)

    for row, run in enumerate(runs):
        ref_path = DATASET_DIR / run.name / "color.png"
        step_images = sorted(run.glob("color_*.png"), key=step_number)

        ax = axes[row, 0]
        if ref_path.exists():
            ax.imshow(mpimg.imread(ref_path))
        else:
            ax.set_facecolor("#cccccc")
        ax.set_title("reference", fontsize=6, pad=2)
        ax.axis("off")

        if row == 0:
            ax.set_title(f"reference\n{run.name[:20]}", fontsize=5, pad=2)
        ax.set_ylabel(run.name[:30], fontsize=5, rotation=0, labelpad=60, va="center")

        for col, img_path in enumerate(step_images, start=1):
            ax = axes[row, col]
            ax.imshow(mpimg.imread(img_path))
            step = step_number(img_path)
            ax.set_title(f"step {step}", fontsize=6, pad=2)
            ax.axis("off")

        # blank out unused columns
        for col in range(len(step_images) + 1, n_cols):
            axes[row, col].axis("off")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
