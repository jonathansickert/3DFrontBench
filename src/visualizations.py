from src.dataset import Eval3DFrontDataset
import matplotlib.pyplot as plt


def visualize_dataset(dataset: Eval3DFrontDataset):
    fig, axes = plt.subplots(10, 12, figsize=(36, 15))
    fig.subplots_adjust(wspace=0.02, hspace=0.02)

    for i, scene_data in enumerate(dataset):
        r = (i // 6) * 2
        c = (i % 6) * 2

        color = scene_data["color"]
        depth = scene_data["depth"]
        color_bbox = scene_data["color_bbox"]
        depth_bbox = scene_data["depth_bbox"]
        name = scene_data["scene_id"]

        axes[r, c].imshow(color)
        axes[r, c + 1].imshow(depth)
        axes[r + 1, c].imshow(color_bbox)
        axes[r + 1, c + 1].imshow(depth_bbox)
        axes[r, c].set_title(name, fontsize=5, pad=1)

        for ax in [axes[r, c], axes[r, c + 1], axes[r + 1, c], axes[r + 1, c + 1]]:
            ax.axis("off")


    fig.savefig("./assets/dataset_visualization.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    dataset = Eval3DFrontDataset("./dataset")
    visualize_dataset(dataset=dataset)


