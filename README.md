# 3DFrontBench

A benchmark for evaluating VLM-driven 3D scene generation using the [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) dataset.
---

## Repository Structure

```
3DFrontBench/
├── src/
│   ├── dataset.py              # Eval3DFrontDataset loader
│   ├── vlm_score.py            # VLM-based scene quality scoring
│   ├── blender/                # Blender rendering utilities
│   └── ...
├── dataset/                    # 30 benchmark scenes (see Dataset section)
├── dataset_cam_params/         # Per-scene camera parameters (JSON)
├── prompts/                    # System prompts for VLM scoring
├── assets/                     # Visualizations and example images
└── outputs/                    # Generated scenes and VLM score results
```

---

## Dataset

The benchmark contains **30 indoor scenes** derived from 3D-FRONT, covering room types including Living Rooms, Living/Dining Rooms, Master Bedrooms, Hallways, and Corridors.

Each scene entry (named `<scene_uuid>_<RoomType-ID>`) provides:

| File | Description |
|---|---|
| `color.png` | Rendered RGB image from the fixed camera viewpoint |
| `depth.png` | Depth map of the scene |
| `color_bbox.png` | RGB render with ground-truth bounding boxes overlaid |
| `depth_bbox.png` | Depth map with bounding boxes |
| `color_scaffold.png` | RGB render with scaffold geometry |
| `depth_scaffold.png` | Depth map with scaffold geometry |
| `scene.glb` | Full 3D scene mesh (GLB) |
| `scene_bbox.glb` | Scene with bounding box geometry |
| `scene_scaffold.glb` | Scene with scaffold geometry |
| `metadata.json` | Furniture list with labels, 3D positions, rotations, scales, and visible item flags |
| `camera.json` | Camera intrinsics and extrinsics for the render viewpoint |
| `visible_furniture/` | Per-item visibility data |

Each `metadata.json` entry records the full furniture layout of the room, including which items are visible from the camera — these are the objects the generation agent is expected to reconstruct.

### Dataset Visualization

![Dataset Visualization](assets/dataset_visualization.png)

Each group of three columns shows, left to right: the RGB render, the depth map, and the bounding-box overlay for a scene in the benchmark.

---

## Installation

**Requirements:** Python 3.10+

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-org>/3DFrontBench.git
   cd 3DFrontBench
   ```

2. Create Python environment & Install dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Copy the environment template and fill in your credentials:
   ```bash
   cp .env.template .env
   ```
   Then edit `.env`:
   ```
   GEMINI_API_KEY=<your Gemini API key>
   GEMINI_PROJECT_NAME=<your GCP project name>
   GEMINI_PROJECT_NUMBER=<your GCP project number>
   ```

