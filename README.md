# YOLOE-26 on Raspberry Pi 4 - Edge Inference & Benchmarking

> **Run open-vocabulary YOLOE-26 (Nano / Small / Medium) on a Raspberry Pi 4 (CPU-only), benchmark per-image latency, and visualise results - all from a local Flask web UI or a single CLI command.**

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Hardware & Software Requirements](#hardware--software-requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Web UI (`app.py`)](#1-web-ui-apppy)
  - [2. Single-Image CLI (`infer_single.py`)](#2-single-image-cli-infer_singlepy)
  - [3. Batch Benchmark (`run_benchmark.sh`)](#3-batch-benchmark-run_benchmarksh)
  - [4. Latency Plot (`plot_benchmark.py`)](#4-latency-plot-plot_benchmarkpy)
  - [5. Accuracy vs Latency (`plot_device_accuracy.py`)](#5-accuracy-vs-latency-plot_device_accuracypy)
- [Output & Results](#output--results)
- [Screenshot](#screenshot)
- [Benchmark Methodology](#benchmark-methodology)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

This repository is part of the **ITPM project** and demonstrates how to deploy **YOLOE-26** - Ultralytics' open-vocabulary detection model - on a **Raspberry Pi 4** for real-world edge inference benchmarking.

Key goals:
- **Run YOLOE-26s/n/m on RPi 4 CPU** with no GPU acceleration.
- **Measure latency, FPS estimate, and detection count** per image.
- **Serve a browser-accessible web UI** on the local network for live image upload and annotation.
- **Generate publication-quality plots** comparing per-image latency and model accuracy vs. speed trade-offs across the YOLOE-26 variant family.

---

## Project Structure

```
Rpi_Yolo26_ITPM/
│
├── app.py                    # Flask web server - upload an image, see YOLOE-26 results
├── infer_single.py           # CLI: infer one image, output JSON + annotated file
├── run_benchmark.sh          # Bash: batch infer a whole directory, produce CSV + plot
├── plot_benchmark.py         # Python: plot per-image latency bar chart from CSV
├── plot_device_accuracy.py   # Python: Pareto curve - mAP vs latency for N/S/M variants
│
├── templates/
│   └── index.html            # Jinja2 template for the Flask web UI
│
├── images/                   # Drop test images here for batch benchmarking
│   └── rpi_accuracy_vs_latency.png   # (generated) accuracy–latency Pareto plot
│
└── runs/
    └── segment/              # Ultralytics validation run artefacts
        ├── val2/
        ├── val4/
        ├── val5/
        └── val6/
```

---

## Hardware & Software Requirements

| Item | Recommended |
|---|---|
| **Board** | Raspberry Pi 4 Model B (4 GB RAM recommended) |
| **OS** | Raspberry Pi OS (64-bit Bookworm) |
| **Python** | 3.9 – 3.11 |
| **Ultralytics** | ≥ 8.3 (YOLOE support) |
| **OpenCV** | `opencv-python-headless` |
| **Flask** | ≥ 3.0 |
| **matplotlib** | ≥ 3.7 |
| **psutil** | ≥ 5.9 |
| **numpy** | ≥ 1.24 |

> All inference runs **CPU-only** on the RPi 4. No CUDA, no NCNN, no ONNX runtime needed.

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/Tusker13-04/Rpi_Yolo26_ITPM.git
cd Rpi_Yolo26_ITPM

# 2. Create a virtual environment (recommended)
python3 -m venv ~/yoloe_env
source ~/yoloe_env/bin/activate

# 3. Install dependencies
pip install ultralytics flask opencv-python-headless psutil matplotlib numpy

# 4. Pre-download the YOLOE-26 small segmentation weights (auto on first run, or manually)
python3 -c "from ultralytics import YOLO; YOLO('yoloe-26s-seg.pt')"
```

---

## Usage

### 1. Web UI (`app.py`)

Launches a **Flask server on port 5000**, accessible from any browser on the same local network.

```bash
python3 app.py
# Access: http://<RPi4-IP-Address>:5000
```

**Features:**
- Upload any JPEG/PNG image (max 16 MB).
- Enter a **dot-separated class prompt** (e.g. `person.car.bicycle`) - YOLOE-26's open-vocabulary interface means you can query *any* class names.
- Adjust **confidence threshold** (`conf`, default `0.15`) and **IoU threshold** (`iou`, default `0.45`).
- Returns annotated image (bounding boxes + labels) rendered in the browser.
- **System stats panel**: CPU %, RAM used/total, CPU temperature (°C), Python version, platform.
- **Inference stats panel**: latency (ms), estimated FPS, number of detections, image resolution, per-box coordinates and confidence scores.

**API Endpoints:**

| Route | Method | Description |
|---|---|---|
| `/` | GET | Serve the web UI |
| `/system` | GET | Return live system stats as JSON |
| `/infer` | POST | Run YOLOE-26 inference; returns annotated image (base64) + stats JSON |

> The model is loaded **once at startup** and cached for all subsequent requests to minimise latency overhead.

---

### 2. Single-Image CLI (`infer_single.py`)

Run YOLOE-26 on a single image from the command line.

```bash
python3 infer_single.py \
    --image  path/to/image.jpg \
    --classes "person.car.bicycle" \
    --conf   0.15 \
    --iou    0.45 \
    --model  yoloe-26s-seg.pt
```

**Output:**
- Prints a **JSON object** to stdout:
  ```json
  {
    "image": "image.jpg",
    "output": "path/to/image_yoloe.jpg",
    "inference_ms": 512.3,
    "detections": 4,
    "labels": ["person", "car"],
    "image_wh": [1280, 720]
  }
  ```
- Saves an **annotated copy** as `<stem>_yoloe.<ext>` in the same directory.

**CLI Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--image` | *(required)* | Path to input image |
| `--classes` | *(required)* | Dot-separated class names |
| `--conf` | `0.15` | Detection confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--model` | `yoloe-26s-seg.pt` | YOLOE model weights file |

---

### 3. Batch Benchmark (`run_benchmark.sh`)

Runs `infer_single.py` over **all images** in a directory, collects latency data into a CSV, and auto-generates a benchmark plot.

```bash
chmod +x run_benchmark.sh
./run_benchmark.sh ./images "person.car.bicycle"
```

**What it does:**
1. Finds all `.jpg`, `.jpeg`, `.png` files in the given directory (skips `*_yoloe.*` outputs).
2. Calls `infer_single.py` for each image, recording `inference_ms` and detection count.
3. Writes `results.csv` to the image directory.
4. Calls `plot_benchmark.py` to generate `benchmark_plot.pdf` + `.png`.
5. Prints a **summary block** - N images, mean latency, estimated FPS - to the terminal.

**Output files:**
```
images/
├── results.csv            # image, inference_ms, detections
└── benchmark_plot.pdf     # latency bar chart (also .png)
```

Automatically activates `~/yoloe_env` if it exists.

---

### 4. Latency Plot (`plot_benchmark.py`)

Standalone script to regenerate the benchmark bar chart from an existing `results.csv`.

```bash
python3 plot_benchmark.py \
    --csv   images/results.csv \
    --out   images/benchmark_plot.pdf \
    --title "YOLOE-26s - Inference latency per image (RPi 4, CPU-only)"
```

**Generated chart includes:**
- Horizontal bar chart: per-image latency (ms), sorted by image name.
- Red dashed mean-latency line with label.
- Summary statistics table: N, mean, std dev, min, max, estimated FPS, total detections.
- Saved as **both PDF and PNG** (300 DPI).

---

### 5. Accuracy vs Latency (`plot_device_accuracy.py`)

Evaluates all three YOLOE-26 variants - **Nano, Small, Medium** - on `coco8-seg.yaml` (COCO 8-image subset) at `imgsz=640, batch=1` on the RPi 4 CPU, then plots a **Pareto frontier** graph.

```bash
python3 plot_device_accuracy.py
```

**Output:**
- `images/rpi_accuracy_vs_latency.png`
- `images/rpi_accuracy_vs_latency.pdf`

The X-axis shows CPU inference latency (ms, lower is better); the Y-axis shows `mAP50-95` (%, higher is better). Each variant (26N, 26S, 26M) is annotated as a data point.

> **Note:** This script runs full Ultralytics `.val()` evaluation and may take **several minutes per model** on a RPi 4.

---

## Output & Results

After a full benchmark run you will have:

```
images/
├── <img>_yoloe.jpg              # Annotated image outputs (per infer_single run)
├── results.csv                  # Batch latency CSV
├── benchmark_plot.pdf/.png      # Per-image latency bar chart
├── rpi_accuracy_vs_latency.pdf  # Pareto: mAP vs latency for N/S/M
└── rpi_accuracy_vs_latency.png

runs/segment/
├── val2/   # Ultralytics val run artefacts
├── val4/
├── val5/
└── val6/
```

---
## Screenshot
<img width="1919" height="1024" alt="image" src="https://github.com/user-attachments/assets/a6f7266b-1e81-4990-813f-924bd3af4fbe" />


## Benchmark Methodology

- **Device**: Raspberry Pi 4 Model B, CPU-only (no hardware acceleration).
- **Model**: `yoloe-26s-seg.pt` (default), with variants `yoloe-26n-seg.pt` and `yoloe-26m-seg.pt` tested in accuracy plots.
- **Timing**: `time.perf_counter()` wraps a single `model.predict()` call per image - includes preprocessing, forward pass, and NMS.
- **Batch size**: 1 (sequential edge inference).
- **Input resolution**: native image resolution for single-image mode; `imgsz=640` for validation.
- **FPS estimate**: `1000 / mean_inference_ms` (single-thread, no pipeline overlap).
- **Classes**: open-vocabulary, specified at runtime via dot-separated prompt string.

---

## Configuration Reference

| Parameter | Where | Default | Notes |
|---|---|---|---|
| `model` | all scripts | `yoloe-26s-seg.pt` | Swap for `yoloe-26n-seg.pt` (faster) or `yoloe-26m-seg.pt` (more accurate) |
| `conf` | `app.py`, `infer_single.py` | `0.15` | Lower = more detections, higher = fewer false positives |
| `iou` | `app.py`, `infer_single.py` | `0.45` | NMS overlap threshold |
| `max_det` | `app.py`, `infer_single.py` | `100` | Maximum detections per image |
| `MAX_CONTENT_LENGTH` | `app.py` | `16 MB` | Max upload size for web UI |
| `port` | `app.py` | `5000` | Flask server port |

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: ultralytics` | Run `pip install ultralytics` inside your venv |
| Model download hangs | Ensure internet access on first run; weights auto-download from Ultralytics Hub |
| `/sys/class/thermal/...` error | CPU temp read is RPi-specific; returns `"N/A"` on non-RPi hardware automatically |
| Flask server not reachable | Check RPi firewall, ensure `host="0.0.0.0"` is set (it is by default) |
| `plot_device_accuracy.py` takes too long | Expected - full `.val()` on CPU can take 5–15 min per model on RPi 4 |
| `run_benchmark.sh: CSV_OUT not exported` | The `export CSV_OUT` at the end of the script handles this; ensure bash, not sh |

---

> **Model weights** (`*.pt` files) are not committed to this repository. They are downloaded automatically by Ultralytics on first use from [https://docs.ultralytics.com](https://docs.ultralytics.com).
