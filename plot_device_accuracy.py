#!/usr/bin/env python3
"""
plot_device_accuracy.py
Evaluates YOLOE-26 (Nano, Small, Medium) on the COCO8 dataset on-device.
Measures physical mAP and latency, then generates a Pareto frontier graph.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLOE

def main():
    # Correct model names for the open-vocabulary branch of YOLO-26
    models_to_test = ["yoloe-26n-seg.pt", "yoloe-26s-seg.pt", "yoloe-26m-seg.pt"]
    
    results_map = []
    results_lat = []
    labels = []

    print("================================================================")
    print(" On-Device Accuracy vs Latency Profiling (YOLOE-26)")
    print("================================================================")
    
    for model_name in models_to_test:
        print(f"\nEvaluating {model_name}...")
        try:
            # Using the specific YOLOE loader for open-vocabulary weights
            model = YOLOE(model_name)
            
            # imgsz=640 is standard benchmark size
            # batch=1 forces sequential inference to get accurate edge latency
            metrics = model.val(data="coco8-seg.yaml", imgsz=640, batch=1, device="cpu", verbose=False)
            
            # Extract mAP50-95 for bounding boxes (convert to percentage)
            map_val = metrics.box.map * 100  
            # Extract inference time per image in milliseconds
            lat_val = metrics.speed['inference']
            
            results_map.append(map_val)
            results_lat.append(lat_val)
            
            # Clean up the label for the graph (e.g. "YOLOE-26N")
            clean_label = model_name.replace('-seg.pt', '').upper()
            labels.append(clean_label)
            
            print(f" -> mAP: {map_val:.1f}% | Latency: {lat_val:.1f} ms")
        except Exception as e:
            print(f" [!] Failed to evaluate {model_name}: {e}")

    if len(results_map) < 2:
        print("\nNot enough data points to plot a curve.")
        return

    # ── Plotting ───────────────────────────────────────────────────
    print("\nGenerating matplotlib graphs...")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", 
        "font.size": 10,
        "axes.facecolor": "#f8f9fa",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    # Plot the YOLOE-26 series curve
    ax.plot(results_lat, results_map, color="#3b82f6", linestyle="-", linewidth=1.5, zorder=1)
    ax.scatter(results_lat, results_map, color="#2563eb", s=100, marker="s", zorder=2)
    
    # Annotate points with model names
    for i, txt in enumerate(labels):
        ax.annotate(txt, (results_lat[i], results_map[i]), textcoords="offset points", xytext=(0, 10), 
                    ha='center', fontsize=9, fontweight="bold", color="#1d4ed8")
    
    ax.set_title("On-Device Accuracy vs Latency (Raspberry Pi 4 CPU)\nCOCO Subset mAP50-95 vs Inference ms", pad=15)
    ax.set_xlabel("CPU Inference Latency (ms) [Lower is better]")
    ax.set_ylabel("Mean Average Precision (mAP) % [Higher is better]")
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Save outputs
    os.makedirs("images", exist_ok=True)
    out_png = "images/rpi_accuracy_vs_latency.png"
    out_pdf = "images/rpi_accuracy_vs_latency.pdf"
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    print("================================================================")
    print(f" Done! Plots successfully saved to:")
    print(f" - {out_png}")
    print(f" - {out_pdf}")
    print("================================================================")

if __name__ == "__main__":
    main()
