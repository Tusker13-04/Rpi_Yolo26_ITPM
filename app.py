#!/usr/bin/env python3
"""
YOLOE-26 Edge Inference Web UI
Runs on Raspberry Pi 4 | Accessible from any browser on the local network
Access: http://<RPi4-IP-Address>:5000
"""

import io, os, time, base64, json, psutil, platform, subprocess
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# ── Model cache (load once, reuse) ──────────────────────────────────────────
_model = None

def get_model(model_name="yoloe-26s-seg.pt"):
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(model_name)
    return _model

# ── Helpers ─────────────────────────────────────────────────────────────────
def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read()) / 1000, 1)
    except Exception:
        return "N/A"

def get_system_stats():
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "ram_used_mb": round(mem.used / 1024**2, 1),
        "ram_total_mb": round(mem.total / 1024**2, 1),
        "ram_percent": mem.percent,
        "cpu_temp": get_cpu_temp(),
        "platform": platform.machine(),
        "python": platform.python_version(),
    }

def run_inference(image_bytes, prompt, conf=0.15, iou=0.45):
    """Run YOLOE-26 inference and return annotated image + stats."""
    model = get_model()

    # Parse classes from prompt
    classes = [c.strip() for c in prompt.split(".") if c.strip()]
    if not classes:
        return None, {"error": "Empty prompt"}

    model.set_classes(classes)

    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = img_bgr.shape[:2]

    # ── Inference ────────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    results = model.predict(img_bgr, conf=conf, iou=iou, max_det=100, verbose=False)
    t_end = time.perf_counter()

    inference_ms = round((t_end - t_start) * 1000, 1)
    fps_estimate  = round(1000 / inference_ms, 2) if inference_ms > 0 else 0

    boxes = results[0].boxes
    detections = []

    # ── Draw annotations ─────────────────────────────────────────────────────
    np.random.seed(42)
    palette = {}
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf_val = float(box.conf[0])
        cls_idx  = int(box.cls[0])
        label    = classes[cls_idx] if cls_idx < len(classes) else f"cls_{cls_idx}"

        if label not in palette:
            palette[label] = [int(c) for c in np.random.randint(80, 230, 3)]
        color = palette[label]

        thickness = max(2, int(min(h, w) * 0.004))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)

        txt  = f"{label} {conf_val:.2f}"
        fs   = max(0.45, min(h, w) * 0.0009)
        ft   = max(1, thickness // 2)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
        cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img_bgr, txt, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), ft)

        detections.append({
            "label": label,
            "confidence": round(conf_val, 4),
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "width_px":  int(x2 - x1),
            "height_px": int(y2 - y1),
        })

    # ── Encode to base64 PNG ──────────────────────────────────────────────────
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img_b64 = base64.b64encode(buf).decode()

    stats = {
        "inference_ms":    inference_ms,
        "fps_estimate":    fps_estimate,
        "num_detections":  len(detections),
        "image_size":      f"{w}×{h}",
        "classes_queried": classes,
        "detections":      detections,
    }
    return img_b64, stats


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/system")
def system():
    return jsonify(get_system_stats())

@app.route("/infer", methods=["POST"])
def infer():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    image_bytes = request.files["image"].read()
    conf = float(request.form.get("conf", 0.15))
    iou  = float(request.form.get("iou",  0.45))

    sys_before = get_system_stats()
    img_b64, stats = run_inference(image_bytes, prompt, conf, iou)
    sys_after  = get_system_stats()

    stats["system"] = {
        "cpu_temp_c":   sys_after["cpu_temp"],
        "cpu_percent":  sys_after["cpu_percent"],
        "ram_used_mb":  sys_after["ram_used_mb"],
        "ram_percent":  sys_after["ram_percent"],
    }

    return jsonify({"image": img_b64, "stats": stats})


if __name__ == "__main__":
    print("=" * 55)
    print(" YOLOE-26 Web UI — Raspberry Pi 4 Edge Deployment")
    print(f" Access at:  http://0.0.0.0:5000")
    print(f" Pre-loading model...")
    get_model()
    print(" Model ready. Starting server...")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
