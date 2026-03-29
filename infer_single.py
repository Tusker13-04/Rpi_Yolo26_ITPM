#!/usr/bin/env python3
"""
infer_single.py
Usage:
    python3 infer_single.py --image path/to/img.jpg --classes "person.car.bicycle"
Output (stdout):
    JSON: {"image": "...", "inference_ms": 512.3, "detections": 4, "labels": [...]}
Saved:
    <same_dir>/<stem>_yoloe.jpg  — annotated bounding boxes
"""

import argparse, json, sys, time
from pathlib import Path
import cv2
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",   required=True, help="Path to input image")
    p.add_argument("--classes", required=True, help="Dot-separated class names, e.g. 'person.car'")
    p.add_argument("--conf",    type=float, default=0.15)
    p.add_argument("--iou",     type=float, default=0.45)
    p.add_argument("--model",   default="yoloe-26s-seg.pt")
    return p.parse_args()

def main():
    args = parse_args()
    img_path = Path(args.image).resolve()
    if not img_path.exists():
        print(json.dumps({"error": f"File not found: {img_path}"}))
        sys.exit(1)

    classes = [c.strip() for c in args.classes.split(".") if c.strip()]

    from ultralytics import YOLO
    model = YOLO(args.model)
    model.set_classes(classes)

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(json.dumps({"error": f"Cannot read image: {img_path}"}))
        sys.exit(1)
    h, w = img_bgr.shape[:2]

    # ── Inference ────────────────────────────────────────────────
    t0 = time.perf_counter()
    results = model.predict(img_bgr, conf=args.conf, iou=args.iou,
                            max_det=100, verbose=False)
    t1 = time.perf_counter()
    inference_ms = round((t1 - t0) * 1000, 2)

    boxes      = results[0].boxes
    detections = []
    np.random.seed(0)
    palette = {}

    # ── Draw bounding boxes ──────────────────────────────────────
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf_val = float(box.conf[0])
        cls_idx  = int(box.cls[0])
        label    = classes[cls_idx] if cls_idx < len(classes) else f"cls_{cls_idx}"

        if label not in palette:
            palette[label] = tuple(int(c) for c in np.random.randint(100, 220, 3))
        color = palette[label]

        thick = max(1, int(min(h, w) * 0.003))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thick)

        txt = f"{label} {conf_val:.2f}"
        fs  = max(0.35, min(h, w) * 0.0007)
        ft  = max(1, thick // 2)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
        cv2.rectangle(img_bgr, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img_bgr, txt, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), ft)

        detections.append({"label": label, "confidence": round(conf_val, 4),
                           "box": [int(x1), int(y1), int(x2), int(y2)]})

    # ── Save annotated image ─────────────────────────────────────
    out_path = img_path.parent / (img_path.stem + "_yoloe" + img_path.suffix)
    cv2.imwrite(str(out_path), img_bgr)

    result = {
        "image":        img_path.name,
        "output":       str(out_path),
        "inference_ms": inference_ms,
        "detections":   len(detections),
        "labels":       [d["label"] for d in detections],
        "image_wh":     [w, h],
    }
    print(json.dumps(result))

if __name__ == "__main__":
    main()
