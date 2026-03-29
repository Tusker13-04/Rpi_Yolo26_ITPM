#!/bin/bash
# ================================================================
# run_benchmark.sh — YOLOE-26 batch inference + performance plot
# Usage:
#   chmod +x run_benchmark.sh
#   ./run_benchmark.sh <image_dir> "<class1>.<class2>.<class3>"
# ================================================================

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <image_dir> \"<classes>\"" >&2
    echo "Example: $0 ./images \"person.car.bicycle\"" >&2
    exit 1
fi

IMAGE_DIR="$1"
CLASSES="$2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFER_SCRIPT="$SCRIPT_DIR/infer_single.py"
PLOT_SCRIPT="$SCRIPT_DIR/plot_benchmark.py"
CSV_OUT="$IMAGE_DIR/results.csv"
PLOT_OUT="$IMAGE_DIR/benchmark_plot.pdf"

if [ -f "$HOME/yoloe_env/bin/activate" ]; then
    source "$HOME/yoloe_env/bin/activate"
fi

echo "================================================================"
echo " YOLOE-26 Batch Benchmark"
echo "  Image directory : $IMAGE_DIR"
echo "  Classes         : $CLASSES"
echo "================================================================"
echo ""

echo "image,inference_ms,detections" > "$CSV_OUT"

shopt -s nullglob
IMAGES=("$IMAGE_DIR"/*.jpg "$IMAGE_DIR"/*.jpeg "$IMAGE_DIR"/*.png)
shopt -u nullglob

FILTERED=()
for img in "${IMAGES[@]}"; do
    base="$(basename "$img")"
    if [[ "$base" != *_yoloe* ]]; then
        FILTERED+=("$img")
    fi
done

N="${#FILTERED[@]}"
if [ "$N" -eq 0 ]; then
    echo "ERROR: No images found in $IMAGE_DIR" >&2
    exit 1
fi
echo "Found $N image(s) to process."
echo ""

PASS=0
FAIL=0

for i in "${!FILTERED[@]}"; do
    IMG="${FILTERED[$i]}"
    BASE="$(basename "$IMG")"
    NUM=$((i + 1))

    printf "[%d/%d] %-40s" "$NUM" "$N" "$BASE"

    RESULT=$(python3 "$INFER_SCRIPT" \
                --image  "$IMG" \
                --classes "$CLASSES" 2>/dev/null)

    if echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if 'error' not in d else 1)" 2>/dev/null; then
        INF_MS=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['inference_ms'])")
        DETS=$(echo  "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['detections'])")
        OUTPUT=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['output'])")

        printf " %7.1f ms  |  %2d det(s)  ->  %s\n" "$INF_MS" "$DETS" "$(basename "$OUTPUT")"

        echo "$BASE,$INF_MS,$DETS" >> "$CSV_OUT"
        PASS=$((PASS + 1))
    else
        printf " FAILED\n"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "Inference complete: $PASS succeeded, $FAIL failed."
echo ""

if [ "$PASS" -gt 0 ]; then
    echo "Generating benchmark plot..."
    python3 "$PLOT_SCRIPT" \
        --csv   "$CSV_OUT" \
        --out   "$PLOT_OUT" \
        --title "YOLOE-26s — Inference latency per image (RPi 4, CPU-only)"
    echo ""
fi

echo "================================================================"
echo " Summary"
echo "================================================================"
python3 - <<'PYEOF'
import csv, sys, os
csv_path = os.environ.get("CSV_OUT", "")
rows = []
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            rows.append(float(row["inference_ms"]))
        except (KeyError, ValueError):
            pass
if rows:
    import statistics
    print(f"  N images       : {len(rows)}")
    print(f"  Mean latency   : {statistics.mean(rows):.1f} ms")
    print(f"  Est. FPS       : {1000.0/statistics.mean(rows):.2f}")
PYEOF
export CSV_OUT
echo "================================================================"
