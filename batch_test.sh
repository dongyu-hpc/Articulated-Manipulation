#!/bin/bash
set -e

# 代理与环境变量
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export HF_ENDPOINT=https://hf-mirror.com
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
export MAX_JOB=1

# 彩色输出
INFO() { echo -e "\033[1;32m[INFO]\033[0m $*"; }
STEP() { echo -e "\033[1;34m[STEP]\033[0m $*"; }

SCANS_DIR="/home/gdy/rsrd/dataset/nerf_recon_dataset/corl_data/scans"
DEMO_DIR="/home/gdy/rsrd/dataset/nerf_recon_dataset/corl_data/demonstrations"
OUTPUT_BASE="/home/gdy/rsrd/outputs/$(date +%Y%m%d)"

mkdir -p "$OUTPUT_BASE"
INFO "Output base: $OUTPUT_BASE"

OBJECTS=(nerfgun redbox scissors stapler sunglasses usb_plug wire_cutters led_light bear)

for OBJ in "${OBJECTS[@]}"; do
    INFO "=========================================="
    INFO "Processing: $OBJ"
    INFO "=========================================="

    STEP "[1/4] ns-train garfield for $OBJ..."
    ns-train garfield \
        --data "$SCANS_DIR/$OBJ" \
        --output-dir "$OUTPUT_BASE" \
        --experiment-name "$OBJ"

    GARFIELD_RUN=$(ls -td "$OUTPUT_BASE/$OBJ/garfield/"*/ | head -1)
    GARFIELD_CKPT="${GARFIELD_RUN}config.yml"
    INFO "Garfield ckpt: $GARFIELD_CKPT"

    STEP "[2/4] ns-train dig for $OBJ..."
    MAX_JOB=1 ns-train dig \
        --data "$SCANS_DIR/$OBJ" \
        --output-dir "$OUTPUT_BASE" \
        --experiment-name "$OBJ" \
        --pipeline.garfield-ckpt "$GARFIELD_CKPT"

    DIG_RUN=$(ls -td "$OUTPUT_BASE/$OBJ/dig/"*/ | head -1)
    DIG_CFG="${DIG_RUN}config.yml"
    INFO "DIG config: $DIG_CFG"

    STEP "[3/4] run_tracker for $OBJ..."
    python scripts/run_tracker.py \
        --is-obj-jointed False \
        --dig-config-path "$DIG_CFG" \
        --video-path "$DEMO_DIR/$OBJ.MOV" \
        --output-dir "$OUTPUT_BASE/$OBJ" \
        --no-save-hand

    STEP "[4/4] run_planner for $OBJ..."
    timeout 120 python scripts/run_planner.py \
        --hand-mode single \
        --track-dir "$OUTPUT_BASE/$OBJ" || true

    INFO "Done: $OBJ"
    echo ""
done

INFO "All objects processed."
