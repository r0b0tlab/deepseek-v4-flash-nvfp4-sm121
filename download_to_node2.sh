#!/bin/bash
# Download all safetensors files from node1 HTTP server
set -euo pipefail

BASE_URL="http://192.168.100.10:8080"
OUTDIR="$HOME/deepseek-v4-nvfp4/weights"
mkdir -p "$OUTDIR"
cd "$OUTDIR"

# Download index file first if not exists
if [ ! -f "model.safetensors.index.json" ]; then
    wget -q "$BASE_URL/model.safetensors.index.json" -O model.safetensors.index.json
fi

# Download all safetensors files (up to 8 parallel)
for i in $(seq -w 2 46); do
    FILE="model-${i}-of-00046.safetensors"
    if [ ! -f "$FILE" ]; then
        wget -q "$BASE_URL/$FILE" -O "$FILE" &
    fi
    # Limit parallel jobs to 8
    if [ $(jobs -r | wc -l) -ge 8 ]; then
        wait -n
    fi
done

# Wait for all background jobs
wait

# Download config files
for f in config.json hf_quant_config.json generation_config.json README.md LICENSE .gitattributes; do
    if [ ! -f "$f" ]; then
        wget -q "$BASE_URL/$f" -O "$f" || true
    fi
done

echo "Download complete. Files:"
ls -la *.safetensors | wc -l
du -sh .
