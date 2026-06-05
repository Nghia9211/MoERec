#!/bin/bash

DATASET=${1:-amazon}
EMB_DIM=${2:-64}

echo "--- Đang xử lý Remap GCN Embedding ---"
echo "Dataset: $DATASET"
echo "Embedding Dim: $EMB_DIM"

python3 remap_gcn_embedding.py \
    --gcn_path     "./gcn_embedding/${DATASET}_gcn_emb.pt" \
    --id2rawid     "../MoE/data/${DATASET}/id2rawid.txt" \
    --output_path  "../MoE/saved_models/light_gcn/${DATASET}_gcn_emb_remapped.pt" \
    --emb_dim      "${EMB_DIM}"

echo "Hoàn thành! File đã lưu tại: ./gcn_embedding/${DATASET}_gcn_emb_remapped.pt"
