#!/bin/bash

# Lấy tên dataset từ tham số thứ nhất, mặc định là 'amazon'
DATASET=${1:-amazon}
# Lấy dimension từ tham số thứ hai, mặc định là 64
EMB_DIM=${2:-64}

echo "--- Đang xử lý Remap GCN Embedding ---"
echo "Dataset: $DATASET"
echo "Embedding Dim: $EMB_DIM"

python3 remap_gcn_embedding.py \
    --gcn_path     "../MoE/saved_models/${DATASET}_gcn_emb.pt" \
    --id2rawid     "../MoE/data/${DATASET}/id2rawid.txt" \
    --output_path  "./gcn_embedding/${DATASET}_gcn_emb_remapped.pt" \
    --emb_dim      "${EMB_DIM}"

echo "Hoàn thành! File đã lưu tại: ./gcn_embedding/${DATASET}_gcn_emb_remapped.pt"