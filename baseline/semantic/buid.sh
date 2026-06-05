#!/bin/bash

# DATASETS=("amazon" "yelp" "goodreads")
DATASETS=("amazon")
EMBED_MODEL="sentence-transformers/all-mpnet-base-v2"

for ds in "${DATASETS[@]}"; do
    python3 "faiss_${ds}.py" \
        --data_path "../gcn/graph_data/item_${ds}_industrial.json" \
        --save_path "../MoE/faiss_dbs/${ds}_industrial_rich" \
        --embed_model "$EMBED_MODEL" \
        --batch_size $BATCH_SIZE
done
