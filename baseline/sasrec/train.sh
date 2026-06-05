#!/bin/bash

DATA_ROOT="../MoE/data"
OUTPUT_DIR="./saved_models"

# Hyperparameters
BATCH_SIZE=256       
LR=0.001
EPOCHS=50
HIDDEN_SIZE=32
NUM_HEADS=2     
DROPOUT=0.2
WEIGHT_DECAY=0.0001

DATASETS=( "yelp")
# DATASETS=("amazon") 

for DATASET in "${DATASETS[@]}"; do
    DATA_DIR="${DATA_ROOT}/${DATASET}"

    python train.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --weight_decay $WEIGHT_DECAY \
        --epochs $EPOCHS \
        --hidden_size $HIDDEN_SIZE \
        --num_heads $NUM_HEADS \
        --dropout $DROPOUT
done