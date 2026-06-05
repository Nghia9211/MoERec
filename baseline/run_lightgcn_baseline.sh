#!/bin/bash
PYTHON="/home/research/nghialt/.venv/bin/python"
DATASETS=("yelp" "amazon" "goodreads" "amazon_musical" "amazon_industrial")
SCENARIOS=("classic" "user_cold_start" )
NUM_TASKS="None"    

for SCENARIO in "${SCENARIOS[@]}"; do
    for TASK_SET in "${DATASETS[@]}"; do
        $PYTHON LightGCN_baseline.py \
            --task_set "$TASK_SET" \
            --scenario "$SCENARIO" \
            --num_tasks "$NUM_TASKS"

        if [ $? -ne 0 ]; then
            exit 1
        fi
    done
done
