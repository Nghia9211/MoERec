#!/bin/bash
# run_moe_all.sh — MoE pipeline runner across datasets and scenarios

set -e

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for _ENV_FILE in "$_SCRIPT_DIR/.env" "$HOME/.env"; do
    if [ -f "$_ENV_FILE" ]; then
        echo "[ENV] Loading API keys from: $_ENV_FILE"
        while IFS= read -r _line || [ -n "$_line" ]; do
            [[ "$_line" =~ ^[[:space:]]*# ]] && continue
            [[ -z "${_line// }" ]] && continue
            if [[ "$_line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
                export "$_line"
            fi
        done < "$_ENV_FILE"
        break
    fi
done

MOE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
cd "$MOE_DIR"


# DATASETS=(\"goodreads\" \"yelp\" \"amazon\" \"amazon_musical\" \"amazon_industrial\")
# DATASETS=( \"amazon\" \"yelp\")
# DATASETS=( \"goodreads\" \"amazon\")
DATASETS=("goodreads")
# DATASETS=("goodreads" "yelp" "amazon")
# DATASETS=("amazon_industrial")

SCENARIOS=("user_cold_start" "classic")
# SCENARIOS=("classic" )

STAGE="test"
CANS_NUM=20
MAX_EPOCH=5
MAX_SAMPLES=-1
MP=32
SEED=303
TEMPERATURE="0.0"
RERANKER_MODE="llm"
USE_RERANKER="false"
USE_USER_AGENT="false"

USE_OPENAI_API="false"

if [ "$USE_OPENAI_API" == "true" ]; then
    # MODEL: gpt-4o-mini (~$6 full run) | gpt-4o (~$97) | gpt-4.1-mini (~$16)
    MODEL="gpt-4o-mini"
    API_KEY="${OPENAI_API_KEY:-FILL_YOUR_KEY_HERE}"
    BASE_URL=""
    MP=32
    echo "[CONFIG] Using OpenAI API | model=$MODEL | key=${API_KEY:0:12}..."
else
    # Cloudflare VLLM URL
    # MODEL="${MODEL:-qwen-small}"
    # API_KEY="${API_KEY:-empty}"
    # BASE_URL="https://sheep-pleasant-settled-synthesis.trycloudflare.com/v1"
    MODEL="${MODEL:-qwen-research}"
    API_KEY="${API_KEY:-EMPTY}"
    BASE_URL="http://localhost:11435/v1"
    echo "[CONFIG] Using local vLLM | model=$MODEL | url=$BASE_URL"
fi

for DS in "${DATASETS[@]}"; do
    for SCENARIO in "${SCENARIOS[@]}"; do
        echo ""
        echo "############################################################"
        echo "  RUNNING MOE PIPELINE FOR: $DS | SCENARIO: $SCENARIO"
        echo "############################################################"

        DATA_DIR="./data/${DS}/"
        MODEL_PATH="./saved_models/${DS}_best_model.pt"
        CANDIDATE_DIR="../dataset/tasks5/${SCENARIO}/${DS}/tasks"
        FAISS_DB_PATH="./faiss_dbs/${DS}_rich"

        GCN_PATH="./saved_models/${DS}_gcn_emb_remapped.pt"
        GATING_MODEL_PATH="./saved_models/moe_fix_gcn/${DS}/ce/${DS}_gating_model.pt"

        # Unified item file path — all datasets use dataset/raw/{DS}/item.json (symlinks)
        ITEMFILE="../dataset/raw/${DS}/item.json"

        if [ "$DS" == "amazon_musical" ] || [ "$DS" == "amazon_industrial" ]; then
            INPUT_JSON_FILE="./data/groundtruth_music_industrial.json"
        else
            INPUT_JSON_FILE="./data/ground_truth.json"
        fi

        P_MODEL="SASRec_MoE"
        NAME="moe_only_sasrec"
        OUTPUT_FILE="./output/experts/${DS}_${SCENARIO}_${NAME}/${P_MODEL}_${MODEL}_SEED${SEED}_ep${MAX_EPOCH}.jsonl"
        RESULT_FILE="./output/experts/${DS}_${SCENARIO}_${NAME}/evaluation_results_${NAME}_${DS}.json"
        mkdir -p "$(dirname "$OUTPUT_FILE")"
        mkdir -p "$(dirname "$RESULT_FILE")"

        start_time=$(date +%s)
        echo ">>> Start running $DS - $SCENARIO at $(date)"

        /home/research/nghialt/.venv/bin/python ./main_moe.py \
            --data_dir="$DATA_DIR" \
            --model_path="$MODEL_PATH" \
            --input_json_file="$INPUT_JSON_FILE" \
            --dataset="$DS" \
            --stage="$STAGE" \
            --cans_num=$CANS_NUM \
            --max_epoch=$MAX_EPOCH \
            --max_samples=$MAX_SAMPLES \
            --candidate_dir="$CANDIDATE_DIR" \
            --item_mapping_file="$ITEMFILE" \
            --faiss_db_path="$FAISS_DB_PATH" \
            --gcn_path="$GCN_PATH" \
            --embed_model_name="sentence-transformers/all-mpnet-base-v2" \
            --gating_model_path="$GATING_MODEL_PATH" \
            --reranker_mode="$RERANKER_MODE" \
            --use_reranker="$USE_RERANKER" \
            --use_user_agent="$USE_USER_AGENT" \
            --reranker_top_llm=10 \
            --model="$MODEL" \
            --api_key="$API_KEY" \
            --base_url="$BASE_URL" \
            --seed=$SEED \
            --mp=$MP \
            --temperature=$TEMPERATURE \
            --output_file="$OUTPUT_FILE" \
            --result_file="$RESULT_FILE" \
            --save_info \
            --save_rec_dir="$SAVE_REC_DIR"

        end_time=$(date +%s)
        duration=$((end_time - start_time))

        TIME_LOG="$(dirname "$OUTPUT_FILE")/execution_time.txt"
        echo "============================================================" >> "$TIME_LOG"
        echo "Dataset: $DS" >> "$TIME_LOG"
        echo "Scenario: $SCENARIO" >> "$TIME_LOG"
        echo "Start: $(date -d @$start_time)" >> "$TIME_LOG"
        echo "End:   $(date -d @$end_time)" >> "$TIME_LOG"
        echo "Duration: ${duration}s ($((duration/60))m $((duration%60))s)" >> "$TIME_LOG"
        echo "============================================================" >> "$TIME_LOG"

        echo ">>> Finished $DS - $SCENARIO. Duration: ${duration}s. Results: $OUTPUT_FILE"

        echo ">>> Cooling down: Sleeping for 30 minutes before the next run..."
        # sleep 1800
    done
done

echo ""
echo "============================================================"
echo "  ALL DATASETS AND SCENARIOS COMPLETED."
echo "============================================================"
