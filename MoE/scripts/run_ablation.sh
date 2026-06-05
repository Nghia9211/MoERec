#!/bin/bash
# run_ablation.sh — Ablation study for Reranker & Feedback Loop
# Sweeps 3 hyperparameters:
#   (a) Fusion weight α ∈ {0.1, 0.2, ..., 0.9}
#   (b) Pool size M     ∈ {10, 15, 20, 25, 30}
#   (c) T_max (rounds)  ∈ {1, 2, 3, 4, 5}
#
# Results saved to: ./output/ablation/amazon_industrial_classic/
# After run: python plot_ablation.py

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

# ── Fixed dataset & scenario for ablation ────────────────────────────────────
ABL_DS="amazon_industrial"
ABL_SCENARIO="classic"

# ── Run parameters (sync with run_moe.sh) ────────────────────────────────────
MAX_EPOCH=5
CANS_NUM=20
MAX_SAMPLES=-1
MP=12
SEED=303
TEMPERATURE="0.0"
RERANKER_MODE="llm"
USE_RERANKER="true"
USE_USER_AGENT="false"    # false for alpha & M sweep (1 round, no feedback loop)
                          # Tmax sweep will override to true

# ── Sweep values ──────────────────────────────────────────────────────────────
ALPHA_VALUES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)  # Chart (a)
M_VALUES=(5 10 15 20)                                  # Chart (b)
TMAX_VALUES=(1 2 3 4 5)                                # Chart (c)

# ── Which sweeps to run ───────────────────────────────────────────────────────
RUN_ALPHA_SWEEP="false"
RUN_M_SWEEP="true"
RUN_TMAX_SWEEP="true"

# ── LLM Backend ───────────────────────────────────────────────────────────────
USE_OPENAI_API="false"

if [ "$USE_OPENAI_API" == "true" ]; then
    MODEL="gpt-4o-mini"
    API_KEY="${OPENAI_API_KEY:-FILL_YOUR_KEY_HERE}"
    BASE_URL=""
    MP=32
    echo "[CONFIG] Using OpenAI API | model=$MODEL | key=${API_KEY:0:12}..."
else
    MODEL="${MODEL:-qwen-research}"
    API_KEY="${API_KEY:-EMPTY}"
    BASE_URL="http://localhost:11435/v1"
    echo "[CONFIG] Using local vLLM | model=$MODEL | url=$BASE_URL"
fi

# ── Paths (sync with run_moe.sh) ──────────────────────────────────────────────
DATA_DIR="./data/${ABL_DS}/"
MODEL_PATH="./saved_models/${ABL_DS}_best_model.pt"
CANDIDATE_DIR="../dataset/tasks5/${ABL_SCENARIO}/${ABL_DS}/tasks"
FAISS_DB_PATH="./faiss_dbs/${ABL_DS}_rich"
GCN_PATH="./saved_models/${ABL_DS}_gcn_emb_remapped.pt"
GATING_MODEL_PATH="./saved_models/moe_fix_gcn/${ABL_DS}/ce/${ABL_DS}_gating_model.pt"
# Unified item file path — uses dataset/raw/{DS}/item.json (symlinks)
ITEMFILE="../dataset/raw/${ABL_DS}/item.json"
INPUT_JSON_FILE="./data/groundtruth_music_industrial.json"

_run_ablation() {
    local sweep_name="$1"
    local sweep_val="$2"
    local extra_args="$3"

    local OUT_DIR="./output/ablation/${ABL_DS}_${ABL_SCENARIO}/${sweep_name}_${sweep_val}"
    local OUT_FILE="${OUT_DIR}/SASRec_MoE_${MODEL}_ablation.jsonl"
    local RES_FILE="${OUT_DIR}/evaluation_results_ablation.json"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [Ablation] Sweep: ${sweep_name} = ${sweep_val}"
    echo "──────────────────────────────────────────────────────────────"
    local t0=$(date +%s)

    /home/research/nghialt/.venv/bin/python ./main_moe.py \
        --data_dir="$DATA_DIR" \
        --model_path="$MODEL_PATH" \
        --input_json_file="$INPUT_JSON_FILE" \
        --dataset="$ABL_DS" \
        --stage="test" \
        --cans_num=$CANS_NUM \
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
        --output_file="$OUT_FILE" \
        --result_file="$RES_FILE" \
        --save_info \
        $extra_args

    local t1=$(date +%s)
    local dur=$((t1 - t0))
    echo ">>> [Ablation] ${sweep_name}=${sweep_val} done in ${dur}s → $RES_FILE"

    local TIME_LOG="${OUT_DIR}/execution_time.txt"
    echo "============================================================" >> "$TIME_LOG"
    echo "Ablation: ${sweep_name} = ${sweep_val}"                       >> "$TIME_LOG"
    echo "Dataset:  $ABL_DS | Scenario: $ABL_SCENARIO"                 >> "$TIME_LOG"
    echo "Start:    $(date -d @$t0)"                                    >> "$TIME_LOG"
    echo "End:      $(date -d @$t1)"                                    >> "$TIME_LOG"
    echo "Duration: ${dur}s ($((dur/60))m $((dur%60))s)"               >> "$TIME_LOG"
    echo "============================================================" >> "$TIME_LOG"
}

echo ""
echo "████████████████████████████████████████████████████████████████"
echo "  ABLATION STUDY — Dataset: $ABL_DS | Scenario: $ABL_SCENARIO"
echo "████████████████████████████████████████████████████████████████"

# ── (a) Alpha Sweep ──────────────────────────────────────────────────────────
if [ "$RUN_ALPHA_SWEEP" == "true" ]; then
    echo ""
    echo "════════ (a) Fusion weight α sweep ════════"
    for ALPHA in "${ALPHA_VALUES[@]}"; do
        _run_ablation "alpha" "$ALPHA" "--alpha=$ALPHA --max_epoch=$MAX_EPOCH"
    done
    echo "✅ Alpha sweep completed."
fi

# ── (b) Pool Size M sweep ────────────────────────────────────────────────────
if [ "$RUN_M_SWEEP" == "true" ]; then
    echo ""
    echo "════════ (b) Pool size M sweep ════════"
    for M in "${M_VALUES[@]}"; do
        _run_ablation "M" "$M" "--top_m=$M --max_epoch=$MAX_EPOCH"
    done
    echo "✅ Pool size M sweep completed."
fi

# ── (c) T_max sweep ──────────────────────────────────────────────────────────
if [ "$RUN_TMAX_SWEEP" == "true" ]; then
    echo ""
    echo "════════ (c) Max feedback rounds T_max sweep ════════"

    # Tmax sweep needs User Agent for actual feedback loop
    USE_USER_AGENT="true"
    echo "[Config] USE_USER_AGENT overridden → true (Tmax sweep needs feedback loop)"

    for TMAX in "${TMAX_VALUES[@]}"; do
        _run_ablation "Tmax" "$TMAX" "--max_epoch=$TMAX"
    done

    USE_USER_AGENT="false"
    echo "✅ T_max sweep completed."
fi

echo ""
echo "████████████████████████████████████████████████████████████████"
echo "  ABLATION STUDY COMPLETED."
echo "  Results: ./output/ablation/${ABL_DS}_${ABL_SCENARIO}/"
echo "  ──────────────────────────────────────────────────────"
echo "  Plot:"
echo "    python plot_ablation.py"
echo "████████████████████████████████████████████████████████████████"
