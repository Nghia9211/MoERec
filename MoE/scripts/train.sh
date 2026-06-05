#!/bin/bash
# train.sh — MoE Gating v2.3 | Train all datasets & losses
#
# Usage:
#   bash train.sh [loss] [epochs] [split]
#
# Args (optional, in order):
#   loss   : ce | bpr | both  (default: both)
#   epochs : number of epochs  (default: 50)
#   split  : train | val       (default: val)
#
# Examples:
#   bash train.sh              # train ce + bpr, 50 epochs
#   bash train.sh ce           # CE loss only
#   bash train.sh bpr 100      # BPR loss, 100 epochs
#   bash train.sh both 80 train

LOSS=${1:-ce}
EPOCHS=${2:-50}
SPLIT=${3:-val}

MOE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
cd "$MOE_DIR"

PYTHON="/home/research/nghialt/.venv/bin/python"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# DATASETS=("amazon" "yelp" "goodreads" "amazon_industrial" "amazon_musical")
DATASETS=("amazon_industrial" "amazon_musical")

CE_LR=1e-3
CE_BATCH=256
CE_BALANCE_EPS=0.1

BPR_LR=1e-3
BPR_BATCH=256
BPR_N_NEG=5
BPR_HARD_RATIO=0.5

if [[ "$LOSS" != "ce" && "$LOSS" != "bpr" && "$LOSS" != "both" ]]; then
    echo -e "${RED}[ERROR] loss must be: ce | bpr | both${NC}"
    exit 1
fi

declare -A RESULTS
TOTAL=0
FAILED=0
START_ALL=$SECONDS

print_header() {
    echo ""
    echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  $1${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
}

run_training() {
    local dataset=$1
    local loss=$2

    local data_dir="./data/${dataset}"
    local model_path="./saved_models/${dataset}_best_model.pt"
    local gcn_path="./saved_models/${dataset}_gcn_emb_remapped.pt"
    local faiss_path="./faiss_dbs/${dataset}_rich"
    local output_dir="./saved_models/moe_fix_gcn/${dataset}/${loss}"

    local missing=0
    for f in "$data_dir" "$model_path"; do
        if [[ ! -e "$f" ]]; then
            echo -e "  ${RED}[SKIP] Not found: $f${NC}"
            missing=1
        fi
    done
    [[ $missing -eq 1 ]] && return 1

    [[ ! -f "$gcn_path"   ]] && echo -e "  ${YELLOW}[WARN] GCN not found: $gcn_path${NC}"
    [[ ! -d "$faiss_path" ]] && echo -e "  ${YELLOW}[WARN] FAISS not found: $faiss_path${NC}"

    mkdir -p "$output_dir"
    local log_file="${output_dir}/train.log"

    echo -e "  ${GREEN}▶ Starting training...${NC}"
    echo -e "  Output dir : $output_dir"
    echo -e "  Log file   : $log_file"

    local start=$SECONDS

    if [[ "$loss" == "ce" ]]; then
        $PYTHON -m gating.train_gating \
            --data_dir    "$data_dir"    \
            --model_path  "$model_path"  \
            --gcn_path    "$gcn_path"    \
            --faiss_path  "$faiss_path"  \
            --output_dir  "$output_dir"  \
            --dataset     "$dataset"     \
            --epochs      "$EPOCHS"      \
            --split       "$SPLIT"       \
            --loss        ce             \
            --lr          $CE_LR         \
            --batch_size  $CE_BATCH      \
            --balance_eps $CE_BALANCE_EPS \
            2>&1 | tee "$log_file"

    else  # bpr
        $PYTHON -m gating.train_gating \
            --data_dir    "$data_dir"    \
            --model_path  "$model_path"  \
            --gcn_path    "$gcn_path"    \
            --faiss_path  "$faiss_path"  \
            --output_dir  "$output_dir"  \
            --dataset     "$dataset"     \
            --epochs      "$EPOCHS"      \
            --split       "$SPLIT"       \
            --loss        bpr            \
            --lr          $BPR_LR        \
            --batch_size  $BPR_BATCH     \
            --n_neg       $BPR_N_NEG     \
            --hard_ratio  $BPR_HARD_RATIO \
            2>&1 | tee "$log_file"
    fi

    local exit_code=${PIPESTATUS[0]}
    local elapsed=$(( SECONDS - start ))

    if [[ $exit_code -eq 0 ]]; then
        echo -e "  ${GREEN}✅ Done! (${elapsed}s)${NC}"
        return 0
    else
        echo -e "  ${RED}❌ FAILED (exit=$exit_code, ${elapsed}s) — see: $log_file${NC}"
        return 1
    fi
}

if [[ "$LOSS" == "both" ]]; then
    LOSSES=("ce" "bpr")
else
    LOSSES=("$LOSS")
fi

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║     MoE Gating v2.3 — Full Training Pipeline        ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
echo -e "  Datasets : ${DATASETS[*]}"
echo -e "  Losses   : ${LOSSES[*]}"
echo -e "  Epochs   : $EPOCHS"
echo -e "  Split    : $SPLIT"
echo ""

for dataset in "${DATASETS[@]}"; do
    for loss in "${LOSSES[@]}"; do
        TOTAL=$(( TOTAL + 1 ))
        key="${dataset}_${loss}"

        print_header "Dataset: ${dataset^^}  |  Loss: ${loss^^}  (${TOTAL}/${#DATASETS[@]}×${#LOSSES[@]})"

        if run_training "$dataset" "$loss"; then
            RESULTS[$key]="✅ OK"
        else
            RESULTS[$key]="❌ FAIL"
            FAILED=$(( FAILED + 1 ))
        fi
    done
done

ELAPSED_ALL=$(( SECONDS - START_ALL ))
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║                   SUMMARY                           ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
printf "  %-18s | %-8s | %s\n" "Dataset" "Loss" "Status"
echo "  ──────────────────────────────────────────"
for dataset in "${DATASETS[@]}"; do
    for loss in "${LOSSES[@]}"; do
        key="${dataset}_${loss}"
        printf "  %-18s | %-8s | %s\n" "$dataset" "$loss" "${RESULTS[$key]:-⚠️  SKIP}"
    done
done
echo "  ──────────────────────────────────────────"
echo -e "  Total: ${TOTAL} jobs | Failed: ${FAILED} | Time: ${ELAPSED_ALL}s"

if [[ $FAILED -eq 0 ]]; then
    echo -e "\n  ${GREEN}${BOLD}🎉 All completed successfully!${NC}"
    exit 0
else
    echo -e "\n  ${RED}${BOLD}⚠️  ${FAILED} job(s) failed. Check logs.${NC}"
    exit 1
fi