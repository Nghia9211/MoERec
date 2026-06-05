#!/bin/bash

echo "====================================================="
echo "1. Choose Dataset to Run Simulator"
echo "====================================================="
echo "[1] amazon"
echo "[2] goodreads"
echo "[3] yelp"
echo ""
read -p "Select Dataset (Ex: 1 2): " ds_choices

echo ""
echo "====================================================="
echo "2. Choose Scenario"
echo "====================================================="
echo "[1] classic"
echo "[2] user_cold_start"
echo "[3] item_cold_start"
echo ""
read -p "Select scenario (Ex: 1 2 3): " sc_choices

echo ""
read -p "Enter number of experiments to run : (Ex: 5): " num_runs

cd baseline || exit

for d in $ds_choices; do
    DS=""
    [ "$d" == "1" ] && DS="amazon"
    [ "$d" == "2" ] && DS="goodreads"
    [ "$d" == "3" ] && DS="yelp"

    if [ -n "$DS" ]; then
        for s in $sc_choices; do
            SC=""
            [ "$s" == "1" ] && SC="classic"
            [ "$s" == "2" ] && SC="user_cold_start"
            [ "$s" == "3" ] && SC="item_cold_start"

            if [ -n "$SC" ]; then
                echo ""
                echo "[STARTING] Dataset: $DS | Scenario: $SC | Provider: $PROV"
                
                for r in $(seq 1 $num_runs); do
                    echo ""
                    echo "[RUN $r/$num_runs] Dataset: $DS | Scenario: $SC"

                    # python CoTAgent_baseline.py --task_set $DS --scenario $SC 
                    # python CoTMemoryAgent_baseline.py --task_set $DS --scenario $SC 
                    # python MemoryAgent_baseline.py --task_set $DS --scenario $SC
                    # python DummyAgent_baseline.py --task_set $DS --scenario $SC 
                    python RecHackerAgent_baseline.py --task_set $DS --scenario $SC 
                    # python Baseline666_baseline.py --task_set $DS --scenario $SC

                    # Currently Run On Server for Test Result
                    # python3 ARAGgcnAgentRetrie.py --task_set $DS --scenario $SC 

                    TARGET_DIR="./results/$SC"
                    FOUND_FILE=""
                    
                    if [ -d "$TARGET_DIR" ]; then
                        for f in $(ls "$TARGET_DIR"/evaluation_results_*_"$DS".json 2>/dev/null); do
                            if [[ ! "$f" == *"_run"* ]]; then
                                FOUND_FILE=$(basename "$f")
                                break
                            fi
                        done
                    fi

                    if [ -n "$FOUND_FILE" ]; then
                        OLD_PATH="$TARGET_DIR/$FOUND_FILE"
                        FILENAME_NO_EXT="${FOUND_FILE%.*}"
                        EXT=".${FOUND_FILE##*.}"
                        
                        NEW_PATH="$TARGET_DIR/${FILENAME_NO_EXT}_run$r$EXT"
    
                        mv "$OLD_PATH" "$NEW_PATH"
                    else
                        echo "[WARNING] Not Found Result File to change name."
                    fi
                done
            fi
        done
    fi
done

cd ..

echo ""
echo "====================================================="
echo "All Expriements is completed."
echo "====================================================="
read -p "Press Enter to continue..."