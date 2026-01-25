#!/bin/bash

DATASETS=("cube" "grid" "mnist" "gas"  "fashion" "big5") # "big5"
# DATASETS=("cube") # "big5"

GPU_ID=0
BASE_DIR="share_random_init"

shopt -s nullglob

for DATASET in "${DATASETS[@]}"
do
    echo "================================================"
    echo "Starting evaluation on dataset: $DATASET"

    FILES=("${BASE_DIR}/${DATASET}/policy_diff_final_${DATASET}_"*_rollout.pt)

    IFS=$'\n' FILES=($(sort -V <<<"${FILES[*]}"))
    unset IFS

    if [ ${#FILES[@]} -eq 0 ]; then
        echo "No models found for $DATASET in $BASE_DIR/$DATASET"
        continue
    fi

    for MODEL_PATH in "${FILES[@]}"
    do
        echo "------------------------------------------------"
        echo "Loading: $MODEL_PATH"
    
        FILENAME=$(basename -- "$MODEL_PATH")
        TEMP="${FILENAME##*policy_diff_final_${DATASET}_}"
        IDX="${TEMP%_rollout.pt}"
        
        echo "Running index: $IDX"

        python timing_code_ablation.py \
            --dataset "$DATASET" \
            --gpu "$GPU_ID" \
            --model_path "$MODEL_PATH"

    done
done