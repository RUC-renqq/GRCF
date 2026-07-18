#!/bin/bash

RELEVANCE_PATH=""
COMPETITION_PATH=""
OUTPUT_DIR="./res"
SEED=2022

for ALPHA in 0.1 0.3 0.5 0.7 0.9
do
    for TOP_K in 5 10 20
    do
        echo "Running MMF: alpha=${ALPHA}, top_k=${TOP_K}"
        python mmf.py \
            --relevance_path ${RELEVANCE_PATH} \
            --competition_path ${COMPETITION_PATH} \
            --top_k ${TOP_K} \
            --alpha ${ALPHA} \
            --seed ${SEED} \
            --output_dir ${OUTPUT_DIR}
    done
done
