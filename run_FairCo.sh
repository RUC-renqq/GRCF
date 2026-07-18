#!/bin/bash

RELEVANCE_PATH=""
COMPETITION_PATH=""
OUTPUT_DIR="./res"

for ALPHA in 0.01 0.05 0.1 0.5
do
    for TOP_K in 5 10 20
    do
        echo "Running FairCo: alpha=${ALPHA}, top_k=${TOP_K}"
        python FairCo_main.py \
            --relevance_path ${RELEVANCE_PATH} \
            --competition_path ${COMPETITION_PATH} \
            --top_k ${TOP_K} \
            --alpha ${ALPHA} \
            --output_dir ${OUTPUT_DIR}
    done
done
