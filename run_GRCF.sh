#!/bin/bash

RELEVANCE_PATH=""
COMPETITION_PATH=""
OUTPUT_DIR=""

# Different alpha values
for ALPHA in 0.1 0.2 0.4 0.8
do
    # Different top-k values
    for TOP_K in 5 10 20
    do
        echo "Running experiment: alpha=${ALPHA}, top_k=${TOP_K}"
        python GRCF_main.py \
            --relevance_path ${RELEVANCE_PATH} \
            --competition_path ${COMPETITION_PATH} \
            --top_k ${TOP_K} \
            --alpha ${ALPHA} \
            --output_dir ${OUTPUT_DIR}
    done
done
