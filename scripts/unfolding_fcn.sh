#!/bin/bash

set -eu

# Datasets for experiments - format: dataset_name;encoding;encoding_type
# path is relative to the dir where the script will be executed (root dir of the repository)
datasets_info="Solesmes_staffLevel;s-gabc;music_aware
Einsiedeln_staffLevel;mei-gabc;new_gabc
Salzinnes_staffLevel;mei-gabc;new_gabc
GregoSynth_staffLevel;gabc;music_aware"

base_dir="$( pwd )"
output_base_dir="$base_dir/out/training"
datasets_base_dir="$base_dir/out/datasets/deduplicated_original"
model="unfolding_fcn"
mkdir -p "$output_base_dir"

while IFS= read -r LINE; do
    if [ -z "$LINE" ]; then
        continue
    fi

    name="$(echo "$LINE" | cut -d';' -f1)"
    encoding="$(echo "$LINE" | cut -d';' -f2)"
    encoding_type="$(echo "$LINE" | cut -d';' -f3)"

    experiment_dir="$output_base_dir/$name/$model"
    mkdir -p "$experiment_dir" && cd "$experiment_dir"

    python "$base_dir/AMNLT_original_models/AMNLT/scripts/dc_base_unfolding_trocr/train.py" --ds_name "$datasets_base_dir/$name" --model_name fcn --encoding_type "$encoding_type" --ctc greedy --batch_size 1
    python "$base_dir/AMNLT_original_models/AMNLT/scripts/compute_amnlt_metrics.py" --encoding "$encoding" "$datasets_base_dir/$name" predictions.txt > metrics.txt

done <<EOF
$datasets_info
EOF
