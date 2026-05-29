#!/bin/bash

set -eu

# Datasets for experiments - format: dataset_name;encoding;config_path
# path is relative to the dir where the script will be executed (root dir of the repository)
datasets_info="Solesmes_staffLevel;s-gabc;configs/smt_dan/Solesmes/Solesmes_orig_dedup.json
Einsiedeln_staffLevel;mei-gabc;configs/smt_dan/Einsiedeln/Einsiedeln_orig_dedup.json
Salzinnes_staffLevel;mei-gabc;configs/smt_dan/Salzinnes/Salzinnes_orig_dedup.json
GregoSynth_staffLevel;gabc;configs/smt_dan/GregoSynth/GregoSynth_orig_dedup.json"

base_dir="$( pwd )"
output_base_dir="$base_dir/out/training"
datasets_base_dir="$base_dir/out/datasets/deduplicated_original"
model="smt"
mkdir -p "$output_base_dir"

while IFS= read -r LINE; do
    if [ -z "$LINE" ]; then
        continue
    fi

    name="$(echo "$LINE" | cut -d';' -f1)"
    encoding="$(echo "$LINE" | cut -d';' -f2)"
    config_path="$(echo "$LINE" | cut -d';' -f3)"

    experiment_dir="$output_base_dir/$name/$model"
    mkdir -p "$experiment_dir" && cd "$experiment_dir"

    python "$base_dir/AMNLT_original_models/AMNLT/scripts/smt_dan/train_smt.py" --config_path "$base_dir/$config_path"
    python "$base_dir/AMNLT_original_models/AMNLT/scripts/compute_amnlt_metrics.py" --encoding "$encoding" "$datasets_base_dir/$name" predictions.txt > metrics.txt

done <<EOF
$datasets_info
EOF