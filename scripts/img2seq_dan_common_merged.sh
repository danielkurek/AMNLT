#!/bin/bash

set -eu

config_path="configs/smt_dan/common/common.json"
# Datasets for experiments - format: dataset_name;encoding;config_path
# path is relative to the dir where the script will be executed (root dir of the repository)
separate_datasets_info="Solesmes_staffLevel;common-gabc;configs/smt_dan/Solesmes/Solesmes_orig_dedup.json
Einsiedeln_staffLevel;common-gabc;configs/smt_dan/Einsiedeln/Einsiedeln_orig_dedup.json
Salzinnes_staffLevel;common-gabc;configs/smt_dan/Salzinnes/Salzinnes_orig_dedup.json
GregoSynth_staffLevel;common-gabc;configs/smt_dan/GregoSynth/GregoSynth_orig_dedup.json"

base_dir="$( pwd )"
output_base_dir="$base_dir/out/training"
datasets_base_dir="$base_dir/out/datasets/deduplicated_original"
model="dan"
name="common"

experiment_dir="$output_base_dir/$name/$model"
mkdir -p "$experiment_dir" && cd "$experiment_dir"

config_exp_name="$(jq .name -r "$base_dir/$config_path")"

PYTHONPATH="$base_dir" python -m src.smt_dan.train_dan_common_grad_acc --config_path "$base_dir/$config_path"

while IFS= read -r LINE; do
    if [ -z "$LINE" ]; then
        continue
    fi

    name="$(echo "$LINE" | cut -d';' -f1)"
    encoding="$(echo "$LINE" | cut -d';' -f2)"
    config_path="$(echo "$LINE" | cut -d';' -f3)"

    python "$base_dir/AMNLT_original_models/AMNLT/scripts/smt_dan/test_dan.py" --config_path "$base_dir/$config_path" --checkpoint_path "weights/$config_exp_name/common_merged_DAN.ckpt"
    mv predictions.txt "predictions_$name.txt"
    python "$base_dir/AMNLT_original_models/AMNLT/scripts/compute_amnlt_metrics.py" --encoding "$encoding" "$datasets_base_dir/$name" "predictions_$name.txt" > "metrics_$name.txt"

done <<EOF
$separate_datasets_info
EOF