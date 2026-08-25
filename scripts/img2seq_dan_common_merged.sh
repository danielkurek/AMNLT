#!/bin/bash

set -eu

config_path="configs/smt_dan/merged/common_merged.json"
# Datasets for experiments - format: dataset_name;encoding;dataset_index
# path is relative to the dir where the script will be executed (root dir of the repository)
separate_datasets_info="Solesmes_staffLevel;common-gabc;3
Einsiedeln_staffLevel;common-gabc;1
Salzinnes_staffLevel;common-gabc;2
GregoSynth_staffLevel;common-gabc;0"

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
    dataset_index="$(echo "$LINE" | cut -d';' -f3)"

    PYTHONPATH="$base_dir" python -m src.smt_dan.test_dan_common --config_path "$base_dir/$config_path" --checkpoint_path "weights/$config_exp_name/common_merged_DAN.ckpt" --dataset_index "$dataset_index"
    mv predictions.txt "predictions_$name.txt"
    python "$base_dir/AMNLT_original_models/AMNLT/scripts/compute_amnlt_metrics.py" --encoding "$encoding" "$datasets_base_dir/$name" "predictions_$name.txt" > "metrics_$name.txt"

done <<EOF
$separate_datasets_info
EOF