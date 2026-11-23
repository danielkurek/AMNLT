#!/bin/bash

set -eu

# Datasets for experiments - format: dataset_name;encoding;config_path
# Each model is trained on each dataset individually.
# path is relative to the dir where the script will be executed (root dir of the repository)
datasets_info="Solesmes_staffLevel;common-gabc;configs/Solesmes/Solesmes_common_dedup_fixed_height.json
Einsiedeln_staffLevel;common-gabc;configs/Einsiedeln/Einsiedeln_common_dedup_fixed_height.json
Salzinnes_staffLevel;common-gabc;configs/Salzinnes/Salzinnes_common_dedup_fixed_height.json
GregoSynth_staffLevel;common-gabc;configs/GregoSynth/GregoSynth_common_dedup_fixed_height.json"

base_dir="$( pwd )"
output_base_dir="$base_dir/out/training"
model="dan"
suffix="common_fixed_height"

while IFS= read -r LINE; do
    if [ -z "$LINE" ]; then
        continue
    fi

    name="$(echo "$LINE" | cut -d';' -f1)"
    encoding="$(echo "$LINE" | cut -d';' -f2)"
    config_path="$(echo "$LINE" | cut -d';' -f3)"

    experiment_dir="$output_base_dir/${name}_${suffix}/$model"
    mkdir -p "$experiment_dir" && cd "$experiment_dir"

    config_exp_name="$(jq .name -r "$base_dir/$config_path")"
    dataset_name="$(jq ".data[0].dataset_name" -r "$base_dir/$config_path")"

    PYTHONPATH="$base_dir" python -m src.smt_dan.train_dan_common_grad_acc --config_path "$base_dir/$config_path"
    PYTHONPATH="$base_dir" python -m src.smt_dan.test_dan_common --config_path "$base_dir/$config_path" --checkpoint_path "weights/$config_exp_name/${config_exp_name}_DAN.ckpt"
    python "$base_dir/AMNLT_original_models/AMNLT/scripts/compute_amnlt_metrics.py" --encoding "$encoding" "$dataset_name" predictions.txt > metrics.txt

done <<EOF
$datasets_info
EOF
