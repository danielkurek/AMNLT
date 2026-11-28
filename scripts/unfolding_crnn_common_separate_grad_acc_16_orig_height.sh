#!/bin/bash

set -eu

# Datasets for experiments - format: dataset_name;encoding;config_path
# Each model is trained on each dataset individually.
# path is relative to the dir where the script will be executed (root dir of the repository)
datasets_info="Solesmes_staffLevel;common-gabc;configs/Solesmes/Solesmes_common_dedup.json
Einsiedeln_staffLevel;common-gabc;configs/Einsiedeln/Einsiedeln_common_dedup.json
Salzinnes_staffLevel;common-gabc;configs/Salzinnes/Salzinnes_common_dedup.json
GregoSynth_staffLevel;common-gabc;configs/GregoSynth/GregoSynth_common_dedup.json"

base_dir="$( pwd )"
output_base_dir="$base_dir/out/training"
model="unfolding_crnn_grad_acc_16"
model_name="crnnunfolding"
suffix="common_orig_height"

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

    PYTHONPATH="$base_dir" python -m src.dc_base_unfolding_trocr.train --config_path "$base_dir/$config_path" --model_name "$model_name" --gradient_accumulation 16
    PYTHONPATH="$base_dir" python -m src.dc_base_unfolding_trocr.test --config_path "$base_dir/$config_path" --checkpoint_path "weights/Baseline-CharacterLevel/${config_exp_name}_${model_name}_greedy.ckpt" --model_name "$model_name"
    python "$base_dir/AMNLT_original_models/AMNLT/scripts/compute_amnlt_metrics.py" --encoding "$encoding" "$dataset_name" predictions.txt > metrics.txt

done <<EOF
$datasets_info
EOF
