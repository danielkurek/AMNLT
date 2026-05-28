#!/bin/bash

datasets_path="out/datasets/deduplicated_original"

python AMNLT_original_models/AMNLT/scripts/dc_base_unfolding_trocr/train.py --ds_name "$datasets_path/Solesmes_staffLevel" --model_name fcn --encoding_type music_aware --ctc greedy --batch_size 1
python AMNLT_original_models/AMNLT/scripts/dc_base_unfolding_trocr/train.py --ds_name "$datasets_path/Einsiedeln_staffLevel" --model_name fcn --encoding_type new_gabc --ctc greedy --batch_size 1
python AMNLT_original_models/AMNLT/scripts/dc_base_unfolding_trocr/train.py --ds_name "$datasets_path/Salzinnes_staffLevel" --model_name fcn --encoding_type new_gabc --ctc greedy --batch_size 1
python AMNLT_original_models/AMNLT/scripts/dc_base_unfolding_trocr/train.py --ds_name "$datasets_path/GregoSynth_staffLevel" --model_name fcn --encoding_type music_aware --ctc greedy --batch_size 1
