#!/bin/bash

python AMNLT_original_models/AMNLT/scripts/smt_dan/train_dan.py --config_path configs/smt_dan/Solesmes/Solesmes_orig_dedup.json
python AMNLT_original_models/AMNLT/scripts/smt_dan/train_dan.py --config_path configs/smt_dan/Einsiedeln/Einsiedeln_orig_dedup.json
python AMNLT_original_models/AMNLT/scripts/smt_dan/train_dan.py --config_path configs/smt_dan/Salzinnes/Salzinnes_orig_dedup.json
python AMNLT_original_models/AMNLT/scripts/smt_dan/train_dan.py --config_path configs/smt_dan/GregoSynth/GregoSynth_orig_dedup.json
