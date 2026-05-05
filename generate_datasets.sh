#!/bin/sh

set -eu

threads=1
output_dir="out"
while getopts t:o: opt
do
    case "$opt" in
      t)  threads="$OPTARG";;
      o)  output_dir="$OPTARG";;
      \?)		# unknown flag
      	  echo >&2 \
	  "usage: $0 [-t num_threads] [-o output_dir]"
	  exit 1;;
    esac
done
shift "$(($OPTIND - 1))"

case "$threads" in
    ''|*[!0-9]*) 
        echo "Error: '$threads' is not a valid number." >&2
        exit 1 ;;
esac

if [ "$threads" -le 0 ]; then
    echo "Error: Threads must be 1 or more." >&2
    exit 1
fi

mkdir -p "$output_dir"

# Defines datasets: "encoding;HuggingFaceDataset"
datasets_info="s-gabc;PRAIG/Solesmes_staffLevel
mei-gabc;PRAIG/Einsiedeln_staffLevel
mei-gabc;PRAIG/Salzinnes_staffLevel
gabc;PRAIG/GregoSynth_staffLevel
"

# Prepend resulting dataset name to each line
datasets_info_new=""
while IFS= read -r LINE; do
    hf_dataset="$(echo "$LINE" | cut -d';' -f2)"
    name="$(echo "$hf_dataset" | python3 -c "from pathlib import Path; print(Path(input()).stem);")"
    if [ -z "$datasets_info_new" ]; then
        datasets_info_new="$name;$LINE"
    else
        datasets_info_new="$(printf "%s\n%s;%s" "$datasets_info_new" "$name" "$LINE")"
    fi
done <<EOF
$datasets_info
EOF

datasets_info="$datasets_info_new"

mkdir -p "$output_dir/datasets"

# Generate common-encoded datasets
echo "Converting datasets to common encoding..."

conversion_output_dir="$output_dir/datasets/common_encoding"
conversion_log_file="$output_dir/conversion_to_common.log"
mkdir -p "$conversion_output_dir"
echo "" > "$conversion_log_file"

while IFS= read -r LINE; do
    name="$(echo "$LINE" | cut -d';' -f1)"
    encoding="$(echo "$LINE" | cut -d';' -f2)"
    hf_dataset="$(echo "$LINE" | cut -d';' -f3)"

    echo "Converting $name..."
    echo ">>>>$name<<<<" >> "$conversion_log_file"

    python3 -m gabcparser.utils.common_encoding \
        -t "$threads" \
        -o "$conversion_output_dir" \
        --remove_failed_rows --remove_mislabeled_custos --delete_without_asking \
        "$encoding" "$hf_dataset" >> "$conversion_log_file"
done <<EOF
$datasets_info
EOF

# Validate conversion

echo "Validation of conversion..."
validation_log_file="$output_dir/conversion_validation.log"
echo -n "" > "$validation_log_file"

while IFS= read -r LINE; do
    name="$(echo "$LINE" | cut -d';' -f1)"

    echo "Validating $name..."
    echo ">>>>$name<<<<" >> "$validation_log_file"

    python3 -m gabcparser.utils.grammar_validation \
        -t "$threads" \
        common-gabc "$conversion_output_dir/$name" >> "$validation_log_file"
done <<EOF
$datasets_info
EOF

if [ "$(grep -v -E "(^>>>>.+<<<<$|^$)" "$validation_log_file" | wc -l)" -gt 0 ]; then
    echo "Error: datasets were not converted correctly"
    exit 1;
fi

# Deduplication of common-encoded datasets

echo "Deduplication of common-encoded datasets"

deduplication_dir="$output_dir/deduplication_files"
deduplicated_dataset_dir="$output_dir/datasets/common_encoding_deduplicated"
similarity_threshold=3
mkdir -p "$deduplication_dir"
mkdir -p "$deduplicated_dataset_dir"

while IFS= read -r LINE; do
    name="$(echo "$LINE" | cut -d';' -f1)"

    echo "Calculating similarity matrix for $name..."

    python3 deduplication/similarity_matrix.py \
        -t "$threads" \
        -o "$deduplication_dir/sim_${name}_common.bin" \
        "$conversion_output_dir/$name"

    echo "Finding duplicates for $name..."
    python3 deduplication/duplicate_finder.py \
        -t "$threads" \
        --threshold "$similarity_threshold" \
        --check_images \
        --output "$deduplication_dir/duplicates_$name.json" \
        "$deduplication_dir/sim_${name}_common.bin"
    
    echo "Generating deduplicated dataset for $name..."
    python3 deduplication/deduplication.py \
        -t "$threads" \
        --delete_without_asking \
        --output_dir "$deduplicated_dataset_dir" \
        "$deduplication_dir/duplicates_$name.json" "$conversion_output_dir/$name"
done <<EOF
$datasets_info
EOF

# Generate deduplicated dataset with original data
# (with same samples as the deduplicated common-encoded datasets)

echo "Generating deduplicated original datasets"

original_deduplicated_dir="$output_dir/datasets/deduplicated_original"
mkdir -p "$original_deduplicated_dir"

while IFS= read -r LINE; do
    name="$(echo "$LINE" | cut -d';' -f1)"
    hf_dataset="$(echo "$LINE" | cut -d';' -f3)"

    echo "Validating $name..."
    echo "---$name---" >> "$validation_log_file"

    python3 deduplication/filter_by_indices.py \
        -t "$threads" \
        --output_dir "$original_deduplicated_dir" \
        "$deduplicated_dataset_dir/$name" "$hf_dataset"
done <<EOF
$datasets_info
EOF

echo "Success: All datasets were generated into $output_dir/datasets directory"
