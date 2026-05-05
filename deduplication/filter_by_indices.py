import argparse
import json
from datasets import load_dataset, DatasetDict
import shutil
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Filter original dataset based on deduplicated dataset")
    parser.add_argument("-t", "--threads", type=int, default=1, help="Process within multiple threads")
    parser.add_argument("--output_dir", default="out/", type=str, help="Output directory for filtered dataset")
    parser.add_argument("--delete_without_asking", default=False, action="store_true", help="Disable prompt before deletion of output folder '[output_dir]/[dataset]'. It will DELETED WITHOUT ASKING.")
    parser.add_argument("--original_index_col", default="original_index", type=str, help="Column name with original indices in deduplicated dataset")
    parser.add_argument("--name_suffix", default="_deduplicated", type=str, help="Suffix that will be appended to original dataset name")
    parser.add_argument("deduplicated_dataset", help="Deduplicated dataset name/path (huggingface dataset)")
    parser.add_argument("original_dataset", help="Original dataset name/path (huggingface dataset)")

def main(args):
    dedup_dataset = load_dataset(args.deduplicated_dataset)
    orig_dataset = load_dataset(args.original_dataset)
    
    indices = {
        split: dedup_dataset[split][args.original_index_col] for split in dedup_dataset.keys()
    }

    filtered_dataset = DatasetDict({
        split: ds.select(indices[split])
        for split, ds in orig_dataset.items()
    })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{Path(args.original_dataset).stem}{args.name_suffix}"
    if save_path.exists():
        delete = True
        if not args.delete_without_asking:
            answer = ""
            while True:
                answer = input(f"Output folder '{save_path}' already exists. Do you wish to delete it? Not deleting it might result in inconsistent dataset. [n/y]: ")
                if answer not in ["n","y"]:
                    print("Unexpected answer. Type 'n' for NO or 'y' for YES")
                else:
                    break
            delete = answer == "y"
        if not delete:
            print("WARNING: Existing files in output directory will not be deleted. This may result in inconsistent or even corrupted dataset.")
        else:
            print(f"Deleting '{save_path}'")
            shutil.rmtree(save_path)
    filtered_dataset.save_to_disk(save_path, num_proc=args.threads)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)