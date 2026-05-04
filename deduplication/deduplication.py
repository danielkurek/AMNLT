import argparse
import json
from datasets import load_dataset, DatasetDict
import shutil
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Filter dataset according to precomputed duplicates")
    parser.add_argument("-t", "--threads", type=int, default=1, help="Process within multiple threads")
    parser.add_argument("--output_dir", default="out/", type=str, help="Output directory for filtered dataset")
    parser.add_argument("--delete_without_asking", default=False, action="store_true", help="Disable prompt before deletion of output folder '[output_dir]/[dataset]'. It will DELETED WITHOUT ASKING.")
    parser.add_argument("duplicates_file", help="File containing duplicates (.json)")
    parser.add_argument("original_dataset_path", help="Original dataset name/path (huggingface dataset)")

def prepare_indices(duplicates, split_priority = ["test", "validation", "train"]):
    """
    split_priority - ordering of splits, remove samples mainly from splits with higher indices
                     (e.g. if there is sample in test and train split, the sample in train split will be removed)
    """
    filter_indices = {split: set() for split in duplicates.keys()}
    split_priority = {split: i for i,split in enumerate(split_priority)}
    # the following filtering logic depends on symmetric relationship -> if a is duplicate of b than b is also duplicate of a
    for split in duplicates.keys():
        for index in duplicates[split].keys():
            index_int = int(index)
            priorities_other = [split_priority[other_split] for other_split in duplicates[split][index].keys()]
            # keep samples only from the highest priority splits
            if any(other_priority < split_priority[split] for other_priority in priorities_other):
                filter_indices[split].add(index_int)
                continue
            # keep samples with minimum index within each split
            if split in duplicates[split][index].keys() and index_int > min(int(x) for x in duplicates[split][index][split]):
                filter_indices[split].add(index_int)
    return filter_indices

def main(args):
    dataset = load_dataset(args.original_dataset_path)
    with open(args.duplicates_file, "r", encoding="utf-8-sig") as f:
        duplicates = json.load(f)

    filter_indices = prepare_indices(duplicates)

    with open("filtered_indices.json", "w", encoding="utf-8") as f:
        json.dump({split: sorted(list(indices)) for split,indices in filter_indices.items()}, f, ensure_ascii=False)
    
    keep_indices = {
        split: [i for i in range(len(dataset[split])) if i not in filter_indices[split]] for split in dataset.keys()
    }

    filtered_dataset = DatasetDict({
        split: ds.select(keep_indices[split])
        for split, ds in dataset.items()
    })

    print("Filtering results:")
    for split in dataset.keys():
        print(f"  {split}: {len(dataset[split])} -> {len(filtered_dataset[split])} ({len(dataset[split]) - len(filtered_dataset[split])} removed)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / args.original_dataset_path.replace("/", "-")
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