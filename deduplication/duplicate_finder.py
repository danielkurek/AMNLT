import numpy as np
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Find duplicates based on precomputed edit distance matrix")
    parser.add_argument("--stats", default=False, action="store_true", help="Print statistics of duplicates")
    parser.add_argument("--threshold", default=3, type=int, help="Edit distance threshold for samples to be considered duplicates")
    parser.add_argument("similarity_matrix", help="Similarity matrix file name (.bin) with accompanying .json file")

# Calculates starting index of each row
def calc_row_starts(n):
    row_starts = np.zeros(n-1, dtype=np.uint64)
    row_starts[1:] = np.cumsum(np.arange(n-1, 1, -1))
    return row_starts

# calculates indices of a column for the flattened upper triangular matrix
def calc_col_indices(i, row_starts):
    return row_starts[:i] + np.arange(i-1, -1, -1, dtype=np.uint64)

def get_split(index, splits):
    """
    get split name and original index from row index in similarity matrix
       `i`       index
       `splits`  sorted list of tuples (`split_name`,`starting_index`) by starting index (ascending)
    """
    for i in range(len(splits)):
        if i == len(splits) - 1 or index < splits[i+1][1]:
            return splits[i][0], index - splits[i][1]

def print_symmetric_matrix(matrix, labels):
    col_width = max(len(x) for x in labels)
    print(f"{" "*col_width}", *[f"{x: <{col_width}}" for x in labels], sep=" | ")
    print("-"*(col_width+1), *["-"*(col_width+2) for _ in range(len(labels))], sep="|")
    for i,row in enumerate(matrix):
        print(f"{labels[i]: <{col_width}}", *[f"{x: <{col_width}}" for x in row], sep=" | ")

def main(args):
    with open(args.similarity_matrix + ".json", "r") as f:
        config = json.load(f)
    if config["dtype"] != "uint16":
        raise ValueError("This program only supports uint16 type of similarity matrix")
    similarity_matrix = np.memmap(args.similarity_matrix, dtype=np.uint16)
    num_samples = config["total_samples"]
    matrix_expected_length = (num_samples * (num_samples-1)) / 2
    if len(similarity_matrix) != matrix_expected_length:
        raise ValueError(f"Loaded matrix has unexpected length - expected: {matrix_expected_length}; got: {len(similarity_matrix)}")
    row_starts = calc_row_starts(num_samples)

    splits = list(config["split_starts"].items())
    splits.sort(key=lambda x: x[1])
    
    if args.stats:
        print(f"Edit distances are clipped to range [0, {np.iinfo(similarity_matrix.dtype).max}]. This may influence the statistics.")
        print(f"Min. edit distance:   {np.min(similarity_matrix)} (number of edit distances: {np.count_nonzero(similarity_matrix <= np.min(similarity_matrix))})")
        print(f"Mean edit distance:   {np.mean(similarity_matrix)} (number of edit distances <= mean: {np.count_nonzero(similarity_matrix <= np.mean(similarity_matrix))})")
        print(f"Max. edit distance:   {np.max(similarity_matrix)}")

        duplicates_by_split = np.zeros((len(splits), len(splits)), dtype=np.uint64)
        split_index = {}
        for i,(split,_) in enumerate(splits):
            split_index[split] = i
        for i in range(num_samples):
            split, _ = get_split(i, splits)
            split_stats = {x:0 for x,_ in splits}
            if i < num_samples - 1:
                row_end = row_starts[i] + num_samples - i - 1
                mask = similarity_matrix[row_starts[i]:row_end] <= args.threshold
                indices = np.nonzero(mask)[0]
                for index in indices:
                    other_split, _ = get_split(i + 1 + index, splits)
                    split_stats[other_split] += 1
            if i >= 1:
                col_indices = calc_col_indices(i, row_starts)
                mask = similarity_matrix[col_indices] <= args.threshold
                indices = np.nonzero(mask)[0]
                for index in indices:
                    other_split, _ = get_split(index, splits)
                    split_stats[other_split] += 1
            for other_split, count in split_stats.items():
                if count > 0:
                    duplicates_by_split[split_index[split], split_index[other_split]] += 1
        print("Duplicates by split:")
        print_symmetric_matrix(duplicates_by_split, [x[0] for x in splits])
        print("(row/col - how many samples from 'row' split are in 'col' split)")
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
