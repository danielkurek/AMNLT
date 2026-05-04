import numpy as np
import argparse
import json
from datasets import load_dataset
from skimage.metrics import structural_similarity as ssim
from multiprocessing import Pool
from functools import partial
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Find duplicates based on precomputed edit distance matrix")
    parser.add_argument("-t", "--threads", type=int, default=1, help="Process within multiple threads")
    parser.add_argument("--stats_only", default=False, action="store_true", help="Print statistics of duplicates - no performance benefit (if --check_images is not used)")
    parser.add_argument("--threshold", default=3, type=int, help="Edit distance threshold for samples to be considered duplicates")
    parser.add_argument("--check_images", default=False, action="store_true", help="Validate duplicates by comparing images. Duplicate will be reported only if both the transcription and the image are same.")
    parser.add_argument("--image_threshold", default=0.97, type=float, help="Threshold for SSIM between image to detect duplicates")
    parser.add_argument("--output", default="duplicates.json", type=str, help="Output json file of duplicates")
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

def basic_matrix_stats(similarity_matrix, chunk_size = 128 * 1024 * 1024):
    min_value = np.iinfo(similarity_matrix.dtype).max
    max_value = 0
    sum_value = np.float64(0.0)

    for offset in tqdm.trange(0, len(similarity_matrix), chunk_size, desc="Calculating basic stats"):
        chunk = similarity_matrix[offset:offset+chunk_size]
        min_value = min(min_value, np.min(chunk))
        max_value = max(max_value, np.max(chunk))
        sum_value += np.sum(chunk, dtype=np.float64)
    
    mean_value = sum_value / len(similarity_matrix)
    return int(min_value), int(max_value), float(mean_value)

def duplicate_finder(similarity_matrix, splits, num_samples, row_starts, threshold, prune_empty_sets = True):
    min_val, max_val, mean_val = basic_matrix_stats(similarity_matrix)

    min_count = 0
    mean_count = 0

    duplicates_by_split = np.zeros((len(splits), len(splits)), dtype=np.uint64)
    duplicates = {split: {} for split, _ in splits}
    split_index = {}
    for i,(split,_) in enumerate(splits):
        split_index[split] = i
        duplicates[split] = {}
    
    # precompute original information about samples
    split_info = [get_split(i, splits) for i in range(num_samples)]

    for i in tqdm.trange(num_samples-1, desc="Finding duplicates"):
        split, original_index = split_info[i]
        
        row_start = row_starts[i]
        row_end = row_start + num_samples - i - 1
        row = similarity_matrix[row_start:row_end]

        min_count += np.count_nonzero(row <= min_val)
        mean_count += np.count_nonzero(row <= mean_val)

        indices = np.nonzero(row <= threshold)[0]
        for other_index in indices:
            other_split, other_orig_index = split_info[i + 1 + other_index]
            if original_index not in duplicates[split]:
                duplicates[split][original_index] = {x: set() for x,_ in splits}
            if other_orig_index not in duplicates[other_split]:
                duplicates[other_split][other_orig_index] = {x: set() for x,_ in splits}
            duplicates[split][original_index][other_split].add(int(other_orig_index))
            duplicates[other_split][other_orig_index][split].add(int(original_index))

    for split,_ in splits:
        for i in duplicates[split].keys():
            for other_split,_ in splits:
                if len(duplicates[split][i][other_split]) >= 1:
                    duplicates[split][i][other_split] = sorted(list(duplicates[split][i][other_split]))
                    duplicates_by_split[split_index[split], split_index[other_split]] += 1
                elif prune_empty_sets:
                    del duplicates[split][i][other_split]
    
    print(f"Edit distances are clipped to range [0, {np.iinfo(similarity_matrix.dtype).max}]. This may influence the statistics.")
    print(f"Min. edit distance:   {min_val} (number of edit distances: {min_count})")
    print(f"Mean edit distance:   {mean_val:.2f} (number of edit distances <= mean: {mean_count})")
    print(f"Max. edit distance:   {max_val}")
    
    print("Duplicates by split:")
    print_symmetric_matrix(duplicates_by_split, [x[0] for x in splits])
    print("(row/col - how many samples from 'row' split are in 'col' split)")

    return duplicates

def image_duplicate_worker_init(dataset_path):
    global ds
    ds = load_dataset(dataset_path)

def image_duplicate_process(item, split, threshold):
    global ds
    index, duplicates = item

    orig_img = ds[split]["image"][int(index)]
    size = (orig_img.width // 4, orig_img.height // 4)
    im1 = np.asarray(orig_img.convert('L').resize(size))

    for other_split in duplicates.keys():
        i = 0
        while i < len(duplicates[other_split]):
            other_index = duplicates[other_split][i]
            im2 = np.asarray(ds[other_split]["image"][int(other_index)].convert('L').resize(size))
            score = ssim(im1, im2, data_range=255.0)
            if score < threshold:
                duplicates[other_split].pop(i)
                continue # next item is at position i
            i += 1
    for other_split in list(duplicates.keys()):
        if len(duplicates[other_split]) == 0:
            del duplicates[other_split]
    if len(duplicates) == 0:
        return index, None
    return index, duplicates

def image_duplicate_filter(duplicates, dataset_path, threshold, threads):
    pool = Pool(threads, initializer=image_duplicate_worker_init, initargs=(dataset_path,))
    for split in duplicates.keys():
        process_fn = partial(image_duplicate_process, split=split, threshold=threshold)
        delete_indices = []
        for result in tqdm.tqdm(pool.imap_unordered(process_fn, duplicates[split].items()), total=len(duplicates[split])):
            index, value = result
            if value is None:
                delete_indices.append(index)
            else:
                duplicates[split][index] = value
        for index in delete_indices:
            del duplicates[split][index]
    pool.close()
    pool.join()
    return duplicates

def main(args):
    with open(args.similarity_matrix + ".json", "r") as f:
        config = json.load(f)
    if config["dtype"] != "uint16":
        raise ValueError("This program only supports uint16 type of similarity matrix")
    similarity_matrix = np.memmap(args.similarity_matrix, mode="r", dtype=np.uint16)
    num_samples = config["total_samples"]
    matrix_expected_length = (num_samples * (num_samples-1)) / 2
    if len(similarity_matrix) != matrix_expected_length:
        raise ValueError(f"Loaded matrix has unexpected length - expected: {matrix_expected_length}; got: {len(similarity_matrix)}")
    row_starts = calc_row_starts(num_samples)

    splits = list(config["split_starts"].items())
    splits.sort(key=lambda x: x[1])
    
    duplicates = duplicate_finder(similarity_matrix, splits, num_samples, row_starts, args.threshold)
    if args.stats_only:
        return
    if args.check_images:
        duplicates = image_duplicate_filter(duplicates, config["dataset"], args.image_threshold, args.threads)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(duplicates, f, ensure_ascii=False)
        
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
