from datasets import load_dataset
import stringzilla as sz
import stringzillas as szs
import numpy as np
from tqdm import tqdm
import argparse
import json
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Compute similarity matrix for a column of strings from dataset")
    parser.add_argument("-t", "--threads", type=int, default=None, help="Process file in multiple threads")
    parser.add_argument("-o", "--output", type=str, default="similarity_stats.bin", help="Similarity stats output")
    parser.add_argument("--column", type=str, default="transcription", help="Column name for which we compute the similarity matrix")
    parser.add_argument("--keep_whitespace", default=False, action="store_true", help="Keep whitespace characters in strings, otherwise they are deleted before distance computation")
    parser.add_argument("--keep_musictag", default=False, action="store_true", help="Keep music tags in strings, otherwise they are deleted before distance computation")
    parser.add_argument("dataset", help="Huggingface dataset name")

from multiprocessing import Pool

def worker_init(keep_whitespace, keep_musictag):
    global compiled_regex
    assert not keep_whitespace or not keep_musictag
    pattern = ""
    if not keep_whitespace and not keep_musictag:
        pattern = r"(\s|<m>)"
    elif not keep_whitespace:
        pattern = r"\s"
    elif not keep_musictag:
        pattern = r"<m>"
    compiled_regex = re.compile(pattern)

def preprocess(sample):
    global compiled_regex
    return compiled_regex.sub("", sample)

def main(args):
    dataset = load_dataset(args.dataset)
    strings = []
    splits_starts = {}
    splits = list(dataset.keys())
    pool = None
    if not args.keep_whitespace or not args.keep_musictag:
        pool = Pool(args.threads, initializer=worker_init, initargs=(args.keep_whitespace, args.keep_musictag))
        print("Preprocessing:")
    for split in splits:
        splits_starts[split] = len(strings)
        if pool is not None:
            strings.extend(tqdm(pool.imap(preprocess, dataset[split][args.column]), desc=split, total=len(dataset[split][args.column])))
        else:
            strings.extend(dataset[split][args.column])
    cpu_scope = szs.DeviceScope(cpu_cores=args.threads)
    engine = szs.LevenshteinDistancesUTF8()
    strings = sz.Strs(strings)
    distances_length = int((len(strings) * (len(strings)-1))/2)
    distances = np.memmap(args.output, dtype=np.uint16, mode="w+", shape=distances_length)
    clip_max = np.iinfo(distances.dtype).max
    index = 0
    print("Similarity computation:")
    with tqdm(total=len(strings)) as pbar:
        for i in range(len(strings)):
            row_distances = engine(sz.Strs([strings[i]] * (len(strings)-i-1)), strings[i+1:], device=cpu_scope)
            distances[index:index+len(row_distances)] = np.clip(row_distances, 0, clip_max)
            index += len(row_distances)
            pbar.update(1)
    distances.flush()
    with open(args.output+".json", "w", encoding="utf-8") as file:
        json.dump({
            "dataset": args.dataset,
            "splits": splits,
            "split_starts": splits_starts,
            "total_samples": len(strings),
            "distances_length": distances_length,
            "keep_whitespace": args.keep_whitespace,
            "keep_musictag": args.keep_musictag,
            "dtype": str(distances.dtype)}, file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
