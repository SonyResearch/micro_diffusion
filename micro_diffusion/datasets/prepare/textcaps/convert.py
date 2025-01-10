import os
import math
from argparse import ArgumentParser
from typing import Dict, List, Any

import numpy as np
import torch
from datasets import load_dataset
from streaming.base import MDSWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

"""
Example usage:
python convert.py --local_mds_dir ./textcaps/mds/
"""


def parse_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--local_mds_dir",
        type=str,
        help="Directory to store mds shards.",
    )
    args = parser.parse_args()
    return args


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    keys = batch[0].keys()
    data = {k: [] for k in keys}
    for b in batch:
        for k, v in b.items():
            data[k].append(v)
    return data


def main():
    args = parse_arguments()

    ds = load_dataset(
        "HuggingFaceM4/TextCaps",
        split="train+validation",
    )
    loader = DataLoader(
        ds,
        batch_size=512,
        collate_fn=collate_fn,
    )

    keys = ["height", "width", "jpg", "image_id", "org_captions"]
    samples = {k: [] for k in keys}

    for i, batch in tqdm(enumerate(loader)):
        samples["height"].extend(batch["image_height"])
        samples["width"].extend(batch["image_width"])
        samples["jpg"].extend(batch["image"])
        samples["image_id"].extend(batch["image_id"])
        samples["org_captions"].extend(batch["reference_strs"])

    print(f"Total {len(samples['jpg'])} samples in textcaps dataset")

    columns = {
        "height": "int32",
        "width": "int32",
        "jpg": "jpeg",
        "image_id": "str",
        "caption": "str",
    }

    writer = MDSWriter(
        out=args.local_mds_dir,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64,
    )

    for i in range(len(samples["jpg"])):
        try:
            mds_sample = {
                "height": samples["height"][i],
                "width": samples["width"][i],
                "jpg": samples["jpg"][i],
                "image_id": samples["image_id"][i],
                "caption": samples["org_captions"][i][0],
            }
            writer.write(mds_sample)
        except Exception as e:
            print(
                f"Something went wrong in reading caption, skipping writing this sample. "
                f"Error: {e}"
            )

    writer.finish()


if __name__ == "__main__":
    main()
