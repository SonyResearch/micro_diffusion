import os
from argparse import ArgumentParser
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from streaming.base import MDSWriter
from streaming.base.util import merge_index


"""Example usage:
python convert.py \
    --images_dir ./diffusionDB/raw/ \
    --local_mds_dir ./diffusionDB/mds/ \
    --num_proc 16 \
    --safety_threshold 0.2
"""


def parse_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Path to local dir with all images",
    )
    parser.add_argument(
        "--local_mds_dir",
        type=str,
        required=True,
        help="Directory to store mds shards.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--safety_threshold",
        type=float,
        default=0.2,
        help="We discard all images with text/image nsfw score above this threshold",
    )
    args = parser.parse_args()
    return args


def write_df(args: ArgumentParser, df: pd.DataFrame, idx: int):
    columns = {
        "width": "int32",
        "height": "int32",
        "jpg": "jpeg",
        "caption": "str",
    }
    writer = MDSWriter(
        out=os.path.join(args.local_mds_dir, str(idx)),
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64,
    )

    total, skipped = 0, 0
    for id, im, pr, s1, s2 in tqdm(
        zip(df["part_id"], df["image_name"], df["prompt"], df["image_nsfw"], df["prompt_nsfw"])
    ):
        if s1 > args.safety_threshold or s2 > args.safety_threshold:
            skipped += 1
            continue

        try:
            img_path = os.path.join(args.images_dir, f"images/part-{id:>06}/{im}")
            if not os.path.exists(img_path):
                # Likely only a subset of dataset is downloaded and this image doesn't exist
                continue

            img = Image.open(img_path)
            w, h = img.size
            mds_sample = {
                "jpg": img,
                "caption": pr,
                "width": w,
                "height": h,
            }
            writer.write(mds_sample)
            total += 1
        except Exception as e:
            print(f"Something went wrong in reading caption, skipping writing this sample. Error: {e}")

    print(f"Total written: {total}, Skipped: {skipped}")
    writer.finish()


def main():
    args = parse_arguments()

    metadata = pd.read_parquet(
        os.path.join(args.images_dir, "metadata-large.parquet"),
        engine="fastparquet",
    )
    # Splitting metadata in num_processes
    metadata = np.array_split(metadata, args.num_proc)
    
    pool_args = [(args, df_sub, i) for i, df_sub in enumerate(metadata)]
    with Pool(processes=args.num_proc) as pool:
        pool.starmap(write_df, pool_args)

    shards_metadata = [
        os.path.join(args.local_mds_dir, str(i), "index.json")
        for i in range(args.num_proc)
    ]
    merge_index(shards_metadata, out=args.local_mds_dir, keep_local=True)


if __name__ == "__main__":
    main()
