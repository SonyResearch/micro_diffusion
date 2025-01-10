import os
import shutil
import argparse
import subprocess
import numpy as np

from glob import iglob
from multiprocessing import Pool
from torchvision import transforms
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError


"""Download Conceptual-Captions-12M dataset.

Example usage:
    python download.py --datadir ./cc12m/wds --valid_ids 0 1 --num_proc 2
"""


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Download Conceptual-Captions-12M dataset.'
    )
    parser.add_argument(
        '--datadir',
        type=str,
        default='./cc12m/wds',
        help='Directory to store wds data.'
    )
    parser.add_argument(
        '--valid_ids',
        type=int,
        nargs='+',
        default=list(np.arange(2176)),
        help='List of valid image IDs (default is 0 to 2176).'
    )
    parser.add_argument(
        '--num_proc',
        type=int,
        default=8,
        help='Number of parallel processes for downloading images.'
    )
    return parser.parse_args()


def download_shard(datadir: str, idx: int) -> None:
    """Downloads a single shard from HuggingFace Hub."""
    hf_hub_download(
        repo_id="pixparse/cc12m-wds",
        repo_type="dataset", 
        filename=f'cc12m-train-{idx:>04}.tar',
        local_dir=datadir,
        local_dir_use_symlinks=False
    )


def main():
    args = parse_arguments()
    
    os.makedirs(args.datadir, exist_ok=True)
    
    hf_hub_download(
        repo_id="pixparse/cc12m-wds",
        repo_type="dataset",
        filename="_info.json",
        local_dir=args.datadir,
        local_dir_use_symlinks=False
    )
    
    # Use multiprocessing to download the wds dataset
    with Pool(processes=args.num_proc) as pool:
        pool.starmap(
            download_shard,
            [(args.datadir, idx) for idx in args.valid_ids]
        )


if __name__ == "__main__":
    main()