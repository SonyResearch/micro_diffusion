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


"""Example usage:
python download.py --datadir ./diffusionDB/ --max_image_size 512 --min_image_size 256 \
    --valid_ids 0 1 --num_proc 2
"""

def parse_arguments():
    """Parse command line arguments for downloading and processing DiffusionDB dataset."""
    parser = argparse.ArgumentParser(
        description='Download, uncompress, and resize images from DiffusionDB dataset.'
    )
    parser.add_argument(
        '--datadir',
        type=str,
        default='./diffusionDB/',
        help='Directory to store data. Will create subdirs for compressed and raw data inside it.'
    )
    parser.add_argument(
        '--max_image_size',
        type=int,
        default=512,
        help='If min(h, w) > max_image_size, then downsize the smaller edge to max_image size.'
    )
    parser.add_argument(
        '--min_image_size',
        type=int,
        default=256,
        help='Skip image if any side is smaller than min_image_size.'
    )
    parser.add_argument(
        '--valid_ids',
        type=int,
        nargs='+',
        default=list(np.arange(1, 14001)),
        help='List of valid image IDs (default is 1 to 14001).'
    )
    parser.add_argument(
        '--num_proc',
        type=int,
        default=8,
        help='Number of parallel processes for downloading and processing images.'
    )
    
    args = parser.parse_args()
    args.datadir_compressed = os.path.join(args.datadir, 'compressed')
    args.datadir_raw = os.path.join(args.datadir, 'raw')
    return args


def download_and_process_metadata(args: argparse.Namespace):
    """Download and copy metadata file to both compressed and raw directories."""
    hf_hub_download(
        repo_id="poloclub/diffusiondb",
        repo_type="dataset",
        filename='metadata-large.parquet',
        local_dir=args.datadir_compressed,
        local_dir_use_symlinks=False
    )
    shutil.copy(
        os.path.join(args.datadir_compressed, 'metadata-large.parquet'),
        os.path.join(args.datadir_raw, 'metadata-large.parquet')
    )


def download_uncompress_resize(args: argparse.Namespace, idx: int):
    """Download, uncompress, and resize images for a given archive index."""
    assert idx in args.valid_ids
    data_split = (
        'diffusiondb-large-part-1' if idx < 10001 else 'diffusiondb-large-part-2'
    )

    print(f"Downloading idx: {idx}")
    hf_hub_download(
        repo_id="poloclub/diffusiondb",
        repo_type="dataset",
        subfolder=data_split,
        filename=f'part-{idx:>06}.zip',
        local_dir=args.datadir_compressed,
        local_dir_use_symlinks=False
    )
    print(f"Downloaded idx: {idx}")

    print(f"Extracting idx: {idx}")
    os.makedirs(
        os.path.join(args.datadir_compressed, f'images/part-{idx:>06}'),
        exist_ok=True
    )
    subprocess.call(
        f'unzip -qd {args.datadir_compressed}/images/part-{idx:>06}/ '
        f'{args.datadir_compressed}/{data_split}/part-{idx:>06}.zip',
        shell=True
    )
    print(f"Extracted idx: {idx}")

    print(f"Removing idx: {idx}")
    os.remove(os.path.join(
        args.datadir_compressed, data_split, f'part-{idx:>06}.zip'
    ))
    print(f"Removed idx: {idx}")

    downsize = transforms.Resize(
        args.max_image_size,
        antialias=True,
        interpolation=transforms.InterpolationMode.BICUBIC
    )
    
    print(f"Downsizing idx: {idx}")
    os.makedirs(
        os.path.join(args.datadir_raw, f'images/part-{idx:>06}'),
        exist_ok=True
    )
    for f in iglob(os.path.join(
        args.datadir_compressed, f'images/part-{idx:>06}/*.webp'
    )):
        try:
            img = Image.open(f)
            w, h = img.size
            if min(w, h) > args.max_image_size:
                img = downsize(img)
            if min(w, h) < args.min_image_size:
                print(
                    f'Skipping image with resolution ({h}, {w}) - '
                    f'Since at least one side has resolution below {args.min_image_size}'
                )
                continue
            img.save(os.path.join(
                args.datadir_raw,
                f'images/part-{idx:>06}',
                os.path.basename(f)
            ))
            os.remove(f)  # Remove the high resolution image
        except (UnidentifiedImageError, OSError) as e:
            print(f"Error {e}, File: {f}")
    print(f"Downsized idx: {idx}")
    
    shutil.move(
        os.path.join(
            args.datadir_compressed,
            f'images/part-{idx:>06}/part-{idx:>06}.json'
        ),
        os.path.join(
            args.datadir_raw,
            f'images/part-{idx:>06}/part-{idx:>06}.json'
        )
    )

    # Writing a dummy file for archives that are successfully processed.
    with open(os.path.join(
        args.datadir_compressed, data_split, f'{idx:>06}.txt'
    ), 'a') as f:
        f.write(f'{idx:>06} architecture processed successfully!')


def main():
    """Main function to orchestrate the download and processing of DiffusionDB."""
    args = parse_arguments()
    
    # Create necessary directories
    os.makedirs(args.datadir, exist_ok=True)
    os.makedirs(args.datadir_compressed, exist_ok=True)
    os.makedirs(args.datadir_raw, exist_ok=True)
    os.makedirs(os.path.join(args.datadir_compressed, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.datadir_raw, 'images'), exist_ok=True)
    os.makedirs(
        os.path.join(args.datadir_compressed, 'diffusiondb-large-part-1'),
        exist_ok=True
    )
    os.makedirs(
        os.path.join(args.datadir_compressed, 'diffusiondb-large-part-2'),
        exist_ok=True
    )
    
    download_and_process_metadata(args)
   
    print(f"Downloading and processing metadata for {len(args.valid_ids)} archieves.")
    # Use multiprocessing to download, uncompress, and resize images
    with Pool(processes=args.num_proc) as pool:
        pool.starmap(
            download_uncompress_resize,
            [(args, idx) for idx in args.valid_ids]
        )


if __name__ == "__main__":
    main()