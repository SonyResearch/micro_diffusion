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

'''
Example usage:
python download.py --datadir ./jdb/ --max_image_size 512 --min_image_size 256 \
    --valid_ids 0 1 --num_proc 2
'''


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Download, uncompress, and resize images from JourneyDB dataset.'
    )
    parser.add_argument(
        '--datadir',
        type=str,
        default='./journeyDB/',
        help='Directory to store data. Will create subdirs for compressed and raw data inside it.',
    )
    parser.add_argument(
        '--max_image_size',
        type=int,
        default=512,
        help='If min(h, w) > max_image_size, then downsize the smaller edge to max_image size.',
    )
    parser.add_argument(
        '--min_image_size',
        type=int,
        default=256,
        help='Skip image if any side is smaller than min_image_size.',
    )
    parser.add_argument(
        '--valid_ids',
        type=int,
        nargs='+',
        default=list(np.arange(200)),
        help='List of valid image IDs (default is 0 to 199).',
    )
    parser.add_argument(
        '--num_proc',
        type=int,
        default=8,
        help='Number of parallel processes to for downloading and processing images.',
    )
    args = parser.parse_args()
    args.datadir_compressed = os.path.join(args.datadir, 'compressed')
    args.datadir_raw = os.path.join(args.datadir, 'raw')
    return args


def download_and_process_metadata(args: argparse.Namespace):
    # Only using a single process for downloading metadata
    metadata_files = [
        ('data/train', 'train_anno.jsonl.tgz'),
        ('data/train', 'train_anno_realease_repath.jsonl.tgz'),
        ('data/valid', 'valid_anno_repath.jsonl.tgz'),
        ('data/test', 'test_questions.jsonl.tgz'),
        ('data/test', 'imgs.tgz'),
    ]

    for subfolder, filename in metadata_files:
        hf_hub_download(
            repo_id="JourneyDB/JourneyDB",
            repo_type="dataset",
            subfolder=subfolder,
            filename=filename,
            local_dir=args.datadir_compressed,
            local_dir_use_symlinks=False,
        )

    metadata_tars = [
        os.path.join(dir, fname) for (dir, fname) in metadata_files
    ]

    for tar_file in metadata_tars:
        subprocess.call(
            f'tar -xvzf {args.datadir_compressed}/{tar_file} '
            f'-C {args.datadir_compressed}/{os.path.dirname(tar_file)}',
            shell=True,
        )

    shutil.copy(
        f'{args.datadir_compressed}/data/train/train_anno_realease_repath.jsonl',
        f'{args.datadir_raw}/train/train_anno_realease_repath.jsonl',
    )
    shutil.copy(
        f'{args.datadir_compressed}/data/valid/valid_anno_repath.jsonl',
        f'{args.datadir_raw}/valid/valid_anno_repath.jsonl',
    )
    shutil.copy(
        f'{args.datadir_compressed}/data/test/test_questions.jsonl',
        f'{args.datadir_raw}/test/test_questions.jsonl',
    )
    shutil.move(
        f'{args.datadir_compressed}/data/test/imgs',
        f'{args.datadir_raw}/test/',
    )


def download_uncompress_resize(
    args: argparse.Namespace,
    split: str,
    idx: int,
):
    """Download, uncompress, and resize images for a given archive index."""
    assert split in ('train', 'valid')
    assert idx in args.valid_ids

    print(f"Downloading idx: {idx}")
    hf_hub_download(
        repo_id="JourneyDB/JourneyDB",
        repo_type="dataset",
        subfolder=f'data/{split}/imgs',
        filename=f'{idx:>03}.tgz',
        local_dir=args.datadir_compressed,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded idx: {idx}")

    print(f"Extracting idx: {idx}")
    subprocess.call(
        f'tar -xzf {args.datadir_compressed}/data/{split}/imgs/{idx:>03}.tgz '
        f'-C {args.datadir_compressed}/data/{split}/imgs/',
        shell=True,
    )
    print(f"Extracted idx: {idx}")

    print(f"Removing idx: {idx}")
    os.remove(f'{args.datadir_compressed}/data/{split}/imgs/{idx:>03}.tgz')
    print(f"Removed idx: {idx}")

    # add bicubic downsize
    downsize = transforms.Resize(
        args.max_image_size,
        antialias=True,
        interpolation=transforms.InterpolationMode.BICUBIC,
    )

    print(f"Downsizing idx: {idx}")
    os.makedirs(
        f'{args.datadir_raw}/{split}/imgs/{idx:>03}/',
        exist_ok=True,
    )
    for f in iglob(f'{args.datadir_compressed}/data/{split}/imgs/{idx:>03}/*'):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
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
                img.save(
                    f'{args.datadir_raw}/{split}/imgs/{idx:>03}/{os.path.basename(f)}'
                )
                os.remove(f)
            except (UnidentifiedImageError, OSError) as e:
                print(f"Error {e}, File: {f}")
    print(f'Downsized idx: {idx}')


def main():
    args = parse_arguments()

    os.makedirs(args.datadir, exist_ok=True)
    os.makedirs(args.datadir_compressed, exist_ok=True)
    os.makedirs(args.datadir_raw, exist_ok=True)
    os.makedirs(os.path.join(args.datadir_raw, 'train', 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(args.datadir_raw, 'valid', 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(args.datadir_raw, 'test'), exist_ok=True)

    download_and_process_metadata(args)

    # Prepare arguments for multiprocessing
    pool_args = [('train', i) for i in args.valid_ids] + [('valid', i) for i in args.valid_ids]

    # Use multiprocessing to download, uncompress, and resize images
    with Pool(processes=args.num_proc) as pool:
        pool.starmap(download_uncompress_resize, [(args, split, idx) for split, idx in pool_args])

if __name__ == "__main__":
    main()