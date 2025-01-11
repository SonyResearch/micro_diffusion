import os
import shutil
import tarfile
import argparse
import subprocess
import numpy as np
import requests
import urllib.request
from tqdm import tqdm
from multiprocessing import Pool
from torchvision import transforms
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
from typing import List
from urllib.error import HTTPError, URLError


"""Example usage:
python download.py --datadir ./sa1b/ --max_image_size 512 --min_image_size 256 \
    --data_fraction 0.01 --skip_existing --num_proc 2
"""

already_extracted = []


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Download, uncompress, and resize images from Segment-anything-1B dataset.'
    )
    parser.add_argument(
        '--datadir',
        type=str,
        default='./sa1b/',
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
        help='Skip image if any side is smaller than min_image_size '
             '(almost unncessary as sa1b images have at least 1500x1500 resolution).'
    )
    parser.add_argument(
        '--data_fraction',
        type=float,
        default=1.0,
        help='Fraction of total dataset to download.'
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip extraction if the file has already been extracted'
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


def download_and_extract(
    file_name: str,
    url: str,
    args: argparse.Namespace
) -> None:
    tar_dir, images_dir = args.datadir_compressed, args.datadir_raw
    downsize = transforms.Resize(
        args.max_image_size,
        antialias=True,
        interpolation=transforms.InterpolationMode.BICUBIC
    )

    # Check if the file already exists
    if not os.path.exists(f'{tar_dir}/{file_name}') and file_name not in already_extracted:
        print(f'Downloading {file_name} from {url}...')
        response = requests.get(url, stream=True)
        with open(f'{tar_dir}/{file_name}', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f'{file_name} already exists in {tar_dir}. Skipping downloading it.')

    # Extract the file if it's a .tar file
    if file_name.endswith('.tar') and file_name not in already_extracted:
        images_subdir = os.path.join(images_dir, os.path.splitext(file_name)[0])
        os.makedirs(images_subdir, exist_ok=True)

        # Check if the file has already been extracted
        if len(os.listdir(images_subdir)) > 0 and args.skip_existing:
            print(f'{file_name} has already been extracted. Skipping extraction.')
        else:
            print(f'Extracting {file_name}...')
            with tarfile.open(f'{tar_dir}/{file_name}') as tar:
                for member in tqdm(tar.getmembers()):
                    try:
                        if member.name.endswith(".jpg"):
                            tar.extract(member, path=images_dir)
                            # Downsample images
                            p = os.path.join(images_dir, member.name.strip('./'))
                            new_p = os.path.join(images_subdir, member.name.strip('./'))
                            img = Image.open(p)
                            w, h = img.size
                            if min(w, h) > args.max_image_size:
                                img = downsize(img)
                            if min(w, h) < args.min_image_size:
                                print(
                                    f'Skipping image with resolution ({h}, {w}) - '
                                    f'Since at least one side has resolution below {args.min_image_size}'
                                )
                                continue
                            img.save(new_p)
                            os.remove(p)
                    except Exception as e:
                        print('Exception occured: ', e)
                
                print(f'{file_name} extracted!')

            already_extracted.append(file_name)
        
        # Delete tar file after extraction
        os.remove(f'{tar_dir}/{file_name}')
    else:
        print(f'{file_name} is not a tar file. Skipping extraction.')
        
  
def main() -> None:
    args = parse_arguments()
    
    os.makedirs(args.datadir, exist_ok=True)
    os.makedirs(args.datadir_compressed, exist_ok=True)
    os.makedirs(args.datadir_raw, exist_ok=True)
    
    print('Downloading Llava synthetic captions for sa1b dataset')
    cap_dir = os.path.join(args.datadir, 'captions')
    os.makedirs(cap_dir, exist_ok=True)
    subprocess.run([
        "wget",
        'https://huggingface.co/datasets/PixArt-alpha/SAM-LLaVA-Captions10M/resolve/main/SA1B_caption.tar.gz',
        "-O",
        os.path.join(args.datadir, 'SA1B_caption.tar.gz')
    ], check=True)
    subprocess.run([
        "tar",
        "-xzvf",
        os.path.join(args.datadir, 'SA1B_caption.tar.gz'),
        "-C",
        cap_dir
    ], check=True)

    try:
        url = ('https://scontent-sjc3-1.xx.fbcdn.net/m1/v/t6/'
               'An8MNcSV8eixKBYJ2kyw6sfPh-J9U4tH2BV7uPzibNa0pu4uHi6fyXdlbADVO4nfvsWpTwR8B0usCARHTz33cBQNrC0kWZsD1MbBWjw'
               '.txt?ccb=10-5&oh=00_AYBg06E6PToKwxq0WE5JXHxxyof9-eehXojLqzoMbneeOw&oe=67A68F58')
        with urllib.request.urlopen(url) as f:
            links = [link.decode('utf-8') for link in f.readlines()[1:]]
    except (HTTPError, URLError) as e:
        print(
            f"Url no valid. Exception: {e}. Please manually update the above urls to the file "
            "containing the urls of each *.tar split. Its link dynamically updates at SA1B dataset "
            "website, thus we can't provide an automated download option permanently. Dataset webpage "
            "for text file: https://ai.meta.com/datasets/segment-anything-downloads/"
        )
        return
   
    print(f'Downloading only {args.data_fraction * 100}% of SA1B dataset')    
    links = links[:int(len(links) * args.data_fraction)]
    
    # Download and extract the files in parallel
    with Pool(processes=args.num_proc) as pool:
        pool.starmap(
            download_and_extract,
            [(*line.strip().split('\t'), args) for line in links]
        )


if __name__ == "__main__":
    main()