import os
from glob import glob
from argparse import ArgumentParser
from multiprocessing import Pool, current_process

import numpy as np
from PIL import Image
from tqdm import tqdm
from streaming.base import MDSWriter
from streaming.base.util import merge_index


"""Example usage:
python convert.py --images_dir ./sa1b/raw/ --captions_dir ./sa1b/captions/ \
    --local_mds_dir ./sa1b/mds/ --num_proc 16
"""


def parse_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '--images_dir',
        type=str,
        required=True,
        help='Path to local dir with all images'
    )
    parser.add_argument(
        '--captions_dir',
        type=str,
        required=True,
        help='Path to local dir with all captions (each caption is stored in a txt file)'
    )
    parser.add_argument(
        '--local_mds_dir',
        type=str,
        default='',
        help='Directory to store mds shards.'
    )
    parser.add_argument(
        '--num_proc',
        type=int,
        default=16
    )
    args = parser.parse_args()
    return args


def current_process_index() -> int:
    # by default it starts from 1
    p = current_process()
    return p._identity[0] - 1


def write_images(images_path: np.ndarray, args: ArgumentParser) -> None:
    print(f"Writing {len(images_path)} images in the {current_process_index()} proccess")
    assert isinstance(images_path, np.ndarray)
    
    columns = {
        'width': 'int32',
        'height': 'int32',
        'jpg': 'jpeg',
        'caption_syn_pixart_llava15': 'str'
    }
    
    save_dir = os.path.join(args.local_mds_dir, str(current_process_index()))
    os.makedirs(save_dir, exist_ok=True)
    
    writer = MDSWriter(
        out=save_dir,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64
    )
    
    for f in tqdm(images_path):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(f)
                w, h = img.size
                # data format is (sa_433759.jpg -> sa_433759.txt)
                cap_path = os.path.join(
                    args.captions_dir,
                    os.path.basename(f).split('.')[0] + '.txt'
                )
                cap = open(cap_path, 'r').read().strip()
                
                mds_sample = {
                    'jpg': img,
                    'caption_syn_pixart_llava15': cap,
                    'width': w,
                    'height': h
                }
                writer.write(mds_sample)
                
            except Exception as e:
                print(
                    "Something went wrong in reading image and caption, "
                    f"skipping writing this sample. Error: {e}"
                )
    
    writer.finish()


def main() -> None:
    args = parse_arguments()
    os.makedirs(args.local_mds_dir, exist_ok=True)
    
    images_path = glob(os.path.join(args.images_dir, "**", "*jpg"))
    print(f"Total {len(images_path)} available in the dataset")
    
    # use one worker per list of images
    images_path = np.array_split(images_path, args.num_proc)
    
    with Pool(processes=args.num_proc) as pool:
        pool.starmap(
            write_images,
            [(im, args) for im in images_path]
        )
    
    shards_metadata = [
        os.path.join(args.local_mds_dir, str(i), 'index.json')
        for i in range(args.num_proc)
    ]
    merge_index(shards_metadata, out=args.local_mds_dir, keep_local=True)


if __name__ == '__main__':
    main()
