import os
import glob
import shutil
import tarfile
import numpy as np
from PIL import Image, UnidentifiedImageError
from argparse import ArgumentParser
from multiprocessing import Pool, current_process
from streaming.base import MDSWriter
from streaming.base.util import merge_index
from torchvision import transforms
from tqdm import tqdm
from typing import List, Generator, Tuple


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        '--wds_dir',
        type=str,
        required=True,
        help='Path to local dir with wds download of cc12m dataset'
    )
    parser.add_argument(
        '--local_mds_dir',
        type=str,
        default='',
        help='Directory to store mds shards.'
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
    parser.add_argument('--num_proc', type=int, default=16)
    return parser.parse_args()


def current_process_index() -> int:
    # by default it starts from 1
    p = current_process()
    return p._identity[0] - 1


def read_tar(path: str, path_out: str) -> Generator[Tuple[Image.Image, str], None, None]:
    os.makedirs(path_out, exist_ok=False)
    with tarfile.open(path, 'r') as tar:
        tar.extractall(path_out)

    txts = sorted(glob.glob(os.path.join(path_out, '*txt')))
    print(f"Found {len(txts)} images in tar file")
    
    for t in txts:
        try:
            with open(t, 'r') as ct:
                cap = ct.read()
            # assuming all files are in jpg
            img = Image.open(t.replace('.txt', '.jpg'))
            yield img, cap
        except Exception as e:
            print(e)
            
    print("Done reading the tar file")
    shutil.rmtree(path_out)
    shutil.rmtree(os.path.dirname(path_out))


def write_tar(tars: List[str], args: ArgumentParser):
    columns = {
        'width': 'int32',
        'height': 'int32',
        'jpg': 'jpeg',
        'caption': 'str'
    }
    
    # make sure that write_tar is only called once per process
    save_dir = os.path.join(args.local_mds_dir, str(current_process_index()))
    os.makedirs(save_dir, exist_ok=True)
    
    # create a writer per process
    writer = MDSWriter(
        out=save_dir,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64
    )
    
    downsize = transforms.Resize(
        args.max_image_size,
        antialias=True,
        interpolation=transforms.InterpolationMode.BICUBIC
    )
    
    temp_dir = os.path.join(save_dir, f'temp/wds_{current_process_index()}')
    
    for tar in tars:
        rejected, total = 0, 0
        for img, cap in tqdm(read_tar(tar, temp_dir)):
            w, h = img.size
            try:
                if min(w, h) > args.max_image_size:
                    img = downsize(img)
                if min(w, h) < args.min_image_size:
                    rejected += 1
                    print(
                        f'Skipping image with resolution ({h}, {w}) - '
                        f'Since at least one side has resolution below {args.min_image_size}'
                    )
                    continue
            except (UnidentifiedImageError, OSError) as e:
                print(f"Error {e}")

            mds_sample = {
                'jpg': img,
                'caption': cap,
                'width': w,
                'height': h
            }
            writer.write(mds_sample)
            total += 1

        print(f"Rejected {rejected}, total {total}, Tar: {tar}")
    writer.finish()


def main():
    args = parse_arguments()
    print(os.path.join(args.wds_dir, '*tar'))
    tars = glob.glob(os.path.join(args.wds_dir, '*tar'))
    print(f"Total {len(tars)} tar files found in cc12m wds dataset path!")

    tars_split = np.array_split(tars, args.num_proc)
    
    with Pool(processes=args.num_proc) as pool:
        pool.starmap(write_tar, [(ts, args) for ts in tars_split])
    
    shards_metadata = [
        os.path.join(args.local_mds_dir, str(i), 'index.json')
        for i in range(args.num_proc)
    ]
    merge_index(shards_metadata, out=args.local_mds_dir, keep_local=True)


if __name__ == '__main__':
    main()