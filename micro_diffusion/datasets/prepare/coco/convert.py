import os
import json
import subprocess
from argparse import ArgumentParser
from PIL import Image
from streaming.base import MDSWriter
from tqdm import tqdm
from typing import Dict, List, Any

'''
Example usage:
python convert.py --datadir ./coco2014/ --local_mds_dir ./coco2014/mds/
'''


def parse_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '--datadir',
        type=str,
        help='Directory to store mds shards.',
    )
    parser.add_argument(
        '--local_mds_dir',
        type=str,
        help='Directory to store mds shards.',
    )
    return parser.parse_args()


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    keys = batch[0].keys()
    data = {k: [] for k in keys}
    for b in batch:
        for k, v in b.items():
            data[k].append(v)
    return data


def main():
    args = parse_arguments()
    os.makedirs(args.datadir, exist_ok=True)

    subprocess.run(["wget", "-P", args.datadir, "http://images.cocodataset.org/zips/val2014.zip"])
    subprocess.run(["wget", "-P", args.datadir, "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"])
    subprocess.run(["unzip", "-d", args.datadir, f"{args.datadir}/val2014.zip"])
    subprocess.run(["unzip", "-d", args.datadir, f"{args.datadir}/annotations_trainval2014.zip"])

    captions_path = os.path.join(args.datadir, 'annotations/captions_val2014.json')
    data = json.load(open(captions_path))

    # Create {image_id: list[captions]} dictionary
    coco_captions = {}
    for sample in data['annotations']:
        image_id = sample['image_id']
        caption = sample['caption']
        if image_id in coco_captions:
            coco_captions[image_id]['captions'].append(caption.replace('\n', ''))
        else:
            coco_captions[image_id] = {'captions': [caption]}

        if 'image_file' not in coco_captions[image_id]:
            image_file = f'{args.datadir}/val2014/COCO_val2014_{image_id:012d}.jpg'
            coco_captions[image_id]['image_file'] = image_file

    columns = {'jpg': 'jpeg', 'caption': 'json'}
    writer = MDSWriter(
        out=args.local_mds_dir,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64,
    )

    for sample in tqdm(coco_captions):
        image = Image.open(coco_captions[sample]['image_file'])
        caption = coco_captions[sample]['captions'][0]  # Select first caption
        mds_sample = {'jpg': image, 'caption': caption}
        writer.write(mds_sample)

    writer.finish()


if __name__ == '__main__':
    main()
