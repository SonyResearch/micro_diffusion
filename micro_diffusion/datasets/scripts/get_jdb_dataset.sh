#!/bin/bash

# Get user input for data directory and dataset size
datadir=$1
dataset_size=$2 # small or all
num_gpus=$3

num_proc=8
batch_size=32 # use batch size of 8 for <16GB GPU memory

# A. Download a small subset (~1%) of the dataset, if specified
if [ "$dataset_size" == "small" ]; then
    echo "Downloading ~1% of the dataset..."
    python micro_diffusion/datasets/prepare/jdb/download.py --datadir $datadir --max_image_size 512 --min_image_size 256 --valid_ids 0 1 --num_proc 2
# Or download the entire dataset, if specified
elif [ "$dataset_size" == "all" ]; then
    echo "Downloading the full dataset..."
    python micro_diffusion/datasets/prepare/jdb/download.py --datadir $datadir --max_image_size 512 --min_image_size 256 --num_proc $num_proc
else
    echo "Invalid dataset size option. Please use 'small' or 'all'."
    exit 1
fi

# B. Convert dataset to MDS format.
python micro_diffusion/datasets/prepare/jdb/convert.py --images_dir "${datadir}/raw/train/imgs/" --captions_jsonl "${datadir}/raw/train/train_anno_realease_repath.jsonl" --local_mds_dir "${datadir}/mds/train/"
python micro_diffusion/datasets/prepare/jdb/convert.py --images_dir "${datadir}/raw/valid/imgs/" --captions_jsonl "${datadir}/raw/valid/valid_anno_repath.jsonl" --local_mds_dir "${datadir}/mds/valid/"

# C. Precompute latents across multiple GPUs.
python -c "from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()"
for split in train valid; do
    accelerate launch --multi_gpu --num_processes $num_gpus micro_diffusion/datasets/prepare/jdb/precompute.py --datadir "${datadir}/mds/$split/" \
        --savedir "${datadir}/mds_latents_sdxl1_dfnclipH14/$split/" --vae stabilityai/stable-diffusion-xl-base-1.0 \
        --text_encoder openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 --batch_size $batch_size
done
