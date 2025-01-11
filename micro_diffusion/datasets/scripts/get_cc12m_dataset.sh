#!/bin/bash

# Get user input for data directory and dataset size
datadir=$1
dataset_size=$2 # small or all
num_gpus=$3

num_proc=16
batch_size=32 # use batch size of 8 for <16GB GPU memory

# A. Download a small subset (~1%) of the dataset, if specified
if [ "$dataset_size" == "small" ]; then
    echo "Downloading ~1% of the dataset..."
    python micro_diffusion/datasets/prepare/cc12m/download.py --datadir "${datadir}/wds/" --valid_ids $(seq 0 22) --num_proc $num_proc

# Or download the entire dataset, if specified
elif [ "$dataset_size" == "all" ]; then
    echo "Downloading the full dataset..."
    python micro_diffusion/datasets/prepare/cc12m/download.py --datadir "${datadir}/wds/" --num_proc $num_proc
else
    echo "Invalid dataset size option. Please use 'small' or 'all'."
    exit 1
fi

# B. Convert dataset to MDS format.
# CC12M has many images with resolution lower than 512x512. Thus we create a separate split of CC12M that has min(h, w) >= 512
python micro_diffusion/datasets/prepare/cc12m/convert.py --wds_dir "${datadir}/wds/" --local_mds_dir "${datadir}/mds/" \
    --max_image_size 512 --min_image_size 256 --num_proc $num_proc
python micro_diffusion/datasets/prepare/cc12m/convert.py --wds_dir "${datadir}/wds/" --local_mds_dir "${datadir}/mds_minres512/" \
    --max_image_size 512 --min_image_size 512 --num_proc $num_proc


# C. Precompute latents across multiple GPUs.
python -c "from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()"
accelerate launch --multi_gpu --num_processes $num_gpus micro_diffusion/datasets/prepare/cc12m/precompute.py --datadir "${datadir}/mds/" \
    --savedir "${datadir}/mds_latents_sdxl1_dfnclipH14/" --vae stabilityai/stable-diffusion-xl-base-1.0 \
    --text_encoder openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 --batch_size $batch_size

python -c "from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()"
accelerate launch --multi_gpu --num_processes $num_gpus micro_diffusion/datasets/prepare/cc12m/precompute.py --datadir "${datadir}/mds_minres512/" \
    --savedir "${datadir}/mds_minres512_latents_sdxl1_dfnclipH14/" --vae stabilityai/stable-diffusion-xl-base-1.0 \
    --text_encoder openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 --batch_size $batch_size