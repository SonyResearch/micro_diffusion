#!/bin/bash

# Get user input for data directory and dataset size
datadir=$1
num_gpus=$2

num_proc=16
batch_size=32 # use batch size of 8 for <16GB GPU memory

# COCO validation is fairly small so we download all of it. We also download and process the data is a single script.
python micro_diffusion/datasets/prepare/coco/convert.py --datadir "${datadir}/raw/" --local_mds_dir "${datadir}/mds/"

# Precompute latents across multiple GPUs.
python -c "from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()"
accelerate launch --multi_gpu --num_processes $num_gpus micro_diffusion/datasets/prepare/coco/precompute.py --datadir "${datadir}/mds/" \
    --savedir "${datadir}/mds_latents_sdxl1_dfnclipH14/" --vae stabilityai/stable-diffusion-xl-base-1.0 \
    --text_encoder openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 --batch_size $batch_size