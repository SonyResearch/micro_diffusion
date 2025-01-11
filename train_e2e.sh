#!/bin/bash

train() {
   # 1: yaml_path, 2: yaml_name, $3: update_args
   pkill -9 python; python -c 'import streaming; streaming.base.util.clean_stale_shared_memory()' # alternative hack: rm -rf /dev/shm/0000*
   rm -rf /tmp/streaming/*
   wait;
   sleep 3

   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HYDRA_FULL_ERROR=1 composer train.py --config-path $1 --config-name $2  $3
}

# Step-1: Pre-training at 256x256 image resolution with 75% patch masking
train ./configs res_256_pretrain.yaml "exp_name=MicroDiTXL_mask_75_res_256_pretrain model.train_mask_ratio=0.75"

# Step-2: Finetuning at 256x256 image resolution with no patch masking
train ./configs res_256_finetune.yaml "exp_name=MicroDiTXL_mask_0_res_256_finetune model.train_mask_ratio=0.0 trainer.load_path=./trained_models/MicroDiTXL_mask_75_res_256_pretrain/latest-rank0.pt"

# Step-3: Finetuning at 512x512 image resolution with 75% patch masking
train ./configs res_512_pretrain.yaml "exp_name=MicroDiTXL_mask_75_res_512_pretrain model.train_mask_ratio=0.75 trainer.load_path=./trained_models/MicroDiTXL_mask_0_res_256_finetune/latest-rank0.pt"

# Step-4: Finetuning at 512x512 image resolution with no patch masking
train ./configs res_512_finetune.yaml "exp_name=MicroDiTXL_mask_0_res_512_finetune model.train_mask_ratio=0.0 trainer.load_path=./trained_models/MicroDiTXL_mask_75_res_512_pretrain/latest-rank0.pt"