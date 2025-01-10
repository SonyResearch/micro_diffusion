## Instructions on Downloading and Preparing Datasets and Precomputed Latents

For each dataset preparation, we follow three steps:
- Download the dataset from its official source. We support multiprocessing to speed up the download of larger datasets.
- Convert datasets to the MDS format required for the streaming-datasets library.
- Precompute the image and text latent embeddings. We assume that latents are precomputed for all datasets during the training of micro-budget models.

We make use of the following five datasets for training: [Conceptual-Captions-12M](https://github.com/google-research-datasets/conceptual-12m), [Segment-Anything-1B](https://ai.meta.com/datasets/segment-anything/), [JourneyDB](https://journeydb.github.io/), [DiffusionDB](https://github.com/poloclub/diffusiondb), and [TextCaps](https://textvqa.org/textcaps/). We use [MS-COCO](https://cocodataset.org/#home) dataset for evaluation. 

For all datasets, we only retain images with a resolution of at least 256×256. Since our final model is trained on 512×512 resolution, we resize images larger than 512×512 to have a smaller edge of 512 pixels while preserving the aspect ratio. You can modify both of these settings in the data downloading scripts.

For each dataset, we also support downloading only ~1% of it for testing purposes. Below, we provide the individual steps for downloading the JourneyDB dataset. We provide scripts for end-to-end dataset handling in the `./scripts` directory.

## Dataset Pipeline Demo with the JourneyDB Dataset
We download the dataset from its official Hugging Face repository. We can either manually execute the three steps or use the script `scripts/get_jdb_dataset.sh` to automate the process.

A. 
Download 1% of the dataset.
```
python download.py --datadir ./datasets/jdb/ --max_image_size 512 --min_image_size 256 --valid_ids 0 1 --num_proc 2
```

Or you can download the full dataset.
```
python download.py --datadir ./datasets/jdb/ --max_image_size 512 --min_image_size 256 --num_proc 8
```

*Note*: Before downloading, you will need to acquire access to the dataset by accepting the terms of service on Hugging Face. You can also remove the *./datasets/jdb/compressed* directory to save disk space.

B. Convert dataset to mds format.
```
python convert.py --images_dir ./datasets/jdb/raw/train/imgs/ --captions_jsonl ./datasets/jdb/raw/train/train_anno_realease_repath.jsonl --local_mds_dir ./datasets/jdb/mds/train/

python convert.py --images_dir ./datasets/jdb/raw/valid/imgs/ --captions_jsonl ./datasets/jdb/raw/valid/valid_anno_repath.jsonl --local_mds_dir ./datasets/jdb/mds/valid/

python convert.py --images_dir ./datasets/jdb/raw/test/imgs/ --captions_jsonl ./datasets/jdb/raw/test/test_questions.jsonl --local_mds_dir ./datasets/jdb/mds/test/
```

C. Precompute latents.
Use multiple GPUs to parallelize latent computation (in this example, we are using 8 GPUs).
```
for split in train valid test; do
    accelerate launch --multi_gpu --num_processes 8 precompute.py --datadir ./datasets/jdb/mds/$split/ --savedir ./datasets/jdb/mds_latents_sdxl1_dfnclipH14/$split/ --vae stabilityai/stable-diffusion-xl-base-1.0 --text_encoder openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 --batch_size 32
done
```

You can also use a script that automates the three steps.
- Download small subset (~1%) of the dataset: 
`bash micro_diffusion/datasets/scripts/get_jdb_dataset.sh ./datasets/jdb small 8`
- Download the whole dataset: 
`bash micro_diffusion/datasets/scripts/get_jdb_dataset.sh ./datasets/jdb all 8`


## All Datasets
Similar to the JourneyDB dataset, all datasets can be processed using the scripts available in the `./scripts` directory. We recommend first downloading the `small` partition before downloading the full dataset for ease of debugging.

```
bash micro_diffusion/datasets/scripts/get_jdb_dataset.sh ./datasets/jdb small 8
bash micro_diffusion/datasets/scripts/get_diffdb_dataset.sh ./datasets/diffdb small 8
bash micro_diffusion/datasets/scripts/get_cc12m_dataset.sh ./datasets/c12m small 8
bash micro_diffusion/datasets/scripts/get_sa1b_dataset.sh ./datasets/sa1b small 8
bash micro_diffusion/datasets/scripts/get_textcaps_dataset.sh ./datasets/textcaps small 8
bash micro_diffusion/datasets/scripts/get_coco_dataset.sh ./datasets/coco2014 small 8
```

Comments:
- When using a large number of processes to download data from the Hugging Face Hub, some files may not download successfully. In that event, you can manually download them by providing the IDs of the unsuccessful archives.
- An added advantage of precomputing latents is that it homogenizes all datasets with different configurations into a unified set of image and latent vectors. All latent datasets are therefore loaded using a single latent_loader, allowing streams from all datasets to be easily mixed during training.
- In the current dataset downloading and processing setup, we store the data locally. However, MDS datasets support streaming from a remote source, which can be advantageous in cases of low local disk storage.
- If the streaming dataloader gets stuck at the start of an epoch, you may need to clean up the cache leftover from any previous runs using `python -c "from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()"`