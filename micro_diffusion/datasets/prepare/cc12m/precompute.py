import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from streaming import MDSWriter
from streaming.base.util import merge_index
from tqdm import tqdm

from micro_diffusion.datasets.prepare.cc12m.base import (
    build_streaming_cc12m_precompute_dataloader,
)
from micro_diffusion.models.utils import UniversalTextEncoder, DATA_TYPES

'''
Example usage:
accelerate launch --multi_gpu --num_processes 8 precompute.py \
    --datadir ./cc12m/mds/ \
    --savedir ./cc12m/mds_latents_sdxl1_dfnclipH14/ \
    --vae stabilityai/stable-diffusion-xl-base-1.0 \
    --text_encoder openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 \
    --batch_size 32
'''


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        '--datadir',
        type=str,
        required=True,
        help='Local directory to store mds shards.',
    )
    parser.add_argument(
        '--savedir',
        type=str,
        default='',
        help='Remote path to upload MDS-formatted shards to.',
    )
    parser.add_argument(
        '--image_resolutions',
        type=int,
        nargs='+',
        default=[256, 512],
        help='List of image resolutions to use for processing.',
    )
    parser.add_argument(
        '--save_images',
        default=False,
        action='store_true',
        help='If True, also save images, else only latents',
    )
    parser.add_argument(
        '--model_dtype',
        type=str,
        choices=('float16', 'bfloat16', 'float32'),
        default='bfloat16',
        help='Data type for the encoding models',
    )
    parser.add_argument(
        '--save_dtype',
        type=str,
        choices=('float16', 'float32'),
        default='float16',
        help='Data type to save the latents',
    )
    parser.add_argument(
        '--vae',
        type=str,
        default='stabilityai/stable-diffusion-xl-base-1.0',
        help='Name of VAE model to use for vision encoding.',
    )
    parser.add_argument(
        '--text_encoder',
        type=str,
        default='openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378',
        help='Name of model to use for text encoding.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size per device to use for encoding.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=2024,
        help='Seed for random number generation.',
    )
    args = parser.parse_args()
    if isinstance(args.image_resolutions, int):
        args.image_resolutions = [args.image_resolutions]
    return args


def main(args):
    """Precompute image and text latents and store them in MDS format.

    By default, we only save the image latents for 256x256 and 512x512 image
    resolutions (using center crop).

    Note that the image latents will be scaled by the vae_scaling_factor.
    """
    cap_key = 'caption'  # Hardcoding the image caption key to 'caption' in MDS dataset

    accelerator = Accelerator()
    device = accelerator.device
    device_idx = int(accelerator.process_index)

    # Set random seeds
    torch.manual_seed(device_idx + args.seed)
    torch.cuda.manual_seed(device_idx + args.seed)
    np.random.seed(device_idx + args.seed)

    dataloader = build_streaming_cc12m_precompute_dataloader(
        datadir=[args.datadir],
        batch_size=args.batch_size,
        resize_sizes=args.image_resolutions,
        drop_last=False,
        shuffle=False,
        caption_key=cap_key,
        tokenizer_name=args.text_encoder,
        prefetch_factor=2,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )
    print(f'Device: {device_idx}, Dataloader sample count: {len(dataloader.dataset)}')

    print(
        f"MP variable -> world size: {os.environ['WORLD_SIZE']}, "
        f"RANK: {os.environ['RANK']}, {device}"
    )

    vae = AutoencoderKL.from_pretrained(
        args.vae,
        subfolder='vae',  # Change subfolder to appropriate one in hf_hub, if needed
        torch_dtype=DATA_TYPES[args.model_dtype],
    )
    print("Created VAE: ", args.vae)
    assert isinstance(vae, AutoencoderKL)

    text_encoder = UniversalTextEncoder(
        args.text_encoder,
        dtype=args.model_dtype,
        pretrained=True,
    )
    print("Created text encoder: ", args.text_encoder)

    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    columns = {
        cap_key: 'str',
        f'{cap_key}_latents': 'bytes',
        'latents_256': 'bytes',
        'latents_512': 'bytes',
    }
    if args.save_images:
        columns['jpg'] = 'jpeg'

    remote_upload = os.path.join(args.savedir, str(accelerator.process_index))
    writer = MDSWriter(
        out=remote_upload,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64,
    )

    for batch in tqdm(dataloader):
        image_256 = torch.stack(batch['image_0']).to(device)
        image_512 = torch.stack(batch['image_1']).to(device)
        captions = torch.stack(batch[cap_key]).to(device)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=DATA_TYPES[args.model_dtype]):
                latent_dist_256 = vae.encode(image_256)
                assert isinstance(latent_dist_256, AutoencoderKLOutput)
                latents_256 = (
                    latent_dist_256['latent_dist'].sample().data * vae.config.scaling_factor
                ).to(DATA_TYPES[args.save_dtype])

                latent_dist_512 = vae.encode(image_512)
                assert isinstance(latent_dist_512, AutoencoderKLOutput)
                latents_512 = (
                    latent_dist_512['latent_dist'].sample().data * vae.config.scaling_factor
                ).to(DATA_TYPES[args.save_dtype])

                attention_mask = None
                if f'{cap_key}_attention_mask' in batch:
                    attention_mask = torch.stack(
                        batch[f'{cap_key}_attention_mask']
                    ).to(device)

                conditioning = text_encoder.encode(
                    captions.view(-1, captions.shape[-1]),
                    attention_mask=attention_mask,
                )[0].to(DATA_TYPES[args.save_dtype])

        try:
            if isinstance(latents_256, torch.Tensor) and isinstance(
                latents_512, torch.Tensor
            ):
                latents_256 = latents_256.detach().cpu().numpy()
                latents_512 = latents_512.detach().cpu().numpy()
            else:
                continue

            if isinstance(conditioning, torch.Tensor):
                conditioning = conditioning.detach().cpu().numpy()
            else:
                continue

            # Write the batch to the MDS file
            for i in range(latents_256.shape[0]):
                mds_sample = {
                    cap_key: batch['sample'][i][cap_key],
                    f'{cap_key}_latents': np.reshape(conditioning[i], -1).tobytes(),
                    'latents_256': latents_256[i].tobytes(),
                    'latents_512': latents_512[i].tobytes(),
                }
                if args.save_images:
                    mds_sample['jpg'] = batch['sample'][i]['jpg']
                writer.write(mds_sample)
        except RuntimeError:
            print('Runtime error CUDA, skipping this batch')

    writer.finish()

    # Wait for all processes to finish
    accelerator.wait_for_everyone()
    print(f'Process {accelerator.process_index} finished')
    time.sleep(10)

    # Merge the mds shards created by each device (only do on main process)
    if accelerator.is_main_process:
        shards_metadata = [
            os.path.join(args.savedir, str(i), 'index.json')
            for i in range(accelerator.num_processes)
        ]
        merge_index(shards_metadata, out=args.savedir, keep_local=True)


if __name__ == '__main__':
    main(parse_args())