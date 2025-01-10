from functools import partial
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.models import ComposerModel
from diffusers import AutoencoderKL
from easydict import EasyDict

from . import dit as model_zoo
from .utils import (
    DATA_TYPES,
    DistLoss,
    UniversalTextEncoder,
    UniversalTokenizer,
    text_encoder_embedding_format,
)


class LatentDiffusion(ComposerModel):
    """Latent diffusion model that generates images from text prompts.

    This model combines a DiT (Diffusion Transformer) model for denoising image latents,
    a VAE for encoding/decoding images to/from the latent space, and a text encoder
    for converting text prompts into embeddings. It implements the EDM (Elucidated
    Diffusion Model) sampling process.

    Args:
        dit (nn.Module): Diffusion Transformer model
        vae (AutoencoderKL): VAE model from diffusers for encoding/decoding images
        text_encoder (UniversalTextEncoder): Text encoder for converting prompts to embeddings
        tokenizer (UniversalTokenizer): Tokenizer for processing text prompts
        image_key (str, optional): Key for image data in batch dict. Defaults to 'image'.
        text_key (str, optional): Key for text data in batch dict. Defaults to 'captions'.
        image_latents_key (str, optional): Key for precomputed image latents in batch dict. Defaults to 'image_latents'.
        text_latents_key (str, optional): Key for precomputed text latents in batch dict. Defaults to 'caption_latents'.
        precomputed_latents (bool, optional): Whether to use precomputed latents (must be in the batch). Defaults to True.
        dtype (str, optional): Data type for model ops. Defaults to 'bfloat16'.
        latent_res (int, optional): Resolution of latent space assuming 8x downsampling by VAE. Defaults to 32.
        p_mean (float, optional): EDM log-normal noise mean. Defaults to -0.6.
        p_std (float, optional): EDM log-normal noise standard-deviation. Defaults to 1.2.
        train_mask_ratio (float, optional): Ratio of patches to mask during training. Defaults to 0.
    """

    def __init__(
        self,
        dit: nn.Module,
        vae: AutoencoderKL,
        text_encoder: UniversalTextEncoder,
        tokenizer: UniversalTokenizer,
        image_key: str = 'image',
        text_key: str = 'captions',
        image_latents_key: str = 'image_latents',
        text_latents_key: str = 'caption_latents',
        precomputed_latents: bool = True,
        dtype: str = 'bfloat16',
        latent_res: int = 32,
        p_mean: float = -0.6,
        p_std: float = 1.2,
        train_mask_ratio: float = 0.
    ):
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.image_key = image_key
        self.text_key = text_key
        self.image_latents_key = image_latents_key
        self.text_latents_key = text_latents_key
        self.precomputed_latents = precomputed_latents
        self.dtype = dtype
        self.latent_res = latent_res
        self.edm_config = EasyDict({
            'sigma_min': 0.002,
            'sigma_max': 80,
            'P_mean': p_mean,
            'P_std': p_std,
            'sigma_data': 0.9,
            'num_steps': 18,
            'rho': 7,
            'S_churn': 0,
            'S_min': 0,
            'S_max': float('inf'),
            'S_noise': 1
        })
        self.train_mask_ratio = train_mask_ratio
        self.eval_mask_ratio = 0.  # no masking during sampling/evaluation
        assert self.train_mask_ratio >= 0, 'Masking ratio must be non-negative!'

        self.randn_like = torch.randn_like
        self.latent_scale = self.vae.config.scaling_factor

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        # freeze vae and text_encoder during training
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        # avoid wrapping the models that we aren't training
        self.text_encoder._fsdp_wrap = False
        self.vae._fsdp_wrap = False
        self.dit._fsdp_wrap = True

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get image latents
        if self.precomputed_latents and self.image_latents_key in batch:
            # Assuming that latents have already been scaled, i.e., multiplied with the scaling factor
            latents = batch[self.image_latents_key]
        else:
            with torch.no_grad():
                images = batch[self.image_key]
                latents = self.vae.encode(
                    images.to(DATA_TYPES[self.dtype])
                )['latent_dist'].sample().data
                latents *= self.latent_scale

        # Get text embeddings
        if self.precomputed_latents and self.text_latents_key in batch:
            conditioning = batch[self.text_latents_key]
        else:
            captions = batch[self.text_key]
            captions = captions.view(-1, captions.shape[-1])
            if 'attention_mask' in batch:
                conditioning = self.text_encoder.encode(
                    captions,
                    attention_mask=batch['attention_mask'].view(-1, captions.shape[-1])
                )[0]
            else:
                conditioning = self.text_encoder.encode(captions)[0]

        # Zero out dropped captions. Needed for classifier-free guidance during inference.
        if 'drop_caption_mask' in batch.keys():
            conditioning *= batch['drop_caption_mask'].view(
                [-1] + [1] * (len(conditioning.shape) - 1)
            )

        loss = self.edm_loss(
            latents.float(),
            conditioning.float(),
            mask_ratio=self.train_mask_ratio if self.training else self.eval_mask_ratio
        )
        return (loss, latents, conditioning)

    def model_forward_wrapper(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        y: torch.Tensor,
        model_forward_fxn: callable,
        mask_ratio: float,
        **kwargs
    ) -> dict:
        """Wrapper for the model call in EDM (https://github.com/NVlabs/edm/blob/main/training/networks.py#L632)"""
        sigma = sigma.to(x.dtype).reshape(-1, 1, 1, 1)
        c_skip = (
            self.edm_config.sigma_data ** 2 /
            (sigma ** 2 + self.edm_config.sigma_data ** 2)
        )
        c_out = (
            sigma * self.edm_config.sigma_data /
            (sigma ** 2 + self.edm_config.sigma_data ** 2).sqrt()
        )
        c_in = 1 / (self.edm_config.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        out = model_forward_fxn(
            (c_in * x).to(x.dtype),
            c_noise.flatten(),
            y,
            mask_ratio=mask_ratio,
            **kwargs
        )
        F_x = out['sample']
        c_skip = c_skip.to(F_x.device)
        x = x.to(F_x.device)
        c_out = c_out.to(F_x.device)
        D_x = c_skip * x + c_out * F_x
        out['sample'] = D_x
        return out

    def edm_loss(self, x: torch.Tensor, y: torch.Tensor, mask_ratio: float = 0, **kwargs) -> torch.Tensor:
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.edm_config.P_std + self.edm_config.P_mean).exp()
        weight = (
            (sigma ** 2 + self.edm_config.sigma_data ** 2) /
            (sigma * self.edm_config.sigma_data) ** 2
        )
        n = self.randn_like(x) * sigma

        model_out = self.model_forward_wrapper(
            x + n,
            sigma,
            y,
            self.dit,
            mask_ratio=mask_ratio,
            **kwargs
        )
        D_xn = model_out['sample']
        loss = weight * ((D_xn - x) ** 2)  # (N, C, H, W)

        if mask_ratio > 0:
            # Masking is not feasible during image generation as it only returns denoised version
            # for non-masked patches. Image generation requires all patches to be denoised.
            assert (
                self.dit.training and 'mask' in model_out
            ), 'Masking is only recommended during training'
            loss = F.avg_pool2d(loss.mean(dim=1), self.dit.patch_size).flatten(1)
            unmask = 1 - model_out['mask']
            loss = (loss * unmask).sum(dim=1) / unmask.sum(dim=1)  # (N,)
        return loss.mean()

    # Composer specific formatting of model loss and eval functions.
    def loss(self, outputs: tuple, batch: dict) -> torch.Tensor:
        # forward pass already computed the loss function
        return outputs[0]

    def eval_forward(self, batch: dict, outputs: Optional[tuple] = None) -> tuple:
        # Skip if output already calculated (e.g., during training forward pass)
        if outputs is not None:
            return outputs
        loss, _, _ = self.forward(batch)
        return loss, None, None

    def get_metrics(self, is_train: bool = False) -> dict:
        # get_metrics expected to return a dict in composer
        return {'loss': DistLoss()}

    def update_metric(self, batch: dict, outputs: tuple, metric: DistLoss):
        metric.update(outputs[0])

    @torch.no_grad()
    def edm_sampler_loop(
        self, x: torch.Tensor, 
        y: torch.Tensor, 
        steps: Optional[int] = None, 
        cfg: float = 1.0, 
        **kwargs
    ) -> torch.Tensor:
        mask_ratio = 0  # no masking during image generation
        model_forward_fxn = (
            partial(self.dit.forward, cfg=cfg) if cfg > 1.0
            else self.dit.forward
        )

        # Time step discretization.
        num_steps = self.edm_config.num_steps if steps is None else steps
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=x.device)
        t_steps = (
            self.edm_config.sigma_max ** (1 / self.edm_config.rho) +
            step_indices / (num_steps - 1) *
            (self.edm_config.sigma_min ** (1 / self.edm_config.rho) -
             self.edm_config.sigma_max ** (1 / self.edm_config.rho))
        ) ** self.edm_config.rho
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])])

        # Main sampling loop.
        x_next = x.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next
            # Increase noise temporarily.
            gamma = (
                min(self.edm_config.S_churn / num_steps, np.sqrt(2) - 1)
                if self.edm_config.S_min <= t_cur <= self.edm_config.S_max else 0
            )
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            x_hat = (
                x_cur +
                (t_hat ** 2 - t_cur ** 2).sqrt() *
                self.edm_config.S_noise *
                self.randn_like(x_cur)
            )

            # Euler step.
            denoised = self.model_forward_wrapper(
                x_hat.to(torch.float32),
                t_hat.to(torch.float32),
                y,
                model_forward_fxn,
                mask_ratio=mask_ratio,
                **kwargs
            )['sample'].to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.model_forward_wrapper(
                    x_next.to(torch.float32),
                    t_next.to(torch.float32),
                    y,
                    model_forward_fxn,
                    mask_ratio=mask_ratio,
                    **kwargs
                )['sample'].to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        return x_next.to(torch.float32)

    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[list] = None,
        tokenized_prompts: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        guidance_scale: Optional[float] = 5.0,
        num_inference_steps: Optional[int] = 30,
        seed: Optional[int] = None,
        return_only_latents: Optional[bool] = False,
        **kwargs
    ) -> torch.Tensor:
        # _check_prompt_given(prompt, tokenized_prompts, prompt_embeds=None)
        assert prompt or tokenized_prompts, "Must provide either prompt or tokenized prompts"
        device = self.vae.device  # hack to identify model device during training
        rng_generator = torch.Generator(device=device)
        if seed:
            rng_generator = rng_generator.manual_seed(seed)

        # Convert prompt text to embeddings (zero out embeddings for classifier-free guidance)
        if tokenized_prompts is None:
            out = self.tokenizer.tokenize(prompt)
            tokenized_prompts = out['input_ids']
            attention_mask = (
                out['attention_mask'] if 'attention_mask' in out else None
            )
        text_embeddings = self.text_encoder.encode(
            tokenized_prompts.to(device),
            attention_mask=attention_mask.to(device) if attention_mask is not None else None
        )[0]

        latents = torch.randn(
            (len(text_embeddings), self.dit.in_channels, self.latent_res, self.latent_res),
            device=device,
            generator=rng_generator,
        )

        # iteratively denoise latents
        latents = self.edm_sampler_loop(
            latents,
            text_embeddings,
            num_inference_steps,
            cfg=guidance_scale
        )

        if return_only_latents:
            return latents

        # Decode latents with VAE
        latents = 1 / self.latent_scale * latents
        torch_dtype = DATA_TYPES[self.dtype]
        image = self.vae.decode(latents.to(torch_dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.float().detach()
        return image
    
    
def create_latent_diffusion(
    vae_name: str = 'stabilityai/stable-diffusion-xl-base-1.0',
    text_encoder_name: str = 'openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378', 
    dit_arch: str = 'MicroDiT_XL_2',
    latent_res: int = 32,
    in_channels: int = 4,
    pos_interp_scale: float = 1.0,
    dtype: str = 'bfloat16',
    precomputed_latents: bool = True,
    p_mean: float = -0.6,
    p_std: float = 1.2,
    train_mask_ratio: float = 0.
) -> LatentDiffusion:
    # retrieve max sequence length (s) and token embedding dim (d) from text encoder
    s, d = text_encoder_embedding_format(text_encoder_name)

    dit = getattr(model_zoo, dit_arch)(
        input_size=latent_res,
        caption_channels=d,
        pos_interp_scale=pos_interp_scale,
        in_channels=in_channels
    )

    vae = AutoencoderKL.from_pretrained(
        vae_name,
        subfolder=None if vae_name=='ostris/vae-kl-f8-d16' else 'vae',
        torch_dtype=DATA_TYPES[dtype],
        pretrained=True
    )

    text_encoder = UniversalTextEncoder(
        text_encoder_name,
        dtype=dtype,
        pretrained=True
    )
    tokenizer = UniversalTokenizer(text_encoder_name)

    model = LatentDiffusion(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        precomputed_latents=precomputed_latents,
        dtype=dtype,
        latent_res=latent_res,
        p_mean=p_mean,
        p_std=p_std,
        train_mask_ratio=train_mask_ratio
    )
    return model