import math
from collections.abc import Iterable
from itertools import repeat
from typing import Optional, Tuple, Dict, Union, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchmetrics import Metric

import open_clip
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel, 
    T5Tokenizer
)

DATA_TYPES = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32
}


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Applies modulation to input tensor using shift and scale factors."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# Ref: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py
class Mlp(nn.Module):
    """MLP implementation from timm (without the dropout layers)
    
    Args:
        in_features (int): Input tensor dimension
        hidden_features (Optional[int], None): Number of hidden features. If None, same as in_features
        out_features (Optional[int], None): Number of output features. If None, same as in_features
        act_layer (Any, nn.GELU): Activation layer constructor. Defaults to GELU with tanh approximation
        norm_layer (Optional[Any], None): Normalization layer constructor. If None, uses Identity
        bias (bool, True): Whether to use bias in linear layers
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Any = lambda: nn.GELU(approximate="tanh"),
        norm_layer: Optional[Any] = None,
        bias: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x
    

def create_norm(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    """Creates a normalization layer based on the specified type."""
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, bias=False)
    elif norm_type == "np_layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
    else:
        raise ValueError('Norm type not supported!')
    

class CrossAttention(nn.Module):
    """Cross attention layer.
    
    Args:
        dim (int): Input and output tensor dimension
        num_heads (int): Number of attention heads
        qkv_bias (bool, True): Whether to use bias in QKV linear layers
        norm_eps (float, 1e-6): Epsilon for normalization layers
        hidden_dim (Optional[int], None): Dimension for qkv space. If None, same as input dim
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        norm_eps: float = 1e-6,
        hidden_dim: Optional[int] = None
    ):
        super(CrossAttention, self).__init__()
        if hidden_dim is None:
            hidden_dim = dim
        assert hidden_dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv_bias = qkv_bias

        self.q_linear = nn.Linear(dim, hidden_dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, hidden_dim*2, bias=qkv_bias)
        self.proj = nn.Linear(hidden_dim, dim, bias=qkv_bias)

        self.ln_q = create_norm('np_layernorm', dim=hidden_dim, eps=norm_eps)
        self.ln_k = create_norm('np_layernorm', dim=hidden_dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_linear(x).reshape(B, N, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).reshape(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        q = self.ln_q(q.view(B, N, self.num_heads * self.head_dim)).view(
            B, N, self.num_heads, self.head_dim).to(q.dtype)
        k = self.ln_k(k.view(B, -1, self.num_heads * self.head_dim)).view(
            B, -1, self.num_heads, self.head_dim).to(k.dtype)

        x = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=False
        ).transpose(1, 2).contiguous()
        
        x = x.view(B, -1, self.num_heads * self.head_dim)
        x = self.proj(x)
        return x
    
    def custom_init(self, init_std: float) -> None:
        for linear in (self.q_linear, self.kv_linear):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=init_std)


class SelfAttention(nn.Module):
    """Self attention layer.
    
    Args:
        dim (int): Input and output tensor dimension
        num_heads (int): Number of attention heads
        qkv_bias (bool, True): Whether to use bias in QKV linear layers
        norm_eps (float, 1e-6): Epsilon for normalization layers
        hidden_dim (Optional[int], None): Dimension for qkv space. If None, same as input dim
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        norm_eps: float = 1e-6,
        hidden_dim: Optional[int] = None
    ):
        super().__init__()
        self.dim = dim
        if hidden_dim is None:
            hidden_dim = dim
        assert hidden_dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv_bias = qkv_bias

        self.qkv = nn.Linear(dim, hidden_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_dim, dim, bias=qkv_bias)

        self.ln_q = create_norm('np_layernorm', dim=hidden_dim, eps=norm_eps)
        self.ln_k = create_norm('np_layernorm', dim=hidden_dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        q = self.ln_q(q.view(B, N, self.num_heads * self.head_dim)).view(
            B, N, self.num_heads, self.head_dim).to(q.dtype)
        k = self.ln_k(k.view(B, N, self.num_heads * self.head_dim)).view(
            B, N, self.num_heads, self.head_dim).to(k.dtype)

        x = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=False
        ).transpose(1, 2).contiguous()
        
        x = x.view(B, N, self.num_heads * self.head_dim)
        x = self.proj(x)
        return x
    
    def custom_init(self, init_std: float) -> None:
        nn.init.trunc_normal_(self.qkv.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=init_std)


class T2IFinalLayer(nn.Module):
    """The final layer of DiT architecture.
    
    Args:
        hidden_size (int): Size of hidden dimension
        time_emb_dim (int): Dimension of timestep embeddings
        patch_size (int): Size of image patches 
        out_channels (int): Number of output channels
        act_layer (Any): Activation layer constructor
        norm_final (nn.Module): Final normalization layer
    """
    def __init__(
        self,
        hidden_size: int,
        time_emb_dim: int,
        patch_size: int,
        out_channels: int,
        act_layer: Any,
        norm_final: nn.Module
    ):
        super().__init__()
        self.linear = nn.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(time_emb_dim, 2 * hidden_size, bias=True)
        )
        self.norm_final = norm_final

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations.
    
    Args:
        hidden_size (int): Size of hidden dimension
        act_layer (Any): Activation layer constructor
        frequency_embedding_size (int, 512): Size of frequency embedding
    """
    def __init__(
        self,
        hidden_size: int,
        act_layer: Any,
        frequency_embedding_size: int = 512
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            act_layer(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0,
                end=half,
                dtype=torch.float32,
                device=t.device
            ) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        return self.mlp(t_freq)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


class CaptionProjection(nn.Module):
    """Projects caption embeddings to model dimension.
    
    Args:
        in_channels (int): Number of input channels
        hidden_size (int): Size of hidden dimension
        act_layer (Any): Activation layer constructor
        norm_layer (Any): Normalization layer constructor
    """
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        act_layer: Any,
        norm_layer: Any
    ) -> None:
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            norm_layer=norm_layer
        )
    
    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        return self.y_proj(caption)


def ntuple(n: int):
    """Converts input into n-tuple."""
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: Union[int, Tuple[int, int]],
    cls_token: bool = False,
    extra_tokens: int = 0,
    pos_interp_scale: float = 1.0,
    base_size: int = 16
) -> np.ndarray:
    """Get 2D sinusoidal positional embeddings."""
    to_2tuple = ntuple(2)
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    # Interpolate position embeddings to adapt model across resolutions. Interestingly, without any interpolation
    # the model does converge slowly at start (~1000 steps) but eventually achieves near similar qualitative performance.
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / pos_interp_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / pos_interp_scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Get 2D sinusoidal positional embeddings from grid."""
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Get 1D sinusoidal positional embeddings from grid."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_mask(batch: int, length: int, mask_ratio: float, device: torch.device) -> Dict[str, torch.Tensor]:
    """Get binary mask for input sequence. 
    
    mask: binary mask, 0 is keep, 1 is remove
    ids_keep: indices of tokens to keep
    ids_restore: indices to restore the original order
    """
    len_keep = int(length * (1 - mask_ratio))
    noise = torch.rand(batch, length, device=device)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]

    mask = torch.ones([batch, length], device=device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return {
        'mask': mask,
        'ids_keep': ids_keep,
        'ids_restore': ids_restore
    }


def mask_out_token(x: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
    """Mask out tokens specified by ids_keep."""
    N, L, D = x.shape  # batch, length, dim
    x_masked = torch.gather(
        x,
        dim=1,
        index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
    )
    return x_masked


def unmask_tokens(x: torch.Tensor, ids_restore: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
    """Unmask tokens using provided mask token."""
    mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
    x_ = torch.cat([x, mask_tokens], dim=1)
    x_ = torch.gather(
        x_,
        dim=1,
        index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
    )  # unshuffle
    return x_


class UniversalTextEncoder(torch.nn.Module):
    """Universal text encoder supporting multiple model types.
    
    Args:
        name (str): Name/path of the model to load
        dtype (str): Data type for model weights
        pretrained (bool, True): Whether to load pretrained weights
    """
    def __init__(self, name: str, dtype: str, pretrained: bool = True):
        super().__init__()
        self.name = name
        if self.name.startswith("openclip:"):
            assert pretrained, 'Load default pretrained model from openclip'
            self.encoder = openclip_text_encoder(
                open_clip.create_model_and_transforms(name.lstrip('openclip:'))[0],
                torch_dtype=DATA_TYPES[dtype]
            )
        elif self.name == "DeepFloyd/t5-v1_1-xxl":
            self.encoder = T5EncoderModel.from_pretrained(
                name,
                torch_dtype=DATA_TYPES[dtype],
                pretrained=pretrained
            )
        else:
            self.encoder = CLIPTextModel.from_pretrained(
                name,
                subfolder='text_encoder',
                torch_dtype=DATA_TYPES[dtype],
                pretrained=pretrained
            )

    def encode(self, tokenized_caption: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.name == "DeepFloyd/t5-v1_1-xxl":
            out = self.encoder(
                tokenized_caption,
                attention_mask=attention_mask
            )['last_hidden_state']
            out = out.unsqueeze(dim=1)
            return out, None
        else:
            return self.encoder(tokenized_caption)


class openclip_text_encoder(torch.nn.Module):
    """OpenCLIP text encoder wrapper.
    
    Args:
        clip_model (Any): OpenCLIP model instance
        dtype (torch.dtype, torch.float32): Data type for model weights
    """
    def __init__(self, clip_model: Any, dtype: torch.dtype = torch.float32, **kwargs) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.device = None
        self.dtype = dtype

    def forward_fxn(self, text: torch.Tensor) -> Tuple[torch.Tensor, None]:
        cast_dtype = self.clip_model.transformer.get_cast_dtype()
        x = self.clip_model.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.clip_model.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x = x.unsqueeze(dim=1) # [batch_size, 1, n_ctx, transformer.width] expected for text_emb
        return x, None # HF encoders expected to return multiple values with first being text emb

    def forward(self, text: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, None]:
        with torch.autocast(device_type='cuda', dtype=self.dtype):
            return self.forward_fxn(text)


def text_encoder_embedding_format(enc: str) -> Tuple[int, int]:
    """Returns sequence length and token embedding dimension for text encoder."""
    if enc in [
        'stabilityai/stable-diffusion-2-base',
        'runwayml/stable-diffusion-v1-5',
        'CompVis/stable-diffusion-v1-4'
    ]:
        return 77, 1024
    if enc in ['openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378']:
        return 77, 1024
    if enc in ["DeepFloyd/t5-v1_1-xxl"]:
        return 120, 4096
    raise ValueError(f'Please specifcy the sequence and embedding size of {enc} encoder')
    
    
class simple_2_hf_tokenizer_wrapper:
    """Simple wrapper to make OpenCLIP tokenizer match HuggingFace interface.
    
    Args:
        tokenizer (Any): OpenCLIP tokenizer instance
    """
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.model_max_length = self.tokenizer.context_length
        
    def __call__(
        self,
        caption: str,
        padding: str = 'max_length',
        max_length: Optional[int] = None,
        truncation: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        return {'input_ids': self.tokenizer(caption, context_length=max_length)}


class UniversalTokenizer:
    """Universal tokenizer supporting multiple model types.
    
    Args:
        name (str): Name/path of the tokenizer to load
    """
    def __init__(self, name: str):
        self.name = name
        s, d = text_encoder_embedding_format(name)
        if self.name.startswith("openclip:"):
            self.tokenizer = simple_2_hf_tokenizer_wrapper(
                open_clip.get_tokenizer(name.lstrip('openclip:'))
            )
            assert s == self.tokenizer.model_max_length, "simply check of text_encoder_embedding_format"
        elif self.name == "DeepFloyd/t5-v1_1-xxl":
            self.tokenizer = T5Tokenizer.from_pretrained(name) # for t5 we would use a smaller than max_seq_length
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(name, subfolder='tokenizer')
            assert s == self.tokenizer.model_max_length, "simply check of text_encoder_embedding_format"
        self.model_max_length = s
        
    def tokenize(self, captions: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        if self.name == "DeepFloyd/t5-v1_1-xxl":
            text_tokens_and_mask = self.tokenizer(
                captions,
                padding='max_length',
                max_length=self.model_max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
            return {
                'input_ids': text_tokens_and_mask['input_ids'],
                'attention_mask': text_tokens_and_mask['attention_mask']
            }
        else:
            # Avoid attention mask for CLIP tokenizers as they are not used
            tokenized_caption = self.tokenizer(
                captions,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt'
            )['input_ids']
            return {'input_ids': tokenized_caption}
        

def _cast_if_autocast_enabled(tensor: torch.Tensor) -> torch.Tensor:
    """Cast tensor if autocast is enabled."""
    if torch.is_autocast_enabled():
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor


class DistLoss(Metric):
    """Distributed loss metric.
    
    Args:
        kwargs (Any): Additional arguments passed to parent class
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("loss", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, value: torch.Tensor) -> None:
        self.loss += value
        self.batches += 1

    def compute(self) -> torch.Tensor:
        return self.loss.float() / self.batches