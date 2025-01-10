import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.vision_transformer import PatchEmbed
from typing import List

from .utils import (CaptionProjection, CrossAttention, Mlp, SelfAttention, T2IFinalLayer,
                   TimestepEmbedder, create_norm, get_2d_sincos_pos_embed, get_mask,
                   mask_out_token, modulate, unmask_tokens)

class AttentionBlockPromptEmbedding(nn.Module):
    """Attention block specifically for processing prompt embeddings.
    
    Args:
        dim (int): Input and output dimension
        head_dim (int): Dimension size per attention head
        mlp_ratio (float): Multiplier for feed-forward network hidden dimension w.r.t input dim
        multiple_of (int): Round feed-forward network hidden dimension up to nearest multiple of this value
        norm_eps (float): Epsilon value for layer normalization
        use_bias (bool): Whether to use bias terms in linear layers
    """
    def __init__(
        self,
        dim: int,
        head_dim: int, 
        mlp_ratio: float,
        multiple_of: int,
        norm_eps: float,
        use_bias: bool,
    ) -> None:
        super().__init__()
        assert dim % head_dim == 0, 'Hidden dimension must be divisible by head dim'
        
        self.dim = dim
        self.num_heads = dim // head_dim
        
        self.norm1 = create_norm('layernorm', dim, eps=norm_eps)
        self.attn = SelfAttention(
            dim=dim,
            num_heads=self.num_heads,
            qkv_bias=use_bias,
            norm_eps=norm_eps,
        )
        self.norm2 = create_norm('layernorm', dim, eps=norm_eps)
        self.mlp = FeedForward(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            multiple_of=multiple_of,
            use_bias=use_bias,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def custom_init(self, init_std: float = 0.02) -> None:
        self.attn.custom_init(init_std)
        self.mlp.custom_init(init_std)


class FeedForward(nn.Module):
    """Feed-forward block with SiLU activation.
    
    Args:
        dim (int): Input and output dimension
        hidden_dim (int): Hidden dimension betwen the two linear layers
        multiple_of (int): Round hidden dimension up to nearest multiple of this value
        use_bias (bool): Whether to use bias terms in linear layers
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        use_bias: bool,
    ):
        super().__init__()
        self.dim = dim
        hidden_dim = int(2 * hidden_dim / 3)
        self.hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.w1 = nn.Linear(dim, self.hidden_dim, bias=use_bias)
        self.w2 = nn.Linear(dim, self.hidden_dim, bias=use_bias)
        self.w3 = nn.Linear(self.hidden_dim, dim, bias=use_bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

    def custom_init(self, init_std: float) -> None:
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class FeedForwardECMoe(nn.Module):
    """Expert-Choice style Mixture of Experts feed-forward layer with GELU activation.
    
    Args:
        num_experts (int): Number of experts in the layer
        expert_capacity (float): Capacity factor determining tokens per expert
        dim (int): Input and output dimension
        hidden_dim (int): Hidden dimension between the two linear layers
        multiple_of (int): Round hidden dimension up to nearest multiple of this value
    """
    def __init__(
        self,
        num_experts: int,
        expert_capacity: float,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.dim = dim
        self.hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Parameter(torch.ones(num_experts, dim, self.hidden_dim))
        self.w2 = nn.Parameter(torch.ones(num_experts, self.hidden_dim, dim))
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 3
        n, t, d = x.shape
        tokens_per_expert = int(self.expert_capacity * t / self.num_experts)

        scores = self.gate(x)  # [n, t, e]
        probs = F.softmax(scores, dim=-1)  # [n, t, e]
        g, m = torch.topk(probs.permute(0, 2, 1), tokens_per_expert, dim=-1)  # [n, e, k], [n, e, k]
        p = F.one_hot(m, num_classes=t).float()  # [n, e, k, t]

        xin = torch.einsum('nekt, ntd -> nekd', p, x)  # [n, e, k, d]
        h = torch.einsum('nekd, edf -> nekf', xin, self.w1)  # [n, e, k, 4d]
        h = self.gelu(h)
        h = torch.einsum('nekf, efd -> nekd', h, self.w2)  # [n, e, k, d]

        out = g.unsqueeze(dim=-1) * h  # [n, e, k, d]
        out = torch.einsum('nekt, nekd -> ntd', p, out)
        return out
    
    def custom_init(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)


class DiTBlock(nn.Module):
    """DiT transformer block comprising Attention and MLP blocks. It supports choosing between 
     dense feed-forward and expert-choice style Mixture-of-Experts feed-forward blocks.
    
    Args:
        dim (int): Input and output dimension of the block
        head_dim (int): Dimension of each attention head
        mlp_ratio (float): Ratio for hidden dimension between linear layers in MLP block
        qkv_ratio (float): Ratio for dimension in qkv layers in attention block
        multiple_of (int): Round hidden dimensions up to nearest multiple of this value in MLP block
        pooled_emb_dim (int): Dimension of pooled caption embeddings
        norm_eps (float): Epsilon for layer normalization
        depth_init (bool): Whether to initialize weights of the last layer in MLP/Attention block based on block index
        layer_id (int): Index of this block in the dit model
        num_layers (int): Total number of blocks in the dit model
        compress_xattn (bool): Whether to scale cross-attention qkv dimension using qkv_ratio 
        use_bias (bool): Whether to use bias in linear layers
        moe_block (bool): Whether to use mixture of experts for MLP block
        num_experts (int): Number of experts if using MoE block
        expert_capacity (float): Capacity factor for each expert if using MoE block
    """
    def __init__(
        self,
        dim: int,
        head_dim: int,
        mlp_ratio: float,
        qkv_ratio: float,
        multiple_of: int,
        pooled_emb_dim: int,
        norm_eps: float,
        depth_init: bool,
        layer_id: int,
        num_layers: int,
        compress_xattn: bool,
        use_bias: bool,
        moe_block: bool,
        num_experts: int,
        expert_capacity: float,
    ):
        super().__init__()
        self.dim = dim
        qkv_hidden_dim = (
            (head_dim * 2) * ((int(dim * qkv_ratio) + head_dim * 2 - 1) // (head_dim * 2))
            if qkv_ratio != 1 else dim
        )
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = create_norm('layernorm', dim, eps=norm_eps)
        self.attn = SelfAttention(
            dim=dim,
            num_heads=qkv_hidden_dim // head_dim,
            qkv_bias=use_bias,
            norm_eps=norm_eps,
            hidden_dim=qkv_hidden_dim,
        )
        self.cross_attn = CrossAttention(
            dim=dim,
            num_heads=qkv_hidden_dim // head_dim if compress_xattn else dim // head_dim,
            qkv_bias=use_bias,
            norm_eps=norm_eps,
            hidden_dim=qkv_hidden_dim if compress_xattn else dim,
        )
        self.norm2 = create_norm('layernorm', dim, eps=norm_eps)
        self.norm3 = create_norm('layernorm', dim, eps=norm_eps)
        
        self.mlp = (
            FeedForwardECMoe(num_experts, expert_capacity, dim, mlp_hidden_dim, multiple_of)
            if moe_block else
            FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.GELU(approximate="tanh"),
            nn.Linear(pooled_emb_dim, 6 * dim, bias=True),
        )
        
        self.weight_init_std = (
            0.02 / (2 * (layer_id + 1)) ** 0.5 if depth_init else
            0.02 / (2 * num_layers) ** 0.5
        )
        
    def forward(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, **kwargs) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + self.cross_attn(self.norm2(x), y)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x

    def custom_init(self):
        for norm in (self.norm1, self.norm2, self.norm3):
            norm.reset_parameters()
        self.attn.custom_init(self.weight_init_std)
        self.cross_attn.custom_init(self.weight_init_std)
        self.mlp.custom_init(self.weight_init_std)
    

class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) model than support conditioning on caption embeddings for text-to-image generation.
    
    Args:
        input_size (int, default: 32): Size of input image (assumed square)
        patch_size (int, default: 2): Size of patches for patch embedding
        in_channels (int, default: 4): Number of input image channels (by default assuming four channel latent space)
        dim (int, default: 1152): Dimension of transformer backbone, i.e., dimension of major transformer layers
        depth (int, default: 28): Number of transformer blocks
        head_dim (int, default: 64): Dimension of each attention head
        multiple_of (int, default: 256): Round hidden dimensions up to nearest multiple of this value in MLP block
        caption_channels (int, default: 4096): Number of channels in caption embeddings
        pos_interp_scale (float, default: 1.0): Scale for positional embedding interpolation (1.0 for 256x256, 2.0 for 512x512)
        norm_eps (float, default: 1e-6): Epsilon for layer normalization
        depth_init (bool, default: True): Whether to use depth-dependent initialization in DiT blocks
        qkv_multipliers (List[float], default: [1.0]): Multipliers for QKV projection dimensions in DiT blocks
        ffn_multipliers (List[float], default: [4.0]): Multipliers for FFN hidden dimensions in DiT blocks
        use_patch_mixer (bool, default: True): Whether to use patch mixer layers
        patch_mixer_depth (int, default: 4): Number of patch mixer blocks
        patch_mixer_dim (int, default: 512): Dimension of patch-mixer layers
        patch_mixer_qkv_ratio (float, default: 1.0): Multipliers for QKV projection dimensions in patch-mixer blocks
        patch_mixer_mlp_ratio (float, default: 1.0): Multipliers for FFN hidden dimensions in patch-mixer blocks
        use_bias (bool, default: True): Whether to use bias in linear layers
        num_experts (int, default: 8):  Number of experts if using MoE block
        expert_capacity (int, default: 1): Capacity factor for each expert if using MoE FFN layers
        experts_every_n (int, default: 2): Add MoE FFN layers every n blocks
    """
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        dim: int = 1152,
        depth: int = 28,
        head_dim: int = 64,
        multiple_of: int = 256,
        caption_channels: int = 1024,
        pos_interp_scale: float = 1.0,
        norm_eps: float = 1e-6,
        depth_init: bool = True,
        qkv_multipliers: List[float] = [1.0],
        ffn_multipliers: List[float] = [4.0],
        use_patch_mixer: bool = True,
        patch_mixer_depth: int = 4,
        patch_mixer_dim: int = 512,
        patch_mixer_qkv_ratio: float = 1.0,
        patch_mixer_mlp_ratio: float = 1.0,
        use_bias: bool = True,
        num_experts: int = 8,
        expert_capacity: int = 1,
        experts_every_n: int = 2
    ):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.head_dim = head_dim
        self.pos_interp_scale = pos_interp_scale
        self.use_patch_mixer = use_patch_mixer
        
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, dim, bias=True
        )
        self.t_embedder = TimestepEmbedder(dim, approx_gelu)

        num_patches = self.x_embedder.num_patches
        self.base_size = input_size // self.patch_size
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, dim))

        self.y_embedder = CaptionProjection(
            in_channels=caption_channels,
            hidden_size=dim,
            act_layer=approx_gelu,
            norm_layer=create_norm('layernorm', dim, eps=norm_eps)
        )

        self.y_emb_preprocess = AttentionBlockPromptEmbedding(
            dim,
            head_dim,
            mlp_ratio=4.0,
            multiple_of=multiple_of,
            norm_eps=norm_eps,
            use_bias=use_bias
        )
        
        self.pooled_y_emb_process = Mlp(
            dim,
            dim,
            dim,
            approx_gelu,
            norm_layer=create_norm('layernorm', dim, eps=norm_eps)
        )

        if self.use_patch_mixer:
            expert_blocks_idx = [
                i for i in range(1, patch_mixer_depth) 
                if (i+1) % experts_every_n == 0
            ]
            is_moe_block = [
                True if i in expert_blocks_idx else False 
                for i in range(patch_mixer_depth)
            ]
                 
            self.patch_mixer = nn.ModuleList([
                DiTBlock(
                    dim=patch_mixer_dim,
                    head_dim=head_dim,
                    mlp_ratio=patch_mixer_mlp_ratio,
                    qkv_ratio=patch_mixer_qkv_ratio,
                    multiple_of=multiple_of,
                    pooled_emb_dim=dim,
                    norm_eps=norm_eps,
                    depth_init=False,
                    layer_id=0,
                    num_layers=depth,
                    compress_xattn=False,
                    use_bias=use_bias,
                    moe_block=is_moe_block[i],
                    num_experts=num_experts,
                    expert_capacity=expert_capacity
                ) for i in range(patch_mixer_depth)
            ])

            # Couple of projection layers
            if patch_mixer_dim != dim:
                self.patch_mixer_map_xin = nn.Sequential(
                    create_norm('layernorm', dim, eps=norm_eps),
                    nn.Linear(dim, patch_mixer_dim, bias=use_bias)
                )
                self.patch_mixer_map_xout = nn.Sequential(
                    create_norm('layernorm', patch_mixer_dim, eps=norm_eps),
                    nn.Linear(patch_mixer_dim, dim, bias=use_bias)
                )
                self.patch_mixer_map_y = nn.Sequential(
                    create_norm('layernorm', dim, eps=norm_eps),
                    nn.Linear(dim, patch_mixer_dim, bias=use_bias)
                )
            else:
                self.patch_mixer_map_xin = nn.Identity()
                self.patch_mixer_map_xout = nn.Identity()
                self.patch_mixer_map_y = nn.Identity()

        assert len(ffn_multipliers) == len(qkv_multipliers)
        if len(ffn_multipliers) == depth:
            qkv_ratios = qkv_multipliers
            mlp_ratios = ffn_multipliers
        else:
            # Distribute the multiplers across each partition
            num_splits = len(ffn_multipliers)
            assert depth % num_splits == 0, 'number of blocks should be divisible by number of splits'
            depth_per_split = depth // num_splits
            qkv_ratios = list(np.array([
                [m]*depth_per_split for m in qkv_multipliers
            ]).reshape(-1))
            mlp_ratios = list(np.array([
                [m]*depth_per_split for m in ffn_multipliers
            ]).reshape(-1))

        # Don't use MoE in last block
        expert_blocks_idx = [
            i for i in range(0, depth - 1) 
            if (i+1) % experts_every_n == 0
        ]
        is_moe_block = [
            True if i in expert_blocks_idx else False 
            for i in range(depth)
        ]
        
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=dim,
                head_dim=head_dim,
                mlp_ratio=mlp_ratios[i],
                qkv_ratio=qkv_ratios[i],
                multiple_of=multiple_of,
                pooled_emb_dim=dim,
                norm_eps=norm_eps,
                depth_init=depth_init,
                layer_id=i,
                num_layers=depth,
                compress_xattn=False,
                use_bias=use_bias,
                moe_block=is_moe_block[i],
                num_experts=num_experts,
                expert_capacity=expert_capacity
            ) for i in range(depth)
        ])
        
        self.register_buffer(
            "mask_token", 
            torch.zeros(1, 1, patch_size ** 2 * self.out_channels)
        )
        self.final_layer = T2IFinalLayer(
            dim,
            dim,
            patch_size,
            self.out_channels,
            approx_gelu,
            create_norm('layernorm', dim, eps=norm_eps)
        )

        self.initialize_weights()

    def forward_without_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        mask_ratio: float = 0,
        **kwargs
    ) -> dict:
        """Forward pass without classifier-free guidance.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            t: Timestep tensor of shape (batch_size,)
            y: Caption embedding tensor of shape (batch_size, 1, seq_len, dim)
            mask_ratio: Ratio of patches to mask during training (between 0 and 1)

        Returns:
            dict: Dictionary containing:
                - 'sample': Output tensor of shape (batch_size, out_channels, height, width)
                - 'mask': Optional binary mask tensor, if masking was applied else None
        """
        self.h = x.shape[-2] // self.patch_size
        self.w = x.shape[-1] // self.patch_size

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t.expand(x.shape[0]))  # (N, D)

        y = self.y_embedder(y)  # (N, 1, L, D)
        y = self.y_emb_preprocess(y.squeeze(dim=1)).unsqueeze(dim=1)  # (N, 1, L, D) -> (N, D)
        y_pooled = self.pooled_y_emb_process(y.mean(dim=-2).squeeze(dim=1))
        t = t + y_pooled

        mask = None
        
        if self.use_patch_mixer:
            x = self.patch_mixer_map_xin(x)
            y_mixer = self.patch_mixer_map_y(y)
            for block in self.patch_mixer:
                x = block(x, y_mixer, t)  # (N, T, D_mixer)
        
        if mask_ratio > 0:
            mask_dict = get_mask(
                x.shape[0], x.shape[1], 
                mask_ratio=mask_ratio, 
                device=x.device
            )
            ids_keep = mask_dict['ids_keep']
            ids_restore = mask_dict['ids_restore']
            mask = mask_dict['mask']
            x = mask_out_token(x, ids_keep)
        
        if self.use_patch_mixer:
            # Project mixer out to backbone transformer dim (do after masking to save compute)
            x = self.patch_mixer_map_xout(x)

        for block in self.blocks:
            x = block(x, y, t)  # (N, T, D)
        
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        
        if mask_ratio > 0:
            x = unmask_tokens(x, ids_restore, self.mask_token)

        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return {'sample': x, 'mask': mask}
    
    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg: float = 1.0,
        mask_ratio: float = 0,
        **kwargs
    ) -> dict:
        """Forward pass with classifier-free guidance.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            t: Timestep tensor of shape (batch_size,)
            y: Caption embedding tensor of shape (batch_size, 1, seq_len, dim)
            cfg: Classifier-free guidance scale (1.0 means no guidance)
            mask_ratio: Ratio of patches to mask during training (between 0 and 1)

        Returns:
            dict: Dict with output tensor of shape (batch_size, out_channels, height, width)
        """
        x = torch.cat([x, x], 0)
        y = torch.cat([y, torch.zeros_like(y)], 0)
        if len(t) != 1:
            t = torch.cat([t, t], 0)
        
        eps = self.forward_without_cfg(x, t, y, mask_ratio, **kwargs)['sample']
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        eps = uncond_eps + cfg * (cond_eps - uncond_eps)
        return {'sample': eps}

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg: float = 1.0,
        **kwargs
    ) -> dict:
        """Routes to appropriate forward pass based on classifier-free guidance value."""
        if cfg != 1.0:
            return self.forward_with_cfg(x, t, y, cfg, **kwargs)
        else:
            return self.forward_without_cfg(x, t, y, **kwargs)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Reverses the patch embedding process to reconstruct the original image dimensions."""
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1]**0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def initialize_weights(self) -> None:
        """Initialize model weights with custom initialization scheme."""
        def zero_bias(m: nn.Module) -> None:
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                zero_bias(module)  # All bias in the model init to zero

        # Baseline init of all parameters
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches**0.5),
            pos_interp_scale=self.pos_interp_scale,
            base_size=self.base_size
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.pooled_y_emb_process.fc1.weight, std=0.02)
        nn.init.normal_(self.pooled_y_emb_process.fc2.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)
        
        # Custom init of blocks
        for block in self.blocks:
            block.custom_init()
        for block in self.patch_mixer:
            block.custom_init()

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)

        for block in self.patch_mixer:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)

        self.y_emb_preprocess.custom_init()
        nn.init.constant_(self.y_emb_preprocess.attn.proj.weight, 0)
        nn.init.constant_(self.y_emb_preprocess.mlp.w3.weight, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)


def MicroDiT_Tiny_2(
    caption_channels: int = 1024,
    qkv_ratio: List[float] = [0.5, 1.0],
    mlp_ratio: List[float] = [0.5, 4.0],
    pos_interp_scale: float = 1.0,
    input_size: int = 32,
    num_experts: int = 8,
    expert_capacity: float = 2.0,
    experts_every_n: int = 2,
    in_channels: int = 4,
    **kwargs
) -> DiT:
    depth = 16
    model = DiT(
        input_size=input_size,
        patch_size=2,
        in_channels=in_channels,
        dim=512,
        depth=depth,
        head_dim=32,
        multiple_of=256,
        caption_channels=caption_channels,
        pos_interp_scale=pos_interp_scale,
        norm_eps=1e-6,
        depth_init=True,
        qkv_multipliers=np.linspace(qkv_ratio[0], qkv_ratio[1], num=depth, dtype=float),
        ffn_multipliers=np.linspace(mlp_ratio[0], mlp_ratio[1], num=depth, dtype=float),
        use_patch_mixer=True,
        patch_mixer_depth=4,
        patch_mixer_dim=512,  # allocating higher budget to mixer layers
        patch_mixer_qkv_ratio=1.0,
        patch_mixer_mlp_ratio=4.0,
        use_bias=False,
        num_experts=num_experts,
        expert_capacity=expert_capacity,
        experts_every_n=experts_every_n,
        **kwargs
    )
    return model


def MicroDiT_XL_2(
    caption_channels: int = 1024,
    qkv_ratio: List[float] = [0.5, 1.0],
    mlp_ratio: List[float] = [0.5, 4.0],
    pos_interp_scale: float = 1.0,
    input_size: int = 32,
    num_experts: int = 8,
    expert_capacity: float = 2.0,
    experts_every_n: int = 2,
    in_channels: int = 4,
    **kwargs
) -> DiT:
    depth = 28
    model = DiT(
        input_size=input_size,
        patch_size=2,
        in_channels=in_channels,
        dim=1024,
        depth=depth,
        head_dim=64,
        multiple_of=256,
        caption_channels=caption_channels,
        pos_interp_scale=pos_interp_scale,
        norm_eps=1e-6,
        depth_init=True,
        qkv_multipliers=np.linspace(qkv_ratio[0], qkv_ratio[1], num=depth, dtype=float),
        ffn_multipliers=np.linspace(mlp_ratio[0], mlp_ratio[1], num=depth, dtype=float),
        use_patch_mixer=True,
        patch_mixer_depth=6,
        patch_mixer_dim=768,  # allocating higher budget to mixer layers
        patch_mixer_qkv_ratio=1.0,
        patch_mixer_mlp_ratio=4.0,
        use_bias=False,
        num_experts=num_experts,
        expert_capacity=expert_capacity,
        experts_every_n=experts_every_n,
        **kwargs
    )
    return model