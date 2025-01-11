import torch
import numpy as np
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from typing import List, Dict, Union, Optional


class StreamingLatentsDataset(StreamingDataset):
    """Dataset class for loading precomputed latents from mds format.
    
    Args:
        streams: List of individual streams (in our case streams of individual datasets)
        shuffle: Whether to shuffle the dataset
        image_size: Size of images (256 or 512)
        cap_seq_size: Context length of text-encoder
        cap_emb_dim: Dimension of caption embeddings
        cap_drop_prob: Probability of using all zeros caption embedding (classifier-free guidance)
        batch_size: Batch size for streaming
    """

    def __init__(
        self,
        streams: Optional[List[Stream]] = None,
        shuffle: bool = False,
        image_size: Optional[int] = None,
        cap_seq_size: Optional[int] = None,
        cap_emb_dim: Optional[int] = None,
        cap_drop_prob: float = 0.0,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__(
            streams=streams,
            shuffle=shuffle,
            batch_size=batch_size,
        )

        self.image_size = image_size
        self.cap_seq_size = cap_seq_size
        self.cap_emb_dim = cap_emb_dim
        self.cap_drop_prob = cap_drop_prob

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str, float]]:
        sample = super().__getitem__(index)
        out = {}

        # Mask for zero'ed out captions in classifier-free guidance (cfg) training.
        # We replace caption embeddings with a zero vector in cfg guidance.
        out['drop_caption_mask'] = (
            0. if torch.rand(1) < self.cap_drop_prob else 1.
        )
        out['caption_latents'] = torch.from_numpy(
            np.frombuffer(sample['caption_latents'], dtype=np.float16)
            .copy()
        ).reshape(1, self.cap_seq_size, self.cap_emb_dim)

        if self.image_size == 256 and 'latents_256' in sample:
            out['image_latents'] = torch.from_numpy(
                np.frombuffer(sample['latents_256'], dtype=np.float16)
                .copy()
            ).reshape(-1, 32, 32)

        if self.image_size == 512 and 'latents_512' in sample:
            out['image_latents'] = torch.from_numpy(
                np.frombuffer(sample['latents_512'], dtype=np.float16)
                .copy()
            ).reshape(-1, 64, 64)

        # out['caption'] = sample['caption']
        return out


def build_streaming_latents_dataloader(
    datadir: Union[str, List[str]],
    batch_size: int,
    image_size: int = 256,
    cap_seq_size: int = 77,
    cap_emb_dim: int = 1024,
    cap_drop_prob: float = 0.0,
    shuffle: bool = True,
    drop_last: bool = True,
    **dataloader_kwargs
) -> DataLoader:
    """Creates a DataLoader for streaming latents dataset."""
    if isinstance(datadir, str):
        datadir = [datadir]

    streams = [Stream(remote=None, local=d) for d in datadir]

    dataset = StreamingLatentsDataset(
        streams=streams,
        shuffle=shuffle,
        image_size=image_size,
        cap_seq_size=cap_seq_size,
        cap_emb_dim=cap_emb_dim,
        cap_drop_prob=cap_drop_prob,
        batch_size=batch_size,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        drop_last=drop_last,
        **dataloader_kwargs,
    )

    return dataloader
