from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from streaming import Stream, StreamingDataset

from micro_diffusion.models.utils import UniversalTokenizer


class StreamingTextcapsDatasetForPreCompute(StreamingDataset):
    """Streaming dataset that resizes images to user-provided resolutions and tokenizes captions."""

    def __init__(
        self,
        streams: Sequence[Stream],
        transforms_list: List[Callable],
        batch_size: int,
        tokenizer_name: str,
        shuffle: bool = False,
        caption_key: str = 'caption_syn_pixart_llava15',
    ) -> None:
        super().__init__(
            streams=streams,
            shuffle=shuffle,
            batch_size=batch_size,
        )

        self.transforms_list = transforms_list
        self.caption_key = caption_key
        self.tokenizer = UniversalTokenizer(tokenizer_name)
        print("Created tokenizer:", tokenizer_name)
        assert self.transforms_list is not None, 'Must provide transforms to resize and center crop images'

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = super().__getitem__(index)
        ret = {}

        out = self.tokenizer.tokenize(sample[self.caption_key])
        ret[self.caption_key] = out['input_ids'].clone().detach()
        if 'attention_mask' in out:
            ret[f'{self.caption_key}_attention_mask'] = out['attention_mask'].clone().detach()

        for i, transform in enumerate(self.transforms_list):
            img = sample['jpg']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = transform(img)
            ret[f'image_{i}'] = img

        ret['sample'] = sample
        return ret


def build_streaming_textcaps_precompute_dataloader(
    datadir: Union[List[str], str],
    batch_size: int,
    resize_sizes: Optional[List[int]] = None,
    drop_last: bool = False,
    shuffle: bool = True,
    caption_key: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    **dataloader_kwargs,
) -> DataLoader:
    """Builds a streaming mds dataloader returning multiple image sizes and text captions."""
    assert resize_sizes is not None, 'Must provide target resolution for image resizing'
    
    datadir = [datadir] if isinstance(datadir, str) else datadir
    streams = [Stream(remote=None, local=path) for path in datadir]

    transforms_list = []
    for size in resize_sizes:
        transforms_list.append(
            transforms.Compose([
                transforms.Resize(
                    size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )

    dataset = StreamingTextcapsDatasetForPreCompute(
        streams=streams,
        shuffle=shuffle,
        transforms_list=transforms_list,
        batch_size=batch_size,
        caption_key=caption_key,
        tokenizer_name=tokenizer_name,
    )

    def custom_collate(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        out = {k: [] for k in batch[0].keys()}
        for sample in batch:
            for k, v in sample.items():
                out[k].append(v)
        return out

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        collate_fn=custom_collate,
        **dataloader_kwargs,
    )

    return dataloader