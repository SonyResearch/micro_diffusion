from setuptools import setup

setup(
    name="micro_diffusion",
    py_modules=["micro_diffusion"],
    install_requires=[
        'accelerate',
        'diffusers',
        'huggingface_hub',
        'torch<=2.3.1',
        'torchvision',
        'transformers',
        'timm',
        'open_clip_torch<=2.24.0',
        'easydict',
        'einops',
        'mosaicml-streaming<=0.9.0',
        'torchmetrics',
        'mosaicml[tensorboard, wandb]<=0.24.1',
        'tqdm',
        'pandas',
        'fastparquet',
        'omegaconf', 
        'datasets', 
        'hydra-core',
        'beautifulsoup4'
    ],
)