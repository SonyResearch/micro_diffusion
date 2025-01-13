from huggingface_hub import hf_hub_download
import torch

from micro_diffusion.models.dit import DiT

model = DiT()

# equip with weights
filepath = hf_hub_download(repo_id="VSehwag24/MicroDiT", filename="dit_16_channel_37M_real_and_synthetic_data.pt")
state_dict = torch.load(filepath, map_location="cpu")
model.load_state_dict()

# push to the hub
model.push_to_hub("sony/MicroDiT-16-channel")

# load from the hub
model = DiT.from_pretrained("sony/MicroDiT-16-channel")