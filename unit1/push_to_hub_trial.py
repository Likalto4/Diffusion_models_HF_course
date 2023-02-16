from huggingface_hub import get_full_repo_name, HfApi, create_repo
from huggingface_hub import ModelCard


#define repo path
from pathlib import Path
import os
repo_path= Path.cwd().resolve().parent
repo_list = os.listdir(repo_path)
if '.gitignore' not in repo_list: raise Exception('The parent directory is not the root directory of the repository')

def main():
    # name the model and get the full name in HF Hub
    model_name = 'Unconditional_Butterflies_x64'
    hub_model_id = get_full_repo_name(model_name)
    #create repo in HF Hub
    create_repo(hub_model_id, exist_ok=True)
    #create API object
    api = HfApi() 
    #upload the scheduler folder
    api.upload_folder(
        folder_path= str(repo_path /'unit1' /'pipelines' /'butterfly_pipeline'/'scheduler'), path_in_repo='', repo_id=hub_model_id)
    #upload the unet folder
    api.upload_folder(
        folder_path=str(repo_path /'unit1'/ 'pipelines' /'butterfly_pipeline'/'unet'), path_in_repo='', repo_id=hub_model_id)
    #upload the single files
    api.upload_file(
        path_or_fileobj=str(repo_path / 'unit1' / 'pipelines' /'butterfly_pipeline'/'model_index.json'),
        path_in_repo='model_index.json',
        repo_id=hub_model_id,
    )

    #create model card
    content = f"""
---
license: mit
tags:
- pytorch
- diffusers
- unconditional-image-generation
- diffusion-models-class
---

# Model Card for a model trained based on the Unit 1 of the [Diffusion Models Class 🧨](https://github.com/huggingface/diffusion-models-class), not using accelarate yet.

This model is a diffusion model for unconditional image generation of cute but small 🦋.
The model was trained with 1000 images using the [DDPM](https://arxiv.org/abs/2006.11239) architecture. Images generated are of 64x64 pixel size.
The model was trained for 50 epochs with a batch size of 64, using around 11 GB of GPU memory. 

## Usage

```python
from diffusers import DDPMPipeline

pipeline = DDPMPipeline.from_pretrained({hub_model_id})
image = pipeline().images[0]
image
```
"""
    card = ModelCard(content)
    card.push_to_hub(hub_model_id)

if __name__ == "__main__":
    main()