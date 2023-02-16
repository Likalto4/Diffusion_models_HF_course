from huggingface_hub import get_full_repo_name, HfApi, create_repo
from huggingface_hub import ModelCard
import yaml

#define repo path
from pathlib import Path
import os
repo_path= Path.cwd().resolve().parent.parent
repo_list = os.listdir(repo_path)
if '.gitignore' not in repo_list: raise Exception('repo_path is not the root directory of the repository')

def main():
    #open config file
    with open(repo_path / 'unit1' / 'unconditional_training' / 'config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # name the model and get the full name in HF Hub
    hub_model_id = get_full_repo_name(config['saving']['hf']['repo_name'])
    #create repo in HF Hub, if it exists continue and overwrite if necessary
    create_repo(hub_model_id, exist_ok=True)
    #create API object
    api = HfApi() 
    #upload the scheduler folder
    pipeline_path = repo_path / config['saving']['pipeline_dir'] / config['saving']['pipeline_name']
    api.upload_folder(
        folder_path= str(pipeline_path /'scheduler'), path_in_repo='', repo_id=hub_model_id)
    #upload the unet folder
    api.upload_folder(
        folder_path=str(pipeline_path /'unet'), path_in_repo='', repo_id=hub_model_id)
    #upload the individual files
    api.upload_file(
        path_or_fileobj=str(pipeline_path /'model_index.json'), path_in_repo='model_index.json', repo_id=hub_model_id)

    #create model card
    # take content from model_card.yaml file
    with open(repo_path / config['saving']['hf']['model_card_path'], 'r') as f:
        content = f.read()
    card = ModelCard(content)
    card.push_to_hub(hub_model_id)

if __name__ == "__main__":
    main()