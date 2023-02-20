#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

#Libraries
import yaml
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
    InterpolationMode,
)
from torch.utils.tensorboard import SummaryWriter
from diffusers import UNet2DModel, DDPMScheduler
from diffusers import DDPMPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

# Check the diffusers version
check_min_version("0.13.0.dev0")

######MAIN######
def main():
    # GPU selection
    selected_gpu = 0 #select the GPU to use
    device = torch.device("cuda:" + str(selected_gpu) if torch.cuda.is_available() else "cpu")
    print(f'The device is: {device}\n')

    #load the config file
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    ### 1. Dataset loading and preprocessing
    # Dataset loading
    dataset = load_dataset(config['processing']['dataset_name'], split="train")
    # Define data augmentations
    preprocess = Compose(
        [
            Resize(config['processing']['resolution'], interpolation= InterpolationMode.BILINEAR), #getattr(InterpolationMode, config['processing']['interpolation'])),  # Smaller edge is resized to 256 preserving aspect ratio
            CenterCrop(config['processing']['resolution']),  # Center crop to the desired squared resolution
            RandomHorizontalFlip(),  # Horizontal randomly flip (data augmentation)
            ToTensor(),  # Convert to tensor (0, 1)
            Normalize([0.5], [0.5]),  # Map to (-1, 1) as a way to make data more similar to a Gaussian distribution
        ]
    )
    def transform(batch_dict):
        """Transform the images in the dataset to the desired format, this generates a dictionary with the key "images" containing the images in a list.
        It should include a formatting function as preproces. A formatting function is a callable that takes a batch as (dict) and returns a batch also as (dict).
        The formatting function is defined outside of the function (not self-contained)

        Args:
            batch_dict (dict): dictionary containing the images in a list under the key "image"

        Returns:
            dict: dictionary containing the images in a list under the key "images"
        """
        images = [preprocess(image.convert("RGB")) for image in batch_dict["image"]]
        return {"images": images}
    #set the transform function to the dataset
    dataset.set_transform(transform)
    # Create the dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config['processing']['batch_size'], num_workers= config['processing']['num_workers'], shuffle=True
    )

    ### 2. Model definition
    model = UNet2DModel(
        sample_size=config['processing']['resolution'],  # the target image resolution
        in_channels=config['model']['in_channels'],  # the number of input channels, 3 for RGB images
        out_channels=config['model']['out_channels'],  # the number of output channels
        layers_per_block=config['model']['layers_per_block'],  # how many ResNet layers to use per UNet block
        block_out_channels=config['model']['block_out_channels'],  # More channels -> more parameters
        down_block_types= config['model']['down_block_types'],
        up_block_types=config['model']['up_block_types'],
    )
    model.to(device) # send the model to the GPU

    ### 3. Training
    # Number of epochs
    num_epochs = config['training']['num_epochs']
    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr= config['training']['optimizer']['learning_rate'], # learning rate of the optimizer
        betas= (config['training']['optimizer']['beta_1'], config['training']['optimizer']['beta_2']), # betas according to the AdamW paper
        weight_decay= config['training']['optimizer']['weight_decay'], # weight decay according to the AdamW paper
        eps= config['training']['optimizer']['eps'] # epsilon according to the AdamW paper
    )
    # learning rate scheduler
    lr_scheduler = get_scheduler(
        name= config['training']['lr_scheduler']['name'], # name of the scheduler
        optimizer= optimizer, # optimizer to use
        num_warmup_steps= config['training']['lr_scheduler']['num_warmup_steps'] * config['training']['gradient_accumulation_steps'],
        num_training_steps= (len(train_dataloader) * num_epochs), #* config['training']['gradient_accumulation_steps']?
    )
    # Noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config['training']['noise_scheduler']['num_train_timesteps'],
        beta_schedule=config['training']['noise_scheduler']['beta_schedule'],
    )
    
    # trackers
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config['training']['gradient_accumulation_steps'])
    max_train_steps = num_epochs * num_update_steps_per_epoch

    print('The training is starting...\n')
    print(f'The number of examples is: {len(dataset)}\n')
    print(f'The number of epochs is: {num_epochs}\n')
    print(f'The number of batches is: {len(train_dataloader)}\n')
    print(f'The batch size is: {config["processing"]["batch_size"]}\n')
    # print(f'The number of update steps per epoch is: {num_update_steps_per_epoch}\n')
    print(f'Total optimization steps: {max_train_steps}\n')
    
    # Training loop
    # Create a TB summary writer
    writer = SummaryWriter()
    # scaler for mixed precision training (not needed for accelerate)
    scaler = torch.cuda.amp.GradScaler()
    # Loop over the epochs
    for epoch in range(num_epochs):
        #set the model to training mode explicitly
        model.train()
        # Create a progress bar
        pbar = tqdm(total=num_update_steps_per_epoch)
        pbar.set_description(f"Epoch {epoch}")
        # Loop over the batches
        for step, batch in enumerate(train_dataloader):
            # Get the images and send them to device (1st thing in device)
            clean_images = batch["images"].to(device)
            # Sample noise to add to the images and also send it to device(2nd thing in device)
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            # batch size variable for later use
            bs = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint( #create bs random integers from init=0 to end=timesteps, and send them to device (3rd thing in device)
                low= 0,
                high= noise_scheduler.num_train_timesteps,
                size= (bs,),
                device=clean_images.device ,
            ).long() #int64
            
            # Forward diffusion process: add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # auto cast for mixed precision training # not needed for accelerate
            with torch.autocast(device_type= "cuda", dtype= torch.float16):
                #### if existant, gradient accumulation starts here #with accelerator.accumulate(model)
                # Get the model prediction, #### This part changes according to the prediction type (e.g. epsilon, sample, etc.)
                noise_pred = model(noisy_images, timesteps).sample # sample tensor
                # Calculate the loss
                loss = F.mse_loss(noise_pred, noise)

            # Log the loss and logarithm loss to tensorboard
            global_step = epoch * len(train_dataloader) + step
            writer.add_scalar(tag= "Loss/train", scalar_value=loss, global_step= global_step) 
            writer.add_scalar(tag= "LogLoss/train", scalar_value= torch.log(loss), global_step= global_step)
            
            # Backpropagate the loss (sacler not needed for accelerate)
            scaler.scale(loss).backward() #loss is used as a gradient, coming from the accumulation of the gradients of the loss function (not implemented yet)
            # Update the parameters
            #optimizer.step()
            # unscale for gradient clipping
            scaler.unscale_(optimizer)
            # clip the gradients (clipping managed differently by accelerate)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # step the scaler
            scaler.step(optimizer)
            # Update the learning rate
            lr_scheduler.step()
            # Zero the gradients
            optimizer.zero_grad()
            # update the scaler (not needed for accelerate)
            scaler.update()
            #### if existant, gradient accumulation ends here
            # Update the progress bar
            pbar.update(1)
        #for each epoch, flush the writer (save the data)
        writer.flush()
        # Close the progress bar at the end of the epoch
        pbar.close()
    # Close the writer
    writer.close()
    print("Finished training!\n")
    
    ###4. Save the model
    # create the pipeline for generating images using the trained model
    image_pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    # save the pipeline
    image_pipe.save_pretrained(str(repo_path / config['saving']['pipeline_dir'] / config['saving']['pipeline_name']))
    
    print("The model has been saved.")

############################################################################################################

if __name__ == "__main__":
    main()