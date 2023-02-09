#Libraries
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from diffusers import UNet2DModel, DDPMScheduler
from diffusers import DDPMPipeline


#####Functions


######MAIN######
def main():
    # GPU selection
    selected_gpu = 0 # here you select the GPU used (0, 1 or 2)
    device = torch.device("cuda:" + str(selected_gpu) if torch.cuda.is_available() else "cpu")
    print(f'The device is: {device}\n')

    ###1. Dataset loading and preprocessing

    # Hyperparameters
    image_size = 64
    batch_size = 64
    # Dataset loading
    dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
    # Define data augmentations
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # Resize
            transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
            transforms.ToTensor(),  # Convert to tensor (0, 1)
            transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
        ]
    )
    def transform(batch_dict):
        """Transform the images in the dataset to the desired format, this generates a dictionary with the key "images" containing the images in a list.
        It should include a formatting function as preproces. A formatting function is a callable that akes a batch as (dict) and returns a batch also as (dict).

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
        dataset, batch_size=batch_size, shuffle=True
    )

    ###2. Model definition
    model = UNet2DModel(
        sample_size=image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(64, 128, 128, 256),  # More channels -> more parameters
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",  # a regular ResNet upsampling block
        ),
    )
    model.to(device) # send the model to the GPU

    ###3. Training
    # Hyperparameters
    num_epochs = 50
    #AdamW optimizer
    learning_rate = 4e-4
    beta_1 = 0.95
    beta_2 = 0.999
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate,
        betas=(beta_1, beta_2), weight_decay=1e-6,
        eps=1e-8
    )
    # Set the noise scheduler
    num_train_timesteps = 1000
    beta_schedule = "squaredcos_cap_v2"
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule
    )

    # Training loop
    # Create a summary writer
    writer = SummaryWriter()
    # Create a progress bar
    pbar = tqdm(total=num_epochs)
    # Loop over the epochs
    for epoch in range(num_epochs):
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
                0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device 
            ).long() #int64

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Get the model prediction
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0] #first element is sample tensor

            # Calculate the loss
            loss = F.mse_loss(noise_pred, noise)
            # Log the loss and logarithm loss to tensorboard
            global_step = epoch * len(train_dataloader) + step
            writer.add_scalar(tag= "Loss/train", scalar_value=loss, global_step= global_step) 
            writer.add_scalar(tag= "LogLoss/train", scalar_value= torch.log(loss), global_step= global_step)
            
            # Backpropagate the loss
            loss.backward(loss) #loss is used as a gradient, coming from the accumulation of the gradients of the loss function (not implemented yet)
            # Update the parameters
            optimizer.step()
            # Zero the gradients
            optimizer.zero_grad()
        #for each epoch, flush the writer (save the data)
        writer.flush()
        # Update the progress bar
        pbar.set_description(f"Epoch {epoch}")
        pbar.update(1)
    # Close the progress bar
    pbar.close()
    # Close the writer
    writer.close()
    print("Finished training!\n")
    
    ###4. Save the model
    # create the pipeline for generating images using the trained model
    image_pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    # save the pipeline
    image_pipe.save_pretrained("butterfly_pipeline")
    
    print("The model has been saved.")

############################################################################################################

if __name__ == "__main__":
    main()