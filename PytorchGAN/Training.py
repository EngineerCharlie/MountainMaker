import torch
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm

from Generator import Generator
from Discriminator import Discriminator
import LossFunctions
import DataSets
import CustomDataset
import os
import torch.nn as nn
import torch.nn.init as init

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("--------------------")
print("Training started with device: ", device)
print("CUDA version:", torch.version.cuda)
print(torch.cuda.get_device_name(0))
print("--------------------")
#print("CUDA_PATH: {}".format(os.environ["CUDA_PATH"]))
#print("CUDA_HOME: {}".format(os.environ["CUDA_HOME"]))

# Initialize the generator and discriminator
generator = Generator(input_channels=3, output_channels=3).to(device=device)
discriminator = Discriminator(input_channels=3).to(device=device)
initialize_weights(generator)
initialize_weights(discriminator)

# Define the optimizers for generator and discriminator
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))#, weight_decay=0.002)
criterion = nn.BCELoss()

# Training parameters
num_epochs = 100
resultsPath = "C:/Users/Juhász/NightSkyify/PytorchGAN/modelresults"

generator.train()
discriminator.train()

# Training loop
for epoch in range(num_epochs):
    # Iterate over the dataset
    for i, (real_images, drawing_images) in enumerate(CustomDataset.data_loader):

        # Move images to device
        real_images = real_images.to(device)
        drawing_images = (drawing_images).to(device)

        #noise = (0.1**0.5)*torch.randn(drawing_images.shape)
        #noise = torch.randn(8, 3, 1, 1).to(device)
        # Generate fake images
        fake_images = generator(drawing_images)
        # + (0.1**0.5)*torch.randn(drawing_images.shape)

        #print("shape: ", real_images.shape)
        #print("real images:", real_images.shape, " ones:", torch.ones_like(real_images[:,:1,:,:]).shape)
        #print("concat:", torch.concat([real_images, torch.ones_like(real_images[:,:1,:,:])], dim=1).shape)
        
        #Train discriminator
        d_real = discriminator(real_images)
        #d_loss_real = criterion(d_real, torch.ones_like(d_real))
        
        d_fake = discriminator(fake_images)
        d_loss = LossFunctions.discriminator_loss(d_real, d_fake)
        #d_loss_fake = criterion(d_fake, torch.zeros_like(d_fake))
        #d_loss = (d_loss_real + d_loss_fake) / 2

        discriminator_optimizer.zero_grad()
        d_loss.backward(retain_graph=True)
        discriminator_optimizer.step() 

        # Train generator
        
        fake_output = discriminator(fake_images)
        g_loss = LossFunctions.generator_loss(fake_output)
        #g_loss = criterion(fake_output, torch.ones_like(fake_output))
        generator_optimizer.zero_grad()
        g_loss.backward()
        generator_optimizer.step()

        print("i --> ",i)
        # Print loss
        if i % 2 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch Step [{i}/{len(DataSets.train_dataloader)}], "
                  f"Generator Loss: {g_loss.item():.4f}, Discriminator Loss: {d_loss.item():.4f}")

        
        # Test the model every few steps (e.g., every 500 steps)
        if i % 20 == 0:
            # Set the generator to evaluation mode
            generator.eval()

            # Iterate over the DataLoader to get a batch of data
            for data in DataSets.manualTestDataloader:
                # Extract input data from the batch
                input_data = data[0].to(device)  # Assuming the input data is in the first index of the batch tuple

                # Generate some sample images
                with torch.no_grad():
                    sample_images = generator(input_data)

                #print("shape:", sample_images.shape, " an: ", input_data.shape)
                #print("together shape:", torch.cat([input_data, sample_images], dim = 0).shape)

                # Save the sample images
                save_image(torch.cat([input_data[:2,...], sample_images[:2,...]], dim = 0), f"{resultsPath}/generated_images_epoch_{epoch+1}_step_{i}.png", nrow=2, normalize=True)

                # Only process one batch of data, then break out of the loop
                break

            # Set the generator back to training mode
            generator.train()
            
    # Save generated images
    save_image(torch.cat([drawing_images[:8,...], fake_images[:8,...], real_images[:8,...]], dim = 0), f"{resultsPath}/generated_images_epoch_{epoch + 1}.png", nrow=8, normalize=True)

torch.save(generator, f"C:/WorkingSets/Model/savedmodel_{epoch + 1}.pt")
