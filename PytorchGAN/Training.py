import torch
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm

from Generator import Generator
from Discriminator import Discriminator
import LossFunctions
import DataSets
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print("--------------------")
print("Training started with device: ", device)
print("CUDA version:", torch.version.cuda)
print("--------------------")
#print("CUDA_PATH: {}".format(os.environ["CUDA_PATH"]))
#print("CUDA_HOME: {}".format(os.environ["CUDA_HOME"]))

# Initialize the generator and discriminator
generator = Generator(input_channels=3, output_channels=3).to(device=device)
discriminator = Discriminator(input_channels=6).to(device=device)

# Define the optimizers for generator and discriminator
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))

# Training parameters
num_epochs = 4
resultsPath = "C:/Users/Juhász/NightSkyify/PytorchGAN/modelresults"

# Training loop
for epoch in range(num_epochs):
    # Iterate over the dataset
    for i, (real_images, _) in enumerate(DataSets.train_dataloader):
        # Move real images to device
        real_images = real_images.to(device)
        #print("REAL SHAPE: ", real_images.shape)
        # Generate fake images
        fake_images = generator(real_images).to(device)
        #print("FAKE SHAPE: ", fake_images.shape)

        # Train discriminator
        discriminator_optimizer.zero_grad()
        real_output = discriminator(torch.cat([real_images, real_images], dim=1))
        fake_output = discriminator(torch.cat([real_images, fake_images], dim=1))
        d_loss = LossFunctions.discriminator_loss(real_output, fake_output)
        d_loss.backward(retain_graph=True)
        discriminator_optimizer.step()

        # Train generator
        generator_optimizer.zero_grad()
        fake_output = discriminator(torch.cat([real_images, fake_images], dim=1))
        g_loss = LossFunctions.generator_loss(fake_output)
        g_loss.backward()
        generator_optimizer.step()

        print("i --> ",i)
        # Print loss
        if i % 2 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Batch Step [{i}/{len(DataSets.train_dataloader)}], "
                  f"Generator Loss: {g_loss.item():.4f}, Discriminator Loss: {d_loss.item():.4f}")

        
        # Test the model every few steps (e.g., every 500 steps)
        if i % 25 == 0:
            # Set the generator to evaluation mode
            generator.eval()

            # Iterate over the DataLoader to get a batch of data
            for data in DataSets.manualTestDataloader:
                # Extract input data from the batch
                input_data = data[0].to(device)  # Assuming the input data is in the first index of the batch tuple

                # Generate some sample images
                with torch.no_grad():
                    sample_images = generator(input_data)

                # Save the sample images
                save_image(sample_images, f"{resultsPath}/generated_images_epoch_{epoch}_step_{i}.png", nrow=8, normalize=True)

                # Only process one batch of data, then break out of the loop
                break

            # Set the generator back to training mode
            generator.train()
            
    # Save generated images
    save_image(fake_images, f"{resultsPath}/generated_images_epoch_{epoch + 1}.png", nrow=8, normalize=True)


torch.save(generator, "C:/WorkingSets/Model/savedmodel.pt")
