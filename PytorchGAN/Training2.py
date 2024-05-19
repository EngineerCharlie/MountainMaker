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

device = "cuda" if torch.cuda.is_available() else "cpu"
print("--------------------")
print("Training started with device: ", device)
print("CUDA version:", torch.version.cuda)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print("--------------------")
#print("CUDA_PATH: {}".format(os.environ["CUDA_PATH"]))
#print("CUDA_HOME: {}".format(os.environ["CUDA_HOME"]))

# Initialize the generator and discriminator
generator = Generator(input_channels=3, output_channels=3).to(device=device)
discriminator = Discriminator(input_channels=6).to(device=device)

# Define the optimizers for generator and discriminator
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=.5)


num_epochs = 200
for epoch in range(1, num_epochs+1): 
 
    D_loss_list, G_loss_list = [], []
    
    for index, (real_images, labels) in enumerate(train_loader):
        D_optimizer.zero_grad()
        real_images = real_images.to(device)
        labels = labels.to(device)
        labels = labels.unsqueeze(1).long()
 
       
        real_target = Variable(torch.ones(real_images.size(0), 1).to(device))
        fake_target = Variable(torch.zeros(real_images.size(0), 1).to(device))
       
        D_real_loss = discriminator_loss(discriminator((real_images, labels)), real_target)
        # print(discriminator(real_images))
        #D_real_loss.backward()
     
        noise_vector = torch.randn(real_images.size(0), latent_dim, device=device)  
        noise_vector = noise_vector.to(device)
         
        
        generated_image = generator((noise_vector, labels))
        output = discriminator((generated_image.detach(), labels))
        D_fake_loss = discriminator_loss(output,  fake_target)
 
     
        # train with fake
        #D_fake_loss.backward()
       
        D_total_loss = (D_real_loss + D_fake_loss) / 2
        D_loss_list.append(D_total_loss)
       
        D_total_loss.backward()
        D_optimizer.step()
 
        # Train generator with real labels
        G_optimizer.zero_grad()
        G_loss = generator_loss(discriminator((generated_image, labels)), real_target)
        G_loss_list.append(G_loss)
 
        G_loss.backward()
        G_optimizer.step()