import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from GAN.utils import save_checkpoint, load_checkpoint, save_some_examples
import GAN.config as config
from GAN.Generator import Generator
from GAN.Discriminator import Discriminator
from torch.utils.data import DataLoader
from numpy import load
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast,GradScaler


def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    source, target = data["arr_0"], data["arr_1"]
    # Normalize scale from [0,255] to [-1,1]
    source = (source - 127.5) / 127.5
    target = (target - 127.5) / 127.5
    return source, target



    
def train_cycle_gan(
            generator_G, 
            optimizer_G,
            generator_scaler,
            generator_F,
            discriminator_A,
            discriminator_scaler_A,
            optimizer_D_A,
            discriminator_B,
            discriminator_scaler_B,
            optimizer_D_B,
            dataloader,
            criterion_GAN,
            criterion_cycle,
            criterion_identity,
            device,
            ):
    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

   
    loop = tqdm(dataloader, leave=True)

    

    # Training Loop
    for i, (real_A, real_B) in enumerate(loop):
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        
        # -----------------------
        #  Train Discriminator A
        # -----------------------
        optimizer_D_A.zero_grad()
        with torch.no_grad():
            fake_B = generator_G(real_A)
            fake_A = generator_F(real_B)

        with autocast():
            
            pred_real_A = discriminator_A(real_A, real_B)
            loss_D_real_A = criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))

            
            

            pred_fake_A = discriminator_A(fake_A.detach(), real_A)
            loss_D_fake_A = criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))

            loss_D_A = (loss_D_real_A + loss_D_fake_A)/2

        
        discriminator_scaler_A.scale(loss_D_A).backward()
        discriminator_scaler_A.step(optimizer_D_A)
        discriminator_scaler_A.update()

        
        # -----------------------
        #  Train Discriminator B
        # -----------------------
        optimizer_D_B.zero_grad()

        with autocast():
            pred_real_B = discriminator_B(real_B, real_A)
            loss_D_real_B = criterion_GAN(pred_real_B, torch.ones_like(pred_real_B)) 

            pred_fake_B = discriminator_B(fake_B.detach(), real_B)
            loss_D_fake_B = criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))

            loss_D_B = (loss_D_real_B + loss_D_fake_B)/2

        discriminator_scaler_B.scale(loss_D_B).backward()
        discriminator_scaler_B.step(optimizer_D_B)
        discriminator_scaler_B.update()

        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()

        with autocast():
            # Identity loss
            loss_id_A = criterion_identity(generator_F(real_A), real_A) * 10.0
            loss_id_B = criterion_identity(generator_G(real_B), real_B) * 10.0

            # GAN loss
            
            pred_fake_B = discriminator_B(fake_B, real_B)
            loss_GAN_A2B = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))

            
            pred_fake_A = discriminator_A(fake_A, real_A)
            loss_GAN_B2A = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

            # Cycle-consistency loss
            recovered_A = generator_F(fake_B)
            loss_cycle_A = criterion_cycle(recovered_A, real_A) *config.L1_LAMBDA

            recovered_B = generator_G(fake_A)
            loss_cycle_B = criterion_cycle(recovered_B, real_B) * config.L1_LAMBDA

            # Total generator loss
            loss_G = (loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B)

        generator_scaler.scale(loss_G).backward()
        generator_scaler.step(optimizer_G)
        generator_scaler.update()

        if i % 10 == 0:
            # Update cmd line outputs based on current accuracy values
            loop.set_postfix(
                D_real=torch.sigmoid(pred_real_A).mean().item(),
                D_fake=torch.sigmoid(pred_fake_A).mean().item(),
            )
    print("G loss = " + str(loss_G.mean().item()))


    




def run():
    device = config.DEVICE
    
    generator_G = Generator(in_channels=3, features=64).to(device)
    generator_F = Generator(in_channels=3, features=64).to(device)
    discriminator_A = Discriminator(in_channels=3).to(device)
    discriminator_B = Discriminator(in_channels=3).to(device)
     # Loads data from compressed numpy array and returns the source and target images

    optimizer_G = optim.Adam(itertools.chain(generator_G.parameters(), generator_F.parameters()), config.LEARNING_RATE_GEN, betas=(config.GEN_BETA1, config.GEN_BETA2))
    optimizer_D_A = optim.Adam(discriminator_A.parameters(), config.LEARNING_RATE_DISC, betas=(config.DISC_BETA1, config.DISC_BETA2))
    optimizer_D_B = optim.Adam(discriminator_B.parameters(), config.LEARNING_RATE_DISC, betas=(config.DISC_BETA1, config.DISC_BETA2))
    # Defines the pytorch scaler function that is used to back propogate values
    # through the models. Due to limited hardware and a fairly complex model, we're
    # using torch's automatic mixed precision to try and reduce training times.
    generator_scaler = torch.cuda.amp.GradScaler()
    discriminator_scaler = torch.cuda.amp.GradScaler()
    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = config.L1_LAMBDA
    if config.LOAD_MODEL:
            load_checkpoint(
                generator_G,
                optimizer_G,
                filename=config.LOAD_GEN + "G.tar",
            )
            load_checkpoint(
                discriminator_A,
                optimizer_D_A,
                filename=config.LOAD_DISC + "G.tar",
            )
            load_checkpoint(
                generator_F,
                optimizer_G,
                filename=config.LOAD_GEN + "F.tar",
            )
            load_checkpoint(
                discriminator_B,
                optimizer_D_B,
                filename=config.LOAD_DISC + "F.tar",
            )

    input_image, target_image = load_real_samples(config.TRAIN_DIR)

    in_reshaped = np.moveaxis(input_image, 3, 1)
    tar_reshaped = np.moveaxis(target_image, 3, 1)
    train_dataset = list(zip(in_reshaped, tar_reshaped))
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    # Defines the pytorch scaler function that is used to back propogate values
    # through the models. Due to limited hardware and a fairly complex model, we're
    # using torch's automatic mixed precision to try and reduce training times.
    generator_scaler = GradScaler()
    discriminator_scaler_A = GradScaler()
    discriminator_scaler_B = GradScaler()
    # Loads data from compressed numpy array and returns the source and target images
    input_image, target_image = load_real_samples(config.VAL_DIR)

    in_reshaped = np.moveaxis(input_image, 3, 1)
    tar_reshaped = np.moveaxis(target_image, 3, 1)

    val_dataset = list(zip(in_reshaped, tar_reshaped))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS-config.EPOCH_OFFSET):
        epoch = epoch+config.EPOCH_OFFSET
        if epoch % 5 == 0:
            print(f"Epoch number {epoch}")
        train_cycle_gan(
            generator_G, 
            optimizer_G,
            generator_scaler,
            generator_F,
            discriminator_A,
            discriminator_scaler_A,
            optimizer_D_A,
            discriminator_B,
            discriminator_scaler_B,
            optimizer_D_B,
            dataloader,
            criterion_GAN,
            criterion_cycle,
            criterion_identity,
            device,
            )
        save_interval = config.SAVE_INTERVAL
        if config.SAVE_MODEL and epoch % save_interval == 0 and epoch > config.EPOCH_OFFSET:
            save_checkpoint(
                generator_G,
                optimizer_G,
                filename=config.CHECKPOINT_GEN + str(epoch) + "G.tar",
            )
            save_checkpoint(
                discriminator_A,
                optimizer_D_A,
                filename=config.CHECKPOINT_DISC + str(epoch) + "G.tar",
            )
            save_checkpoint(
                generator_F,
                optimizer_G,
                filename=config.CHECKPOINT_GEN + str(epoch) + "F.tar",
            )
            save_checkpoint(
                discriminator_B,
                optimizer_D_B,
                filename=config.CHECKPOINT_DISC + str(epoch) + "F.tar",
            )

        save_some_examples(
            generator_G,
            val_loader,
            epoch,
            folder=config.SAMPLE_FOLDER,
        )

    
