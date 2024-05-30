import torch
import numpy as np
from GAN.utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import GAN.config as config
from GAN.Generator import Generator
from GAN.Discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from numpy import load

torch.backends.cudnn.benchmark = True


def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    source, target = data["arr_0"], data["arr_1"]
    # Normalize scale from [0,255] to [-1,1]
    source = (source - 127.5) / 127.5
    target = (target - 127.5) / 127.5
    return source, target


def train_fn(
    discriminator,
    generator,
    loader,
    optimizer_discriminator,
    optimizer_generator,
    l1_loss,
    bce,
    generator_scaler,
    discriminator_scaler,
):
    # Load data and display progress bar as each piece of data is loaded
    loop = tqdm(loader, leave=True)

    for idx, (input_images, targets_real) in enumerate(loop):
        # Load target and source images to optimal device
        input_images = input_images.to(config.DEVICE)
        targets_real = targets_real.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            # Use generator to  generate fake images
            targets_fake = generator(input_images)
            # Get discriminator real/fake predictions from real source and targets
            # Output is array of [batch size, 1, 30, 30]
            real_predictions = discriminator(input_images, targets_real)
            # Use discriminator output to produce a scalar that is used to feed back into model
            real_losses = bce(real_predictions, torch.ones_like(real_predictions))
            # Get discriminator real/fake predictions from real source and fake targets
            fake_predictions = discriminator(input_images, targets_fake.detach())
            # Use discriminator output for single loss scalar
            fake_losses = bce(fake_predictions, torch.zeros_like(fake_predictions))
            # Compute average accuracy of discriminator of both real and fake over the current batch
            overall_discriminator_loss = (real_losses + fake_losses) / 2

        # Zero the gradients before we calculate them for back propogation
        discriminator.zero_grad()
        # Scales the losses stop gradients prematurely becoming 0, then computes gradients
        discriminator_scaler.scale(overall_discriminator_loss).backward()
        # Unscales the gradients back and updates the optimizer parameters
        discriminator_scaler.step(optimizer_discriminator)
        # Adjusts the scaling factor to allow for amp
        discriminator_scaler.update()

        # Using updated discriminator, recompute real/fake predictions
        with torch.cuda.amp.autocast():
            # Computes predictions for fake input images
            fake_predictions = discriminator(input_images, targets_fake)
            # Computes losses of predictions
            generator_fake_losses = bce(
                fake_predictions, torch.ones_like(fake_predictions)
            )
            # Computes the mean absolute error (or difference) between the fake and real outputs
            # If a generated image closely resembles the original, then the L1_loss will be low
            L1 = l1_loss(targets_fake, targets_real) * config.L1_LAMBDA
            # By summing these two values, the  generator losses are highest when it both:
            # A) - doesn't fool the discriminator
            # B) - doesn't resemble the original output image
            generator_overall_losses = generator_fake_losses + L1

        optimizer_generator.zero_grad()
        generator_scaler.scale(generator_overall_losses).backward()
        generator_scaler.step(optimizer_generator)
        generator_scaler.update()

        if idx % 10 == 0:
            # Update cmd line outputs based on current accuracy values
            loop.set_postfix(
                D_real=torch.sigmoid(real_predictions).mean().item(),
                D_fake=torch.sigmoid(fake_predictions).mean().item(),
            )
    print("G loss = " + str(generator_overall_losses.mean().item()))


def run():
    discriminator = Discriminator(in_channels=3).to(config.DEVICE)
    generator = Generator(in_channels=3, features=64).to(config.DEVICE)

    optimizer_disc = optim.Adam(
        discriminator.parameters(),
        lr=config.LEARNING_RATE_DISC,
        betas=(config.DISC_BETA1, config.DISC_BETA2),
    )
    optimizer_gen = optim.Adam(
        generator.parameters(),
        lr=config.LEARNING_RATE_GEN,
        betas=(config.GEN_BETA1, config.GEN_BETA2),
    )
    # Combines sigmoid activation fnc and cross entropy loss fnc into a single function
    BCE = nn.BCEWithLogitsLoss()
    # Uses "per pixel" losses to compare real and generated output images, returning
    # a single scalar as  the average of all pixel losses
    L1_LOSS = nn.L1Loss()

    # Loads in pre-trained modeel
    if config.LOAD_MODEL:
        load_checkpoint(
            config.LOAD_GEN + ".tar",
            generator,
            optimizer_gen,
            config.LEARNING_RATE_GEN,
        )
        load_checkpoint(
            config.LOAD_DISC + ".tar",
            discriminator,
            optimizer_disc,
            config.LEARNING_RATE_DISC,
        )

    # Loads data from compressed numpy array and returns the source and target images
    input_image, target_image = load_real_samples(config.TRAIN_DIR)

    in_reshaped = np.moveaxis(input_image, 3, 1)
    tar_reshaped = np.moveaxis(target_image, 3, 1)
    train_dataset = list(zip(in_reshaped, tar_reshaped))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    # Defines the pytorch scaler function that is used to back propogate values
    # through the models. Due to limited hardware and a fairly complex model, we're
    # using torch's automatic mixed precision to try and reduce training times.
    generator_scaler = torch.cuda.amp.GradScaler()
    discriminator_scaler = torch.cuda.amp.GradScaler()
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
        train_fn(
            discriminator,
            generator,
            train_loader,
            optimizer_disc,
            optimizer_gen,
            L1_LOSS,
            BCE,
            generator_scaler,
            discriminator_scaler,
        )
        save_interval = config.SAVE_INTERVAL
        if config.SAVE_MODEL and epoch % save_interval == 0 and epoch > config.EPOCH_OFFSET:
            save_checkpoint(
                generator,
                optimizer_gen,
                #filename=config.CHECKPOINT_GEN + str(epoch) + ".tar"
                filename=config.CHECKPOINT_GEN + str(epoch) + "blur.tar",
            )
            save_checkpoint(
                discriminator,
                optimizer_disc,
                #filename=config.CHECKPOINT_DISC + str(epoch) + ".tar"
                filename=config.CHECKPOINT_DISC + str(epoch) + "blur.tar",
            )

        save_some_examples(
            generator,
            val_loader,
            epoch,
            folder=config.SAMPLE_FOLDER,
        )
