import torch
import numpy as np
from GAN.utils import load_checkpoint
import torch.optim as optim
import GAN.config as config
from GAN.Generator import Generator
from matplotlib import pyplot as plt
from PIL import Image
import itertools


def generate_mountain_from_file(
    filepath="my_mountain.png", save_images=False, model_filename=None
):
    if not model_filename:
        model_filename = config.EVALUATION_GEN + ".tar"

    torch.backends.cudnn.benchmark = True
    gen_G = Generator(in_channels=3, features=64).to(config.DEVICE)
    gen_G.eval()
    gen_F = Generator(in_channels=3, features=64).to(config.DEVICE)
    gen_F.eval()
    opt_gen = optim.Adam(itertools.chain(gen_G.parameters(), gen_F.parameters()),
        lr=config.LEARNING_RATE_GEN,
        betas=(config.GEN_BETA1, config.GEN_BETA2),
    )
    load_checkpoint(
        model_filename,
        gen_G,
        opt_gen,
        config.LEARNING_RATE_GEN,
    )

    def load_single_image(filename, target_size=(256, 256)):
        # Load image
        img = Image.open(filename)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        # Resize if necessary
        if img.size != target_size:
            img = img.resize(target_size, Image.ANTIALIAS)
        # Convert image to numpy array
        img_array = np.array(img)
        # Normalize pixel values to range [-1, 1]
        img_array = (img_array - 127.5) / 127.5
        # Convert numpy array to PyTorch tensor
        img_tensor = (
            torch.tensor(img_array.transpose(2, 0, 1), dtype=torch.float32)
            .unsqueeze(0)
            .to(config.DEVICE)
        )
        return img_tensor

    image = load_single_image(filepath)
    output_image = gen_G(image)

    # Convert output tensor to numpy array and reshape if necessary
    output_image_np = output_image.squeeze(0).detach().cpu().numpy()
    output_image_np = (output_image_np * 0.5 + 0.5) * 255  # De-normalize pixel values
    output_image_np = output_image_np.transpose(1, 2, 0).astype(np.uint8)

    if save_images:
        # TODO:
        output_image_pil = Image.fromarray(output_image_np)
        output_image_pil.save("generated_image.jpg")
    # Display input and output images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.array(Image.open(filepath)))  # Display input image
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    axes[1].imshow(output_image_np)  # Display output image
    axes[1].set_title("Generated Image")
    axes[1].axis("off")
    plt.show()

def load_single_image(filename, target_size=(256, 256)):
        # Load image
        img = Image.open(filename)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        # Resize if necessary
        if img.size != target_size:
            img = img.resize(target_size, Image.ANTIALIAS)
        # Convert image to numpy array
        img_array = np.array(img)
        # Normalize pixel values to range [-1, 1]
        img_array = (img_array - 127.5) / 127.5
        # Convert numpy array to PyTorch tensor
        img_tensor = (
            torch.tensor(img_array.transpose(2, 0, 1), dtype=torch.float32)
            .unsqueeze(0)
            .to(config.DEVICE)
        )
        return img_tensor

def gen_mount_img(
        filepath="my_mountain.png",  model_filename=None
        ):
    
    
    if not model_filename:
        model_filename = config.EVALUATION_GEN + ".tar"

    torch.backends.cudnn.benchmark = True
    gen_G = Generator(in_channels=3, features=64).to(config.DEVICE)
    gen_G.eval()
    gen_F = Generator(in_channels=3, features=64).to(config.DEVICE)
    gen_F.eval()
    opt_gen = optim.Adam(itertools.chain(gen_G.parameters(), gen_F.parameters()),
        lr=config.LEARNING_RATE_GEN,
        betas=(config.GEN_BETA1, config.GEN_BETA2),
    )
    load_checkpoint(
        model_filename,
        gen,
        opt_gen,
        config.LEARNING_RATE_GEN,
    )

   

    image = load_single_image(filepath)
    output_image = gen(image)

    # Convert output tensor to numpy array and reshape if necessary
    output_image_np = output_image.squeeze(0).detach().cpu().numpy()
    output_image_np = (output_image_np * 0.5 + 0.5) * 255  # De-normalize pixel values
    output_image_np = output_image_np.transpose(1, 2, 0).astype(np.uint8)
    return output_image_np


def plot_images(images,labels):
    """
    Plot a list of images with labels 'Original', 'epoch 15', 'epoch 30', ..., 'epoch n'.
    
    Parameters:
    images (list of np.ndarray): List of images to be plotted. The first image is the original,
                                 followed by images at intervals of 15 epochs.
    """
    num_images = len(images)
    if num_images < 1:
        raise ValueError("The list of images must contain at least one image.")
    
    # Ensure all elements are valid numpy arrays with correct dimensions
    for idx, img in enumerate(images):
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Image at index {idx} is not a numpy array.")
        if img.ndim not in [2, 3]:
            raise ValueError(f"Image at index {idx} has invalid shape {img.shape}. Expected a 2D or 3D array.")
        print(f"Image {idx} shape: {img.shape}")
    
   
    
    # Create a figure
    plt.figure(figsize=(15, 5))
    
    for idx, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(1, num_images, idx + 1)
        plt.imshow(image, cmap='gray' if image.ndim == 2 else None)
        plt.title(label)
        plt.axis('off')  # Hide the axis
    
    plt.tight_layout()
    plt.show()



def show_model_evo(filepath,BaseModelPath,start,tar):
    
    input = np.array(Image.open(filepath))
    
    outputs = [input]

    labels = ['Original']

    original = BaseModelPath +f"{start}.tar"
    out = gen_mount_img(filepath,original)
    outputs.append(out)


    for i in range(start,tar+1,15):
        try:
            path = BaseModelPath +f"{i}.tar"
            out = gen_mount_img(filepath,path)
            labels +=   [f"epoch {i}"]
            
        except:
           #if for some reason there was in issue in saving the checkpoint for that epoch it will just use the previous one 
           path = BaseModelPath+f"{i-15}.tar"
           labels +=  [f"epoch {i-15}"]
           pass

        
        out = gen_mount_img(filepath,path)
        outputs.append(out)
    


    plot_images(outputs,labels)

    
    








    

