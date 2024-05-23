import torch
import numpy as np
from utils import load_checkpoint
import torch.optim as optim
import config
from Generator import Generator
from matplotlib import pyplot as plt
from PIL import Image

filename = "GAN/mount_sketch_diego.jpg"
model_filename = config.EVALUATION_GEN + "135.tar"

torch.backends.cudnn.benchmark = True
gen = Generator(in_channels=3, features=64).to(config.DEVICE)
gen.eval()
opt_gen = optim.Adam(
    gen.parameters(),
    lr=config.LEARNING_RATE_GEN,
    betas=(config.GEN_BETA1, config.GEN_BETA2),
)
load_checkpoint(
    model_filename,
    gen,
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


image = load_single_image(filename)
output_image = gen(image)

# Convert output tensor to numpy array and reshape if necessary
output_image_np = output_image.squeeze(0).detach().cpu().numpy()
output_image_np = (output_image_np * 0.5 + 0.5) * 255  # De-normalize pixel values
output_image_np = output_image_np.transpose(1, 2, 0).astype(np.uint8)

# Display input and output images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(np.array(Image.open(filename)))  # Display input image
axes[0].set_title("Input Image")
axes[0].axis("off")
axes[1].imshow(output_image_np)  # Display output image
axes[1].set_title("Generated Image")
axes[1].axis("off")
plt.show()
