from numpy import load
from matplotlib import pyplot
import numpy as np
import GAN.config as config

# load the dataset
data = load(config.TRAIN_DIR)
src_images, tar_images = data["arr_0"], data["arr_1"]

# Print the shapes of the loaded datasets
print("Loaded: ", src_images.shape, tar_images.shape)

# Set the number of samples to display
n_samples = 3

# Plot source images
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + i)
    pyplot.axis("off")
    # Ensure the image is in the correct format (H, W, C)
    image = src_images[i]
    if image.shape[0] == 3 and image.shape[1] == 256 and image.shape[2] == 256:
        # Transpose to (256, 256, 3) if needed
        image = np.transpose(image, (1, 2, 0))
    pyplot.imshow(image.astype("uint8"))

# Plot target images
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + n_samples + i)
    pyplot.axis("off")
    # Ensure the image is in the correct format (H, W, C)
    image = tar_images[i]
    if image.shape[0] == 3 and image.shape[1] == 256 and image.shape[2] == 256:
        # Transpose to (256, 256, 3) if needed
        image = np.transpose(image, (1, 2, 0))
    pyplot.imshow(image.astype("uint8"))

# Display the plots
pyplot.show()
