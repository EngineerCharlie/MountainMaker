from numpy import load
from matplotlib import pyplot
import numpy as np

# load the dataset
path = ".venv/testDataUnfiltered/"
data = load(path + "mountains_256.npz")
src_images, tar_images = data["arr_0"], data["arr_1"]
in_reshaped = np.moveaxis(src_images, 3, 1)
tar_reshaped = np.moveaxis(tar_images, 3, 1)
print("Loaded: ", src_images.shape, tar_images.shape)
# plot source images
n_samples = 3
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + i)
    pyplot.axis("off")
    pyplot.imshow(in_reshaped[i].astype("uint8"))
# plot target image
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + n_samples + i)
    pyplot.axis("off")
    pyplot.imshow(tar_reshaped[i].astype("uint8"))
pyplot.show()