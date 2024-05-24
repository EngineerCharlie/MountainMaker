import numpy as np
from numpy import savez_compressed
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as pyplot
from torchvision import transforms
from data_prep import load_images


# class MapDataset(Dataset):
#     def __init__(self, root_dir):
#         self.root_dir = root_dir
#         self.list_files = os.listdir(self.root_dir)

#     def __len__(self):
#         return len(self.list_files)

#     def __getitem__(self, index):
#         img_file = self.list_files[index]
#         img_path = os.path.join(self.root_dir, img_file)
#         image = np.array(Image.open(img_path))
#         input_image = image[:, :600, :]
#         target_image = image[:, 600:, :]

#         augmentations = config.both_transform(image=input_image, image0=target_image)
#         input_image = augmentations["image"]
#         target_image = augmentations["image0"]

#         input_image = config.transform_only_input(image=input_image)["image"]
#         target_image = config.transform_only_mask(image=target_image)["image"]

#         return input_image, target_image


if __name__ == "__main__":
   # Load data for validation


    # dataset path
    val_data = np.load(".venv/valData/val_256.npz")
    src_images, tar_images = val_data["arr_0"], val_data["arr_1"]
    print("Loaded: ", src_images.shape, tar_images.shape)




    data = np.load(".venv/testDataUnfiltered/mountains_256.npz")
    


    input_image = data["arr_0"]
    target_image = data["arr_1"]



    src_images, tar_images = data["arr_0"], data["arr_1"]
    print("Loaded: ", src_images.shape, tar_images.shape)
# plot source images
    n_samples = 3
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + i)
        pyplot.axis("off")
        pyplot.imshow(src_images[i].astype("uint8"))
# plot target image
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + n_samples + i)
        pyplot.axis("off")
        pyplot.imshow(tar_images[i].astype("uint8"))
    pyplot.show()
