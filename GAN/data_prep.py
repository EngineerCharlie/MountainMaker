# load, split and scale the maps dataset ready for training
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed


# load all images in a directory into memory
def load_images(path_original, path_processed):
    src_list, tar_list = list(), list()
    num_images = len(listdir(path_original))
    # enumerate filenames in directory, assume all are images
    for i in range(num_images):
        try:
            original_filename = f"Mountain-{str(i)}.jpg"
            processed_filename = f"Mountain_processed-{str(i)}.jpg"
            # load and resize the image
            original_pixels = load_img(
                path_original + original_filename, target_size=(256, 256)
            )
            # convert to numpy array
            original_pixels = img_to_array(original_pixels)
            # load and resize the image
            processed_pixels = load_img(
                path_processed + processed_filename, target_size=(256, 256)
            )
            # convert to numpy array
            processed_pixels = img_to_array(processed_pixels)
            # split into satellite and map
            src_list.append(processed_pixels)
            tar_list.append(original_pixels)
        except:
            print(f"error on iteration {i}")
    return [asarray(src_list), asarray(tar_list)]


# dataset path
path_original = "testDataUnfiltered/scaled/"
# path_processed = "testDataUnfiltered/scaled_processed_downsized/"
path_processed = "testDataUnfiltered/scaled_processed/"

# load dataset
[src_images, tar_images] = load_images(path_original, path_processed)
print("Loaded: ", src_images.shape, tar_images.shape)
# save as compressed numpy array
filename = "testDataUnfiltered/mountains_256.npz"
savez_compressed(filename, src_images, tar_images)
print("Saved dataset: ", filename)
