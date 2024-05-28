# load, split and scale the maps dataset ready for training
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed

# dataset path
path_original_images = "testDataUnfiltered/valid/"
# path_processed = "testDataUnfiltered/scaled_processed_downsized/"
path_processed_images = "testDataUnfiltered/caravaggio/traced-images/"
filename = "testDataUnfiltered/mountains_Diego.npz"

# load all images in a directory into memory
def load_images(path_original, path_processed):
    src_list, tar_list = list(), list()
    num_images = len(listdir(path_original))
    # enumerate filenames in directory, assume all are images
    for i in range(num_images):
        try:
            original_filename = f"Mountain-{str(i)}.jpg"
            processed_filename = f"traced-Mountain-{str(i)}.jpg"
            # load and resize the image
            original_image = load_img(path_original + original_filename, target_size=(256,256))
            # convert to numpy array
            original_image = img_to_array(original_image)
            # load and resize the image
            processed_image = load_img(
                path_processed + processed_filename, target_size=(256, 256)
            )
            # convert to numpy array
            processed_image = img_to_array(processed_image)
            # split into drawified and photo
            src_list.append(processed_image)
            tar_list.append(original_image)
        except:
            print(f"error on iteration {i}")
    return [asarray(src_list), asarray(tar_list)]

# load dataset
[src_images, tar_images] = load_images(path_original_images, path_processed_images)
print("Loaded: ", src_images.shape, tar_images.shape)
# save as compressed numpy array

savez_compressed(filename, src_images, tar_images)
print("Saved dataset: ", filename)
