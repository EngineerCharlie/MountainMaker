import tensorflow as tf
import tensorflow_hub as hub
from io import BytesIO
from PIL import Image as PilImage
from PIL import ImageOps
import numpy as np
from matplotlib import pyplot as plt
#from six.moves.urllib.request import urlopen
from urllib.request import urlopen

def read_image(filename):
    fd = None
    if(filename.startswith('http')):
      fd = urlopen(filename)
    else:
      fd = tf.io.gfile.GFile(filename, 'rb')

    pil_image = PilImage.open(fd)
    width, height = pil_image.size
    # crop to make the image square
    pil_image = pil_image.crop((0, 0, height, height))
    pil_image = pil_image.resize((257,257),PilImage.LANCZOS)
    image_unscaled = np.array(pil_image)
    image_np = np.expand_dims(
        image_unscaled.astype(np.float32) / 255., axis=0)
    return image_np


def visualize_output_comparison(img_original, img_masked, img_filled):
  plt.figure(figsize=(24,12))
  plt.subplot(131)
  plt.imshow((np.squeeze(img_original)))
  plt.title("Original", fontsize=24)
  plt.axis('off')
  plt.subplot(132)
  plt.imshow((np.squeeze(img_masked)))
  plt.title("Masked", fontsize=24)
  plt.axis('off')
  plt.subplot(133)
  plt.imshow((np.squeeze(img_filled)))
  plt.title("Generated", fontsize=24)
  plt.axis('off')
  plt.show()

from PIL import Image

def shift_image(image, direction):
    width, height = image.size
    
    # Calculate the size of the left and right parts
    left_width = width // 2
    
    # Depending on the direction, determine the crop and paste positions
    if direction == "left":
        # Crop the right half of the original image
        cropped_part = image.crop((left_width, 0, width, height))
        
        # Create a new black image with the same size as the original image
        shifted_image = Image.new("RGB", (width, height), color="black")
        
        # Paste the right half onto the left side of the new image
        shifted_image.paste(cropped_part, (0, 0))
    elif direction == "right":
        # Crop the left half of the original image
        cropped_part = image.crop((0, 0, left_width, height))
        
        # Create a new black image with the same size as the original image
        shifted_image = Image.new("RGB", (width, height), color="black")
        
        # Paste the left half onto the right side of the new image
        shifted_image.paste(cropped_part, (left_width, 0))
    else:
        raise ValueError("Invalid direction. Please specify 'left' or 'right'.")
    
    return shifted_image

def Prepare_Img(shifted_pil_image):
   
    #image_unscaled = np.array(shifted_pil_image)
    shifted_pil_image = shifted_pil_image.resize((257,257),PilImage.LANCZOS)

    image_unscaled = np.array(shifted_pil_image)
    input_img = np.expand_dims(image_unscaled.astype(np.float32) / 255., axis=0)
    
    return input_img


def InferModel(model, input_img):

    result = model.signatures['default'](tf.constant(input_img))
    generated_image =  result['default']
    masked_image = result['masked_image']
    
    #visualize_output_comparison(input_img, masked_image, generated_image)

    return generated_image

def CreateInferation(model, pil_image_rgb, num):
    orig_array = np.asarray(pil_image_rgb)
    orig_array = orig_array / np.max(orig_array)
    separateIndex = orig_array.shape[1]//2
    
    final = orig_array[:, :separateIndex]

    previus_gen = orig_array
    for i in range(num):
        second_input = np.zeros_like(previus_gen)
        second_input[:, :128] = previus_gen[:, 128:256]
        second_input = np.expand_dims(second_input.astype(np.float32), axis=0)

        gen2 = InferModel(model, second_input)
        gen2 = np.squeeze(gen2)
        final = np.concatenate((final, gen2[:, :separateIndex]), axis=1)

        previus_gen = gen2

    return final

def CreateInferModelInferation(fd, width_factor=3):
    model_name = 'Boundless Half' # @param ['Boundless Half', 'Boundless Quarter', 'Boundless Three Quarters']
    model_handle_map = {
        'Boundless Half' : 'https://tfhub.dev/google/boundless/half/1',
        'Boundless Quarter' : 'https://tfhub.dev/google/boundless/quarter/1', 
        'Boundless Three Quarters' : 'https://tfhub.dev/google/boundless/three_quarter/1'
    }

    model_handle = model_handle_map[model_name]
    print("Loading model {} ({})".format(model_name, model_handle))
    model = hub.load(model_handle)


    pil_image = PilImage.open(fd)
    pil_image_rgb = pil_image.convert('RGB')
    pil_image_rgb = pil_image_rgb.resize((257,257),PilImage.LANCZOS)

    
    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(np.asarray(pil_image_rgb))
    axs[0].axis('off')
    #current_position = axs[0].get_position()
    #axs[0].set_position([current_position.x0 - 0.3, current_position.y0, current_position.width, current_position.height])

    final_right = CreateInferation(model, pil_image_rgb, width_factor)
    final_right = np.asarray(final_right)


    image_mirrored = ImageOps.mirror(pil_image_rgb)
    final_left = CreateInferation(model, image_mirrored, width_factor)
    final_left = np.flip(final_left, axis=1)
    final_left = np.asarray(final_left)


    final = np.full((257, 256*width_factor-1, 3), 0, dtype=np.float32)
    #print("final shape: ", final.shape, " ", final_left.shape, " ", final_right.shape)
    final[:, :final_left.shape[1], :] = final_left
    final[:, final_left.shape[1]:] = final_right[:, 257:final.shape[1]]

    axs[1].imshow(final)
    axs[1].axis('off')

    plt.show()

    return final

if __name__ == '__main__':

    CreateInferModelInferation(width_factor=3)

    #visualize_output_comparison(pil_image, shifted_pil_image, shifted_pil_image)

    #print("shape: ", input_img.shape)
    #print(input_img)

    #wikimedia = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Nusfjord_road%2C_2010_09.jpg/800px-Nusfjord_road%2C_2010_09.jpg"
    # wikimedia = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Beech_forest_M%C3%A1tra_in_winter.jpg/640px-Beech_forest_M%C3%A1tra_in_winter.jpg"
    # wikimedia = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Marmolada_Sunset.jpg/640px-Marmolada_Sunset.jpg"
    # wikimedia = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Aegina_sunset.jpg/640px-Aegina_sunset.jpg"

    #input_img = read_image(wikimedia)


   


