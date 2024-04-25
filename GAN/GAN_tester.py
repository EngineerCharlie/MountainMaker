import GAN1
OUTPUT_PATH = "output/"
model = "model_016240.h5"
image_path = "GAN/have_a_go_image.jpg"
GAN1.gen_mountains(OUTPUT_PATH + model, image_path)
