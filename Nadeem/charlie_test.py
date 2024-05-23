import config
from Nadeem.GenerateMountain import generate_mountain_from_file

filename = "GAN/mount_sketch_diego.jpg"
model_filename = config.EVALUATION_GEN + ".tar"
generate_mountain_from_file(filename, save_images=True, model_filename=model_filename)
