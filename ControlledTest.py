import GAN.config as config
from GAN.GenerateMountain import generate_mountain_from_file
from GAN.GenerateMountain import gen_mount_img

filename = "test_im_1.jpg"
#model_filename = config.EVALUATION_GEN + '.tar'
model_filename = "C:/Users/nadee/Documents/Mountains/gen.pth60G" + '.tar'
generate_mountain_from_file(filename, save_images=True, model_filename=model_filename)
# refiner = "C:/Users/nadee/Documents/Mountains/gen.pth105blur.tar"
# filename = "generated_image.jpg"
# generate_mountain_from_file(filename, save_images=True, model_filename=refiner)
