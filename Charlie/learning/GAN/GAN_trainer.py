import GAN1

# load image data
# map data can be downlaoded here:
# https://www.kaggle.com/datasets/alincijov/pix2pix-maps?resource=download-directory
path = "Charlie/learning/GAN/maps/"
dataset = GAN1.load_real_samples(path + "maps_256.npz")
print("Loaded", dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# define the models
d_model = GAN1.define_discriminator(image_shape)
print("defined discriminator")
g_model = GAN1.define_generator(image_shape)
print("defined generator")
# define the composite model
gan_model = GAN1.define_gan(g_model, d_model, image_shape)
print("defined gan, about to start training ")
# train model
GAN1.train(d_model, g_model, gan_model, dataset)
