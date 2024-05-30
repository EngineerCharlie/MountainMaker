import GAN.config as config
from GAN.GenerateMountain import show_model_evo



filename = "Gan/mount_sketch_diego.jpg"
model_filename = config.CHECKPOINT_GEN 
#Note do not start at zero also try keep the number of images shown <= 13 for readablitity 
show_model_evo(filename,model_filename,105,255)