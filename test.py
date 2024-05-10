import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


import torch


print("TensorFlow Version:", tf.__version__)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device = "GPU" if torch.cuda.is_available() else "CPU"
print(device)