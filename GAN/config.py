import torch
import GAN.config_local as config_local

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
LEARNING_RATE_DISC = 2e-4
LEARNING_RATE_GEN = 2e-4
BATCH_SIZE = 16
L1_LAMBDA = 100
LAMBDA_GP = 12
DISC_BETA1 = 0.49
DISC_BETA2 = 0.999
GEN_BETA1 = 0.49
GEN_BETA2 = 0.999
NUM_EPOCHS = 301
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
SAVE_INTERVAL = 30
LOAD_MODEL = False 
EPOCH_OFFSET = 255 if LOAD_MODEL else 0
SAVE_MODEL = True
CHECKPOINT_DISC = config_local.CHECKPOINT_DISC
CHECKPOINT_GEN = config_local.CHECKPOINT_GEN
LOAD_DISC = config_local.LOAD_DISC
LOAD_GEN = config_local.LOAD_GEN
EVALUATION_GEN = config_local.EVALUATION_GEN
SAMPLE_FOLDER = config_local.SAMPLE_FOLDER
TRAIN_DIR = config_local.TRAIN_DIR
VAL_DIR = config_local.VAL_DIR
