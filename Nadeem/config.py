import torch 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "C:/Users/Charl/Programming/NightSkyify/testDataUnfiltered/mountains_256.npz"
VAL_DIR = "C:/Users/Charl/Programming/NightSkyify/testDataUnfiltered/mountains_256.npz"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "C:/Users/Charl/Programming/NightSkyify/output2/disc.pth.tar"
CHECKPOINT_GEN = "C:/Users/Charl/Programming/NightSkyify/output2/gen.pth.tar"
