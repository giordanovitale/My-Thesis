"""
Config file to specify the parameters for training the model and paths.
"""
import os
import sys
sys.path.append('Code/UNet/')
sys.path.append('Code/DeepLab/')
sys.path.append('Code/Attention/')
from transfer_learning import resnet50_unet, efficientnet_unet, resnet101_unet #type: ignore
from deeplab import DeepLabV3Plus #type: ignore
from tensorflow.keras.applications import resnet, efficientnet #type: ignore
from attention_unet import attention_unet #type: ignore
from unet import UNet #type: ignore

# Model parameters
IMAGE_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 100
NUM_CLASSES = 7
EXPERIMENT = 9


# Model Architecture
MODEL_FAMILY = 'Attention_UNet'
PREPROCESSING = None # efficientnet.preprocess_input if 'EfficientNet' in MODEL_FAMILY else resnet.preprocess_input
MODEL_CHOICE = attention_unet(input_size=(IMAGE_SIZE, IMAGE_SIZE, 3))

# Configuration Name
MODEL_NAME = f"Experiment_Number_{EXPERIMENT}"

# Directory settings
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
    )
LOG_DIR = 'Logs/' + MODEL_FAMILY + '/' + MODEL_NAME
print(f"Logs will be saved in {LOG_DIR}")
print(f"model will be saved in Models_saved/{MODEL_FAMILY}/{MODEL_NAME}.keras")

# MODEL_CHOICE.summary()