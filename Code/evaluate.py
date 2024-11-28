"""
Script to evaluate the model on the validation set.
The metrics considered are: 
    - Mean IoU
    - Single class IoU

Moreover, the confusion matrix is computed.
"""

import tensorflow as tf
from dataset_loader import *
from custom_loss import *
from helpers import *
import sys
sys.path.append('Config/')
# import config
from tensorflow.keras.applications import resnet, efficientnet #type: ignore
from tensorflow.keras.metrics import OneHotIoU, OneHotMeanIoU #type: ignore

BATCH_SIZE = 4
NUM_CLASSES = 7

model = tf.keras.models.load_model('Models_saved/TransferLearning/ResNet101-UNet/Experiment_Number_34.keras', 
                                   compile=False)

# Validation Set
x_valid_dir = 'LoveDA/val/img/'
y_valid_dir = 'LoveDA/val/masks/'

val_dataset = DatasetCreator(x_valid_dir, y_valid_dir, training=False, size=(512,512),
                             preprocessing=resnet.preprocess_input)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Confusion Matrix
# print("\nConfusion matrix:")
# cm = compute_confusion_matrix(model=model, dataset=val_dataloader)
# print("Confusion Matrix", cm)

# Metrics: IoUs
print("\nMetrics:")
mean_iou = compute_iou_validation(model=model, dataset=val_dataloader)

iou_single_class = compute_single_iou_validation(model=model, dataset=val_dataloader)
print("\n Mean IoU: ", mean_iou)
for i, single_iou in enumerate(iou_single_class):
    print(f"Single IoU for class {i}: ", single_iou)