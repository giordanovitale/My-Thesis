"""
This script is used to train the specified model on the LoveDA dataset.
The final model is then saved in the Models_saved folder.
"""

import sys
sys.path.append('Code/')
sys.path.append('Config/')
from dataset_loader import DatasetCreator, DataLoader
from custom_loss import *
from helpers import *
import config #type: ignore
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping #type: ignore
from tensorflow.keras.metrics import OneHotIoU, OneHotMeanIoU #type: ignore


# Create Training 
x_train_dir = 'LoveDA/train/img/'
y_train_dir = 'LoveDA/train/masks/'
train_dataset = DatasetCreator(x_train_dir, y_train_dir, training=True, size=(config.IMAGE_SIZE,config.IMAGE_SIZE),
                               preprocessing=config.PREPROCESSING)
train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

# Create Validation
x_valid_dir = 'LoveDA/val/img/'
y_valid_dir = 'LoveDA/val/masks/'
val_dataset = DatasetCreator(x_valid_dir, y_valid_dir, training=False, size=(config.IMAGE_SIZE,config.IMAGE_SIZE),
                             preprocessing=config.PREPROCESSING)
val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

print("\nTraining and Validation Sets created.") 
print("\nNumber of training images: ", len(train_dataset))
print("Number of validation images: ", len(val_dataset))
print("\nModelling part starting now...")


# Learning Rate Scheduler, reduce on plateau
lr_schedule = ReduceLROnPlateau(monitor='val_loss', 
                                factor=0.5, 
                                patience=12, 
                                verbose=1,
                                mode='min')


# # Create the PolynomialDecay learning rate schedule
# lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
#     initial_learning_rate=0.001,
#     decay_steps=config.EPOCHS,
#     end_learning_rate=0,
#     power=0.6
# )

# Modelling
model = config.MODEL_CHOICE
model.compile(optimizer=tf.keras.optimizers.Nadam(),
              loss=paper_loss_function, 
              metrics=[OneHotMeanIoU(num_classes=config.NUM_CLASSES),
                       OneHotIoU(num_classes=config.NUM_CLASSES, target_class_ids=[0], name='iou_background'),
                       OneHotIoU(num_classes=config.NUM_CLASSES, target_class_ids=[1], name='iou_building'),
                       OneHotIoU(num_classes=config.NUM_CLASSES, target_class_ids=[2], name='iou_road'),
                       OneHotIoU(num_classes=config.NUM_CLASSES, target_class_ids=[3], name='iou_water'),
                       OneHotIoU(num_classes=config.NUM_CLASSES, target_class_ids=[4], name='iou_barren'),
                       OneHotIoU(num_classes=config.NUM_CLASSES, target_class_ids=[5], name='iou_forest'),
                       OneHotIoU(num_classes=config.NUM_CLASSES, target_class_ids=[6], name='iou_agriculture')
                       ])

# TensorBoard LogDir
tensorboard_callback = TensorBoard(log_dir=config.LOG_DIR, histogram_freq=1)

# Model Checkpoint
checkpoint = ModelCheckpoint(f'Models_saved/{config.MODEL_FAMILY}/{config.MODEL_NAME}.keras', 
                             monitor='val_loss', 
                             save_best_only=True,
                             verbose=1, 
                             mode='min')

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=70, verbose=1, mode='min', min_delta=0.001)


csvlogger = tf.keras.callbacks.CSVLogger(f'CSV_Logger/{config.MODEL_FAMILY}/{config.MODEL_NAME}.csv', separator=',', append=False)

# Training
model.fit(train_dataloader, epochs=config.EPOCHS,
          validation_data=val_dataloader,
          callbacks=[tensorboard_callback, checkpoint, lr_schedule, early_stopping, csvlogger])