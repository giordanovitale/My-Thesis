import os
import numpy as np
import tensorflow as tf
tf.keras.backend.clear_session()
import sys
sys.path.append('/home/giordano_vitale/ETE-Thesis-2024-ML-002/Code/UNet/')
sys.path.append('/home/giordano_vitale/ETE-Thesis-2024-ML-002/Code/')
from custom_loss import * # type: ignore
from Deployment.oos_data_loader import *

# Dataset Loading
oos_dataset = OosDatasetCreator('Out_of_sample/Images/', 
                                size=(512, 512),
                                preprocessing=tf.keras.applications.resnet.preprocess_input)

oos_dataloader = OosDataLoader(oos_dataset, 
                               batch_size=4, 
                               shuffle=False)

# Model Loading
model = tf.keras.models.load_model('Models_saved/DeepLabV3Plus/Experiment_Number_6.keras',
                                   compile=False)

# Create output directory for individual predictions
output_dir = 'Out_of_sample/Predictions/Individual/'
os.makedirs(output_dir, exist_ok=True)

# Predict and save each batch
for i, batch in enumerate(oos_dataloader):
    images = batch
    batch_predictions = model.predict(images)
    
    for j, prediction in enumerate(batch_predictions):
        np.savez_compressed(os.path.join(output_dir, f'prediction_{i * oos_dataloader.batch_size + j}.npz'), prediction)
    
    # Free memory to ensure that the kernel doesnt crash
    del batch_predictions
    tf.keras.backend.clear_session()