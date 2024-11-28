import os
import tensorflow as tf  # type: ignore
import matplotlib.pyplot as plt 
from PIL import Image 
import numpy as np
from matplotlib.colors import ListedColormap

"""
This script takes 4 images, creates the corresponding batch, and performs the prediction using the DeepLab model.
The output is saved in the output_images folder as images, and in the predictions folder as npy arrays.
The images are displayed using a custom color map.
"""

# Custom color map, but we might want to change them
cmap = ListedColormap(['black', 'green', 'blue', 'yellow', 'red', 'purple', 'brown'])

# Load the DeepLab model
deeplab = tf.keras.models.load_model('Deployment/deeplab_model.keras', compile=False)

# Image directories
input_dir = 'Deployment/input_images/'
output_dir = 'Deployment/output_images/'

for image_name in os.listdir(input_dir):
    if image_name.endswith('.tif'):
        input_image_path = os.path.join(input_dir, image_name)
        input_image = Image.open(input_image_path)
        numpy_image = np.array(input_image)

        numpy_image = tf.image.resize(numpy_image, (512, 512))
        numpy_image = tf.keras.applications.resnet.preprocess_input(numpy_image)
        numpy_image = tf.expand_dims(numpy_image, axis=0)

        prediction = deeplab.predict(numpy_image)

        # Save the npy array of the prediction
        np.save(f"Deployment/predictions/{os.path.splitext(image_name)[0]}.npy", prediction)
        
        # Save the predicted mask
        prediction_argmaxed = np.argmax(prediction[0], axis=-1)
        plt.imsave(f"Deployment/output_images/{os.path.splitext(image_name)[0]}.png", prediction_argmaxed, cmap=cmap)