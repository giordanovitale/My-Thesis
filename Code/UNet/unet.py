import numpy as np 
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import sys
sys.path.append('Code/')
from helpers import conv_batch_relu, conv_transpose_concat


def UNet(input_size):

    inputs = Input(shape=input_size)

    # First single encoder block
    x = conv_batch_relu(inputs, filters=32)
    x = conv_batch_relu(x, filters=32)
    first_sc = x
    x = AveragePooling2D(pool_size=(2,2))(first_sc)

    # Second single encoder block
    x = conv_batch_relu(x, filters=64)
    x = conv_batch_relu(x, filters=64)
    second_sc = x
    x = AveragePooling2D(pool_size=(2,2))(second_sc)

    # Third single encoder block
    x = conv_batch_relu(x, filters=128)
    x = conv_batch_relu(x, filters=128)
    third_sc = x
    x = AveragePooling2D(pool_size=(2,2))(third_sc)
    x = layers.SpatialDropout2D(0.1)(x)

    # Fourth single encoder block
    x = conv_batch_relu(x, filters=256)
    x = conv_batch_relu(x, filters=256)
    x = conv_batch_relu(x, filters=256)
    fourth_sc = x
    x = AveragePooling2D(pool_size=(2,2))(fourth_sc)
    x = layers.SpatialDropout2D(0.2)(x)

    # Fifth single encoder block
    x = conv_batch_relu(x, filters=512)
    x = conv_batch_relu(x, filters=512)
    x = conv_batch_relu(x, filters=512)
    x = layers.SpatialDropout2D(0.3)(x)  

    #########################################################################
    #########################################################################

    # First Decoder Block
    x = conv_transpose_concat(input=x, skip_connection=fourth_sc, filters=512)
    x = conv_batch_relu(x, filters=512)
    x = conv_batch_relu(x, filters=256)
    x = layers. SpatialDropout2D(0.2)(x)

    # Second Decoder Block
    x = conv_transpose_concat(input=x, skip_connection=third_sc, filters=256)
    x = conv_batch_relu(x, filters=256)
    x = conv_batch_relu(x, filters=128)
    x = layers.SpatialDropout2D(0.1)(x)

    # Third Decoder Block
    x = conv_transpose_concat(input=x, skip_connection=second_sc, filters=128)
    x = conv_batch_relu(x, filters=128)
    x = conv_batch_relu(x, filters=64)

    # Fourth Decoder Block
    x = conv_transpose_concat(input=x, skip_connection=first_sc, filters=64)
    x = conv_batch_relu(x, filters=64)
    x = conv_batch_relu(x, filters=32)

    ###########################################################################
    ###########################################################################

    # Output layer
    x = Conv2D(filters=7, kernel_size=(1,1), activation='softmax', kernel_initializer='he_normal')(x)
    
    model = Model(inputs=[inputs], outputs=[x], name='UNET')

    return model