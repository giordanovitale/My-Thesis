import tensorflow as tf
from tensorflow.keras import layers #type: ignore
from tensorflow.keras import backend as K #type: ignore

import sys
sys.path.append('Code/')
from helpers import *

"""
Attention U-Net model with pre-trained resnet50 encoder.
"""

def attention_block(x, g, inter_channel):
    """
    This function defines the attention block.

    Args:
        x: input tensor, feature maps
        g: gating tensor
        inter_channel: number of filters for the intermediate convolutional layer

    Returns:
        att_x: attention applied tensor
    """

    shape_x = K.int_shape(x)

    
    # Theta path: getting x to the same shape as g
    theta_x = layers.Conv2D(filters=inter_channel, 
                            kernel_size=(1, 1), 
                            strides=(2, 2), 
                            padding='same',
                            kernel_initializer='he_normal', 
                            use_bias=False)(x)

    # Phi path: getting g to the same shape as x
    phi_g = layers.Conv2D(filters=inter_channel, 
                          kernel_size=(1, 1), 
                          strides=(1, 1), 
                          padding='same',
                          kernel_initializer='he_normal', 
                          use_bias=False)(g)

    # Adding the two paths
    concat_xg = layers.Add()([theta_x, phi_g])
    activation_xg = layers.Activation('relu')(concat_xg)

    # psi path: producing the attention weights
    psi = layers.Conv2D(filters=1, 
                        kernel_size=(1, 1), 
                        strides=(1, 1), 
                        padding='same',
                        kernel_initializer='he_normal', 
                        use_bias=False)(activation_xg)
    
    # Normalizing the weights
    sigmoid = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid)

    # Upscaling the weights to the same shape as x
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1],
                                        shape_x[2] // shape_sigmoid[2]),
                                        data_format='channels_last'
                                   )(sigmoid)

    # Applying the attention weights
    y = layers.Multiply()([upsample_psi, x])

    # Returning the attention applied tensor
    result = layers.Conv2D(filters=inter_channel,
                           kernel_size=(1,1),
                           padding='same',
                           kernel_initializer='he_normal',
                           use_bias=False)(y)
    result_bn = layers.BatchNormalization()(result)

    return result_bn