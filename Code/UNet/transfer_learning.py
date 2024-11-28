"""
UNet with pretrained ResNet50 encoder for transfer learning. 
ImageNet weights are used for the encoder part of the network.
"""

import sys
sys.path.append('Code/UNet/')
sys.path.append('Code/')
sys.path.append('Config/')
import tensorflow as tf
tf.keras.backend.clear_session()
from custom_loss import *
from tensorflow.keras.applications import ResNet50, EfficientNetB5, ResNet101
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dropout, Conv2DTranspose, SpatialDropout2D
from helpers import conv_batch_relu, conv_transpose_concat
NUM_CLASSES = 7

def resnet50_unet(input_size):
    
    input = Input(shape=input_size)

    #################
    # Encoder block #
    #################
    encoder = ResNet50(input_tensor=input, weights='imagenet', include_top=False, input_shape=input_size)

    # Set the first N layers to be non-trainable (frozen)
    freeze_until = 143
    for layer in encoder.layers[:freeze_until]:
        layer.trainable = False

    # Set the remaining layers to be trainable
    for layer in encoder.layers[freeze_until:]:
        layer.trainable = True

    skip_connections = [
        encoder.get_layer('conv1_relu').output,
        encoder.get_layer('conv2_block3_out').output,
        encoder.get_layer('conv3_block4_out').output,
        encoder.get_layer('conv4_block6_out').output,
    ]
    encoder_output = encoder.get_layer('conv5_block3_out').output

    #################
    # Decoder block #
    #################

    x = conv_batch_relu(input=encoder_output, filters=256)
    x = Dropout(0.3, name='test_name')(x)

    # # Atrous Convolution in the bottleneck
    # x = layers.Conv2D(filters=512, kernel_size=3, padding='same', dilation_rate=3, kernel_initializer='he_normal',
    #                   name='atrous_conv')(encoder_output)
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU()(x) 

    x = conv_transpose_concat(input=x, skip_connection=skip_connections[3], filters=1024)
    x = conv_batch_relu(x, filters=1024)
    x = Dropout(0.3)(x)

    x = conv_transpose_concat(input=x, skip_connection=skip_connections[2], filters=512)
    x = conv_batch_relu(x, filters=512)
    x = Dropout(0.3)(x)

    x = conv_transpose_concat(input=x, skip_connection=skip_connections[1], filters=256)
    x = conv_batch_relu(x, filters=256)
    x = Dropout(0.2)(x)

    x = conv_transpose_concat(input=x, skip_connection=skip_connections[0], filters=128)
    x = conv_batch_relu(x, filters=128)
    x = Dropout(0.1)(x)

    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)

    outputs = layers.Conv2D(NUM_CLASSES, (1, 1), activation='softmax')(x)

    model = models.Model(inputs=input, outputs=outputs, name='Pretrained_ResNet50_UNet')

    return model

# model = resnet50_unet(input_size=(128,128,3))
# model.summary()




def efficientnet_unet(input_size):
    input = Input(shape=input_size)

    #################
    # Encoder block #
    #################
    encoder = EfficientNetB5(input_tensor=input, weights='imagenet', include_top=False, input_shape=input_size)

    # Set the first N layers to be non-trainable (frozen)
    freeze_until = 500
    for layer in encoder.layers[:freeze_until]:
        layer.trainable = False

    # Set the remaining layers to be trainable
    for layer in encoder.layers[freeze_until:]:
        layer.trainable = True

    skip_connections = [
        encoder.get_layer('block2a_expand_activation').output,
        encoder.get_layer('block3a_expand_activation').output,
        encoder.get_layer('block4a_expand_activation').output,
        encoder.get_layer('block6a_expand_activation').output 
    ]

    encoder_output = encoder.output

    #################
    # Decoder block #
    #################

    x = conv_batch_relu(input=encoder_output, filters=256)
    x = SpatialDropout2D(0.3, name='test_name')(x)

    x = conv_transpose_concat(input=x, skip_connection=skip_connections[3], filters=1024)
    x = conv_batch_relu(x, filters=1024)
    x = SpatialDropout2D(0.2)(x)

    x = conv_transpose_concat(input=x, skip_connection=skip_connections[2], filters=512)
    x = conv_batch_relu(x, filters=512)
    x = SpatialDropout2D(0.2)(x)

    x = conv_transpose_concat(input=x, skip_connection=skip_connections[1], filters=256)
    x = conv_batch_relu(x, filters=256)
    x = SpatialDropout2D(0.1)(x)

    x = conv_transpose_concat(input=x, skip_connection=skip_connections[0], filters=128)
    x = conv_batch_relu(x, filters=128)

    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)

    outputs = layers.Conv2D(NUM_CLASSES, (1, 1), activation='softmax')(x)

    model = models.Model(inputs=input, outputs=outputs, name='Pretrained_EfficientNetB5_UNet')
    
    return model

# model = efficientnet_unet(input_size=(256,256,3))
# model.summary()

def resnet101_unet(input_size):
    
    input = Input(shape=input_size)

    #################
    # Encoder block #
    #################
    encoder = ResNet101(input_tensor=input, weights='imagenet', include_top=False, input_shape=input_size)

    # Set the first N layers to be non-trainable (frozen)
    freeze_until = 313
    for layer in encoder.layers[:freeze_until]:
        layer.trainable = False

    # Set the remaining layers to be trainable
    for layer in encoder.layers[freeze_until:]:
        layer.trainable = True

    skip_connections = [
        encoder.get_layer('conv1_relu').output,
        encoder.get_layer('conv2_block3_out').output,
        encoder.get_layer('conv3_block4_out').output,
        encoder.get_layer('conv4_block6_out').output,
    ]
    encoder_output = encoder.get_layer('conv5_block3_out').output

    #################
    # Decoder block #
    #################

    x = conv_transpose_concat(input=encoder_output, skip_connection=skip_connections[3], filters=1024)
    x = conv_batch_relu(x, filters=1024)
    x = SpatialDropout2D(rate=0.4)(x)

    x = conv_transpose_concat(input=x, skip_connection=skip_connections[2], filters=512)
    x = conv_batch_relu(x, filters=512)
    x = SpatialDropout2D(rate=0.3)(x)

    x = conv_transpose_concat(input=x, skip_connection=skip_connections[1], filters=256)
    x = conv_batch_relu(x, filters=256)
    x = SpatialDropout2D(rate=0.2)(x)

    x = conv_transpose_concat(input=x, skip_connection=skip_connections[0], filters=128)
    x = conv_batch_relu(x, filters=128)
    x = SpatialDropout2D(rate=0.1)(x)

    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    outputs = layers.Conv2D(NUM_CLASSES, (1, 1), activation='softmax')(x)

    model = models.Model(inputs=input, outputs=outputs, name='Pretrained_ResNet101_UNet')

    return model

# model = resnet101_unet(input_size=(256, 256, 3))
# model.summary()
# for i, layer in enumerate(encoder.layers):
#     print(f"Layer {i}: {layer.name} - Trainable: {layer.trainable}")