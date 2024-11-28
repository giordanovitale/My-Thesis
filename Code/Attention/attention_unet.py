from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras import backend as K # type: ignore
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D # type: ignore

import sys
sys.path.append('Code/')
sys.path.append('Code/Attention/')
from helpers import *
from attention_block import attention_block

def attention_unet(input_size):

    inputs = layers.Input(shape=input_size)

    #################
    # Encoder block #
    #################
    encoder = ResNet50(input_tensor=inputs, weights='imagenet', include_top=False, input_shape=input_size)

    # Set the first N layers to be non-trainable (frozen)
    freeze_until = 113
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
    #   Bottleneck  #
    #################

    bottleneck = conv_batch_relu(input=encoder_output, filters=1024)
    bottleneck = layers.SpatialDropout2D(0.3, name='test_name')(bottleneck)

    #################
    # Decoder block #
    #################
    
    # First decoder block
    dec_1 = attention_block(x=skip_connections[3], g=bottleneck, inter_channel=1024)
    dec_1 = layers.SpatialDropout2D(0.3)(dec_1)

    dec_1 = conv_batch_relu(input=dec_1, filters=512)
    dec_1 = layers.SpatialDropout2D(0.3)(dec_1)
    dec_1 = conv_batch_relu(input=dec_1, filters=512)
    dec_1 = layers.SpatialDropout2D(0.3)(dec_1)

    # Second decoder block
    dec_2 = attention_block(x=skip_connections[2], g=dec_1, inter_channel=512)
    dec_2 = layers.SpatialDropout2D(0.2)(dec_2)

    dec_2 = conv_batch_relu(input=dec_2, filters=256)
    dec_2 = layers.SpatialDropout2D(0.2)(dec_2)
    dec_2 = conv_batch_relu(input=dec_2, filters=256)
    dec_2 = layers.SpatialDropout2D(0.2)(dec_2)

    # Third decoder block
    dec_3 = attention_block(x=skip_connections[1], g=dec_2, inter_channel=256)
    dec_3 = layers.SpatialDropout2D(0.1)(dec_3)

    dec_3 = conv_batch_relu(input=dec_3, filters=128)
    dec_3 = layers.SpatialDropout2D(0.1)(dec_3)
    dec_3 = conv_batch_relu(input=dec_3, filters=128)
    dec_3 = layers.SpatialDropout2D(0.1)(dec_3)

    # Fourth decoder block
    dec_4 = attention_block(x=skip_connections[0], g=dec_3, inter_channel=128)
    dec_4 = layers.SpatialDropout2D(0.2)(dec_4)

    dec_4 = conv_batch_relu(input=dec_4, filters=64)
    dec_4 = layers.SpatialDropout2D(0.05)(dec_4)
    dec_4 = conv_batch_relu(input=dec_4, filters=64)
    dec_4 = layers.SpatialDropout2D(0.05)(dec_4)

    # Final upsampling
    x = layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2,2),
                               padding='same', kernel_initializer='he_normal')(dec_4)
    x = layers.BatchNormalization(name='test_name2')(x)
    x = layers.ReLU()(x)

    # Output layer
    outputs = layers.Conv2D(filters=7, kernel_size=(1, 1), activation='softmax')(dec_4)

    model = models.Model(inputs=inputs, outputs=outputs, name='Attention_UNet')

    return model

# model = attention_unet((512,512,3))
# model.summary()


def simple_attention_unet(input_size):

    input = layers.Input(shape=input_size)

    #################
    # Encoder block #
    #################
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

    #################
    #   Bottleneck  #
    #################

    bottleneck = conv_batch_relu(input=x, filters=1024)
    bottleneck = layers.SpatialDropout2D(0.3, name='test_name')(bottleneck)

    #################
    # Decoder block #
    #################
    
    # First decoder block
    dec_1 = attention_block(x=fourth_sc, g=bottleneck, inter_channel=1024)
    dec_1 = layers.SpatialDropout2D(0.3)(dec_1)

    dec_1 = conv_batch_relu(input=dec_1, filters=512)
    dec_1 = layers.SpatialDropout2D(0.3)(dec_1)
    dec_1 = conv_batch_relu(input=dec_1, filters=512)
    dec_1 = layers.SpatialDropout2D(0.3)(dec_1)

    # Second decoder block
    dec_2 = attention_block(x=third_sc, g=dec_1, inter_channel=512)
    dec_2 = layers.SpatialDropout2D(0.2)(dec_2)

    dec_2 = conv_batch_relu(input=dec_2, filters=256)
    dec_2 = layers.SpatialDropout2D(0.2)(dec_2)
    dec_2 = conv_batch_relu(input=dec_2, filters=256)
    dec_2 = layers.SpatialDropout2D(0.2)(dec_2)

    # Third decoder block
    dec_3 = attention_block(x=second_sc, g=dec_2, inter_channel=256)
    dec_3 = layers.SpatialDropout2D(0.1)(dec_3)

    dec_3 = conv_batch_relu(input=dec_3, filters=128)
    dec_3 = layers.SpatialDropout2D(0.1)(dec_3)
    dec_3 = conv_batch_relu(input=dec_3, filters=128)
    dec_3 = layers.SpatialDropout2D(0.1)(dec_3)

    # Fourth decoder block
    dec_4 = attention_block(x=first_sc, g=dec_3, inter_channel=128)
    dec_4 = layers.SpatialDropout2D(0.2)(dec_4)

    dec_4 = conv_batch_relu(input=dec_4, filters=64)
    dec_4 = layers.Dropout(0.05)(dec_4)
    dec_4 = conv_batch_relu(input=dec_4, filters=64)
    dec_4 = layers.SpatialDropout2D(0.05)(dec_4)

    # Final upsampling
    x = layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(1,1),
                               padding='same', kernel_initializer='he_normal')(dec_4)
    x = layers.BatchNormalization(name='test_name2')(x)
    x = layers.ReLU()(x)

    # Output layer
    outputs = layers.Conv2D(filters=7, kernel_size=(1, 1), activation='softmax')(x)

    model = models.Model(inputs=input, outputs=outputs, name='Attention_ResNet50_UNet')

    return model

# model = simple_attention_unet((512,512,3))
# model.summary()