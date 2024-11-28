from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, SpatialDropout2D
from tensorflow.keras.layers import AveragePooling2D, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
import sys
sys.path.append('Code/')
NUM_CLASSES = 7


def ASPP(inputs):
    """
    This function applies multiple convolutional layers with different dilation rates to the input tensor,
    followed by batch normalization and leaky_relu activation. It also includes an average pooling layer followed
    by a 1x1 convolution, batch normalization, leaky_relu activation, and upsampling.

    Args:
        inputs (tf.Tensor): Input tensor of shape (batch_size, height, width, channels).

    Returns:
        tf.Tensor: Output tensor after applying ASPP.
    
    """
    shape = inputs.shape

    # Pooling
    y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]), name='average_pooling')(inputs)
    y_pool = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False, kernel_initializer='he_normal')(y_pool)
    y_pool = BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = Activation('leaky_relu', name=f'leaky_relu_1')(y_pool)
    y_pool = SpatialDropout2D(0.1)(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

    # First Layer
    aspp_1 = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', use_bias=False, kernel_initializer='he_normal')(inputs)
    aspp_1 = BatchNormalization()(aspp_1)
    aspp_1 = Activation('leaky_relu')(aspp_1)
    aspp_1 = SpatialDropout2D(0.15)(aspp_1)

    # Second Layer
    aspp_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same', use_bias=False, kernel_initializer='he_normal')(inputs)
    aspp_2 = BatchNormalization()(aspp_2)
    aspp_2 = Activation('leaky_relu')(aspp_2)
    aspp_2 = SpatialDropout2D(0.20)(aspp_2)

    # Third Layer
    aspp_3 = Conv2D(filters=512, kernel_size=3, dilation_rate=12, padding='same', use_bias=False, kernel_initializer='he_normal')(inputs)
    aspp_3 = BatchNormalization()(aspp_3)
    aspp_3 = Activation('leaky_relu')(aspp_3)
    aspp_3 = SpatialDropout2D(0.25)(aspp_3)

    # Fourth Layer
    aspp_4 = Conv2D(filters=512, kernel_size=3, dilation_rate=18, padding='same', use_bias=False, kernel_initializer='he_normal')(inputs)
    aspp_4 = BatchNormalization()(aspp_4)
    aspp_4 = Activation('leaky_relu')(aspp_4)
    aspp_4 = SpatialDropout2D(0.30)(aspp_4)

    # Concatenate
    y = Concatenate()([y_pool, aspp_1, aspp_2, aspp_3, aspp_4])

    # Final Convolution
    y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', use_bias=False, kernel_initializer='he_normal')(y)
    y = BatchNormalization()(y)
    y = Activation('leaky_relu')(y)
    return y

def DeepLabV3Plus(input_size):
    """
    DeepLabV3+ model.

    This function builds the DeepLabV3+ model using a pre-trained ResNet50 with imagenet weights as the backbone. 
    It applies the ASPP module to the output of the ResNet50 backbone and upsamples the result.

    Args:
        input_size (tuple): Shape of the input tensor, e.g., (height, width, channels).

    Returns:
        tf.keras.Model: DeepLabV3+ model.
    """
    # Input
    inputs = Input(shape=input_size)

    # Backbone
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs, input_shape=input_size)

    # Freezing policy
    freeze_until = 143  
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    # Set the remaining layers to be trainable
    for layer in base_model.layers[freeze_until:]:
        layer.trainable = True

    # Retrieve the encoder output and apply ASPP module
    image_features = base_model.get_layer('conv4_block6_out').output
    x_a = ASPP(image_features)
    x_a = UpSampling2D((4, 4), interpolation="bilinear")(x_a)

    # Low-level features
    x_b = base_model.get_layer('conv2_block3_out').output
    x_b = Conv2D(filters=128, kernel_size=1, padding='same', use_bias=False, kernel_initializer='he_normal')(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation('leaky_relu')(x_b)
    x_b = SpatialDropout2D(0.1)(x_b)

    # Concatenate high-level features (ASPP) with low-level features
    x = Concatenate()([x_a, x_b])

    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('leaky_relu')(x)
    x = SpatialDropout2D(0.2)(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('leaky_relu')(x)
    x = UpSampling2D((4, 4), interpolation="bilinear")(x)

    # Output layer
    outputs = Conv2D(NUM_CLASSES, (1, 1), name='output_layer', activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs, name='DeepLabV3Plus')
    return model

# model = DeepLabV3Plus((128,128,3))
# model.summary()

# base_model = ResNet50(weights='imagenet', include_top=False)
# base_model.summary()