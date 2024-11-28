import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

# may be an alternative: https://youtu.be/KfDMv2873uM?t=437

"""
Classes:
    0 = background
    1 = building
    2 = road
    3 = water
    4 = barren
    5 = forest
    6 = agriculture
"""

# Hyperparameters
epsilon = 1e-5
smooth = 1
alpha = 0.8

# Weights computed in trial.ipynb [0.373920, 1.342083, 2.816789, 2.321565, 2.845884, 0.923515, 0.734992]
# the following have been manually updated to reflect the corrected weights
background_weight = 0.28
building_weight = 1.342083
road_weight = 2.816789
water_weight = 2.321565
barren_weight = 2.845884 # this was tried at 2.6, 2.3
forest_weight = 0.923515 # this was tried at 1.2, 1.3, 1.8, 0.8
agriculture_weight = 0.734992
weights_sum = background_weight + building_weight + road_weight + water_weight + barren_weight + forest_weight + agriculture_weight

"""
Tversky Loss:
    TP/(TP + alpha*FP + (1-alpha)*FN)

alpha is a hyperparameter that controls the balance between false positives and false negatives.
1-alpha is the weight of false negatives.
"""

def tversky_background(y_true, y_pred, num_channel=0):
    true_positives = K.sum(y_true[:, :, :, num_channel] * y_pred[:, :, :, num_channel] )
    false_negatives = K.sum(y_true[:, :, :, num_channel] * (1 - y_pred[:, :, :, num_channel]) )
    false_positives = K.sum((1 - y_true[:, :, :, num_channel]) * y_pred[:, :, :, num_channel] )
    return K.mean(true_positives / (true_positives + alpha*false_positives + (1-alpha)*false_negatives + epsilon))

def tversky_building(y_true, y_pred, num_channel=1):
    true_positives = K.sum(y_true[:, :, :, num_channel] * y_pred[:, :, :, num_channel] )
    false_negatives = K.sum(y_true[:, :, :, num_channel] * (1 - y_pred[:, :, :, num_channel]) )
    false_positives = K.sum((1 - y_true[:, :, :, num_channel]) * y_pred[:, :, :, num_channel] )
    return K.mean(true_positives / (true_positives + alpha*false_positives + (1-alpha)*false_negatives + epsilon))

def tversky_road(y_true, y_pred, num_channel=2):
    true_positives = K.sum(y_true[:, :, :, num_channel] * y_pred[:, :, :, num_channel] )
    false_negatives = K.sum(y_true[:, :, :, num_channel] * (1 - y_pred[:, :, :, num_channel]) )
    false_positives = K.sum((1 - y_true[:, :, :, num_channel]) * y_pred[:, :, :, num_channel] )
    return K.mean(true_positives / (true_positives + alpha*false_positives + (1-alpha)*false_negatives + epsilon))

def tversky_water(y_true, y_pred, num_channel=3):
    true_positives = K.sum(y_true[:, :, :, num_channel] * y_pred[:, :, :, num_channel] )
    false_negatives = K.sum(y_true[:, :, :, num_channel] * (1 - y_pred[:, :, :, num_channel]) )
    false_positives = K.sum((1 - y_true[:, :, :, num_channel]) * y_pred[:, :, :, num_channel] )
    return K.mean(true_positives / (true_positives + alpha*false_positives + (1-alpha)*false_negatives + epsilon))

def tversky_barren(y_true, y_pred, num_channel=4, alpha=0.5):
    true_positives = K.sum(y_true[:, :, :, num_channel] * y_pred[:, :, :, num_channel] )
    false_negatives = K.sum(y_true[:, :, :, num_channel] * (1 - y_pred[:, :, :, num_channel]) )
    false_positives = K.sum((1 - y_true[:, :, :, num_channel]) * y_pred[:, :, :, num_channel] )
    return K.mean(true_positives / (true_positives + alpha*false_positives + (1-alpha)*false_negatives + epsilon))

def tversky_forest(y_true, y_pred, num_channel=5, alpha=0.3):
    true_positives = K.sum(y_true[:, :, :, num_channel] * y_pred[:, :, :, num_channel] )
    false_negatives = K.sum(y_true[:, :, :, num_channel] * (1 - y_pred[:, :, :, num_channel]) )
    false_positives = K.sum((1 - y_true[:, :, :, num_channel]) * y_pred[:, :, :, num_channel] )
    return K.mean(true_positives / (true_positives + alpha*false_positives + (1-alpha)*false_negatives + epsilon))

def tversky_agriculture(y_true, y_pred, num_channel=6, alpha=0.3):
    true_positives = K.sum(y_true[:, :, :, num_channel] * y_pred[:, :, :, num_channel] )
    false_negatives = K.sum(y_true[:, :, :, num_channel] * (1 - y_pred[:, :, :, num_channel]) )
    false_positives = K.sum((1 - y_true[:, :, :, num_channel]) * y_pred[:, :, :, num_channel] )
    return K.mean(true_positives / (true_positives + alpha*false_positives + (1-alpha)*false_negatives + epsilon))

def tversky_index(y_true, y_pred):
    return(
        (background_weight*tversky_background(y_true, y_pred) +
        building_weight*tversky_building(y_true, y_pred) +
        road_weight*tversky_road(y_true, y_pred) +
        water_weight*tversky_water(y_true, y_pred) +
        barren_weight*tversky_barren(y_true, y_pred) +
        forest_weight*tversky_forest(y_true, y_pred) +
        agriculture_weight*tversky_agriculture(y_true, y_pred)) / weights_sum
    )

def foc_tversky(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    pt_2 = tversky_index(y_true, y_pred)
    gamma = 4.0/3.0
    return K.pow((1-pt_2), gamma)


"""
Dice Metrics:
    2*TP/(2*TP + FP + FN)
"""

def dice_background(y_true, y_pred, num_channel=0):
    true_positives = K.sum(y_true[:, :, :, num_channel] * y_pred[:, :, :, num_channel] )
    false_negatives = K.sum(y_true[:, :, :, num_channel] * (1 - y_pred[:, :, :, num_channel]) )
    false_positives = K.sum((1 - y_true[:, :, :, num_channel]) * y_pred[:, :, :, num_channel] )
    return K.mean((2*true_positives) / (2*true_positives + false_positives + false_negatives + epsilon))

def dice_building(y_true, y_pred, num_channel=1):
    true_positives = K.sum(y_true[:, :, :, num_channel] * y_pred[:, :, :, num_channel] )
    false_negatives = K.sum(y_true[:, :, :, num_channel] * (1 - y_pred[:, :, :, num_channel]) )
    false_positives = K.sum((1 - y_true[:, :, :, num_channel]) * y_pred[:, :, :, num_channel] )
    return K.mean((2*true_positives) / (2*true_positives + false_positives + false_negatives + epsilon))

def dice_road(y_true, y_pred, num_channel=2):
    true_positives = K.sum(y_true[:, :, :, num_channel] * y_pred[:, :, :, num_channel] )
    false_negatives = K.sum(y_true[:, :, :, num_channel] * (1 - y_pred[:, :, :, num_channel]) )
    false_positives = K.sum((1 - y_true[:, :, :, num_channel]) * y_pred[:, :, :, num_channel] )
    return K.mean((2*true_positives) / (2*true_positives + false_positives + false_negatives + epsilon))

def dice_water(y_true, y_pred, num_channel=3):
    true_positives = K.sum(y_true[:, :, :, num_channel] * y_pred[:, :, :, num_channel] )
    false_negatives = K.sum(y_true[:, :, :, num_channel] * (1 - y_pred[:, :, :, num_channel]) )
    false_positives = K.sum((1 - y_true[:, :, :, num_channel]) * y_pred[:, :, :, num_channel] )
    return K.mean((2*true_positives) / (2*true_positives + false_positives + false_negatives + epsilon))

def dice_barren(y_true, y_pred, num_channel=4):
    true_positives = K.sum(y_true[:, :, :, num_channel] * y_pred[:, :, :, num_channel] )
    false_negatives = K.sum(y_true[:, :, :, num_channel] * (1 - y_pred[:, :, :, num_channel]) )
    false_positives = K.sum((1 - y_true[:, :, :, num_channel]) * y_pred[:, :, :, num_channel] )
    return K.mean((2*true_positives) / (2*true_positives + false_positives + false_negatives + epsilon))

def dice_forest(y_true, y_pred, num_channel=5):
    true_positives = K.sum(y_true[:, :, :, num_channel] * y_pred[:, :, :, num_channel] )
    false_negatives = K.sum(y_true[:, :, :, num_channel] * (1 - y_pred[:, :, :, num_channel]) )
    false_positives = K.sum((1 - y_true[:, :, :, num_channel]) * y_pred[:, :, :, num_channel] )
    return K.mean((2*true_positives) / (2*true_positives + false_positives + false_negatives + epsilon))

def dice_agriculture(y_true, y_pred, num_channel=6):
    true_positives = K.sum(y_true[:, :, :, num_channel] * y_pred[:, :, :, num_channel] )
    false_negatives = K.sum(y_true[:, :, :, num_channel] * (1 - y_pred[:, :, :, num_channel]) )
    false_positives = K.sum((1 - y_true[:, :, :, num_channel]) * y_pred[:, :, :, num_channel] )
    return K.mean((2*true_positives) / (2*true_positives + false_positives + false_negatives + epsilon))

def dice_overall(y_true, y_pred):
    
    dice_loss = 1 - ((background_weight*dice_background(y_true, y_pred) +
        building_weight*dice_building(y_true, y_pred) +
        road_weight*dice_road(y_true, y_pred) +
        water_weight*dice_water(y_true, y_pred) +
        barren_weight*dice_barren(y_true, y_pred) +
        forest_weight*dice_forest(y_true, y_pred) +
        agriculture_weight*dice_agriculture(y_true, y_pred)) / weights_sum)
    
    return dice_loss


"""
Custom Loss as per the paper:

    L = ce + alpha*bce + beta*dice

where:
    ce = Categorical Cross-Entropy
    bce = Binary Cross-Entropy applied wrt the background class
    dice = Dice Loss
"""
def paper_loss_function(y_true, y_pred, alpha=0.3, beta=0.7):

    # Categorical Cross-Entropy (Reduce to scalar)
    ce = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
        tf.reshape(y_true, [-1, y_true.shape[-1]]), 
        tf.reshape(y_pred, [-1, y_pred.shape[-1]])
    ))

    # Binary Cross-Entropy for background class (Reduce to scalar)
    bce = tf.keras.losses.binary_crossentropy(
        tf.reshape(y_true[..., 0], [-1]), 
        tf.reshape(y_pred[..., 0], [-1]))

    # Dice Loss
    ftl = foc_tversky(y_true, y_pred)

    # total Loss
    total_loss = ce + alpha*bce + beta*ftl 
    return total_loss