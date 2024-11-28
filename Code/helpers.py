import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, LeakyReLU
from sklearn.metrics import confusion_matrix
from custom_loss import *
import sys


def conv_batch_relu(input, filters, strides=(1,1), kernel_size=(3,3), padding='same', kernel_initializer='he_normal'):
    """
    Creates a block of Conv2D -> BatchNormalization -> ReLU.
    :param filters: int. The number of filters.
    :param kernel_size: Tuple. The kernel size.
    :param input: tensor. The input tensor.
    :param padding: str. The padding to use.
    :param kernel_initializer: str. The kernel initializer to use.
    :return: tensor. The output tensor.
    """
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides = strides, 
               padding=padding, kernel_initializer=kernel_initializer, 
               use_bias=False)(input)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    return x

def conv_transpose_concat(input, skip_connection, filters):
    """
    Performs a 2D transposed convolution followed by a concatenation with a skip connection.
    :param input: tensor. The input tensor.
    :param skip_connection: tensor. The skip connection tensor.
    :param filters: int. The number of filters.
    :return: tensor. The output tensor.
    """
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', 
                               use_bias=False, kernel_initializer='he_normal')(input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Concatenate()([x, skip_connection])
    return x


def compute_confusion_matrix(model, dataset):
    """
    Computes the confusion matrix for a given model and dataset.
    :param model: The model to use for predictions.
    :param dataset: The dataset to use for predictions.
    :param num_classes: The number of classes.
    :return: The confusion matrix.
    """
    all_predictions = []
    all_ground_truth = []

    for batch in dataset:
        images, ground_truth = batch

        # Make predictions for the current batch
        predictions = model.predict(images)

        # Convert one-hot encoded predictions to class labels
        predicted_classes = np.argmax(predictions, axis=-1) 

        # Convert one-hot encoded ground truth to class labels 
        true_classes = np.argmax(ground_truth, axis=-1)  

        all_predictions.append(predicted_classes)
        all_ground_truth.append(true_classes)

    # After going through all batches, concatenate everything
    all_predictions = np.concatenate(all_predictions, axis=0) 
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)

    # Flatten the arrays to get pixel-level predictions and ground truth
    flat_predictions = all_predictions.flatten()
    flat_true_labels = all_ground_truth.flatten()

    conf_matrix = confusion_matrix(flat_true_labels, flat_predictions)

    return conf_matrix


def compute_iou_validation(model, dataset):
    """
    Computes the IoU metric for each class, and the overall MeanIoU.
    """
    # Initialize the metric
    metric = tf.keras.metrics.OneHotMeanIoU(num_classes=7)

    for batch in dataset:
        image, mask = batch
        batch_predictions = model.predict(image)
        metric.update_state(y_true=mask, y_pred=batch_predictions)

    overall_mean_iou = metric.result().numpy()

    metric.reset_state()

    return overall_mean_iou

def compute_single_iou_validation(model, dataset):
    """
    Computes the IoU metric for a single class.
    """
    # Initialize IoU metrics for each class
    metrics = [tf.keras.metrics.OneHotIoU(num_classes=7, target_class_ids=[i]) for i in range(7)]

    # Go through the dataset and predict only once
    for batch in dataset:
        image, mask = batch
        batch_predictions = model.predict(image)

        # Update state for each class-specific metric
        for metric in metrics:
            metric.update_state(y_true=mask, y_pred=batch_predictions)

    # Compute the IoU for each class
    iou_per_class = [metric.result().numpy() for metric in metrics]

    # Reset the state for all metrics
    for metric in metrics:
        metric.reset_state()

    return iou_per_class

def reverse_preprocess_input(img):
    """
    Reverse the preprocessing applied by keras.applications.resnet50.preprocess_input.
    
    Args:
    img: Preprocessed image (with 3 channels, values subtracted by ImageNet mean).
    
    Returns:
    Reversed image in the range [0, 255].
    """
    # If the input is a TensorFlow tensor, convert it to a NumPy array
    if isinstance(img, tf.Tensor):
        img = img.numpy()

    img = img.astype(np.float64)
    # Reverse the mean subtraction
    img[..., 0] += 103.939  # Add back the ImageNet mean for the R channel
    img[..., 1] += 116.779  # Add back the ImageNet mean for the G channel
    img[..., 2] += 123.68   # Add back the ImageNet mean for the B channel

    # Clip the values to ensure they are within valid pixel range
    img = np.clip(img, 0, 255)

    return img.astype(np.uint8)

def plot_history(history, figsize=(12,6), y_lim_left=(0,1), y_lim_right=(0,1.02)):
    """
    Plots the loss and accuracy curves.
    :param df: the data frame containing the histories we want to plot
    :param y_lim: Tuple. If not differently specified, it's (0,1). Only applies to the accuracy plot.
    :return: None
    """
    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=2)


    # Plot Loss
    sns.lineplot(data=history,
                 x="epoch",
                 y="val_loss",
                 ax=ax[0],
                 label="Validation Loss",
                 linewidth=2,
                 color="#F58939",
                 errorbar=None)

    sns.lineplot(data=history,
                 x="epoch",
                 y="loss",
                 ax=ax[0],
                 label="Training Loss",
                 linewidth=2,
                 color="#40CA37",
                 errorbar=None)

    # Plot Accuracy
    sns.lineplot(data=history,
                 x="epoch",
                 y="val_accuracy",
                 ax=ax[1],
                 label="Validation Accuracy",
                 linewidth=2,
                 color="#F58939",
                 errorbar=None)

    sns.lineplot(data=history,
                 x="epoch",
                 y="accuracy",
                 ax=ax[1],
                 label="Training Accuracy",
                 linewidth=2,
                 color="#40CA37",
                 errorbar=None)

    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Categorical Crossentropy")
    ax[0].set_title("Training Loss vs Validation Loss")
    ax[0].set_ylim(y_lim_left)
    ax[0].legend()

    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy (%)")
    ax[1].set_title("Training Accuracy vs Validation Accuracy")
    ax[1].set_ylim(y_lim_right)
    ax[1].legend(loc="lower right")

    plt.show()