'https://github.com/qubvel/segmentation_models/blob/master/examples/multiclass%20segmentation%20(camvid).ipynb'

import os
import numpy as np
import tensorflow as tf
tf.keras.backend.clear_session()
import sys
sys.path.append('/home/giordano_vitale/ETE-Thesis-2024-ML-002/Code/UNet/')
import albumentations as A
import keras

"""
In the DatasetCreator class, the __getitem__ method reads the .npy files from disk 
only when they are requested during training. 
This means that the data is not preloaded into memory but is fetched dynamically during training or validation.
"""

# Data loading and preprocessing
class DatasetCreator:
    """Read images, apply preprocessing and augmentation transformations.
    
    Args:
        images_dir (str): path to images folder.
        masks_dir (str): path to segmentation masks folder.
        training (bool): if true, apply augmentation.
        size (tuple): image and mask sizes. Default is (128, 128).
    """
    
    def __init__(self, images_dir, masks_dir, size=(128,128), training=False, preprocessing=None):
        self.images_fps = sorted([os.path.join(images_dir, image_id) for image_id in os.listdir(images_dir) if image_id.endswith('.npy')])
        self.masks_fps = sorted([os.path.join(masks_dir, mask_id) for mask_id in os.listdir(masks_dir) if mask_id.endswith('.npy')])
        self.training = training
        self.size = size
        self.preprocessing = preprocessing

        # Augmentation pipeline
        # https://albumentations.ai/docs/examples/example_kaggle_salt/?h=segmentation
        self.augmentation_all = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomCrop(height=int(self.size[0]/2), 
                         width=int(self.size[1]/2), 
                         p=0.2)
        ]) 
        self.augmentation_images_only = A.Compose([
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.3),
        ])
        # p=1.0, random_state=42 random_state is set for reproducibility 
        # https://albumentations.ai/docs/examples/example_kaggle_salt/?h=segmentation#:~:text=We%20fix%20the,transformations%20each%20time.
        

    def __getitem__(self, i):
        # read data
        image = np.load(self.images_fps[i])
        mask = np.load(self.masks_fps[i])

        # Apply augmentation only for Training Set
        if self.training:
            augmented = self.augmentation_all(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

            image = self.augmentation_images_only(image=image)['image']

        # resize image and mask
        image = tf.image.resize(image, self.size)
        mask = tf.image.resize(mask, self.size, method='nearest')

        # Preprocessing function based on the backbone used
        if self.preprocessing is not None:
            image = self.preprocessing(image)
        else:
            image = image / 255.0
        # JUSTIFICATION HERE: 
        # https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50#:~:text=Note%3A%20each,dataset%2C%20without%20scaling
        # as an alternative, 
        # https://www.tensorflow.org/tutorials/images/transfer_learning#data_preprocessing:~:text=preprocess_input%20%3D%20tf.keras.applications.mobilenet_v2.preprocess_input

        return image, mask
        
    def __len__(self):
        return len(self.images_fps)
    

class DataLoader(tf.keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=4, shuffle=False, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset.__getitem__(self.indexes[j]))
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)




######################################
#### Check that the process is ok ####
######################################

if __name__ == "__main__":

    # Specify the directories for the training and validation images and masks
    x_train_dir = 'LoveDA/train/img/'
    y_train_dir = 'LoveDA/train/masks/'
    x_valid_dir = 'LoveDA/val/img/'
    y_valid_dir = 'LoveDA/val/masks/'

    # Create Training Dataset 
    train_dataset = DatasetCreator(x_train_dir, y_train_dir, training=True)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Create Validation Dataset 
    val_dataset = DatasetCreator(x_valid_dir, y_valid_dir, training=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # After initializing the DataLoader
    print("Initial indexes: ", train_dataloader.indexes)

    # Manually trigger shuffling and print the shuffled indexes
    train_dataloader.on_epoch_end()
    print("Shuffled indexes: ", train_dataloader.indexes)

    val_dataloader.on_epoch_end()
    print("Shuffled indexes: ", val_dataloader.indexes)

    import matplotlib.pyplot as plt

    # Fetch one batch
    batch_images, batch_masks = train_dataloader[0]

    # Visualize the batch
    def visualize_batch(images, batch_size, is_mask=False):
        plt.figure(figsize=(20, 10))
        for i in range(batch_size):
            # Display image
            plt.subplot(2, batch_size, i + 1)
            
            if is_mask:
                plt.imshow(np.argmax(images[i], axis=-1))
            else:
                plt.imshow(images[i])
            plt.title(f'Image {i+1}')
            plt.axis('off')
        plt.show()

    visualize_batch(batch_images, batch_size=8)

    print(f"image.shape, mask.shape: {batch_images.shape, batch_masks.shape}")
    print("Number of training images: ", len(train_dataset))
    print("Number of validation images: ", len(val_dataset))
    visualize_batch(batch_masks, batch_size=8, is_mask=True)