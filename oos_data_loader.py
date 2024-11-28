import os
import numpy as np
import tensorflow as tf
tf.keras.backend.clear_session()
from PIL import Image


# Data loading and preprocessing
class OosDatasetCreator:
    """Read images, resizes them.
    
    Args:
        images_dir (str): path to images folder.
        size (tuple): image size. Default is (128, 128).
    """
    
    def __init__(self, images_dir, size=(128,128), preprocessing=None):
        self.images_fps = sorted([os.path.join(images_dir, image_id) for image_id in os.listdir(images_dir) if image_id.endswith('.tif')])
        self.size = size
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image = Image.open(self.images_fps[i])

        # convert to npy array
        image = np.array(image)[:,:,:3]

        # resize image and mask
        image = tf.image.resize(image, self.size)

        # Preprocessing function based on the backbone used
        # image = image / 255.0
        if self.preprocessing:
            image = self.preprocessing(image)
        
        return (image,)
        
    def __len__(self):
        return len(self.images_fps)
    


class OosDataLoader(tf.keras.utils.Sequence):
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