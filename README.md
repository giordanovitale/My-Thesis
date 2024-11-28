# REPOSITORY USAGE
After cloning this repo, the user who wants to execute the code has to open the `Code/train.py` and execute the file. 
It generates dataset batches using the classes that are specified in `Code/dataset_loader.py`, and performs model training. 
The `Config/config.py` file contains all the required parameters, including the model that the user wants to train.

The dataset used can be found [here](https://datasetninja.com/remote-sensing-land-cover-dataset). 
To enable a fast and already up-to-date execution of `train.py`, please ensure to download the data and set up the following directory organization:
```
LoveDA/
    ├───Train/
        ├─── ann/
        ├─── img/
        └─── masks/
    └───Val/
        ├─── ann/
        ├─── img/
        └─── masks/
```

It is required to generate numpy arrays of the images and the corresponding masks and store them in the above-mentioned folder organization, 
in order to obtain a seamless application of the models. 

# FILES DESCRIPTION
### UNet Model Construction - `UNet/unet.py`:
This script contains the structure of the UNet model. It creates a Keras Model can be speficied in the `config.py` called in `Code/train.py`.
It consists of 5 encoder and 5 decoder blocks, for a total of 15M parameters.

### ResNet50 U-Net Model Construction - `UNet/transfer_learning.py`:
This file contains the creation of a U-Net model with pretrained ResNet50. It is possible to customize the number of trainable layers.

### DeepLabV3+ model Contstruction - `DeepLab/deeplab.py`:
This file is responsible for the creation of a DeepLabV3+ model with pre-trained ResNet50 backbone.

### Attention U-Net Construction - `Attention/attention_unet.py`:
Here the U-Net with attention gates is created. The architecture of the attention gates is created in the `Attention/attention_block.py` file.

### Models Training - `train.py`:
This script creates the batches using the `dataset_loader` file, performs preprocessing, performs augmentation, and then starts the training of the model performance.
It includes a Learning Rate schedule, an Early Stopping policy, and a TensorBoard logger.

### Loss Functions - `custom_loss.py`:
This file contains the experimented loss functions.

### GPU Check - `check_tf_gpu.py`:
This script checks whether a GPU is detected.

### Helpers - `helpers.py`: 
This script contains useful functions for conv-batch-activation modules, transposed conv-batchnorm-activation modules, model evaluation functions.

### Dataset Batch Loading - `dataset_loader.py`: 
Here the batch processing is handled. It includes data augmentation transformations, preprocessing strategy, and batch generation.