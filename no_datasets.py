from edflow.data.dataset import DatasetMixin, ProcessedDataset, ConcatenatedDataset, CachedDataset, cachable, SubDataset
from edflow.custom_logging import get_logger
from edflow.util import pprint
from edflow.iterators.batches import resize_float32 as resize

import PIL
import os
import glob

import numpy as np
import pandas as pd
import scipy.io as sio

def MNIST_w_noise(config):
    p = lambda **kwargs: noise(config, **kwargs)
    return ProcessedDataset(CelebA(config), p)

def SVHN_w_Noise(config):
    p = lambda **kwargs: noise(config, **kwargs)
    return ProcessedDataset(PortraitsFromWikiArt(config), p)

def noise(config, **kwargs):
    latent_size = config.get('latent_size', 512) # Latent vector (Z) dimensionality.
    latents = np.random.randn(1, latent_size)
    return {'latent':latents[0], **kwargs}

def MNISTnSVHN(config):
    balanced = config.get('balanced_datasets', False)
    return ConcatenatedDataset(CelebA(config), PortraitsFromWikiArt(config), balanced=balanced)

def MNISTnSVHN_w_Noise(config):
    p = lambda **kwargs: noise(config, **kwargs)
    return ProcessedDataset(CelebAnPortraits(config), p)

class MNIST(DatasetMixin):
    def __init__(self, config):
        self.config = config

        data_dir = config.get('mnist_dir', './mnist')
        data_path = os.path.join(data_dir, "mnist.npz")
        # Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        self.data_train, self.data_test = tf.keras.datasets.mnist.load_data(
            path=data_path
        )
        self.im_shape = config.get(
            "spatial_size", [32, 32]
        )  # if not found, default value
        if isinstance(self.im_shape, int):
            self.im_shape = [self.im_shape] * 2

    def preprocess(self, image):
        image = image.astype(np.float32)
        image = image /255 *2 - 1.0
        r = resize(image, self.im_shape)
        #include conversion to rgb image
        return np.expand_dims(r, -1)

    def get_example(self, idx):
        example = dict()

        image = self.data_train[0][idx]
        class_ = self.data_train[1][idx]
        example["image"] = self.preprocess(image)
        example["target"] = class_
        example["painted"] = np.array([1,0,-1])
        return example

    def __len__(self):
        return len(self.data_train[0])


class SVHN(DatasetMixin):
    def __init__(self, config):
        self.config = config

        data_dir = config.get('svhn_dir', './svhn')
        data_path = os.path.join(data_dir, 'train_32x32.mat')
        self.data = sio.loadmat(data_path)
        pprint(self.data)

        self.im_shape = config.get(
            "spatial_size", [32, 32]
        )  # if not found, default value
        if isinstance(self.im_shape, int):
            self.im_shape = [self.im_shape] * 2

    def preprocess(self, image):
        image = image.astype(np.float32)
        image = image / 255 *2 - 1.0
        r = resize(image, self.im_shape)
        return np.expand_dims(r, -1)

    def get_example(self, idx):
        example = dict()

        image = self.data_train['X'][:,:,:,idx]
        class_ = self.data_train['y'][idx]
        example["image"] = self.preprocess(image)
        example["target"] = class_
        example["painted"] = np.array([0,1,-1])
        return example

    def __len__(self):
        return len(self.data_train[0])


