from edflow.data.dataset import DatasetMixin, ProcessedDataset, ConcatenatedDataset, CachedDataset, cachable
from edflow.custom_logging import get_logger
from edflow.util import pprint

import PIL
import os
import glob

import numpy as np
import pandas as pd

def CelebA_w_Noise(config):
    p = lambda **kwargs: noise(config, **kwargs)
    return ProcessedDataset(CelebA(config), p)

def noise(config, **kwargs):
    latent_size = config.get('latent_size', 512) # Latent vector (Z) dimensionality.
    latents = np.random.randn(1, latent_size)
    return {'latent':latents[0], **kwargs}

def CelebAnPortraits(config):
    balanced = config.get('balanced_datasets', False)
    return ConcatenatedDataset(CelebA(config), PortraitsFromWikiArt(config), balanced=balanced)

def CelebAnPortraits_w_Noise(config):
    p = lambda **kwargs: noise(config, **kwargs)
    return ProcessedDataset(CelebAnPortraits(config), p)

@cachable('../CelebAnPortraits/cached/Dataset.zip')
def CelebAnPortraitsCached(config):
    return CelebAnPortraits_w_Noise(config)

def CelebAnPortraits_mirrored(config):
    double_set = ConcatenatedDataset(CelebAnPortraits(config), CelebAnPortraits(config))
    p = lambda **kwargs: mirror(config, double_set, **kwargs)
    return ProcessedDataset(double_set, p)

def mirror(config, dataset, **kwargs):
    arg = config.get('mirror_arg', image)
    doubled_len = len(dataset)
    if kwargs['idx'] >= int(doubled_len/2):
        kwargs['arg'] = np.flip(kwargs['arg'], 2)
    return kwargs




class CelebA(DatasetMixin):
    #returns 128x128 images from CelebA
    def __init__(self, config):
        celeba_dir = config.get('celeba_dir', './celeba-dataset')
        self.logger = get_logger(self)
        self.logger.info('Loading CelebA from "%s"' % celeba_dir)
        glob_pattern = os.path.join(celeba_dir, 'img_align_celeba', '*.jpg')
        self.image_filepaths = sorted(glob.glob(glob_pattern))
        self.attributes_df = pd.read_csv(os.path.join(celeba_dir, 'list_attr_celeba.csv'), index_col='image_id')
        expected_images = 202599
        if len(self.image_filepaths) != expected_images:
            error('Expected to find %d images' % expected_images)

    def __name__(self):
        return "CelebA"

    def __len__(self):
        return len(self.image_filepaths)


    def get_example(self, idx):
        cx=89; cy=121
        image_filepath = self.image_filepaths[idx]
        image_filename =  image_filepath.split('/')[-1]

        img = np.asarray(PIL.Image.open(image_filepath))
        features = self.attributes_df.loc[image_filename,:].to_dict()
        features_vec = np.array([val for val in features.values()])
        features_vec = []
        assert img.shape == (218, 178, 3)
        img = img[cy - 64 : cy + 64, cx - 64 : cx + 64]
        img = img.transpose(2, 0, 1) # HWC => CHW
        example = {'image':img/255 *2 -1, 'painted':np.array([1, 0]), 'feature_vec':features_vec, 'idx':idx}
        return example

class PortraitsFromWikiArt(DatasetMixin):
    #returns 128x128 images from WikiartsPortraits
    def __init__(self, config):
        celeba_dir = config.get('wikiart_dir', './artworks')
        self.logger = get_logger(self)
        self.logger.info('Loading portraits from "%s"' % celeba_dir)
        glob_pattern = os.path.join(celeba_dir, 'portrait_align', '*.jpg')
        self.image_filepaths = sorted(glob.glob(glob_pattern))
        #self.attributes_df = pd.read_csv(os.path.join(celeba_dir, 'wikiart.csv'), index_col='id', usecols=['id'])
        expected_images = 784
        #if len(self.image_filepaths) != expected_images:
            #error('Expected to find %d images' % expected_images)

    def __name__(self):
        return "WikiArtPortraits"

    def __len__(self):
        return len(self.image_filepaths)

    def get_example(self, idx):
        cx=109; cy=94
        image_filepath = self.image_filepaths[idx]
        image_filename =  image_filepath.split('/')[-1]
        image_filename =  image_filename.split('.')[0]

        img = np.asarray(PIL.Image.open(image_filepath))
        #features = self.attributes_df.loc[image_filename,:].to_dict()
        #features_vec = np.array([val for val in features.values()])
        features_vec = []
        assert img.shape == (218, 218, 3)
        img = img[cy - 64 : cy + 64, cx - 64 : cx + 64]
        img = img.transpose(2, 0, 1) # HWC => CHW
        example = {'image':img/255 *2 -1, 'painted':np.array([0, 1]), 'feature_vec':features_vec, 'idx':idx}
        return example
