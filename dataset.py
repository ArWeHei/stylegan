from edflow.data.dataset import DatasetMixin, ProcessesDataset
from edflow.custom_logging import get_logger
from edflow.util import pprint

import PIL
import os
import glob

import numpy as np
import pandas as pd

def CelebA_w_Noise(config)
    return ProcessesDataset(CelebA(config), noise_process)

def noise_process(config):
    latent_size = config.get('latent_size', 512) # Latent vector (Z) dimensionality.
    latents = np.random.randn(1, latent_size)
    return {'latent':latent}
    
 

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
        assert img.shape == (218, 178, 3)
        img = img[cy - 64 : cy + 64, cx - 64 : cx + 64]
        img = img.transpose(2, 0, 1) # HWC => CHW
        example = {'image':img, 'features':features, 'feature_vec':features_vec}
        return example
