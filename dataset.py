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

def Portraits_w_Noise(config):
    p = lambda **kwargs: noise(config, **kwargs)
    return ProcessedDataset(PortraitsFromWikiArt(config), p)

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

def Merged_CelebAnPortraits(config):
    balanced = config.get('balanced_datasets', False)
    return MergedDataset(CelebA(config), PortraitsFromWikiArt(config), balanced=balanced)

def MergedCelebAnPortraits_w_Noise(config):
    p = lambda **kwargs: noise(config, **kwargs)
    return ProcessedDataset(Merged_CelebAnPortraits(config), p)


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
        example = {'image':img/255 *2 -1, 'painted':np.array([1, 0, 1]), 'feature_vec':features_vec, 'idx':idx}
        return example

    def get_empty_example(self):
        example = {'feature_vec':np.zeros(40)}
        return example

    @property
    def shape(self):
        return {'image':(128, 128, 3), 'painted':(3), 'feature_vec':(40), 'idx':(1)}

class PortraitsFromWikiArt(DatasetMixin):
    #returns 128x128 images from WikiartsPortraits
    def __init__(self, config):
        celeba_dir = config.get('wikiart_dir', './artworks')
        self.logger = get_logger(self)
        self.logger.info('Loading portraits from "%s"' % celeba_dir)
        glob_pattern = os.path.join(celeba_dir, 'portrait_align', '*.jpg')
        self.image_filepaths = sorted(glob.glob(glob_pattern))
        hdf_path = config.get('hdf', './artworks/art_faces_info.hdf5')
        self.attributes_df = pd.read_hdf(hdf_path), index_col='image_id')
        #if len(self.image_filepaths) != expected_images:
            #error('Expected to find %d images' % expected_images)

    def __name__(self):
        return "Art_Faces"

    def __len__(self):
        return len(self.image_filepaths)

    def get_example(self, idx):
        cx=109; cy=94
        image_filepath = self.image_filepaths[idx]
        image_filename =  image_filepath.split('/')[-1]
        image_filename =  image_filename.split('.')[0]
        idx_filename =  image_filename.split('-')[0]

        img = np.asarray(PIL.Image.open(image_filepath))
        features = self.attributes_df.loc[idx_filename,:].to_list()
        print(features)
        features_vec = np.array([val for val in features.values()])
        assert img.shape == (218, 218, 3)
        img = img[cy - 64 : cy + 64, cx - 64 : cx + 64]
        img = img.transpose(2, 0, 1) # HWC => CHW
        example = {'image':img/255 *2 -1, 'painted':np.array([0, 1, -1]), 'feature_vec':features_vec, 'idx':idx}
        return example

    def get_empty_example(self):
        example = {'feature_vec':np.zeros(1)}
        return example

class MergedDataset(DatasetMixin):
    """A dataset which merges given datasets."""

    def __init__(self, *datasets, balanced=False):
        self.datasets = list(datasets)
        self.lengths = [len(d) for d in self.datasets]
        self.boundaries = np.cumsum(self.lengths)
        self.balanced = balanced
        if self.balanced:
            max_length = np.max(self.lengths)
            for data_idx in range(len(self.datasets)):
                data_length = len(self.datasets[data_idx])
                if data_length != max_length:
                    cycle_indices = [i % data_length for i in range(max_length)]
                    self.datasets[data_idx] = SubDataset(
                        self.datasets[data_idx], cycle_indices
                    )
        self.lengths = [len(d) for d in self.datasets]
        self.boundaries = np.cumsum(self.lengths)
        self.feature_shape = 

   @property
   def shape(self):
        shapes = dict(self.datasets[0].shape)
        for i in range(1, len(self.datasets)):
            new_shapes = self.datasets[i].shape
            for k, v in shapes.keys():
                shapes[k] = shapes[k] + new_shapes[k]
                del new_shapes[k]
            shapes.update(new_shapes)

   def get_example(self, i):
        """Get example and add dataset index to it."""
        did = np.where(i < self.boundaries)[0][0]
        if did > 0:
            local_i = i - self.boundaries[did - 1]
        else:
            local_i = i
        example["dataset_index_"] = did

        for i, d in enumerate(self.datasets):
            if i == did:
                x = self.datasets[did][local_i]
            else:
                x = self.datasets[did].get_empty_example()

            for k, v in x.items():
                if k in example.keys():
                    example[k] = np.concatenate((example[k], v), axis=None)
                else:
                    example.update({k:v})

        return example


    def __len__(self):
        return sum(self.lengths)
