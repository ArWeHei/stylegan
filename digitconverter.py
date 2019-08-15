import os
import scipy.io as sio
from edflow.iterators.batches import resize_float32 as resize
from PIL import Image

import pandas as pd
import numpy as np
import tensorflow as tf

#data_dir = './svhn'
#out_dir = './svhn/image_align'
#data_path = os.path.join(data_dir, 'train_32x32.mat')
#data_train = sio.loadmat(data_path)
#data_train['y'][data_train['y']==[10]] = [0]
#
#df = pd.DataFrame(data_train['y'])
#print(df)
#
#im_shape = [32, 32]
#
#data_train['X'] = np.moveaxis(data_train['X'], -1, 0)
#
#if not os.path.exists(out_dir):
#    os.makedirs(out_dir)
#
#if isinstance(im_shape, int):
#    im_shape = [im_shape] * 2
#
#def preprocess(image):
#    image = image.astype(np.float32)
#    image = image / 255 *2 - 1.0
#    r = resize(image, im_shape)
#    return r
#
#for i, X in enumerate(data_train['X']):
#    print(X.shape)
#    image = X
#    image = preprocess(image)
#    out_path = os.path.join(out_dir, f'{i:06d}.jpg')
#    img = Image.fromarray(np.uint8((image+1)/2*255))
#    img.save(out_path)
#
#df.to_csv(data_dir+'/labels.csv')

data_dir = './mnist'
out_dir = './mnist/image_align'
data_train, data_test = tf.keras.datasets.mnist.load_data(
            path="mnist.npz"
        )

df = pd.DataFrame(data_train[1])
print(df)

im_shape = [32, 32]

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if isinstance(im_shape, int):
    im_shape = [im_shape] * 2

def preprocess(image):
    image = image.astype(np.float32)
    image = image / 255 *2 - 1.0
    r = resize(image, im_shape)
    return r

for i, X in enumerate(data_train[0]):
    print(X.shape)
    image = X
    image = preprocess(image)
    out_path = os.path.join(out_dir, f'{i:06d}.jpg')
    img = Image.fromarray(np.uint8((image+1)/2*255))
    img.save(out_path)

df.to_csv(data_dir+'/labels.csv')


