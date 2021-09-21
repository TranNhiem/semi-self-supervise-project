'''
RandAug implementation for Self-Supervised Learning
paper: https://arxiv.org/abs/1909.13719
RandAug with 2 Version 
Version 1 from Original Paper (14 Transformations ) 
Version 2 Modify with Adding Multiple Transformation (22 transformation from Imgaug API)
'''

import imgaug.augmenters as iaa
from official.vision.image_classification.augment import RandAugment
import tensorflow as tf
import numpy as np

'''Version 1 RandAug Augmentation'''


def tfa_randaug(image, num_transform, magnitude):
    '''
    Args:
     image: A tensor [ with, height, channels]
     augmenter: a function to apply Random transformation 
    Return: 
      Image: A tensor of Applied transformation [with, height, channels]
    '''
    '''Version 1 RandAug Augmentation'''
    augmenter_apply = RandAugment(
        num_layers=num_transform, magnitude=magnitude)
    image = augmenter_apply.distort(image)
    image = tf.cast(image, dtype=tf.float32)

    return image


'''Version2  RandAug Augmentation'''
# rand_aug = iaa.RandAugment(n=2, m=7)


def imgaug_randaug(images, num_transform, magnitude):
    '''
    Args:
     images: A batch tensor [batch, with, height, channels]
     rand_aug: a function to apply Random transformation 
    Return: 
      Images: A batch of Applied transformation [batch, with, height, channels]
    '''
    rand_aug_apply = iaa.RandAugment(n=num_transform, m=magnitude)

    # Input to `augment()` is a TensorFlow tensor which
    # is not supported by `imgaug`. This is why we first
    # convert it to its `numpy` variant.
    images = tf.cast(images, tf.uint8)
    images = rand_aug_apply(images=images.numpy())
    #images = (images.astype(np.float32))/255.
    images = tf.cast(images, tf.float32)/255.
    return images
