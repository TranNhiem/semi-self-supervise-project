'''
Auto Augmentation Policy Focus on Object Detection Task 
#Reference Implementation on Object Detection with 
1. AutoAugment 3 Policies (V0- V3)
 Barret, et al. Learning Data Augmentation Strategies for Object Detection.
    Arxiv: https://arxiv.org/abs/1906.11172
2. RandomAugment --> Also apply for object Detection Models

## Reference GitHub for implementation
[1] https://github.com/google/automl/blob/master/efficientdet/aug/autoaugment.py
[2] https://github.com/tensorflow/models/blob/master/official/vision/image_classification/augment.py
'''
# Current implementation will Deploy for Images WITHOUT BOX
import tensorflow as tf
from official.vision.image_classification.augment import AutoAugment


def tfa_AutoAugment(image):
    '''
    Args:
     image: A tensor [ with, height, channels]
     AutoAugment: a function to apply Policy transformation [v0, policy_simple]
    Return: 
      Image: A tensor of Applied transformation [with, height, channels]
    '''
    '''Version 1 RandAug Augmentation'''
    augmenter_apply = AutoAugment(augmentation_name='v0')
    image = augmenter_apply.distort(image)
    image = tf.cast(image, dtype=tf.float32)
    return image
