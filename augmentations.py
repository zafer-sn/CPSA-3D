import torch
import numpy as np
from torchvision.transforms.functional import erase

class SilhouetteAugmentation(object):
    """
    This augmentation blacks out the RGB channels of an image tensor, 
    leaving only the alpha channel intact. It forces the model to learn 
    from the silhouette of the object.
    The original implementation incorrectly used parameters for random erasing.
    This version is simplified to reflect its actual behavior: erasing the whole image.
    """
    def __init__(self, p=0.5, value=0, inplace=False):
        """
        Args:
            p (float): probability of applying the augmentation.
            value (int): the value to erase the channels with (usually 0 for black).
            inplace (bool): whether to perform the operation in-place.
        """
        self.p = p
        self.value = value
        self.inplace = inplace

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be augmented.
                          It is expected to be the RGB channels portion.
        Returns:
            Tensor: Augmented image.
        """
        if np.random.uniform(0, 1) > self.p:
            return img

        # Erase the entire image (all channels passed to it)
        return erase(img, 0, 0, img.size(1), img.size(2), v=self.value, inplace=self.inplace)