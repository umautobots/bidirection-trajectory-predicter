# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
import torch
import torchvision
from torchvision.transforms import functional as F
from bitrap.structures.bounding_box import BoxList
import pdb

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size=None, max_size=None, enforced_size=None):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.enforced_size = enforced_size
    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, images, target):
        '''
        images: a list of PIL Image object 
        '''
        if self.enforced_size is None:
            size = self.get_size(images.size)
            images = F.resize(images, size)
            # for i, img in enumerate(images):
            #     images[i] = F.resize(img, size)
        else:
            images = images.resize(self.enforced_size)
            # for i, img in enumerate(images):
                # images[i] = img.resize(self.enforced_size)

        return images, target

    def __str__(self):
        return 'Resize(): Min {} | Max {}'.format(self.min_size, self.max_size)


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            if isinstance(image, list):
                image = [F.hflip(img) for img in image]
                if isinstance(target, BoxList):
                    target = target.transpose(0)
            else:
                image = F.hflip(image)
                if isinstance(target, BoxList):
                    target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        if isinstance(image, list):
            image = [F.to_tensor(img) for img in image]
            image = torch.stack(image, dim=0)
        else:
            image = F.to_tensor(image)
        return image, target

class PointCrop(object):
    '''
    NOTE: crop the image centered at the defined pixel point
    '''
    def __init__(self, crop_size):
        if isinstance(crop_size, (tuple, list)):
            self.crop_w, self.crop_h = crop_size
        elif isinstance(crop_size, (int, float)):
            self.crop_w, self.crop_h = int(crop_size), int(crop_size)
        else:
            raise ValueError()

    def __call__(self, image, pt):
        '''img:'''
        t, l = int(pt[1] - self.crop_h), int(pt[0] - self.crop_w/2)
        image = F.crop(image, t, l, self.crop_h, self.crop_w)
        return image
        
class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        if self.mean is not None and self.std is not None:
            image = F.normalize(image, mean=self.mean, std=self.std)
        else:
            image = image*2 - 1
        return image, target