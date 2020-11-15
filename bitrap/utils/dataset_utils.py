import numpy as np
import torch
import copy
import dill
import PIL
from PIL import Image

def restore(data):
    """
    In case we dilled some structures to share between multiple process this function will restore them.
    If the data input are not bytes we assume it was not dilled in the first place

    :param data: Possibly dilled data structure
    :return: Un-dilled data structure
    """
    if type(data) is bytes:
        return dill.loads(data)
    return data
    
def squarify(bbox, squarify_ratio, img_width):
    width = abs(bbox[0] - bbox[2])
    height = abs(bbox[1] - bbox[3])
    width_change = height * squarify_ratio - width
    # width_change = float(bbox[4])*self._squarify_ratio - float(bbox[3])
    bbox[0] = bbox[0] - width_change/2
    bbox[2] = bbox[2] + width_change/2
    # bbox[1] = str(float(bbox[1]) - width_change/2)
    # bbox[3] = str(float(bbox[3]) + width_change)
    # Squarify is applied to bounding boxes in Matlab coordinate starting from 1
    if bbox[0] < 0:
        bbox[0] = 0
    
    # check whether the new bounding box goes beyond image boarders
    # If this is the case, the bounding box is shifted back
    if bbox[2] > img_width:
        # bbox[1] = str(-float(bbox[3]) + img_dimensions[0])
        bbox[0] = bbox[0]-bbox[2] + img_width
        bbox[2] = img_width
    return bbox

def img_pad(img, mode = 'warp', size = 224):
    '''
    Pads a given image.
    Crops and/or pads a image given the boundries of the box needed
    img: the image to be coropped and/or padded
    bbox: the bounding box dimensions for cropping
    size: the desired size of output
    mode: the type of padding or resizing. The modes are,
        warp: crops the bounding box and resize to the output size
        same: only crops the image
        pad_same: maintains the original size of the cropped box  and pads with zeros
        pad_resize: crops the image and resize the cropped box in a way that the longer edge is equal to
        the desired output size in that direction while maintaining the aspect ratio. The rest of the image is
        padded with zeros
        pad_fit: maintains the original size of the cropped box unless the image is biger than the size in which case
        it scales the image down, and then pads it
    '''
    assert(mode in ['same', 'warp', 'pad_same', 'pad_resize', 'pad_fit']), 'Pad mode %s is invalid' % mode
    image = img.copy()
    if mode == 'warp':
        warped_image = image.resize((size,size),PIL.Image.NEAREST)
        return warped_image
    elif mode == 'same':
        return image
    elif mode in ['pad_same','pad_resize','pad_fit']:
        img_size = image.size  # size is in (width, height)
        ratio = float(size)/max(img_size)
        if mode == 'pad_resize' or  \
            (mode == 'pad_fit' and (img_size[0] > size or img_size[1] > size)):
            img_size = tuple([int(img_size[0]*ratio),int(img_size[1]*ratio)])
            image = image.resize(img_size, PIL.Image.NEAREST)
        padded_image = PIL.Image.new("RGB", (size, size))
        padded_image.paste(image, ((size-img_size [0])//2,
                    (size-img_size [1])//2))
        return padded_image

def bbox_to_goal_map(bbox, image_size=(1920,1080), target_size=(256,256)):
    '''
    Params:
        a future bbox in x1y1x2y2 format
    Return:
        score_map: (H, W)
    '''
    if isinstance(bbox, (np.ndarray, list)):
        bbox = torch.tensor(bbox)
    W, H = image_size
    WW, HH = target_size
    # resize box
    bbox = copy.deepcopy(bbox)
    bbox[[0, 2]] = bbox[[0, 2]] * WW / W
    bbox[[1, 3]] = bbox[[1, 3]] * HH / H
    bbox = bbox.type(torch.long)
    bbox[[0,2]] = torch.clamp(bbox[[0,2]], min=1, max=WW)
    bbox[[1,3]] = torch.clamp(bbox[[1,3]], min=1, max=HH)
    
    # enforce the box to be at list 1*1 size
    bbox[2] = max([bbox[2], bbox[0]+1])
    bbox[3] = max([bbox[3], bbox[1]+1])
    # Generate gaussian
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    sigma = torch.tensor([w, h])

    x_locs = torch.arange(0, w, 1).type(torch.float)
    y_locs = torch.arange(0, h, 1).type(torch.float)
    y_locs = y_locs[:, np.newaxis]

    x0 = w // 2
    y0 = h // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- (((x_locs - x0) ** 2)/sigma[0] + ((y_locs - y0) ** 2)/sigma[1]) / 2 )        
    
    # generate goal heat map
    goal_map = torch.zeros(WW, HH)
    goal_map[bbox[1]:bbox[3], bbox[0]:bbox[2]] += g
        
    return goal_map