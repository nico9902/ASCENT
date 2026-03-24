from PIL import Image
import cv2
import math
import torch
import os
import random
import numpy as np

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_img(img_path):
    filename, extension = os.path.splitext(img_path)
    img = Image.open(img_path)
    img = np.array(img).astype(float)
    return img

def get_mask(img, mask, value=1):
    mask = mask != value
    img[mask] = 0
    return img

def get_box(img, box, perc_border=.0):
    # Sides
    l_h = box[2] - box[0]
    l_w = box[3] - box[1]
    # Border
    diff_1 = math.ceil((abs(l_h - l_w) / 2))
    diff_2 = math.floor((abs(l_h - l_w) / 2))
    border = int(perc_border * diff_1)
    # Img dims
    img_h = img.shape[0]
    img_w = img.shape[1]
    if l_h > l_w:
        if box[0]-border < 0:
            pad = 0-(box[0]-border)
            img = np.vstack([np.zeros((pad, img.shape[1])), img])
            box[0] += pad
            box[2] += pad
            img_h += pad
        if box[2]+border > img_h:
            pad = (box[2]+border)-img_h
            img = np.vstack([img, np.zeros((pad, img.shape[1]))])
        if box[1]-diff_1-border < 0:
            pad = 0-(box[1]-diff_1-border)
            img = np.hstack([np.zeros((img.shape[0], pad)), img])
            box[1] += pad
            box[3] += pad
            img_w += pad
        if box[3]+diff_2+border > img_w:
            pad = (box[3]+diff_2+border)-img_w
            img = np.hstack([img, np.zeros((img.shape[0], pad))])
        img = img[box[0]-border:box[2]+border, box[1]-diff_1-border:box[3]+diff_2+border]
    elif l_w > l_h:
        if box[0]-diff_1-border < 0:
            pad = 0-(box[0]-diff_1-border)
            img = np.vstack([np.zeros((pad, img.shape[1])), img])
            box[0] += pad
            box[2] += pad
            img_h += pad
        if box[2]+diff_2+border > img_h:
            pad = (box[2]+diff_2+border)-img_h
            img = np.vstack([img, np.zeros((pad, img.shape[1]))])
        if box[1]-border < 0:
            pad = 0-(box[1]-border)
            img = np.hstack([np.zeros((img.shape[0], pad)), img])
            box[1] += pad
            box[3] += pad
            img_w += pad
        if box[3]+border > img_w:
            pad = (box[3]+border)-img_w
            img = np.hstack([img, np.zeros((img.shape[0], pad))])
        img = img[box[0]-diff_1-border:box[2]+diff_2+border, box[1]-border:box[3]+border]
    else:
        if box[0]-border < 0:
            pad = 0-(box[0]-border)
            img = np.vstack([np.zeros((pad, img.shape[1])), img])
            box[0] += pad
            box[2] += pad
            img_h += pad
        if box[2]+border > img_h:
            pad = (box[2]+border)-img_h
            img = np.vstack([img, np.zeros((pad, img.shape[1]))])
        if box[1]-border < 0:
            pad = 0-(box[1]-border)
            img = np.hstack([np.zeros((img.shape[0], pad)), img])
            box[1] += pad
            box[3] += pad
            img_w += pad
        if box[3]+border > img_w:
            pad = (box[3]+border)-img_w
            img = np.hstack([img, np.zeros((img.shape[0], pad))])
        img = img[box[0]-border:box[2]+border, box[1]-border:box[3]+border]
    return img

def loader(img, img_dim, mask_path=False, box=False, clip=False, scale=False, step="train", aug=False):
    """
        Load and preprocess image
    """
    loaded_img = []  
    # iterate over slices
    for slice in img:
        slice = np.array(slice).astype(float)
        # to Grayscale
        if slice.ndim > 2:
            slice = slice.mean(axis=2)
        # filter Mask
        if mask_path:
            mask = load_img(mask_path)
            slice = get_mask(slice, mask, value=1)
        # select Box Area
        if box:
            slice = get_box(slice, box, perc_border=0.5)
        # resize
        slice = cv2.resize(slice, (img_dim, img_dim))
        # clip
        if clip:
            slice = np.clip(slice, slice['min'], slice['max'])

        # collect slices
        loaded_img.append(slice)

    return loaded_img
