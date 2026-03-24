# import libraries
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import numpy as np
import torch
import os
import cv2  
from PIL import Image

import src.utils.util_data as util_data


### MASK FUNCTIONS
'''
    fc_mask2      : To calculate LOSS_1 (log-likelihood loss)
    fc_mask3      : To calculate LOSS_2 (ranking loss)
'''
def f_get_fc_mask2(time, label, num_Event, num_Category):
    '''
        mask4 is required to get the log-likelihood loss
        mask4 size is [N, num_Event, num_Category]
            if not censored : one element = 1 (0 elsewhere)
            if censored     : fill elements with 1 after the censoring time (for all events)
    '''
    mask = np.zeros([np.shape(time)[0], num_Event, num_Category]) # for the first loss function
    for i in range(np.shape(time)[0]):
        if label[i] != 0:  # not censored
            mask[i,int(label[i]-1),int(time[i])] = 1
        else: # label[i,2]==0: censored
            mask[i,:,int(time[i]+1):] =  1 # fill 1 until from the censoring time (to get 1 - \sum F)
    return mask


def f_get_fc_mask3(time, meas_time, num_Category):
    '''
        mask5 is required calculate the ranking loss (for pair-wise comparision)
        mask5 size is [N, num_Category].
        - For longitudinal measurements:
             1's from the last measurement to the event time (exclusive and inclusive, respectively)
             denom is not needed since comparing is done over the same denom
        - For single measurement:
             1's from start to the event time(inclusive)
    '''
    mask = np.zeros([np.shape(time)[0], num_Category]) # for the first loss function
    if np.shape(meas_time):  # lonogitudinal measurements
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i]) # last measurement time
            t2 = int(time[i]) # censoring/event time
            mask[i,(t1+1):(t2+1)] = 1  #this excludes the last measurement time and includes the event time
    else:                    # single measurement
        for i in range(np.shape(time)[0]):
            t = int(time[i]) # censoring/event time
            mask[i,:(t+1)] = 1  # this excludes the last measurement time and includes the event time
    return mask

def import_mask(time, label, num_Event, num_Category):
    '''
        mask1 is required to get the log-likelihood loss
        mask2 is required calculate the ranking loss (for pair-wise comparision)
    '''
    mask1 = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask2 = f_get_fc_mask3(time, -1, num_Category)

    return mask1, mask2

### DATASET CLASS
class ImgDataset(torch.utils.data.Dataset):
    '''
        Dataset class for the image data
    '''
    def __init__(self, patientIDs, cfg_data, time, label, augmentation, step,
                 num_Event=None, num_Category=None, transform=None, mu=None, 
                 std=None):
        self.patientIDs = patientIDs
        self.cfg_data = cfg_data
        self.times = time
        self.labels = label
        self.augmentation = augmentation
        self.clip = cfg_data["clip"]
        self.scale = cfg_data["scale"]
        self.img_dim = cfg_data["img_dim"]
        self.step = step
        self.transform = transform

        # Build an index of slice paths per patient (no data loaded yet)
        self._paths = [self._list_slices(pid) for pid in self.patientIDs]
        self.survival_times = list(self.times)
        self.targets = list(self.labels)

        # Stats: either use provided mu/std or compute (streaming) on a subset for train
        self.mu = mu
        self.std = std
        if step=='train' and (self.mu is None or self.std is None):
            self.mu, self.std = self._compute_mean_std_streaming()

        # Precompute masks for DeepHit (if requested)
        self.masks1 = None
        self.masks2 = None
        if num_Event is not None and num_Category is not None:
            self.masks1, self.masks2 = import_mask(
                time=self.survival_times, label=self.targets,
                num_Event=num_Event, num_Category=num_Category
            )

    def _list_slices(self, patient_id):
        pdir = os.path.join(self.cfg_data['data_dir'], str(patient_id))
        # robust listing, ignore hidden files
        elements = [f for f in os.listdir(pdir) if not f.startswith(".")]
        elements.sort()
        return [os.path.join(pdir, f) for f in elements]

    def _compute_mean_std_streaming(self):
        # Sample a subset for speed
        sel_patients = self._paths[:len(self._paths)]
        total = 0
        s = 0.0
        ss = 0.0
        for paths in sel_patients:
            for p in paths[:len(paths)]:
                x = np.load(p)  # keep as np.ndarray
                x = x.astype(np.float32, copy=False)
                s += float(x.sum())
                ss += float((x * x).sum())
                total += x.size
        eps = 1e-8
        mu = s / max(total, 1)
        var = ss / max(total, 1) - mu * mu
        std = float(np.sqrt(max(var, eps)))
        return float(mu), std

    def __len__(self):
        return len(self.patientIDs)

    def __getitem__(self, index):
        # Load all slices for this patient lazily
        paths = self._paths[index]
        # NOTE: if memory spikes, consider mmap_mode='r'
        slices = [cv2.resize(np.load(p), (224, 224)) for p in paths]
        sample = np.stack(slices, axis=0)  # [S, H, W] or similar

        # Apply your existing preprocessing pipeline
        sample = util_data.loader(sample, self.img_dim,
                                  clip=self.clip, scale=self.scale,
                                  step=self.step, aug=self.augmentation)

        # Optional per-slice transforms (e.g., augment + normalize)
        if self.transform:
            sample = [self.transform(Image.fromarray(x)) for x in sample]

        survival_time = self.survival_times[index]
        target = self.targets[index]

        if self.masks1 is not None and self.masks2 is not None:
            mask1 = self.masks1[index]
            mask2 = self.masks2[index]
        else:
            mask1 = 0
            mask2 = 0

        pid = self.patientIDs[index]
        return sample, survival_time, target, mask1, mask2, pid