# import libraries
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def mask_slices(inputs, mask_percentage=0.2):
    """
        Mask a percentage of slices in the input tensor
    """
    masked_inputs = []
    for input in inputs:
        num_slices = input.shape[0]
        masked_input = input.clone()  

        num_slices_to_mask = int(num_slices * mask_percentage)
        masked_slices_idx = np.random.choice(num_slices, num_slices_to_mask, replace=False)
        masked_input[masked_slices_idx, ...] = 0  # Imposta le slice selezionate a zero

        masked_inputs.append(masked_input)

    return masked_inputs

class MyCollator(object):
    """
        Collate function for the DataLoader
    """
    def __init__(self, stage='train', mask_percentage=0.2):
        self.stage = stage
        self.mask_percentage = mask_percentage
    def __call__(self, batch):
        # inputs (pad the input sequences)
        inputs = [torch.stack(input) for input, _, _, _, _, _ in batch]
        
        if self.stage == 'train' and self.mask_percentage != 0:
            inputs, masked_slices = mask_slices(inputs, self.mask_percentage)
        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        # survival times
        survival_times = [survival_time for _, survival_time, _, _, _, _ in batch]
        survival_times = torch.tensor(survival_times, dtype=torch.float32)
        # targets
        targets = [target for _, _, target, _, _, _ in batch]
        targets = torch.tensor(targets, dtype=torch.float32)
        # mask1
        mask1 = [mask for _, _, _, mask, _, _ in batch]
        mask1 = torch.tensor(np.array(mask1), dtype=torch.float32)
        # mask2
        mask2 = [mask for _, _, _, _, mask, _ in batch]
        mask2 = torch.tensor(np.array(mask2), dtype=torch.float32)
        # pid
        pids = [pid for pid, _, _, _, _, _ in batch]

        return inputs, survival_times, targets, mask1, mask2, pids

class Convert(object):
    """
        Convert input to tensor.
        If the input is a list of images, convert each image to a tensor.
    """
    def __call__(self, img):
        return torch.unsqueeze(torch.from_numpy(np.array(img)), 0).float()