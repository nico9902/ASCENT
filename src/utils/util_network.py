# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

def safe_softmax(x):
    # check if all elements in each sequence of the batch are -inf
    all_neg_inf = torch.all(x == -float('inf'), dim=1)
    
    # create a tensor of zeros for sequences where all elements are -inf
    result = torch.zeros_like(x)
    
    # calculate softmax only for sequences where not all elements are -inf
    if not torch.all(all_neg_inf):
        exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)  # Subtracting max for numerical stability
        result[~all_neg_inf] = exp_x[~all_neg_inf] / torch.sum(exp_x[~all_neg_inf], dim=1, keepdim=True)

    return result

def majority_voting(logits, mask):
    """
    Effettua il majority voting su un insieme di probabilità per ogni campione della batch.

    Parameters:
    - logits: Lista di probabilità delle varie slice. e.g. tensor [B,D]
    - mask: Maschera booleana che indica quali slice considerare nel majority voting per ciascun campione della batch.

    Returns:
    - Probabilità finale dopo il majority voting per ciascun campione della batch.
    """
    batch_size = logits.size(0)
    num_classes = logits.size(2)

    # determine device
    device = logits.device

    # initialize final votes tensor
    final_votes = torch.zeros(batch_size, num_classes, dtype=torch.float32, device=device)

    for i in range(batch_size):
        # filter out the masked logits
        masked_logits = logits[i][mask[i]]
        
        # compute the mean of the masked logits
        if len(masked_logits) > 0:
            final_votes[i] = torch.mean(masked_logits, axis=0)

    return final_votes

# feedforward network
def create_FCNet(input_dim, num_layers, h_dim, h_fn, o_dim, o_fn, limit=None, dropout=0.0, w_reg=None):
    '''
    GOAL             : Create FC network with different specifications 
    inputs (tensor)  : input tensor
    num_layers       : number of layers in FCNet
    h_dim  (int)     : number of hidden units
    h_fn             : activation function for hidden layers (default: tf.nn.relu)
    o_dim  (int)     : number of output units
    o_fn             : activation function for output layers (defalut: None)
    w_init           : initialization for weight matrix (defalut: Xavier)
    keep_prob        : keep probabilty [0, 1]  (if None, dropout is not employed)
    '''
    # default active functions (hidden: relu, out: None)
    if h_fn is None:
        h_fn = nn.ReLU()
    if o_fn is None:
        o_fn = None
    
    layers = []
    if num_layers == 1:
        fc = nn.Linear(input_dim, o_dim)
        if not limit is None:
            nn.init.uniform_(fc.weight, a=-limit, b=limit)
        else:
            nn.init.xavier_normal_(fc.weight)
        layers.append(fc)
        if not o_fn is None:
            layers.append(o_fn)
        return nn.Sequential(*layers)
    else:
        fc1 = nn.Linear(input_dim, h_dim)
        if not limit is None:
            nn.init.uniform_(fc1.weight, a=-limit, b=limit)
        else:
            nn.init.xavier_normal_(fc1.weight)
        layers.append(fc1)
        layers.append(h_fn)
        if not dropout is None:
            layers.append(dropout)

        for _ in range(1, num_layers-1):
            fc2 = nn.Linear(h_dim, h_dim)
            if not limit is None:
                nn.init.uniform_(fc2.weight, a=-limit, b=limit)
            else:
                nn.init.xavier_normal_(fc2.weight)
            layers.append(fc2)
            layers.append(h_fn)
            if not dropout is None:
                layers.append(dropout)

        fc3 = nn.Linear(h_dim, o_dim)
        if not limit is None:
            nn.init.uniform_(fc3.weight, a=-limit, b=limit)
        else:
            nn.init.xavier_normal_(fc3.weight)
        layers.append(fc3)
        if not o_fn is None:
            layers.append(o_fn)
        return nn.Sequential(*layers)
