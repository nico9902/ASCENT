# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# user-defined functions
_EPSILON = 1e-08
def log(x):
    return torch.log(x + _EPSILON)

def div(x, y):
    return torch.div(x, (y + _EPSILON))

# create loss function
# LOSS-FUNCTION 1 -- Log-likelihood loss
def loss_Log_Likelihood(out, mask1, k):
    I_1 = torch.sign(k)

    #for uncenosred: log P(T=t,K=k|x)
    tmp1 = torch.sum(torch.sum(mask1 * out, dim=2), dim=1, keepdim=True)
    tmp1 = I_1 * log(tmp1)

    #for censored: log \sum P(T>t|x)
    tmp2 = torch.sum(torch.sum(mask1 * out, dim=2), dim=1, keepdim=True)
    tmp2 = (1. - I_1) * log(tmp2)

    loss1 = - torch.mean(tmp1 + 1.0*tmp2)

    return loss1

# LOSS-FUNCTION 2 -- Ranking loss
def loss_Ranking(out, num_Event, num_Category, mask2, t, k):
    
    sigma1 = 0.1
    eta = []
    for e in range(num_Event):
        one_vector = torch.ones_like(t, dtype=torch.float32)
        I_2 = torch.eq(k, e+1)
        I_2 = I_2.type(torch.float32) #indicator for event
        I_2 = torch.diag(torch.squeeze(I_2))
        tmp_e = torch.reshape(out[:,e,:], (-1, num_Category)) #event specific joint prob.
        #tmp_e = torch.reshape(tf.slice(out, [0, e, 0], [-1, 1, -1]), (-1, num_Category)) #event specific joint prob.

        R = torch.matmul(tmp_e, torch.transpose(mask2, 0, 1)) #no need to divide by each individual dominator
        # r_{ij} = risk of i-th pat based on j-th time-condition (last meas. time ~ event time) , i.e. r_i(T_{j})

        diag_R = torch.reshape(torch.diagonal(R,0), (-1, 1))
        R = torch.matmul(one_vector, torch.transpose(diag_R, 0, 1)) - R # R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
        R = torch.transpose(R, 0, 1)                                    # Now, R_{ij} (i-th row j-th column) = r_{i}(T_{i}) - r_{j}(T_{i})

        criterion = nn.ReLU()
        T = criterion(torch.sign(torch.matmul(one_vector, torch.transpose(t,0,1)) - torch.matmul(t, torch.transpose(one_vector,0,1))))
        # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j

        T = torch.matmul(I_2, T) # only remains T_{ij}=1 when event occured for subject i

        tmp_eta = torch.mean(T * torch.exp(-R/sigma1), dim=1, keepdim=True)

        eta.append(tmp_eta)
    eta = torch.stack(eta, dim=1) #stack referenced on subjects
    eta = torch.mean(torch.reshape(eta, (-1, num_Event)), dim=1, keepdim=True)

    loss2 = torch.sum(eta) #sum over num_Events
    return loss2

# # LOSS-FUNCTION 3 -- Calibration Loss
# def loss_Calibration(out, num_Event, num_Category, mask2, t, k):

#     eta = []
#     for e in range(num_Event):
#         one_vector = torch.ones_like(t, dtype=torch.float32)
#         I_2 = torch.eq(k, e+1)
#         I_2 = I_2.type(torch.float32) #indicator for event
#         tmp_e = torch.reshape(out[:,e,:], (-1, num_Category)) #event specific joint prob.

#         r = torch.sum(tmp_e * mask2, dim=0) #no need to divide by each individual dominator
#         tmp_eta = torch.mean((r - I_2)**2, dim=1, keepdim=True)

#         eta.append(tmp_eta)
#     eta = torch.stack(eta, dim=1) #stack referenced on subjects
#     eta = torch.mean(torch.reshape(eta, (-1, num_Event)), dim=1, keepdim=True)

#     loss3 = torch.sum(eta) #sum over num_Events
#     return loss3

class DeepHitLoss(nn.Module):

    def __init__(self, alpha, beta, num_Event, num_Category):
        super(DeepHitLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_Event = num_Event
        self.num_Category = num_Category
    
    def forward(self, preds, times, targets, mask1, mask2):
        loss = self.alpha*loss_Log_Likelihood(preds, mask1, targets) + self.beta*loss_Ranking(preds, self.num_Event, self.num_Category, mask2, times, targets) #+ gamma*loss_Calibration(out, num_Event, num_Category, mask2, t, k)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()