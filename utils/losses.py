import torch
from torch.autograd import Function
import numpy as np
from sklearn.metrics import jaccard_score,f1_score


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

# IOU
SMOOTH = 1e-6

def iou_pytorch(pred: torch.Tensor, gt: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape

    #print("##Pred = ", pred)
    #print("##Pred - shape=", pred.shape)
    #print("##gt =", gt)
    #print("##gt.shape=", gt.shape)

    pred = pred.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    gt = gt.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    pred = pred.int()
    gt = gt.int()

    intersection = (pred & gt).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (pred | gt).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    #thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    #print("iou pytorch=", iou)
    
    return iou #thresholded  # Or thresholded.mean() if you are interested in average across the batch
    
    
# Numpy version
# Well, it's the same function, so I'm going to omit the comments

def iou_numpy(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze(1)
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return thresholded  # Or thresholded.mean()

# Added by Vajira Thambawit
def iou_sklearn(pred: torch.Tensor, gt: torch.Tensor, avg_type: str):
    #print("Pred = ", pred)
   # print("Pred - shape=", pred.shape)
   # print("gt =", gt)
   # print("gt.shape=", gt.shape)

    # convert 1x1xHxW into HxW
    pred = pred.reshape((pred.shape[2], pred.shape[3]))
    gt = gt.reshape((gt.shape[2], gt.shape[3])) 

    # convert into int
    pred = pred.int()
    gt = gt.int()

    # convert inot numpy
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()

    iou = jaccard_score(gt, pred, average=avg_type)
    #print("IOU skalearn=", iou)
    return iou

# Added by Vajira Thambawit
def dice_using_sklearn(pred: torch.Tensor, gt: torch.Tensor):
    #print("Pred = ", pred)
    #print("Pred - shape=", pred.shape)
    #print("gt =", gt)
    #print("gt.shape=", gt.shape)

    # convert 1x1xHxW into HxW
    pred = pred.reshape((pred.shape[2], pred.shape[3]))
    gt = gt.reshape((gt.shape[2], gt.shape[3])) 

    # convert into int
    pred = pred.int()
    gt = gt.int()

    # flat 2d to 1d
    pred = pred.reshape(-1)
    gt = pred.reshape(-1)

    # convert inot numpy
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()


    dice = f1_score(gt, pred)
    #print("IOU skalearn=", iou)
    return dice 