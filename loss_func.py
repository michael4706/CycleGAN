import torch
import torch.nn as nn
import torch.nn.functional as F

def real_mse_loss(D_out):
    # how close is the produced output from being "real"?
    return torch.mean((D_out - 1) ** 2)
    

def fake_mse_loss(D_out):
    # how close is the produced output from being "false"?
    return torch.mean(D_out ** 2)
    

def cycle_consistency_loss(real_im, reconstructed_im, lambda_weight):
    # calculate reconstruction loss 
    # return weighted loss
    return torch.mean(torch.abs(real_im-reconstructed_im)) * lambda_weight
    