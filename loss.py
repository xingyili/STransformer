import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MSE_loss = nn.MSELoss()

def calculate_loss(expression, recon_expression, tile, recon_tile):
    recon_loss_exp = MSE_loss(expression, recon_expression)
    recon_loss_tile = MSE_loss(tile, recon_tile) #mouse_brain

    cosine_loss_exp = 1 - F.cosine_similarity(expression, recon_expression, dim=-1).mean()
    cosine_loss_tile = 1 - F.cosine_similarity(expression, recon_expression, dim=-1).mean()

    return recon_loss_exp, recon_loss_tile, cosine_loss_exp, cosine_loss_tile