import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_func import *
from model import *
from train import *
from helpers import *

G_XtoY, G_YtoX, D_X, D_Y = create_model()
print_models(G_XtoY, G_YtoX, D_X, D_Y)

# hyperparams for Adam optimizers
lr= 0.0002
beta1= 0.5
beta2= 0.999
n_epochs = 1000 
g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters

dataloader_X, test_dataloader_X = get_data_loader(image_type='summer', image_dir = "summer2winter_yosemite")
dataloader_Y, test_dataloader_Y = get_data_loader(image_type='winter', image_dir = "summer2winter_yosemite")

# Create optimizers for the generators and discriminators
g_optimizer = optim.Adam(g_params, lr, [beta1, beta2])
d_x_optimizer = optim.Adam(D_X.parameters(), lr, [beta1, beta2])
d_y_optimizer = optim.Adam(D_Y.parameters(), lr, [beta1, beta2])


losses = training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, 
G_YtoX, G_XtoY, D_X, D_Y, d_x_optimizer, d_y_optimizer, g_optimizer, n_epochs=3000)