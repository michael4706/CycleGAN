import torch
import torch.nn as nn
import torch.nn.functional as F

# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    
    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()

        # Define all convolutional layers
        # Should accept an RGB image as input and output a single value
        
        #128x128x3 --> (128-4+2)/2+1 = 64 --> 64x64x64 
        self.conv1 = conv(in_channels = 3, out_channels = conv_dim, kernel_size = 4, 
                          batch_norm = False)
        
        #64x64x64 --> (64-4+2)/2+1 = 32 --> 32x32x128
        self.conv2 = conv(in_channels = conv_dim, out_channels = conv_dim * 2, kernel_size = 4, 
                          batch_norm = True)
        
        #32x32x128 --> (32-4+2)/2+1 = 16 --> 16x16x256
        self.conv3 = conv(in_channels = conv_dim * 2, out_channels = conv_dim * 4, kernel_size = 4, 
                          batch_norm = True)
        
        #16x16x256 --> (16-4+2)/2+1 = 8 --> 8x8x512
        self.conv4 = conv(in_channels = conv_dim * 4, out_channels = conv_dim * 8, kernel_size = 4, 
                          batch_norm = True)
        
        #8x8x512 --> (8-8+0)/1+1 = 1 --> 1x1x1
        
        
        self.conv5 = conv(in_channels = conv_dim * 8, out_channels = 1, kernel_size = 4, stride = 1,
                          batch_norm = False)
        
        

    def forward(self, x):
        # define feedforward behavior
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        
        return x

class ResidualBlock(nn.Module):
    """Defines a residual block.
       This adds an input x to a convolutional layer (applied to x) with the same size input and output.
       These blocks allow a model to learn an effective transformation from one domain to another.
    """
    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        # conv_dim = number of inputs
        self.conv1 = conv(in_channels = conv_dim, out_channels = conv_dim, kernel_size = 3, stride = 1,
                          padding = 1,batch_norm = True)
        self.conv2 = conv(in_channels = conv_dim, out_channels = conv_dim, kernel_size = 3, stride = 1,
                          padding = 1,batch_norm = True)
        
        
    def forward(self, x):
        res_x = F.relu(self.conv1(x))
        res_x = self.conv2(res_x)
        
        return x + res_x

# helper deconv function
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transpose convolutional layer, with optional batch normalization.
    """
    layers = []
    # append transpose conv layer
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    # optional batch norm layer
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

class CycleGenerator(nn.Module):
    
    def __init__(self, conv_dim=64, n_res_blocks=6):
        super(CycleGenerator, self).__init__()

        # 1. Define the encoder part of the generator
        
        #128x128x3 --> (128-4+2)/2+1 = 64 --> 64x64x64
        #64x64x64 --> (64-4+2)/2+1 = 32 --> 32x32x128
        #32x32x128 --> (32-4+2)/2+1 = 16 --> 16x16x256

        self.encoder = nn.Sequential(conv(in_channels = 3, out_channels = conv_dim, kernel_size = 4,
                                     batch_norm = True),
                            nn.ReLU(),
                            conv(in_channels = conv_dim, out_channels = conv_dim * 2, kernel_size = 4,
                                 batch_norm = True),
                            nn.ReLU(),
                            conv(in_channels = conv_dim * 2, out_channels = conv_dim * 4, kernel_size = 4,
                                 batch_norm = True),
                            nn.ReLU()
                            )
        res_layers = []
        for i in range(n_res_blocks):
            res_layers.append(ResidualBlock(conv_dim*4))
        
        self.resblock = nn.Sequential(*res_layers)
        
        #(16-1)*2-2+4 = 32x32x128
        #(32-1)*2-2+4 = 64x64x64
        #(32-1)*2-2+4 = 128x128x3
        
        self.decoder = nn.Sequential(deconv(in_channels = conv_dim*4, out_channels = conv_dim * 2, kernel_size = 4,
                                            batch_norm = True),
                                    nn.ReLU(),
                                    deconv(in_channels = conv_dim*2, out_channels = conv_dim, kernel_size = 4,
                                            batch_norm = True),
                                    nn.ReLU(),
                                    deconv(in_channels = conv_dim, out_channels = 3, kernel_size = 4,
                                           batch_norm = False),
                                    nn.Tanh()
                                    )
        
    def forward(self, x):
        """Given an image x, returns a transformed image."""
        # define feedforward behavior, applying activations as necessary
        encode_out = self.encoder(x)
        res_out = self.resblock(encode_out)
        decode_out = self.decoder(res_out)

        return decode_out


def create_model(g_conv_dim=64, d_conv_dim=64, n_res_blocks=6):
    """Builds the generators and discriminators."""
    
    # Instantiate generators
    G_XtoY = CycleGenerator(conv_dim = g_conv_dim, n_res_blocks = n_res_blocks)
    G_YtoX = CycleGenerator(conv_dim = g_conv_dim, n_res_blocks = n_res_blocks)
    # Instantiate discriminators
    D_X = Discriminator(conv_dim = d_conv_dim)
    D_Y = Discriminator(conv_dim = d_conv_dim)

    # move models to GPU, if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        G_XtoY.to(device)
        G_YtoX.to(device)
        D_X.to(device)
        D_Y.to(device)
        print('Models moved to GPU.')
    else:
        print('Only CPU available.')

    return G_XtoY, G_YtoX, D_X, D_Y


def print_models(G_XtoY, G_YtoX, D_X, D_Y):
    """Prints model information for the generators and discriminators.
    """
    print("                     G_XtoY                    ")
    print("-----------------------------------------------")
    print(G_XtoY)
    print()

    print("                     G_YtoX                    ")
    print("-----------------------------------------------")
    print(G_YtoX)
    print()

    print("                      D_X                      ")
    print("-----------------------------------------------")
    print(D_X)
    print()

    print("                      D_Y                      ")
    print("-----------------------------------------------")
    print(D_Y)
    print()
    

