import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import torch.backends.cudnn as cudnn

def slicegan_net(pth, imtype='twophase', df=None, gf=None):
    """define networks for slicegan

    Args:
        pth (str): path to project
        training (Bool): Training or loading nets
        imtype (str): nphase or continuous data
        df (tupe of ints, optional): Filters for the discriminator network. Defaults to None.
        gf (tuple of ints, optional): Filters for generator layers. Defaults to None.

    Returns:
        tuple: Discriminator and Generator
    """
    
    # if fresh training, save params
    with open(pth + '_params.data', 'rb') as filehandle:
        # read the data as binary data stream
        df, gf = pickle.load(filehandle)


    class Generator(nn.Module):
        """Create generator class according to filters given or loaded
        type:"""
        def __init__(self):
            super(Generator, self).__init__()
            self.lays = len(gf)-1
            self.tconvs1 = nn.ModuleList()
            self.tconvs2 = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
            self.relu = nn.LeakyReLU(0.2)
            self.imtype = imtype
            self.out_act = nn.Softmax(1) if 'phase' in imtype else nn.Sigmoid()
            for lay in range(self.lays):
                if lay < self.lays - 1:
                    self.tconvs1.append(nn.ConvTranspose3d(gf[lay], gf[lay+1] - 1, 2, 1, 1, bias=False))
                    self.tconvs2.append(nn.ConvTranspose3d(gf[lay+1] - 1, gf[lay+1] - 1, 2, 1, 1, bias=False))
                else:
                    self.tconvs1.append(nn.ConvTranspose3d(gf[lay], gf[lay+1], 2, 1, 1, bias=False))
                    self.tconvs2.append(nn.ConvTranspose3d(gf[lay+1], gf[lay+1], 2, 1, 1, bias=False))
                self.bns.append(nn.BatchNorm3d(gf[lay+1] - 1))


        def forward(self, x, threed, slice_dim):
            """Forward pass for the generator

            Args:
                x (torch.Tensor): Input noise
                dim (bool, optional): the dim to crop if training. Defaults to False.
                to be held constant during gen. Defaults to False.

            Returns:
                torch.tensor: batch of generated microstructures
            """
            dim = 0 if threed else slice_dim + 2
            x = self.crop(x, dim)
            for i, (tconv1, tconv2) in enumerate(zip(
                self.tconvs1[:-1], self.tconvs2[:-1])):
                x = self.relu(tconv2(tconv1(x)))
                x = self.up(x)
                x = x[...,1:-1,1:-1,1:-1]
                noise_shape = list(x.size())
                noise_shape[1] = 1
                x = torch.cat((x, torch.randn(size=noise_shape, device=x.device)) , dim=1)
                
            x = self.out_act(self.tconvs2[-1](self.tconvs1[-1](x)))
            return self.crop(x, dim, final=True)


        def crop(self, x, dim, final=False, all_slices=False):
            """Croping of intermediate layers during generation

            Args:
                x (torch.Tensor): layer output 
                dim (int): which dim to
                final (bool, optional): Whether its the final output layer. Defaults to False.

            Returns:
                torch.tensor: cropped input
            """
            if not dim:
                return x
            if not final:
                return x.transpose(-1, dim)[...,:-1].transpose(-1, dim)
            if all_slices:
                x = x.transpose(2, dim)[...,2:-2, 2:-2]
                x = x.transpose(1,2)
                return x.reshape(x.size(0)*x.size(1), x.size(2), x.size(3), x.size(4))
            return x.transpose(-1, dim)[...,2:-2, 2:-2,0]

    return Generator


def load_generator(Project_path):
    """Load generator for validation"""
    
    ## Create Networks
    netG = slicegan_net(Project_path)
    
    netG = netG()
    netG = netG.cuda()
    return netG