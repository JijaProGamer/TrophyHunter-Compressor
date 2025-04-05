import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torchsummary import summary

from layers import SelfAttentionModule, CBAM, ResNetLayer, DownscaleFilters, elastic_net_regularization

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    
    tiled_x = x.unsqueeze(1).expand(x_size, y_size, dim)
    tiled_y = y.unsqueeze(0).expand(x_size, y_size, dim)
    
    return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / float(dim))

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

"""class Discriminator(nn.Module):
    def __init__(self, latent_dims):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(latent_dims, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x"""


class Encoder(nn.Module):
    def __init__(self, args, input_resolution, latent_dims):
        super(Encoder, self).__init__()

        x_downscale = int(input_resolution[0] / 16)
        y_downscale = int(input_resolution[1] / 16)
        out_dims = 512 * x_downscale * y_downscale
        
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn = nn.BatchNorm2d(32)
        
        self.res1 = ResNetLayer(in_channels=32, out_channels=64, scale='down', num_layers=1)
        self.res2 = ResNetLayer(in_channels=64, out_channels=128, scale='down', num_layers=2)
        self.res3 = ResNetLayer(in_channels=128, out_channels=256, scale='down', num_layers=3)
        self.res4 = ResNetLayer(in_channels=256, out_channels=512, scale='down', num_layers=4)

        #self.simplify_1 = DownscaleFilters(in_channels=256, out_channels=16)
        self.attention1 = CBAM(512, 2, 5)#SelfAttentionModule(attention_features=256)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_dims, latent_dims)

    def forward_conv(self, x):
        x = F.gelu(self.bn(self.conv(x)))

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        #x = self.simplify_1(x)
        x = self.attention1(x)
    
        return x
    def continue_conv(self, x):
        x = self.flatten(x)

        mu = self.fc(x)

        return mu
    def forward(self, x):
        x = self.forward_conv(x)
        x = self.flatten(x)

        mu = self.fc(x)

        return mu

class Decoder(nn.Module):
    def __init__(self, args, input_resolution, latent_dims):
        super(Decoder, self).__init__()

        x_downscale = int(input_resolution[0] / 16)
        y_downscale = int(input_resolution[1] / 16)

        self.input_resolution = input_resolution
        self.x_downscale = x_downscale
        self.y_downscale = y_downscale

        self.transformer = nn.Linear(latent_dims, x_downscale * y_downscale * 512, bias=False)
        self.transformer_bn = nn.BatchNorm1d(x_downscale * y_downscale * 512)

        #self.simplify_1 = DownscaleFilters(in_channels=16, out_channels=256)

        self.res1 = ResNetLayer(in_channels=512, out_channels=256, scale='up', num_layers=4)
        self.res2 = ResNetLayer(in_channels=256, out_channels=128, scale='up', num_layers=3)
        self.res3 = ResNetLayer(in_channels=128, out_channels=64, scale='up', num_layers=2)
        self.res4 = ResNetLayer(in_channels=64, out_channels=32, scale='up', num_layers=1)

        self.output = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        x = F.gelu(self.transformer_bn(self.transformer(x)))
        x = x.view(-1, 512, self.x_downscale, self.y_downscale)

        #x = self.simplify_1(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        x = self.output(x)
        
        return x

class VAE(nn.Module):
    def __init__(self, args, dataset_size):
        super().__init__()

        self.args = args
        self.dataset_size = dataset_size

        self.ssim_module = SSIM(data_range=1, size_average=True, channel=3, win_size=5, win_sigma=1)
       
        self.encoder = Encoder(
            input_resolution=args["resolution"],
            latent_dims = args["latent_size"],
            args = args
        )

        self.decoder = Decoder(
            input_resolution=args["resolution"],
            latent_dims = args["latent_size"],
            args = args
        )

        #self.discriminator = Discriminator(
        #    latent_dims = args["latent_size"]
        #)

        summary(self, (3, self.args["resolution"][0], self.args["resolution"][1]), device="cpu")

    def reconstruction_loss(self, x, y):
        #lpips = self.lpips(x, y).mean()
        #l1 =    F.l1_loss(x, y)
        ssim = 1 - self.ssim_module((x + 1) / 2, (y + 1) / 2)
        #combo = lpips * 0.1 + ssim * 0.8 + l1 * 0.1
        combo = ssim #* 0.9# + l1 * 0.1

        return combo * (self.args["resolution"][0] * self.args["resolution"][1] * 3)

    def disentanglement_loss(self, z):
        prior_samples = torch.randn_like(z)
        return compute_mmd(prior_samples, z)
    
    def loss(self, x, y, z):
        #regularization = elastic_net_regularization(self, l1_lambda=self.args['l1_lambda'], l2_lambda=self.args['l2_lambda'], accepted_names=[".res"])
        reconstruction = self.reconstruction_loss(x, y)
        disentanglement = self.disentanglement_loss(z)

        return reconstruction + disentanglement * self.args["beta"]

    def forward(self, x):
        z = self.encoder.forward(x)
        decoded = self.decoder(z)

        return z, decoded
    
    def muforward(self, x):
        z = self.encoder.forward(x)
        decoded = self.decoder(z)

        return z, decoded    
    def zforward(self, x):
        z = self.encoder.forward(x)

        return z