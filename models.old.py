import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torchsummary import summary

from layers import SelfAttentionModule, ResNetLayer, DenseResNetLayer, DownscaleFilters, elastic_net_regularization

class Encoder(nn.Module):
    def __init__(self, args, input_resolution, latent_dims):
        super(Encoder, self).__init__()

        x_downscale = int(input_resolution[0] / 16)
        y_downscale = int(input_resolution[1] / 16)
        out_dims = 16 * x_downscale * y_downscale
        
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn = nn.BatchNorm2d(64)
        
        self.res1 = ResNetLayer(in_channels=64, out_channels=128, scale='down', num_layers=1)
        self.res2 = ResNetLayer(in_channels=128, out_channels=256, scale='down', num_layers=2)
        self.res3 = ResNetLayer(in_channels=256, out_channels=512, scale='down', num_layers=3)
        self.res4 = ResNetLayer(in_channels=512, out_channels=1024, scale='down', num_layers=4)

        self.simplify_1 = DownscaleFilters(in_channels=1024, out_channels=16)
        #self.simplify_2 = DownscaleFilters(in_channels=256, out_channels=32)
        
        self.flatten = nn.Flatten()

        self.fc_mean = nn.Linear(out_dims, latent_dims)
        self.fc_var = nn.Linear(out_dims, latent_dims)

    def forward_conv(self, x):
        x = F.gelu(self.bn(self.conv(x)))

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        x = self.simplify_1(x)
        #x = self.simplify_2(x)

    
        return x
    def forward(self, x):
        x = self.forward_conv(x)
        x = self.flatten(x)

        mu = self.fc_mean(x)
        logvar = self.fc_var(x)

        return mu, logvar
    def forward_mu(self, x):
        x = self.forward_conv(x)
        x = self.flatten(x)

        mu = self.fc_mean(x)

        return mu

class Decoder(nn.Module):
    def __init__(self, args, input_resolution, latent_dims):
        super(Decoder, self).__init__()

        x_downscale = int(input_resolution[0] / 16)
        y_downscale = int(input_resolution[1] / 16)

        self.input_resolution = input_resolution
        self.x_downscale = x_downscale
        self.y_downscale = y_downscale

        self.transformer = nn.Linear(latent_dims, x_downscale * y_downscale * 16, bias=False)
        self.transformer_bn = nn.BatchNorm1d(x_downscale * y_downscale * 16)

        #self.simplify_1 = DownscaleFilters(in_channels=32, out_channels=256)
        self.simplify_2 = DownscaleFilters(in_channels=16, out_channels=1024)

        self.res1 = ResNetLayer(in_channels=1024, out_channels=512, scale='up', num_layers=4)
        self.res2 = ResNetLayer(in_channels=512, out_channels=256, scale='up', num_layers=3)
        self.res3 = ResNetLayer(in_channels=256, out_channels=128, scale='up', num_layers=2)
        self.res4 = ResNetLayer(in_channels=128, out_channels=64, scale='up', num_layers=1)

        self.output = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        x = F.gelu(self.transformer_bn(self.transformer(x)))
        x = x.view(-1, 16, self.x_downscale, self.y_downscale)

        #x = self.simplify_1(x)
        x = self.simplify_2(x)

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

        self.ssim_module = SSIM(data_range=1, size_average=True, channel=3, win_size=11, win_sigma=0.1)
       
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

        summary(self, (3, self.args["resolution"][0], self.args["resolution"][1]), device="cpu")


    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        #std = torch.clamp(std, min=0, max=100)

        eps = torch.randn_like(std)

        return mean + eps * std
    
    #def reconstruction_loss(self, x, y):
    #    x_slim = (x + 1) / 2
    #    y_slim = (y + 1) / 2
    #
    #    ssim = 1 - self.ssim_module(x_slim, y_slim)
    #    return ssim * (self.args["resolution"][0] * self.args["resolution"][1] * 3)


    def reconstruction_loss(self, x, y):
        #lpips = self.lpips(x, y).mean()
        l1 =    F.l1_loss(x, y)
        ssim = 1 - self.ssim_module((x + 1) / 2, (y + 1) / 2)
        #combo = lpips * 0.1 + ssim * 0.8 + l1 * 0.1
        combo = ssim * 0.9 + l1 * 0.1

        return combo * (self.args["resolution"][0] * self.args["resolution"][1] * 3)

    def disentanglement_loss(self, step, z, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    def loss(self, step, x, y, z, mu, logvar):
        #regularization = elastic_net_regularization(self, l1_lambda=self.args['l1_lambda'], l2_lambda=self.args['l2_lambda'], accepted_names=[".res"])
        reconstruction = self.reconstruction_loss(x, y)

        if self.args["disentangle"]:
            disentanglement = self.disentanglement_loss(step, z, mu, logvar)

            return reconstruction + disentanglement
        else:
            return reconstruction

    def forward(self, x):
        if self.args["disentangle"]:
            mu, logvar = self.encoder(x)

            z = self.reparameterize(mu, logvar)
            decoded = self.decoder(z)
            return mu, logvar, z, decoded
        else:
            mu = self.encoder.forward_mu(x)

            decoded = self.decoder(mu)

            return mu, torch.zeros_like(mu), mu, decoded
    def muforward(self, x):
        mu = self.encoder.forward_mu(x)
        decoded = self.decoder(mu)

        return mu, decoded    
    def zforward(self, x, disable_disentanglement):
        if self.args["disentangle"] and not disable_disentanglement:
            mu, logvar = self.encoder(x)

            z = self.reparameterize(mu, logvar)
            return mu, logvar, z
        else:
            mu = self.encoder.forward_mu(x)

            return mu