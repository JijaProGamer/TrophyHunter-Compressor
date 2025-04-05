import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from pytorch_msssim import SSIM
from torchsummary import summary

from layers import CBAM, ResNetLayer, DownscaleFilters, VectorQuantizer, ConvLayer


class Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn = nn.BatchNorm2d(64)
        
        #self.res1 = ResNetLayer(in_channels=64, out_channels=128, scale='down', num_layers=1)
        #self.res2 = ResNetLayer(in_channels=128, out_channels=256, scale='down', num_layers=2)
        #self.res3 = ResNetLayer(in_channels=256, out_channels=512, scale='down', num_layers=3)
        #self.res4 = ResNetLayer(in_channels=512, out_channels=1024, scale='down', num_layers=4)

        self.res1 = ConvLayer(64, 128, scale='down')
        self.res2 = ConvLayer(128, 256, scale='down')
        self.res3 = ConvLayer(256, 512, scale='down')
        self.res4 = ConvLayer(512, 1024, scale='down')

        self.simplify_1 = DownscaleFilters(in_channels=1024, out_channels=embedding_dim)

    def forward(self, x):
        x = F.gelu(self.bn(self.conv(x)))

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        x = self.simplify_1(x)

    
        return x

class Decoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Decoder, self).__init__()

        self.simplify_1 = DownscaleFilters(in_channels=embedding_dim, out_channels=1024)

        #self.res1 = ResNetLayer(in_channels=1024, out_channels=512, scale='up', num_layers=4)
        #self.res2 = ResNetLayer(in_channels=512, out_channels=256, scale='up', num_layers=3)
        #self.res3 = ResNetLayer(in_channels=256, out_channels=128, scale='up', num_layers=2)
        #self.res4 = ResNetLayer(in_channels=128, out_channels=64, scale='up', num_layers=1)

        self.res1 = ConvLayer(1024, 512, scale='up')
        self.res2 = ConvLayer(512, 256, scale='up')
        self.res3 = ConvLayer(256, 128, scale='up')
        self.res4 = ConvLayer(128, 64, scale='up')

        self.output = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        x = self.simplify_1(x)

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
            embedding_dim = args["embedding_dim"]
        ).to(args["device"])

        self.quantizer = VectorQuantizer(
            embedding_dim = args["embedding_dim"], 
            num_embeddings = args["num_embeddings"], 
            use_ema = args["use_ema"], 
            decay = args["decay"], 
            epsilon = args["epsilon"]
        ).to(args["device"])

        self.decoder = Decoder(
            embedding_dim = args["embedding_dim"]
        ).to(args["device"])

        #summary(self, (3, self.args["resolution"][0], self.args["resolution"][1]), device=args["device"])

    def reconstruction_loss(self, x, y):
        #ssim = 1 - self.ssim_module((x + 1) / 2, (y + 1) / 2)
        ssim = F.mse_loss(x, y)
        return ssim #* (self.args["resolution"][0] * self.args["resolution"][1] * 3)

    def loss(self, x, y, dictionary_loss, commitment_loss):
        reconstruction = self.reconstruction_loss(x, y)
        commitment = self.args["beta"] * commitment_loss
        loss = reconstruction + commitment

        if not self.args["use_ema"]:
            loss += dictionary_loss

        return loss

    """def forward(self, x):
        z = self.encoder.forward(x)
        decoded = self.decoder(z)

        return z, decoded"""
    
    def quantize(self, x):
        z = self.encoder(x)
        (z_quantized, dictionary_loss, commitment_loss, encoding_indices) = self.quantizer(z)
        return z, z_quantized, dictionary_loss, commitment_loss, encoding_indices

    def forward(self, x):
        z, z_quantized, dictionary_loss, commitment_loss, encoding_indices = self.quantize(x)
        x_recon = self.decoder(z_quantized)
        return z, x_recon, z_quantized, dictionary_loss, commitment_loss, encoding_indices
    def zforward(self, x):
        z = self.encoder(x)
        (z_quantized, _, _, encoding_indices) = self.quantize(z)

        return z, z_quantized, encoding_indices