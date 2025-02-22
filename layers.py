import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=3, scale='none'):
        super(ResNetLayer, self).__init__()
        
        self.identity_proccess = in_channels != out_channels or scale != 'none'
        self.scale = scale
        self.num_layers = num_layers

        if scale == 'up':
            conv_first_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        else:
            conv_first_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)

        self.conv_first = nn.Sequential(
            conv_first_layer,
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

        layers = []
        for i in range(1, num_layers):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.SiLU())

        self.conv_stack = nn.Sequential(*layers)

        if self.identity_proccess:
            if scale == 'up':
                self.skip_connection = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=2, output_padding=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.skip_connection = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride= 1 if scale == "none" else 2, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

    def forward(self, x):
        identity = x

        if self.identity_proccess:
            identity = self.skip_connection(x)

        out = self.conv_first(x)

        out = self.conv_stack(out)
        out += identity
        out = F.silu(out)

        return out



class DownscaleFilters(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownscaleFilters, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.silu(out)
        
        return out

class DenseResNetLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseResNetLayer, self).__init__()

        self.identity_process = in_features != out_features

        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)

        if self.identity_process:
            self.skip_connection = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
    
    def forward(self, x):
        identity = x

        if self.identity_process:
            identity = self.skip_connection(x)
        
        out = self.linear1(x)
        out = self.bn1(out)
        out = F.silu(out)

        out += identity
        out = F.silu(out)
        
        return out

class SelfAttentionModule(nn.Module):
    def __init__(self, attention_features):
        super(SelfAttentionModule, self).__init__()
        
        self.f = nn.Conv2d(in_channels=attention_features, out_channels=attention_features, kernel_size=1)
        self.g = nn.Conv2d(in_channels=attention_features, out_channels=attention_features, kernel_size=1)
        self.h = nn.Conv2d(in_channels=attention_features, out_channels=attention_features, kernel_size=1)

        self.scale = nn.Parameter(torch.zeros(1))

    def flatten_hw(self, x):
        batch_size, channels, height, width = x.size()
        return x.view(batch_size, channels, -1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        f = self.f(x)
        g = self.g(x)
        h = self.h(x)

        f_flatten = self.flatten_hw(f)
        g_flatten = self.flatten_hw(g)
        h_flatten = self.flatten_hw(h)
        
        s = torch.bmm(g_flatten.permute(0, 2, 1), f_flatten)
        b = F.softmax(s, dim=-1)
        
        o = torch.bmm(h_flatten, b.permute(0, 2, 1))
        o = o.view(batch_size, channels, height, width)
        
        y = self.scale * o
        
        return x + y

"""
class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale='none', kernel_size=3, stride=1):
        super(ResNetLayer, self).__init__()

        self.identity_proccess = in_channels != out_channels or stride > 1

        if scale == 'down':
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=1)
        elif scale == 'none':
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1)

        if self.identity_proccess:
            if scale == 'down':
                self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            elif scale == 'none':
                self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
                
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1)
    
    def forward(self, x):
        identity = x

        if self.identity_proccess:
            identity = self.skip_connection(x)
        
        out = self.conv1(x)
        out = F.silu(out)
        
        out = self.conv2(out)
        
        out += identity
        out = F.silu(out)
        
        return out
    
class DenseResNetLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseResNetLayer, self).__init__()

        self.identity_process = in_features != out_features

        self.linear1 = nn.Linear(in_features, out_features)

        if self.identity_process:
            self.skip_connection = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        identity = x

        if self.identity_process:
            identity = self.skip_connection(x)
        
        out = self.linear1(x)
        out = F.silu(out)

        out += identity
        out = F.silu(out)

        return out

class SelfAttentionModule(nn.Module):
    def __init__(self, attention_features):
        super(SelfAttentionModule, self).__init__()
        
        self.f = nn.Conv2d(in_channels=attention_features, out_channels=attention_features, kernel_size=1)
        self.g = nn.Conv2d(in_channels=attention_features, out_channels=attention_features, kernel_size=1)
        self.h = nn.Conv2d(in_channels=attention_features, out_channels=attention_features, kernel_size=1)
        
        self.scale = nn.Parameter(torch.zeros(1))

    def flatten_hw(self, x):
        batch_size, channels, height, width = x.size()
        return x.view(batch_size, channels, -1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        f = self.f(x)
        g = self.g(x)
        h = self.h(x)

        f_flatten = self.flatten_hw(f)
        g_flatten = self.flatten_hw(g)
        h_flatten = self.flatten_hw(h)
        
        s = torch.bmm(g_flatten.permute(0, 2, 1), f_flatten)
        b = F.softmax(s, dim=-1)
        
        o = torch.bmm(h_flatten, b.permute(0, 2, 1))
        o = o.view(batch_size, channels, height, width)
        
        y = self.scale * o
        
        return x + y
"""
        
def elastic_net_regularization(model, l1_lambda, l2_lambda, accepted_names):
    l1_norm = 0.0
    l2_norm = 0.0
    
    for name, param in model.named_parameters():
        if isinstance(param, nn.Parameter):
            if any(substring in name for substring in accepted_names):
                l1_norm += torch.norm(param, 1)
                l2_norm += torch.norm(param, 2)
    
    return l1_lambda * l1_norm + l2_lambda * l2_norm