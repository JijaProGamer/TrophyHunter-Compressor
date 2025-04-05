import torch
import torch.nn as nn
import torch.nn.functional as F


"""class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=3, scale='none'):
        super(ResNetLayer, self).__init__()
        
        self.identity_proccess = in_channels != out_channels or scale != 'none'
        self.scale = scale

        if scale == 'up':
            conv_first_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        else:
            conv_first_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)

        self.conv_first = nn.Sequential(
            conv_first_layer,
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        layers = []
        for i in range(1, num_layers):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.GELU())

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
        out = F.gelu(out)

        return out"""

class ResNetSingleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale='none'):
        super(ResNetSingleLayer, self).__init__()
        
        self.identity_proccess = in_channels != out_channels or scale != 'none'
        self.scale = scale

        if scale == 'up':
            conv_first_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        elif scale == "down":
            conv_first_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            conv_first_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv1 = nn.Sequential(
            conv_first_layer,
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

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

        out = self.conv1(x)
        out = self.conv2(out)

        out += identity
        out = F.gelu(out)

        return out

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, scale='none'):
        super(ResNetLayer, self).__init__()

        self.first_layer = ResNetSingleLayer(in_channels, out_channels, scale)
        layers = []

        for _ in range(1, num_layers):
            layers.append(ResNetSingleLayer(out_channels, out_channels, scale='none'))

        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        out = self.first_layer(x)
        out = self.layers(out)
        return out
    
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale='down'):
        super(ConvLayer, self).__init__()

        if scale == 'up':
            conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        elif scale == "down":
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.layers = nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    def forward(self, x):
        return self.layers(x)

class DownscaleFilters(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownscaleFilters, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.gelu(out)
        
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
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio, kernel_size=7):
        super(CBAM, self).__init__()

        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        residual = x
        x = self.ca(x) * x
        x = self.sa(x) * x
        return F.gelu(residual + x)

def elastic_net_regularization(model, l1_lambda, l2_lambda, accepted_names):
    l1_norm = 0.0
    l2_norm = 0.0
    
    for name, param in model.named_parameters():
        if isinstance(param, nn.Parameter):
            if any(substring in name for substring in accepted_names):
                l1_norm += torch.norm(param, 1)
                l2_norm += torch.norm(param, 2)
    
    return l1_lambda * l1_norm + l2_lambda * l2_norm

class SonnetExponentialMovingAverage(nn.Module):
    # See: https://github.com/deepmind/sonnet/blob/5cbfdc356962d9b6198d5b63f0826a80acfdf35b/sonnet/src/moving_averages.py#L25.
    # They do *not* use the exponential moving average updates described in Appendix A.1
    # of "Neural Discrete Representation Learning".
    def __init__(self, decay, shape):
        super().__init__()
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros(*shape))
        self.register_buffer("average", torch.zeros(*shape))

    def update(self, value):
        self.counter += 1
        with torch.no_grad():
            self.hidden -= (self.hidden - value) * (1 - self.decay)
            self.average = self.hidden / (1 - self.decay ** self.counter)

    def __call__(self, value):
        self.update(value)
        return self.average

# Following 2 classes were taken from https://github.com/airalcorn2/vqvae-pytorch/blob/master/vqvae.py

class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, use_ema, decay, epsilon):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.use_ema = use_ema
        self.decay = decay
        self.epsilon = epsilon

        limit = 3 ** 0.5
        e_i_ts = torch.FloatTensor(embedding_dim, num_embeddings).uniform_(
            -limit, limit
        )
        if use_ema:
            self.register_buffer("e_i_ts", e_i_ts)
        else:
            self.register_parameter("e_i_ts", nn.Parameter(e_i_ts))

        self.N_i_ts = SonnetExponentialMovingAverage(decay, (num_embeddings,))
        self.m_i_ts = SonnetExponentialMovingAverage(decay, e_i_ts.shape)

    def forward(self, x):
        flat_x = x.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        distances = (
            (flat_x ** 2).sum(1, keepdim=True)
            - 2 * flat_x @ self.e_i_ts
            + (self.e_i_ts ** 2).sum(0, keepdim=True)
        )
        encoding_indices = distances.argmin(1)
        quantized_x = F.embedding(
            encoding_indices.view(x.shape[0], *x.shape[2:]), self.e_i_ts.transpose(0, 1)
        ).permute(0, 3, 1, 2)

        if not self.use_ema:
            dictionary_loss = ((x.detach() - quantized_x) ** 2).mean()
        else:
            dictionary_loss = None

        commitment_loss = ((x - quantized_x.detach()) ** 2).mean()
        quantized_x = x + (quantized_x - x).detach()

        if self.use_ema and self.training:
            with torch.no_grad():

                encoding_one_hots = F.one_hot(
                    encoding_indices, self.num_embeddings
                ).type(flat_x.dtype)
                n_i_ts = encoding_one_hots.sum(0)
                self.N_i_ts(n_i_ts)

                embed_sums = flat_x.transpose(0, 1) @ encoding_one_hots
                self.m_i_ts(embed_sums)

                N_i_ts_sum = self.N_i_ts.average.sum()
                N_i_ts_stable = (
                    (self.N_i_ts.average + self.epsilon)
                    / (N_i_ts_sum + self.num_embeddings * self.epsilon)
                    * N_i_ts_sum
                )
                self.e_i_ts = self.m_i_ts.average / N_i_ts_stable.unsqueeze(0)

        return (
            quantized_x,
            dictionary_loss,
            commitment_loss,
            encoding_indices.view(x.shape[0], -1),
        )