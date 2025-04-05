from tqdm import tqdm
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from pytorch_msssim import SSIM

lr = 3e-4
num_codes = 512
seed = 1234
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(seed)

    
class Encoder(nn.Module):
    def __init__(self, top_dims, middle_dims, bottom_dims):
        super().__init__()

        self.downscale = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(32),
        )


        self.top_layer = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(64),
        )

        self.middle_layer = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(128),
        )

        self.bottom_layer = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(256),
        )
        
        self.top_refilter = nn.Conv2d(64, top_dims, kernel_size=1, stride=1, padding=0)
        self.middle_refilter = nn.Conv2d(128, middle_dims, kernel_size=1, stride=1, padding=0)
        self.bottom_refilter = nn.Conv2d(256, bottom_dims, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        downscale = self.downscale(x)

        top_filters = self.top_layer(downscale)
        middle_filters = self.middle_layer(top_filters)
        bottom_filters = self.bottom_layer(middle_filters)

        return self.top_refilter(top_filters), self.middle_refilter(middle_filters), self.bottom_refilter(bottom_filters)

class Decoder(nn.Module):
    def __init__(self, top_dims, middle_dims, bottom_dims):
        super().__init__()

        self.bottom_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(bottom_dims, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(256)
        )

        self.middle_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256 + middle_dims, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(128)
        )

        self.top_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128 + top_dims, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(64)
        )

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, quantized_top, quantized_middle, quantized_bottom):
        bottom_filters = self.bottom_layer(quantized_bottom)
        middle_filters = self.middle_layer(torch.cat((bottom_filters, quantized_middle), dim=1))
        top_filters = self.top_layer(torch.cat((middle_filters, quantized_top), dim=1))

        return self.up(top_filters)
    
"""class AECritic(nn.Module):
    def __init__(self):
        super(AECritic, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(128),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1) 
        )

    def forward(self, img):
        return self.model(img)"""

class VQVAE(nn.Module):
    def __init__(self, top_codebook_size, top_dims, middle_codebook_size, middle_dims, bottom_codebook_size, bottom_dims):
        super().__init__()
        #self.critic = AECritic()
        self.encoder = Encoder(top_dims, middle_dims, bottom_dims)
        self.decoder = Decoder(top_dims, middle_dims, bottom_dims)

        self.top_codebook_size = top_codebook_size
        self.middle_codebook_size = middle_codebook_size
        self.bottom_codebook_size = bottom_codebook_size

        self.top_quantizer = VectorQuantize(accept_image_fmap=True, rotation_trick=True, kmeans_init=True, kmeans_iters=200, codebook_size=top_codebook_size, dim=top_dims)#threshold_ema_dead_code=4)
        self.middle_quantizer = VectorQuantize(accept_image_fmap=True, rotation_trick=True, kmeans_init=True, kmeans_iters=200, codebook_size=middle_codebook_size, dim=middle_dims)#threshold_ema_dead_code=4)
        self.bottom_quantizer = VectorQuantize(accept_image_fmap=True, rotation_trick=True, kmeans_init=True, kmeans_iters=200, codebook_size=bottom_codebook_size, dim=bottom_dims)#threshold_ema_dead_code=4)

    def forward(self, x):
        encoded_top, encoded_middle, encoded_bottom = self.encoder(x)

        quantized_top, indices_top, commitment_loss_top = self.top_quantizer(encoded_top)
        quantized_middle, indices_middle, commitment_loss_middle = self.middle_quantizer(encoded_middle)
        quantized_bottom, indices_bottom, commitment_loss_bottom = self.bottom_quantizer(encoded_bottom)

        commitment_loss = commitment_loss_top + commitment_loss_middle + commitment_loss_bottom
        decoded = self.decoder(quantized_top, quantized_middle, quantized_bottom)
        return decoded, (indices_top, indices_middle, indices_bottom), commitment_loss
    
def validate(model, val_loader):
    model.eval()

    total_rec_loss = 0
    total_cmt_loss = 0
    total_active_top = 0
    total_active_middle = 0
    total_active_bottom = 0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for x, _ in pbar:
            x = x.to(device)

            out, (indices_top, indices_middle, indices_bottom), cmt_loss = model(x)

            rec_loss = (out - x).abs().mean()

            unique_codes_top = indices_top.unique().numel()
            active_percent_top = (unique_codes_top / model.top_codebook_size) * 100

            unique_codes_middle = indices_middle.unique().numel()
            active_percent_middle = (unique_codes_middle / model.middle_codebook_size) * 100

            unique_codes_bottom = indices_bottom.unique().numel()
            active_percent_bottom = (unique_codes_bottom / model.bottom_codebook_size) * 100

            total_rec_loss += rec_loss.item()
            total_cmt_loss += cmt_loss.item()
            total_active_top += active_percent_top
            total_active_middle += active_percent_middle
            total_active_bottom += active_percent_bottom
            num_batches += 1

    avg_rec_loss = total_rec_loss / num_batches
    avg_active_top = total_active_top / num_batches
    avg_active_middle = total_active_middle / num_batches
    avg_active_bottom = total_active_bottom / num_batches

    print(f"[Validation] Avg rec_loss: {avg_rec_loss:.3f} | "
          f"Avg active %: {avg_active_top:.2f} | "
          f"Avg active %: {avg_active_middle:.2f} | "
          f"Avg active %: {avg_active_bottom:.2f}")


def train(model, train_loader, val_loader, test_dataset, epochs=50, alpha=10):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    #optimizer_critic = optim.Adam(model.parameters(), lr=lr)
    ssim = SSIM(data_range=1, size_average=True, channel=3, nonnegative_ssim=True)

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for x, _ in pbar:
            model.train()

            x = x.to(device)

            """# train critic

            out_critic, _, _ = model(x)
            real_validity = model.critic(x)
            fake_validity = model.critic(out_critic)
            gradient_penalty = compute_gradient_penalty(model.critic, x, out_critic)
            critic_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            # train autoencoder"""

            out, indices, cmt_loss = model(x)

            ssim_loss = 1 - ssim((x + 1) / 2, (out + 1) / 2)
            l1_loss = (out - x).abs().mean()

            rec_loss = ssim_loss
            #rec_loss = ssim_loss * 0.3 + l1_loss * 0.7# - torch.mean(model.critic(out))
            loss = rec_loss + alpha * cmt_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        validate(model, val_loader)
        evaluate_and_save(model, test_loader=test_dataset)

def evaluate_and_save(model, test_loader, filename="reconstructions.png"):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x.to(device)

        out, _, _ = model(x)
        out = out.clamp(-1., 1.)

        grid_img = vutils.make_grid(torch.cat([x, out], dim=0), nrow=test_dataset.batch_size, normalize=True, scale_each=True)
        plt.figure(figsize=(grid_img.size(2) / 100, grid_img.size(1) / 100), dpi=100)
        plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)
        plt.close()

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))


transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = DataLoader(
    datasets.STL10(root="data", split='train+unlabeled', download=True, transform=transform),
    batch_size=128,
    shuffle=True,
    pin_memory = True
)

test_dataset = DataLoader(
    datasets.STL10(root="data", split='test', download=True, transform=transform),
    batch_size=32,
    shuffle=False,
    pin_memory = True
)

val_dataset = DataLoader(
    datasets.STL10(root="data", split='test', download=True, transform=transform),
    batch_size=32,
    shuffle=True,
    pin_memory=True
)

model = VQVAE(
    top_codebook_size=32, top_dims=16, 
    middle_codebook_size=64, middle_dims=32,
    bottom_codebook_size=128, bottom_dims=64,
).to(device)


train(model, train_loader=train_dataset, val_loader=val_dataset, test_dataset=test_dataset)
save_model(model, "model.pt")
load_model(model, "model.pt")

for param in model.parameters():
    param.requires_grad = False

model.eval()

# Reconstruction
evaluate_and_save(model, test_dataset)

#

class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0


def encode_and_codebook(model, dataset):
    model.eval()
    encoded_data = []

    with torch.no_grad():
        pbar = tqdm(dataset, desc=f"Loading generator Images", leave=False)
        for x, _ in pbar:
            encoded = model.encoder(x.to(device))
            quantized, _, _ = model.quantizer(encoded)
            encoded_data.extend(quantized.cpu().numpy())

    return encoded_data


encoded_data = encode_and_codebook(model, train_dataset)
new_train_dataset = EncodedDataset(encoded_data)

new_train_loader = DataLoader(new_train_dataset, batch_size=256, shuffle=True)


# Random images
from generator import Generator, Critic, train_generator

def generate_images(generator, device, num_images, save_path="random_images.png"):
    generator.eval()

    noise = torch.randn(num_images**2, generator.model[0].in_features).to(device)
    with torch.no_grad():
        generated_features = generator(noise)

    generated_features, _, _ = model.quantizer.vq(generated_features)
    generated_imgs = model.decoder(generated_features).clamp(-1, 1)
    
    grid = vutils.make_grid(generated_imgs.cpu(), nrow=num_images, padding=2, normalize=True)
    plt.imsave(save_path, grid.permute(1, 2, 0).numpy())

generator_latent_dim = 128

generator = Generator(latent_dim=generator_latent_dim).to(device)
critic = Critic().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizer_C = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

train_generator(train_dataset=new_train_loader, generator=generator, critic=critic, optimizer_G=optimizer_G, optimizer_C=optimizer_C, device=device, generate_images=generate_images, latent_dim=generator_latent_dim, lambda_gp=10, critic_iterations=10, epochs=10000)

generate_images(generator, device, num_images=8)