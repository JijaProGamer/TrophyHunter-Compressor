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
from generator import compute_gradient_penalty

lr = 3e-4
num_codes = 512
seed = 1234
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(seed)

    
class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        return self.net(x)

class Quantizer(nn.Module):
    def __init__(self, **vq_kwargs):
        super().__init__()
        self.vq = VectorQuantize(accept_image_fmap=True, **vq_kwargs)

    def forward(self, x):
        return self.vq(x)

class Decoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)
    
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
    def __init__(self, **vq_kwargs):
        super().__init__()
        dim = vq_kwargs["dim"]
        #self.critic = AECritic()
        self.encoder = Encoder(dim)
        self.quantizer = Quantizer(**vq_kwargs)
        self.decoder = Decoder(dim)

    def forward(self, x):
        encoded = self.encoder(x)
        quantized, indices, commitment_loss = self.quantizer(encoded)
        decoded = self.decoder(quantized)
        return decoded, indices, commitment_loss
    
def validate(model, val_loader):
    model.eval()

    total_rec_loss = 0
    total_cmt_loss = 0
    total_active_percent = 0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for x, _ in pbar:
            x = x.to(device)

            out, indices, cmt_loss = model(x)
            #out = out.clamp(-1., 1.)

            rec_loss = (out - x).abs().mean()
            unique_codes = indices.unique().numel()
            active_percent = (unique_codes / num_codes) * 100

            total_rec_loss += rec_loss.item()
            total_cmt_loss += cmt_loss.item()
            total_active_percent += active_percent
            num_batches += 1

            pbar.set_postfix({
                "rec_loss": f"{rec_loss.item():.3f}",
                "cmt_loss": f"{cmt_loss.item():.3f}",
                "active_%": f"{active_percent:.2f}"
            })

    avg_rec_loss = total_rec_loss / num_batches
    avg_cmt_loss = total_cmt_loss / num_batches
    avg_active_percent = total_active_percent / num_batches

    print(f"[Validation] Avg rec_loss: {avg_rec_loss:.3f} | "
          f"Avg cmt_loss: {avg_cmt_loss:.3f} | "
          f"Avg active %: {avg_active_percent:.2f}")


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

            unique_codes = indices.unique().numel()
            active_percent = (unique_codes / num_codes) * 100

            pbar.set_postfix({
                "rec_loss": f"{rec_loss.item():.3f}",
                "cmt_loss": f"{cmt_loss.item():.3f}",
                "active_%": f"{active_percent:.2f}"
            })

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
    rotation_trick=True,
    #threshold_ema_dead_code=4,
    kmeans_init=True,
    kmeans_iters=200,
    codebook_size=num_codes,
    dim=16,

    heads=8,
    separate_codebook_per_head=True,

    # orthogonal_reg_weight=10, 
    # orthogonal_reg_max_codes=128,
    # orthogonal_reg_active_codes_only=False
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