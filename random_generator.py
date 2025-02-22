import torch
import os
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torch.nn as nn
from torch.autograd import grad

import torchvision.utils as vutils

from models import VAE

args = {
    "device": torch.device("cuda"),
    "batch_size": 64,
    "resolution": [224, 400],
    "latent_size": 2048,
    "disentangle": True
}

class FlatFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, raw_shape=(128, 128, 3)):
        self.image_paths = list(Path(root_dir).rglob("*.*"))
        self.transform = transform
        self.raw_shape = raw_shape

    def __len__(self):
        return len(self.image_paths)

    def _load_raw_image(self, image_path):
        with open(image_path, "rb") as f:
            raw_data = np.frombuffer(f.read(), dtype=np.uint8)
        image = raw_data.reshape(self.raw_shape)
        return image

    def __getitem__(self, idx):
        max_attempts = len(self.image_paths)

        for _ in range(max_attempts):
            image_path = self.image_paths[idx]

            try:
                if image_path.suffix.lower() == ".raw":
                    image = self._load_raw_image(image_path)
                else:
                    image = Image.open(image_path).convert("RGB")
                    image = np.array(image)

                image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
                image = image / 127.5 - 1

                #if self.transform:
                #    image = self.transform(image)
                
                return image, 0

            except Exception as e:
                os.remove(image_path)
                del self.image_paths[idx]
                idx = idx % len(self.image_paths) if self.image_paths else 0
        
        raise RuntimeError("All images in dataset are corrupted!")

transform = transforms.Compose([
    transforms.Resize((args["resolution"][0], args["resolution"][1])),
])

image_dataset = FlatFolderDataset(root_dir="./images", transform=transform, raw_shape=(224, 400, 3))


train_loader = DataLoader(image_dataset, batch_size=args["batch_size"], shuffle=True)

model = VAE(args, 0).to(args["device"])


checkpoint_path = "model.pt"
def load_model():
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])

images_data = []
load_model()

def load_dataset():
    model.eval()
    
    progress_bar = tqdm(train_loader, desc=f"Loading Images", unit="batch")

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(progress_bar):
            if images.size(0) < train_loader.batch_size:
                continue

            if np.isnan(images).any():
                print("NaN images")
                continue

            images = images.to(args["device"])
            mu = model.zforward(images, disable_disentanglement=True)
            images_data.append(mu)

load_dataset()

gan_args = {
    "batch_size": 128,
    "latent_size": 256,
    "n_critic": 8,
    "lambda_gp": 10,
    "epochs": 1000,
    "test_size": 5
}

images_data_tensor = torch.cat(images_data, dim=0)
images_dataset = TensorDataset(images_data_tensor)
images_dataloader = DataLoader(images_dataset, batch_size=gan_args["batch_size"], shuffle=True)


class Generator(nn.Module):
    def __init__(self, latent_dim, seq_length):
        super(Generator, self).__init__()
        """self.model = nn.Sequential(
            nn.Linear(latent_dim, 512, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, seq_length)
        )"""

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, seq_length)
        )

    def forward(self, z):
        return self.model(z)

class Critic(nn.Module):
    def __init__(self, seq_length):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(seq_length, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1).to(args["device"])
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones(d_interpolates.size()).to(args["device"])
    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

generator = Generator(gan_args["latent_size"], args["latent_size"]).to(args["device"])
critic = Critic(args["latent_size"]).to(args["device"])

#optimizer_G = optim.Adam(generator.parameters(), lr=5e-4, betas=(0.9, 0.99))
#optimizer_C = optim.Adam(critic.parameters(), lr=5e-4, betas=(0.9, 0.99))
optimizer_G = optim.RMSprop(generator.parameters(), lr=1e-4)
optimizer_C = optim.RMSprop(critic.parameters(), lr=1e-4)

fixed_z = torch.randn(gan_args["test_size"] * gan_args["test_size"], gan_args["latent_size"]).to(args["device"])
def save_generated_images(save_path):
    generator.eval()
    with torch.no_grad():
        fake_samples = generator(fixed_z)

        reconstructed_images = model.decoder(fake_samples)
        reconstructed_images = (reconstructed_images + 1) / 2
        reconstructed_images = reconstructed_images.clamp(0, 1)

        grid_images = reconstructed_images.cpu().numpy()

        fig, axes = plt.subplots(gan_args["test_size"], gan_args["test_size"], figsize=(gan_args["test_size"]*5, gan_args["test_size"]*5))
        axes = axes.flatten()

        for i in range(gan_args["test_size"] * gan_args["test_size"]):
            img = grid_images[i].transpose(1, 2, 0)
            axes[i].imshow(img)
            axes[i].axis('off')

        for i in range(gan_args["test_size"] * gan_args["test_size"], len(axes)):
            axes[i].axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)

        plt.tight_layout(pad=0)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    generator.train()

for epoch in range(gan_args["epochs"]):
    generator.train()
    critic.train()
    
    epoch_progress = tqdm(images_dataloader, desc=f"Epoch [{epoch+1}/{gan_args['epochs']}]", unit="batch")
    
    for i, (real_samples,) in enumerate(epoch_progress):
        real_samples = real_samples.to(args["device"])

        for _ in range(gan_args["n_critic"]):
            z = torch.randn(real_samples.shape[0], gan_args["latent_size"]).to(args["device"])
            fake_samples = generator(z)

            real_validity = critic(real_samples)
            fake_validity = critic(fake_samples)

            gradient_penalty = compute_gradient_penalty(critic, real_samples, fake_samples)

            critic_loss = -(torch.mean(real_validity) - torch.mean(fake_validity)) + gan_args["lambda_gp"] * gradient_penalty

            optimizer_C.zero_grad()
            critic_loss.backward()
            optimizer_C.step()

        z = torch.randn(gan_args["batch_size"], gan_args["latent_size"]).to(args["device"])
        fake_samples = generator(z)
        fake_validity = critic(fake_samples)
        generator_loss = -torch.mean(fake_validity)

        optimizer_G.zero_grad()
        generator_loss.backward()
        optimizer_G.step()

        epoch_progress.set_postfix({
            "Critic Loss": f"{critic_loss.item():.4f}",
            "Generator Loss": f"{generator_loss.item():.4f}"
        })

    save_generated_images(f"random_gan_epoch.png")


#torch.save(generator.state_dict(), "generator.pth")
#torch.save(critic.state_dict(), "critic.pth")

#print("Training complete.")