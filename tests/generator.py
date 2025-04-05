import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128, bias=False),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256, bias=False),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 512, bias=False),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 16 * 12 * 12)
        )

    def forward(self, z):
        return self.model(z).view(-1, 16, 12, 12)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(16 * 12 * 12, 512, bias=False),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256, bias=False),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128, bias=False),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )

    def forward(self, img):
        return self.model(img.view(img.size(0), -1))

def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = critic(interpolates)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_generator(train_dataset, generator, critic, optimizer_G, optimizer_C, device, generate_images, latent_dim, lambda_gp, critic_iterations, epochs):
    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(train_dataset):
            generator.train()
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            for _ in range(critic_iterations):
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_imgs = generator(z).detach()

                real_validity = critic(real_imgs)
                fake_validity = critic(fake_imgs)
                gradient_penalty = compute_gradient_penalty(critic, real_imgs, fake_imgs)
                critic_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

                optimizer_C.zero_grad()
                critic_loss.backward()
                optimizer_C.step()

            z = torch.randn(batch_size, latent_dim).to(device)
            generated_imgs = generator(z)
            generator_loss = -torch.mean(critic(generated_imgs))

            optimizer_G.zero_grad()
            generator_loss.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch+1}/{epochs}] - Critic Loss: {critic_loss.detach().item():.4f}, Generator Loss: {generator_loss.detach().item():.4f}")
        generate_images(generator, device, num_images=8)
