import torch
import os
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
#import lpips
import torchvision.utils as vutils

from models import VAE

args = {
    "device": torch.device("cuda"),

    "beta": 0.1,#10,
    "disentangle": True,

    "gradient_clip": 2,
    "l1_lambda": 5e-7,
    "l2_lambda": 1e-6,

    "batch_size": 32,
    "lr": 5e-4,

    #"resolution_raw": [224, 400],
    #"resolution": [176, 400],
    "resolution": [224, 400],
    
    "latent_size": 2048,

    "test_amount": 32,

    #"is_mss": True,
    #"steps_anneal": 20000,
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


#def init_weights(m):
#    if isinstance(m, torch.nn.Linear):
#        torch.nn.init.xavier_uniform_(m.weight)
#        m.bias.data.fill_(0.01)

#transform = transforms.Compose([
#    transforms.Resize((args["resolution"][0], args["resolution"][1])),
#])

#image_dataset = FlatFolderDataset(root_dir="./images", transform=transform, raw_shape=(224, 400, 3))
image_dataset = FlatFolderDataset(root_dir="./images", raw_shape=(244, 400, 3))

train_dataset, test_dataset = torch.utils.data.random_split(image_dataset, [len(image_dataset) - args["test_amount"], args["test_amount"]])

train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False, pin_memory=True)

if __name__ == '__main__':
    model = VAE(args, len(train_dataset)).to(args["device"])
    #model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=args["lr"])

    #torch.backends.cudnn.deterministic = True


    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = 0
    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    #def lr_schedule(epoch):
    #    if epoch == 0:
    #       return 0.2
    #    elif epoch >= 1:
    #        return 1.0

    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-6)
    #scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)

    def test(save_path="reconstructions.png"):
        model.eval()

        test_iter = iter(test_loader)
        samples, _ = next(test_iter)
        samples = samples.to(args["device"])

        with torch.no_grad():
            _, reconstructions = model.muforward(samples)

        samples = samples.cpu()
        reconstructions = reconstructions.cpu()


        combined_images = []
        for i in range(samples.size(0)):
            original_img = samples[i]
            reconstructed_img = reconstructions[i]

            original_img = (original_img + 1) / 2
            reconstructed_img = (reconstructed_img + 1) / 2

            original_img = torch.clamp(original_img, 0, 1)
            reconstructed_img = torch.clamp(reconstructed_img, 0, 1)

            combined = torch.cat((original_img, reconstructed_img), dim=1)

            combined_images.append(combined)

        final_grid = torch.cat(combined_images, dim=2)

        plt.figure(figsize=(final_grid.size(2) / 100, final_grid.size(1) / 100), dpi=100)
        plt.imshow(final_grid.permute(1, 2, 0).numpy())
        plt.axis("off")

        plt.tight_layout(pad=0)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()



    def random_sample(save_path="random_samples.png"):
        model.eval()
        z = torch.randn((args["test_amount"], args["latent_size"])).to(args["device"])
        
        with torch.no_grad():
            images = model.decoder(z)
            
        grid = vutils.make_grid(images, nrow=args["test_amount"], padding=2, normalize=True)
        
        plt.figure(figsize=(args["test_amount"] * 2, 2))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')

        plt.tight_layout(pad=0)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    def latent_traversal(save_path="latent_traversals.png", max_grid_height=128):
        model.eval()

        steps = torch.arange(-1.0, 1.1, 0.2)
        latent_size = args["latent_size"]
        device = args["device"]
        
        all_images = []
        
        for i in range(latent_size):
            if len(all_images) >= max_grid_height:
                break
            
            z = torch.zeros((len(steps), latent_size), device=device)
            
            z[:, i] = steps

            with torch.no_grad():
                images = model.decoder(z)
            
            grid = vutils.make_grid(images, nrow=len(steps), padding=2, normalize=True)
            all_images.append(grid)

        full_grid = torch.cat(all_images, dim=1)

        plt.figure(figsize=(len(steps) * 2, min(latent_size, max_grid_height) * 2))
        plt.imshow(full_grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')

        plt.tight_layout(pad=0)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()



    checkpoint_path = "model.pt"
    def save_checkpoint(train_step, epoch):
        checkpoint = {
            'model_state_dict': {k: v for k, v in model.state_dict().items() if not k.startswith("lpips.")},
            'optimizer_state_dict': optimizer.state_dict(),
            #'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'train_step': train_step,
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint():
        if os.path.exists(checkpoint_path):        
            checkpoint = torch.load(checkpoint_path, map_location=args["device"])
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']
            train_step = checkpoint['train_step']
            
            return epoch, train_step

        return 0, 0

    total_updates = 10

    def train(epochs=5):
        start_epoch, train_step = load_checkpoint()

        #model.lpips = lpips.LPIPS(net='alex', spatial=True).to(args["device"])
        #train_step = 0

        #test()
        
        for epoch in range(start_epoch, epochs):
            model.train()
            running_loss = 0.0
            last_update = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

            for batch_idx, (images, _) in enumerate(progress_bar):
                train_step += 1

                progress = batch_idx / len(train_loader)
                if progress >= last_update / total_updates:
                    last_update += 1
                    test()
                    model.train()

                if images.size(0) < train_loader.batch_size:
                    continue

                if np.isnan(images).any():
                    print("NaN images")
                    continue

                images = images.to(args["device"])
                mu, logvar, z, decoded = model(images)

                loss = model.loss(train_step, images, decoded, z, mu, logvar)

                if torch.isnan(loss):
                    print("NaN encountered in loss, skipping batch.")
                    #break
                    continue

                optimizer.zero_grad()
                loss.backward()

                #torch.nn.utils.clip_grad_norm_(model.parameters(), args["gradient_clip"])

                optimizer.step()


                #scheduler.step(epoch + batch_idx / len(train_loader))

                running_loss += loss.item()

            #scheduler.step()
            
            save_checkpoint(train_step, epoch)
            #test()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

    load_checkpoint()
    #random_sample()
    #latent_traversal()
    test()
    train(epochs=1000)