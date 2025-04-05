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
    "device": "cuda",#torch.device("cuda"),

    "gradient_clip": 2,
    "l1_lambda": 5e-7,
    "l2_lambda": 1e-6,

    "batch_size": 64,
    #"gradient_steps": 4,
    "lr": 1e-3,#5e-4,

    "embedding_dim": 4,
    "num_embeddings": 1024,
    "use_ema": True,
    "decay": 0.99,
    "beta": 0.25,
    "epsilon": 1e-5,

    "resolution": [224, 400],
    
    "test_amount": 32,
}

class FlatFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, raw_shape=(128, 128, 3)):
        self.image_paths = list(Path(root_dir).rglob("*.*"))
        self.len = len(self.image_paths)
        self.transform = transform
        self.raw_shape = raw_shape

    def __len__(self):
        return self.len

    def _load_raw_image(self, image_path):
        with open(image_path, "rb") as f:
            raw_data = np.frombuffer(f.read(), dtype=np.uint8)
        image = raw_data.reshape(self.raw_shape)
        return image
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        if image_path.suffix.lower() == ".raw":
            image = self._load_raw_image(image_path)
        else:
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        image = image / 127.5 - 1

        #if self.transform:
        #    image = self.transform(image)
        
        return image

image_dataset = FlatFolderDataset(root_dir="./images", raw_shape=(244, 400, 3))

train_dataset, test_dataset = torch.utils.data.random_split(image_dataset, [len(image_dataset) - args["test_amount"], args["test_amount"]])

#train_loader = DataLoader(train_dataset, batch_size=args["batch_size"])
train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False, pin_memory=True)

#scaler = torch.amp.GradScaler(device="cuda")

if __name__ == '__main__':
    model = VAE(args, len(train_dataset))#.to(args["device"])

    optimizer = optim.Adam(model.parameters(), lr=args["lr"])

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = 0
    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    def test(save_path="reconstructions.png"):
        model.eval()

        test_iter = iter(test_loader)
        samples = next(test_iter)
        samples = samples.to(args["device"])

        with torch.no_grad():
            _, reconstructions, _, _, _, _ = model(samples)

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


    checkpoint_path = "model.pt"
    def save_checkpoint(epoch):
        checkpoint = {
            'model_state_dict': {k: v for k, v in model.state_dict().items() if not k.startswith("lpips.")},
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint():
        if os.path.exists(checkpoint_path):        
            checkpoint = torch.load(checkpoint_path, map_location=args["device"])
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            
            return epoch

        return 0

    total_updates = 100

    def train(epochs):
        start_epoch = load_checkpoint()
        
        for epoch in range(start_epoch, epochs):
            model.train()
            running_loss = 0.0
            running_scale = 0.0
            last_update = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

            for batch_idx, images in enumerate(progress_bar):
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

                #with torch.amp.autocast(device_type="cuda"):
                _, decoded, _, dictionary_loss, commitment_loss, _ = model(images)

                loss = model.loss(images, decoded, dictionary_loss, commitment_loss)


                if torch.isnan(loss):
                    print(" NaN encountered in recon loss, skipping batch.")
                    continue

                """scaler.scale(loss).backward()

                if (batch_idx + 1) % args["gradient_steps"] == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()"""
                
                loss.backward()


                optimizer.step()                
                optimizer.zero_grad()


                running_loss += loss.item()

            save_checkpoint(epoch)

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

    load_checkpoint()
    test()
    train(epochs=1000)