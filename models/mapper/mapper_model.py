import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

args = {
    "device": torch.device("cuda"),
    "disentangle": True,
    "resolution": [224, 400],
    "latent_size": 2048,

    "test_image_folder": "./test_images/",
    "image_folder": "./annotated_images/images/",
    "annotation_folder": "./annotated_images/annotations/",
    "validation_image_folder": "./validation_images/images/",
    "validation_annotation_folder": "./validation_images/annotations/",
    "map_size": (72, 45),
    "num_blocks": 6,
    "batch_size": 16,

    "lr": 5e-4,
}

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark_limit = 0
torch.backends.cudnn.allow_tf32 = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        
classColors = {
    0: "#FFFFFF00",                  # Transparent
    1: "#FF5733",                   # Red-ish color
    2: "#888888",                   # Gray
    3: "#00BFFF",                   # Blue
    4: "#0d5e0d",                   # Dark Green
    5: "#00FF00"                    # Bright Green
}

class AnnotatedImageDataset(Dataset):
    def __init__(self, image_folder, annotation_folder, vae_model, device):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.device = device
        self.vae_model = vae_model

        # Get all annotation filenames (without extensions)
        annotation_filenames = {f.rsplit('.', 1)[0] for f in os.listdir(annotation_folder) if f.endswith('.txt')}

        # Filter image files to only those with matching annotation
        self.image_files = sorted([f for f in os.listdir(image_folder) 
                                   if f.rsplit('.', 1)[0] in annotation_filenames and f.endswith(('.png', '.jpg', '.jpeg'))])

        self.annotation_files = sorted([f + '.txt' for f in annotation_filenames if f + '.txt' in os.listdir(annotation_folder)])

        self.latent_vectors = []
        self.annotations = []

        self._process_data()

    def _process_data(self):
        for img_file, ann_file in tqdm(zip(self.image_files, self.annotation_files), total=len(self.image_files)):
            img_path = os.path.join(self.image_folder, img_file)
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            img = img / 127.5 - 1
            img = img.unsqueeze(0).to(args["device"])

            ann_path = os.path.join(self.annotation_folder, ann_file)
            with open(ann_path, "r") as f:
                annotation = np.loadtxt(f, dtype=np.int32).reshape(-1)

            with torch.no_grad():
                latent_vector = self.vae_model.encoder.forward_conv(img).cpu()

            self.latent_vectors.append(latent_vector)
            self.annotations.append(torch.tensor(annotation, dtype=torch.int8))

    def __len__(self):
        return len(self.latent_vectors)

    def __getitem__(self, idx):
        return self.latent_vectors[idx], self.annotations[idx]

    
class MapperModel(nn.Module):
    def __init__(self, num_classes, begin_filters, mid_filters, final_filters):
        super(MapperModel, self).__init__()
        self.num_classes = num_classes

        begin_layers = []

        last_filters = 16
        for idx, filters in enumerate(begin_filters):
            padding = 1
            if idx == 0:
                padding = 2
            
            begin_layers.append(nn.Conv2d(last_filters, filters, kernel_size=3, padding=padding))
            begin_layers.append(nn.BatchNorm2d(filters))
            begin_layers.append(nn.GELU())
            last_filters = filters

        self.begin = nn.Sequential(*begin_layers)

        self.up = nn.ConvTranspose2d(begin_filters[-1], mid_filters, kernel_size=3, stride=3, padding=1)
        self.bn_up = nn.BatchNorm2d(mid_filters)


        out_layers = []

        last_filters = mid_filters
        for filters in final_filters:
            #out_layers.append(nn.Conv2d(last_filters, filters, kernel_size=3, padding=1))
            out_layers.append(nn.Conv2d(last_filters, filters, kernel_size=1, padding=0))
            out_layers.append(nn.BatchNorm2d(filters))
            out_layers.append(nn.GELU())
            last_filters = filters

        self.end = nn.Sequential(*out_layers)

        self.out = nn.Conv2d(final_filters[-1], self.num_classes, kernel_size=3, padding=1)
                
    def forward(self, x):
        x = self.begin(x)

        x = F.gelu(self.bn_up(self.up(x)))
        x = x[:, :, :-1, :-7]

        x = self.end(x)

        x = self.out(x)

        return x

class Mapper:
    def __init__(self, image_folder, annotation_folder, num_classes, vae_model, device, test_image_folder=None):
        self.image_folder = image_folder
        self.num_classes = num_classes
        self.annotation_folder = annotation_folder
        self.test_image_folder = test_image_folder
        self.vae_model = vae_model
        self.device = device

        self.criterion = nn.CrossEntropyLoss()

    def predict(self, latent_vector):
        with torch.no_grad():
            predicted_annotation = self.mapper_model(latent_vector.squeeze(0))
            predicted_annotation = torch.argmax(predicted_annotation, dim=1).cpu().numpy()

        return predicted_annotation

    def display_test(self, image, annotations):
        image = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        image = (image + 1) / 2
        
        height, width = annotations.shape

        colored_image = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(width):
            for j in range(height):
                block_class_id = annotations[j, i].item()
                color = self.get_color_from_class(block_class_id)
                colored_image[j, i] = color

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title("Original Image")
        
        axes[1].imshow(colored_image)
        axes[1].axis('off')
        axes[1].set_title("Predicted Map")

        plt.tight_layout()
        plt.show()

    def get_color_from_class(self, class_id):
        hex_color = classColors.get(class_id, "#FFFFFF")
        return np.array([int(hex_color[i:i+2], 16) for i in (1, 3, 5)], dtype=np.uint8)

    def make_model(self, lr, begin_filters, mid_filters, final_filters):
        self.mapper_model = MapperModel(begin_filters=begin_filters, mid_filters=mid_filters, final_filters=final_filters, num_classes=self.num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.mapper_model.parameters(), lr=lr)
    
    def train_step(self, latent_vectors, annotations, l1_lambda=1e-5):
        self.optimizer.zero_grad()

        outputs = self.mapper_model(latent_vectors)
        annotations = annotations.reshape([outputs.shape[0], outputs.shape[2], outputs.shape[3]]).long()
        loss = self.criterion(outputs, annotations)

        #l1_loss = sum(torch.norm(param, p=1) for param in self.mapper_model.parameters()) # L1
        #l1_loss = sum(torch.norm(param, p=2) for param in self.mapper_model.parameters())  # L2 
        #loss += l1_loss * l1_lambda

        loss.backward()
        self.optimizer.step()

        return loss, outputs
    def train(self, epochs, dataloader, validation_dataloader):
        # train

        for epoch in range(epochs):
            self.mapper_model.train()

            running_loss = 0.0
            correct_preds = 0
            total_preds = 0

            for latent_vectors, annotations in dataloader: 
                latent_vectors += torch.randn_like(latent_vectors) * 0.1 


                latent_vectors = latent_vectors.squeeze(1).to(args["device"])
                annotations = annotations.to(args["device"])

                loss, outputs = self.train_step(latent_vectors, annotations)

                running_loss += loss

                _, predicted = torch.max(outputs, 1)
                predicted = predicted.view(-1)
                annotations = annotations.view(-1) 

                correct_preds += (predicted == annotations).sum().item()
                total_preds += annotations.numel()

        # validation

        self.mapper_model.eval()

        correct_preds = 0
        total_preds = 0
        
        for latent_vectors, annotations in validation_dataloader: 
            latent_vectors = latent_vectors.squeeze(1).to(args["device"])
            annotations = annotations.to(args["device"])

            outputs = self.mapper_model(latent_vectors)

            _, predicted = torch.max(outputs, 1)
            predicted = predicted.view(-1)
            annotations = annotations.view(-1) 

            correct_preds += (predicted == annotations).sum().item()
            total_preds += annotations.numel()

        return 100 * correct_preds / total_preds