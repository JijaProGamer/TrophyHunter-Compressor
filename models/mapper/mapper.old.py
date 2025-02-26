import torch
import sys
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torchsummary import summary
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models import VAE

args = {
    "device": torch.device("cuda"),
    "disentangle": True,
    "resolution": [224, 400],
    "latent_size": 2048,
    "loader_batch_size": 64,

    "test_image_folder": "./test_images/",
    "image_folder": "./annotated_images/images/",
    "annotation_folder": "./annotated_images/annotations/",
    "small_map_size": (24, 15),
    "map_size": (72, 45),
    "num_blocks": 6,
    "batch_size": 64,

    "lr": 5e-4,

}

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark_limit = 0
torch.backends.cudnn.allow_tf32 = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

VAEmodel = VAE(args, 0).to(args["device"])

def load_vae_checkpoint(checkpoint_path = "model.pt"):
    checkpoint = torch.load(checkpoint_path, map_location=args["device"])
    VAEmodel.load_state_dict(checkpoint['model_state_dict'])

load_vae_checkpoint("../../model.pt")
VAEmodel.eval()
        
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

        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.annotation_files = sorted([f for f in os.listdir(annotation_folder) if f.endswith('.txt')])

        assert len(self.image_files) == len(self.annotation_files), "Mismatch between images and annotations"

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
                latent_vector = self.vae_model.zforward(img).cpu()

            self.latent_vectors.append(latent_vector)
            self.annotations.append(torch.tensor(annotation, dtype=torch.int8))

    def __len__(self):
        return len(self.latent_vectors)

    def __getitem__(self, idx):
        return self.latent_vectors[idx], self.annotations[idx]
    
class MapperModel(nn.Module):
    def __init__(self, latent_size, small_map_size, map_size, num_classes):
        super(MapperModel, self).__init__()
        self.latent_size = latent_size
        self.map_size = map_size
        self.small_map_size = small_map_size
        self.num_classes = num_classes
        
        self.fc1 = nn.Linear(latent_size, np.prod(small_map_size), bias=False)
        self.bn1 = nn.BatchNorm1d(np.prod(small_map_size))

        self.fc2 = nn.Linear(np.prod(small_map_size), np.prod(small_map_size), bias=False)
        self.bn2 = nn.BatchNorm1d(np.prod(small_map_size))


        self.fc3 = nn.Linear(np.prod(small_map_size), np.prod(map_size) * num_classes)
       # self.fc3 = nn.Linear(np.prod(small_map_size), np.prod(map_size))

        #self.conv1 = nn.Conv2d(1, self.num_classes, kernel_size=3, padding=1)
                
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        #x = F.gelu(self.bn1(self.fc1(x)))
        #x = F.gelu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        
        x = x.view(-1, self.num_classes, self.map_size[1], self.map_size[0])
        #x = x.view(-1, 1, self.map_size[1], self.map_size[0])

        #x = self.conv1(x)

        return x
    
class Mapper:
    def __init__(self, image_folder, annotation_folder, small_map_size, map_size, vae_model, latent_size, num_classes, lr=5e-4, test_image_folder=None):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.test_image_folder = test_image_folder
        self.vae_model = vae_model
        self.latent_size = latent_size
        self.num_classes = num_classes
        
        self.mapper_model = MapperModel(latent_size=latent_size, small_map_size=small_map_size, map_size=map_size, num_classes=num_classes).to(args["device"])
        
        self.optimizer = torch.optim.Adam(self.mapper_model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        summary(self.mapper_model, (latent_size, ), device="cuda")

    def predict(self, latent_vector):
        with torch.no_grad():
            predicted_annotation = self.mapper_model(latent_vector)
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

    """def train_step(self, latent_vectors, annotations):        
        self.optimizer.zero_grad()
        
        outputs = self.mapper_model(latent_vectors)
        outputs = torch.argmax(outputs, dim=1)

        annotations = annotations.reshape(outputs.shape)


        
        return 0,outputs"""
    def train_step(self, latent_vectors, annotations, l1_lambda=1e-5):
        self.optimizer.zero_grad()

        outputs = self.mapper_model(latent_vectors)
        annotations = annotations.reshape([outputs.shape[0], outputs.shape[2], outputs.shape[3]]).long()
        loss = self.criterion(outputs, annotations)

        #l1_penalty = 0

        #for name, param in self.mapper_model.named_parameters():
        #    if "fc" in name and "weight" in name:
        #        l1_penalty += param.norm(1, dim=1).sum()

        #for name, param in self.mapper_model.named_parameters():
        #    if "fc" in name and "weight" in name:
        #        abs_weights = param.abs()
        #        max_weight_per_neuron = abs_weights.max(dim=1, keepdim=True)[0]
        #        l1_penalty += (abs_weights - max_weight_per_neuron).sum()

        #loss += l1_lambda * l1_penalty

        loss.backward()
        self.optimizer.step()

        return loss, outputs




mapper = Mapper(
    image_folder=args["image_folder"], 
    annotation_folder=args["annotation_folder"], 
    test_image_folder=args["test_image_folder"], 
    small_map_size=args["small_map_size"],
    map_size=args["map_size"], 
    vae_model=VAEmodel, 
    latent_size=args["latent_size"], 
    num_classes=args["num_blocks"], 
    lr=args["lr"]
)

dataset = AnnotatedImageDataset(args["image_folder"], args["annotation_folder"], VAEmodel, args["device"])
dataloader = DataLoader(dataset, batch_size=args["loader_batch_size"], shuffle=True)

num_epochs = 100
for epoch in range(num_epochs):
    mapper.mapper_model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for latent_vectors, annotations in tqdm(dataloader):
        latent_vectors = latent_vectors.squeeze(1).to(args["device"])
        annotations = annotations.to(args["device"])

        loss, outputs = mapper.train_step(latent_vectors, annotations)

        running_loss += loss

        _, predicted = torch.max(outputs, 1)
        predicted = predicted.view(-1)
        annotations = annotations.view(-1) 

        correct_preds += (predicted == annotations).sum().item()
        total_preds += annotations.numel()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100 * correct_preds / total_preds
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")


mapper.mapper_model.eval()



image_index = 0



test_image = True
real_annotations = False

if test_image:
    test_image_files = sorted([f for f in os.listdir(args["test_image_folder"]) if f.endswith(('.png', '.jpg', '.jpeg'))])
    test_image_path = os.path.join(args["test_image_folder"], test_image_files[image_index])
    image = Image.open(test_image_path).convert("RGB")

    image = np.array(image)
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    image = image / 127.5 - 1
    image = image.unsqueeze(0).to(args["device"])

    with torch.no_grad():
        latent_vector = VAEmodel.zforward(image).to(args["device"])
    
    annotations = mapper.predict(latent_vector)
    annotations = annotations[0,...]
else:
    latent_vector, annotation = dataset[image_index]
    latent_vector = latent_vector.to(args["device"])

    image_path = os.path.join(args["image_folder"], dataset.image_files[image_index])
    image = Image.open(image_path).convert("RGB")

    image = np.array(image)
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    image = image / 127.5 - 1
    image = image.unsqueeze(0).to(args["device"])

    if real_annotations:
        annotations = annotation.cpu().numpy().reshape(args["map_size"][1], args["map_size"][0])
    else:
        annotations = mapper.predict(latent_vector)
        annotations = annotations[0,...]

mapper.display_test(image, annotations)
