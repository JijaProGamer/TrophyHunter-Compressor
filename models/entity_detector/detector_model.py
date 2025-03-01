import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark_limit = 0
torch.backends.cudnn.allow_tf32 = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

def calculate_iou(pred_box, true_box):
    pred_xmin = pred_box[0] - pred_box[2] / 2
    pred_ymin = pred_box[1] - pred_box[3] / 2
    pred_xmax = pred_box[0] + pred_box[2] / 2
    pred_ymax = pred_box[1] + pred_box[3] / 2
    
    true_xmin = true_box[0] - true_box[2] / 2
    true_ymin = true_box[1] - true_box[3] / 2
    true_xmax = true_box[0] + true_box[2] / 2
    true_ymax = true_box[1] + true_box[3] / 2
    
    inter_xmin = max(pred_xmin, true_xmin)
    inter_ymin = max(pred_ymin, true_ymin)
    inter_xmax = min(pred_xmax, true_xmax)
    inter_ymax = min(pred_ymax, true_ymax)
    
    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    intersection = inter_width * inter_height
    
    pred_area = pred_box[2] * pred_box[3]
    true_area = true_box[2] * true_box[3]
    union = pred_area + true_area - intersection
    
    iou = intersection / union
    return iou
        
class LabeledImageDataset(Dataset):
    def __init__(self, grid_size, images_dir, labels_dir, vae_model, device):
        self.grid_size = grid_size
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.label_files = sorted(os.listdir(labels_dir))
        self.vae_model = vae_model
        self.device = device

        self.latent_vectors = []
        self.labels = []

        self._process_data()

    def _process_data(self):
        #i = 0
        for img_file, ann_file in tqdm(zip(self.image_files, self.label_files), total=len(self.image_files)):
            img_path = os.path.join(self.images_dir, img_file)
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            img = img / 127.5 - 1
            img = img.unsqueeze(0).to(self.device)

            #i += 1
            #if i > 1:
            #    break

            label_path = os.path.join(self.labels_dir, ann_file)
            annotation = self.__parse_an__(label_path, img)

            with torch.no_grad():
                #latent_vector = self.vae_model.zforward(img, disable_disentanglement=True).cpu()
                latent_vector = self.vae_model.encoder.forward_conv(img).cpu()

            self.latent_vectors.append(latent_vector)
            self.labels.append(annotation)#torch.tensor(annotation, dtype=torch.int8))

    def __parse_an__(self, label_path, img):
        _, _, H, W = img.shape
        grid_H, grid_W = self.grid_size
        boxes = []

        with open(label_path, 'r') as file:
            for line in file.readlines():
                line_split = line.strip().split()
                class_id = int(line_split[0])
                x_center = float(line_split[1]) * W
                y_center = float(line_split[2]) * H
                width = float(line_split[3]) * W
                height = float(line_split[4]) * H

                grid_x = int(x_center / W * grid_W)
                grid_y = int(y_center / H * grid_H)
                rel_x = (x_center / W * grid_W) - grid_x
                rel_y = (y_center / H * grid_H) - grid_y
                rel_w = width / W
                rel_h = height / H

                boxes.append([class_id, grid_x, grid_y, rel_x, rel_y, rel_w, rel_h])

        boxes = torch.tensor(boxes)
        return self.pad_annotations(boxes)

    def pad_annotations(self, boxes):
        num_boxes = boxes.shape[0]
        max_boxes = self.grid_size[0] * self.grid_size[1]

        if num_boxes < max_boxes:
            padding = torch.full((max_boxes - num_boxes, 7), -1.0)
            boxes = torch.cat([boxes, padding], dim=0)
        elif num_boxes > max_boxes:
            boxes = boxes[:max_boxes]
        return boxes
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        return self.latent_vectors[idx], self.labels[idx]
    
class DetectorModel(nn.Module):
    def __init__(self, num_classes, filters):
        super(DetectorModel, self).__init__()
        self.num_classes = num_classes
        self.num_outputs = 5 + num_classes

        layers = []

        last_filters = 16
        for filter_amount in filters:
            layers.append(nn.Conv2d(last_filters, filter_amount, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(filter_amount))
            layers.append(nn.GELU())
            last_filters = filter_amount

        self.layers = nn.Sequential(*layers)


        self.out = nn.Conv2d(filters[-1], self.num_outputs, kernel_size=3, padding=1)
                
    def forward(self, x):
        x = self.layers(x)
        x = self.out(x)

        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, H, W, self.num_outputs)
        
        x[..., 0:2] = torch.sigmoid(x[..., 0:2])
        x[..., 2:4] = torch.exp(x[..., 2:4])
        x[..., 4] = torch.sigmoid(x[..., 4])
        x[..., 5:] = torch.softmax(x[..., 5:], dim=-1)
        
        return x

class Detector:
    def __init__(self, num_classes, grid_size, vae_model, device):
        self.num_classes = num_classes
        self.vae_model = vae_model
        self.device = device
        self.grid_size = grid_size

        self.criterion = nn.CrossEntropyLoss()


    def make_model(self, lr, filters):
        self.detector_model = DetectorModel(num_classes=self.num_classes, filters=filters).to(self.device)
        self.optimizer = torch.optim.Adam(self.detector_model.parameters(), lr=lr)
    
    def train_step(self, latent_vectors, annotations, lambda_coord=5, lambda_noobj=0.5):
        self.optimizer.zero_grad()
        outputs = self.detector_model(latent_vectors)

        pred_boxes = outputs[..., 0:4]
        pred_obj = outputs[..., 4]
        pred_class = outputs[..., 5:]

        B, H, W, _ = pred_boxes.shape

        true_boxes = annotations[..., 2:6]
        true_obj = annotations[..., 6].view(B, H, W)
        true_obj = torch.where(true_obj == -1, torch.tensor(0).cuda(), true_obj)
        true_class = annotations[..., 0].long()
        true_class = torch.where(true_class == -1, torch.tensor(0).cuda(), true_class)
        
        true_boxes = true_boxes.view(B, H, W, 4)

        box_loss = lambda_coord * F.mse_loss(pred_boxes, true_boxes, reduction='sum')
        obj_loss = F.binary_cross_entropy(pred_obj, true_obj, reduction='sum')
        noobj_loss = lambda_noobj * F.binary_cross_entropy(pred_obj, torch.zeros_like(pred_obj), reduction='sum')
        class_loss = F.cross_entropy(pred_class.view(-1, self.num_classes), true_class.view(-1), reduction='sum')

        total_loss = box_loss + obj_loss + noobj_loss + class_loss
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), outputs
    def train(self, epochs, dataloader, validation_dataloader):
        # train

        for epoch in range(epochs):
            self.detector_model.train()

            for latent_vectors, annotations in dataloader: 
                latent_vectors = latent_vectors.squeeze(1).to(self.device)
                annotations = annotations.to(self.device)

                self.train_step(latent_vectors, annotations)

        # validation

        #avg_iou, precision, recall = self.validate(validation_dataloader, 0.5)
        loss = self.validate(validation_dataloader, 0.5)

        return loss#recall * 3 + precision * 2 + avg_iou
    
    def validate(self, validation_dataloader, iou_threshold=0.5):
        self.detector_model.eval()
        
        lambda_coord = 5
        lambda_noobj = 0.5

        with torch.no_grad():
            for latent_vectors, annotations in validation_dataloader:
                latent_vectors = latent_vectors.squeeze(1).to(self.device)
                annotations = annotations.to(self.device)

                outputs = self.detector_model(latent_vectors)

                pred_boxes = outputs[..., 0:4]
                pred_obj = outputs[..., 4]
                pred_class = outputs[..., 5:]

                B, H, W, _ = pred_boxes.shape

                true_boxes = annotations[..., 2:6]
                true_obj = annotations[..., 6].view(B, H, W)
                true_obj = torch.where(true_obj == -1, torch.tensor(0).cuda(), true_obj)
                true_class = annotations[..., 0].long()
                true_class = torch.where(true_class == -1, torch.tensor(0).cuda(), true_class)
                
                true_boxes = true_boxes.view(B, H, W, 4)

                box_loss = lambda_coord * F.mse_loss(pred_boxes, true_boxes, reduction='sum')
                obj_loss = F.binary_cross_entropy(pred_obj, true_obj, reduction='sum')
                noobj_loss = lambda_noobj * F.binary_cross_entropy(pred_obj, torch.zeros_like(pred_obj), reduction='sum')
                class_loss = F.cross_entropy(pred_class.view(-1, self.num_classes), true_class.view(-1), reduction='sum')

                total_loss = box_loss + obj_loss + noobj_loss + class_loss

        return total_loss
