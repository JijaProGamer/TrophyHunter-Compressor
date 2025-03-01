
import sys
import torch
import os
from PIL import Image
import optuna
import numpy as np
import cv2
from torchsummary import summary
from torch.utils.data import DataLoader

from detector_model import Detector, LabeledImageDataset

args = {
    "device": torch.device("cuda"),
    "disentangle": True,
    "resolution": [224, 400],
    "latent_size": 2048,

    "grid_size": [int(224/16), int(400/16)],
    "batch_size": 16,
    "lr": 5e-4,
}

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models import VAE

VAEmodel = VAE(args, 0).to(args["device"])

def load_vae_checkpoint(checkpoint_path = "model.pt"):
    checkpoint = torch.load(checkpoint_path, map_location=args["device"])
    VAEmodel.load_state_dict(checkpoint['model_state_dict'])

load_vae_checkpoint("../../model.pt")
VAEmodel.eval()

classes = ["Ball", "Enemy", "Friendly", "Gem", "Hot_Zone", "Me", "PP", "PP_Box", "Safe_Enemy", "Safe_Friendly"]
detector = Detector(
    grid_size=args["grid_size"],
    num_classes=len(classes),
    vae_model=VAEmodel, 
    device=args["device"], 
)

dataset = LabeledImageDataset(args["grid_size"], "./dataset/train/images/", "./dataset/train/labels/", VAEmodel, args["device"])
dataloader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=True)

validation_dataset = LabeledImageDataset(args["grid_size"], "./dataset/valid/images/", "./dataset/valid/labels/", VAEmodel, args["device"])
validation_dataloader = DataLoader(validation_dataset, batch_size=args["batch_size"])

best_accuracy = 0.0
def objective(trial):
    global best_accuracy
    
    num_layers = trial.suggest_int('num_layers', 1, 6)
    filters = [2 ** trial.suggest_int(f'filter_{i}', 3, 8) for i in range(num_layers)]

    lr = trial.suggest_float("lr", 1e-5, 1e-3)
    epochs = trial.suggest_int("epochs", 10, 200)

    detector.make_model(lr=lr, filters=filters)
    accuracy = detector.train(epochs, dataloader, validation_dataloader)


    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save({
            'model_state_dict': detector.detector_model.state_dict(),
            'accuracy': accuracy,
            'architecture': {
                'num_layers': num_layers,
                'filters': filters,
                'lr': lr,
                'epochs': epochs
            }
        }, "best_detector_model.pt")
    
    return accuracy

#study = optuna.create_study(direction="maximize") 
study = optuna.create_study(direction="minimize") 
study.optimize(objective, n_trials=100)




"""best_trial = study.best_trial

best_middle_filters = 2 ** best_trial.params['middle_filters']
best_num_layers_before = best_trial.params['num_layers_before']
best_num_layers_after = best_trial.params['num_layers_after']

best_filters_before = [2 ** best_trial.params[f'filters_before_{i}'] for i in range(best_num_layers_before)]
best_filters_after = [2 ** best_trial.params[f'filters_after_{i}'] for i in range(best_num_layers_after)]

print("best_middle_filters", best_middle_filters)
print("best_filters_before", best_filters_before)
print("best_filters_after", best_filters_after)
print("lr", best_trial.params["lr"])
print("epochs", best_trial.params["epochs"])"""

"""checkpoint = torch.load("best_detector_model.pt", map_location=args["device"])
best_arch = checkpoint['architecture']

detector.make_model(
    lr=best_arch['lr'], 
    begin_filters=best_arch['filters_before'], 
    mid_filters=best_arch['middle_filters'], 
    final_filters=best_arch['filters_after']
)

detector.detector_model.load_state_dict(checkpoint['model_state_dict'])
detector.detector_model.eval()

summary(detector.detector_model, (16, int(args["resolution"][0]/16), int(args["resolution"][1]/16)), device="cuda")




test_image_files = sorted([f for f in os.listdir(args["test_image_folder"]) if f.endswith(('.png', '.jpg', '.jpeg'))])

original_images = []
annotation_images = []

for image_index, test_image_file in enumerate(test_image_files):
    test_image_path = os.path.join(args["test_image_folder"], test_image_file)
    image = Image.open(test_image_path).convert("RGB")

    image_np = np.array(image)
    image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1)
    image_tensor = image_tensor / 127.5 - 1
    image_tensor = image_tensor.unsqueeze(0).to(args["device"])

    with torch.no_grad():
        latent_vector = VAEmodel.encoder.forward_conv(image_tensor).to(args["device"]).unsqueeze(0)

    annotations = detector.predict(latent_vector)
    annotations = annotations[0, ...]

    height, width = annotations.shape
    colored_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(width):
        for j in range(height):
            block_class_id = annotations[j, i].item()
            colored_image[j, i] = detector.get_color_from_class(block_class_id)

    original_images.append(image_np)
    annotation_images.append(colored_image)

top_row = np.hstack(original_images)
bottom_row = np.hstack(annotation_images)

target_width = top_row.shape[1]
original_width = bottom_row.shape[1]
original_height = bottom_row.shape[0]

scale_factor = target_width / original_width
new_height = int(original_height * scale_factor)

bottom_row_resized = cv2.resize(bottom_row, (target_width, new_height), interpolation=cv2.INTER_NEAREST)

if new_height > top_row.shape[0]:  
    bottom_row_resized = bottom_row_resized[:top_row.shape[0], :, :]
else:  
    top_row = top_row[:new_height, :, :]

final_combined = np.vstack((top_row, bottom_row_resized))

final_image_path = os.path.join("./", "mapper_tests.png")
cv2.imwrite(final_image_path, cv2.cvtColor(final_combined, cv2.COLOR_RGB2BGR))

image_index = 1

test_image_files = sorted([f for f in os.listdir(args["test_image_folder"]) if f.endswith(('.png', '.jpg', '.jpeg'))])
test_image_path = os.path.join(args["test_image_folder"], test_image_files[image_index])
image = Image.open(test_image_path).convert("RGB")

image = np.array(image)
image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
image = image / 127.5 - 1
image = image.unsqueeze(0).to(args["device"])

with torch.no_grad():
    latent_vector = VAEmodel.encoder.forward_conv(image).to(args["device"]).unsqueeze(0)

for _ in range(1, 100):
    annotations = detector.predict(torch.rand_like(latent_vector))
    annotations = annotations[0,...]

average_time = 0
times = 0

import time
for _ in range(1, 100):
    start = time.time()
    annotations = detector.predict(latent_vector)
    annotations = annotations[0,...]
    times += 1
    average_time += time.time() - start

print(average_time / times * 1000)"""