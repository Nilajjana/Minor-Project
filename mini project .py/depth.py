import torch
import cv2
import numpy as np
from rembg import remove
from PIL import Image

def load_model():
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    model.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, transform, device

def remove_bg(image_path):
    img = Image.open(image_path)
    img = remove(img)
    img = np.array(img)

    #REMOVE ALPHA CHANNEL
    if img.shape[2] == 4:
        img = img[:, :, :3]

    return img

def get_depth(image_path, model, transform, device):

    img = cv2.imread(image_path)
    if img is None:
       raise ValueError(f"Image not found at path: {image_path}")

    img = cv2.resize(img, (384, 384))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    if len(input_batch.shape) == 3:
       input_batch = input_batch.unsqueeze(0)

    print("input shape:", input_batch.shape)

    with torch.no_grad():
        depth = model(input_batch)

    depth = depth.squeeze().cpu().numpy()

    # normalize
    depth = (depth - depth.min()) / (depth.max() - depth.min())

    depth = np.power(depth, 1.5)

    depth = cv2.resize(depth, (img.shape[1], img.shape[0]))

    depth = cv2.GaussianBlur(depth, (3,3), 0)

    print("Depth shape:", depth.shape)
    print("Image shape:", img.shape)
    return depth, img
