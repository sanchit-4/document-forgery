import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
from model import ForensicDINO
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Load the BEST model (Epoch 14), not the last one (Epoch 15 dropped a bit)
MODEL_PATH = "checkpoints/best_heatmap_model.pth" 
DATA_ROOT = "test" # Adjust to your folder

# --- LOAD MODEL ---
print(f"Loading model from {MODEL_PATH}...")
model = ForensicDINO(freeze_dino=False).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint)
model.eval()

# --- TRANSFORM ---
transform = A.Compose([
    A.Resize(518, 518),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def predict_and_viz(image_path):
    # 1. Load Image
    original_img = cv2.imread(image_path)
    if original_img is None: return
    
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w, _ = original_img.shape
    
    # 2. Preprocess
    augmented = transform(image=original_img)
    img_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    
    # 3. Predict
    with torch.no_grad():
        cls_logits, mask_logits = model(img_tensor)
    
    # 4. Process Output
    cls_score = torch.sigmoid(cls_logits).item()
    heatmap = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
    heatmap = cv2.resize(heatmap, (w, h))
    
    # CLEAN UP: Thresholding
    # Only show red if model is > 50% confident about that pixel
    heatmap_clean = heatmap.copy()
    heatmap_clean[heatmap_clean < 0.5] = 0  # Hide low confidence noise
    
    # Visualization Overlay
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_clean), cv2.COLORMAP_JET)
    
    # If heatmap is empty (model detected fake globally but not locally), use a different blend
    if heatmap_clean.max() == 0:
        print("Warning: Fake detected globally, but no specific region found.")
    
    overlay = cv2.addWeighted(original_img, 0.7, heatmap_colored, 0.3, 0)
    
    # 6. Plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    axs[0].imshow(original_img)
    axs[0].set_title("Input Fake ID")
    axs[0].axis('off')
    
    axs[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    axs[1].set_title(f"Heatmap (Blue=Real, Red=Fake)")
    axs[1].axis('off')
    
    axs[2].imshow(overlay)
    axs[2].set_title(f"Prediction (Fake Prob: {cls_score:.4f})")
    axs[2].axis('off')
    
    filename = os.path.basename(image_path)
    plt.savefig(f"final_result_{filename}")
    print(f"Saved visualization to final_result_{filename}")
    plt.close()

# --- RUN ON RANDOM FAKES ---
# Recursively find all JPG/PNG
fake_images = []
for ext in ['*.jpg', '*.png', '*.jpeg']:
    fake_images.extend(glob.glob(os.path.join(DATA_ROOT, '**', ext), recursive=True))

if len(fake_images) > 0:
    selected_fakes = random.sample(fake_images, min(5, len(fake_images)))
    for img_path in selected_fakes:
        print(f"Testing: {img_path}")
        predict_and_viz(img_path)
else:
    print(f"No images found in {DATA_ROOT}")