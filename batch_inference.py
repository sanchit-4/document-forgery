import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UniversalForensicDINO
import os
import glob
import random

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/best_universal_model.pth"
OUTPUT_DIR = "inference_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset Paths
FAKE_DIR = "templates/Images/fakes"
REAL_DIR = "templates/Images/reals"

# Mapping
TYPE_MAP_REV = {0: "Real", 1: "Crop/Replace", 2: "Inpaint", 3: "Copy/Move", 4: "AI Generated"}

def get_mixed_batch(total_images=10):
    """
    Selects a mix of AI, Standard Fakes, and Real images.
    """
    all_fakes = glob.glob(os.path.join(FAKE_DIR, "*"))
    all_reals = glob.glob(os.path.join(REAL_DIR, "*"))
    
    # Filter for AI images (Gemini) vs Standard Fakes
    ai_fakes = [f for f in all_fakes if "gemini" in os.path.basename(f).lower()]
    standard_fakes = [f for f in all_fakes if "gemini" not in os.path.basename(f).lower()]
    
    selected_images = []
    
    # 1. Pick 2 AI Images (or as many as available)
    num_ai = min(2, len(ai_fakes))
    if num_ai > 0:
        selected_images.extend(random.sample(ai_fakes, num_ai))
        
    # 2. Pick 2 Real Images
    num_real = min(2, len(all_reals))
    if num_real > 0:
        selected_images.extend(random.sample(all_reals, num_real))
        
    # 3. Fill the rest with Standard Fakes
    remaining_slots = total_images - len(selected_images)
    if remaining_slots > 0 and len(standard_fakes) > 0:
        selected_images.extend(random.sample(standard_fakes, min(remaining_slots, len(standard_fakes))))
        
    print(f"Selected: {len(selected_images)} images ({num_ai} AI, {num_real} Real, {remaining_slots} Spliced)")
    return selected_images

def predict_and_viz(model, image_path, idx):
    img = cv2.imread(image_path)
    if img is None: return
    
    # Keep original for visualization
    orig_viz = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    # Preprocess
    transform = A.Compose([
        A.Resize(518, 518),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    tensor = transform(image=orig_viz)['image'].unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        bin_logits, mask_logits, type_logits = model(tensor)
        
    # Decode
    fake_prob = torch.sigmoid(bin_logits).item()
    
    type_probs = torch.softmax(type_logits, dim=1).cpu().numpy()[0]
    pred_type_idx = np.argmax(type_probs)
    pred_type_str = TYPE_MAP_REV.get(pred_type_idx, "Unknown")
    type_conf = type_probs[pred_type_idx]
    
    heatmap = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
    heatmap = cv2.resize(heatmap, (w, h))
    
    # Threshold heatmap for cleaner look
    heatmap_clean = heatmap.copy()
    heatmap_clean[heatmap_clean < 0.3] = 0 # Hide low confidence noise
    
    # Overlay
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_clean), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    
    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    
    filename = os.path.basename(image_path)
    
    # Info String
    info = (
        f"File: {filename}\n"
        f"FAKE Probability: {fake_prob*100:.2f}%\n"
        f"Type: {pred_type_str} ({type_conf*100:.2f}%)"
    )
    
    axs[0].imshow(orig_viz)
    axs[0].set_title("Original Input")
    axs[0].axis('off')
    
    axs[1].imshow(overlay)
    axs[1].set_title(info, fontsize=12, loc='left', backgroundcolor='white')
    axs[1].axis('off')
    
    save_path = f"{OUTPUT_DIR}/result_{idx:02d}_{pred_type_str}.jpg"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def main():
    # Load Model Once
    print(f"Loading Universal Model from {MODEL_PATH}...")
    model = UniversalForensicDINO(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    
    # Get Images
    images = get_mixed_batch(total_images=10)
    
    # Run Loop
    for i, img_path in enumerate(images):
        predict_and_viz(model, img_path, i)
        
    print("Batch Inference Complete.")

if __name__ == "__main__":
    main()