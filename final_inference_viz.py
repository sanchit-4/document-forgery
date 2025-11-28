# import torch
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from model import UniversalForensicDINO
# import os
# import glob
# import random

# # --- CONFIG ---
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_PATH = "checkpoints/refined_universal_model.pth"
# OUTPUT_DIR = "final_visuals"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# TYPE_MAP_REV = {0: "Real", 1: "Crop/Replace", 2: "Inpaint", 3: "Copy/Move", 4: "AI Generated"}

# def predict_and_viz(model, image_path, idx):
#     img = cv2.imread(image_path)
#     if img is None: return
    
#     orig_viz = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     h, w, _ = img.shape
    
#     transform = A.Compose([
#         A.Resize(518, 518),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2()
#     ])
#     tensor = transform(image=orig_viz)['image'].unsqueeze(0).to(DEVICE)
    
#     with torch.no_grad():
#         bin_logits, mask_logits, type_logits = model(tensor)
        
#     # Decode
#     fake_prob = torch.sigmoid(bin_logits).item()
#     type_probs = torch.softmax(type_logits, dim=1).cpu().numpy()[0]
#     pred_type_idx = np.argmax(type_probs)
#     pred_type_str = TYPE_MAP_REV.get(pred_type_idx, "Unknown")
    
#     # --- HEATMAP PROCESSING ---
#     heatmap = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
#     heatmap = cv2.resize(heatmap, (w, h))
    
#     # 1. The Raw "Blue" Heatmap (JET Colormap)
#     # We want 0.0 -> Blue, 1.0 -> Red
#     heatmap_norm = np.uint8(255 * heatmap)
#     raw_blue_mask = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    
#     # 2. The Clean Overlay
#     # Hard Threshold to remove the faint wash
#     heatmap_clean = heatmap.copy()
#     heatmap_clean[heatmap_clean < 0.4] = 0 # Cut off weak noise
#     heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_clean), cv2.COLORMAP_JET)
#     overlay = cv2.addWeighted(img, 0.7, heatmap_colored, 0.3, 0)
    
#     # --- PLOTTING ---
#     fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
#     axs[0].imshow(orig_viz)
#     axs[0].set_title("Input Image")
#     axs[0].axis('off')
    
#     axs[1].imshow(cv2.cvtColor(raw_blue_mask, cv2.COLOR_BGR2RGB))
#     axs[1].set_title("Raw Heatmap (Blue=Low, Red=High)")
#     axs[1].axis('off')
    
#     axs[2].imshow(overlay)
#     axs[2].set_title(f"Prob: {fake_prob*100:.1f}% | Type: {pred_type_str}")
#     axs[2].axis('off')
    
#     save_path = f"{OUTPUT_DIR}/viz_{idx:02d}_{pred_type_str}.jpg"
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Saved: {save_path}")

# def main():
#     print(f"Loading Model...")
#     model = UniversalForensicDINO(num_classes=5).to(DEVICE)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
#     model.eval()
    
#     # Get standard fakes
#     fakes = glob.glob("templates/Images/fakes/*.jpg")
#     # Get gems
#     gems = glob.glob("templates/Images/fakes/gemini*.jpg")
    
#     # Pick random mix
#     selection = []
#     if len(fakes) > 0: selection.extend(random.sample(fakes, min(8, len(fakes))))
#     if len(gems) > 0: selection.extend(random.sample(gems, min(2, len(gems))))
    
#     for i, path in enumerate(selection):
#         predict_and_viz(model, path, i)

# if __name__ == "__main__":
#     main()



import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UniversalForensicDINO
import os
import argparse
import sys

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/refined_universal_model.pth"
OUTPUT_DIR = "single_inference_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TYPE_MAP_REV = {0: "Real", 1: "Crop/Replace", 2: "Inpaint", 3: "Copy/Move", 4: "AI Generated"}

def predict_and_viz_single(model, image_path):
    # 1. Check file existence
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at '{image_path}'. Check format/integrity.")
        return
    
    # Get filename for saving
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 2. Preprocess
    orig_viz = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    transform = A.Compose([
        A.Resize(518, 518),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    tensor = transform(image=orig_viz)['image'].unsqueeze(0).to(DEVICE)
    
    # 3. Inference
    print(f"Processing: {base_name}...")
    with torch.no_grad():
        bin_logits, mask_logits, type_logits = model(tensor)
        
    # 4. Decode
    fake_prob = torch.sigmoid(bin_logits).item()
    type_probs = torch.softmax(type_logits, dim=1).cpu().numpy()[0]
    pred_type_idx = np.argmax(type_probs)
    pred_type_str = TYPE_MAP_REV.get(pred_type_idx, "Unknown")
    
    # Print text results to console
    print(f"--- Results for {base_name} ---")
    print(f"Fake Probability: {fake_prob*100:.2f}%")
    print(f"Predicted Type:   {pred_type_str}")
    
    # 5. Heatmap Processing
    heatmap = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
    heatmap = cv2.resize(heatmap, (w, h))
    
    # A. Raw Blue Heatmap
    heatmap_norm = np.uint8(255 * heatmap)
    raw_blue_mask = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    
    # B. Clean Overlay (Hard Threshold)
    heatmap_clean = heatmap.copy()
    heatmap_clean[heatmap_clean < 0.4] = 0 
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_clean), cv2.COLORMAP_JET)
    
    # Convert overlay to RGB for matplotlib
    overlay_bgr = cv2.addWeighted(img, 0.7, heatmap_colored, 0.3, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    
    # 6. Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    axs[0].imshow(orig_viz)
    axs[0].set_title("Input Image")
    axs[0].axis('off')
    
    axs[1].imshow(cv2.cvtColor(raw_blue_mask, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Raw Heatmap")
    axs[1].axis('off')
    
    axs[2].imshow(overlay_rgb)
    axs[2].set_title(f"Prob: {fake_prob*100:.1f}% | Type: {pred_type_str}")
    axs[2].axis('off')
    
    save_path = f"{OUTPUT_DIR}/{base_name}_result.jpg"
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to: {save_path}")

def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description="Run Forensic DINO on a single image.")
    parser.add_argument("path", type=str, help="Path to the input image")
    args = parser.parse_args()

    # Load Model
    print(f"Loading Model from {MODEL_PATH}...")
    try:
        model = UniversalForensicDINO(num_classes=5).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model.eval()
    except FileNotFoundError:
        print("Error: Model checkpoint not found. Please check MODEL_PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Run Inference
    predict_and_viz_single(model, args.path)

if __name__ == "__main__":
    main()