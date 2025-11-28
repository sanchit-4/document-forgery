import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UniversalForensicDINO

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/best_universal_model.pth" # Use the BEST one (Epoch 15)
TYPE_MAP_REV = {0: "Real", 1: "Crop/Replace", 2: "Inpaint", 3: "Copy/Move", 4: "AI Generated"}

def predict_universal(image_path):
    # 1. Load Model
    # Note: num_classes=5 must match training
    model = UniversalForensicDINO(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    
    # 2. Load & Preprocess Image
    img = cv2.imread(image_path)
    if img is None: return
    orig_viz = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(518, 518),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    tensor = transform(image=orig_viz)['image'].unsqueeze(0).to(DEVICE)
    
    # 3. Predict
    with torch.no_grad():
        bin_logits, mask_logits, type_logits = model(tensor)
        
    # 4. Decode Outputs
    # A. Binary Score
    fake_prob = torch.sigmoid(bin_logits).item()
    
    # B. Type
    type_probs = torch.softmax(type_logits, dim=1).cpu().numpy()[0]
    pred_type_idx = np.argmax(type_probs)
    pred_type_str = TYPE_MAP_REV.get(pred_type_idx, "Unknown")
    type_conf = type_probs[pred_type_idx]
    
    # C. Heatmap
    heatmap = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # 5. Visualization
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    
    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Text Report
    report = (
        f"FAKE Probability: {fake_prob*100:.2f}%\n"
        f"Detected Type:    {pred_type_str}\n"
        f"Type Confidence:  {type_conf*100:.2f}%"
    )
    
    axs[0].imshow(orig_viz)
    axs[0].set_title("Input Image")
    axs[0].axis('off')
    
    axs[1].imshow(overlay)
    axs[1].set_title(report, fontsize=12, loc='left')
    axs[1].axis('off')
    
    out_name = "universal_result_" + os.path.basename(image_path)
    plt.savefig(out_name)
    print(f"Saved: {out_name} -> Type: {pred_type_str}")
    plt.close()

if __name__ == "__main__":
    import os
    # Test on your Gemini image
    predict_universal("templates/Images/fakes/gemini_fake_01.jpg") 
    
    # Test on a random dataset fake (Crop & Replace)
    # Replace this path with a real file from your dataset
    # predict_universal("templates/Images/fakes/alb_id_00_fake_6_25.jpg")