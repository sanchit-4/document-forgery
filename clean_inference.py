import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UniversalForensicDINO
import os

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/best_universal_model.pth"
TYPE_MAP_REV = {0: "Real", 1: "Crop/Replace", 2: "Inpaint", 3: "Copy/Move", 4: "AI Generated"}

def robust_preprocess(image_path):
    """
    A safer pipeline to convert phone photos to 'Scanner-like' images
    without destroying color or inverting the image.
    """
    img = cv2.imread(image_path)
    if img is None: return None
    
    # --- STEP 1: PERSPECTIVE CROP (Find the card) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    # Close gaps in lines
    kernel = np.ones((5,5), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)
    
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    warped = img # Fallback
    
    for c in cnts:
        if cv2.contourArea(c) < (img.shape[0]*img.shape[1]*0.1): continue
        
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int32(box) # Fix for Numpy 2.0
        
        # Ordering points
        pts = box.reshape(4, 2).astype("float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        ordered_pts = np.zeros((4, 2), dtype="float32")
        ordered_pts[0] = pts[np.argmin(s)]      # TL
        ordered_pts[2] = pts[np.argmax(s)]      # BR
        ordered_pts[1] = pts[np.argmin(diff)]   # TR
        ordered_pts[3] = pts[np.argmax(diff)]   # BL
        
        (tl, tr, br, bl) = ordered_pts
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
            
        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        break

    # --- STEP 2: GENTLE RESTORATION ---
    
    # A. Bilateral Filter: Removes noise but keeps text edges sharp
    # d=9 (diameter), sigmaColor=75 (mix colors), sigmaSpace=75 (how far to look)
    clean = cv2.bilateralFilter(warped, 9, 75, 75)
    
    # B. Lighting Correction (CLAHE) on L-channel only
    # This fixes shadows without changing the card color
    lab = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # C. Mild Sharpening (To make text look scanned)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    final = cv2.filter2D(final, -1, sharpen_kernel)
    
    # Blend slightly with original to look natural
    final = cv2.addWeighted(final, 0.6, clean, 0.4, 0)

    return img, final

def predict(image_path):
    print(f"Analyzing {image_path}...")
    original, processed = robust_preprocess(image_path)
    
    if processed is None: return

    # Prepare for AI
    transform = A.Compose([
        A.Resize(518, 518),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    tensor = transform(image=cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))['image'].unsqueeze(0).to(DEVICE)
    
    # Load Model
    model = UniversalForensicDINO(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    
    with torch.no_grad():
        bin_logits, mask_logits, type_logits = model(tensor)
        
    # Decode
    fake_prob = torch.sigmoid(bin_logits).item()
    type_probs = torch.softmax(type_logits, dim=1).cpu().numpy()[0]
    pred_type = TYPE_MAP_REV.get(np.argmax(type_probs), "Unknown")
    
    # Visualization
    heatmap = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
    heatmap = cv2.resize(heatmap, (processed.shape[1], processed.shape[0]))
    
    # Clean heatmap: only show high confidence
    heatmap_clean = heatmap.copy()
    heatmap_clean[heatmap_clean < 0.4] = 0 
    
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_clean), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(processed, 0.7, heatmap_colored, 0.3, 0)
    
    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")
    axs[0].axis("off")
    
    axs[1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Cleaned Input (Scanner Sim)")
    axs[1].axis("off")
    
    axs[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    color = "red" if fake_prob > 0.5 else "green"
    axs[2].set_title(f"Prob: {fake_prob*100:.2f}% | {pred_type}", color=color, weight="bold")
    axs[2].axis("off")
    
    plt.savefig("clean_inference_result.jpg")
    print(f"Saved result. Probability: {fake_prob:.4f}")

if __name__ == "__main__":
    predict("templates/real.jpg") # Update path if needed