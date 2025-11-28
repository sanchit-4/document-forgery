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

def scanner_simulation_pipeline(image_path):
    """
    Turns a phone photo into a 'Digital Scan' lookalike.
    """
    img = cv2.imread(image_path)
    if img is None: return None, None
    
    # --- STEP 1: ROBUST CROP & WARP ---
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use morphological operations to merge text blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # Canny Edge Detection
    edges = cv2.Canny(morph, 50, 150)
    
    # Find Contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    warped = img # Default to original if fail
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area < (img.shape[0] * img.shape[1] * 0.1): continue 
        
        # Min Area Rect (Handles rotation)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        
        # --- FIX FOR NUMPY 2.0 ---
        # np.int0 is removed in NumPy 2.0. Use np.int32 instead.
        box = np.int32(box)
        
        # Perspective Transform logic
        pts = box.reshape(4, 2).astype("float32")
        
        # Order points: TL, TR, BR, BL
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        ordered_pts = np.zeros((4, 2), dtype="float32")
        ordered_pts[0] = pts[np.argmin(s)]      # TL
        ordered_pts[2] = pts[np.argmax(s)]      # BR
        ordered_pts[1] = pts[np.argmin(diff)]   # TR
        ordered_pts[3] = pts[np.argmax(diff)]   # BL
        
        (tl, tr, br, bl) = ordered_pts
        
        # Compute width/height
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

    # --- STEP 2: ILLUMINATION CORRECTION ---
    rgb_planes = cv2.split(warped)
    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(norm_img)
        
    shadow_free = cv2.merge(result_planes)

    # --- STEP 3: DENOISING ---
    clean = cv2.fastNlMeansDenoisingColored(shadow_free, None, 10, 10, 7, 21)
    
    # --- STEP 4: SHARPENING ---
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(clean, -1, kernel)

    return img, sharpened

def predict(image_path):
    print(f"Processing {image_path}...")
    
    # 1. Run the Pipeline
    original, processed = scanner_simulation_pipeline(image_path)
    
    if processed is None:
        print("Failed to load image.")
        return

    # 2. Prepare for Model
    transform = A.Compose([
        A.Resize(518, 518),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # We predict on the PROCESSED image
    proc_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    tensor = transform(image=proc_rgb)['image'].unsqueeze(0).to(DEVICE)
    
    # 3. Load Model
    model = UniversalForensicDINO(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    
    # 4. Inference
    with torch.no_grad():
        bin_logits, mask_logits, type_logits = model(tensor)
        
    # 5. Decode
    fake_prob = torch.sigmoid(bin_logits).item()
    
    type_probs = torch.softmax(type_logits, dim=1).cpu().numpy()[0]
    pred_type_idx = np.argmax(type_probs)
    pred_type_str = TYPE_MAP_REV.get(pred_type_idx, "Unknown")
    
    heatmap = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
    heatmap = cv2.resize(heatmap, (processed.shape[1], processed.shape[0]))
    
    # 6. Visualization
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(processed, 0.6, heatmap_colored, 0.4, 0)
    
    # Show 3 things: Original Photo, Simulated Scan, Heatmap Result
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    axs[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Photo")
    axs[0].axis('off')
    
    axs[1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Simulated Scan (Input to AI)")
    axs[1].axis('off')
    
    axs[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    title_color = "red" if fake_prob > 0.5 else "green"
    axs[2].set_title(f"Prob: {fake_prob*100:.2f}% | {pred_type_str}", color=title_color, fontsize=12, weight='bold')
    axs[2].axis('off')
    
    out_name = "enhanced_result_" + os.path.basename(image_path)
    plt.savefig(out_name)
    print(f"Saved result: {out_name}")
    print(f"Final Probability: {fake_prob:.4f}")

if __name__ == "__main__":
    # Replace with your college ID path
    predict("templates/real.jpg")