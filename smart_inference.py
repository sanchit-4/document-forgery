import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import ForensicDINO

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/best_heatmap_model.pth"

def get_id_card_crop(image_path):
    """
    Robust Cropper using MinAreaRect (Handles rounded corners & rotation)
    """
    img = cv2.imread(image_path)
    if img is None: return None
    orig = img.copy()
    h_orig, w_orig = img.shape[:2]
    
    # 1. Preprocessing (Aggressive)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) to separate white card from light background
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Blur and Threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Canny with looser thresholds
    edged = cv2.Canny(blurred, 30, 150)
    
    # Dilate to close gaps in the edge
    kernel = np.ones((5,5), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)
    
    # 2. Find Contours
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return img # Fallback
    
    # Sort by area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    target_cnt = None
    for c in cnts:
        area = cv2.contourArea(c)
        # Filter small noise (must be at least 10% of image)
        if area < (h_orig * w_orig * 0.1):
            continue
            
        # Instead of looking for 4 points, finding the Minimum Area Rectangle
        # This works for rounded corners
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        target_cnt = box
        break
        
    if target_cnt is None:
        print("Warning: No suitable ID card contour found. Using center crop.")
        return img[h_orig//4:3*h_orig//4, w_orig//4:3*w_orig//4]

    # 3. Perspective Warp (Un-rotate)
    pts = target_cnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    # Order points
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL
    rect[2] = pts[np.argmax(s)] # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    
    (tl, tr, br, bl) = rect
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
        
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    
    return warped

def predict(image_path):
    # 1. Load Model
    model = ForensicDINO(freeze_dino=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    
    # 2. Get Smart Crop
    crop_img = get_id_card_crop(image_path)
    
    # Save crop to see what model sees
    cv2.imwrite("debug_crop.jpg", crop_img) 
    
    # 3. Transform
    transform = A.Compose([
        A.Resize(518, 518),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    tensor = transform(image=crop_rgb)['image'].unsqueeze(0).to(DEVICE)
    
    # 4. Predict
    with torch.no_grad():
        cls_logits, mask_logits = model(tensor)
        
    score = torch.sigmoid(cls_logits).item()
    heatmap = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
    
    # 5. Viz
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.resize(heatmap_colored, (crop_img.shape[1], crop_img.shape[0]))
    overlay = cv2.addWeighted(crop_img, 0.6, heatmap_colored, 0.4, 0)
    
    cv2.imwrite(f"smart_result_{score:.4f}.jpg", overlay)
    print(f"Prediction on Cropped ID: {score:.4f}")
    print("Saved smart_result.jpg and debug_crop.jpg")

# Run on your Gemini image
predict("test\Gemini_Generated_Image_61lwfw61lwfw61lw.png")