import os
import pandas as pd
import cv2

# CONFIG
DATA_ROOT = "."  # Current folder
CSV_FILE = "splits_balanced/train.csv"

def check_masks():
    print(f"Checking masks based on {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    
    # Filter for FAKES only (label == 1)
    fakes = df[df['label'] == 1]
    print(f"Found {len(fakes)} fake images in CSV.")
    
    found_count = 0
    missing_count = 0
    
    # Check first 20 fakes
    for idx, row in fakes.head(20).iterrows():
        rel_img_path = row['image_path']
        img_full_path = os.path.join(DATA_ROOT, rel_img_path)
        
        # Current Logic in your dataset.py
        rel_mask_path = rel_img_path.replace('Images', 'Annotations')
        mask_path_jpg = os.path.join(DATA_ROOT, rel_mask_path)
        
        # Alternative Logic (Try PNG)
        mask_path_png = os.path.splitext(mask_path_jpg)[0] + ".png"
        
        print(f"\nImage: {rel_img_path}")
        print(f" -> Looking for JPG mask: {mask_path_jpg}")
        
        if os.path.exists(mask_path_jpg):
            print("    [OK] Found JPG mask!")
            found_count += 1
        elif os.path.exists(mask_path_png):
             print(f"    [OK] Found PNG mask instead: {mask_path_png}")
             found_count += 1
        else:
            print("    [FAIL] Mask NOT found.")
            missing_count += 1
            
    print(f"\nSummary of first 20: Found {found_count}, Missing {missing_count}")

if __name__ == "__main__":
    check_masks()