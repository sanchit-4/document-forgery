import os
import shutil
import json
import glob

# --- CONFIGURATION ---
SOURCE_FOLDER = "raw_gemini"  # Put your ALREADY CROPPED images here
DEST_IMG_FOLDER = "templates/Images/fakes"
DEST_JSON_FOLDER = "templates/Annotations/fakes"
CSV_PATH = "splits_balanced/train.csv"

# Ensure output directories exist
os.makedirs(DEST_IMG_FOLDER, exist_ok=True)
os.makedirs(DEST_JSON_FOLDER, exist_ok=True)

def process_gemini_data():
    # 1. Gather all images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(SOURCE_FOLDER, ext)))
    
    if not files:
        print(f"No images found in '{SOURCE_FOLDER}'. Please create the folder and add images.")
        return

    print(f"Found {len(files)} images. Processing...")
    
    new_csv_lines = []
    
    for i, file_path in enumerate(files):
        # 1. Generate New Name (gemini_fake_01.jpg, etc.)
        # We enforce .jpg for consistency with the rest of the dataset
        new_filename = f"gemini_fake_{i+1:02d}.jpg"
        dest_img_path = os.path.join(DEST_IMG_FOLDER, new_filename)
        
        # 2. Copy and Rename File
        # We use shutil to copy. If source isn't jpg, we might want to convert, 
        # but usually just copying is fine if your loader handles it. 
        # To be safe, let's just copy bytes.
        shutil.copy2(file_path, dest_img_path)
        
        # 3. Generate JSON Metadata
        # This is CRITICAL. The 'ctype' tells the Universal Model this is Class 4.
        json_filename = f"gemini_fake_{i+1:02d}.json"
        dest_json_path = os.path.join(DEST_JSON_FOLDER, json_filename)
        
        json_data = {
            "src": new_filename,
            "ctype": "fully_ai_generated",  # <--- Triggers Class 4 in dataset.py
            "field": "whole_image",
            "loader": "gemini_custom"
        }
        
        with open(dest_json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
            
        # 4. Prepare CSV Line
        # Format matches SIDTD: label_name, label, image_path, class_id, class_name
        # label = 1 (Binary Fake)
        rel_path = f"templates/Images/fakes/{new_filename}"
        line = f"fakes,1,{rel_path},999,AI"
        new_csv_lines.append(line)
        
        print(f"  [OK] Saved {new_filename} + JSON")

    # 5. Append to Train CSV
    if new_csv_lines:
        print(f"Appending {len(new_csv_lines)} entries to {CSV_PATH}...")
        with open(CSV_PATH, 'a') as f: # 'a' for append
            for line in new_csv_lines:
                f.write("\n" + line)
        print("Success! Your Gemini data is now part of the training set.")
    else:
        print("No lines to add.")

if __name__ == "__main__":
    process_gemini_data()