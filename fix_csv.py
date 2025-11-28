import os
import glob

# CONFIG
CSV_PATH = "splits_balanced/train.csv"
GEMINI_FOLDER = "templates/Images/fakes"

def fix_csv():
    print(f"Reading {CSV_PATH}...")
    
    # 1. Read existing lines and filter out the bad ones
    valid_lines = []
    with open(CSV_PATH, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split(',')
        # Keep empty newlines OR lines that look like standard format (2 columns)
        # We assume standard is: label,path OR label_name,label,path... 
        # But based on your error, your valid lines have exactly 2 columns.
        if len(parts) == 2:
            valid_lines.append(line.strip())
        elif len(parts) > 2 and "gemini" not in line: 
            # Keep header if it exists and has more cols, but skip gemini lines
            valid_lines.append(line.strip())
            
    print(f"Cleaned CSV. Retained {len(valid_lines)} valid lines.")
    
    # 2. Find Gemini Images
    gemini_images = glob.glob(os.path.join(GEMINI_FOLDER, "gemini_fake_*.jpg"))
    # Sort them to be nice
    gemini_images.sort()
    
    if not gemini_images:
        print("No Gemini images found to append!")
        return

    print(f"Found {len(gemini_images)} Gemini images to append...")
    
    # 3. Append them in the CORRECT format (2 Columns: Label, Path)
    # Label is 1 (Fake)
    for img_path in gemini_images:
        # img_path is like "templates/Images/fakes\gemini_fake_01.jpg"
        # We need relative path with forward slashes usually
        rel_path = img_path.replace("\\", "/")
        
        # The Format: 1,relative_path
        new_line = f"1,{rel_path}"
        valid_lines.append(new_line)
        
    # 4. Write back to file
    with open(CSV_PATH, 'w') as f:
        f.write("\n".join(valid_lines))
        
    print("Success! CSV fixed and Gemini data added correctly.")

if __name__ == "__main__":
    fix_csv()