# import os
# import cv2
# import json
# import torch
# import pandas as pd
# import numpy as np
# from torch.utils.data import Dataset
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# class SIDTDDataset(Dataset):
#     def __init__(self, root_dir, csv_file, transform=None, mode='train'):
#         self.root_dir = root_dir
#         self.data = pd.read_csv(csv_file)
#         self.transform = transform
#         self.mode = mode
        
#         # Cache for the heavy master JSONs to speed up training
#         self.master_annotations = {} 

#     def __len__(self):
#         return len(self.data)

#     def get_mask_from_metadata(self, fake_json_path, img_shape):
#         h, w = img_shape[:2]
#         mask = np.zeros((h, w), dtype=np.float32)
        
#         try:
#             with open(fake_json_path, 'r') as f:
#                 fake_meta = json.load(f)
            
#             # 1. Extract critical info from Fake JSON
#             modified_field = fake_meta.get('field')     # e.g., "expiry_date"
#             original_filename = fake_meta.get('src')    # e.g., "alb_id_00.jpg" or "00.jpg"
            
#             # 2. Identify the Master JSON file
#             # filename usually is country_type_id.jpg (e.g., alb_id_00.jpg)
#             # We need to find 'alb_id.json' in 'templates/Annotations/reals/'
            
#             # Logic: Split by '_' and take the first two parts (alb, id)
#             parts = os.path.basename(fake_json_path).split('_')
#             # Construct "alb_id"
#             doc_type_prefix = f"{parts[0]}_{parts[1]}" 
            
#             master_json_name = f"{doc_type_prefix}.json"
#             master_json_path = os.path.join(self.root_dir, "templates", "Annotations", "reals", master_json_name)
            
#             # 3. Load Master JSON (with caching)
#             if master_json_path not in self.master_annotations:
#                 if os.path.exists(master_json_path):
#                     with open(master_json_path, 'r') as f:
#                         self.master_annotations[master_json_path] = json.load(f)
#                 else:
#                     # Fallback: Try searching recursively if path is slightly different
#                     return mask # Return black if master not found
            
#             master_data = self.master_annotations[master_json_path]
#             via_metadata = master_data.get('_via_img_metadata', {})
            
#             # 4. Find the specific image entry in Master JSON
#             # The keys in Master JSON are like "00.jpg1859173" (filename + size)
#             # The 'src' in fake JSON is like "alb_id_00.jpg"
            
#             target_entry = None
            
#             # Clean the src name (remove alb_id_ prefix if it exists in src but not in master)
#             # Master keys usually start with "00.jpg..." or "alb_id_00.jpg..."
            
#             # Strategy: Check if any key STARTS with the relevant part of original_filename
#             # Extract "00.jpg" from "alb_id_00.jpg" just in case
#             if original_filename.startswith(doc_type_prefix):
#                 short_name = original_filename.replace(f"{doc_type_prefix}_", "") # "00.jpg"
#             else:
#                 short_name = original_filename

#             for key, val in via_metadata.items():
#                 # Check for "alb_id_00.jpg" OR "00.jpg"
#                 if key.startswith(original_filename) or key.startswith(short_name):
#                     target_entry = val
#                     break
            
#             if target_entry:
#                 # 5. Find the Region for the Modified Field
#                 for region in target_entry.get('regions', []):
#                     region_attr = region.get('region_attributes', {})
#                     field_name = region_attr.get('field_name')
                    
#                     # Check if this is the field that was faked
#                     if field_name == modified_field:
#                         shape = region.get('shape_attributes', {})
#                         if shape.get('name') == 'rect':
#                             x, y = int(shape['x']), int(shape['y'])
#                             rw, rh = int(shape['width']), int(shape['height'])
#                             cv2.rectangle(mask, (x, y), (x+rw, y+rh), 1.0, -1)
#                         elif shape.get('name') == 'polygon':
#                             pts_x = shape['all_points_x']
#                             pts_y = shape['all_points_y']
#                             pts = np.array(list(zip(pts_x, pts_y)), np.int32)
#                             cv2.fillPoly(mask, [pts], 1.0)
                        
#                         # Found it, stop searching regions
#                         break
                        
#         except Exception as e:
#             # print(f"Mask Gen Error: {e}")
#             pass
            
#         return mask

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
        
#         # 1. Get Image
#         rel_img_path = row['image_path'].replace("/", os.sep)
#         img_full_path = os.path.join(self.root_dir, rel_img_path)
        
#         image = cv2.imread(img_full_path)
#         if image is None:
#             return torch.zeros((3, 518, 518)), torch.zeros((1, 518, 518)), torch.tensor(0.0)
            
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         # 2. Label
#         label = int(row['label'])

#         # 3. Mask Generation
#         if label == 0:
#             mask = np.zeros(image.shape[:2], dtype=np.float32)
#         else:
#             # Path to Fake Metadata JSON
#             # templates/Images/fakes/alb_id_00_fake...jpg -> templates/Annotations/fakes/alb_id_00_fake...json
#             rel_json_path = rel_img_path.replace('Images', 'Annotations')
#             rel_json_path = os.path.splitext(rel_json_path)[0] + ".json"
#             json_full_path = os.path.join(self.root_dir, rel_json_path)
            
#             if os.path.exists(json_full_path):
#                 mask = self.get_mask_from_metadata(json_full_path, image.shape)
#             else:
#                 mask = np.zeros(image.shape[:2], dtype=np.float32)

#         # 4. Transform
#         if self.transform:
#             augmented = self.transform(image=image, mask=mask)
#             image = augmented['image']
#             mask = augmented['mask']

#         if mask.ndim == 2:
#             mask = mask.unsqueeze(0)

#         return image, mask, torch.tensor(label, dtype=torch.float32)

# def get_transforms(img_size=518):
#     return A.Compose([
#         A.Resize(img_size, img_size),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2()
#     ])


import os
import cv2
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# MAPPING FORGERY TYPES TO INTEGERS
# We define a standard mapping based on SIDTD + A new "Fully AI" class
TYPE_MAP = {
    "real": 0,
    "crop_and_replace": 1,
    "crop_and_replace_ocr": 1, # Treat OCR replace same as crop replace
    "inpaint_and_rewrite": 2,
    "copy_and_move": 3,
    "fully_ai_generated": 4    # <--- New Class for your Gemini images
}

class SIDTDDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, mode='train'):
        self.root_dir = root_dir
        try:
            # Try reading with headers first
            self.data = pd.read_csv(csv_file)
            
            # Check if 'image_path' column exists. If not, reload assuming no header.
            if 'image_path' not in self.data.columns:
                # Assuming 2 columns: label, image_path
                self.data = pd.read_csv(csv_file, header=None, names=['label', 'image_path'])
                
        except Exception as e:
            print(f"CSV Read Error: {e}")
            # Fallback for simple structure
            self.data = pd.read_csv(csv_file, header=None, names=['label', 'image_path'])

        # Ensure strings
        self.data['image_path'] = self.data['image_path'].astype(str)
        self.transform = transform
        self.mode = mode
        self.master_annotations = {} 

    def __len__(self):
        return len(self.data)

    def get_mask_and_type_from_metadata(self, fake_json_path, img_shape):
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        forgery_type_id = 0 # Default to Real
        
        try:
            with open(fake_json_path, 'r') as f:
                fake_meta = json.load(f)
            
            # 1. GET FORGERY TYPE
            ctype_str = fake_meta.get('ctype', 'real').lower().replace(" ", "_")
            # Fuzzy matching for the mapping
            if "crop" in ctype_str: forgery_type_id = 1
            elif "inpaint" in ctype_str: forgery_type_id = 2
            elif "copy" in ctype_str: forgery_type_id = 3
            elif "ai" in ctype_str or "gan" in ctype_str: forgery_type_id = 4
            else: forgery_type_id = 1 # Default fallback for unknown fakes
            
            # 2. GET MASK (Existing logic...)
            modified_field = fake_meta.get('field')
            original_filename = fake_meta.get('src')
            
            parts = os.path.basename(fake_json_path).split('_')
            doc_type_prefix = f"{parts[0]}_{parts[1]}" 
            master_json_name = f"{doc_type_prefix}.json"
            master_json_path = os.path.join(self.root_dir, "templates", "Annotations", "reals", master_json_name)
            
            if master_json_path not in self.master_annotations:
                if os.path.exists(master_json_path):
                    with open(master_json_path, 'r') as f:
                        self.master_annotations[master_json_path] = json.load(f)
            
            if master_json_path in self.master_annotations:
                master_data = self.master_annotations[master_json_path]
                via_metadata = master_data.get('_via_img_metadata', {})
                
                if original_filename.startswith(doc_type_prefix):
                    short_name = original_filename.replace(f"{doc_type_prefix}_", "")
                else:
                    short_name = original_filename

                target_entry = None
                for key, val in via_metadata.items():
                    if key.startswith(original_filename) or key.startswith(short_name):
                        target_entry = val
                        break
                
                if target_entry:
                    for region in target_entry.get('regions', []):
                        region_attr = region.get('region_attributes', {})
                        if region_attr.get('field_name') == modified_field:
                            shape = region.get('shape_attributes', {})
                            if shape.get('name') == 'rect':
                                x, y, rw, rh = int(shape['x']), int(shape['y']), int(shape['width']), int(shape['height'])
                                cv2.rectangle(mask, (x, y), (x+rw, y+rh), 1.0, -1)
                            elif shape.get('name') == 'polygon':
                                pts = np.array(list(zip(shape['all_points_x'], shape['all_points_y'])), np.int32)
                                cv2.fillPoly(mask, [pts], 1.0)
                            break
                            
        except Exception:
            pass
            
        return mask, forgery_type_id

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        rel_img_path = row['image_path'].replace("/", os.sep)
        img_full_path = os.path.join(self.root_dir, rel_img_path)
        
        image = cv2.imread(img_full_path)
        if image is None:
            return torch.zeros((3, 518, 518)), torch.zeros((1, 518, 518)), torch.tensor(0.0), torch.tensor(0)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(row['label'])
        type_label = 0 # Real

        if label == 0:
            mask = np.zeros(image.shape[:2], dtype=np.float32)
            type_label = 0
        else:
            rel_json_path = rel_img_path.replace('Images', 'Annotations').replace('.jpg', '.json').replace('.png', '.json')
            json_full_path = os.path.join(self.root_dir, rel_json_path)
            
            if os.path.exists(json_full_path):
                mask, type_label = self.get_mask_and_type_from_metadata(json_full_path, image.shape)
            else:
                mask = np.zeros(image.shape[:2], dtype=np.float32)
                type_label = 1 # Default to generic fake if JSON missing

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if mask.ndim == 2: mask = mask.unsqueeze(0)

        return image, mask, torch.tensor(label, dtype=torch.float32), torch.tensor(type_label, dtype=torch.long)

def get_transforms(img_size=518):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])