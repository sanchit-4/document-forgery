# # # utils.py
# # import os
# # import csv
# # from torch.utils.data import Dataset
# # from PIL import Image

# # class SIDTDDataset(Dataset):
# #     """Robust dataset loader for SIDTD from CSV files."""
# #     def __init__(self, csv_file, root_dir, processor, for_features=False):
# #         self.root_dir = root_dir
# #         self.processor = processor
# #         self.image_files = []
# #         self.labels = []
# #         self.for_features = for_features

# #         try:
# #             with open(csv_file, 'r', newline='') as f:
# #                 reader = csv.reader(f)
# #                 header = next(reader, None)
# #                 if header is None: return
# #                 path_idx, label_idx = 1, 0 # Assuming 'label', 'image_path' order

# #                 for row in reader:
# #                     full_path = os.path.join(self.root_dir, row[path_idx].replace('templates/Images/', ''))
# #                     if os.path.isfile(full_path):
# #                         self.image_files.append(full_path)
# #                         self.labels.append(int(row[label_idx]))
# #         except FileNotFoundError:
# #             print(f"FATAL ERROR: The CSV file was not found at {csv_file}")
# #             exit()

# #     def __len__(self):
# #         return len(self.image_files)

# #     def __getitem__(self, idx):
# #         img_path = self.image_files[idx]
# #         try:
# #             image = Image.open(img_path).convert("RGB")
# #             label = self.labels[idx]
# #             if self.for_features:
# #                 processed_inputs = self.processor(images=image, return_tensors="pt")
# #                 return processed_inputs['pixel_values'].squeeze(0), label
# #             return image, label
# #         except Exception:
# #             return None


# # utils.py (Modified for Hybrid Model Inputs)

# # utils.py (LEAN AND MEMORY-SAFE VERSION)
# import os
# import csv
# from torch.utils.data import Dataset
# from PIL import Image

# class SIDTDDataset(Dataset):
#     """
#     MODIFIED: This version is lean and memory-safe.
#     It only loads PIL images and applies lightweight torchvision transforms.
#     The heavy HuggingFace processor is NOT used here.
#     """
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.image_files = []
#         self.labels = []
#         self.transform = transform

#         try:
#             with open(csv_file, 'r', newline='') as f:
#                 reader = csv.reader(f)
#                 header = next(reader, None);_ = header # Skip header
#                 path_idx, label_idx = 1, 0

#                 for row in reader:
#                     # Your robust path logic from before
#                     raw_path = row[path_idx].strip()
#                     normalized_path = raw_path.replace('\\\\', '/').replace('\\', '/').lstrip('./')
#                     parts = normalized_path.split('/')
#                     if len(parts) >= 3 and parts[0].lower() == 'templates' and parts[1].lower() == 'images':
#                         path_suffix = '/'.join(parts[2:])
#                     else:
#                         path_suffix = normalized_path
                    
#                     full_path = os.path.join(self.root_dir, path_suffix)
#                     if os.path.isfile(full_path):
#                         self.image_files.append(full_path)
#                         self.labels.append(int(row[label_idx]))
#         except FileNotFoundError:
#             print(f"FATAL: CSV file not found at {csv_file}"); exit()

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_path = self.image_files[idx]
#         try:
#             # The worker only does this light work
#             image = Image.open(img_path).convert("RGB")
#             label = self.labels[idx]

#             if self.transform:
#                 image = self.transform(image)
            
#             # Return the PIL image, not a tensor
#             return image, label
            
#         except Exception as e:
#             print(f"Warning: Could not load image {img_path}. Error: {e}")
#             return None





import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class ExplainableLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_cls = nn.BCEWithLogitsLoss()
        self.bce_seg = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).cuda()) # Penalty for missing white pixels
        self.dice = DiceLoss()

    def forward(self, cls_pred, mask_pred, label_true, mask_true):
        # 1. Classification Loss
        loss_cls = self.bce_cls(cls_pred.view(-1), label_true)
        
        # 2. Segmentation Loss
        # We multiply by 'label_true' so we ONLY calculate mask loss on FAKE images
        # This prevents the model from learning "Black is always correct" from Real images
        mask_loss_active = (label_true.view(-1, 1, 1, 1) == 1).float()
        
        loss_bce_seg = self.bce_seg(mask_pred, mask_true) * mask_loss_active
        loss_dice = self.dice(mask_pred, mask_true) * mask_loss_active
        
        # Average the segmentation loss only over the fake images in the batch
        num_fakes = mask_loss_active.sum() + 1e-6
        loss_seg_total = (loss_bce_seg.sum() + loss_dice.sum()) / num_fakes

        # Heavily weight the segmentation task (5.0x)
        return loss_cls + (5.0 * loss_seg_total)