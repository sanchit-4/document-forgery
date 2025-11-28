# import os
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader, WeightedRandomSampler
# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import tqdm
# from sklearn.metrics import roc_auc_score
# from torch.amp import autocast, GradScaler

# from dataset import SIDTDDataset, get_transforms
# from model import ForensicDINO
# from utils import ExplainableLoss

# # --- CONFIG ---
# BATCH_SIZE = 2
# ACCUM_STEPS = 8 
# LEARNING_RATE = 1e-4 # Lower LR for fine-tuning
# EPOCHS = 15
# DEVICE = "cuda"
# NUM_WORKERS = 0 # Set to 0 for Windows stability
# SAVE_DIR = "./checkpoints"
# VIS_DIR = "./training_visuals"

# os.makedirs(SAVE_DIR, exist_ok=True)
# os.makedirs(VIS_DIR, exist_ok=True)

# def get_balanced_sampler(dataset):
#     """
#     Creates a sampler that ensures 50/50 Real/Fake distribution in batches
#     """
#     targets = []
#     print("Computing class weights for balanced sampling...")
#     for i in tqdm(range(len(dataset))):
#         _, _, label = dataset[i]
#         targets.append(int(label.item()))
    
#     targets = torch.tensor(targets)
#     class_sample_count = torch.tensor([(targets == t).sum() for t in torch.unique(targets, sorted=True)])
    
#     # Weight = 1 / count
#     weight = 1. / class_sample_count.float()
#     samples_weight = torch.tensor([weight[t] for t in targets])
    
#     sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
#     return sampler

# def save_debug_images(images, masks_true, masks_pred, epoch, batch_idx):
#     img = images[0].permute(1, 2, 0).float().cpu().numpy()
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     img = std * img + mean
#     img = np.clip(img, 0, 1)

#     gt = masks_true[0].squeeze().cpu().numpy()
    
#     # Use Sigmoid to get 0-1 probability
#     pred = torch.sigmoid(masks_pred[0]).squeeze().detach().cpu().float().numpy()

#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#     axs[0].imshow(img)
#     axs[0].set_title("Input ID")
#     axs[0].axis('off')

#     axs[1].imshow(gt, cmap='gray')
#     axs[1].set_title("Ground Truth")
#     axs[1].axis('off')

#     # Use 'inferno' or 'jet' to visualize intensity
#     axs[2].imshow(pred, cmap='inferno', vmin=0, vmax=1)
#     axs[2].set_title(f"Pred (Max: {pred.max():.2f})")
#     axs[2].axis('off')

#     plt.savefig(f"{VIS_DIR}/epoch_{epoch}_batch_{batch_idx}.png")
#     plt.close()

# def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch):
#     model.train()
#     loop = tqdm(loader, desc=f"Training Epoch {epoch}")
    
#     running_loss = 0.0
#     optimizer.zero_grad()
    
#     for batch_idx, (images, masks, labels) in enumerate(loop):
#         images, masks, labels = images.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)
        
#         with autocast(device_type='cuda', dtype=torch.float16):
#             cls_logits, mask_logits = model(images)
#             loss = criterion(cls_logits, mask_logits, labels, masks)
#             loss = loss / ACCUM_STEPS 
        
#         scaler.scale(loss).backward()
        
#         if (batch_idx + 1) % ACCUM_STEPS == 0:
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()
        
#         running_loss += loss.item() * ACCUM_STEPS
#         loop.set_postfix(loss=loss.item() * ACCUM_STEPS)

#     return running_loss / len(loader)

# def validate(model, loader, criterion, epoch):
#     model.eval()
#     loop = tqdm(loader, desc="Validating")
    
#     running_loss = 0.0
#     all_labels = []
#     all_preds_cls = []
#     iou_scores = []

#     with torch.no_grad():
#         for batch_idx, (images, masks, labels) in enumerate(loop):
#             images, masks, labels = images.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)
            
#             with autocast(device_type='cuda', dtype=torch.float16):
#                 cls_logits, mask_logits = model(images)
#                 loss = criterion(cls_logits, mask_logits, labels, masks)
            
#             running_loss += loss.item()
            
#             probs_cls = torch.sigmoid(cls_logits).squeeze().cpu().float().numpy()
#             labels_np = labels.cpu().numpy()
            
#             if np.ndim(probs_cls) == 0: probs_cls = [probs_cls]; labels_np = [labels_np]
#             all_preds_cls.extend(probs_cls)
#             all_labels.extend(labels_np)
            
#             # IoU Logic - STRICTLY ON FAKES
#             pred_mask_bin = (torch.sigmoid(mask_logits) > 0.5).float()
#             for i in range(labels.size(0)):
#                 if labels[i] == 1: # Only calculate IoU for Fakes
#                     intersection = (pred_mask_bin[i] * masks[i]).sum()
#                     union = pred_mask_bin[i].sum() + masks[i].sum() - intersection
#                     iou = (intersection + 1e-6) / (union + 1e-6)
#                     iou_scores.append(iou.item())

#             if batch_idx == 0:
#                 save_debug_images(images, masks, mask_logits, epoch, batch_idx)

#     epoch_loss = running_loss / len(loader)
    
#     try:
#         auc = roc_auc_score(all_labels, all_preds_cls)
#     except:
#         auc = 0.5
        
#     # If no fakes in validation batch, return 0 IoU to show something is wrong
#     mean_iou = np.mean(iou_scores) if len(iou_scores) > 0 else 0.0
    
#     print(f"\nEpoch {epoch} | Loss: {epoch_loss:.4f} | AUC: {auc:.4f} | Fake IoU: {mean_iou:.4f}")
#     return epoch_loss, auc, mean_iou

# def main():
#     DATA_ROOT = "." 
#     TRAIN_CSV = os.path.join(DATA_ROOT, "splits_balanced", "train.csv")
#     VAL_CSV = os.path.join(DATA_ROOT, "splits_balanced", "val.csv")
    
#     print("Loading Datasets...")
#     train_dataset = SIDTDDataset(root_dir=DATA_ROOT, csv_file=TRAIN_CSV, transform=get_transforms(), mode='train')
#     val_dataset = SIDTDDataset(root_dir=DATA_ROOT, csv_file=VAL_CSV, transform=get_transforms(), mode='val')
    
#     # --- KEY CHANGE: BALANCED SAMPLER ---
#     sampler = get_balanced_sampler(train_dataset)
    
#     # Shuffle MUST be False if Sampler is provided
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, drop_last=True)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=True)
    
#     print(f"Initializing Forensic-DINO on {DEVICE}...")
#     model = ForensicDINO(freeze_dino=False).to(DEVICE) # Use the U-Net Model
    
#     optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
#     criterion = ExplainableLoss().to(DEVICE)
#     scaler = torch.amp.GradScaler()
    
#     best_iou = 0.0 # Track IoU now, not AUC (since AUC is easy)
    
#     print("Starting Balanced Training...")
#     for epoch in range(1, EPOCHS + 1):
#         train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch)
#         val_loss, val_auc, val_iou = validate(model, val_loader, criterion, epoch)
        
#         # Save based on IoU (Heatmap Quality)
#         if val_iou > best_iou:
#             best_iou = val_iou
#             torch.save(model.state_dict(), f"{SAVE_DIR}/best_heatmap_model.pth")
#             print(f"Saved Best Heatmap Model (IoU: {best_iou:.4f})")
        
#         torch.save(model.state_dict(), f"{SAVE_DIR}/last_model.pth")

# if __name__ == "__main__":
#     import multiprocessing
#     multiprocessing.freeze_support()
#     main()


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.amp import autocast, GradScaler

# --- IMPORTS FROM YOUR PROJECT ---
# Ensure your dataset.py and model.py are updated to the "Universal" versions provided previously
from dataset import SIDTDDataset, get_transforms
from model import UniversalForensicDINO

# --- CONFIGURATION ---
BATCH_SIZE = 2          # Keep low for VRAM safety
ACCUM_STEPS = 8         # Effective Batch Size = 16
LEARNING_RATE = 1e-4    # Low LR for fine-tuning
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0         # Windows stability
SAVE_DIR = "./checkpoints"
VIS_DIR = "./training_visuals"

# Forgery Types for logging
TYPE_MAP_REV = {0: "Real", 1: "Crop", 2: "Inpaint", 3: "Copy", 4: "AI"}

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# --- LOSS FUNCTIONS (Included here for completeness) ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class UniversalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()   # For Real vs Fake
        self.ce = nn.CrossEntropyLoss()     # For Type (Multi-class)
        self.dice = DiceLoss()              # For Heatmap

    def forward(self, binary_pred, mask_pred, type_pred, 
                binary_target, mask_target, type_target):
        
        # 1. Binary Loss (Is it Fake?)
        loss_bin = self.bce(binary_pred.view(-1), binary_target)
        
        # 2. Type Loss (What kind of Fake?)
        # type_target needs to be Long/Int for CrossEntropy
        loss_type = self.ce(type_pred, type_target)
        
        # 3. Segmentation Loss (Where is it Fake?)
        # Only calculate mask loss for images that ARE fake (binary_target == 1)
        mask_loss_active = (binary_target.view(-1, 1, 1, 1) == 1).float()
        
        # Calculate Dice only on active (fake) images
        # We add a small epsilon to avoid division by zero if batch has no fakes
        loss_seg = (self.dice(mask_pred, mask_target) * mask_loss_active).sum() / (mask_loss_active.sum() + 1e-6)
        
        # TOTAL LOSS WEIGHTING
        # Binary is easy -> 1.0
        # Type is hard -> 2.0
        # Segmentation is very hard -> 5.0
        return loss_bin + (2.0 * loss_type) + (5.0 * loss_seg)

# --- HELPER FUNCTIONS ---
def get_balanced_sampler(dataset):
    """
    Optimized sampler that reads directly from Dataframe to save time.
    Ensures 50/50 split between Real (0) and Fakes (1).
    """
    print("Computing class weights for balanced sampling...")
    # Read labels directly from the pandas dataframe in the dataset object
    # This avoids loading images from disk
    targets = dataset.data['label'].values.astype(int)
    
    class_sample_count = np.array([np.sum(targets == t) for t in np.unique(targets)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in targets])
    
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

def save_debug_images(images, masks_true, masks_pred, type_preds, type_true, epoch, batch_idx):
    img = images[0].permute(1, 2, 0).float().cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    gt = masks_true[0].squeeze().cpu().numpy()
    pred = torch.sigmoid(masks_pred[0]).squeeze().detach().cpu().float().numpy()
    
    # Get predicted type
    pred_type_idx = torch.argmax(type_preds[0]).item()
    true_type_idx = type_true[0].item()
    pred_str = TYPE_MAP_REV.get(pred_type_idx, "?")
    true_str = TYPE_MAP_REV.get(true_type_idx, "?")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img)
    axs[0].set_title(f"Input ({true_str})")
    axs[0].axis('off')

    axs[1].imshow(gt, cmap='gray')
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis('off')

    axs[2].imshow(pred, cmap='inferno', vmin=0, vmax=1)
    axs[2].set_title(f"Pred: {pred_str} (Max:{pred.max():.2f})")
    axs[2].axis('off')

    plt.savefig(f"{VIS_DIR}/epoch_{epoch}_batch_{batch_idx}.png")
    plt.close()

# --- TRAINING LOOP ---
def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch):
    model.train()
    loop = tqdm(loader, desc=f"Training Epoch {epoch}")
    
    running_loss = 0.0
    optimizer.zero_grad()
    
    # Unpack 4 values now
    for batch_idx, (images, masks, binary_labels, type_labels) in enumerate(loop):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        binary_labels = binary_labels.to(DEVICE)
        type_labels = type_labels.to(DEVICE)
        
        with autocast(device_type='cuda', dtype=torch.float16):
            # Model returns 3 outputs
            binary_logits, mask_logits, type_logits = model(images)
            
            loss = criterion(binary_logits, mask_logits, type_logits, 
                             binary_labels, masks, type_labels)
            loss = loss / ACCUM_STEPS 
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        running_loss += loss.item() * ACCUM_STEPS
        loop.set_postfix(loss=loss.item() * ACCUM_STEPS)

    return running_loss / len(loader)

def validate(model, loader, criterion, epoch):
    model.eval()
    loop = tqdm(loader, desc="Validating")
    
    running_loss = 0.0
    
    # Storage for metrics
    all_bin_labels = []
    all_bin_preds = []
    
    all_type_labels = []
    all_type_preds = []
    
    iou_scores = []

    with torch.no_grad():
        for batch_idx, (images, masks, binary_labels, type_labels) in enumerate(loop):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            binary_labels = binary_labels.to(DEVICE)
            type_labels = type_labels.to(DEVICE)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                binary_logits, mask_logits, type_logits = model(images)
                loss = criterion(binary_logits, mask_logits, type_logits, 
                                 binary_labels, masks, type_labels)
            
            running_loss += loss.item()
            
            # --- 1. Binary Metrics ---
            probs_bin = torch.sigmoid(binary_logits).squeeze().cpu().float().numpy()
            labels_bin_np = binary_labels.cpu().numpy()
            if np.ndim(probs_bin) == 0: probs_bin = [probs_bin]; labels_bin_np = [labels_bin_np]
            all_bin_preds.extend(probs_bin)
            all_bin_labels.extend(labels_bin_np)
            
            # --- 2. Type Metrics ---
            preds_type = torch.argmax(type_logits, dim=1).cpu().numpy()
            labels_type_np = type_labels.cpu().numpy()
            all_type_preds.extend(preds_type)
            all_type_labels.extend(labels_type_np)
            
            # --- 3. Segmentation Metrics (IoU on Fakes only) ---
            pred_mask_bin = (torch.sigmoid(mask_logits) > 0.5).float()
            for i in range(binary_labels.size(0)):
                if binary_labels[i] == 1: # If Fake
                    intersection = (pred_mask_bin[i] * masks[i]).sum()
                    union = pred_mask_bin[i].sum() + masks[i].sum() - intersection
                    iou = (intersection + 1e-6) / (union + 1e-6)
                    iou_scores.append(iou.item())

            if batch_idx == 0:
                save_debug_images(images, masks, mask_logits, type_logits, type_labels, epoch, batch_idx)

    epoch_loss = running_loss / len(loader)
    
    # Calculate Final Metrics
    try:
        auc = roc_auc_score(all_bin_labels, all_bin_preds)
    except:
        auc = 0.5
        
    type_acc = accuracy_score(all_type_labels, all_type_preds)
    
    mean_iou = np.mean(iou_scores) if len(iou_scores) > 0 else 0.0
    
    print(f"\nEpoch {epoch} Results:")
    print(f"  Loss:      {epoch_loss:.4f}")
    print(f"  Bin AUC:   {auc:.4f}")
    print(f"  Type Acc:  {type_acc:.4f}")
    print(f"  Fake IoU:  {mean_iou:.4f}")
    
    return epoch_loss, auc, type_acc, mean_iou

# --- MAIN ---
def main():
    DATA_ROOT = "." 
    TRAIN_CSV = os.path.join(DATA_ROOT, "splits_balanced", "train.csv")
    VAL_CSV = os.path.join(DATA_ROOT, "splits_balanced", "val.csv")
    
    print("Loading Universal Datasets...")
    train_dataset = SIDTDDataset(root_dir=DATA_ROOT, csv_file=TRAIN_CSV, transform=get_transforms(), mode='train')
    val_dataset = SIDTDDataset(root_dir=DATA_ROOT, csv_file=VAL_CSV, transform=get_transforms(), mode='val')
    
    # Balanced Sampler (Real vs Fake)
    sampler = get_balanced_sampler(train_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=True)
    
    print(f"Initializing Universal Forensic-DINO (5 Classes)...")
    model = UniversalForensicDINO(num_classes=5).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = UniversalLoss().to(DEVICE)
    scaler = torch.amp.GradScaler()
    
    best_iou = 0.0 
    
    print("Starting Universal Training...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch)
        val_loss, val_auc, val_acc, val_iou = validate(model, val_loader, criterion, epoch)
        
        # Save based on IoU (The hardest task)
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_universal_model.pth")
            print(f"Saved Best Model (IoU: {best_iou:.4f})")
        
        torch.save(model.state_dict(), f"{SAVE_DIR}/last_universal_model.pth")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()