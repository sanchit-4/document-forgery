import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import numpy as np

# Imports
from dataset import SIDTDDataset, get_transforms
from model import UniversalForensicDINO

# CONFIG
BATCH_SIZE = 2
LR = 5e-5 # Very low learning rate
EPOCHS = 5
DEVICE = "cuda"
SAVE_DIR = "./checkpoints"

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

def main():
    print("Initializing Fine-Tuner...")
    
    # 1. Load Data
    train_dataset = SIDTDDataset(root_dir=".", csv_file="splits_balanced/train.csv", transform=get_transforms(), mode='train')
    
    # We only want FAKES for this phase! 
    # (Training heatmap on Real images is useless as mask is black)
    # Filter indices where label == 1
    fake_indices = [i for i, x in enumerate(train_dataset.data['label']) if x == 1]
    
    train_sampler = torch.utils.data.SubsetRandomSampler(fake_indices)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
    
    # 2. Load Model
    model = UniversalForensicDINO(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load(f"{SAVE_DIR}/best_universal_model.pth", weights_only=True))
    
    # 3. FREEZE EVERYTHING EXCEPT DECODER
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze Decoder layers
    for param in model.bottleneck.parameters(): param.requires_grad = True
    for param in model.up1.parameters(): param.requires_grad = True
    for param in model.cat1.parameters(): param.requires_grad = True
    for param in model.up2.parameters(): param.requires_grad = True
    for param in model.cat2.parameters(): param.requires_grad = True
    for param in model.up3.parameters(): param.requires_grad = True
    for param in model.cat3.parameters(): param.requires_grad = True
    for param in model.up4.parameters(): param.requires_grad = True
    for param in model.cat4.parameters(): param.requires_grad = True
    for param in model.final_mask.parameters(): param.requires_grad = True
    
    print("Model Frozen. Only Decoder is trainable.")
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    dice_loss = DiceLoss().to(DEVICE)
    scaler = torch.amp.GradScaler()
    
    print("Starting Heatmap Refinement...")
    
    for epoch in range(1, EPOCHS+1):
        model.train()
        loop = tqdm(train_loader, desc=f"Refining Epoch {epoch}")
        running_loss = 0.0
        
        for images, masks, _, _ in loop: # We ignore labels/types
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                # Forward
                _, mask_logits, _ = model(images)
                
                # Loss (Pure Dice)
                loss = dice_loss(mask_logits, masks)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch} Loss: {running_loss/len(train_loader):.4f}")
        
        # Save
        torch.save(model.state_dict(), f"{SAVE_DIR}/refined_universal_model.pth")
        print("Saved refined model.")

if __name__ == "__main__":
    main()