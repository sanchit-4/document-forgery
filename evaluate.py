import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm

# Imports from your project
from dataset import SIDTDDataset, get_transforms
from model import ForensicDINO

# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/best_heatmap_model.pth"
DATA_ROOT = "." 
TEST_CSV = "splits_balanced/test.csv"  # Or val.csv if you don't have test
BATCH_SIZE = 2

def evaluate_model():
    # 1. Load Data
    test_dataset = SIDTDDataset(root_dir=DATA_ROOT, csv_file=TEST_CSV, transform=get_transforms(), mode='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 2. Load Model
    model = ForensicDINO(freeze_dino=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    
    y_true = []
    y_pred = []
    
    print("Evaluating...")
    with torch.no_grad():
        for images, masks, labels in tqdm(test_loader):
            images = images.to(DEVICE)
            
            # Forward pass
            cls_logits, _ = model(images)
            probs = torch.sigmoid(cls_logits).squeeze().cpu().numpy()
            
            # Handle batch size 1 edge case
            if np.ndim(probs) == 0: probs = [probs]
            
            # Binarize predictions (Threshold 0.5)
            preds = [1 if p > 0.5 else 0 for p in probs]
            
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            
    # 3. Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n=== FINAL METRICS ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

if __name__ == "__main__":
    evaluate_model()