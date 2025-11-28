# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # --- 1. The Forensic Filter (BayarConv) ---
# class BayarConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=5, padding=2):
#         super(BayarConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
#         self.kernel_size = kernel_size

#     def forward(self, x):
#         # Constrain weights for forensic noise extraction
#         self.conv.weight.data[:, :, self.kernel_size // 2, self.kernel_size // 2] = 0.0
#         sum_weights = self.conv.weight.data.sum(dim=(2, 3), keepdim=True)
#         self.conv.weight.data = self.conv.weight.data / (sum_weights + 1e-7)
#         self.conv.weight.data[:, :, self.kernel_size // 2, self.kernel_size // 2] = -1.0
#         return self.conv(x)

# # --- 2. Double Conv Block for Decoder ---
# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)

# # --- 3. The "Perfect" Forensic-DINO (U-Net Style) ---
# class ForensicDINO(nn.Module):
#     def __init__(self, freeze_dino=False): # Changed default to False logic handling inside
#         super().__init__()
        
#         # A. DINOv2 Backbone
#         self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
#         self.embed_dim = 1024
        
#         # OPTIMIZATION: Freeze most of DINO, Unfreeze ONLY the last block
#         # This gives high accuracy without OOM on RTX 4060
#         for param in self.dino.parameters():
#             param.requires_grad = False
        
#         # Unfreeze the last 2 blocks for adaptation
#         for param in self.dino.blocks[-2:].parameters():
#             param.requires_grad = True
#         self.dino.norm.weight.requires_grad = True
#         self.dino.norm.bias.requires_grad = True

#         # B. Forensic Branch (High Res Features)
#         # We create "Skip Connections" levels
#         self.forensic_l1 = nn.Sequential(BayarConv2d(3, 32), nn.ReLU())  # 518x518
#         self.forensic_l2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()) # 259x259
#         self.forensic_l3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU()) # 129x129
#         self.forensic_l4 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU()) # 65x65

#         # C. Decoder (U-Net Style)
#         # DINO output is 37x37. We need to get back to 518x518
        
#         # Project DINO 1024 -> 512
#         self.bottleneck = nn.Conv2d(1024, 512, 1)
        
#         # Upsample Blocks
#         self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # 37->74
#         self.cat1 = DoubleConv(512 + 256, 256) # Fuse with forensic_l4
        
#         self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # 74->148
#         self.cat2 = DoubleConv(256 + 128, 128) # Fuse with forensic_l3
        
#         self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # 148->296
#         self.cat3 = DoubleConv(128 + 64, 64)   # Fuse with forensic_l2
        
#         self.up4 = nn.Upsample(size=(518, 518), mode='bilinear', align_corners=True) # 296->518
#         self.cat4 = DoubleConv(64 + 32, 32)    # Fuse with forensic_l1
        
#         self.final_mask = nn.Conv2d(32, 1, 1)
        
#         # D. Classification Head
#         self.cls_head = nn.Sequential(
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 1)
#         )

#     def forward(self, x):
#         B, C, H, W = x.shape
        
#         # 1. Forensic Features (Skip Connections)
#         f1 = self.forensic_l1(x)
#         f2 = self.forensic_l2(f1)
#         f3 = self.forensic_l3(f2)
#         f4 = self.forensic_l4(f3)
        
#         # 2. DINO Features
#         dino_out = self.dino.forward_features(x)
#         patch_tokens = dino_out['x_norm_patchtokens'] # (B, 1369, 1024)
        
#         # Reshape DINO
#         grid_size = int(patch_tokens.shape[1]**0.5) # 37
#         dino_map = patch_tokens.permute(0, 2, 1).reshape(B, 1024, grid_size, grid_size)
        
#         # 3. Decoder (The U-Net path)
#         x_dec = self.bottleneck(dino_map) # 1024->512
        
#         # Block 1
#         x_dec = self.up1(x_dec) # 37 -> 74
#         # Resize f4 to match x_dec if slightly off due to padding
#         if f4.shape[2:] != x_dec.shape[2:]:
#             f4 = F.interpolate(f4, size=x_dec.shape[2:], mode='nearest')
#         x_dec = torch.cat([x_dec, f4], dim=1)
#         x_dec = self.cat1(x_dec)
        
#         # Block 2
#         x_dec = self.up2(x_dec)
#         if f3.shape[2:] != x_dec.shape[2:]:
#             f3 = F.interpolate(f3, size=x_dec.shape[2:], mode='nearest')
#         x_dec = torch.cat([x_dec, f3], dim=1)
#         x_dec = self.cat2(x_dec)
        
#         # Block 3
#         x_dec = self.up3(x_dec)
#         if f2.shape[2:] != x_dec.shape[2:]:
#             f2 = F.interpolate(f2, size=x_dec.shape[2:], mode='nearest')
#         x_dec = torch.cat([x_dec, f2], dim=1)
#         x_dec = self.cat3(x_dec)
        
#         # Block 4 (Final Resolution)
#         x_dec = self.up4(x_dec)
#         x_dec = torch.cat([x_dec, f1], dim=1)
#         x_dec = self.cat4(x_dec)
        
#         mask_logits = self.final_mask(x_dec)
        
#         # 4. Classification
#         cls_logits = self.cls_head(dino_out['x_norm_clstoken'])
        
#         return cls_logits, mask_logits



import torch
import torch.nn as nn
import torch.nn.functional as F

class BayarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2):
        super(BayarConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.kernel_size = kernel_size

    def forward(self, x):
        self.conv.weight.data[:, :, self.kernel_size // 2, self.kernel_size // 2] = 0.0
        sum_weights = self.conv.weight.data.sum(dim=(2, 3), keepdim=True)
        self.conv.weight.data = self.conv.weight.data / (sum_weights + 1e-7)
        self.conv.weight.data[:, :, self.kernel_size // 2, self.kernel_size // 2] = -1.0
        return self.conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class UniversalForensicDINO(nn.Module):
    def __init__(self, num_classes=5): # Real, Crop, Inpaint, Copy, AI
        super().__init__()
        
        # Backbone
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.embed_dim = 1024
        
        # Forensic Branch
        self.forensic_l1 = nn.Sequential(BayarConv2d(3, 32), nn.ReLU())
        self.forensic_l2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU())
        self.forensic_l3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU())
        self.forensic_l4 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU())

        # Decoder (Heatmap)
        self.bottleneck = nn.Conv2d(1024, 512, 1)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cat1 = DoubleConv(512 + 256, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cat2 = DoubleConv(256 + 128, 128)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cat3 = DoubleConv(128 + 64, 64)
        self.up4 = nn.Upsample(size=(518, 518), mode='bilinear', align_corners=True)
        self.cat4 = DoubleConv(64 + 32, 32)
        self.final_mask = nn.Conv2d(32, 1, 1)
        
        # HEAD 1: Binary Classification (Is it Fake?)
        self.binary_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        # HEAD 2: Type Classification (How was it faked?)
        # We use the CLS token for this
        self.type_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes) 
        )

        # Freeze DINO optimization (Keep last blocks trainable)
        for param in self.dino.parameters(): param.requires_grad = False
        for param in self.dino.blocks[-2:].parameters(): param.requires_grad = True
        self.dino.norm.weight.requires_grad = True
        self.dino.norm.bias.requires_grad = True

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Forensic Path
        f1 = self.forensic_l1(x)
        f2 = self.forensic_l2(f1)
        f3 = self.forensic_l3(f2)
        f4 = self.forensic_l4(f3)
        
        # DINO Path
        dino_out = self.dino.forward_features(x)
        cls_token = dino_out['x_norm_clstoken']
        patch_tokens = dino_out['x_norm_patchtokens']
        
        # Decoder
        grid_size = int(patch_tokens.shape[1]**0.5)
        dino_map = patch_tokens.permute(0, 2, 1).reshape(B, 1024, grid_size, grid_size)
        x_dec = self.bottleneck(dino_map)
        x_dec = self.cat1(torch.cat([self.up1(x_dec), F.interpolate(f4, size=(grid_size*2, grid_size*2))], dim=1))
        x_dec = self.cat2(torch.cat([self.up2(x_dec), F.interpolate(f3, size=(grid_size*4, grid_size*4))], dim=1))
        x_dec = self.cat3(torch.cat([self.up3(x_dec), F.interpolate(f2, size=(grid_size*8, grid_size*8))], dim=1))
        x_dec = self.cat4(torch.cat([self.up4(x_dec), f1], dim=1))
        
        # OUTPUTS
        mask_logits = self.final_mask(x_dec)       # Heatmap
        binary_logits = self.binary_head(cls_token)# Real vs Fake
        type_logits = self.type_head(cls_token)    # Real vs Crop vs AI
        
        return binary_logits, mask_logits, type_logits