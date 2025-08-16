import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from PIL import Image, ImageFilter
from datasets import load_dataset
import numpy as np
import torch.nn.functional as F
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import Subset


# load dataset 
dataset = load_dataset("tanganke/stanford_cars")

# required preprocessing
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # required for resnet mean
        std=[0.229, 0.224, 0.225]    # required for resnet standard dev
    )
])

# augmentation pipeline
augment_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=1))),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),  # Gaussian noise
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # required for resnet mean
        std=[0.229, 0.224, 0.225]    # required for resnet standard dev
    )
])



class StanfordCarsDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        img = example['image'].convert('RGB')  # already a PIL.Image
        label =example['label']
        img = self.transform(img)
        return img, torch.tensor(label)
    

# ----- split indices for train/val (10% val) -----
total = len(dataset['train'])
val_size = int(0.10 * total)
train_size = total - val_size
gen = torch.Generator().manual_seed(42)
train_idx_subset, val_idx_subset = torch.utils.data.random_split(
    range(total), [train_size, val_size], generator=gen
)
hf_train = Subset(dataset['train'], train_idx_subset.indices)
hf_val   = Subset(dataset['train'], val_idx_subset.indices)

# ----- wrappers -----
train_base = StanfordCarsDataset(hf_train, base_transform)
train_aug  = StanfordCarsDataset(hf_train, augment_transform)  # augment only training
train_combined = ConcatDataset([train_base, train_aug])

val_set  = StanfordCarsDataset(hf_val,  base_transform)        # no aug
test_set = StanfordCarsDataset(dataset["test"], base_transform)

# ----- loaders -----
BATCH_SIZE = 64
train_loader = DataLoader(train_combined, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,       batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_set,      batch_size=BATCH_SIZE, shuffle=False)

        

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#----------------------------------MODEL ARCHITECTURE---------------------------------------------------------------------------------------------------
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX



class ResNet50Backbone(nn.Module):
    # def __init__(self):
    #     super().__init__()
    #     resnet = resnet50(weights=None)  # no pretrained because you load your own weights

    #     #Please note that this section assumes that the 
    #     #program is being used on a cuda-enabled device (loads from gpu). map_loation="cpu" is used to fix this for non-cuda devices
    #     state_dict = torch.load("resnet50_stanfordcars.pth", map_location="cuda")
    #     resnet.load_state_dict(state_dict)

    #     # Don't avg pool
    #     self.backbone = nn.Sequential(
    #         resnet.conv1,   # [B, 64, 112, 112]
    #         resnet.bn1,
    #         resnet.relu,
    #         resnet.maxpool,
    #         resnet.layer1,
    #         resnet.layer2,
    #         resnet.layer3,
    #         resnet.layer4    # [B, 2048, 7, 7]
    #     )

    # def forward(self, x):
    #     return self.backbone(x)  # [B, 2048, 7, 7]


    '''
    def __init__(self, ckpt_path, device="cuda"):
        super().__init__()
        resnet = resnet50(weights=None)

        sd = torch.load(ckpt_path, map_location=device)
        sd_backbone = {k: v for k, v in sd.items()
                       if k.startswith(("conv1","bn1","layer1","layer2","layer3","layer4"))}
        resnet.load_state_dict(sd_backbone, strict=False)  # will skip fc/avgpool

        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

    def forward(self, x):
        return self.backbone(x)

'''
    def __init__(self, ckpt_path, device="cuda", freeze=True):
        super().__init__()
        resnet = resnet50(weights=None)

        # Load checkpoint
        sd = torch.load(ckpt_path, map_location=device)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}  # remove DP prefix

        # Keep only backbone layers
        sd_backbone = {k: v for k, v in sd.items()
                       if k.startswith(("conv1","bn1","layer1","layer2","layer3","layer4"))}

        # Load into resnet
        missing, unexpected = resnet.load_state_dict(sd_backbone, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)





class Navigator(nn.Module):
    def __init__(self,in_channels=2048,num_regions=8):
        super().__init__()
        self.num_regions = num_regions

        #convv head to score proposed region
        self.score_head = nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),  # [32, 1, H, W]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,1,kernel_size=1)
        )
        #PAST HERE IS PURE CHATGPT :-)
    def forward(self, feature_map):
        score_map = self.score_head(feature_map)  # [B, 1, H, W]
        B, _, H, W = score_map.shape
        score_map = score_map.view(B, -1)  # [B, H*W]

        # Top-k most informative regions
        topk_scores, topk_indices = torch.topk(score_map, self.num_regions, dim=1)

        return topk_scores, topk_indices  # [B, k], [B, k]




def extract_topk_patches(img, topk_indices, patch_size=64, fmap_size=7, img_size=224):
    """
    Extract image patches based on top-k indices from the Navigator.

    Args:
        img: Tensor of shape [B, 3, 224, 224]
        topk_indices: Tensor of shape [B, k]
        patch_size: size of the patch to crop (default: 64x64)
        fmap_size: spatial size of the feature map (default: 7x7)
        img_size: size of the original input image (default: 224)

    Returns:
        patches: Tensor of shape [B, k, 3, patch_size, patch_size]
    """
    B, _, H, W = img.shape
    k = topk_indices.shape[1]
    stride = img_size // fmap_size  # 224 / 7 = 32
    patches = []

    for b in range(B):
        batch_patches = []
        for idx in topk_indices[b]:
            idx = idx.item()
            row = idx // fmap_size
            col = idx % fmap_size

            # Compute crop coordinates
            center_x = col * stride + stride // 2
            center_y = row * stride + stride // 2

            # Crop patch from original image
            x1 = max(center_x - patch_size // 2, 0)
            y1 = max(center_y - patch_size // 2, 0)
            x2 = min(center_x + patch_size // 2, img_size)
            y2 = min(center_y + patch_size // 2, img_size)

            patch = img[b, :, y1:y2, x1:x2]

            # Resize to standard patch size (in case edges are clipped)
            patch = F.interpolate(patch.unsqueeze(0), size=(patch_size, patch_size), mode='bilinear', align_corners=False)
            batch_patches.append(patch)

        patches.append(torch.cat(batch_patches, dim=0))  # [k, 3, patch_size, patch_size]

    return torch.stack(patches)  # [B, k, 3, patch_size, patch_size]














'''
class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        # Always use pretrained ImageNet1K_V1 weights
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Backbone up to avgpool (exclude fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # [B, 2048, 1, 1]
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # Single output score per patch
        self.score_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, 1),
            nn.Sigmoid()
        )
        
    def forward(self, patches):
        B, k, C, H, W = patches.shape
        patches = patches.view(B * k, C, H, W)
        
        features = self.backbone(patches)          # [B*k, 2048, 1, 1]
        pooled = self.avgpool(features).view(B*k, -1)  # [B*k, 2048]
        
        scores = self.score_layer(pooled)          # [B*k, 1]
        scores = scores.view(B, k)
        
        return scores
'''

class Teacher(nn.Module):
    def __init__(self, ckpt_path, device="cuda"):
        super().__init__()
        resnet = resnet50(weights=None)

        # Load fine-tuned backbone weights
        sd = torch.load(ckpt_path, map_location=device)
        sd_backbone = {k: v for k, v in sd.items()
                       if k.startswith(("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"))}
        resnet.load_state_dict(sd_backbone, strict=False)

        # Backbone up to avgpool (exclude fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # [B, 2048, 1, 1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Increased capacity in the scoring head
        self.score_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, 1024),  # extra hidden layer
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, patches):
        B, k, C, H, W = patches.shape
        patches = patches.view(B * k, C, H, W)

        features = self.backbone(patches)               # [B*k, 2048, 1, 1]
        pooled = self.avgpool(features).view(B * k, -1) # [B*k, 2048]

        scores = self.score_layer(pooled)               # [B*k, 1]
        scores = scores.view(B, k)

        return scores




# def reorder_patches_by_teacher_scores(patches, scores):
#     """
#     patches: [B, k, 3, H, W]
#     scores: [B, k]
#     returns reordered patches by descending score
#     """
#     sorted_indices = torch.argsort(scores, dim=1, descending=True)  # [B, k]
    
#     B, k, C, H, W = patches.shape
#     batch_indices = torch.arange(B).unsqueeze(1).expand(-1, k)  # [B, k]
    
#     reordered_patches = patches[batch_indices, sorted_indices, :, :, :]  # [B, k, 3, H, W]
#     return reordered_patches

def reorder_patches_by_teacher_scores(patches, scores):
    """
    patches: [B, k, 3, H, W]
    scores:  [B, k]
    returns patches sorted by descending score per example
    """
    sorted_idx = torch.argsort(scores, dim=1, descending=True)  # [B, k]
    B, k, C, H, W = patches.shape
    idx = sorted_idx.view(B, k, 1, 1, 1).expand(B, k, C, H, W)
    return torch.gather(patches, dim=1, index=idx)





'''
class Scrutinizer(nn.Module):
    def __init__(self, num_classes=196):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1
        resnet = resnet50(weights=weights)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # [B, 2048, 1, 1]
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)
        
    def forward(self, patches):
        B, k, C, H, W = patches.shape
        patches = patches.view(B * k, C, H, W)
        
        features = self.backbone(patches)      # [B*k, 2048, 1, 1]
        pooled = self.avgpool(features).view(B, k, -1)  # [B, k, 2048]
        
        agg_features = pooled.mean(dim=1)     # average across patches [B, 2048]
        out = self.fc(agg_features)           # [B, num_classes]
        return out
'''

# class Scrutinizer(nn.Module):
#     def __init__(self, ckpt_path, num_classes=196, device="cuda"):
#         super().__init__()

#         # Load ResNet50 with no pretrained weights
#         resnet = resnet50(weights=None)

#         # Load only the backbone weights from your fine-tuned model
#         sd = torch.load(ckpt_path, map_location=device)
#         sd_backbone = {k: v for k, v in sd.items()
#                        if k.startswith(("conv1","bn1","layer1","layer2","layer3","layer4"))}
#         resnet.load_state_dict(sd_backbone, strict=False)  # skip fc/avgpool

#         # Backbone (same as Teacher & Backbone modules)
#         self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # [B, 2048, 1, 1]
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

#         # Increased capacity classification head
#         self.fc = nn.Sequential(
#             nn.Linear(resnet.fc.in_features, 1024),  # 2048 → 1024
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(1024, 512),  # 1024 → 512
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(512, num_classes)  # Final classification layer
#         )

#     def forward(self, patches):
#         B, k, C, H, W = patches.shape
#         patches = patches.view(B * k, C, H, W)

#         # Feature extraction
#         features = self.backbone(patches)  # [B*k, 2048, 1, 1]
#         pooled = self.avgpool(features).view(B, k, -1)  # [B, k, 2048]

#         # Average patch features
#         agg_features = pooled.mean(dim=1)  # [B, 2048]

#         # Classification
#         out = self.fc(agg_features)  # [B, num_classes]
#         return out

class Scrutinizer(nn.Module):
    def __init__(self, ckpt_path, num_classes=196, device="cuda"):
        super().__init__()
        resnet = resnet50(weights=None)

        # load backbone weights only
        sd = torch.load(ckpt_path, map_location=device)
        sd_backbone = {k: v for k, v in sd.items()
                       if k.startswith(("conv1","bn1","layer1","layer2","layer3","layer4"))}
        resnet.load_state_dict(sd_backbone, strict=False)

        # CNN trunk (no pool/fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # [B, 2048, 1, 1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # beefier head (yours)
        self.fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, patches, scores=None):
        # patches: [B, k, 3, H, W], scores: [B, k] teacher scores or None
        B, k, C, H, W = patches.shape
        x = patches.view(B * k, C, H, W)
        feats = self.backbone(x)                          # [B*k, 2048, 1, 1]
        pooled = self.avgpool(feats).view(B, k, -1)       # [B, k, 2048]

        if scores is not None:
            w = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, k, 1]
            agg = (pooled * w).sum(dim=1)                   # [B, 2048]
        else:
            agg = pooled.mean(dim=1)

        return self.fc(agg)








def navigator_ranking_loss(nav_scores, teacher_scores, margin=0.1):
    """
    nav_scores: [B, k] - Navigator's scores for k patches
    teacher_scores: [B, k] - Teacher's scores for k patches (as soft ground truth)
    margin: margin for ranking loss
    """
    B, k = nav_scores.shape
    loss = 0.0
    count = 0
    
    for i in range(B):
        for j in range(k):
            for l in range(k):
                if teacher_scores[i, j] > teacher_scores[i, l]:
                    # We want nav_scores[i, j] > nav_scores[i, l] + margin
                    target = torch.ones(1, device=nav_scores.device)
                    loss += F.margin_ranking_loss(nav_scores[i, j].unsqueeze(0), 
                                                 nav_scores[i, l].unsqueeze(0), 
                                                 target, margin=margin)
                    count += 1
    if count > 0:
        loss /= count
    return loss






def teacher_bce_loss(teacher_scores, nav_scores, threshold=0.5):
    """
    teacher_scores: [B, k] (Teacher's output, sigmoid activated)
    nav_scores: [B, k] (Navigator's output before sigmoid, or after if you prefer)
    
    We treat patches with nav_scores > threshold as positives.
    """
    pos_mask = (nav_scores > threshold).float()
    neg_mask = 1.0 - pos_mask
    labels = pos_mask  # pseudo labels from Navigator
    
    bce = F.binary_cross_entropy(teacher_scores, labels)
    return bce




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate models
backbone = ResNet50Backbone("resnet50_stanfordcars.pth", device=device).to(device)
navigator = Navigator().to(device)
teacher = Teacher("resnet50_stanfordcars.pth", device= device).to(device)
scrutinizer = Scrutinizer("resnet50_stanfordcars.pth", num_classes=196, device=device).to(device)

#------------------------------------------------------------------------------


# ------------ training loop ------------------------
backbone.eval()
for p in backbone.parameters(): p.requires_grad = False
teacher.eval()
for p in teacher.parameters():  p.requires_grad = False

optimizer = torch.optim.Adam(
    list(navigator.parameters()) + list(scrutinizer.parameters()), lr=1e-4
)
criterion = nn.CrossEntropyLoss()

epochs = 10

tr_loss_hist, va_loss_hist = [], []
tr_acc_hist,  va_acc_hist  = [], []

with open("training_log.txt", "w") as log_file:
    log_file.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    for epoch in range(1, epochs+1):
        # ---- TRAIN ----
        navigator.train(); scrutinizer.train()
        running_loss = 0.0; correct = 0; total = 0

        for images, labels in train_loader:
            images = images.to(device); labels = labels.to(device)
            optimizer.zero_grad()

            # 1) regions
            feat_map = backbone(images)                           # [B, 2048, 7, 7]
            nav_scores, topk_indices = navigator(feat_map)        # [B, k], [B, k]
            patches = extract_topk_patches(images, topk_indices, 64).to(device)

            # 2) teacher ranking + scrutinizer
            t_scores = teacher(patches)                           # [B, k]
            patches  = reorder_patches_by_teacher_scores(patches, t_scores)
            logits   = scrutinizer(patches, scores=t_scores)      # [B, 196]

            # 3) losses
            loss_nav  = navigator_ranking_loss(nav_scores, t_scores)
            loss_cls  = criterion(logits, labels)
            loss      = loss_cls + 1.0 * loss_nav
            loss.backward(); optimizer.step()

            # 4) stats
            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc  = 100.0 * correct / total
        tr_loss_hist.append(train_loss); tr_acc_hist.append(train_acc)

        # ---- VAL ----
        navigator.eval(); scrutinizer.eval()
        val_running_loss = 0.0; v_correct = 0; v_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device); labels = labels.to(device)

                feat_map = backbone(images)
                nav_scores, topk_indices = navigator(feat_map)
                patches = extract_topk_patches(images, topk_indices, 64).to(device)

                t_scores = teacher(patches)
                patches  = reorder_patches_by_teacher_scores(patches, t_scores)
                logits   = scrutinizer(patches, scores=t_scores)

                v_loss = criterion(logits, labels)
                val_running_loss += v_loss.item()

                preds = logits.argmax(dim=1)
                v_total += labels.size(0)
                v_correct += (preds == labels).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_acc  = 100.0 * v_correct / v_total
        va_loss_hist.append(val_loss); va_acc_hist.append(val_acc)

        print(f"Epoch [{epoch}/{epochs}] "
              f"Train: loss {train_loss:.4f} acc {train_acc:.2f}% | "
              f"Val: loss {val_loss:.4f} acc {val_acc:.2f}%")

        with open("training_log.txt", "a") as log_file2:
            log_file2.write(f"{epoch},{train_loss:.6f},{train_acc:.2f},{val_loss:.6f},{val_acc:.2f}\n")

# ----- accuracy curve -----
plt.figure(figsize=(7,5))
plt.plot(range(1, epochs+1), tr_acc_hist, label="Train Acc")
plt.plot(range(1, epochs+1), va_acc_hist, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy: Train vs Val")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# (optional) loss curve
plt.figure(figsize=(7,5))
plt.plot(range(1, epochs+1), tr_loss_hist, label="Train Loss")
plt.plot(range(1, epochs+1), va_loss_hist, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss: Train vs Val")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


log_file.close()




torch.save({
     'backbone': backbone.state_dict(),
     'navigator': navigator.state_dict(),
     'teacher': teacher.state_dict(),
     'scrutinizer': scrutinizer.state_dict(),
 }, 'ntsnet.pth')


# =========================
# TEST ONLY: load & evaluate
# =========================
def load_models_for_test(device):
    # Recreate architectures
    backbone = ResNet50Backbone("resnet50_stanfordcars.pth", device=device).to(device)
    navigator = Navigator().to(device)
    teacher   = Teacher("resnet50_stanfordcars.pth", device=device).to(device)
    scrutinizer = Scrutinizer("resnet50_stanfordcars.pth", num_classes=196, device=device).to(device)

    # Load trained weights
    ckpt = torch.load("ntsnet.pth", map_location=device)
    backbone.load_state_dict(ckpt["backbone"])
    navigator.load_state_dict(ckpt["navigator"])
    teacher.load_state_dict(ckpt["teacher"])
    scrutinizer.load_state_dict(ckpt["scrutinizer"])
    return backbone, navigator, teacher, scrutinizer



'''
@torch.no_grad()


def evaluate(backbone, navigator, teacher, scrutinizer, loader, device):
    backbone.eval(); navigator.eval(); teacher.eval(); scrutinizer.eval()
    correct = total = 0

    for images, labels in loader:
        images = images.to(device); labels = labels.to(device)

        # Backbone -> regions -> patches
        feat_map = backbone(images)                                 # [B, 2048, 7, 7]
        nav_scores, topk_indices = navigator(feat_map)              # [B, k], [B, k]
        patches = extract_topk_patches(images, topk_indices, 64).to(device)  # [B, k, 3, 64, 64]

        # Rank by teacher and classify with scrutinizer
        t_scores = teacher(patches)                                 # [B, k]
        patches = reorder_patches_by_teacher_scores(patches, t_scores)
        logits  = scrutinizer(patches)                              # [B, 196]
        preds   = logits.argmax(dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

    acc = 100.0 * correct / total
    return acc

# ---- run test ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone, navigator, teacher, scrutinizer = load_models_for_test(device)

test_acc = evaluate(backbone, navigator, teacher, scrutinizer, test_loader, device)
print(f"Test accuracy: {test_acc:.2f}%")
'''



import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

@torch.no_grad()
def evaluate_with_samples(backbone, navigator, teacher, scrutinizer, loader, device, dataset):
    backbone.eval(); navigator.eval(); teacher.eval(); scrutinizer.eval()
    correct = total = 0

    all_preds = []
    all_labels = []
    all_images = []

    for images, labels in loader:
        images = images.to(device); labels = labels.to(device)

        # Backbone -> regions -> patches
        feat_map = backbone(images)
        nav_scores, topk_indices = navigator(feat_map)
        patches = extract_topk_patches(images, topk_indices, 64).to(device)

        # Rank by teacher and classify with scrutinizer
        t_scores = teacher(patches)
        patches  = reorder_patches_by_teacher_scores(patches, t_scores)
        logits   = scrutinizer(patches, scores=t_scores)

        preds   = logits.argmax(dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_images.append(images.cpu())

    # Flatten tensors
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_images = torch.cat(all_images).numpy()  # [N, 3, H, W]

    acc = 100.0 * correct / total

    # Confusion matrix and per-class accuracy
    num_classes = 196
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    # Identify 5 worst classes
    worst_idx = np.argsort(per_class_acc)[:5]

    # Identify 5 best classes
    best_idx = np.argsort(per_class_acc)[-5:][::-1]
    
    print(f"Test accuracy: {acc:.2f}%\n")

    print("\n5 best classes by index and accuracy:")
    for idx in best_idx:
        print(f"Class {idx} - Accuracy: {per_class_acc[idx]*100:.2f}%")

    print("5 worst classes by index and accuracy:")
    for idx in worst_idx:
        print(f"Class {idx} - Accuracy: {per_class_acc[idx]*100:.2f}%")


    # Show one sample image per worst class
    for idx in worst_idx:
        # Find first occurrence of this class in the dataset
        class_positions = np.where(all_labels == idx)[0]
        if len(class_positions) == 0:
            continue
        img = all_images[class_positions[0]]  # [3, H, W]
        img = np.transpose(img, (1,2,0))      # Convert to HWC
        img = (img * 0.5 + 0.5)               # Denormalize from [-1,1] to [0,1]
        plt.imshow(img)
        plt.title(f"Sample image - Class {idx}")
        plt.axis('off')
        plt.show()

    return acc, cm, per_class_acc



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone, navigator, teacher, scrutinizer = load_models_for_test(device)

test_acc, cm, per_class_acc = evaluate_with_samples(
    backbone, navigator, teacher, scrutinizer, test_loader, device, dataset
)


