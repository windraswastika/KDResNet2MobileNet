# Implementasi Knowledge Distillation: ResNet-50 → MobileNetV3-Small
**Dataset:** Breast Ultrasound Images (BUSI)

---

## Struktur Proyek

```
project/
├── data/
│   └── BUSI/
│       ├── benign/
│       ├── malignant/
│       └── normal/
├── checkpoints/
├── dataset.py
├── model.py
├── train_teacher.py
├── train_student_kd.py
└── evaluate.py
```

---

## 1. Environment Setup

```bash
pip install torch torchvision scikit-learn scipy matplotlib seaborn
pip install scikit-image
```

---

## 2. `dataset.py`

```python
import os
import numpy as np
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

CLASS_MAP = {"benign": 0, "malignant": 1, "normal": 2}

def load_paths_labels(root_dir):
    paths, labels = [], []
    for cls_name, cls_idx in CLASS_MAP.items():
        cls_dir = os.path.join(root_dir, cls_name)
        for fname in os.listdir(cls_dir):
            if fname.endswith(".png") and "mask" not in fname:
                paths.append(os.path.join(cls_dir, fname))
                labels.append(cls_idx)
    return paths, labels

def speckle_reduce(img_array):
    img_float = img_array.astype(np.float32) / 255.0
    denoised = denoise_tv_chambolle(img_float, weight=0.1, channel_axis=-1)
    return (denoised * 255).astype(np.uint8)

class BUSIDataset(Dataset):
    def __init__(self, paths, labels, transform=None, apply_denoise=True):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.apply_denoise = apply_denoise

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.apply_denoise:
            img = Image.fromarray(speckle_reduce(np.array(img)))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def get_transforms(is_train=True):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def get_dataloaders(root_dir, batch_size=16, seed=42):
    paths, labels = load_paths_labels(root_dir)

    idx = list(range(len(paths)))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, labels, test_size=0.30, stratify=labels, random_state=seed
    )
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )

    def subset(indices, is_train):
        p = [paths[i] for i in indices]
        l = [labels[i] for i in indices]
        return BUSIDataset(p, l, transform=get_transforms(is_train))

    train_ds = subset(idx_train, True)
    val_ds   = subset(idx_val,   False)
    test_ds  = subset(idx_test,  False)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2),
    )
```

---

## 3. `model.py`

```python
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 3

def get_teacher():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def get_student():
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    return model
```

---

## 4. `train_teacher.py`

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import get_teacher
from dataset import get_dataloaders

def unfreeze_schedule(model, epoch):
    if epoch < 10:
        for name, param in model.named_parameters():
            param.requires_grad = "fc" in name
    else:
        for param in model.parameters():
            param.requires_grad = True

def train_teacher(root_dir, save_path="checkpoints/teacher_best.pth", seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_dataloaders(root_dir, seed=seed)

    model = get_teacher().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    best_val_loss = float("inf")
    patience_counter = 0
    EARLY_STOP = 15

    for epoch in range(1, 101):
        unfreeze_schedule(model, epoch)

        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_loss += criterion(out, labels).item()
                correct += (out.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print(f"Early stopping at epoch {epoch}")
                break

if __name__ == "__main__":
    train_teacher(root_dir="data/BUSI")
```

---

## 5. `train_student_kd.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import get_teacher, get_student
from dataset import get_dataloaders

def kd_loss(student_logits, teacher_logits, labels, T, alpha):
    ce = F.cross_entropy(student_logits, labels)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    kd = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)
    return alpha * ce + (1 - alpha) * kd

def grid_search_hyperparams(root_dir, teacher_path, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_dataloaders(root_dir, seed=seed)

    teacher = get_teacher().to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()

    T_values     = [2, 4, 6, 8]
    alpha_values = [0.3, 0.5, 0.7]
    results = []

    for T in T_values:
        for alpha in alpha_values:
            student = get_student().to(device)
            optimizer = Adam(student.parameters(), lr=1e-4, weight_decay=1e-5)

            for epoch in range(20):
                student.train()
                for imgs, labels in train_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    with torch.no_grad():
                        t_logits = teacher(imgs)
                    s_logits = student(imgs)
                    loss = kd_loss(s_logits, t_logits, labels, T, alpha)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            student.eval()
            correct = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    correct += (student(imgs).argmax(1) == labels).sum().item()
            val_acc = correct / len(val_loader.dataset)
            results.append((T, alpha, val_acc))
            print(f"T={T} | alpha={alpha} | val_acc={val_acc:.4f}")

    best = max(results, key=lambda x: x[2])
    print(f"\nBest: T={best[0]}, alpha={best[1]}, val_acc={best[2]:.4f}")
    return best[0], best[1]

def train_student_kd(root_dir, teacher_path, T, alpha,
                     save_path="checkpoints/student_kd_best.pth", seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_dataloaders(root_dir, seed=seed)

    teacher = get_teacher().to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()

    student = get_student().to(device)
    optimizer = Adam(student.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    best_val_loss = float("inf")
    patience_counter = 0
    EARLY_STOP = 15

    for epoch in range(1, 101):
        student.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                t_logits = teacher(imgs)
            s_logits = student(imgs)
            loss = kd_loss(s_logits, t_logits, labels, T, alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        student.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = student(imgs)
                val_loss += F.cross_entropy(out, labels).item()
                correct += (out.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(student.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print(f"Early stopping at epoch {epoch}")
                break

if __name__ == "__main__":
    T, alpha = grid_search_hyperparams("data/BUSI", "checkpoints/teacher_best.pth")
    train_student_kd("data/BUSI", "checkpoints/teacher_best.pth", T=T, alpha=alpha)
```

---

## 6. `evaluate.py`

```python
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from scipy.stats import wilcoxon
from model import get_teacher, get_student
from dataset import get_dataloaders

CLASS_NAMES = ["Benign", "Malignant", "Normal"]
SEEDS = [0, 1, 2, 3, 4]

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def measure_inference_time(model, device, n_runs=100):
    dummy = torch.randn(1, 3, 224, 224).to(device)
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.time()
            model(dummy)
            times.append((time.time() - t0) * 1000)
    return np.mean(times[10:])

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)

def run_multi_seed_eval(model_fn, weight_path_template, test_loader, device):
    accs, precs, recs, f1s = [], [], [], []
    for seed in SEEDS:
        path = weight_path_template.format(seed)
        model = model_fn().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        preds, labels = evaluate_model(model, test_loader, device)
        accs.append(accuracy_score(labels, preds))
        precs.append(precision_score(labels, preds, average="macro", zero_division=0))
        recs.append(recall_score(labels, preds, average="macro", zero_division=0))
        f1s.append(f1_score(labels, preds, average="macro", zero_division=0))
    return np.array(accs), np.array(precs), np.array(recs), np.array(f1s)

def print_metrics(name, accs, precs, recs, f1s):
    print(f"\n{'='*45}")
    print(f"Model: {name}")
    print(f"  Accuracy  : {accs.mean():.4f} ± {accs.std():.4f}")
    print(f"  Precision : {precs.mean():.4f} ± {precs.std():.4f}")
    print(f"  Recall    : {recs.mean():.4f} ± {recs.std():.4f}")
    print(f"  F1-Score  : {f1s.mean():.4f} ± {f1s.std():.4f}")

def plot_confusion_matrix(model, weight_path, test_loader, device, title, save_path):
    model.load_state_dict(torch.load(weight_path, map_location=device))
    preds, labels = evaluate_model(model, test_loader, device)
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def wilcoxon_test(f1_baseline, f1_proposed):
    stat, p = wilcoxon(f1_baseline, f1_proposed)
    print(f"\nWilcoxon Signed-Rank Test (Baseline-Finetune vs Proposed-KD)")
    print(f"  statistic={stat:.4f}, p-value={p:.4f}")
    if p < 0.05:
        print("  → Perbedaan signifikan secara statistik (α=0.05)")
    else:
        print("  → Perbedaan TIDAK signifikan secara statistik (α=0.05)")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders("data/BUSI", seed=42)

    # Evaluasi multi-seed
    # Sesuaikan path template setelah melatih semua seed
    # Contoh: "checkpoints/student_finetune_seed{}.pth"
    # accs_base, precs_base, recs_base, f1s_base = run_multi_seed_eval(
    #     get_student, "checkpoints/student_finetune_seed{}.pth", test_loader, device
    # )
    # accs_kd, precs_kd, recs_kd, f1s_kd = run_multi_seed_eval(
    #     get_student, "checkpoints/student_kd_seed{}.pth", test_loader, device
    # )
    # print_metrics("Baseline-Finetune (S2)", accs_base, precs_base, recs_base, f1s_base)
    # print_metrics("Proposed-KD (S4)", accs_kd, precs_kd, recs_kd, f1s_kd)
    # wilcoxon_test(f1s_base, f1s_kd)

    # Efisiensi model
    teacher = get_teacher().to(device)
    student = get_student().to(device)
    print(f"\nParameter Teacher : {count_params(teacher):.2f}M")
    print(f"Parameter Student : {count_params(student):.2f}M")
    print(f"Inference Teacher : {measure_inference_time(teacher, device):.2f} ms")
    print(f"Inference Student : {measure_inference_time(student, device):.2f} ms")

    # Confusion matrix (gunakan seed terbaik)
    # plot_confusion_matrix(
    #     get_student(), "checkpoints/student_kd_best.pth",
    #     test_loader, device, "Confusion Matrix – Proposed KD",
    #     "checkpoints/cm_kd.png"
    # )
```

---

## 7. Urutan Eksekusi

```bash
# Step 1 – Latih teacher
python train_teacher.py

# Step 2 – Grid search hyperparameter KD, lalu latih student
python train_student_kd.py

# Step 3 – Evaluasi dan bandingkan semua skenario
python evaluate.py
```

---

## 8. Catatan Reprodusibilitas Multi-Seed

Untuk menjalankan 5 seed berbeda, tambahkan loop berikut di script training:

```python
for seed in [0, 1, 2, 3, 4]:
    train_teacher(root_dir="data/BUSI",
                  save_path=f"checkpoints/teacher_seed{seed}.pth",
                  seed=seed)
    train_student_kd(root_dir="data/BUSI",
                     teacher_path=f"checkpoints/teacher_seed{seed}.pth",
                     T=best_T, alpha=best_alpha,
                     save_path=f"checkpoints/student_kd_seed{seed}.pth",
                     seed=seed)
```
