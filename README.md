# 🌿 Cassava Leaf Disease Classification using EfficientNet-B0

## 📌 Overview

This project classifies cassava leaf images into 5 disease categories using **EfficientNet-B0** with transfer learning. The pipeline handles real-world challenges like class imbalance and limited data, using Stratified K-Fold Cross Validation for reliable evaluation.

---

## 🎯 Objectives

- Classify cassava leaf images into 5 disease categories
- Handle class imbalance using per-fold weighted loss
- Improve generalization with Stratified K-Fold Cross Validation
- Optimize using weighted F1-score

---

## 📂 Dataset

- Source: Kaggle — Cassava Leaf Disease Classification
- Loaded via PyTorch `ImageFolder`
- 5 disease classes with imbalanced distribution
- Class weights computed per fold (`1 / class_count`) to address imbalance

---

## ⚙️ Methodology

### Model — EfficientNet-B0

- Pretrained on ImageNet (`EfficientNet_B0_Weights.DEFAULT`)
- Custom classification head:
  ```
  Dropout(p=0.4) → Linear(1280 → 5)
  ```

### Training Config

| Parameter     | Value                        |
|---------------|------------------------------|
| Folds         | 3 (Stratified K-Fold)        |
| Epochs        | 3 per fold                   |
| Batch Size    | 32                           |
| Learning Rate | 3e-4 (Adam)                  |
| Scheduler     | CosineAnnealingLR            |
| Loss          | CrossEntropyLoss (weighted)  |
| Precision     | Mixed (AMP + GradScaler)     |
| Device        | CUDA (GPU)                   |

### Data Augmentation

| Split | Transforms |
|-------|-----------|
| Train | `RandomResizedCrop(224)`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation(15)`, `ColorJitter` |
| Val   | `Resize(256)` → `CenterCrop(224)` |

Both normalized with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

---

## 📊 Results

> **Evaluation Metric: Weighted F1-Score**
> The model is evaluated using weighted F1-score, which accounts for class imbalance by weighting each class's F1 by its support (number of samples). This is the primary metric used to select and save the best model.

### Fold 1
| Epoch | Train Loss | Weighted F1 |
|-------|------------|-------------|
| 1     | 1.0072     | 0.7736      |
| 2     | 0.8007     | **0.8310** ✅ global best |
| 3     | 0.6922     | 0.8222      |

### Fold 2
| Epoch | Train Loss | Weighted F1 |
|-------|------------|-------------|
| 1     | 1.0181     | 0.7903      |
| 2     | 0.7784     | 0.8018      |
| 3     | 0.6998     | 0.8284      |

### Fold 3
| Epoch | Train Loss | Weighted F1 |
|-------|------------|-------------|
| 1     | 0.9888     | 0.7991      |
| 2     | 0.7882     | 0.8126      |
| 3     | 0.6978     | 0.8203      |

### Summary

| Fold   | Best Weighted F1 |
|--------|-----------------|
| Fold 1 | **0.8310** ✅    |
| Fold 2 | 0.8284          |
| Fold 3 | 0.8203          |

🏆 **Final Best Weighted F1-Score: 0.8310** — model saved as `efficientnet_best.pth`

---

## �️ Tech Stack

- Python
- PyTorch + Torchvision
- Scikit-learn
- CUDA (GPU acceleration)

---

## ▶️ How to Run

```bash
git clone https://github.com/shreya1078/orbit.git
cd orbit
pip install torch torchvision scikit-learn
# Open and run main/train.ipynb in Jupyter or Kaggle
```

> The notebook is configured for Kaggle. Update `DATA_PATH` if running locally.

---

## 📌 Future Improvements

- Try EfficientNet-B3/B4 for better accuracy
- Add test-time augmentation (TTA)
- Deploy model using Streamlit
- Integrate real-time disease detection
