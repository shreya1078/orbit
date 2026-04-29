🌿 Cassava Leaf Disease Classification using EfficientNet
📌 Overview

This project focuses on classifying cassava leaf diseases using a deep learning approach. Accurate detection of plant diseases helps farmers take early action and improve crop productivity.

We use EfficientNet along with Stratified K-Fold Cross Validation to achieve robust and reliable performance.

🎯 Objectives
Classify cassava leaf images into multiple disease categories
Handle class imbalance effectively
Improve model generalization using cross-validation
Optimize performance using F1-score
📂 Dataset
Loaded using PyTorch ImageFolder
Images organized into class-specific folders
5 disease categories
Imbalanced dataset handled using stratified sampling
⚙️ Methodology
🔹 Model
Pretrained EfficientNet (Transfer Learning)
Final classification layer modified for 5 classes
🔹 Training Strategy
Stratified K-Fold Cross Validation (K = 3)
Maintains class distribution in each fold
🔹 Techniques Used
Data augmentation (flip, rotation, resize)
Mixed precision training (AMP)
GPU acceleration
F1-score as evaluation metric
📊 Results
Fold	Best F1 Score
Fold 1	0.8310
Fold 2	0.8284
Fold 3	0.8203
✅ Final Best F1 Score: 0.8310
🚀 Features
Handles imbalanced dataset effectively
Uses efficient deep learning architecture
Cross-validation for better generalization
Optimized training pipeline
🛠️ Tech Stack
Python
PyTorch
Torchvision
Scikit-learn
▶️ How to Run
git clone https://github.com/your-username/repo-name.git
cd repo-name
pip install -r requirements.txt
python train.py
📌 Future Improvements
Try EfficientNet-B3/B4 for better accuracy
Add test-time augmentation (TTA)
Deploy model using Streamlit
Integrate real-time disease detection
