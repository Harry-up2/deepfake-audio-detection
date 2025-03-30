# Deepfake Audio Detection

## Overview
This project focuses on detecting deepfake audio using three different approaches:
1. **Machine Learning Approach**: Feature extraction and model comparison.
2. **Deep Learning Approach**: CNN-based model with Mel-Spectrograms.
3. **Pre-trained Deep Learning Model**: Fine-tuning ResNet for best performance.

## Approaches and Results

### 1️⃣ Machine Learning Approach
- Extracted key audio features (e.g., mean, std, min, max, Mel-Spectrograms).
- Evaluated multiple ML models, selecting **HistGradientBoosting** as the best.
- **Best Model**: HistGradientBoosting.
- **AUC Score**: **0.999812** (training) | **0.99059** (hidden test data).

### 2️⃣ Deep Learning Approach (CNN)
- Converted audio to **Mel-Spectrograms** using **Librosa**.
- Built a **CNN model** with batch normalization and dropout for better generalization.
- Implemented **early stopping** to prevent overfitting.
- **AUC Score**: **0.99208**.

### 3️⃣ Pre-trained Deep Learning Model (ResNet)
- Fine-tuned **ResNet-18** by modifying the fully connected layers.
- Used **dropout and batch normalization** to improve training.
- **Best Performance Achieved**: **AUC = 0.99994**.

## Dependencies
To run this project, install the following dependencies:
```bash
pip install numpy pandas librosa torch torchvision torchaudio tensorflow tqdm lightgbm catboost
```

## Running the Models

### Machine Learning Approach
Run `mlapproach.ipynb` to extract features and train ML models.

### Deep Learning Approach
Run `dlapproach.ipynb` to train a CNN-based model using Mel-Spectrograms.

### Pre-trained Model (ResNet)
Run `dlpresnet.ipynb` to fine-tune ResNet-18 for deepfake audio detection.

## Key Takeaways
- **Feature engineering** played a crucial role in the ML-based approach.
- **CNN with Mel-Spectrograms** provided a deep learning alternative with good results.
- **Pre-trained ResNet** significantly outperformed other models with an **AUC of 0.99994**.

## Future Work
- Experiment with **larger pre-trained models**.
- Apply **self-supervised learning** to improve feature extraction.
- Test on **real-world deepfake datasets** for robustness.

---
🚀 **Best Model:** **Fine-tuned ResNet-18** (AUC = **0.99994**).

