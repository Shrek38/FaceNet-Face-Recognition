# FaceNet Face Recognition — Comparative Study

A deep learning project implementing and comparing two approaches to face verification
using the FaceNet framework: training from scratch vs fine-tuning a pretrained model.

---

## Overview

This project explores the FaceNet approach to face recognition, where a deep neural
network maps face images into a compact embedding space. Similar faces are closer
together and dissimilar faces are farther apart in this space. We implemented two
different training strategies and compared their performance on a standard benchmark.

---

## Models

### Model 1 — Training from Scratch
- Trained a deep neural network from scratch using the **VGGFace2 dataset**
- No pretrained weights used
- Model learns facial embeddings directly from raw data
- **LFW Accuracy: 92.95%**

### Model 2 — Fine-tuning Pretrained Model
- Used a pretrained **InceptionResNetV1** model (pretrained on VGGFace2)
- Fine-tuned on the **CASIA-WebFace dataset**
- Initial layers frozen, deeper layers retrained
- **LFW Accuracy: 94.60%**

---

## Datasets

| Dataset | Purpose | Size |
|---|---|---|
| VGGFace2 | Training from scratch | ~3.31M images |
| CASIA-WebFace | Fine-tuning | ~494K images |
| LFW (Labeled Faces in the Wild) | Evaluation / Testing | ~13.2K images |

---

## Training Details

| Parameter | Value |
|---|---|
| Optimizer | SGD + AdaGrad |
| Learning Rate | 0.05 |
| Input Size | 160 × 160 |
| Batch Size | 360 |
| Epochs | 20 |
| Early Stopping Patience | 3 |
| Loss Function | Triplet Loss |

---

## Results

| Model | Training Type | LFW Accuracy |
|---|---|---|
| Model 1 | From Scratch | 92.95% |
| Model 2 | Fine-tuned (InceptionResNetV1) | **94.60%** |

The fine-tuned model outperformed the scratch model despite CASIA-WebFace being
significantly smaller than VGGFace2. This demonstrates the power of transfer learning —
pretrained weights provide strong feature initialization, leading to faster convergence
and better generalization.

---

## Evaluation Metrics

- Verification Accuracy on LFW dataset
- Embedding distance threshold: **1.242** (as specified in the FaceNet paper)
- ROC Curve (Receiver Operating Characteristic)
- AUC Score (Area Under the Curve)

---

## Repository Structure
