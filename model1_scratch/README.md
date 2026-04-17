# Model 1 — Training from Scratch

This folder contains the implementation of FaceNet trained completely from scratch
using the VGGFace2 dataset.

---

## Files

| File | Description |
|---|---|
| `FaceNet_Trained_Model.ipynb` | Full training pipeline — data loading, triplet loss, semi-hard mining, LFW evaluation |
| `FaceNet_Inference_NewDataset.ipynb` | Run inference using the trained weights on any new dataset on Kaggle |
| `facenet_model.pth` | Final trained model weights |

---

## How to Run Inference on a New Dataset (Kaggle)

1. Upload `facenet_model.pth` as a Kaggle dataset
2. Open `FaceNet_Inference_NewDataset.ipynb` in a Kaggle notebook
3. Update the model path in the notebook:
```python
   MODEL_PATH = "/kaggle/input/your-dataset-name/facenet_model.pth"
```
4. Add your image dataset to Kaggle and update the image paths
5. Run all cells

---
