# FaceNet Face Verification Demo

NN2 · 128-dim embeddings · Triplet Loss · VGGFace2

---

## Project Structure

```
facenet-demo/
├── backend/
│   ├── app.py               ← Flask server (all model code inside)
│   └── requirements.txt
└── frontend/
    └── index.html           ← Open this in browser
```

---

## Setup (Step by Step)

### Step 1 — Install Python dependencies

Open a terminal inside the `backend/` folder:

```bash
cd backend
pip install -r requirements.txt
```

> If you already have PyTorch installed, you can skip torch/torchvision from requirements.txt
> and just run:  pip install flask flask-cors Pillow

### Step 2 — Start the backend

```bash
cd backend
python app.py
```

You should see:
```
[INFO] Loading model from: facenet_model.pth
[INFO] Model loaded successfully ✓
 * Running on http://0.0.0.0:5000
```

### Step 3 — Open the frontend

Simply open `frontend/index.html` in your browser.
No server needed for the frontend — it's a plain HTML file.

**OR** use VS Code's Live Server extension for a cleaner experience:
- Right click `index.html` → Open with Live Server

---

## Using the Demo

1. **Upload or Webcam** — Click "Upload" to select an image from your device,
   or "Webcam" to capture a live photo from your camera.
2. Do the same for Face 2.
3. Adjust the **Threshold** slider if needed (default: 0.900 from training).
4. Click **Compare Faces**.
5. See the verdict, L2 distance, similarity %, and bar chart.

---

## How it Works

```
Browser  →  POST /verify { image1: base64, image2: base64 }
           ↓
Flask    →  decode → PIL → resize 160×160 → normalize(0.5, 0.5)
           ↓
NN2      →  128-dim L2-normalised embedding for each image
           ↓
         distance = ||emb1 - emb2||₂
         is_same  = distance < threshold (1.242)
           ↓
Browser  ←  { distance, similarity_pct, is_same, threshold }
```

---

## Common Issues

| Problem | Fix |
|---|---|
| "Cannot reach backend" | Make sure `python app.py` is running in the backend folder |
| "No module named flask" | Run `pip install flask flask-cors` |
| "FileNotFoundError: facenet_model.pth" | Put the .pth file inside the backend/ folder |
| Webcam not working | Allow camera permission in browser settings |
| CORS error in console | Backend already has CORS enabled — check Flask is running on port 5000 |

---

## Threshold Guide

| Distance | Meaning |
|---|---|
| < 0.8  | Very likely same person |
| 0.8 – 1.242 | Probably same person |
| > 1.242 | Different person (default cutoff) |
| > 1.8  | Clearly different |

The default threshold of **1.242** was determined during LFW evaluation in your training notebook.
