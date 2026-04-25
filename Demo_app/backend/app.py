"""
FaceNet Demo Backend — Flask API
Uses pretrained InceptionResnetV1 (VGGFace2) + MTCNN from facenet-pytorch.
Install: pip install facenet-pytorch flask flask-cors
"""

import io
import base64
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN

from flask import Flask, request, jsonify
from flask_cors import CORS

# ─────────────────────────────────────────────
# 1. Load Models
# ─────────────────────────────────────────────

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.9   # good default for pretrained vggface2 model (L2 distance)

print(f"[INFO] Device: {DEVICE}")

# MTCNN — detects and crops faces to 160x160
mtcnn = MTCNN(
    image_size=160,
    margin=20,           # extra pixels around the detected face
    keep_all=False,      # return only the most prominent face
    device=DEVICE,
    post_process=True,   # returns normalized tensor ready for resnet
)

# Pretrained FaceNet — InceptionResnetV1 trained on VGGFace2
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

print("[INFO] Pretrained FaceNet + MTCNN loaded successfully.")

# ─────────────────────────────────────────────
# 2. Helper Functions
# ─────────────────────────────────────────────

def decode_base64_image(b64_str):
    """Decode a base64 image string (with or without data URI prefix) to PIL Image."""
    if ',' in b64_str:
        b64_str = b64_str.split(',', 1)[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert('RGB')


@torch.no_grad()
def get_embedding(pil_img):
    """
    1. Run MTCNN to detect and crop the face
    2. Run InceptionResnetV1 to get 512-D L2-normalized embedding
    Returns: numpy array of shape (512,), or raises ValueError if no face found
    """
    # MTCNN returns a (3, 160, 160) tensor if face found, else None
    face_tensor = mtcnn(pil_img)

    if face_tensor is None:
        raise ValueError("No face detected in the image. Please use a clear face photo.")

    # Add batch dimension and move to device
    face_tensor = face_tensor.unsqueeze(0).to(DEVICE)

    # Get embedding — already L2 normalized by InceptionResnetV1
    embedding = resnet(face_tensor).squeeze(0).cpu().numpy()
    return embedding


def l2_distance(e1, e2):
    return float(np.sum((e1 - e2) ** 2))


# ─────────────────────────────────────────────
# 3. Flask App
# ─────────────────────────────────────────────

app = Flask(__name__)
CORS(app)


@app.route('/verify', methods=['POST'])
def verify():
    try:
        data = request.get_json()
        if not data or 'image1' not in data or 'image2' not in data:
            return jsonify({'error': 'Missing image1 or image2'}), 400

        threshold = float(data.get('threshold', THRESHOLD))

        img1 = decode_base64_image(data['image1'])
        img2 = decode_base64_image(data['image2'])

        emb1 = get_embedding(img1)
        emb2 = get_embedding(img2)

        distance   = l2_distance(emb1, emb2)
        is_same    = bool(distance <= threshold)

        # Similarity: 100% at distance=0, 0% at distance=threshold*2
        similarity = max(0.0, min(100.0, (1.0 - distance / (threshold * 2)) * 100))

        return jsonify({
            'distance':   round(float(distance), 4),
            'threshold':  round(float(threshold), 4),
            'is_same':    is_same,
            'similarity': round(float(similarity), 1),
        })

    except ValueError as ve:
        # Face not detected — send a clean message to frontend
        return jsonify({'error': str(ve)}), 422

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'device': str(DEVICE)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)