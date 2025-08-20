import os, json
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
import keras 
from PIL import Image
import cv2
import argparse

DEFAULT_MODEL_PATH = os.getenv('MODEL_PATH', 'models/mlp_model.keras')
DEFAULT_LABEL_PATH = os.getenv("LABEL_PATH", 'models/labels.json')

def load_model_and_label(model_path = DEFAULT_MODEL_PATH, label_path = DEFAULT_LABEL_PATH):
    model = keras.models.load_model(model_path)
    with open(label_path) as f:
       m = json.load(f)                 
    labels = [m[str(i)] for i in range(len(m))]
    return model, labels


def process_input_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    #Resize the Image
    img = cv2.resize(img, (28,28))
    
    # Normalize to [0-1]
    arr = np.array(img)/255.
    arr = np.expand_dims(arr, axis=0)
    
    return arr

def predict(image_path, top_k=3):
    model, labels = load_model_and_label()
    arr = process_input_image(image_path)
    probs = model.predict(arr, verbose=0)[0]
    top_idx = probs.argsort()[-top_k:][::-1]
    return [(labels[i], float(probs[i])) for i in top_idx]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    a = p.parse_args()
    for label, score in predict(a.image):
        print(f"{label}: {score:.4f}")