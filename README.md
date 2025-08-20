# Fashion Image Recognition — Vanilla MLP (Keras + TensorFlow)

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


A basic, learning-focused project that trains a vanilla fully-connected neural network (MLP) to classify Fashion-MNIST clothing images.  
The goal is to understand the full workflow — data preprocessing → training → saving → deployment — with a simple baseline before moving to CNNs.

---

## 🧪 Status: Work in Progress

This project is still evolving as I learn. It currently uses a simple **vanilla MLP** (not a CNN), so accuracy is not perfect.  
But that's the point — the current model is a baseline to help me understand the end-to-end ML + deployment pipeline.

I plan to update this with better models (CNNs, regularization, etc.) as I progress.

---

## 🌐 Live Demo

🚀 Try it out:  
**[https://fashion-karanveer.onrender.com](https://fashion-image-recognition.onrender.com/)**  
(hosted on [Render.com](https://render.com))  
Upload a test image and see what the model predicts!

---

## ✨ Highlights

- Simple, well-commented **Keras MLP** baseline
- Fully reproducible training script
- Auto-saves model + plots
- Deployed via **Flask** + **Gunicorn**


---

## 🗂️ Project Structure
<pre> . 
├── <a href="https://github.com/sidhu66/fashion-image-recognition/tree/main/app">app/</a> 
│ ├── <a href="https://github.com/sidhu66/fashion-image-recognition/blob/main/app/app.py">app.py</a> # Flask entry point 
│ ├── <a href="https://github.com/sidhu66/fashion-image-recognition/tree/main/app/static">static/</a> 
│ │ └── <a href="https://github.com/sidhu66/fashion-image-recognition/tree/main/app/static/uploads">uploads/</a> # Uploaded image storage 
│ └── <a href="https://github.com/sidhu66/fashion-image-recognition/tree/main/models/app/templates">templates/</a> 
│ | └── <a href="https://github.com/sidhu66/fashion-image-recognition/blob/main/app/templates/index.html">index.html</a> # Web interface 
├── <a href="https://github.com/sidhu66/fashion-image-recognition/tree/main/models">models/</a> # Saved models (auto-created) 
├── <a href="https://github.com/sidhu66/fashion-image-recognition/tree/main/notebooks">notebooks/</a> 
│ └── <a href="https://github.com/sidhu66/fashion-image-recognition/blob/main/notebooks/MNIST_Fashion_Project.ipynb">MNIST_Fashion_Project.ipynb</a> # Original notebook 
├── <a href="https://github.com/sidhu66/fashion-image-recognition/tree/main/src">src/</a> 
│ ├── <a href="https://github.com/sidhu66/fashion-image-recognition/tree/main/src/inference">inference/</a> # Inference logic 
| ├── <a href="https://github.com/sidhu66/fashion-image-recognition/tree/main/src/models">models/</a> # Model Building Logic
│ └── <a href="https://github.com/sidhu66/fashion-image-recognition/tree/main/src/training">training/</a> # Training logic 
├── <a href="https://github.com/sidhu66/fashion-image-recognition/blob/main/.gitignore">.gitignore</a> 
├── <a href="https://github.com/sidhu66/fashion-image-recognition/blob/main/LICENSE">LICENSE</a> 
├── <a href="https://github.com/sidhu66/fashion-image-recognition/blob/main/requirements.txt">requirements.txt</a> 
└── <a href="https://github.com/sidhu66/fashion-image-recognition/blob/main/README.md">README.md</a> </pre>


---

## 🚀 Quickstart

```bash
# Create & activate virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Train model
python src/train.py --epochs 10 

# Run inference
python src/infer.py --image image.jpg


# To run the Flask app locally:
gunicorn app.app:app --bind 0.0.0.0:8000
```

## 🙋‍♂️ Want to Help or Learn More?

⭐ this repo if it helped you learn too

Fork it and try adding a CNN, dropout, or batch normalization

Reach out on GitHub if you have ideas to improve this further
