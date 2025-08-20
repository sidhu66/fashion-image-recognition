# Fashion Image Recognition — Vanilla MLP (Keras + TensorFlow)

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/sidhu66/fashion-image-recognition/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

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
**[https://fashion-karanveer.onrender.com](https://fashion-karanveer.onrender.com)**  
(hosted on [Render.com](https://render.com))  
Upload a test image and see what the model predicts!

---

## ✨ Highlights

- Simple, well-commented **Keras MLP** baseline
- Fully reproducible training script
- Auto-saves model + plots
- Deployed via **Flask** + **Gunicorn**
- Ready-to-use **CI badge** with GitHub Actions
- MIT Licensed

---

## 🗂️ Project Structure

├─ app/ 
| ├─app.py
| ├─static
| |  └─uploads
| └─templates
|    └─index.html
├─ models/ # saved models (auto-created)
├─ notebooks/
│ └─ MNIST_Fashion_Project.ipynb # original notebook
├─ src/
│ ├─ inference # model training script
│ └─ models # model inference script
| └─ training
| 
├─ .gitignore
├─ LICENSE
├─ requirements.txt
└─ README.md


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
