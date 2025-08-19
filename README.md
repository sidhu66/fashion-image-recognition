# Fashion-MNIST CNN (Keras/TensorFlow)

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

A clean, reproducible implementation of a CNN on the Fashion‑MNIST dataset using Keras.
Includes a notebook for exploration and a scriptable training pipeline for quick runs, plus
figures and saved models for easy showcasing on GitHub.

## ✨ Highlights
- Minimal, well‑commented **Keras CNN** baseline
- Reproducible training via a **single command**
- Exports **training curves**, **confusion matrix**, and **sample predictions**
- Ready‑to‑use **GitHub Actions CI** to smoke‑test the code
- Clear project structure and **MIT license**

## 🗂️ Project structure
```
.
├─ assets/                # saved plots & figures (auto-created)
├─ models/                # saved models (auto-created)
├─ notebooks/
│  └─ MNIST_Fashion_Project.ipynb   # your original notebook goes here
├─ src/
│  ├─ train.py            # train & export artifacts
│  └─ infer.py            # quick inference on a few samples
├─ .github/workflows/ci.yml
├─ .gitignore
├─ LICENSE
├─ requirements.txt
└─ README.md
```

## 🚀 Quickstart
```bash
# 1) Create & activate a virtualenv (macOS/Linux)
python3 -m venv .venv
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Train (saves model + plots into ./models and ./assets)
python src/train.py --epochs 10 --batch-size 128 --model-path models/fashion_mnist_cnn.keras
```

To run the simple inference demo (prints predicted labels for a few test images and saves a grid figure):
```bash
python src/infer.py --model-path models/fashion_mnist_cnn.keras
```

## 📊 Results (baseline)
> Update with your actual run numbers and images!
- Test accuracy: **~0.90–0.92** (typical baseline)
- Artifacts:
  - `assets/training_curves.png`
  - `assets/confusion_matrix.png`
  - `assets/sample_predictions.png`

## 📝 Using the notebook
Place your notebook at: `notebooks/MNIST_Fashion_Project.ipynb`. Keep both the notebook and
the script so recruiters can see **exploration** *and* **productionized** code.

## 🧪 CI
A minimal GitHub Actions workflow runs lint + a quick import check so your repo shows a green badge.

## 📦 Environment
See `requirements.txt`. If you prefer, export an exact lockfile:
```bash
pip freeze > requirements.lock.txt
```

## 📜 License
This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

### 🧭 What to showcase on GitHub
- Clear README with **what/why/how**, results, and a couple of figures
- Short, readable **source files** (under `src/`)
- **Notebook** for exploration
- **Badges** (CI, license, Python version)
- Optional: link to a short blog post/Gist explaining your choices

> If you find this useful, ⭐ the repo!