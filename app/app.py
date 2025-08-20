import os
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from src.inference.predict import predict

ALLOWED_EXTENTIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret")

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR/"app"/"static"/"uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENTIONS

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)

@app.route("/", methods=["POST"])
def handle_predict():
    if "image" not in request.files:
        flash("No File part in request")
        return redirect(url_for("index"))
    file = request.files["image"]
    if file.filename == "":
        flash("No File Selected")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Unsupported File Type")
        return redirect(url_for("index"))
    
    filename = secure_filename(file.filename)
    save_path = UPLOAD_DIR / filename
    file.save(save_path)
    
    try:
        result = predict(str(save_path))
        return render_template("index.html", result={
            "filename":filename,
            "url": url_for('static', filename=f"uploads/{filename}"),
            "topk": [{"label": lbl, "score": float(score)} for lbl, score in result],
            "best": result[0][0]
        })
    except Exception as e:
        app.logger.exception(e)
        flash("Prediction Failed")
        return redirect(url_for("index"))
    
@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)