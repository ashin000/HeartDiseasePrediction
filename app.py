from flask import Flask, render_template, request
import pickle, os, re
import pandas as pd
import numpy as np
import pytesseract, cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)


model = pickle.load(open("models/heart_model.pkl", "rb"))
FEATURES = pickle.load(open("models/features.pkl", "rb"))  # strict feature order


def extract_text(image_file):
    image = Image.open(image_file).convert("RGB")
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray)

def extract_values(text):
    values = {}
    patterns = {
        "age": r"Age[:\s]+(\d+)",
        "chol": r"Cholesterol[:\s]+(\d+)",
        "trestbps": r"BP[:\s]+(\d+)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            values[key] = int(match.group(1))
    return values


def create_graph(chol, bp):
    os.makedirs("static", exist_ok=True)
    plt.figure(figsize=(5, 3))
    plt.bar(["Cholesterol", "Blood Pressure"], [chol, bp])
    plt.title("Heart Health Parameters")
    plt.tight_layout()
    plt.savefig("static/result_graph.png")
    plt.close()


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    if "manual" in request.form:
        input_dict = {
            "age": int(request.form["age"]),
            "sex": int(request.form["sex"]),
            "cp": int(request.form["cp"]),
            "trestbps": int(request.form["trestbps"]),
            "chol": int(request.form["chol"]),
            "fbs": int(request.form["fbs"]),
            "restecg": int(request.form["restecg"]),
            "thalach": int(request.form["thalach"]),
            "exang": int(request.form["exang"]),
            "oldpeak": float(request.form["oldpeak"]),
            "slope": int(request.form["slope"]),
            "ca": int(request.form["ca"]),
            "thal": int(request.form["thal"]),
        }

    else:
        image = request.files["report"]
        text = extract_text(image)
        values = extract_values(text)

        input_dict = {
            "age": values.get("age", 50),
            "sex": 1,
            "cp": 0,
            "trestbps": values.get("trestbps", 120),
            "chol": values.get("chol", 200),
            "fbs": 0,
            "restecg": 1,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 1,
            "ca": 0,
            "thal": 2,
        }

    chol = input_dict["chol"]
    bp = input_dict["trestbps"]

    if chol < 100 or chol > 400 or bp < 80 or bp > 250:
        return render_template(
            "result.html",
            result="Invalid clinical values detected",
            chol=chol,
            bp=bp
        )

    input_df = pd.DataFrame([input_dict], columns=FEATURES)

    pred = model.predict(input_df)[0]
    result = "Heart Disease Detected" if pred == 1 else "No Heart Disease"

    create_graph(chol, bp)

    return render_template(
        "result.html",
        result=result,
        chol=chol,
        bp=bp
    )

if __name__ == "__main__":
    app.run(debug=True)
