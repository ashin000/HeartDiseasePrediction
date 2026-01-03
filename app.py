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

# Load model & scaler
model = pickle.load(open("models/heart_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# ---------- CV FUNCTIONS ----------

def extract_text(image_file):
    image = Image.open(image_file).convert("RGB")
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray)

def extract_values(text):
    values = {}
    age = re.search(r'Age[:\s]+(\d+)', text)
    chol = re.search(r'Cholesterol[:\s]+(\d+)', text)
    bp = re.search(r'BP[:\s]+(\d+)', text)

    if age: values["age"] = int(age.group(1))
    if chol: values["chol"] = int(chol.group(1))
    if bp: values["trestbps"] = int(bp.group(1))
    return values

# ---------- GRAPH ----------

def create_graph(chol, bp):
    if not os.path.exists("static"):
        os.makedirs("static")

    plt.figure(figsize=(5,3))
    plt.bar(["Cholesterol", "Blood Pressure"], [chol, bp])
    plt.title("Heart Health Parameters")
    plt.tight_layout()
    plt.savefig("static/result_graph.png")
    plt.close()

# ---------- ROUTES ----------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    # Manual input
    if "manual" in request.form:
        data = [
            int(request.form["age"]),
            int(request.form["sex"]),
            int(request.form["cp"]),
            float(request.form["trestbps"]),
            float(request.form["chol"]),
            int(request.form["fbs"]),
            int(request.form["restecg"]),
            float(request.form["thalach"]),
            int(request.form["exang"]),
            float(request.form["oldpeak"]),
            int(request.form["slope"]),
            int(request.form["ca"]),
            int(request.form["thal"])
        ]
        chol = float(request.form["chol"])
        bp = float(request.form["trestbps"])

    # CV input
    else:
        image = request.files["report"]
        text = extract_text(image)
        values = extract_values(text)

        chol = values.get("chol", 200)
        bp = values.get("trestbps", 120)

        data = [
            values.get("age", 50),
            1, 0, bp, chol,
            0, 1, 150, 0, 1.0, 1, 0, 2
        ]

    input_df = pd.DataFrame([data], columns=[
        "age","sex","cp","trestbps","chol",
        "fbs","restecg","thalach","exang",
        "oldpeak","slope","ca","thal"
    ])

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    pred = model.predict(input_scaled)[0]
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
