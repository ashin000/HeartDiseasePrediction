import os
import pickle
import datetime
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")  # IMPORTANT for Render
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file
from PIL import Image
import pytesseract

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# -------------------- APP SETUP --------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# -------------------- LOAD MODEL & SCALER --------------------
model = pickle.load(open(os.path.join(BASE_DIR, "models", "heart_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "models", "scaler.pkl"), "rb"))

# -------------------- UTILITY FUNCTIONS --------------------
def safe_float(val):
    try:
        return float(val)
    except:
        return 0.0


def generate_graph(values):
    plt.figure(figsize=(6, 3))
    plt.bar(
        ["Age", "BP", "Cholesterol", "Max HR"],
        values,
        color=["#3498db", "#e74c3c", "#2ecc71", "#f1c40f"]
    )
    plt.title("Patient Health Parameters")
    plt.tight_layout()
    graph_path = os.path.join(STATIC_FOLDER, "result_graph.png")
    plt.savefig(graph_path)
    plt.close()
    return graph_path


def generate_pdf(result, age, bp, chol, max_hr):
    pdf_path = os.path.join(STATIC_FOLDER, "heart_report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Heart Disease Prediction Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 90, f"Date: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M')}")

    c.drawString(50, height - 130, f"Age: {age}")
    c.drawString(50, height - 150, f"Blood Pressure: {bp}")
    c.drawString(50, height - 170, f"Cholesterol: {chol}")
    c.drawString(50, height - 190, f"Max Heart Rate: {max_hr}")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 230, f"Prediction Result: {result}")

    graph_file = os.path.join(STATIC_FOLDER, "result_graph.png")
    if os.path.exists(graph_file):
        c.drawImage(graph_file, 50, height - 480, width=400, height=200)

    c.showPage()
    c.save()
    return pdf_path


def run_ocr(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(img)
        print("OCR TEXT:", text)
        return text
    except Exception as e:
        print("OCR ERROR:", e)
        return ""


def extract_value(pattern, text):
    match = re.search(pattern, text, re.IGNORECASE)
    return safe_float(match.group(1)) if match else 0.0


# -------------------- ROUTES --------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ---------- MANUAL INPUT ----------
        age = safe_float(request.form.get("age"))
        sex = safe_float(request.form.get("sex"))
        cp = safe_float(request.form.get("cp"))
        trestbps = safe_float(request.form.get("trestbps"))
        chol = safe_float(request.form.get("chol"))
        fbs = safe_float(request.form.get("fbs"))
        restecg = safe_float(request.form.get("restecg"))
        thalach = safe_float(request.form.get("thalach"))
        exang = safe_float(request.form.get("exang"))
        oldpeak = safe_float(request.form.get("oldpeak"))
        slope = safe_float(request.form.get("slope"))
        ca = safe_float(request.form.get("ca"))
        thal = safe_float(request.form.get("thal"))

        # ---------- OCR INPUT ----------
        if "report" in request.files and request.files["report"].filename != "":
            file = request.files["report"]
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            ocr_text = run_ocr(file_path)

            if ocr_text:
                age = extract_value(r"age[:\s]+(\d+)", ocr_text)
                trestbps = extract_value(r"blood pressure[:\s]+(\d+)", ocr_text)
                chol = extract_value(r"cholesterol[:\s]+(\d+)", ocr_text)
                thalach = extract_value(r"heart rate[:\s]+(\d+)", ocr_text)

        # ---------- MODEL INPUT ----------
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak,
                                slope, ca, thal]])

        if input_data.shape[1] != scaler.n_features_in_:
            return render_template("result.html", result="Invalid input features.")

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        generate_graph([age, trestbps, chol, thalach])
        generate_pdf(result, age, trestbps, chol, thalach)

        return render_template("result.html", result=result)

    except Exception as e:
        print("PREDICT ERROR:", e)
        return render_template("result.html", result="Internal error occurred.")


@app.route("/download")
def download_report():
    return send_file(
        os.path.join(STATIC_FOLDER, "heart_report.pdf"),
        as_attachment=True
    )


# -------------------- MAIN --------------------
if __name__ == "__main__":
    app.run(debug=True)
