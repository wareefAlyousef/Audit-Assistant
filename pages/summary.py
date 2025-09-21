# summry.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
import traceback

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__, template_folder="../templates", static_folder="../static")

UPLOAD_FOLDER = "../uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"csv", "xlsx"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_summary(text):
    """
    Generate summary using OpenAI GPT model
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize the following data in plain language:\n{text}"}
            ],
            temperature=0.5
        )
        summary = response.choices[0].message["content"]
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

@app.route("/")
def index():
    return render_template("summary.html")  # صفحة HTML فيها upload button

@app.route("/get_summary", methods=["POST"])
def get_summary():
    file_path = None
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Unsupported file type"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # قراءة البيانات
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        if df.empty:
            return jsonify({"error": "File is empty"}), 400

        # نحول أول 100 صف إلى نص لتلخيصه
        text_preview = df.head(100).to_string()

        summary = generate_summary(text_preview)
        return jsonify({"summary": summary})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        # تنظيف الملف المؤقت
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
