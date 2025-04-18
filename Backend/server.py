# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tempfile
import pdfplumber
import docx2txt
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend interaction

# Load better semantic model
model = SentenceTransformer('all-mpnet-base-v2')

# Helper function to extract text from files using pdfplumber

def extract_text(file):
    filename = secure_filename(file.filename)
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    file.save(temp_path)

    if filename.endswith('.pdf'):
        text = ""
        with pdfplumber.open(temp_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    elif filename.endswith('.docx'):
        return docx2txt.process(temp_path)
    else:
        return ""

# Compute cosine similarity

def get_similarity(resume_text, jd_text):
    emb_resume = model.encode(resume_text, convert_to_tensor=True)
    emb_jd = model.encode(jd_text, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb_resume, emb_jd)
    return round(float(score[0][0]) * 100, 2)

@app.route('/api/screen', methods=['POST'])
def screen_resumes():
    jd_text = request.form.get("job_description")
    resume_files = request.files.getlist("resumes")

    results = []
    for file in resume_files:
        resume_text = extract_text(file)
        score = get_similarity(resume_text, jd_text)
        results.append({
            "name": file.filename,
            "score": score
        })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
