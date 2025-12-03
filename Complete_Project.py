# Project.py
import os
import io
import json
from datetime import datetime

from flask import (
    Flask,
    request,
    jsonify,
    render_template_string,
    redirect,
    url_for,
    session,
)

from google import genai
from google.genai import types

from PyPDF2 import PdfReader       # for PDF
import docx                        # for DOCX

# --- MongoDB Imports ---
from pymongo import MongoClient
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
import gridfs

# --- Flask Setup ---

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24).hex())

# --- MongoDB Setup ---

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "textbook_quality_db")

mongo_client = MongoClient(MONGODB_URI)
mongo_db = mongo_client[MONGODB_DB_NAME]

users_collection = mongo_db["users"]
evaluations_collection = mongo_db["evaluations"]
files_fs = gridfs.GridFS(mongo_db)  # for storing uploaded document binaries

# --- Gemini API Setup ---

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

try:
    client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
    if client:
        print("Gemini Client initialized successfully.")
    else:
        print("Gemini Client not initialized: GEMINI_API_KEY missing.")
except Exception as e:
    print(f"ERROR: Could not initialize Gemini Client. Error: {e}")
    client = None

# --- Gemini Schema and Evaluation Logic ---

EVALUATION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "accuracy_score": types.Schema(
            type=types.Type.NUMBER,
            description="Numerical score from 0 to 100 representing content accuracy."
        ),
        "accuracy": types.Schema(
            type=types.Type.STRING,
            description=(
                "Short explanation/justification of the accuracy score, "
                "commenting on factual correctness and domain alignment."
            ),
        ),
        "readability_score": types.Schema(
            type=types.Type.NUMBER,
            description="Numerical score from 0 to 100 representing readability."
        ),
        "readability": types.Schema(
            type=types.Type.STRING,
            description=(
                "Short explanation/justification of the readability score, "
                "including clarity, structure, and ease of understanding."
            ),
        ),
        "consistency_score": types.Schema(
            type=types.Type.NUMBER,
            description="Numerical score from 0 to 100 representing internal consistency."
        ),
        "consistency": types.Schema(
            type=types.Type.STRING,
            description=(
                "Short explanation/justification of the consistency score, "
                "commenting on terminology, notation, tone, and logical flow."
            ),
        ),
        "overall_rating": types.Schema(
            type=types.Type.NUMBER,
            description=(
                "Overall quality rating on a 0 to 5 scale. "
                "You may use one decimal place (e.g., 4.3)."
            ),
        ),
        "summary": types.Schema(
            type=types.Type.STRING,
            description=(
                "2–4 sentence summary of the main strengths and weaknesses "
                "of the material."
            ),
        ),
    },
    required=[
        "accuracy_score",
        "accuracy",
        "readability_score",
        "readability",
        "consistency_score",
        "consistency",
        "overall_rating",
        "summary",
    ],
)

# ---------- Helper to extract text from non-image files ----------

def extract_text_from_file_bytes(filename: str, raw_bytes: bytes) -> str:
    """
    Extracts text from bytes of an uploaded file.
    Supports PDF, DOCX and TXT. Images are handled separately.
    """
    filename_lower = (filename or "").lower()

    if filename_lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages_text = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:
                continue
        text = "\n".join(pages_text)

    elif filename_lower.endswith(".docx"):
        doc = docx.Document(io.BytesIO(raw_bytes))
        text = "\n".join(p.text for p in doc.paragraphs)

    elif filename_lower.endswith(".txt"):
        text = raw_bytes.decode("utf-8", errors="ignore")

    else:
        raise ValueError("Unsupported document type. Please upload PDF, DOCX, or TXT.")

    # Limit text size to avoid overloading the model (trim very long books)
    MAX_CHARS = 20000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    if not text.strip():
        raise ValueError("No readable text found in the uploaded file.")

    return text


# ---------- Gemini evaluation (text and/or image) ----------

def evaluate_textbook_gemini(
    text_content: str | None = None,
    image_bytes: bytes | None = None,
    image_mime: str | None = None,
):
    """
    Calls Gemini to evaluate either:
      - pure text (pdf/doc/txt), or
      - an image (photo of a page), or
      - both together.
    Returns a dict following EVALUATION_SCHEMA.
    """
    if client is None:
        return {
            "error": "Analysis Error: Gemini Client not initialized. Set GEMINI_API_KEY correctly."
        }

    if (not text_content or not text_content.strip()) and image_bytes is None:
        return {"error": "No content to analyze."}

    prompt = (
        "You are an educational content quality reviewer. "
        "You will be given either textual content or an image of textbook/reference book pages. "
        "If an image is provided, read the text from the image first, then evaluate it.\n\n"
        "Based only on the provided material (it may be partial), evaluate the content on:\n\n"
        "1. Accuracy – Give a numerical score from 0 to 100 for factual correctness and "
        "   alignment with standard knowledge in the field, and a short explanation.\n"
        "2. Readability – Give a numerical score from 0 to 100 for clarity, structure, "
        "   and ease of understanding, and a short explanation.\n"
        "3. Consistency – Give a numerical score from 0 to 100 for internal consistency "
        "   (terminology, notation, tone, logical flow), and a short explanation.\n"
        "4. Overall Rating – Give one overall quality rating on a 0 to 5 scale "
        "   (you may use one decimal place, e.g., 4.2).\n\n"
        "Return your answer strictly using the provided JSON schema, filling:\n"
        "- accuracy_score (0–100), accuracy (text explanation)\n"
        "- readability_score (0–100), readability (text explanation)\n"
        "- consistency_score (0–100), consistency (text explanation)\n"
        "- overall_rating (0–5, number)\n"
        "- summary (2–4 sentences summary).\n"
    )

    contents: list = [prompt]

    if text_content and text_content.strip():
        contents.append(text_content)

    if image_bytes is not None:
        mime = image_mime or "image/jpeg"
        img_part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
        contents.append(img_part)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EVALUATION_SCHEMA,
                temperature=0.0,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini API call failed: {e}")
        return {
            "error": f"Analysis Error: API call failed or returned invalid data. Details: {e}"
        }


# ===================== Flask Template Strings =====================

login_html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Textbook Quality Assessment — Login</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
  <script src="https://unpkg.com/lucide@latest"></script>
  <style>
    :root{
      --color-primary: #2563EB;
      --color-accent: #22C55E;
      --color-bg-start: #0f172a;
      --color-bg-end: #1d4ed8;
      --color-card: #FFFFFF;
      --color-text: #111827;
      --color-warn: #DC2626;
    }
    *{ box-sizing:border-box; margin: 0; padding: 0; }
    body{
      min-height:100vh; color:var(--color-text);
      font-family: 'Inter', sans-serif;
      background: radial-gradient(circle at top, var(--color-bg-end), var(--color-bg-start));
      display:flex; justify-content:center; align-items:center;
      padding: 20px;
      background-color: #e5e7eb;
    }
    .card-container {
        opacity: 0;
        transform: translateY(20px);
        animation: fadeIn 0.5s ease-out forwards;
        animation-delay: 0.2s;
        width: 100%;
        max-width: 380px;
    }
    @keyframes fadeIn {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .card{
      background: var(--color-card);
      border-radius: 12px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.1), 0 0 0 1px rgba(0,0,0,0.05);
      padding: 35px;
      width: 100%;
      transition: all 0.3s ease;
      border-top: 5px solid var(--color-primary);
    }
    .head{ text-align: center; margin-bottom: 25px; }
    .title-wrapper {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        margin-bottom: 5px;
        transition: all 0.3s ease;
    }
    .title-wrapper:hover{
        transform: scale(1.05);
        transition: all 0.3s ease;
    }
    .title{
        font-size: 24px; font-weight: 800; color: var(--color-primary);
        letter-spacing: -0.5px;
    }
    .subtitle{ color: var(--color-text); font-size: 13px; opacity: 0.75; }
    .field{ margin-bottom: 20px; }
    label{ display: block; font-size: 14px; margin-bottom: 6px; font-weight: 600; color: var(--color-text); }
    .input{
      width: 100%; padding: 12px; border: 1px solid #D1D5DB; border-radius: 8px;
      font-size: 16px; transition: border-color 0.3s, box-shadow 0.3s;
      background-color: #F9FAFB;
    }
    .input:focus{
        border-color: var(--color-primary);
        outline: none;
        box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.25);
    }
    .actions{ display: flex; justify-content: space-between; gap: 15px; margin-top: 25px; }
    .btn{
      flex: 1; border: none; border-radius: 8px; color: white; font-weight: 700;
      padding: 12px 15px; cursor: pointer; font-size: 15px;
      transition: background-color 0.3s, transform 0.1s, box-shadow 0.3s;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.08);
    }
    .btn:active {
        transform: translateY(1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
    }
    .btn.primary{ background-color: var(--color-primary); }
    .btn.primary:hover{ background-color: #1d4ed8; }
    .btn.alt{ background-color: var(--color-accent); }
    .btn.alt:hover{ background-color: #16a34a; }
    .error{ color: var(--color-warn); text-align: center; margin-top: 15px; font-size: 13px; font-weight: 600; }
    .message{ color: var(--color-accent); text-align: center; margin-top: 15px; font-size: 13px; font-weight: 600; }
    .login-form:hover {
        transform: scale(1.02);
        transition: all 0.4s ease;
    }
  </style>
</head>
<body>
  <div class="card-container">
    <main class="card" role="main">
      <header class="head">
        <div class="title-wrapper">
            <i data-lucide="book-open-check" style="color: var(--color-primary); width: 26px; height: 26px;"></i>
            <div class="title">TEXTBOOK QUALITY APP</div>
            <i data-lucide="star" style="color: var(--color-accent); width: 24px; height: 24px;"></i>
        </div>
        <div class="subtitle">Sign in to assess textbooks, reference books, and e-books.</div>
      </header>
      {% if message %}<p class="message">{{ message }}</p>{% endif %}
      {% if error %}<p class="error" role="alert">{{ error }}</p>{% endif %}
      <form action="{{ url_for('login') }}" method="post" autocomplete="on" class="login-form">
        <div class="field">
          <label for="username">Email or User ID</label>
          <input id="username" class="input" type="text" name="username" placeholder="e.g., reviewer@university.edu" required/>
        </div>
        <div class="field">
          <label for="password">Password</label>
          <input id="password" class="input" type="password" name="password" placeholder="••••••••" required/>
        </div>
        <div class="actions">
          <button type="submit" class="btn primary">Login</button>
          <button type="button" class="btn alt" onclick="window.location.href = '{{ url_for('register') }}'">Sign Up</button>
        </div>
      </form>
    </main>
  </div>
  <script>
    lucide.createIcons();
  </script>
</body>
</html>
"""

register_html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Textbook Quality Assessment — Registration</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
  <script src="https://unpkg.com/lucide@latest"></script>
  <style>
    :root{
      --color-primary: #2563EB;
      --color-accent: #22C55E;
      --color-bg-start: #f1f5f9;
      --color-bg-end: #e5e7eb;
      --color-card: #FFFFFF;
      --color-text: #111827;
      --color-warn: #DC2626;
    }
    *{ box-sizing:border-box; margin: 0; padding: 0; }
    body{
      min-height:100vh; color:var(--color-text);
      font-family: 'Inter', sans-serif;
      background: linear-gradient(135deg, var(--color-bg-start) 0%, var(--color-bg-end) 100%);
      display:flex; justify-content:center; align-items:center;
      padding: 20px;
    }
    .card-container {
        opacity: 0;
        transform: translateY(20px);
        animation: fadeIn 0.5s ease-out forwards;
        animation-delay: 0.2s;
        width: 100%;
        max-width: 420px;
    }
    @keyframes fadeIn {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .card{
      background: var(--color-card);
      border-radius: 12px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.04);
      padding: 35px;
      width: 100%;
      transition: all 0.3s ease;
      border-top: 5px solid var(--color-accent);
    }
    .head{ text-align: center; margin-bottom: 25px; }
    .title-wrapper {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        margin-bottom: 5px;
        transition: all 0.3s ease;
    }
    .title{
        font-size: 24px; font-weight: 800; color: var(--color-accent);
        letter-spacing: -0.5px;
    }
    .subtitle{ color: var(--color-text); font-size: 13px; opacity: 0.75; }
    .field{ margin-bottom: 20px; }
    label{ display: block; font-size: 14px; margin-bottom: 6px; font-weight: 600; color: var(--color-text); }
    .input{
      width: 100%; padding: 12px; border: 1px solid #D1D5DB; border-radius: 8px;
      font-size: 16px; transition: border-color 0.3s, box-shadow 0.3s;
      background-color: #F9FAFB;
    }
    .input:focus{
        border-color: var(--color-accent);
        outline: none;
        box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.2);
    }
    .actions{ display: flex; justify-content: space-between; gap: 15px; margin-top: 25px; }
    .btn{
      flex: 1; border: none; border-radius: 8px; color: white; font-weight: 700;
      padding: 12px 15px; cursor: pointer; font-size: 15px;
      transition: background-color 0.3s, transform 0.1s, box-shadow 0.3s;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.08);
    }
    .btn:active {
        transform: translateY(1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
    }
    .btn.primary{ background-color: var(--color-accent); }
    .btn.primary:hover{ background-color: #16a34a; }
    .btn.alt{ background-color: var(--color-primary); }
    .btn.alt:hover{ background-color: #1d4ed8; }
    .error{ color: var(--color-warn); text-align: center; margin-top: 15px; font-size: 13px; font-weight: 600; }
    .message{ color: var(--color-accent); text-align: center; margin-top: 15px; font-size: 13px; font-weight: 600; }
  </style>
</head>
<body>
  <div class="card-container">
    <main class="card" role="main">
      <header class="head">
        <div class="title-wrapper">
            <i data-lucide="user-plus" style="color: var(--color-accent); width: 26px; height: 26px;"></i>
            <div class="title">CREATE REVIEWER ACCOUNT</div>
            <i data-lucide="shield-check" style="color: var(--color-primary); width: 24px; height: 24px;"></i>
        </div>
        <div class="subtitle">Register to assess the quality of textbooks and e-resources.</div>
      </header>
      {% if message %}<p class="message">{{ message }}</p>{% endif %}
      {% if error %}<p class="error" role="alert">{{ error }}</p>{% endif %}
      <form action="{{ url_for('register') }}" method="post" autocomplete="on">
        <div class="field">
          <label for="username">Username</label>
          <input id="username" class="input" type="text" name="username" placeholder="Choose a username" required/>
        </div>
        <div class="field">
          <label for="password">Password</label>
          <input id="password" class="input" type="password" name="password" placeholder="Create a strong password" required/>
        </div>
        <div class="actions">
          <button type="submit" class="btn primary">Register Account</button>
          <button type="button" class="btn alt" onclick="window.location.href = '{{ url_for('login') }}'">Back to Login</button>
        </div>
      </form>
    </main>
  </div>
  <script>
    lucide.createIcons();
  </script>
</body>
</html>
"""

# ===================== MAIN APP WITH CAMERA OPTION =====================

main_app_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Application for Assessment of Quality of Textbook / Reference Books / E-Books</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
        :root{
            --color-primary: #2563EB;
            --color-accent: #22C55E;
            --color-text: #0F172A;
            --color-muted: #6B7280;
            --color-bg-content: rgba(255,255,255,0.96);
            --color-bg-page: #EFF6FF;
            --color-warn: #EF4444;
        }
        *{ box-sizing:border-box; margin: 0; padding: 0; }
        html, body { height:100%; }
        body {
            color: var(--color-text);
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Inter", sans-serif;
            background-color: var(--color-bg-page);
            background-image: linear-gradient(to bottom right, #dbeafe, #eff6ff);
            background-repeat: no-repeat;
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        .page { min-height: 100%; display: flex; flex-direction: column; }
        header.topbar {
            background-color: rgba(255, 255, 255, 0.95);
            position: sticky; top: 0; z-index: 1000;
            backdrop-filter: blur(6px);
            border-bottom: 1px solid #E5E7EB;
            display: flex; align-items: center; justify-content: space-between;
            padding: 10px 20px;
        }
        .brand {
            font-weight: 800; font-size: 16px; letter-spacing:.2px;
            color: var(--color-primary);
        }
        .brand span {
            display:block;
            font-size: 11px;
            font-weight:500;
            color: var(--color-muted);
        }
        .header-actions {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .action-btn {
            border: none;
            border-radius: 999px;
            color: var(--color-primary);
            font-weight:600;
            padding: 6px 12px;
            cursor:pointer;
            font-size: 12px;
            background-color: rgba(37, 99, 235, 0.1);
            transition: background-color 0.2s, transform 0.1s;
        }
        .action-btn:hover {
            transform: translateY(-1px);
            background-color: rgba(37, 99, 235, 0.18);
        }
        .logout-btn {
            border: none; border-radius: 999px; color:white; font-weight:600;
            padding: 6px 12px; cursor:pointer; font-size: 12px;
            background-color: var(--color-accent);
            transition: background-color 0.2s, transform 0.1s;
        }
        .logout-btn:hover { transform: translateY(-1px); background-color: #16a34a; }

        .hero {
            text-align: center; color: #0b1120; padding: 32px 16px 20px;
        }
        .hero h1 {
            font-size: 22px;
            font-weight: 800;
            margin-bottom: 6px;
        }
        .hero p {
            font-size: 14px;
            color: var(--color-muted);
            max-width: 650px;
            margin: 0 auto;
        }

        main { flex:1; padding: 10px; }
        .content {
            max-width: 780px; margin: 0 auto 24px;
            background: var(--color-bg-content);
            border-radius: 14px;
            padding: 18px 18px 22px;
            box-shadow: 0 5px 18px rgba(15, 23, 42, 0.12);
            border: 1px solid #E5E7EB;
        }
        h1.title {
            text-align: left; font-weight: 800; color: var(--color-primary);
            font-size: 20px; margin: 0 0 12px 0;
        }
        .subtitle-main {
            font-size: 13px;
            color: var(--color-muted);
            margin-bottom: 16px;
        }

        .file-wrap {
            border-radius: 10px;
            padding: 10px;
            background-color: #F9FAFB;
            border: 1px dashed #CBD5F5;
            margin-bottom: 16px;
        }
        .file-wrap:focus-within {
            background-color: #F3F4FF;
            box-shadow: 0 6px 18px rgba(37, 99, 235, 0.05);
        }
        input[type="file"] {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 8px;
            border: 2px solid var(--color-primary);
            cursor: pointer;
            font-size: 14px;
            background-color: #fff;
            transition: border-color .25s ease, box-shadow .25s ease, transform .2s ease;
        }
        input[type="file"]:hover {
            border-color: var(--color-accent);
            box-shadow: 0 0 6px rgba(37, 99, 235, 0.35);
            transform: translateY(-1px);
        }
        input[type="file"]:focus,
        input[type="file"]:focus-visible {
            outline: none;
            border-color: var(--color-accent);
            box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.3);
            transform: translateY(-1px);
        }

        .hint {
            font-size: 12px;
            color: var(--color-muted);
            margin-bottom: 10px;
        }

        .actions { display:flex; gap:10px; justify-content:flex-start; align-items:center; margin-top: 6px; flex-wrap: wrap; }
        .btn {
            border: none; border-radius: 6px; color:white; font-weight: 600; font-size: 14px;
            padding: 8px 16px; cursor: pointer; min-width: 120px;
            background-color: var(--color-primary);
            transition: background-color 0.2s, transform 0.1s, box-shadow 0.2s;
        }
        .btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 3px 8px rgba(15, 23, 42, 0.18);
            background-color: #1d4ed8;
        }
        .btn.secondary {
            background-color: #111827;
        }
        .btn.secondary:hover {
            background-color: #020617;
        }
        .btn.outline {
            background-color: transparent;
            color: var(--color-primary);
            border: 1px solid var(--color-primary);
        }
        .btn.outline:hover {
            background-color: rgba(37, 99, 235, 0.06);
        }

        /* CAMERA SECTION */
        .camera-wrap {
            border-radius: 10px;
            padding: 10px;
            background-color: #F9FAFB;
            border: 1px dashed #A7F3D0;
        }
        .camera-title {
            font-size: 13px;
            font-weight: 700;
            color: var(--color-accent);
            margin-bottom: 6px;
        }
        .camera-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 10px;
        }
        @media (min-width: 640px) {
            .camera-grid {
                grid-template-columns: 2fr 1fr;
            }
        }
        .camera-preview-box {
            background-color: #000;
            border-radius: 8px;
            overflow: hidden;
            position: relative;
            min-height: 200px;
        }
        video#cameraVideo, canvas#cameraCanvas {
            width: 100%;
            height: 100%;
            object-fit: contain;
            background-color: #000;
        }
        .camera-side {
            font-size: 12px;
            color: var(--color-muted);
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .camera-status {
            font-size: 12px;
            color: var(--color-muted);
        }
        .divider-or {
            text-align: center;
            font-size: 11px;
            color: var(--color-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 10px 0;
        }

        #result {
            margin-top: 18px;
            border-radius: 12px;
            overflow: hidden;
            opacity: 0;
            transform: translateY(-8px);
            transition: opacity .35s ease, transform .35s ease;
        }
        #result.show {
            opacity: 1;
            transform: translateY(0);
        }
        .result-card {
            background: #ffffff;
            border-radius: 12px;
            border: 1px solid #E5E7EB;
            padding: 16px 18px;
            font-size: 14px;
            color: var(--color-text);
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
        }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            gap: 8px;
            flex-wrap: wrap;
        }
        .result-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            background: rgba(37, 99, 235, 0.08);
            color: var(--color-primary);
        }
        .result-pill-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: var(--color-accent);
        }
        .result-time {
            font-size: 11px;
            color: var(--color-muted);
        }
        .result-title {
            margin: 4px 0 6px 0;
            font-size: 18px;
            font-weight: 800;
            color: var(--color-primary);
        }
        .result-subtitle {
            font-size: 13px;
            color: var(--color-muted);
            margin-bottom: 10px;
        }
        .result-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 10px;
        }
        @media (min-width: 640px) {
            .result-grid {
                grid-template-columns: 1fr 1fr;
            }
        }
        .result-section {
            background: #F9FAFB;
            border-radius: 10px;
            padding: 8px 10px;
            border: 1px solid #E5E7EB;
        }
        .result-section h3 {
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: .06em;
            color: var(--color-muted);
        }
        .result-section p {
            margin: 0;
            font-size: 13px;
            line-height: 1.5;
        }
        .result-note {
            margin-top: 10px;
            font-size: 12px;
            color: var(--color-muted);
            border-top: 1px dashed #E5E7EB;
            padding-top: 6px;
        }

        .history-panel-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(15, 23, 42, 0.6);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 2000;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.25s, visibility 0.25s;
        }
        .history-panel-overlay.active {
            opacity: 1;
            visibility: visible;
        }
        .history-panel-content {
            background: var(--color-bg-content);
            border-radius: 12px;
            padding: 22px;
            width: 90%;
            max-width: 520px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.45);
            position: relative;
            transform: scale(0.94);
            transition: transform 0.25s ease-out;
        }
        .history-panel-overlay.active .history-panel-content {
            transform: scale(1);
        }
        .close-btn {
            position: absolute;
            top: 8px;
            right: 12px;
            background: none;
            border: none;
            font-size: 22px;
            color: var(--color-muted);
            cursor: pointer;
            line-height: 1;
            transition: color 0.2s;
        }
        .close-btn:hover {
            color: var(--color-warn);
        }
        .history-title {
            font-size: 18px;
            margin-top: 0;
            margin-bottom: 10px;
            font-weight: 700;
        }
        .history-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .history-item {
            background-color: #F9FAFB;
            border: 1px solid #E5E7EB;
            border-left: 4px solid var(--color-accent);
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: transform 0.15s, box-shadow 0.15s;
        }
        .history-item:hover {
            box-shadow: 0 4px 10px rgba(15, 23, 42, 0.08);
            transform: translateY(-1px);
        }
        .history-details strong {
            color: var(--color-text);
            font-weight: 700;
            font-size: 14px;
        }
        .history-time {
            font-size: 11px;
            color: var(--color-muted);
            text-align: right;
        }

        footer.site {
            margin-top: auto; padding: 10px 20px; text-align: center; color: var(--color-muted);
            background-color: rgba(255, 255, 255, 0.9); border-top: 1px solid #E5E7EB;
            font-size: 11px;
        }
    </style>
</head>
<body>
    <div class="page">
        <header class="topbar">
            <div class="brand">
              Application for Assessment of Quality of Textbook / Reference Books / E-Books
              <span>Upload academic materials and obtain AI-assisted quality insights.</span>
            </div>
            <div class="header-actions">
                <button id="history-toggle-btn" class="action-btn" title="View Previous Evaluations">History</button>
                <a class="logout-btn" href="{{ url_for('logout') }}">Logout</a>
            </div>
        </header>
        <section class="hero" role="banner" aria-label="Textbook quality assessment intro">
            <h1>Evaluate Textbook & E-Book Quality in Minutes</h1>
            <p>
              Upload a PDF, Word document, text file, or a photo of a textbook page — either from your files
              or directly using your device camera — to receive an AI-assisted review.
            </p>
        </section>
        <main>
            <div class="content">
                <h1 class="title"><b>Textbook / Reference / E-Book Analyzer</b></h1>
                <p class="subtitle-main">
                    This tool provides a preliminary, automated assessment of educational material quality.
                    Use the output as a supporting opinion, not as a replacement for expert academic review.
                </p>

                <!-- FILE UPLOAD SECTION -->
                <div class="file-wrap">
                  <form id="uploadForm" action="/analyze" method="post" enctype="multipart/form-data">
                    <p class="hint">
                        Supported formats: <b>PDF (.pdf)</b>, <b>Word (.docx)</b>, <b>Text (.txt)</b>,
                        or <b>image files</b> of textbook pages.
                    </p>
                    <input type="file" id="fileInput" name="file"
                           accept=".pdf,.docx,.txt,image/*"
                           required><br>
                    <div class="actions">
                      <button class="btn" type="submit"><b>Analyze Uploaded File</b></button>
                    </div>
                  </form>
                </div>

                <div class="divider-or">or use integrated camera</div>

                <!-- CAMERA CAPTURE SECTION -->
                <div class="camera-wrap">
                    <div class="camera-title">Capture a Textbook Page with Camera</div>
                    <div class="camera-grid">
                        <div class="camera-preview-box">
                            <video id="cameraVideo" autoplay playsinline></video>
                            <canvas id="cameraCanvas" style="display:none;"></canvas>
                        </div>
                        <div class="camera-side">
                            <div class="hint">
                                1. Click <b>Start Camera</b> and allow browser permission.<br>
                                2. Hold the textbook page clearly in front of the camera.<br>
                                3. Click <b>Capture Photo</b> and then <b>Analyze Photo</b>.
                            </div>
                            <div class="actions">
                                <button id="cameraStartBtn" class="btn outline" type="button">Start Camera</button>
                                <button id="cameraCaptureBtn" class="btn secondary" type="button">Capture Photo</button>
                            </div>
                            <div class="actions">
                                <button id="cameraAnalyzeBtn" class="btn" type="button">Analyze Photo</button>
                            </div>
                            <div id="cameraStatus" class="camera-status">
                                Camera not started. Click "Start Camera" to begin.
                            </div>
                        </div>
                    </div>
                </div>

                <div id="result"></div>
            </div>
        </main>
        <footer class="site">
            <small>
                This is a <b>supporting tool</b> for educational quality review. Always combine AI feedback
                with judgment from qualified subject experts.
            </small>
        </footer>
    </div>

    <div id="history-panel" class="history-panel-overlay">
        <div class="history-panel-content">
            <button class="close-btn" onclick="toggleHistory(false)">×</button>
            <div class="history-section">
                <div class="history-title">Recent Evaluation History</div>
                <ul class="history-list">
                    <li class="history-item" style="justify-content: center; background-color: #FFF; color: var(--color-muted); border-left: none;">
                        Click "History" to load your full evaluation history.
                    </li>
                </ul>
            </div>
        </div>
    </div>

    <script>
    let historyLoadedOnce = false;

    function toggleHistory(show) {
        const panel = document.getElementById('history-panel');
        if (show === undefined) {
            panel.classList.toggle('active');
        } else if (show) {
            panel.classList.add('active');
        } else {
            panel.classList.remove('active');
        }
    }

    async function loadHistory() {
        const panel = document.getElementById('history-panel');
        const list = panel.querySelector('.history-list');

        toggleHistory(true);

        list.innerHTML = `
            <li class="history-item" style="justify-content: center; background-color: #FFF; color: var(--color-muted); border-left: none;">
                Loading history...
            </li>
        `;

        try {
            const res = await fetch('/user_history', {
                method: 'GET',
                credentials: 'same-origin'
            });

            if (!res.ok) {
                throw new Error('Server error: ' + res.status);
            }

            const data = await res.json();
            const history = data.history || [];

            if (!history.length) {
                list.innerHTML = `
                    <li class="history-item" style="justify-content: center; background-color: #FFF; color: var(--color-muted); border-left: none;">
                        No previous evaluation history found.
                    </li>
                `;
                return;
            }

            list.innerHTML = history.map(item => {
                const timestamp = item.timestamp || '';
                const datePart = timestamp.split(' ')[0] || '';
                const overall = item.overall_rating || 'No rating';
                const shortSummary = (item.summary || '').slice(0, 80);
                const summaryDisplay = shortSummary ? shortSummary + (item.summary.length > 80 ? '…' : '') : 'No summary stored.';
                return `
                    <li class="history-item">
                        <div class="history-details">
                            <strong>${overall}</strong>
                            <br>
                            <small>${summaryDisplay}</small>
                        </div>
                        <div class="history-time">${datePart}</div>
                    </li>
                `;
            }).join('');
        } catch (err) {
            console.error(err);
            list.innerHTML = `
                <li class="history-item" style="justify-content: center; background-color: #FFF; color: var(--color-warn); border-left: none;">
                    Error loading history: ${err.message}
                </li>
            `;
        }
    }

    const uploadForm = document.getElementById('uploadForm');
    const resultBox = document.getElementById('result');

    // Reusable function to call /analyze with any FormData (file upload OR camera)
    async function analyzeFormData(fd, submitBtn) {
      submitBtn.disabled = true;
      const oldLabel = submitBtn.textContent;
      submitBtn.textContent = 'Analyzing...';
      resultBox.classList.remove('show');
      resultBox.innerHTML = '';

      try {
        const res = await fetch('/analyze', {
          method: 'POST',
          body: fd,
          credentials: 'same-origin'
        });

        let data = null;
        try {
          data = await res.json();
        } catch (_) {
          data = null;
        }

        if (!res.ok) {
          const msg = (data && data.error) ? data.error : `Unexpected server error (${res.status})`;
          throw new Error(msg);
        }

        const now = new Date().toLocaleString();

        const accuracyScore = data.accuracy_score;
        const readabilityScore = data.readability_score;
        const consistencyScore = data.consistency_score;
        const overallScore = data.overall_rating;

        const accuracy = data.accuracy ?? 'Not available';
        const readability = data.readability ?? 'Not available';
        const consistency = data.consistency ?? 'Not available';
        const summary = data.summary ?? 'Not available';

        const overallDisplay = (overallScore !== undefined && overallScore !== null)
          ? `${overallScore}/5`
          : 'Not available';

        const formatPercent = (val) => {
          if (val === undefined || val === null || isNaN(val)) return 'N/A';
          return `${val}%`;
        };

        resultBox.innerHTML = `
          <div class="result-card">
            <div class="result-header">
              <div class="result-pill">
                <span class="result-pill-dot"></span>
                <span>AI Quality Evaluation</span>
              </div>
              <div class="result-time">${now}</div>
            </div>

            <h2 class="result-title">${overallDisplay}</h2>
            <p class="result-subtitle">
              This card summarizes the AI-assisted evaluation of the uploaded or captured textbook / reference / e-book content.
            </p>

            <div class="result-grid">
              <div class="result-section">
                <h3>Accuracy (${formatPercent(accuracyScore)})</h3>
                <p>${accuracy}</p>
              </div>
              <div class="result-section">
                <h3>Readability (${formatPercent(readabilityScore)})</h3>
                <p>${readability}</p>
              </div>
              <div class="result-section">
                <h3>Consistency (${formatPercent(consistencyScore)})</h3>
                <p>${consistency}</p>
              </div>
              <div class="result-section">
                <h3>Summary</h3>
                <p>${summary}</p>
              </div>
            </div>

            <p class="result-note">
              This is a <strong>preliminary AI-based assessment</strong>, not a formal academic review.
              Please combine these insights with evaluations by qualified subject-matter experts before making
              publishing or curriculum decisions.
            </p>
          </div>
        `;
        resultBox.classList.add('show');
        resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

      } catch (err) {
        resultBox.innerHTML = `<p style="color:#EF4444; font-size:13px;"><strong>Error:</strong> ${err.message}</p>`;
        resultBox.classList.add('show');
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = oldLabel;
      }
    }

    // Handle normal file upload form
    uploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const form = e.currentTarget;
      const fd = new FormData(form);
      const submitBtn = form.querySelector('button[type="submit"]');
      await analyzeFormData(fd, submitBtn);
    });

    // CAMERA LOGIC
    let cameraStream = null;
    let capturedImageBlob = null;

    const cameraVideo = document.getElementById('cameraVideo');
    const cameraCanvas = document.getElementById('cameraCanvas');
    const cameraStatus = document.getElementById('cameraStatus');
    const cameraStartBtn = document.getElementById('cameraStartBtn');
    const cameraCaptureBtn = document.getElementById('cameraCaptureBtn');
    const cameraAnalyzeBtn = document.getElementById('cameraAnalyzeBtn');

    cameraStartBtn.addEventListener('click', async () => {
      try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          cameraStatus.textContent = 'Camera not supported in this browser/device.';
          return;
        }
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
        cameraVideo.srcObject = cameraStream;
        cameraVideo.style.display = 'block';
        cameraCanvas.style.display = 'none';
        cameraStatus.textContent = 'Camera is on. Position the page and click "Capture Photo".';
      } catch (err) {
        console.error(err);
        cameraStatus.textContent = 'Unable to access camera: ' + err.message;
      }
    });

    cameraCaptureBtn.addEventListener('click', () => {
      if (!cameraStream) {
        cameraStatus.textContent = 'Start camera first.';
        return;
      }
      const trackSettings = cameraStream.getVideoTracks()[0].getSettings();
      const width = trackSettings.width || 640;
      const height = trackSettings.height || 480;

      cameraCanvas.width = width;
      cameraCanvas.height = height;
      const ctx = cameraCanvas.getContext('2d');
      ctx.drawImage(cameraVideo, 0, 0, width, height);

      cameraCanvas.toBlob((blob) => {
        if (!blob) {
          cameraStatus.textContent = 'Failed to capture image from camera.';
          return;
        }
        capturedImageBlob = blob;
        cameraVideo.style.display = 'none';
        cameraCanvas.style.display = 'block';
        cameraStatus.textContent = 'Photo captured. Click "Analyze Photo" to evaluate.';
      }, 'image/jpeg', 0.9);
    });

    cameraAnalyzeBtn.addEventListener('click', async () => {
      if (!capturedImageBlob) {
        cameraStatus.textContent = 'No captured photo found. Capture a photo first.';
        return;
      }
      const fd = new FormData();
      fd.append('file', capturedImageBlob, 'camera_capture.jpg');
      await analyzeFormData(fd, cameraAnalyzeBtn);
    });

    // Stop camera when leaving page (optional)
    window.addEventListener('beforeunload', () => {
      if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
      }
    });

    document.addEventListener('DOMContentLoaded', () => {
      document.getElementById('history-toggle-btn').addEventListener('click', () => {
        loadHistory();
      });
    });
    </script>
</body>
</html>
"""

# ========================= Flask Routes ==========================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = users_collection.find_one({"username": username})
        if user and check_password_hash(user.get("password_hash", ""), password):
            session["logged_in"] = True
            session["username"] = username
            session["user_id"] = str(user["_id"])
            return redirect(url_for("home"))

        return render_template_string(login_html, error="Invalid credentials")

    return render_template_string(login_html, message=request.args.get("message"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        existing = users_collection.find_one({"username": username})
        if existing:
            return render_template_string(
                register_html,
                error="Username already exists. Please choose a different one.",
            )

        password_hash = generate_password_hash(password)

        users_collection.insert_one({
            "username": username,
            "password_hash": password_hash,
            "created_at": datetime.utcnow(),
        })

        return redirect(
            url_for("login", message="Account created successfully! Please log in.")
        )

    return render_template_string(register_html)


@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    session.pop("username", None)
    session.pop("user_id", None)
    return redirect(url_for("login"))


@app.route("/")
def home():
    if "logged_in" not in session:
        return redirect(url_for("login"))

    return render_template_string(main_app_html)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "logged_in" not in session or "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file found in request"}), 400

    file = request.files["file"]

    if not file.filename:
        # if file comes from camera blob, filename might be present but safe-guard:
        filename = getattr(file, "filename", "") or "camera_capture.jpg"
    else:
        filename = file.filename

    content_type = file.content_type or ""

    try:
        # Read bytes once
        file_bytes = file.read()

        # Store file in GridFS
        file_id = files_fs.put(
            file_bytes,
            filename=filename,
            content_type=content_type,
            uploaded_at=datetime.utcnow(),
            user_id=session["user_id"],
        )

        # Decide if it's an image or document
        lower_name = filename.lower()
        is_image = (
            content_type.startswith("image/")
            or lower_name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"))
        )

        text_content = None
        image_bytes = None
        image_mime = None

        if is_image:
            image_bytes = file_bytes
            image_mime = content_type if content_type.startswith("image/") else "image/jpeg"
        else:
            # Extract text from pdf/docx/txt
            text_content = extract_text_from_file_bytes(filename, file_bytes)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to process file: {e}"}), 400

    evaluation_data = evaluate_textbook_gemini(
        text_content=text_content or "",
        image_bytes=image_bytes,
        image_mime=image_mime,
    )
    if "error" in evaluation_data:
        return jsonify(evaluation_data), 500

    # Overall rating (string form for display)
    overall_num = evaluation_data.get("overall_rating", None)
    if isinstance(overall_num, (int, float)):
        overall_str = f"{overall_num}/5"
    else:
        overall_str = str(overall_num) if overall_num is not None else "No rating"

    # --- Save FULL evaluation document in MongoDB ---
    input_type = "image" if image_bytes is not None else "document"
    try:
        eval_doc = {
            "user_id": ObjectId(session["user_id"]),
            "username": session.get("username"),
            "timestamp": datetime.utcnow(),

            "input_type": input_type,

            # File info
            "file_id": file_id,
            "original_filename": filename,
            "content_type": content_type,

            # Scores
            "accuracy_score": evaluation_data.get("accuracy_score"),
            "readability_score": evaluation_data.get("readability_score"),
            "consistency_score": evaluation_data.get("consistency_score"),

            # Full explanation texts from analyzer
            "accuracy_text": evaluation_data.get("accuracy", ""),
            "readability_text": evaluation_data.get("readability", ""),
            "consistency_text": evaluation_data.get("consistency", ""),

            # Overall rating (numeric + string)
            "overall_rating_value": evaluation_data.get("overall_rating"),
            "overall_rating": overall_str,

            # Summary text (full)
            "summary": evaluation_data.get("summary", ""),

            # Full extracted text from the uploaded document (if any)
            "full_extracted_text": text_content if text_content else None,

            # Raw JSON from Gemini
            "raw_evaluation": evaluation_data,
        }
        evaluations_collection.insert_one(eval_doc)
    except Exception as e:
        print(f"Error saving evaluation to MongoDB: {e}")

    # --- Response to frontend ---
    return jsonify(
        {
            "accuracy_score": evaluation_data.get("accuracy_score"),
            "accuracy": evaluation_data.get("accuracy", "AI analysis unavailable."),
            "readability_score": evaluation_data.get("readability_score"),
            "readability": evaluation_data.get(
                "readability", "AI analysis unavailable."
            ),
            "consistency_score": evaluation_data.get("consistency_score"),
            "consistency": evaluation_data.get(
                "consistency", "AI analysis unavailable."
            ),
            "overall_rating": evaluation_data.get(
                "overall_rating", "AI analysis unavailable."
            ),
            "summary": evaluation_data.get("summary", "AI analysis unavailable."),
        }
    )


@app.route("/user_history", methods=["GET"])
def user_history():
    if "logged_in" not in session or "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        user_id = ObjectId(session["user_id"])
    except Exception:
        return jsonify({"history": []})

    cursor = (
        evaluations_collection
        .find({"user_id": user_id})
        .sort("timestamp", -1)
        .limit(50)
    )

    history = []
    for doc in cursor:
        ts = doc.get("timestamp")
        if isinstance(ts, datetime):
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_str = str(ts) if ts else ""
        history.append(
            {
                "timestamp": ts_str,
                "overall_rating": doc.get("overall_rating", "No rating"),
                "summary": doc.get("summary", ""),
                "original_filename": doc.get("original_filename", ""),
            }
        )

    return jsonify({"history": history})


if __name__ == "__main__":
    app.run(debug=True)
