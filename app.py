# ═══════════════════════════════════════════════════════════════════
#  SpamShield AI — Intelligent Spam Email Detection System
#  Flask Web Application (Single File)
#  Run: python app.py
#  Then open: http://localhost:5000
# ═══════════════════════════════════════════════════════════════════

import os
import re
import pickle
import nltk
from flask import Flask, request, jsonify, render_template_string

# Download required NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.corpus    import stopwords
from nltk.stem      import PorterStemmer
from nltk.tokenize  import word_tokenize

# ─────────────────────────────────────────────
#  Initialize Flask App
# ─────────────────────────────────────────────
app = Flask(__name__)

# ─────────────────────────────────────────────
#  Load Trained Model and Vectorizer
# ─────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "spamshield_model.pkl")
VECTORIZER_PATH= os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

model      = None
vectorizer = None

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    print("✅ Model and vectorizer loaded successfully!")
else:
    print("⚠️  Model files not found. Please run the Jupyter Notebook first to train and save the model.")

# ─────────────────────────────────────────────
#  Text Cleaning Pipeline (mirrors the notebook)
# ─────────────────────────────────────────────
stemmer        = PorterStemmer()
stop_words_set = set(stopwords.words("english"))

def clean_email_text(raw_text):
    """
    Cleans raw email text:
    1. Lowercase
    2. Remove non-alphabetic characters
    3. Tokenize
    4. Remove stopwords
    5. Stem each word
    """
    text    = raw_text.lower()
    text    = re.sub(r"[^a-z\s]", "", text)
    tokens  = word_tokenize(text)
    cleaned = [
        stemmer.stem(w)
        for w in tokens
        if w not in stop_words_set and len(w) > 2
    ]
    return " ".join(cleaned)


# ═══════════════════════════════════════════════════════════════════
#  HTML Template — Full Page (Embedded in Python)
# ═══════════════════════════════════════════════════════════════════
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SpamShield AI – Spam Email Detector</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        /* ─────────────────────────────────────────
           CSS Variables & Reset
        ───────────────────────────────────────── */
        :root {
            --blue-50 : #EFF6FF;
            --blue-100: #DBEAFE;
            --blue-200: #BFDBFE;
            --blue-300: #93C5FD;
            --blue-400: #60A5FA;
            --blue-500: #3B82F6;
            --blue-600: #2563EB;
            --blue-700: #1D4ED8;
            --blue-800: #1E40AF;
            --blue-900: #1E3A8A;

            --slate-50 : #F8FAFC;
            --slate-100: #F1F5F9;
            --slate-200: #E2E8F0;
            --slate-300: #CBD5E1;
            --slate-500: #64748B;
            --slate-600: #475569;
            --slate-700: #334155;
            --slate-900: #0F172A;

            --spam-red   : #EF4444;
            --spam-light : #FEF2F2;
            --spam-border: #FECACA;

            --ham-green  : #10B981;
            --ham-light  : #ECFDF5;
            --ham-border : #A7F3D0;

            --shadow-sm  : 0 1px 2px rgba(0,0,0,0.05);
            --shadow-md  : 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -1px rgba(0,0,0,0.04);
            --shadow-lg  : 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -2px rgba(0,0,0,0.04);
            --shadow-xl  : 0 20px 25px -5px rgba(0,0,0,0.08), 0 10px 10px -5px rgba(0,0,0,0.03);

            --radius-sm  : 8px;
            --radius-md  : 12px;
            --radius-lg  : 16px;
            --radius-xl  : 24px;
        }

        *, *::before, *::after {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        html { scroll-behavior: smooth; }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: var(--blue-50);
            color: var(--slate-700);
            min-height: 100vh;
            line-height: 1.6;
            position: relative;
            overflow-x: hidden;
        }

        /* ─────────────────────────────────────────
           Background Decoration
        ───────────────────────────────────────── */
        body::before {
            content: '';
            position: fixed;
            top: -200px;
            right: -200px;
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
        }

        body::after {
            content: '';
            position: fixed;
            bottom: -200px;
            left: -200px;
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(99,179,237,0.10) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
        }

        /* ─────────────────────────────────────────
           Layout
        ───────────────────────────────────────── */
        .page-wrapper {
            position: relative;
            z-index: 1;
            max-width: 860px;
            margin: 0 auto;
            padding: 40px 20px 60px;
        }

        /* ─────────────────────────────────────────
           Header / Banner
        ───────────────────────────────────────── */
        .site-header {
            text-align: center;
            margin-bottom: 40px;
            animation: slideDown 0.6s ease both;
        }

        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-24px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        .shield-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 72px;
            height: 72px;
            background: linear-gradient(135deg, var(--blue-500), var(--blue-700));
            border-radius: var(--radius-xl);
            margin-bottom: 20px;
            box-shadow: 0 8px 24px rgba(37,99,235,0.30);
            font-size: 36px;
            animation: pulse-glow 3s ease-in-out infinite;
        }

        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 8px 24px rgba(37,99,235,0.30); }
            50%       { box-shadow: 0 8px 36px rgba(37,99,235,0.50); }
        }

        .site-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--blue-700) 0%, var(--blue-500) 60%, #38BDF8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.5px;
            line-height: 1.2;
            margin-bottom: 10px;
        }

        .site-subtitle {
            font-size: 1rem;
            color: var(--slate-500);
            font-weight: 400;
            max-width: 480px;
            margin: 0 auto;
        }

        .badge-row {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 16px;
            flex-wrap: wrap;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.73rem;
            font-weight: 600;
            letter-spacing: 0.3px;
        }

        .badge-blue   { background: var(--blue-100); color: var(--blue-700); }
        .badge-purple { background: #EDE9FE; color: #5B21B6; }
        .badge-green  { background: var(--ham-light); color: #065F46; }

        /* ─────────────────────────────────────────
           Cards
        ───────────────────────────────────────── */
        .card {
            background: #ffffff;
            border-radius: var(--radius-lg);
            padding: 32px;
            box-shadow: var(--shadow-lg);
            border: 1px solid rgba(219,234,254,0.6);
            animation: fadeUp 0.5s ease both;
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(20px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        .card + .card {
            margin-top: 24px;
        }

        .card-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--slate-700);
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .card-title span.icon {
            font-size: 1.2rem;
        }

        /* ─────────────────────────────────────────
           Main Input Card
        ───────────────────────────────────────── */
        .input-card { animation-delay: 0.1s; }

        .textarea-wrapper {
            position: relative;
        }

        textarea {
            width: 100%;
            min-height: 180px;
            padding: 16px 18px;
            border: 2px solid var(--blue-200);
            border-radius: var(--radius-md);
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 0.95rem;
            color: var(--slate-700);
            background: var(--blue-50);
            resize: vertical;
            transition: border-color 0.2s, box-shadow 0.2s, background 0.2s;
            line-height: 1.7;
            outline: none;
        }

        textarea:focus {
            border-color: var(--blue-500);
            background: #ffffff;
            box-shadow: 0 0 0 3px rgba(59,130,246,0.12);
        }

        textarea::placeholder {
            color: var(--slate-300);
            font-size: 0.93rem;
        }

        .char-counter {
            text-align: right;
            font-size: 0.78rem;
            color: var(--slate-300);
            margin-top: 6px;
            font-variant-numeric: tabular-nums;
        }

        /* ─────────────────────────────────────────
           Buttons
        ───────────────────────────────────────── */
        .btn-row {
            display: flex;
            gap: 12px;
            margin-top: 20px;
            flex-wrap: wrap;
        }

        .btn-primary {
            flex: 1;
            min-width: 160px;
            padding: 14px 28px;
            background: linear-gradient(135deg, var(--blue-600), var(--blue-500));
            color: #ffffff;
            border: none;
            border-radius: var(--radius-md);
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 0.97rem;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s;
            box-shadow: 0 4px 12px rgba(37,99,235,0.35);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            letter-spacing: 0.2px;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(37,99,235,0.40);
        }

        .btn-primary:active {
            transform: translateY(0);
            box-shadow: 0 2px 6px rgba(37,99,235,0.30);
        }

        .btn-primary:disabled {
            opacity: 0.65;
            cursor: not-allowed;
            transform: none;
        }

        .btn-secondary {
            padding: 14px 22px;
            background: var(--slate-100);
            color: var(--slate-600);
            border: 1.5px solid var(--slate-200);
            border-radius: var(--radius-md);
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 0.92rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s, border-color 0.15s, transform 0.15s;
        }

        .btn-secondary:hover {
            background: var(--slate-200);
            border-color: var(--slate-300);
            transform: translateY(-1px);
        }

        /* ─────────────────────────────────────────
           Result Card
        ───────────────────────────────────────── */
        .result-card {
            display: none;
            animation-delay: 0.05s;
        }

        .result-card.show-spam {
            display: block;
            border-color: var(--spam-border);
            background: linear-gradient(135deg, #ffffff 60%, var(--spam-light));
        }

        .result-card.show-ham {
            display: block;
            border-color: var(--ham-border);
            background: linear-gradient(135deg, #ffffff 60%, var(--ham-light));
        }

        .result-header {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 20px;
        }

        .result-icon {
            width: 56px;
            height: 56px;
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 26px;
            flex-shrink: 0;
        }

        .result-icon.spam { background: var(--spam-light); }
        .result-icon.ham  { background: var(--ham-light);  }

        .result-label {
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: -0.3px;
        }

        .result-label.spam { color: var(--spam-red); }
        .result-label.ham  { color: var(--ham-green); }

        .result-sublabel {
            font-size: 0.88rem;
            color: var(--slate-500);
            font-weight: 500;
            margin-top: 2px;
        }

        /* ─────────────────────────────────────────
           Confidence Bar
        ───────────────────────────────────────── */
        .confidence-section {
            margin-top: 8px;
        }

        .confidence-label {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--slate-600);
            margin-bottom: 8px;
        }

        .confidence-pct {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1rem;
            font-weight: 700;
        }

        .confidence-bar-track {
            height: 10px;
            background: var(--slate-100);
            border-radius: 999px;
            overflow: hidden;
        }

        .confidence-bar-fill {
            height: 100%;
            border-radius: 999px;
            width: 0%;
            transition: width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
        }

        .confidence-bar-fill.spam { background: linear-gradient(90deg, #FCA5A5, var(--spam-red)); }
        .confidence-bar-fill.ham  { background: linear-gradient(90deg, #6EE7B7, var(--ham-green)); }

        .confidence-note {
            font-size: 0.78rem;
            color: var(--slate-400);
            margin-top: 8px;
            font-style: italic;
        }

        /* ─────────────────────────────────────────
           Sample Emails Panel
        ───────────────────────────────────────── */
        .samples-card { animation-delay: 0.2s; }

        .samples-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }

        @media (max-width: 600px) {
            .samples-grid { grid-template-columns: 1fr; }
        }

        .sample-item {
            padding: 14px 16px;
            border-radius: var(--radius-sm);
            cursor: pointer;
            font-size: 0.84rem;
            line-height: 1.55;
            color: var(--slate-600);
            transition: transform 0.15s, box-shadow 0.15s;
            border: 1.5px solid transparent;
            position: relative;
        }

        .sample-item:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }

        .sample-item.spam-sample {
            background: var(--spam-light);
            border-color: var(--spam-border);
        }

        .sample-item.spam-sample:hover { border-color: var(--spam-red); }

        .sample-item.ham-sample {
            background: var(--ham-light);
            border-color: var(--ham-border);
        }

        .sample-item.ham-sample:hover { border-color: var(--ham-green); }

        .sample-tag {
            display: inline-block;
            font-size: 0.68rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 999px;
            margin-bottom: 6px;
            letter-spacing: 0.5px;
        }

        .sample-tag.spam { background: #FECACA; color: #991B1B; }
        .sample-tag.ham  { background: #A7F3D0; color: #065F46; }

        .sample-click-hint {
            font-size: 0.73rem;
            color: var(--slate-400);
            margin-top: 6px;
            font-style: italic;
        }

        /* ─────────────────────────────────────────
           Pipeline Section
        ───────────────────────────────────────── */
        .pipeline-card { animation-delay: 0.3s; }

        .pipeline-steps {
            display: flex;
            align-items: center;
            gap: 0;
            flex-wrap: wrap;
            row-gap: 12px;
        }

        .pipeline-step {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            flex: 1;
            min-width: 90px;
        }

        .step-circle {
            width: 46px;
            height: 46px;
            border-radius: 50%;
            background: var(--blue-100);
            border: 2px solid var(--blue-200);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            margin-bottom: 6px;
            transition: transform 0.2s;
        }

        .step-circle:hover { transform: scale(1.1); }

        .step-name {
            font-size: 0.73rem;
            font-weight: 600;
            color: var(--slate-600);
            line-height: 1.3;
        }

        .pipeline-arrow {
            font-size: 1.2rem;
            color: var(--blue-300);
            padding: 0 4px;
            flex-shrink: 0;
            margin-bottom: 24px;
        }

        /* ─────────────────────────────────────────
           Loading Spinner
        ───────────────────────────────────────── */
        .spinner {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 2.5px solid rgba(255,255,255,0.4);
            border-top-color: #fff;
            border-radius: 50%;
            animation: spin 0.7s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* ─────────────────────────────────────────
           Info Banner (Model Not Found)
        ───────────────────────────────────────── */
        .info-banner {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            padding: 16px 20px;
            background: #FFFBEB;
            border: 1.5px solid #FDE68A;
            border-radius: var(--radius-md);
            margin-bottom: 24px;
            font-size: 0.88rem;
            color: #92400E;
            line-height: 1.6;
        }

        .info-banner .icon { font-size: 1.2rem; flex-shrink: 0; }

        .info-banner code {
            background: rgba(0,0,0,0.07);
            padding: 1px 6px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }

        /* ─────────────────────────────────────────
           Stats Row
        ───────────────────────────────────────── */
        .stats-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin-bottom: 24px;
            animation: fadeUp 0.5s ease both;
            animation-delay: 0.05s;
        }

        .stat-box {
            background: #ffffff;
            border: 1px solid var(--blue-100);
            border-radius: var(--radius-md);
            padding: 18px 16px;
            text-align: center;
            box-shadow: var(--shadow-sm);
        }

        .stat-number {
            font-size: 1.6rem;
            font-weight: 800;
            color: var(--blue-600);
            font-variant-numeric: tabular-nums;
            line-height: 1;
        }

        .stat-label {
            font-size: 0.78rem;
            color: var(--slate-500);
            margin-top: 5px;
            font-weight: 500;
        }

        /* ─────────────────────────────────────────
           Footer
        ───────────────────────────────────────── */
        .site-footer {
            text-align: center;
            padding: 32px 20px 16px;
            color: var(--slate-400);
            font-size: 0.82rem;
            animation: fadeUp 0.5s ease both;
            animation-delay: 0.4s;
        }

        .footer-brand {
            font-weight: 700;
            color: var(--blue-500);
        }

        .footer-divider {
            width: 48px;
            height: 2px;
            background: var(--blue-200);
            margin: 12px auto;
            border-radius: 999px;
        }

        /* ─────────────────────────────────────────
           Responsive
        ───────────────────────────────────────── */
        @media (max-width: 520px) {
            .site-title { font-size: 1.7rem; }
            .stats-row  { grid-template-columns: 1fr 1fr; }
            .stats-row .stat-box:last-child { grid-column: span 2; }
            .card { padding: 22px 18px; }
        }
    </style>
</head>

<body>
<div class="page-wrapper">

    <!-- ═══ HEADER ═══ -->
    <header class="site-header">
        <div class="shield-icon">🛡️</div>
        <h1 class="site-title">SpamShield AI</h1>
        <p class="site-subtitle">Intelligent Spam Email Detection using Gradient Boosting Classifier</p>
        <div class="badge-row">
            <span class="badge badge-blue">⚡ Gradient Boosting</span>
            <span class="badge badge-purple">🔤 TF-IDF Vectorizer</span>
            <span class="badge badge-green">✅ ~97% Accuracy</span>
        </div>
    </header>

    <!-- ═══ WARNING: Model Not Loaded ═══ -->
    {% if not model_loaded %}
    <div class="info-banner">
        <span class="icon">⚠️</span>
        <div>
            <strong>Model not found.</strong> Run the Jupyter Notebook first to train and save the model.
            The files <code>spamshield_model.pkl</code> and <code>tfidf_vectorizer.pkl</code>
            must be in the same directory as <code>app.py</code>.
        </div>
    </div>
    {% endif %}

    <!-- ═══ STATS ROW ═══ -->
    <div class="stats-row">
        <div class="stat-box">
            <div class="stat-number">5,574</div>
            <div class="stat-label">Training Emails</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">97%+</div>
            <div class="stat-label">Model Accuracy</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">5,000</div>
            <div class="stat-label">TF-IDF Features</div>
        </div>
    </div>

    <!-- ═══ MAIN INPUT CARD ═══ -->
    <div class="card input-card">
        <div class="card-title">
            <span class="icon">📧</span>
            Analyze Email
        </div>

        <div class="textarea-wrapper">
            <textarea
                id="emailInput"
                placeholder="Paste your email content here...&#10;&#10;Example: 'Congratulations! You've won a FREE prize. Click here to claim your $1000 cash reward now!'"
                oninput="updateCounter()"
            ></textarea>
        </div>
        <div class="char-counter"><span id="charCount">0</span> characters</div>

        <div class="btn-row">
            <button class="btn-primary" id="predictBtn" onclick="analyzeEmail()" {% if not model_loaded %}disabled{% endif %}>
                <span id="btnIcon">🔍</span>
                <span id="btnText">Analyze Email</span>
            </button>
            <button class="btn-secondary" onclick="clearAll()">🗑 Clear</button>
        </div>
    </div>

    <!-- ═══ RESULT CARD ═══ -->
    <div class="card result-card" id="resultCard">
        <div class="result-header">
            <div class="result-icon" id="resultIcon"></div>
            <div>
                <div class="result-label" id="resultLabel"></div>
                <div class="result-sublabel" id="resultSublabel"></div>
            </div>
        </div>

        <div class="confidence-section">
            <div class="confidence-label">
                <span>Model Confidence</span>
                <span class="confidence-pct" id="confidencePct">--%</span>
            </div>
            <div class="confidence-bar-track">
                <div class="confidence-bar-fill" id="confidenceBar"></div>
            </div>
            <div class="confidence-note" id="confidenceNote"></div>
        </div>
    </div>

    <!-- ═══ SAMPLE EMAILS CARD ═══ -->
    <div class="card samples-card">
        <div class="card-title">
            <span class="icon">📋</span>
            Try Sample Emails
        </div>

        <div class="samples-grid">
            <!-- Spam Samples -->
            <div class="sample-item spam-sample" onclick="useSample(this)">
                <span class="sample-tag spam">SPAM</span>
                <div>WINNER!! Claim your FREE prize now! You've been selected for a $1000 cash reward. Click the link to verify and collect immediately!</div>
                <div class="sample-click-hint">👆 Click to use this sample</div>
            </div>

            <div class="sample-item spam-sample" onclick="useSample(this)">
                <span class="sample-tag spam">SPAM</span>
                <div>URGENT: Your bank account has been suspended. Verify your details NOW to restore access. Failure to act in 24 hours will result in permanent closure.</div>
                <div class="sample-click-hint">👆 Click to use this sample</div>
            </div>

            <!-- Ham Samples -->
            <div class="sample-item ham-sample" onclick="useSample(this)">
                <span class="sample-tag ham">HAM</span>
                <div>Hi, just checking in about the meeting tomorrow at 10am. Could you send me the agenda beforehand? Looking forward to catching up.</div>
                <div class="sample-click-hint">👆 Click to use this sample</div>
            </div>

            <div class="sample-item ham-sample" onclick="useSample(this)">
                <span class="sample-tag ham">HAM</span>
                <div>Please find the Q3 report attached for your review. Let me know if you'd like to schedule a call to discuss the findings. Thanks!</div>
                <div class="sample-click-hint">👆 Click to use this sample</div>
            </div>
        </div>
    </div>

    <!-- ═══ PREPROCESSING PIPELINE CARD ═══ -->
    <div class="card pipeline-card">
        <div class="card-title">
            <span class="icon">⚙️</span>
            Preprocessing Pipeline
        </div>

        <div class="pipeline-steps">
            <div class="pipeline-step">
                <div class="step-circle">📝</div>
                <div class="step-name">Raw Email Text</div>
            </div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">
                <div class="step-circle">🔡</div>
                <div class="step-name">Lowercase</div>
            </div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">
                <div class="step-circle">🧹</div>
                <div class="step-name">Remove Noise</div>
            </div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">
                <div class="step-circle">✂️</div>
                <div class="step-name">Tokenize</div>
            </div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">
                <div class="step-circle">🚫</div>
                <div class="step-name">Stop Words</div>
            </div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">
                <div class="step-circle">🌱</div>
                <div class="step-name">Stemming</div>
            </div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">
                <div class="step-circle">📊</div>
                <div class="step-name">TF-IDF</div>
            </div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">
                <div class="step-circle">🤖</div>
                <div class="step-name">GB Model</div>
            </div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">
                <div class="step-circle">🏷️</div>
                <div class="step-name">Prediction</div>
            </div>
        </div>
    </div>

    <!-- ═══ FOOTER ═══ -->
    <footer class="site-footer">
        <div class="footer-divider"></div>
        <p>
            <span class="footer-brand">SpamShield AI</span> &nbsp;·&nbsp;
            Built with Python, Scikit-learn &amp; Flask &nbsp;·&nbsp;
            Gradient Boosting Classifier
        </p>
        <p style="margin-top:6px; font-size:0.76rem; color: #94a3b8;">
            Machine Learning Project &nbsp;·&nbsp; NLP &amp; Text Classification
        </p>
    </footer>

</div><!-- end page-wrapper -->

<script>
    // ─────────────────────────────────────────
    //  Character Counter
    // ─────────────────────────────────────────
    function updateCounter() {
        const len = document.getElementById('emailInput').value.length;
        document.getElementById('charCount').textContent = len.toLocaleString();
    }

    // ─────────────────────────────────────────
    //  Use Sample Email
    // ─────────────────────────────────────────
    function useSample(el) {
        // Get only the text content (not the tag or hint)
        const children = el.querySelectorAll('div');
        const emailText = children[0].textContent.trim();
        document.getElementById('emailInput').value = emailText;
        updateCounter();

        // Scroll to input
        document.getElementById('emailInput').focus();
        document.getElementById('emailInput').scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    // ─────────────────────────────────────────
    //  Clear Everything
    // ─────────────────────────────────────────
    function clearAll() {
        document.getElementById('emailInput').value = '';
        updateCounter();

        const resultCard = document.getElementById('resultCard');
        resultCard.className = 'card result-card';
        resultCard.style.display = 'none';
    }

    // ─────────────────────────────────────────
    //  Main: Analyze Email
    // ─────────────────────────────────────────
    async function analyzeEmail() {
        const emailText = document.getElementById('emailInput').value.trim();

        if (!emailText) {
            alert('Please enter some email text to analyze.');
            return;
        }

        // Show loading state
        const btn      = document.getElementById('predictBtn');
        const btnIcon  = document.getElementById('btnIcon');
        const btnText  = document.getElementById('btnText');

        btn.disabled = true;
        btnIcon.innerHTML = '<span class="spinner"></span>';
        btnText.textContent = 'Analyzing...';

        try {
            // Send POST request to Flask backend
            const response = await fetch('/predict', {
                method : 'POST',
                headers: { 'Content-Type': 'application/json' },
                body   : JSON.stringify({ email: emailText })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Prediction failed.');
            }

            const data = await response.json();
            showResult(data);

        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            // Restore button
            btn.disabled = false;
            btnIcon.textContent = '🔍';
            btnText.textContent = 'Analyze Email';
        }
    }

    // ─────────────────────────────────────────
    //  Display Result
    // ─────────────────────────────────────────
    function showResult(data) {
        const isSpam     = data.is_spam;
        const confidence = data.confidence;

        const resultCard    = document.getElementById('resultCard');
        const resultIcon    = document.getElementById('resultIcon');
        const resultLabel   = document.getElementById('resultLabel');
        const resultSublabel= document.getElementById('resultSublabel');
        const confPct       = document.getElementById('confidencePct');
        const confBar       = document.getElementById('confidenceBar');
        const confNote      = document.getElementById('confidenceNote');

        // Set classes
        resultCard.className = 'card result-card ' + (isSpam ? 'show-spam' : 'show-ham');
        resultCard.style.display = 'block';

        // Icon
        resultIcon.className = 'result-icon ' + (isSpam ? 'spam' : 'ham');
        resultIcon.textContent = isSpam ? '🚨' : '✅';

        // Label
        resultLabel.className = 'result-label ' + (isSpam ? 'spam' : 'ham');
        resultLabel.textContent = isSpam ? 'SPAM DETECTED' : 'HAM — Legitimate';

        // Sub-label
        resultSublabel.textContent = isSpam
            ? 'This email shows strong spam indicators. Be cautious!'
            : 'This email appears to be legitimate and safe.';

        // Confidence
        confPct.textContent = confidence.toFixed(1) + '%';

        confBar.className = 'confidence-bar-fill ' + (isSpam ? 'spam' : 'ham');

        // Animate bar (with slight delay for CSS transition)
        setTimeout(() => {
            confBar.style.width = confidence + '%';
        }, 80);

        // Confidence note
        let noteText = '';
        if (confidence >= 95)      noteText = '🔒 Very high confidence — model is very sure.';
        else if (confidence >= 85) noteText = '✅ High confidence — reliable prediction.';
        else if (confidence >= 70) noteText = '⚠️ Moderate confidence — review the email manually.';
        else                       noteText = '❓ Low confidence — the email content is ambiguous.';

        confNote.textContent = noteText;

        // Smooth scroll to result
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    // ─────────────────────────────────────────
    //  Keyboard Shortcut: Ctrl+Enter = Analyze
    // ─────────────────────────────────────────
    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            analyzeEmail();
        }
    });
</script>

</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════
#  Flask Routes
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    """Serve the main SpamShield AI web page."""
    return render_template_string(HTML_TEMPLATE, model_loaded=(model is not None))


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive email text via JSON POST request.
    Returns prediction label, confidence score, and is_spam flag.
    """
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded. Please run the Jupyter Notebook first."}), 503

    # Parse incoming JSON data
    data       = request.get_json(force=True)
    email_text = data.get("email", "").strip()

    if not email_text:
        return jsonify({"error": "No email text provided. Please enter some text."}), 400

    # Preprocess → Vectorize → Predict
    cleaned    = clean_email_text(email_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    proba      = model.predict_proba(vectorized)[0]

    # Confidence for the predicted class
    confidence = round(float(proba[prediction]) * 100, 2)
    label      = "SPAM" if prediction == 1 else "HAM"

    return jsonify({
        "label"     : label,
        "confidence": confidence,
        "is_spam"   : bool(prediction == 1),
        "ham_prob"  : round(float(proba[0]) * 100, 2),
        "spam_prob" : round(float(proba[1]) * 100, 2)
    })


@app.route("/health")
def health():
    """Simple health check endpoint."""
    return jsonify({
        "status"      : "ok",
        "model_loaded": model is not None,
        "app"         : "SpamShield AI"
    })


# ═══════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  🛡️  SpamShield AI — Spam Email Detection System")
    print("=" * 55)
    print("  🌐 Open in browser: http://localhost:5000")
    print("  🔍 API endpoint  : POST /predict")
    print("  💚 Health check  : GET  /health")
    print("=" * 55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
