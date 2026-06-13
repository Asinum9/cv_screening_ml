"""Streamlit app for ML-based CV screening and candidate ranking.

Run from the project folder:
    streamlit run app.py
"""

from pathlib import Path
import base64

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from utils.openai_explanation import generate_candidate_explanation
from utils.preprocessing import clean_text
from utils.scoring import calculate_final_score, calculate_skill_score
from utils.skills import match_skills
from utils.text_extraction import extract_text_from_uploaded_file


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "resume_classifier.pkl"
VECTORIZER_FILE = BASE_DIR / "models" / "tfidf_vectorizer.pkl"
HERO_IMAGE_FILE = BASE_DIR / "assets" / "hero_cv_screening.png"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_UPLOAD_FILES = 10
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
TARGET_ROLE_OPTIONS = [
    "Frontend Developer",
    "Backend Developer",
    "AI Engineer",
    "Machine Learning Engineer",
    "QA Engineer",
    "Flutter Developer",
]


st.set_page_config(
    page_title="AI-Powered CV Screening System",
    layout="wide",
)


def apply_custom_styles() -> None:
    """Add visual styling only. This does not change model or ranking logic."""
    st.markdown(
        """
        <style>
        :root {
            --primary: #2563eb;
            --primary-dark: #1e40af;
            --ink: #172033;
            --muted: #64748b;
            --surface: #ffffff;
            --soft: #eef5ff;
            --line: #dbe6f3;
            --success: #0f766e;
            --warning: #b45309;
            --danger: #b91c1c;
            color-scheme: light;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 32rem),
                linear-gradient(180deg, #f8fbff 0%, #eef3f8 100%);
            color: var(--ink);
        }

        .block-container {
            padding-top: .55rem;
            padding-bottom: 3rem;
            max-width: 1560px;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
        }

        header[data-testid="stHeader"] {
            display: none;
        }

        [data-testid="stToolbar"] {
            display: none;
        }

        #MainMenu {
            visibility: hidden;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            background: linear-gradient(135deg, #003b7a 0%, #0057b8 48%, #003b7a 100%);
            color: #ffffff;
            border-radius: 12px 12px 0 0;
            padding: 1rem 1.25rem;
            box-shadow: 0 16px 35px rgba(0, 59, 122, 0.22);
            margin-bottom: 0;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: .7rem;
        }

        .brand-icon {
            width: 38px;
            height: 38px;
            border-radius: 9px;
            background: #ffffff;
            color: #0057b8;
            display: grid;
            place-items: center;
            font-weight: 900;
            box-shadow: inset 0 0 0 2px rgba(0, 87, 184, .12);
        }

        .brand-title {
            font-size: 1.12rem;
            font-weight: 850;
            line-height: 1.05;
        }

        .brand-subtitle {
            font-size: .78rem;
            opacity: .88;
            margin-top: .18rem;
        }

        .nav-pills {
            display: flex;
            align-items: center;
            gap: .45rem;
            flex-wrap: wrap;
        }

        .nav-pills a {
            color: #ffffff;
            text-decoration: none;
            font-weight: 700;
            font-size: .9rem;
            padding: .55rem .85rem;
            border-radius: 9px;
            background: rgba(255, 255, 255, .08);
            border: 1px solid rgba(255, 255, 255, .12);
        }

        .nav-pills a.active,
        .nav-pills a:hover {
            background: rgba(255, 255, 255, .18);
        }

        label,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p {
            color: var(--ink) !important;
            font-weight: 700 !important;
        }

        .stTextArea textarea {
            background: #ffffff !important;
            color: var(--ink) !important;
            border: 1px solid var(--line) !important;
            border-radius: 12px !important;
            caret-color: #1d4ed8 !important;
            outline: none !important;
            color-scheme: light !important;
        }

        .stTextArea textarea:focus {
            border: 2px solid var(--primary) !important;
            box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.14) !important;
        }

        .stTextArea textarea::placeholder {
            color: #64748b !important;
            opacity: 1 !important;
        }

        .stTextArea textarea::selection {
            background: #bfdbfe !important;
            color: var(--ink) !important;
        }

        div[data-baseweb="select"] > div {
            background: #ffffff !important;
            border: 1px solid var(--line) !important;
            color: var(--ink) !important;
            border-radius: 12px !important;
            color-scheme: light !important;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] svg {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            fill: var(--ink) !important;
        }

        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        div[role="listbox"],
        ul[role="listbox"] {
            background: #ffffff !important;
            color: var(--ink) !important;
            border: 1px solid var(--line) !important;
            box-shadow: 0 18px 36px rgba(15, 23, 42, 0.14) !important;
            color-scheme: light !important;
        }

        ul[role="listbox"] li,
        div[role="listbox"] *,
        div[data-baseweb="popover"] *,
        div[data-baseweb="menu"] *,
        div[role="option"] {
            background: #ffffff !important;
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }

        ul[role="listbox"] li:hover,
        div[role="listbox"] *:hover,
        div[role="option"]:hover,
        div[aria-selected="true"] {
            background: #eaf2ff !important;
            color: var(--primary-dark) !important;
            -webkit-text-fill-color: var(--primary-dark) !important;
        }

        div[role="option"] * {
            color: inherit !important;
        }

        div[role="option"][aria-selected="true"],
        div[role="option"][aria-selected="true"] * {
            background: #dbeafe !important;
            color: var(--primary-dark) !important;
            -webkit-text-fill-color: var(--primary-dark) !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: #ffffff !important;
            border: 1px dashed #93c5fd !important;
            border-radius: 12px !important;
        }

        [data-testid="stFileUploaderDropzone"] * {
            color: var(--ink) !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            border-color: #bfdbfe !important;
            color: var(--primary-dark) !important;
            background: #eff6ff !important;
        }

        [data-testid="stFileUploaderDropzone"] small {
            display: none !important;
        }

        [data-testid="stFileUploader"] {
            color: var(--ink) !important;
            color-scheme: light !important;
        }

        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploader"] ul,
        [data-testid="stFileUploader"] li,
        [data-testid="stFileUploader"] div,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] small {
            color: var(--ink) !important;
        }

        [data-testid="stFileUploaderFile"] {
            background: #ffffff !important;
            color: var(--ink) !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 10px !important;
            margin-top: .35rem !important;
            padding: .35rem .5rem !important;
            opacity: 1 !important;
        }

        [data-testid="stFileUploaderFile"] * {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            opacity: 1 !important;
        }

        [data-testid="stFileUploaderFile"] svg {
            display: none !important;
        }

        [data-testid="stFileUploaderFileName"],
        [data-testid="stFileUploaderFileSize"] {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            opacity: 1 !important;
        }

        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] p,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] small,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] div,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] [class*="file"],
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] [class*="File"],
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] [title] {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            text-shadow: none !important;
            opacity: 1 !important;
        }

        [data-testid="stFileUploaderDeleteBtn"] svg {
            display: none !important;
        }

        [data-testid="stFileUploaderDeleteBtn"] {
            width: 32px !important;
            height: 32px !important;
            border-radius: 8px !important;
            display: grid !important;
            place-items: center !important;
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            font-size: 0 !important;
        }

        [data-testid="stFileUploaderDeleteBtn"]::before {
            content: "×";
            color: var(--ink);
            -webkit-text-fill-color: var(--ink);
            font-size: 1.55rem;
            line-height: 1;
            font-weight: 500;
        }

        [data-testid="stFileUploaderDeleteBtn"]:hover {
            background: #eff6ff !important;
        }

        .hero-shell {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(37, 99, 235, 0.18);
            background:
                linear-gradient(135deg, rgba(255,255,255,.96) 0%, rgba(244,249,255,.94) 56%, rgba(226,239,255,.96) 100%);
            border-radius: 0 0 18px 18px;
            padding: 2rem;
            box-shadow: 0 22px 55px rgba(30, 64, 175, 0.12);
            margin-bottom: 1.25rem;
        }

        .hero-layout {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(330px, .85fr);
            align-items: center;
            gap: 2rem;
        }

        .hero-copy {
            min-width: 0;
        }

        .hero-brand {
            display: flex;
            align-items: center;
            gap: .9rem;
            margin-bottom: 1.75rem;
        }

        .hero-logo-mark,
        .brand-logo-mark {
            display: grid;
            place-items: center;
            background: #ffffff;
            border: 1px solid var(--line);
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.10);
        }

        .hero-logo-mark {
            width: 58px;
            height: 58px;
            border-radius: 16px;
        }

        .brand-logo-mark {
            width: 42px;
            height: 42px;
            border-radius: 12px;
        }

        .hero-logo-mark svg,
        .brand-logo-mark svg {
            width: 76%;
            height: 76%;
        }

        .hero-brand-title {
            color: #0b4fc3;
            font-weight: 900;
            font-size: 1.25rem;
            letter-spacing: .02em;
        }

        .hero-brand-subtitle {
            color: #475569;
            font-size: .9rem;
            margin-top: .15rem;
        }

        .hero-media {
            position: relative;
            min-height: 360px;
            border-radius: 18px;
            overflow: hidden;
            background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);
            box-shadow: 0 20px 48px rgba(37, 99, 235, 0.18);
        }

        .hero-media img {
            width: 100%;
            height: 100%;
            min-height: 360px;
            object-fit: cover;
            object-position: center;
            display: block;
        }

        .hero-media-placeholder {
            height: 100%;
            min-height: 360px;
            display: grid;
            place-items: center;
            padding: 1.5rem;
            text-align: center;
            color: var(--primary-dark);
        }

        .hero-media-placeholder strong {
            display: block;
            font-size: 1.15rem;
            margin-bottom: .45rem;
        }

        .hero-media-placeholder span {
            color: #475569;
            line-height: 1.5;
        }

        .hero-kicker {
            display: none;
            align-items: center;
            gap: .45rem;
            font-size: .78rem;
            letter-spacing: .04em;
            text-transform: uppercase;
            font-weight: 700;
            color: var(--primary-dark);
            background: #dbeafe;
            border: 1px solid #bfdbfe;
            border-radius: 999px;
            padding: .35rem .7rem;
            margin-bottom: .9rem;
        }

        .hero-title {
            margin: 0;
            color: var(--ink);
            font-size: clamp(2rem, 3.45vw, 3.25rem);
            line-height: 1.05;
            letter-spacing: 0;
        }

        .hero-subtitle {
            margin-top: 1rem;
            max-width: 690px;
            color: #334155;
            font-size: 1.04rem;
            line-height: 1.65;
        }

        .hero-note {
            display: none;
        }

        .visual-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .8rem;
            margin-top: 1.6rem;
        }

        .visual-step {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: .95rem .75rem;
            min-height: 116px;
            display: grid;
            justify-items: center;
            align-content: center;
            text-align: center;
            gap: .45rem;
        }

        .visual-step strong {
            display: block;
            color: var(--ink);
            margin-bottom: 0;
            font-size: .86rem;
        }

        .visual-icon {
            width: 42px;
            height: 42px;
            color: var(--primary);
        }

        .visual-icon svg {
            width: 100%;
            height: 100%;
            stroke: currentColor;
            fill: none;
            stroke-width: 2.2;
            stroke-linecap: round;
            stroke-linejoin: round;
        }

        .visual-icon .fill-blue {
            fill: #dbeafe;
            stroke: #2563eb;
        }

        .insight-banner {
            display: grid;
            grid-template-columns: 56px 1fr;
            align-items: center;
            gap: 1rem;
            margin-top: 1.5rem;
            max-width: 600px;
            background: linear-gradient(135deg, #eef6ff 0%, #eaf2ff 100%);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1rem;
        }

        .insight-icon {
            width: 48px;
            height: 48px;
            border-radius: 14px;
            background: linear-gradient(135deg, #2563eb, #0b4fc3);
            color: #ffffff;
            display: grid;
            place-items: center;
        }

        .insight-icon svg {
            width: 30px;
            height: 30px;
        }

        .insight-banner strong {
            display: block;
            color: var(--ink);
            margin-bottom: .2rem;
        }

        .insight-banner span {
            color: var(--muted);
            font-size: .92rem;
        }

        .workflow-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .8rem;
            margin: .7rem 0 1.2rem;
        }

        .workflow-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
        }

        .workflow-card b {
            color: var(--primary-dark);
            display: block;
            margin-bottom: .35rem;
        }

        .workflow-card span {
            color: var(--muted);
            font-size: .92rem;
        }

        .section-card {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1.1rem 1.2rem;
            margin: .8rem 0 1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }

        .about-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .about-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1.2rem;
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.06);
            min-height: 132px;
        }

        .about-card h4 {
            margin: .1rem 0 .45rem;
            color: var(--primary-dark);
            font-size: 1rem;
        }

        .about-card p,
        .about-card li {
            color: #334155;
            font-size: .9rem;
            line-height: 1.45;
        }

        .about-card ul {
            margin: .25rem 0 0 1rem;
            padding: 0;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: .85rem 1rem;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="stMetric"] * {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            opacity: 1 !important;
        }

        div[data-testid="stMetricValue"] {
            color: var(--primary-dark) !important;
            -webkit-text-fill-color: var(--primary-dark) !important;
            font-weight: 850 !important;
            font-size: 1.65rem !important;
            line-height: 1.15 !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            word-break: normal !important;
        }

        div[data-testid="stMetricValue"] *,
        div[data-testid="stMetricValue"] div {
            color: var(--primary-dark) !important;
            -webkit-text-fill-color: var(--primary-dark) !important;
            font-size: inherit !important;
            line-height: inherit !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            word-break: normal !important;
        }

        div[data-testid="stAlert"] {
            border-radius: 12px !important;
            border: 1px solid #fde68a !important;
            background: #fffbeb !important;
            color: #78350f !important;
            -webkit-text-fill-color: #78350f !important;
        }

        div[data-testid="stAlert"] *,
        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] div,
        div[data-testid="stAlert"] span {
            color: #78350f !important;
            -webkit-text-fill-color: #78350f !important;
            opacity: 1 !important;
        }

        div[data-testid="stAlertContent"],
        div[data-testid="stAlertContent"] *,
        div[role="alert"],
        div[role="alert"] *,
        [data-baseweb="notification"],
        [data-baseweb="notification"] * {
            color: #78350f !important;
            -webkit-text-fill-color: #78350f !important;
            opacity: 1 !important;
        }

        div[data-testid="stAlert"] svg {
            color: #b45309 !important;
            fill: #b45309 !important;
        }

        .stButton > button,
        div[data-testid="stDownloadButton"] > button {
            border: 0;
            border-radius: 10px;
            background: var(--primary);
            color: white;
            -webkit-text-fill-color: white;
            font-weight: 700;
            padding: .7rem 1.1rem;
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.25);
            transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
        }

        .stButton > button:hover,
        .stButton > button:focus,
        .stButton > button:active,
        div[data-testid="stDownloadButton"] > button:hover {
            background: var(--primary-dark);
            color: white;
            -webkit-text-fill-color: white;
            transform: translateY(-1px);
            box-shadow: 0 16px 30px rgba(37, 99, 235, 0.32);
        }

        .ranking-table-wrap {
            width: 100%;
            overflow-x: auto;
            border: 1px solid var(--line);
            border-radius: 12px;
            background: #ffffff;
            box-shadow: 0 14px 28px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }

        table.ranking-table {
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
            color: var(--ink);
            font-size: .9rem;
        }

        table.ranking-table thead th {
            background: #eaf2ff;
            color: var(--primary-dark);
            font-weight: 800;
            border-bottom: 1px solid var(--line);
            padding: .7rem .75rem;
            text-align: left;
            white-space: nowrap;
        }

        table.ranking-table tbody td {
            color: var(--ink);
            border-bottom: 1px solid #e2e8f0;
            padding: .7rem .75rem;
            vertical-align: top;
            white-space: nowrap;
        }

        table.ranking-table tbody tr:nth-child(even) {
            background: #f8fbff;
        }

        table.ranking-table tbody tr:hover {
            background: #eff6ff;
        }

        table.ranking-table tbody tr:last-child td {
            border-bottom: 0;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 14px 28px rgba(15, 23, 42, 0.06);
        }

        .footer-note {
            color: var(--muted);
            font-size: .9rem;
            text-align: center;
            padding-top: 1rem;
        }

        @media (max-width: 900px) {
            .visual-strip,
            .workflow-grid,
            .about-grid {
                grid-template-columns: 1fr;
            }

            .hero-shell {
                padding: 1.35rem;
            }

            .hero-layout {
                grid-template-columns: 1fr;
            }

            .hero-media,
            .hero-media img,
            .hero-media-placeholder {
                min-height: 260px;
            }

            .topbar {
                align-items: flex-start;
                flex-direction: column;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model_artifacts():
    """Load the trained ML model and TF-IDF vectorizer."""
    if not MODEL_FILE.exists() or not VECTORIZER_FILE.exists():
        return None, None

    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    return model, vectorizer


def predict_candidate_role(cleaned_cv_text: str, model, vectorizer) -> tuple:
    """Predict role and ML confidence for one cleaned CV."""
    cv_features = vectorizer.transform([cleaned_cv_text])
    predicted_role = model.predict(cv_features)[0]

    # Logistic Regression supports predict_proba, which gives confidence.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(cv_features)[0]
        ml_confidence = float(max(probabilities) * 100)
    else:
        ml_confidence = 0.0

    return predicted_role, ml_confidence


def calculate_job_similarity(cleaned_cv_text: str, cleaned_job_description: str, vectorizer) -> float:
    """Compare CV text with the job description using cosine similarity."""
    vectors = vectorizer.transform([cleaned_cv_text, cleaned_job_description])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return float(similarity * 100)


def apply_role_match_penalty(final_score: float, predicted_role: str, target_role: str) -> tuple:
    """Reduce the final score by 50% when the predicted role does not match the selected role."""
    role_match = "Yes" if predicted_role == target_role else "No"
    if role_match == "No":
        return round(final_score * 0.50, 2), role_match, "Yes"
    return final_score, role_match, "No"


def get_recommendation_with_role_match(final_score: float, role_match: str) -> str:
    """Return the recommendation using both score and selected-role match."""
    if role_match == "No" or final_score < 60:
        return "Weak Fit / Not Recommended"
    if final_score >= 80:
        return "Strong Fit"
    return "Medium Fit"


def screen_candidate(uploaded_file, job_description: str, target_role: str, model, vectorizer) -> dict:
    """Process one CV file and return a complete ranking result."""
    file_extension = Path(uploaded_file.name).suffix.lower()
    if file_extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file format. Please upload PDF, DOCX, or TXT files only.")

    raw_cv_text = extract_text_from_uploaded_file(uploaded_file)
    if not raw_cv_text or len(raw_cv_text.strip()) < 30:
        raise ValueError("Text could not be extracted from this CV. Please check the file or upload another version.")

    cleaned_cv_text = clean_text(raw_cv_text)
    cleaned_job_description = clean_text(job_description)
    if not cleaned_cv_text:
        raise ValueError("Text could not be extracted from this CV. Please check the file or upload another version.")

    predicted_role, ml_confidence = predict_candidate_role(cleaned_cv_text, model, vectorizer)
    jd_similarity = calculate_job_similarity(cleaned_cv_text, cleaned_job_description, vectorizer)
    matched_skills, missing_skills = match_skills(cleaned_cv_text, cleaned_job_description, predicted_role)
    skill_match_score = calculate_skill_score(matched_skills, missing_skills)
    base_final_score = calculate_final_score(jd_similarity, skill_match_score, ml_confidence)
    final_score, role_match, penalty_applied = apply_role_match_penalty(
        base_final_score,
        predicted_role,
        target_role,
    )
    recommendation = get_recommendation_with_role_match(final_score, role_match)

    explanation = generate_candidate_explanation(
        predicted_role=predicted_role,
        target_role=target_role,
        role_match=role_match,
        recommendation=recommendation,
        final_score=final_score,
        confidence_score=ml_confidence,
        similarity_score=jd_similarity,
        skill_score=skill_match_score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
    )

    return {
        "Candidate file name": uploaded_file.name,
        "Extracted text length": len(raw_cv_text),
        "Selected target role": target_role,
        "Predicted role": predicted_role,
        "Role match": role_match,
        "ML confidence": round(ml_confidence, 2),
        "Job description similarity": round(jd_similarity, 2),
        "Skill match score": round(skill_match_score, 2),
        "Base final score": base_final_score,
        "Penalty applied": penalty_applied,
        "Final score": final_score,
        "Recommendation": recommendation,
        "Matched skills": ", ".join(matched_skills) if matched_skills else "None",
        "Missing skills": ", ".join(missing_skills) if missing_skills else "None",
        "Explanation": explanation,
    }


def render_about_project() -> None:
    """Render project evidence as dashboard cards instead of a sidebar page."""
    st.markdown(
        """
        <div id="about-project" class="section-card">
            <h3>About Project</h3>
            <p>AI-powered CV screening and candidate ranking system for public service recruitment.</p>
        </div>
        <div class="about-grid">
            <div class="about-card">
                <div>
                    <h4>Project Overview</h4>
                    <p>This prototype uses machine learning to analyse CVs, predict IT roles,
                    compare candidates with a job description, and rank applicants for HR review.</p>
                </div>
            </div>
            <div class="about-card">
                <div>
                    <h4>Real-World Problem</h4>
                    <p>Manual CV screening takes time and can be inconsistent.
                    This system helps HR review candidates faster and more transparently.</p>
                </div>
            </div>
            <div class="about-card">
                <div>
                    <h4>Technical Method</h4>
                    <ul>
                        <li>Text vectorization: TF-IDF</li>
                        <li>Machine learning model: Logistic Regression</li>
                        <li>Similarity method: Cosine similarity</li>
                        <li>Application framework: Streamlit</li>
                    </ul>
                </div>
            </div>
            <div class="about-card">
                <div>
                    <h4>Final Score Formula</h4>
                    <p>50% AI model confidence + 30% job description match + 20% skill match.
                    Role mismatch reduces the final score and prevents unsuitable Top 3 recommendations.</p>
                </div>
            </div>
            <div class="about-card">
                <div>
                    <h4>Ethical Considerations</h4>
                    <ul>
                        <li>Synthetic anonymized dataset</li>
                        <li>Privacy masking for personal data</li>
                        <li>Human HR review required</li>
                        <li>Transparent scores and explanations</li>
                    </ul>
                </div>
            </div>
            <div class="about-card">
                <div>
                    <h4>Limitations</h4>
                    <ul>
                        <li>Synthetic data may not fully match real CVs</li>
                        <li>Soft skills and interviews are not evaluated</li>
                        <li>Not a replacement for human hiring judgement</li>
                    </ul>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_light_table(table_data: pd.DataFrame) -> None:
    """Render results as a light HTML table so it stays readable in every theme."""
    table_html = table_data.to_html(
        index=False,
        classes="ranking-table",
        border=0,
        escape=True,
    )
    st.markdown(
        f'<div class="ranking-table-wrap">{table_html}</div>',
        unsafe_allow_html=True,
    )


def clear_job_description() -> None:
    """Clear only the job description input."""
    st.session_state["job_description_text"] = ""


def get_hero_media_html() -> str:
    """Return hero image HTML when the local image file exists."""
    if not HERO_IMAGE_FILE.exists():
        return """
        <div class="hero-media-placeholder">
            <div>
                <strong>Add your hero image here</strong>
                <span>Save the image as assets/hero_cv_screening.png to display it in this section.</span>
            </div>
        </div>
        """

    image_bytes = HERO_IMAGE_FILE.read_bytes()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    return (
        '<img src="data:image/png;base64,'
        f'{encoded_image}" alt="AI CV screening dashboard illustration">'
    )


def main() -> None:
    """Render the university demo interface."""
    apply_custom_styles()
    hero_media_html = get_hero_media_html()
    logo_svg = """
        <svg viewBox="0 0 64 64" aria-hidden="true">
            <rect x="13" y="10" width="31" height="42" rx="5" fill="none" stroke="#2563eb" stroke-width="4"/>
            <path d="M36 10v12h12" fill="none" stroke="#2563eb" stroke-width="4" stroke-linejoin="round"/>
            <text x="19" y="36" font-size="13" font-weight="900" fill="#2563eb" font-family="Arial, sans-serif">CV</text>
            <circle cx="43" cy="43" r="11" fill="none" stroke="#172033" stroke-width="5"/>
            <path d="M51 51l10 10" stroke="#172033" stroke-width="5" stroke-linecap="round"/>
        </svg>
    """
    st.markdown(
        f"""
        <div class="topbar">
            <div class="brand">
                <div class="brand-logo-mark">{logo_svg}</div>
                <div>
                    <div class="brand-title">AI CV SCREENING</div>
                    <div class="brand-subtitle">Smart Hiring. Better Future.</div>
                </div>
            </div>
            <div class="nav-pills">
                <a class="active" href="#dashboard">Dashboard</a>
                <a href="#about-project">About Project</a>
            </div>
        </div>
        <div class="hero-shell">
            <div class="hero-layout">
                <div class="hero-copy">
                    <h1 class="hero-title">AI-Powered CV Screening and Candidate Ranking System</h1>
                    <div class="hero-subtitle">
                        Upload CVs, match job requirements, and rank candidates with AI-assisted recommendations.
                    </div>
                    <div class="hero-note">
                        This prototype helps HR specialists review CVs faster by using machine learning,
                        job description matching, and skill comparison. The final result is a recommendation
                        only; final hiring decisions should be made by a human HR specialist.
                    </div>
                    <div class="visual-strip">
                        <div class="visual-step">
                            <div class="visual-icon">
                                <svg viewBox="0 0 48 48">
                                    <path d="M14 7h15l8 8v26H14z"/>
                                    <path d="M29 7v9h8"/>
                                    <path d="M19 26h10M19 32h8"/>
                                    <path d="M33 34v-9M29 29l4-4 4 4"/>
                                </svg>
                            </div>
                            <strong>Upload CVs</strong>
                        </div>
                        <div class="visual-step">
                            <div class="visual-icon">
                                <svg viewBox="0 0 48 48">
                                    <rect x="14" y="16" width="20" height="17" rx="6" class="fill-blue"/>
                                    <rect x="20" y="8" width="8" height="6" rx="3" fill="#2563eb" stroke="#2563eb"/>
                                    <path d="M24 14v2M14 24H8M40 24h-6M14 29H8M40 29h-6M24 33v6"/>
                                    <circle cx="20" cy="24" r="1.5" fill="#2563eb" stroke="#2563eb"/>
                                    <circle cx="28" cy="24" r="1.5" fill="#2563eb" stroke="#2563eb"/>
                                    <path d="M19 28c2 3 8 3 10 0"/>
                                </svg>
                            </div>
                            <strong>AI Matching</strong>
                        </div>
                        <div class="visual-step">
                            <div class="visual-icon">
                                <svg viewBox="0 0 48 48">
                                    <path d="M12 36V24h7v12zM22 36V16h7v20zM32 36V9h7v27z"/>
                                </svg>
                            </div>
                            <strong>Ranking</strong>
                        </div>
                        <div class="visual-step">
                            <div class="visual-icon">
                                <svg viewBox="0 0 48 48">
                                    <path class="fill-blue" d="M24 5l15 6v11c0 10-6 17-15 21-9-4-15-11-15-21V11z"/>
                                    <path d="M18 24l4 4 8-9" stroke="#ffffff" stroke-width="4"/>
                                </svg>
                            </div>
                            <strong>Secure & Private</strong>
                        </div>
                    </div>
                    <div class="insight-banner">
                        <div class="insight-icon">
                            <svg viewBox="0 0 48 48" aria-hidden="true">
                                <path d="M24 5l15 6v11c0 10-6 17-15 21-9-4-15-11-15-21V11z" fill="none" stroke="currentColor" stroke-width="4"/>
                                <path d="M17 24l5 5 10-11" fill="none" stroke="currentColor" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div>
                            <strong>AI-driven insights. Faster screening. Better hires.</strong>
                            <span>Built for HR teams and recruiters.</span>
                        </div>
                    </div>
                </div>
                <div class="hero-media">
                    {hero_media_html}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**How it works:**")
    st.markdown(
        """
        <div class="workflow-grid">
            <div class="workflow-card"><b>1. Select role</b><span>Choose the vacancy target role.</span></div>
            <div class="workflow-card"><b>2. Add job description</b><span>Paste the vacancy requirements.</span></div>
            <div class="workflow-card"><b>3. Upload CVs</b><span>Add one or more candidate files.</span></div>
            <div class="workflow-card"><b>4. Review ranking</b><span>Check all candidates and the Top 3.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, vectorizer = load_model_artifacts()
    if model is None or vectorizer is None:
        st.error("Model files were not found. Please run python train_model.py first.")
        st.stop()

    st.markdown('<div id="dashboard"></div>', unsafe_allow_html=True)

    target_role = st.selectbox(
        "Select Target Role",
        TARGET_ROLE_OPTIONS,
        index=None,
        placeholder="Choose target role",
    )

    if "job_description_text" not in st.session_state:
        st.session_state["job_description_text"] = ""

    title_col, clear_col = st.columns([5, 1])
    title_col.markdown("### Job Description")
    clear_col.button(
        "Clear",
        key="clear_job_description_button",
        on_click=clear_job_description,
        use_container_width=True,
    )
    job_description = st.text_area(
        "Job Description",
        height=180,
        placeholder=(
            "Paste the job description here. Include the required role, skills, "
            "tools, experience level, and responsibilities.\n\n"
            "Example: We are looking for a Backend Developer with Python, FastAPI, "
            "PostgreSQL, REST API, Docker, SQL, and 2+ years of experience."
        ),
        label_visibility="collapsed",
        key="job_description_text",
    )

    st.markdown(
        """
        <div class="section-card">
            <h3>Upload Candidate CVs</h3>
            <p>The system will extract text and analyse each candidate.</p>
            <p><strong>Supported formats:</strong> PDF, DOCX, TXT<br>
            <strong>Maximum file size:</strong> 5MB per CV<br>
            <strong>Upload limit:</strong> You can upload up to 10 CV files at once.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        "Choose CV files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT. Maximum file size: 5MB per CV. You can upload up to 10 CV files at once.",
    )

    if st.button("Start CV Analysis", type="primary"):
        if not target_role:
            st.warning("Please select a target role before starting the analysis.")
            st.stop()

        if not job_description or not job_description.strip():
            st.warning("Please paste a job description before starting the analysis.")
            st.stop()

        if not uploaded_files:
            st.warning("Please upload at least one candidate CV.")
            st.stop()

        if len(uploaded_files) > MAX_UPLOAD_FILES:
            st.warning("You can upload up to 10 CV files at once.")
            st.stop()

        oversized_files = [
            uploaded_file.name
            for uploaded_file in uploaded_files
            if uploaded_file.size > MAX_FILE_SIZE_BYTES
        ]
        if oversized_files:
            st.warning("Each CV file must be 5MB or smaller.")
            for file_name in oversized_files:
                st.write(f"- {file_name}")
            st.stop()

        results = []
        errors = []

        with st.spinner("Extracting CV text, predicting roles, and ranking candidates..."):
            for uploaded_file in uploaded_files:
                try:
                    results.append(screen_candidate(uploaded_file, job_description, target_role, model, vectorizer))
                except Exception as exc:
                    errors.append(f"{uploaded_file.name}: {exc}")

        if errors:
            st.warning("Some files could not be processed.")
            for error in errors:
                st.write(error)

        if not results:
            st.error("No candidates could be ranked. Please check the uploaded files.")
            st.stop()

        ranked_results = sorted(results, key=lambda item: item["Final score"], reverse=True)
        ranked_table = pd.DataFrame(ranked_results)
        ranked_table.insert(0, "Rank", range(1, len(ranked_table) + 1))

        strong_count = sum(item["Recommendation"] == "Strong Fit" for item in ranked_results)
        medium_count = sum(item["Recommendation"] == "Medium Fit" for item in ranked_results)
        weak_count = sum(item["Recommendation"] == "Weak Fit / Not Recommended" for item in ranked_results)
        highest_score = max(item["Final score"] for item in ranked_results)
        suitable_candidates = [
            item for item in ranked_results
            if item["Role match"] == "Yes" and item["Final score"] >= 60
        ]

        display_columns = [
            "Rank",
            "Candidate file name",
            "Selected target role",
            "Predicted role",
            "Role match",
            "ML confidence",
            "Job description similarity",
            "Skill match score",
            "Final score",
            "Recommendation",
        ]

        display_table = ranked_table[display_columns].rename(columns={
            "Candidate file name": "Candidate File",
            "Selected target role": "Selected Target Role",
            "Predicted role": "Predicted Role",
            "Role match": "Role Match",
            "ML confidence": "AI Confidence",
            "Job description similarity": "Job Match",
            "Skill match score": "Skill Match",
            "Final score": "Final Score",
        })

        st.markdown(
            """
            <div class="section-card">
                <h3>Analysis Summary</h3>
                <p>Quick overview of the screening results for this vacancy.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total CVs Analyzed", len(ranked_results))
        col2.metric("Strong Fit Candidates", strong_count)
        col3.metric("Medium Fit Candidates", medium_count)
        col4.metric("Weak Fit Candidates", weak_count)
        col5.metric("Highest Score", f"{highest_score}/100")

        st.markdown(
            """
            <div class="section-card">
                <h3>Candidate Ranking Results</h3>
                <p>Candidates are ranked by their final score. A higher score means the CV
                is a closer match to the job description and selected target role.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_light_table(display_table)

        st.markdown(
            """
            <div class="section-card">
                <h3>Top 3 Recommended Candidates</h3>
                <p>Only candidates with Role Match = Yes and Final Score >= 60 can appear here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not suitable_candidates:
            st.warning(
                "No suitable candidates found for this role. Please review more CVs "
                "or adjust the job description."
            )

        for rank, candidate in enumerate(suitable_candidates[:3], start=1):
            title = f"Candidate Details: {candidate['Candidate file name']}"
            with st.expander(title, expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Selected Target Role", candidate["Selected target role"])
                col2.metric("Predicted Role", candidate["Predicted role"])
                col3.metric("Role Match", candidate["Role match"])
                col4.metric("Final Score", f"{candidate['Final score']}/100")

                col5, col6, col7 = st.columns(3)
                col5.metric("AI Confidence", f"{candidate['ML confidence']}%")
                col6.metric("Job Match", f"{candidate['Job description similarity']}%")
                col7.metric("Skill Match", f"{candidate['Skill match score']}%")

                st.write(f"Role mismatch penalty applied: {candidate['Penalty applied']}")
                st.write(f"Recommendation: {candidate['Recommendation']}")

                st.write("Matched Skills:")
                st.write(candidate["Matched skills"])
                st.write("Missing Skills:")
                st.write(candidate["Missing skills"])
                st.write("Explanation:")
                st.info(candidate["Explanation"])
                st.write("HR Review Note:")
                st.warning(
                    "This result is an AI-assisted recommendation only. Final hiring "
                    "decisions should be reviewed and approved by HR specialists."
                )

        st.markdown(
            """
            <div class="section-card">
                <h3>Candidate Debugging Information</h3>
                <p>Use this section to inspect how each candidate score was calculated.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for candidate in ranked_results:
            debug_title = f"Debug: {candidate['Candidate file name']}"
            with st.expander(debug_title):
                st.write(f"Extracted text length: {candidate['Extracted text length']}")
                st.write(f"Selected target role: {candidate['Selected target role']}")
                st.write(f"Predicted role: {candidate['Predicted role']}")
                st.write(f"Role match status: {candidate['Role match']}")
                st.write(f"ML confidence: {candidate['ML confidence']}%")
                st.write(f"JD similarity: {candidate['Job description similarity']}%")
                st.write(f"Skill match: {candidate['Skill match score']}%")
                st.write(f"Role mismatch penalty applied: {candidate['Penalty applied']}")
                st.write(f"Final score: {candidate['Final score']}/100")
                st.write(f"Recommendation: {candidate['Recommendation']}")
                st.write(f"Matched Skills: {candidate['Matched skills']}")
                st.write(f"Missing Skills: {candidate['Missing skills']}")
                st.write(f"Explanation: {candidate['Explanation']}")
                if candidate["Role match"] == "No":
                    st.warning(
                        "Role mismatch detected. This CV may be suitable for another "
                        "IT role, but it does not match the selected vacancy."
                    )
                st.write(
                    "HR Review Note: This result is an AI-assisted recommendation only. "
                    "Final hiring decisions should be reviewed and approved by HR specialists."
                )

        csv_data = display_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download ranking results as CSV",
            data=csv_data,
            file_name="candidate_ranking_results.csv",
            mime="text/csv",
        )

    render_about_project()

    st.markdown(
        """
        <div class="footer-note">
            Developed for BTEC Unit 2 Independent Project - AI-assisted CV screening
            prototype for public service recruitment.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
