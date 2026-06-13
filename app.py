"""Streamlit app for ML-based CV screening and candidate ranking.

Run from the project folder:
    streamlit run app.py
"""

from pathlib import Path

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
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
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


def show_project_notes() -> None:
    """Show concise academic and ethical notes in the sidebar."""
    with st.sidebar:
        st.header("Project Evidence")
        st.markdown("**AI Model:**")
        st.write(
            "This system uses a trained machine learning model to predict the most "
            "suitable IT role for each CV."
        )

        st.markdown("**Technical Method:**")
        st.write("TF-IDF Vectorization + Logistic Regression")

        st.markdown("**What is TF-IDF Vectorization?**")
        st.write(
            "TF-IDF is used to convert CV text into numerical features that the "
            "machine learning model can understand. It identifies important words "
            "and skills in the CV, such as Python, React, SQL, Selenium, Flutter, "
            "or Machine Learning."
        )

        st.markdown("**What is Logistic Regression?**")
        st.write(
            "Logistic Regression is the machine learning classification model used "
            "in this project. It learns from the training dataset and predicts which "
            "IT role a candidate's CV is most suitable for, such as Frontend "
            "Developer, Backend Developer, QA Engineer, AI Engineer, Machine "
            "Learning Engineer, or Flutter Developer."
        )

        st.markdown("**Simple explanation:**")
        st.write(
            "TF-IDF reads the CV text and turns it into numbers. Logistic Regression "
            "uses those numbers to predict the most suitable role for the candidate."
        )

        st.markdown("**Example:**")
        st.write(
            "If a CV contains skills such as Postman, Jira, API Testing, SQL, and "
            "Selenium, the model may predict the candidate as a QA Engineer."
        )

        st.markdown("**What the system checks:**")
        st.write("- AI model confidence")
        st.write("- Similarity between the CV and job description")
        st.write("- Matched and missing skills")

        st.markdown("**Final Score:**")
        st.write("50% AI model confidence")
        st.write("30% job description match")
        st.write("20% skill match")

        st.markdown("**Recommendation Levels:**")
        st.write("80–100 and Role Match = Yes: Strong Fit")
        st.write("60–79 and Role Match = Yes: Medium Fit")
        st.write("Below 60 or Role Match = No: Weak Fit / Not Recommended")

        st.markdown("**Strong Fit:**")
        st.write(
            "This candidate is highly suitable for the role based on the CV content, "
            "matched skills, and job description comparison."
        )

        st.markdown("**Medium Fit:**")
        st.write(
            "This candidate partially matches the role but may have some missing "
            "skills or weaker job description alignment."
        )

        st.markdown("**Weak Fit:**")
        st.write(
            "This candidate has limited alignment with the job description and may "
            "not be the best match for this role."
        )

        st.markdown("**Privacy Note:**")
        st.write(
            "Emails, phone numbers, links, and address-like text are masked during "
            "preprocessing."
        )


def main() -> None:
    """Render the university demo interface."""
    st.title("AI-Powered CV Screening and Candidate Ranking System")
    st.caption(
        "Upload candidate CVs, add a job description, and the system will analyze, "
        "score, and rank candidates based on how well they match the role."
    )
    st.write(
        "This prototype helps HR specialists review CVs faster by using machine "
        "learning, job description matching, and skill comparison. The final result "
        "is a recommendation only; final hiring decisions should be made by a human "
        "HR specialist."
    )
    st.markdown("**How it works:**")
    st.write("1. Paste the job description.")
    st.write("2. Upload one or more candidate CV files.")
    st.write("3. Click “Start CV Analysis”.")
    st.write("4. Review the ranked results and Top 3 recommended candidates.")

    model, vectorizer = load_model_artifacts()
    if model is None or vectorizer is None:
        st.error("Model files were not found. Please run python train_model.py first.")
        st.stop()

    show_project_notes()

    target_role = st.selectbox(
        "Select Target Role",
        TARGET_ROLE_OPTIONS,
    )

    st.subheader("Job Description")
    st.write(
        "Paste the job description here. Include the required role, skills, tools, "
        "experience level, and responsibilities."
    )
    job_description = st.text_area(
        "Job Description",
        height=180,
        placeholder=(
            "Example: We are looking for a Backend Developer with Python, FastAPI, "
            "PostgreSQL, REST API, Docker, SQL, and 2+ years of experience."
        ),
    )

    st.subheader("Upload Candidate CVs")
    st.write(
        "Upload one or more CV files in PDF, DOCX, or TXT format. The system will "
        "extract text and analyze each candidate."
    )
    uploaded_files = st.file_uploader(
        "Choose CV files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload one or more CV files in PDF, DOCX, or TXT format.",
    )

    if st.button("Start CV Analysis", type="primary"):
        if not job_description or not job_description.strip():
            st.warning("Please paste a job description before starting the analysis.")
            st.stop()

        if not uploaded_files:
            st.warning("Please upload at least one candidate CV.")
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

        st.subheader("Analysis Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total CVs Analyzed", len(ranked_results))
        col2.metric("Strong Fit Candidates", strong_count)
        col3.metric("Medium Fit Candidates", medium_count)
        col4.metric("Weak Fit Candidates", weak_count)
        col5.metric("Highest Score", f"{highest_score}/100")

        st.subheader("Candidate Ranking Results")
        st.write(
            "Candidates are ranked by their final score. A higher score means the CV "
            "is a closer match to the job description."
        )
        st.dataframe(display_table, use_container_width=True)

        st.subheader("Top 3 Recommended Candidates")
        st.write(
            "These candidates received the highest final scores and are recommended "
            "for HR review."
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

        st.subheader("Candidate Debugging Information")
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

    st.caption(
        "Developed for BTEC Unit 2 Independent Project — AI-assisted CV screening "
        "prototype for public service recruitment."
    )


if __name__ == "__main__":
    main()
