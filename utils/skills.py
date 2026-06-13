"""Skill extraction and matching helpers."""

from typing import Dict, List, Tuple

from utils.preprocessing import clean_text


SKILL_BANK: Dict[str, List[str]] = {
    "Frontend Developer": [
        "html", "css", "javascript", "typescript", "react", "vue", "angular",
        "redux", "responsive design", "accessibility", "tailwind", "bootstrap",
        "next.js", "api integration", "jest", "cypress", "web performance",
        "figma", "storybook", "vite",
    ],
    "Backend Developer": [
        "python", "java", "node.js", "django", "flask", "fastapi", "spring",
        "sql", "postgresql", "mysql", "mongodb", "rest api", "docker",
        "kubernetes", "spring boot", "authentication", "jwt", "microservices",
        "api documentation", "azure", "aws",
    ],
    "AI Engineer": [
        "artificial intelligence", "nlp", "computer vision", "deep learning",
        "tensorflow", "pytorch", "transformers", "llm", "prompt engineering",
        "rag", "langchain", "openai api", "vector databases", "embeddings",
        "fine-tuning", "guardrails", "agentic ai", "pinecone", "chromadb",
    ],
    "Machine Learning Engineer": [
        "machine learning", "scikit-learn", "pandas", "numpy", "model training",
        "feature engineering", "classification", "regression", "mlops", "statistics",
        "pytorch", "tensorflow", "mlflow", "model deployment", "model evaluation",
        "feature store", "model registry", "batch inference",
    ],
    "ML Engineer": [
        "machine learning", "scikit-learn", "sklearn", "pandas", "numpy",
        "model training", "feature engineering", "classification", "regression",
        "tensorflow", "pytorch", "statistics", "mlflow", "model deployment",
    ],
    "QA Engineer": [
        "testing", "manual testing", "automation testing", "selenium", "pytest",
        "test cases", "bug tracking", "jira", "api testing", "postman",
        "quality assurance", "regression testing", "testrail", "allure reports",
        "selenium webdriver", "ci/cd",
    ],
    "Business Analyst": [
        "business analysis", "requirements gathering", "stakeholder management",
        "user stories", "process mapping", "sql", "excel", "power bi",
        "tableau", "reporting", "documentation",
    ],
    "Flutter Developer": [
        "flutter", "dart", "firebase", "mobile development", "android", "ios",
        "state management", "bloc", "provider", "rest api",
        "mobile ui", "push notifications", "app store deployment", "sqlite",
        "flutter devtools", "cross-platform widgets",
    ],
}


def all_known_skills() -> List[str]:
    """Return a sorted unique list of all skills used by the system."""
    skills = set()
    for role_skills in SKILL_BANK.values():
        skills.update(role_skills)
    return sorted(skills)


def extract_skills(text: str, role_hint: str = "") -> List[str]:
    """Find known skills that appear in the text."""
    cleaned = clean_text(text)
    possible_skills = set(all_known_skills())

    if role_hint in SKILL_BANK:
        possible_skills.update(SKILL_BANK[role_hint])

    found = []
    for skill in sorted(possible_skills):
        normalized_skill = clean_text(skill)
        if normalized_skill and normalized_skill in cleaned:
            found.append(skill)
    return found


def match_skills(cv_text: str, job_description: str, role_hint: str = "") -> Tuple[List[str], List[str]]:
    """Return skills found in both CV and JD, and JD skills missing from CV."""
    cv_skills = set(extract_skills(cv_text, role_hint))
    jd_skills = set(extract_skills(job_description, role_hint))

    matched = sorted(cv_skills.intersection(jd_skills))
    missing = sorted(jd_skills.difference(cv_skills))
    return matched, missing
