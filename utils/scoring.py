"""Candidate scoring helpers."""

from typing import List


def calculate_skill_score(matched_skills: List[str], missing_skills: List[str]) -> float:
    """Calculate a percentage score for skill overlap."""
    total_required = len(matched_skills) + len(missing_skills)
    if total_required == 0:
        return 0.0
    return (len(matched_skills) / total_required) * 100


def calculate_final_score(similarity_score: float, skill_score: float, confidence_score: float) -> float:
    """Combine scores using the required formula.

    Final Score = 50% ML confidence + 30% job description similarity
    + 20% skill match score.
    """
    final_score = (
        0.50 * confidence_score
        + 0.30 * similarity_score
        + 0.20 * skill_score
    )
    return round(float(final_score), 2)


def get_recommendation(final_score: float) -> str:
    """Convert the final score into an HR review recommendation."""
    if final_score >= 80:
        return "Strong Fit"
    if final_score >= 60:
        return "Medium Fit"
    return "Weak Fit"
