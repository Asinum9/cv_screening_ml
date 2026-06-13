"""Optional OpenAI explanation helper with a rule-based fallback.

The app does not send names, phone numbers, emails, addresses, raw CV text,
or uploaded file names to OpenAI. Only job-related scores and skill evidence
are used for explanation.
"""

import os
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv

    PROJECT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(PROJECT_ENV_FILE)
except ImportError:
    # The app still works without python-dotenv; users can install requirements
    # or set OPENAI_API_KEY in their environment.
    pass


PLACEHOLDER_KEYS = {
    "",
    "your_openai_api_key_here",
    "paste_your_openai_api_key_here",
    "sk-your-key-here",
}


def build_rule_based_explanation(
    predicted_role: str,
    target_role: str,
    role_match: str,
    recommendation: str,
    final_score: float,
    confidence_score: float,
    similarity_score: float,
    skill_score: float,
    matched_skills: List[str],
    missing_skills: List[str],
) -> str:
    """Create a transparent explanation without using an external API."""
    matched_text = ", ".join(matched_skills[:8]) if matched_skills else "no clear required skills"
    missing_text = ", ".join(missing_skills[:8]) if missing_skills else "no major required skills"

    if role_match == "No":
        threshold_text = ""
        if final_score < 60:
            threshold_text = (
                " This candidate is not recommended because the final score is "
                "below the minimum recommendation threshold."
            )
        return (
            f"The model predicts this CV is closer to {predicted_role}, but the "
            f"selected vacancy is {target_role}. Therefore, the candidate is not "
            "recommended for this specific vacancy, even if they may be suitable "
            "for another role. Because the role does not match and the job/skill "
            "match scores may be low, this candidate is not recommended for this vacancy."
            f"{threshold_text} "
            f"Final score: {final_score}/100. Matched skills include: {matched_text}. "
            f"Missing or weaker areas include: {missing_text}."
        )

    if final_score < 60:
        return (
            "This candidate matches the selected target role, but is not recommended "
            "because the final score is below the minimum recommendation threshold. "
            f"The final score is {final_score}/100. This score combines ML confidence "
            f"({confidence_score:.2f}/100), job description similarity "
            f"({similarity_score:.2f}/100), and skill match ({skill_score:.2f}/100). "
            f"Matched skills include: {matched_text}. Missing or weaker areas include: {missing_text}."
        )

    return (
        "This candidate matches the selected target role and is recommended based "
        "on the final score, matched skills, job description similarity, and ML confidence. "
        f"The result is {recommendation} with a final score of {final_score}/100. "
        f"This score combines ML confidence ({confidence_score:.2f}/100), "
        f"job description similarity ({similarity_score:.2f}/100), and "
        f"skill match ({skill_score:.2f}/100). "
        f"Matched skills include: {matched_text}. "
        f"Missing or weaker areas include: {missing_text}. "
        "This is only a recommendation for HR review and must not be treated as a final hiring decision."
    )


def generate_candidate_explanation(
    predicted_role: str,
    target_role: str,
    role_match: str,
    recommendation: str,
    final_score: float,
    confidence_score: float,
    similarity_score: float,
    skill_score: float,
    matched_skills: List[str],
    missing_skills: List[str],
) -> str:
    """Generate an explanation using OpenAI when configured, otherwise fallback."""
    fallback = build_rule_based_explanation(
        predicted_role,
        target_role,
        role_match,
        recommendation,
        final_score,
        confidence_score,
        similarity_score,
        skill_score,
        matched_skills,
        missing_skills,
    )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key in PLACEHOLDER_KEYS:
        return fallback

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = f"""
Write a short HR-friendly explanation for this CV screening result.
Use only job-related evidence. Do not make a final hiring decision.
Clearly say this result is only a recommendation for HR review.
No personal data or raw CV text is provided.

Predicted role: {predicted_role}
Selected target role: {target_role}
Role match: {role_match}
Recommendation: {recommendation}
Final score: {final_score}/100
ML confidence: {confidence_score:.2f}/100
Job description similarity: {similarity_score:.2f}/100
Skill match score: {skill_score:.2f}/100
Matched skills: {', '.join(matched_skills) if matched_skills else 'None'}
Missing skills: {', '.join(missing_skills) if missing_skills else 'None'}

If role match is Yes, say:
This candidate matches the selected target role and is recommended based on the final score, matched skills, job description similarity, and ML confidence.

If role match is No, say:
This candidate was predicted as {predicted_role}, but the selected target role is {target_role}. Therefore, the candidate is not recommended for this specific vacancy, even if they may be suitable for another role.

If final score is below 60, say:
This candidate is not recommended because the final score is below the minimum recommendation threshold.
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You explain recruitment screening recommendations fairly, "
                        "briefly, and without making final hiring decisions."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=180,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return fallback
