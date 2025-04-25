# tasks.py

from typing import List
from interview.gemini_client import gemini_chat

def generate_question(job_title: str, history: List[dict], qnum: int) -> str:
    """
    Build a single-text prompt then call Gemini to generate question #qnum.
    """
    prompt = (
        f"You are an expert interviewer for the '{job_title}' position. "
        "Generate exactly one clear, focused next interview question."
    )
    if history:
        for i, h in enumerate(history, start=1):
            prompt += f"\nQ{i}: {h['question']}\nA{i}: {h['answer']}"
    else:
        prompt += "\nNo prior responses."

    prompt += f"\nNow, generate interview question #{qnum}."
    return gemini_chat(prompt)


def evaluate_interview(job_title: str, history: List[dict]) -> str:
    """
    Build a prompt asking Gemini to evaluate the full history, then call it.
    """
    prompt = (
        f"You are an HR expert evaluating a candidate for '{job_title}'. "
        "Analyze the entire interview and provide:\n"
        "1. Overall Decision (PASS/FAIL)\n"
        "2. Score (0–100)\n"
        "3. Key Strengths\n"
        "4. Areas for Improvement\n"
        "5. Specific Tips\n\n"
        "Format clearly with bullet points."
    )
    for i, h in enumerate(history, start=1):
        prompt += f"\nQ{i}: {h['question']}\nA{i}: {h['answer']}"

    prompt += "\nPlease produce your detailed evaluation now."
    return gemini_chat(prompt)
