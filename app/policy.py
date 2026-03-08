from __future__ import annotations

from app.config import settings


def decide(score: float, amount: float | None = None) -> tuple[str, list[str]]:
    reasons: list[str] = [f"ml_score={score:.4f}"]

    if amount is not None and amount > 5000:
        reasons.append("high_amount_signal")

    if score >= settings.decline_threshold:
        return "DECLINE", reasons
    if score >= settings.challenge_threshold:
        return "CHALLENGE", reasons
    if score >= settings.approve_threshold:
        return "REVIEW", reasons
    return "APPROVE", reasons
