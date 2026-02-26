from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from predict.helper import load_model_and_scaler, transform_profile_to_features


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_predict_artifacts(model_dir: Optional[str] = None):
    """
    Load model + scaler once and cache them.
    """
    if model_dir is None:
        model_dir = str(_project_root() / "model" / "predict")
    model, scaler = load_model_and_scaler(model_dir)
    return model, scaler

def predict_from_profile(
    profile: Dict[str, Any],
    *,
    model: Any = None,
    scaler: Any = None,
) -> Dict[str, Any]:
    if not isinstance(profile, dict) or not profile:
        raise ValueError("profile must be a non-empty dict")

    if model is None or scaler is None:
        model, scaler = get_predict_artifacts()

    X = transform_profile_to_features(profile, scaler, model)

    # sklearn-like outputs
    y_pred = model.predict(X)[0]
    proba_no_risk: Optional[float] = None
    proba_at_risk: Optional[float] = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        if len(proba) >= 2:
            proba_no_risk = float(proba[0])
            proba_at_risk = float(proba[1])
        elif len(proba) == 1:
            proba_at_risk = float(proba[0])

    risk_score = proba_at_risk
    if risk_score is None:
        risk_score = 1.0 if int(y_pred) == 1 else 0.0

    # Simple buckets (tunable)
    if risk_score >= 0.66:
        risk_level = "HIGH"
    elif risk_score >= 0.33:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    return {
        "prediction": int(y_pred) if str(y_pred).isdigit() else y_pred,
        "risk_score": float(risk_score),
        "risk_level": risk_level,
        "proba": (
            None
            if (proba_no_risk is None and proba_at_risk is None)
            else {"no_risk": proba_no_risk, "at_risk": proba_at_risk}
        ),
    }

if __name__ == "__main__":
    import json
    import sys

    dummy_state_moderate_risk = {
        "profile": {
            "gender": "Female",
            "age": 23,
            "academic_pressure": 5,
            "study_satisfaction": 1,
            "study_hours": 9.0,
            "degree": "B.Tech",
            "cgpa": 3.2,
            "sleep_duration": "5-6 hours",
            "dietary_habits": "Moderate",
            "suicidal_thoughts": True,
            "family_history": True,
            "financial_stress": 1
        },
        "messages": [],
        "missing_fields": []
    }

    out = predict_from_profile(dummy_state_moderate_risk["profile"])
    print(json.dumps(out, ensure_ascii=False, indent=2))
