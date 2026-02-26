from typing import Annotated, Literal, Optional, List, Dict, Any
from typing_extensions import TypedDict
from dataclasses import dataclass
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, validator

# Type definitions
Gender = Literal["Male", "Female"]
SleepDuration = Literal["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
DietaryHabits = Literal["Unhealthy", "Moderate", "Healthy"]
Degree = Literal[
    "Class 12", "BA", "BSc", "B.Com", "BE", "B.Tech", "BCA", "B.Ed", "BHM", "B.Pharm", "LLB",
    "MA", "MSc", "M.Com", "M.Tech", "MBA", "MCA", "M.Ed", "M.Pharm", "MD", "PhD"
]

Intent = Literal[
    "PREDICT",
    "EXTRACT_PROFILE",
    "DIRECT_RESPONSE",
    "RAG_QA",
    "EVALUATE_AND_ADVISE",
    "UNKNOWN",
]

RiskLevel = Literal["LOW", "MODERATE", "HIGH"]

@dataclass
class UserProfile(BaseModel):
    """User profile with strict validation"""
    
    # Demographics
    gender: Optional[Gender] = Field(None, description="User's gender")
    age: Optional[int] = Field(None, ge=18, le=40, description="User's age (18-40)")
    
    # Academic & Study
    academic_pressure: Optional[int] = Field(None, ge=1, le=5, description="Academic pressure level (1-5)")
    study_satisfaction: Optional[int] = Field(None, ge=1, le=5, description="Study satisfaction level (1-5)")
    study_hours: Optional[float] = Field(None, ge=0, le=24, description="Study hours per day (0-24)")
    degree: Optional[Degree] = Field(None, description="Current degree/program")
    cgpa: Optional[float] = Field(None, ge=0.0, le=10.0, description="Cumulative GPA (0.0-10.0)")
    
    # Lifestyle
    sleep_duration: Optional[SleepDuration] = Field(None, description="Sleep duration category")
    dietary_habits: Optional[DietaryHabits] = Field(None, description="Dietary habits quality")
    
    # Mental Health
    suicidal_thoughts: Optional[bool] = Field(None, description="History of suicidal thoughts")
    family_history: Optional[bool] = Field(None, description="Family history of mental illness")
    
    # Financial
    financial_stress: Optional[int] = Field(None, ge=1, le=5, description="Financial stress level (1-5)")
    
    def get_missing_fields(self) -> List[str]:
        """Get list of fields that are still None"""
        missing = []
        for field_name, field_value in self.dict().items():
            if field_value is None:
                missing.append(field_name)
        return missing
    
    def get_completion_rate(self) -> float:
        """Get percentage of completed fields"""
        total_fields = len(self.dict())
        missing_count = len(self.get_missing_fields())
        return ((total_fields - missing_count) / total_fields) * 100

class IntakerOutput(BaseModel):
    profile: Optional[UserProfile] = Field(default=None, description="Extracted profile information")

class IntentClassifierOutput(BaseModel):
    intent: Intent = Field(description="Intent from user message")
    confidence: float = Field(description="Confidence (0-1)", ge=0, le=1)
    reason: str = Field(description="Reason for classifying this intent")


class PredictProba(BaseModel):
    no_risk: Optional[float] = None
    at_risk: Optional[float] = None

class PredictResult(BaseModel):
    """
    Persistable prediction result (non-diagnostic screening output).
    """
    prediction: Any
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    proba: Optional[PredictProba] = None


class ChatState(TypedDict, total=False):
    """
    Complete state definition for agent chat.
    """
    thread_id: str 
    pending_agents: List[str]
    completed_agents: List[str]
    intent: Intent
    profile: UserProfile
    missing_fields: List[str]
    prediction: PredictResult
    messages: Annotated[list, add_messages]
