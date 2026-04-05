from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class LegalObservation(BaseModel):
    case_text: str = Field(..., description="Description of the legal issue.")
    language: str = Field(..., description="User's language: english, kannada, or hindi.")
    location: str = Field(..., description="User's city.")
    case_type: Optional[str] = None
    priority: Optional[str] = None
    legal_steps_generated: List[str] = []
    assigned_lawyer: Optional[str] = None
    reward: float = 0.0
    done: bool = False
    info: dict = {}

class LegalAction(BaseModel):
    action_type: Literal["classify_case", "set_priority", "generate_guidance", "assign_lawyer"]
    value: str

class LegalReward(BaseModel):
    reward: float = 0.0
    done: bool = False
    info: dict = {}