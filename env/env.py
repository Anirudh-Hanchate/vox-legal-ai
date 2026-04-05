import asyncio
from typing import Dict, List, Any, Optional
from openenv.core.env_server.interfaces import Environment
from .models import LegalObservation, LegalAction

# Real-world Mapping: (Type, Location) -> Lawyer
LAWYER_DIRECTORY = {
    ("property", "bengaluru"): "Adv. Shankar (Property Specialist)",
    ("finance", "mumbai"): "Adv. Mishra (Financial Crimes)",
    ("civil", "delhi"): "Adv. Gupta (Civil Litigation)",
    ("criminal", "delhi"): "Adv. Khan (Criminal Defense)",
    ("property", "mumbai"): "Adv. Deshmukh (Real Estate)",
}

CASE_SAMPLES = [
    {
        "id": "task_ramesh_kn",
        "text": "ನನ್ನ ತಮ್ಮ ನನ್ನ ಜಮೀನನ್ನು ಅಕ್ರಮವಾಗಿ ವಶಪಡಿಸಿಕೊಂಡಿದ್ದಾನೆ. (My brother took my land illegally.)",
        "language": "kannada",
        "location": "bengaluru",
        "gt": {"case_type": "property", "priority": "high", "lawyer": "Adv. Shankar (Property Specialist)"}
    },
    {
        "id": "task_rahul_hi",
        "text": "कंपनी ने पिछले 3 महीनों से मेरा वेतन नहीं दिया है। (Company hasn't paid my salary for 3 months.)",
        "language": "hindi",
        "location": "mumbai",
        "gt": {"case_type": "finance", "priority": "medium", "lawyer": "Adv. Mishra (Financial Crimes)"}
    },
    {
        "id": "task_john_en",
        "text": "My neighbor is encroaching on my driveway with a new fence.",
        "language": "english",
        "location": "delhi",
        "gt": {"case_type": "property", "priority": "medium", "lawyer": "Adv. Deshmukh (Real Estate)"}
    }
]

class VolksLegalEnv(Environment[LegalAction, LegalObservation, Any]):
    def __init__(self, task_id: str = "task_ramesh_kn"):
        self.task_id = task_id
        self.current_case = next((c for c in CASE_SAMPLES if c["id"] == task_id), CASE_SAMPLES[0])
        self.state_data = {
            "case_type": None, "priority": None, 
            "steps": [], "lawyer": None, "completed_actions": set()
        }
        self.total_reward = 0.0
        self.step_count = 0

    def reset(self):
        self.state_data = {"case_type": None, "priority": None, "steps": [], "lawyer": None, "completed_actions": set()}
        self.total_reward = 0.0
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self) -> LegalObservation:
        return LegalObservation(
            case_text=self.current_case["text"],
            language=self.current_case["language"],
            location=self.current_case["location"],
            case_type=self.state_data["case_type"],
            priority=self.state_data["priority"],
            legal_steps_generated=self.state_data["steps"],
            assigned_lawyer=self.state_data["lawyer"]
        )

    def step(self, action: LegalAction):
        self.step_count += 1
        reward = 0.0
        gt = self.current_case["gt"]
        
        if action.action_type == "classify_case":
            if action.value.lower() == gt["case_type"] and "classify" not in self.state_data["completed_actions"]:
                reward = 0.25
                self.state_data["case_type"] = action.value.lower()
                self.state_data["completed_actions"].add("classify")
            else: reward = -0.1

        elif action.action_type == "set_priority":
            if action.value.lower() == gt["priority"] and "priority" not in self.state_data["completed_actions"]:
                reward = 0.20
                self.state_data["priority"] = action.value.lower()
                self.state_data["completed_actions"].add("priority")
            else: reward = -0.05

        elif action.action_type == "generate_guidance":
            steps = [s for s in action.value.split('\n') if len(s) > 10]
            if len(steps) >= 3 and "guidance" not in self.state_data["completed_actions"]:
                reward = 0.25
                self.state_data["steps"] = steps
                self.state_data["completed_actions"].add("guidance")
            else: reward = -0.1

        elif action.action_type == "assign_lawyer":
            if action.value == gt["lawyer"] and "lawyer" not in self.state_data["completed_actions"]:
                reward = 0.30
                self.state_data["lawyer"] = action.value
                self.state_data["completed_actions"].add("lawyer")
            else: reward = -0.2

        self.total_reward = max(0.0, min(1.0, self.total_reward + reward))
        done = len(self.state_data["completed_actions"]) >= 4 or self.step_count >= 8
        return self._get_obs(), reward, done, {"total_score": self.total_reward}

    def state(self):
        return self.state_data

    def close(self):
        pass