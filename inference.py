import os
import json
import asyncio
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
from env.env import VolksLegalEnv
from env.models import LegalAction

API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_ID = os.getenv("TASK_ID", "task_ramesh_kn")

SYSTEM_PROMPT = """You are Volks Legal AI. Respond ONLY with a JSON object.
You are playing a turn-based game. You can ONLY output ONE action per step. NEVER output multiple actions at once.
Your JSON must strictly have exactly two keys: "action_type" and "value".

Here is the exact sequence of 4 actions you must take, one at a time:
1. {"action_type": "classify_case", "value": "property|finance|civil|criminal"}
2. {"action_type": "set_priority", "value": "low|medium|high"}
3. {"action_type": "generate_guidance", "value": "Step 1...\\nStep 2...\\nStep 3..."}
4. {"action_type": "assign_lawyer", "value": "Adv. Name"}

Look at the Status in the prompt to see which fields are already filled, and generate the NEXT single logical action. Do NOT repeat actions.

Lawyers mapping (case_type, location):
- (property, bengaluru): Adv. Shankar (Property Specialist)
- (finance, mumbai): Adv. Mishra (Financial Crimes)
- (civil, delhi): Adv. Gupta (Civil Litigation)
"""

async def run_inference():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = VolksLegalEnv(task_id=TASK_ID)
    
    print(f"[START] task={TASK_ID} env=volks-legal-ai model={MODEL_NAME}")
    
    obs = env.reset()
    rewards = []
    
    for step in range(1, 7):
        prompt = f"Case: {obs.case_text}\nLang: {obs.language}\nLoc: {obs.location}\nStatus: {obs.model_dump_json()}"
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        act_dict = json.loads(response.choices[0].message.content)
        if isinstance(act_dict, list) and len(act_dict) > 0:
            act_dict = act_dict[0]
        if isinstance(act_dict, dict):
            if "actions" in act_dict and isinstance(act_dict["actions"], list) and len(act_dict["actions"]) > 0:
                act_dict = act_dict["actions"][0]
            elif "action" in act_dict and isinstance(act_dict["action"], dict):
                act_dict = act_dict["action"]
        action = LegalAction(**act_dict)
        
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        
        print(f"[STEP] step={step} action={act_dict['action_type']} reward={reward:.2f} done={str(done).lower()} error=null")
        
        if done: break

    total_score = sum(rewards)
    print(f"[END] success={str(total_score >= 0.7).lower()} steps={len(rewards)} rewards={','.join(f'{r:.2f}' for r in rewards)}")

if __name__ == "__main__":
    asyncio.run(run_inference())