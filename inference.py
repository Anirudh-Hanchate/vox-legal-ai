import os
import json
import asyncio
from openai import OpenAI
from env.env import VolksLegalEnv
from env.models import LegalAction

API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_ID = os.getenv("TASK_ID", "task_ramesh_kn")

SYSTEM_PROMPT = """You are Volks Legal AI. Respond ONLY with a JSON object.
Actions: 
- {"action_type": "classify_case", "value": "property|finance|civil|criminal"}
- {"action_type": "set_priority", "value": "low|medium|high"}
- {"action_type": "generate_guidance", "value": "Step 1...\\nStep 2...\\nStep 3..."} (In user's language)
- {"action_type": "assign_lawyer", "value": "Adv. Name"}

Lawyers:
- (property, bengaluru): Adv. Shankar (Property Specialist)
- (finance, mumbai): Adv. Mishra (Financial Crimes)
- (civil, delhi): Adv. Gupta (Civil Litigation)
"""

async def run_inference():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = VolksLegalEnv(task_id=TASK_ID)
    
    print(f"[START] task={TASK_ID} env=volks-legal-ai model={MODEL_NAME}")
    
    obs = await env.reset()
    rewards = []
    
    for step in range(1, 7):
        prompt = f"Case: {obs.case_text}\nLang: {obs.language}\nLoc: {obs.location}\nStatus: {obs.model_dump_json()}"
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        act_dict = json.loads(response.choices[0].message.content)
        action = LegalAction(**act_dict)
        
        obs, reward, done, info = await env.step(action)
        rewards.append(reward)
        
        print(f"[STEP] step={step} action={act_dict['action_type']} reward={reward:.2f} done={str(done).lower()} error=null")
        
        if done: break

    total_score = sum(rewards)
    print(f"[END] success={str(total_score >= 0.7).lower()} steps={len(rewards)} rewards={','.join(f'{r:.2f}' for r in rewards)}")

if __name__ == "__main__":
    asyncio.run(run_inference())