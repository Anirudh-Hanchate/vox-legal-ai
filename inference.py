# filename: inference.py

import os
import json
import asyncio
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# FIX: Corrected import path
from server.env.env import VolksLegalEnv
from server.env.models import LegalAction

# FIX: Respect injected environment variables
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_ID = os.getenv("TASK_ID", "task_ramesh_kn")

SYSTEM_PROMPT = """You are Volks Legal AI. Respond ONLY with a JSON object.
Keys: "action_type" and "value".
Sequence: 1. classify_case, 2. set_priority, 3. generate_guidance, 4. assign_lawyer.
Lawyers:
- (property, bengaluru): Adv. Shankar (Property Specialist)
- (finance, mumbai): Adv. Mishra (Financial Crimes)
- (civil, delhi): Adv. Gupta (Civil Litigation)"""

async def run_inference():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = VolksLegalEnv(task_id=TASK_ID)
    
    # REQUIRED STDOUT FORMAT
    print(f"[START] task={TASK_ID} env=volks-legal-ai model={MODEL_NAME}", flush=True)
    
    # FIX: Added await
    obs = await env.reset()
    rewards = []
    steps_taken = 0
    success = False
    
    for step in range(1, 7):
        prompt = f"Case: {obs.case_text}\nLang: {obs.language}\nLoc: {obs.location}\nStatus: {obs.model_dump_json()}"
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            act_dict = json.loads(response.choices[0].message.content)
            # Handle potential nested JSON from LLM
            if "action" in act_dict: act_dict = act_dict["action"]
            
            action = LegalAction(**act_dict)
            
            # FIX: Added await
            obs, reward, done, info = await env.step(action)
            rewards.append(float(reward))
            steps_taken = step
            
            # REQUIRED STDOUT FORMAT
            print(f"[STEP] step={step} action={act_dict['action_type']} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            
            if done: break
        except Exception as e:
            print(f"[STEP] step={step} action=error reward=0.00 done=true error={str(e)}", flush=True)
            break

    total_score = sum(rewards)
    success = total_score >= 0.7
    # REQUIRED STDOUT FORMAT
    rw_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps_taken} rewards={rw_str}", flush=True)

if __name__ == "__main__":
    asyncio.run(run_inference())