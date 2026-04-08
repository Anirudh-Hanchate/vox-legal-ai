# filename: server/app.py
import uvicorn
import os
from openenv.core.env_server.http_server import create_app

# FIX: Absolute imports for server reliability
from server.env.env import VolksLegalEnv
from server.env.models import LegalAction, LegalObservation

app = create_app(
    VolksLegalEnv,
    LegalAction,
    LegalObservation,
    env_name="volks-legal-ai",
    max_concurrent_envs=1,
)

def main():
    # HF Spaces expects port 7860
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()