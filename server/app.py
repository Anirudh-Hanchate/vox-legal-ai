import uvicorn
from openenv.core.env_server.http_server import create_app
from env.env import VolksLegalEnv
from env.models import LegalAction, LegalObservation

app = create_app(
    VolksLegalEnv,
    LegalAction,
    LegalObservation,
    env_name="volks-legal-ai",
    max_concurrent_envs=10,
)

def main(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run("server.app:app", host=host, port=port)

if __name__ == "__main__":
    main()
