from fastapi.testclient import TestClient
from server.app import app
import traceback

client = TestClient(app)
try:
    print("Sending POST /reset...")
    resp = client.post("/reset")
    print("RESET RESPONSE:", resp.status_code, resp.text)
except Exception as e:
    traceback.print_exc()
