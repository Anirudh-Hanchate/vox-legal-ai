from fastapi import FastAPI
import uvicorn

app = FastAPI(title="volks-legal-ai")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "volks-legal-ai API is ready"}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
