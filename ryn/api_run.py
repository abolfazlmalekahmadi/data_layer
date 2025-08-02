# api_run.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.get_api:app", host="0.0.0.0", port=1235, reload=True)