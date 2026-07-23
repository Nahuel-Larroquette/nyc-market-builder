from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "NYC Market Builder API is running"}