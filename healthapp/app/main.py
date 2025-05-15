from fastapi import FastAPI

app = FastAPI(title="Disease Predictor API")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Disease Predictor API"}
