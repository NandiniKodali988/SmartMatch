from fastapi import FastAPI
from src.pipeline.run_pipeline_updated import run_pipeline

app = FastAPI()


@app.get("/")
def root():
    return {"message": "SmartMatch API running"}


@app.get("/match")
def match(text: str):

    results = run_pipeline(text)

    # handle different pipeline outputs safely
    if isinstance(results, list):
        return {"results": results}

    return {"output": results}