"""doc
"""

from fastapi import FastAPI
import uvicorn


app = FastAPI()


@app.post("/predict")
def inference():
    data = "Hello"
    return data


if __name__ == "__main__":
    uvicorn.run("main:app", port=3030, reload=True)
