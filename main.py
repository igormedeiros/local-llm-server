from fastapi import FastAPI, Request
from transformers import pipeline

app = FastAPI()

model = pipeline('text-generation', model='YOUR_MODEL_HERE')

@app.post("/generate")
async def generate_text(request: Request):
    data = await request.json()
    text = data.get("text")
    generated_text = model(text)
    return {"generated_text": generated_text}
