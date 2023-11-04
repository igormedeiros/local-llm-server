from fastapi import FastAPI, Request, Body
from transformers import AutoTokenizer, AutoModelForCausalLM
import pydantic

app = FastAPI()

class GenerateRequest(pydantic.BaseModel):
    prompts: list
    max_tokens: int
    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    stop: list

def load_model(model_name="mistralai/Mistral-7B-v0.1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


@app.post("/v1/{model_name}")
async def generate_text(model_name: str, request: GenerateRequest):
    tokenizer, model = load_model(model_name)
    inputs = tokenizer(request.prompts[0], return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=request.max_tokens, temperature=request.temperature, top_p=request.top_p)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}