from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
import pydantic

app = FastAPI()

# Pydantic class to validate request body.
class GenerateRequest(pydantic.BaseModel):
    prompts: list
    max_tokens: int
    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    stop: list

# Maps a short model name to its full name.
def map_model_name(model_name_short: str) -> str:
    model_mapping = {
        "mistral": "mistralai/Mistral-7B-v0.1"
    }
    return model_mapping.get(model_name_short.lower(), model_name_short)

# Loads the tokenizer and model based on the provided name.
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Endpoint to generate text based on a specific model.
@app.post("/v1/{model_name_short}/completions")
async def generate_text(model_name_short: str, request: GenerateRequest):
    model_name = map_model_name(model_name_short)
    tokenizer, model = load_model(model_name)
    inputs = tokenizer(request.prompts[0], return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=request.max_tokens, temperature=request.temperature, top_p=request.top_p)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

# Health check endpoint.
@app.get("/health")
def health_check():
    # Here you could add additional checks, like database connection, etc.
    return JSONResponse(content={"status": "ok"})
