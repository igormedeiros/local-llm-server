from fastapi import FastAPI, Request, Query
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

def load_model(model_name):
    if model_name == "Mistral":
        model_path = "/caminho/para/Mistral-7B-Instruct-v0.1-GGUF"
    else:
        model_path = model_name  # assume que é um modelo da Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

@app.get("/v1/{model_name}")
async def generate_text(model_name: str, prompt: str = Query(...)):
    tokenizer, model = load_model(model_name)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}
