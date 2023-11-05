from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel

app = FastAPI()
router = APIRouter(prefix="/v1")

# Model and Tokenizer are loaded once when the server starts
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

class GenerateRequest(BaseModel):
    prompts: list
    max_tokens: int
    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    stop: list

@router.post("/{model_name_short}/completions")
async def generate_text(model_name_short: str, request: GenerateRequest):
    inputs = tokenizer(request.prompts[0], return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=request.max_tokens,
                             temperature=request.temperature,
                             top_p=request.top_p)
    return {"generated_text": tokenizer.decode(outputs[0], skip_special_tokens=True)}

@router.get("/health")
def health_check():
    return JSONResponse(content={"status": "ok"})

app.include_router(router)
