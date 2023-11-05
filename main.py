import logging
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
router = APIRouter(prefix="/v1")

# Initialize variables to None
tokenizer = None
model = None

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
    if tokenizer is None or model is None:
        logger.error("Model or tokenizer not loaded.")
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    
    logger.info("Generating text...")
    inputs = tokenizer(request.prompts[0], return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=request.max_tokens,
                             temperature=request.temperature,
                             top_p=request.top_p)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info("Text generated.")
    return {"generated_text": generated_text}

@router.get("/health")
def health_check():
    return JSONResponse(content={"status": "Model is loaded and service is healthy." if model and tokenizer else "Model is not loaded yet."})

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    logger.info("Loading model, this might take a while...")
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    logger.info("Model loaded.")

app.include_router(router)