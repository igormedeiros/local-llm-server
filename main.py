from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
import logging
import json

app = FastAPI()
router = APIRouter(prefix="/v1")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to map the short name to the actual model name
def map_model_name(model_name_short: str) -> str:
    model_name_map = {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.1.Q4_K_L.gguf"
    }
    return model_name_map.get(model_name_short.lower(), model_name_short)

# Global variables for model and tokenizer
model = None
tokenizer = None

class GenerateRequest(BaseModel):
    prompts: list
    max_tokens: int
    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    stop: list

# Function to load API token from a JSON file
def load_api_token():
    try:
        with open('api_token.json', 'r') as file:
            data = json.load(file)
            return data['api_token']
    except Exception as e:
        logger.error(f"Error loading API token from JSON: {e}")
        return None

# Load API token from JSON file
api_token = load_api_token()

# New endpoint to initialize the model
@router.get("/init_model/{model_name_short}")
async def init_model(model_name_short: str):
    global model, tokenizer

    if model is None or tokenizer is None:
        model_name = map_model_name(model_name_short)
        try:
            logger.info(f"Initializing model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            logger.info("Model initialized successfully.")
            return {"message": f"Model {model_name} loaded successfully"}
        except Exception as e:
            logger.error(f"Error loading the model: {e}")
            raise HTTPException(status_code=500, detail="Model could not be loaded")
    else:
        return {"message": "Model already initialized"}

# The generate text endpoint now assumes the model is already loaded
@router.post("/{model_name_short}/completions")
async def generate_text(model_name_short: str, request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model is not initialized")

    inputs = tokenizer(request.prompts[0], return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=request.max_tokens,
                             temperature=request.temperature,
                             top_p=request.top_p,
                             frequency_penalty=request.frequency_penalty,
                             presence_penalty=request.presence_penalty,
                             stop_words=request.stop)
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info("Generated text successfully.")
    return {"generated_text": response_text}

@router.get("/health")
def health_check():
    return JSONResponse(content={"status": "ok"})

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1234)
