from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Inicializa o app FastAPI
app = FastAPI(title="API de Atendimento IA", version="1.0")

# Carrega o modelo leve (FLAN-T5-SMALL)
generator = pipeline("text2text-generation", model="google/flan-t5-small")

class Prompt(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "API online", "message": "Envie um POST para /chat com um prompt"}

@app.post("/chat")
def chat(data: Prompt):
    try:
        resposta = generator(data.prompt, max_length=200, num_return_sequences=1)
        texto = resposta[0]['generated_text']
        return {"response": texto}
    except Exception as e:
        return {"error": str(e)}
