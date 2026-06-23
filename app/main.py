from fastapi import FastAPI, UploadFile, File
from app.api.routes import router
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from app.agents.vision_agent import analyze_image
from app.agents.classification_agent import classification_agent
app = FastAPI(
    title="Multimodal Image Analysis API",
    version="1.0.0",
    description="AI-powered image analysis using Groq + Llama 4"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api")

@app.get("/")
def home():
    return {
        "message": "Multimodal Image Analysis API Running"
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": "1.0.0"
    }

@app.post("/api/analyze-image")
def analyze(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"

    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_image(temp_file)   

    result = classification_agent(result)

    return {
        "result": result
    }

    