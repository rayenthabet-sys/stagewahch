from fastapi import FastAPI
import uvicorn
from routes_audio import router as audio_router
from routes_image import router as image_router

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Sujet 04: Tunisian Agricultural AI", description="Agentic Workflow for Tunisian Olive Farmers")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(audio_router)
app.include_router(image_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
