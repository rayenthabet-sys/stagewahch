import os
import uuid
import base64
import edge_tts
from fastapi import HTTPException
from config import groq_client

async def transcribe_with_groq(file_path: str) -> str:
    if not groq_client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is missing.")
    with open(file_path, "rb") as file:
        transcription = await groq_client.audio.transcriptions.create(
            file=(os.path.basename(file_path), file.read()),
            model="whisper-large-v3",
        )
        return transcription.text

import asyncio

async def generate_tts(text: str) -> str:
    """Generates TTS locally via edge-tts with retries, and returns base64 format."""
    output_path = f"tmp_{uuid.uuid4()}.mp3"
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            communicate = edge_tts.Communicate(text, "ar-TN-HediNeural")
            await communicate.save(output_path)
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
                return base64.b64encode(audio_bytes).decode('utf-8')
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"TTS attempt {attempt + 1} failed: {e}. Retrying in 2 seconds...")
            await asyncio.sleep(2)
        finally:
            if attempt == max_retries - 1 or os.path.exists(output_path):
                try:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                except:
                    pass
