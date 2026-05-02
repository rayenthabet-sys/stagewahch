import os
import asyncio
import uuid
import traceback
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from utils import transcribe_with_groq, generate_tts
from agents.gatekeeper import agent_1_3assas
from agents.librarian import agent_2_koutbi
from agents.researcher import agent_3_ba7at
from agents.constructor import agent_4_mou2allef

router = APIRouter()

@router.post("/api/process-audio")
async def process_audio_endpoint(file: UploadFile = File(...)):
    try:
        ext = file.filename.split('.')[-1]
        temp_audio_path = f"temp_in_{uuid.uuid4()}.{ext}"
        try:
            with open(temp_audio_path, "wb") as f:
                f.write(await file.read())
            transcription = await transcribe_with_groq(temp_audio_path)
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
        is_valid, gatekeeper_msg = await agent_1_3assas(transcription)
        if not is_valid:
            tts_b64 = await generate_tts(gatekeeper_msg)
            return {"query": transcription, "response_darija": gatekeeper_msg, "audio_base64": tts_b64, "sources": []}
            
        kb_data, web_data = await asyncio.gather(agent_2_koutbi(transcription), agent_3_ba7at(transcription))
        final_response = await agent_4_mou2allef(transcription, kb_data, web_data)
        tts_b64 = await generate_tts(final_response)
        
        sources = []
        if kb_data and "No specific" not in kb_data: sources.append("El Maktaba")
        if web_data and "No relevant" not in web_data and "error" not in web_data.lower(): sources.append("Al Internet")
        return {"query": transcription, "response_darija": final_response, "sources": sources, "audio_base64": tts_b64}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})
