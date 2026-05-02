import os
import asyncio
import uuid
import traceback
import PIL.Image
from typing import List
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from utils import transcribe_with_groq, generate_tts
from config import gemini_client
from agents.gatekeeper import agent_1_3assas
from agents.librarian import agent_2_koutbi
from agents.researcher import agent_3_ba7at
from agents.constructor import agent_4_mou2allef

router = APIRouter()

@router.post("/api/process-image")
async def process_image_endpoint(voice_query: UploadFile = File(...), images: List[UploadFile] = File(...)):
    try:
        ext_audio = voice_query.filename.split('.')[-1]
        temp_audio_path = f"temp_in_{uuid.uuid4()}.{ext_audio}"
        
        img_contents = []
        temp_img_paths = []
        try:
            with open(temp_audio_path, "wb") as f: f.write(await voice_query.read())
            transcription = await transcribe_with_groq(temp_audio_path)
            
            for image in images:
                ext_img = image.filename.split('.')[-1]
                temp_img_path = f"temp_in_{uuid.uuid4()}.{ext_img}"
                with open(temp_img_path, "wb") as f: f.write(await image.read())
                temp_img_paths.append(temp_img_path)
                img = PIL.Image.open(temp_img_path)
                img.load()  # Force read pixels into memory so the file handle is released
                img_contents.append(img)
                
            prompt = ["Analyze these plants. Focus on olive diseases, nutritional deficiencies, or pest issues."] + img_contents
            vision_resp = await gemini_client.aio.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt
            )
            vision_analysis = vision_resp.text
        finally:
            # Close all PIL images first to release file handles (Windows locks)
            for img in img_contents:
                try: img.close()
                except: pass
            if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
            for p in temp_img_paths:
                if os.path.exists(p): os.remove(p)

        combined = f"Farmer Speech: {transcription}. Vision Info: {vision_analysis}"
        is_valid, gatekeeper_msg = await agent_1_3assas(combined)
        if not is_valid:
            return {"query": transcription, "vision_analysis": vision_analysis, "response_darija": gatekeeper_msg, "audio_base64": await generate_tts(gatekeeper_msg), "sources": []}
            
        kb_data, web_data = await asyncio.gather(agent_2_koutbi(combined), agent_3_ba7at(combined))
        final_response = await agent_4_mou2allef(transcription, kb_data, web_data, vision_data=vision_analysis)
        
        sources = ["Vision"]
        if kb_data and "No specific" not in kb_data: sources.append("El Maktaba")
        if web_data and "No relevant" not in web_data and "error" not in web_data.lower(): sources.append("Al Internet")
        return {"query": transcription, "vision_analysis": vision_analysis, "response_darija": final_response, "audio_base64": await generate_tts(final_response), "sources": sources}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})
