import os
import asyncio
import base64
import uuid
import tempfile
import requests
import PIL.Image

from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from dotenv import load_dotenv

from google import genai
from groq import AsyncGroq
import edge_tts

# Load Environment Variables
load_dotenv(os.path.join(os.path.dirname(__file__), 'agents', '.env'))

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Sujet 04: Tunisian Agricultural AI", description="Agentic Workflow for Tunisian Olive Farmers")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure APIs and Clients
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

MOCK_KB = """
[El Maktaba - Tunisian Olive PDF Data]:
- Best varieties in Tunisia are Chemlali in the south (drought resistant) and Chetoui in the north.
- Harvesting typically starts in November when color turns purple/black.
- Common diseases: Olive fruit fly (Dacus oleae) and Peacock spot (Spilocaea oleagina).
- Watering requirement: Usually rain-fed, but supplementary irrigation in summer boosts yield by 30%.
- Pruning (Zebra) should clear the center of the tree to allow wind and sunlight to enter.
"""

# ================= AGENTS =================

async def agent_1_3assas(query: str) -> bool:
    """Agent 1 (Gatekeeper): Checks if query is about Tunisian agriculture/olives."""
    prompt = f"""
    You are '3assas', a strict gatekeeper for an agricultural app in Tunisia.
    Determine if the query specifically relates to agriculture, farming, crops, livestock, or olive trees.
    Reply with ONLY 'YES' or 'NO'. Do not include extra text.
    Query: {query}
    """
    try:
        response = await groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        text = response.choices[0].message.content.strip().upper()
        return "YES" in text
    except Exception as e:
        print("3assas Error:", e)
        return True # Default to allow if service is unstable

async def agent_2_koutbi(query: str) -> str:
    """Agent 2 (Librarian): Mock RAG"""
    await asyncio.sleep(0.5) # simulate Vector DB latency
    if any(word in query.lower() for word in ["zaytoun", "olive", "chajra", "zitoun", "saba", "harvest", "disease", "pruning", "taks", "weather", "traba", "soil"]):
        return MOCK_KB
    return "No specific internal data found in PDFs."

async def agent_3_ba7at(query: str) -> str:
    """Agent 3 (Researcher): Tavily Web Search"""
    def _search():
        if not TAVILY_API_KEY:
            return "No Tavily API key."
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY, 
                    "query": query + " Tunisia olive farming recent news or weather", 
                    "search_depth": "basic", 
                    "include_answer": True
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return data.get("answer", "No relevant external data found.")
        except requests.exceptions.Timeout:
            return "Search API timed out."
        except Exception as e:
            return f"Search API error: {str(e)}"
            
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _search)
    except Exception as e:
        return f"Async Ba7at Error: {e}"

async def agent_4_mou2allef(query: str, kb_data: str, web_data: str, vision_data: str = "") -> str:
    """Agent 4 (Constructor): Synthesizer using Gemini."""
    prompt = f"""
    You are 'Mou2allef', a highly knowledgeable and friendly agricultural assistant for Tunisian olive farmers.
    You MUST speak in friendly Tunisian Darija using Latin characters (Arabizi) or Arabic script depending on typical farmer preferences (Arabizi is great). 
    
    Synthesize an informative, encouraging, and accurate response to the user's query utilizing the provided sources.
    - If you use facts from the internal dataset, cite "El Maktaba".
    - If you use facts from the web, cite "Al Internet".
    - If the user submitted a picture, explain the vision analysis clearly.
    
    User Query: {query}
    Vision Data (if any): {vision_data}
    Internal Knowledge (El Maktaba): {kb_data}
    Web Data (Al Internet): {web_data}
    
    Make sure your response sounds natural in Tunisian Darija! (e.g., "Asslema khouya l felle7...", "barcha", "behi", "chwaya")
    """
    try:
        response = await groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sama7ni, sar mochkel fel systéme: {e}"

# ================= UTILS =================

async def transcribe_with_groq(file_path: str) -> str:
    if not groq_client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is missing.")
    with open(file_path, "rb") as file:
        transcription = await groq_client.audio.transcriptions.create(
            file=(os.path.basename(file_path), file.read()),
            model="whisper-large-v3",
        )
        return transcription.text

async def generate_tts(text: str) -> str:
    """Generates TTS locally via edge-tts and returns base64 format."""
    output_path = f"tmp_{uuid.uuid4()}.mp3"
    try:
        # ar-TN-HemsNeural is the official Tunisian Arabic voice for Azure/Edge TTS
        communicate = edge_tts.Communicate(text, "ar-TN-HediNeural")
        await communicate.save(output_path)
        with open(output_path, "rb") as f:
            audio_bytes = f.read()
            return base64.b64encode(audio_bytes).decode('utf-8')
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)

# ================= ENDPOINTS =================

@app.post("/api/process-audio")
async def process_audio_endpoint(file: UploadFile = File(...)):
    """Receives audio file, runs the full Agentic Pipeline, returns JSON."""
    ext = file.filename.split('.')[-1]
    temp_audio_path = f"temp_in_{uuid.uuid4()}.{ext}"
    
    try:
        with open(temp_audio_path, "wb") as f:
            f.write(await file.read())
            
        # Step 1: Speech to Text (ASR)
        transcription = await transcribe_with_groq(temp_audio_path)
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
    # Step 2: Agent 1 - Gatekeeper
    is_valid = await agent_1_3assas(transcription)
    if not is_valid:
        msg = "Sama7ni, ena n3awnek ken fil zaytoun wil felle7a fil tounes."
        tts_b64 = await generate_tts(msg)
        return {
            "query": transcription, 
            "response_darija": msg, 
            "audio_base64": tts_b64, 
            "sources": []
        }
        
    # Step 3: Agent 2 & 3 - Parallel Execution for Speed
    kb_data, web_data = await asyncio.gather(
        agent_2_koutbi(transcription),
        agent_3_ba7at(transcription)
    )
    
    # Step 4: Agent 4 - Synthesis
    final_response = await agent_4_mou2allef(transcription, kb_data, web_data)
    
    # Generate Output Audio
    tts_b64 = await generate_tts(final_response)
    
    # Identify sources used
    sources = []
    if kb_data and "No specific" not in kb_data:
        sources.append("El Maktaba")
    if web_data and "No relevant" not in web_data and "error" not in web_data.lower():
        sources.append("Al Internet")
        
    return {
        "query": transcription,
        "response_darija": final_response,
        "sources": sources,
        "audio_base64": tts_b64
    }


@app.post("/api/process-image")
async def process_image_endpoint(voice_query: UploadFile = File(...), image: UploadFile = File(...)):
    """Handles an image upload and an audio query for Multimodal reasoning."""
    ext_audio = voice_query.filename.split('.')[-1]
    ext_img = image.filename.split('.')[-1]
    temp_audio_path = f"temp_in_{uuid.uuid4()}.{ext_audio}"
    temp_img_path = f"temp_in_{uuid.uuid4()}.{ext_img}"
    
    try:
        with open(temp_audio_path, "wb") as f:
            f.write(await voice_query.read())
        transcription = await transcribe_with_groq(temp_audio_path)
        
        with open(temp_img_path, "wb") as f:
            f.write(await image.read())
            
        img = PIL.Image.open(temp_img_path)
        img.load()  # Force read pixels into memory so the file handle is released
        
        # Multimodal Vision analysis
        vision_resp = await gemini_client.aio.models.generate_content(
            model='gemini-2.0-flash',
            contents=["Analyze this image of a plant/leaf. Focus on identifying olive diseases, nutritional deficiencies, or pest issues.", img]
        )
        vision_analysis = vision_resp.text
    finally:
        try: img.close()
        except: pass
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

    combined_query = f"Farmer Speech: {transcription}. Background Vision Info: {vision_analysis}"
    
    # Agent 1 - Gatekeeper
    is_valid = await agent_1_3assas(combined_query)
    
    if not is_valid:
        msg = "Sama7ni, ena n3awnek ken fil zaytoun wil felle7a wil chjar."
        tts_b64 = await generate_tts(msg)
        return {
            "query": transcription, 
            "vision_analysis": vision_analysis, 
            "response_darija": msg, 
            "audio_base64": tts_b64,
            "sources": []
        }
        
    # Parallel Agents
    kb_data, web_data = await asyncio.gather(
        agent_2_koutbi(combined_query),
        agent_3_ba7at(combined_query)
    )
    
    # Constructor
    final_response = await agent_4_mou2allef(transcription, kb_data, web_data, vision_data=vision_analysis)
    
    # TTS
    tts_b64 = await generate_tts(final_response)
    
    sources = ["Vision"]
    if kb_data and "No specific" not in kb_data:
        sources.append("El Maktaba")
    if web_data and "No relevant" not in web_data and "error" not in web_data.lower():
        sources.append("Al Internet")
        
    return {
        "query": transcription,
        "vision_analysis": vision_analysis,
        "response_darija": final_response,
        "audio_base64": tts_b64,
        "sources": sources
    }

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
