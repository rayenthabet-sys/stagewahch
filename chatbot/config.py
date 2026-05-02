import os
from dotenv import load_dotenv
from google import genai
from groq import AsyncGroq

load_dotenv(os.path.join(os.path.dirname(__file__), 'agents', '.env'))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
