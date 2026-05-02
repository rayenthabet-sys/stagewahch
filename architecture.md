# Falla7 AI - System Architecture

This document describes the current, working agentic architecture for the **Falla7 AI** application (Tunisian Agricultural Assistant).

## 1. System Overview

The system architecture relies on a **parallel agentic workflow**, mixing different AI models to balance cost, speed, and capability.
- **Frontend**: A modularized Gradio web UI.
- **Backend**: FastAPI modular server (`main.py` -> `routes_audio.py` & `routes_image.py`).

## 2. Processing Pipeline

When a user submits audio (and optionally images):

### Step 1: Transcription (Groq)
The audio file is sent to Groq's high-speed `whisper-large-v3` model to transcribe the Tunisian dialect into text.

### Step 2: Vision Analysis (Gemini) *[If Image Provided]*
Images are sent to `gemini-2.0-flash` to detect olive diseases, nutritional deficiencies, or pests. The images are pre-loaded in memory using `PIL.Image.load()` to prevent Windows file-locking (`WinError 32`) during temporary file cleanup.

### Step 3: Gatekeeper (Agent 1 - Gemini)
**Model**: `gemini-2.0-flash`
The core input (and vision context, if any) is evaluated by the Gatekeeper to see if it is remotely related to agriculture, farming, weather, animals, or nature.
- **Valid Input**: The gatekeeper returns exactly `VALID` and the pipeline continues.
- **Invalid Input**: The gatekeeper *dynamically generates* a polite, friendly rejection message in Tunisian Darija. This skips the rest of the pipeline and answers the user immediately, avoiding hardcoded static responses.

### Step 4: Parallel Context Gathering (Agent 2 & Agent 3)
If the query is approved by the Gatekeeper, two agents run concurrently to fetch context:
- **Agent 2 - The Librarian (Internal Knowledge)**: A mock RAG system that cross-references the query with established agricultural data specifically for Tunisia (e.g., Chemlali olives, local pruning techniques).
- **Agent 3 - The Researcher (Tavily)**: A web search agent that browses for live, up-to-date information, news, and weather regarding Tunisian olive farming.

### Step 5: The Constructor (Agent 4 - Groq)
**Model**: `llama-3.3-70b-versatile` (via Groq API)
The Constructor takes all available context:
1. User transcription
2. Vision analysis
3. Internal PDF knowledge
4. Live web research

It then synthesizes a final, highly accurate, and helpful response entirely in friendly Tunisian Darija. Groq was chosen here to bypass Gemini quota limits and provide blazing-fast text generation.

### Step 6: Text-to-Speech (Edge TTS)
The final synthesized text (from either the Gatekeeper or the Constructor) is passed to `edge-tts`.
- **Voice**: `ar-TN-HediNeural` (Official Tunisian dialect voice).
- **Resilience**: The TTS is wrapped in a 3-try loop. Because Microsoft's Edge WebSocket servers frequently drop connections, this ensures the system reliably generates the audio file without throwing generic 500 Server Errors.
- Returns as base64 audio and decoded by the Gradio frontend to play automatically.

## 3. Notable Improvements Implemented
- **Modularization**: Refactored massive `backend.py` routing logic into `routes_audio` and `routes_image`, and split the heavy Gradio logic into `api_client.py` ensuring files stay under ~50 lines of code.
- **Windows File Handle Locks**: Resolved `WinError 32` crashes during temp file cleanup by manually closing PIL image handles in `routes_image.py`.
- **Quota Optimizations**: Shifted heavy text-generation pipelines off Gemini onto Groq's high-tier free endpoint (`llama-3.3-70b-versatile`) while preserving Gemini for complex tasks like zero-shot routing and multimodal vision.
- **Network Resilience**: Hardened TTS rendering with automated ping retry blocks.
