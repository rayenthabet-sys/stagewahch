import os
import requests
import base64
import tempfile

API_URL = "http://localhost:8000"

def submit_query(audio_path, image_files, chat_history):
    """
    audio_path: filepath from gr.Audio (microphone or uploaded)
    image_files: list of filepaths from gr.File
    chat_history: list of {"role":..., "content":...} dicts (Gradio 6 messages format)
    """
    if not audio_path:
        chat_history = chat_history + [
            {"role": "user", "content": "(no audio provided)"},
            {"role": "assistant", "content": "Sama7ni, lazem t7ot tsjil sowti (audio). Isna3 wela arfa3 fichier sowti!"},
        ]
        return chat_history, None, None, None

    # Build user-side display
    user_parts = "🎙️ Audio submitted"
    if image_files:
        user_parts += f" + {len(image_files)} image(s)"

    chat_history = chat_history + [{"role": "user", "content": user_parts}]

    # Determine which API to call
    try:
        # Collect image paths
        img_paths = []
        if image_files:
            for f in image_files:
                # gr.File returns filepath string in Gradio 6
                path = f if isinstance(f, str) else getattr(f, "name", str(f))
                img_paths.append(path)

        if img_paths:
            # --- Image + Audio endpoint ---
            with open(audio_path, "rb") as f_audio:
                audio_bytes = f_audio.read()

            post_files = [
                ("voice_query", (os.path.basename(audio_path), audio_bytes, "audio/wav")),
            ]
            for img_p in img_paths:
                with open(img_p, "rb") as f_img:
                    post_files.append(("images", (os.path.basename(img_p), f_img.read(), "image/jpeg")))

            response = requests.post(f"{API_URL}/api/process-image", files=post_files, timeout=120)
        else:
            # --- Audio-only endpoint ---
            with open(audio_path, "rb") as f_audio:
                audio_bytes = f_audio.read()

            post_files = [
                ("file", (os.path.basename(audio_path), audio_bytes, "audio/wav")),
            ]
            response = requests.post(f"{API_URL}/api/process-audio", files=post_files, timeout=120)

        if response.status_code == 200:
            data = response.json()

            # Decode TTS audio
            audio_b64 = data.get("audio_base64", "")
            resp_audio_path = None
            if audio_b64:
                audio_decoded = base64.b64decode(audio_b64)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tmp.write(audio_decoded)
                tmp.close()
                resp_audio_path = tmp.name

            # Build bot response text
            reply = data.get("response_darija", "No response.")
            sources = data.get("sources", [])
            vision = data.get("vision_analysis", "")

            if sources:
                reply += f"\n\n📚 **Sources:** {', '.join(sources)}"
            if vision:
                reply += f"\n\n🔍 **Vision:** {vision}"

            chat_history = chat_history + [{"role": "assistant", "content": reply}]
            return chat_history, resp_audio_path, None, None
        else:
            err = f"❌ Backend error {response.status_code}: {response.text[:300]}"
            chat_history = chat_history + [{"role": "assistant", "content": err}]
            return chat_history, None, None, None

    except Exception as e:
        err = f"❌ Connection error: {e}"
        chat_history = chat_history + [{"role": "assistant", "content": err}]
        return chat_history, None, None, None
