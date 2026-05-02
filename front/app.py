import gradio as gr
from api_client import submit_query

# ── UI ──────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Falla7 AI") as demo:
    gr.HTML("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <h1 style="margin:0; font-size:2rem;">🌿 Falla7 AI</h1>
        <p style="color:#888; margin:6px 0 0 0;">Your Tunisian Agricultural Assistant — Record your voice, attach plant photos, and get expert advice</p>
    </div>
    """)

    chatbot = gr.Chatbot(
        label="Chat",
        height=420,
        show_label=False,
    )

    gr.HTML("<hr style='margin:8px 0'>")

    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="🎙️ Record or upload audio (required)",
            scale=2,
        )
        image_input = gr.File(
            file_count="multiple",
            file_types=["image"],
            label="📷 Upload plant images (optional, multiple OK)",
            scale=2,
        )

    submit_btn = gr.Button("🚀 Send to Falla7 AI", variant="primary", size="lg")

    audio_output = gr.Audio(label="🔊 Bot Voice Response", autoplay=True)

    # Wire up callback
    submit_btn.click(
        submit_query,
        inputs=[audio_input, image_input, chatbot],
        outputs=[chatbot, audio_output, audio_input, image_input],
    )

if __name__ == "__main__":
    demo.launch(server_port=7862, share=False)
