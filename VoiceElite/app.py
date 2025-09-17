import gradio as gr
import whisper
import requests
import os

# Load Whisper model locally
model = whisper.load_model("base")

# Hugging Face Inference API (free tier, need your HF token)
HF_TOKEN = os.getenv("HF_TOKEN")  # set in Hugging Face "Settings > Secrets"
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_hf_model(prompt):
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"⚠️ Error from HF API: {response.text}"

def rewrite_message(original_text, style):
    prompt = f"Rewrite this message in a {style} style:\n\n{original_text}\n\nRewritten:"
    return query_hf_model(prompt)

def transcribe_and_rewrite(audio, style):
    if audio is None:
        return "Please upload or record your voice."
    result = model.transcribe(audio)
    text = result["text"]
    rewritten = rewrite_message(text, style)
    return rewritten

# Gradio UI
iface = gr.Interface(
    fn=transcribe_and_rewrite,
    inputs=[
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="🎤 Record or Upload"),
        gr.Radio(["Original", "Funny", "Romantic", "Professional"], label="✨ Choose Style")
    ],
    outputs=gr.Textbox(label="📝 AI Message (Copy & Use)"),
    title="VoiceElite MVP",
    description="Upload or record your voice. Whisper transcribes it, then AI rewrites it in the style you choose."
)

iface.launch()

