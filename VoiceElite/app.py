# app.py
import gradio as gr
import whisper

# Load a small Whisper model (fast, works on CPU)
model = whisper.load_model("base")

def transcribe(audio, style):
    # Transcribe the uploaded audio
    result = model.transcribe(audio)
    text = result["text"]

    # Apply style transformation
    if style == "Funny":
        text = f"😂 Funny twist: {text}"
    elif style == "Romantic":
        text = f"❤️ Romantic version: {text}"
    elif style == "Professional":
        text = f"💼 Professional tone: {text}"
    else:
        text = f"✨ Original: {text}"

    return text

# Gradio interface
demo = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Upload or record your voice"),
        gr.Radio(["Original", "Funny", "Romantic", "Professional"], label="Choose Style")
    ],
    outputs=gr.Textbox(label="Your Message"),
    title="VoiceElite MVP",
    description="Upload or record your voice → get an AI-generated message in different styles."
)

if __name__ == "__main__":
    demo.launch()
