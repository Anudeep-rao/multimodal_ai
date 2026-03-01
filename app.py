import streamlit as st
import os
import tempfile
from PIL import Image
import pytesseract
from gtts import gTTS
from backend.speechtotext import process_voice
from transformers import MarianMTModel, MarianTokenizer
import torch

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(page_title="Multimodal Vernacular AI", layout="wide")

# Set Tesseract Path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ==============================
# LANGUAGE MAP (LOCAL MODELS)
# ==============================

language_models = {
    "Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "Telugu": "Helsinki-NLP/opus-mt-en-te",
    "Tamil": "Helsinki-NLP/opus-mt-en-ta",
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "German": "Helsinki-NLP/opus-mt-en-de",
    "Spanish": "Helsinki-NLP/opus-mt-en-es"
}

languages = list(language_models.keys())

st.markdown("## 🩺 Multimodal Vernacular Prescription AI")

target_language = st.selectbox("Select Language", languages)

# ==============================
# LOAD LOCAL MODEL (CACHED)
# ==============================

@st.cache_resource(show_spinner=False)
def load_model(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

# ==============================
# LOCAL TRANSLATION FUNCTION
# ==============================

def local_translate(text, target_language):
    if not text or not text.strip():
        return "No valid text detected."

    try:
        model_name = language_models[target_language]
        tokenizer, model = load_model(model_name)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )

        result = tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )

        return result[0]

    except Exception as e:
        return f"Translation failed: {str(e)}"

# ==============================
# AUDIO GENERATION
# ==============================

def generate_audio(text):
    tts = gTTS(text)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

# ==============================
# TABS
# ==============================

tab1, tab2, tab3 = st.tabs([
    "📝 Text Input",
    "🖼 Image Input",
    "🎤 Voice Input"
])

# =========================================================
# TEXT INPUT
# =========================================================

with tab1:
    text_input = st.text_area("Enter Prescription", key="text_input")

    if st.button("Process Text", key="text_btn"):
        if text_input:
            with st.spinner("Translating locally..."):
                translated = local_translate(text_input, target_language)

            st.success("Translated Text")
            st.write(translated)

            audio_file = generate_audio(translated)
            st.audio(audio_file)

# =========================================================
# IMAGE INPUT (OCR)
# =========================================================

with tab2:
    uploaded_image = st.file_uploader(
        "Upload Prescription Image",
        type=["png", "jpg", "jpeg"],
        key="image_uploader"
    )

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image)

        if st.button("Process Image", key="image_btn"):
            with st.spinner("Extracting text..."):
                extracted_text = pytesseract.image_to_string(image)

            st.info("Extracted Text")
            st.write(extracted_text)

            with st.spinner("Translating locally..."):
                translated = local_translate(extracted_text, target_language)

            st.success("Translated Text")
            st.write(translated)

            audio_file = generate_audio(translated)
            st.audio(audio_file)

# =========================================================
# VOICE INPUT
# =========================================================

with tab3:
    uploaded_audio = st.file_uploader(
        "Upload Voice File",
        type=["mp3", "wav", "m4a"],
        key="audio_uploader"
    )

    if uploaded_audio:
        st.audio(uploaded_audio)

        if st.button("Process Voice", key="voice_btn"):
            try:
                transcript_text = process_voice(uploaded_audio)

                st.info("Transcribed Text")
                st.write(transcript_text)

                with st.spinner("Translating locally..."):
                    translated = local_translate(transcript_text, target_language)

                st.success("Translated Text")
                st.write(translated)

                audio_file = generate_audio(translated)
                st.audio(audio_file)

            except Exception as e:
                st.error(f"Voice processing failed: {str(e)}")