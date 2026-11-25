"""
Real-Time Voice Translation - website
Speech (Whisper) -> translation (my fine-tuned mT5-small, 12 languages) -> speech (gTTS)

Run locally:  python app.py   then open the address it prints (http://127.0.0.1:7860)
"""
import os
import tempfile
import uuid

import gradio as gr
import torch
from faster_whisper import WhisperModel
from gtts import gTTS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_ID = "Gunal13/mt5-small-11lang"
AUTO = "Detect automatically"

# name: (Whisper language code, gTTS language code)
LANGUAGES = {
    "English":   ("en", "en"),
    "Hindi":     ("hi", "hi"),
    "Marathi":   ("mr", "mr"),
    "Tamil":     ("ta", "ta"),
    "Kannada":   ("kn", "kn"),
    "Malayalam": ("ml", "ml"),
    "Telugu":    ("te", "te"),
    "Chinese":   ("zh", "zh-CN"),
    "Arabic":    ("ar", "ar"),
    "Russian":   ("ru", "ru"),
    "Spanish":   ("es", "es"),
    "Afrikaans": ("af", "af"),
}

# Scripts for languages that don't use English letters
SCRIPTS = {
    "Hindi":     ("\u0900", "\u097F"),
    "Marathi":   ("\u0900", "\u097F"),
    "Tamil":     ("\u0B80", "\u0BFF"),
    "Kannada":   ("\u0C80", "\u0CFF"),
    "Malayalam": ("\u0D00", "\u0D7F"),
    "Telugu":    ("\u0C00", "\u0C7F"),
    "Chinese":   ("\u4E00", "\u9FFF"),
    "Arabic":    ("\u0600", "\u06FF"),
    "Russian":   ("\u0400", "\u04FF"),
}

WHISPER_TO_NAME = {codes[0]: name for name, codes in LANGUAGES.items()}

# ---------- Models (loaded once when the app starts) ----------
print("Loading translation model...")
tok = AutoTokenizer.from_pretrained("google/mt5-small")   # same tokenizer the model was trained with
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, tie_word_embeddings=False)
model.set_input_embeddings(model.get_input_embeddings())   # encoder + decoder use the trained 'shared' table
model.eval()
print("Encoder/decoder use trained table:",
      model.encoder.embed_tokens.weight.data_ptr() == model.shared.weight.data_ptr())
print("Output table kept separate:",
      model.lm_head.weight.data_ptr() != model.shared.weight.data_ptr())

_test = tok("translate English to Spanish: Where is the train station?", return_tensors="pt")
with torch.no_grad():
    _out = model.generate(**_test, max_new_tokens=30, num_beams=4)
print("Self-test raw output:", tok.decode(_out[0], skip_special_tokens=False))

print("Loading speech recognition model...")
whisper = WhisperModel("small", device="cpu", compute_type="int8")
print("Ready.")


# ---------- Translation ----------
def _generate(text, src, dst):
    prompt = f"translate {src} to {dst}: {text}"
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=128, num_beams=4, no_repeat_ngram_size=3)
    return tok.decode(out[0], skip_special_tokens=True)


def translate(text, src, dst):
    """Returns (translation, english_step). english_step is None unless we pivoted."""
    if src == "English" or dst == "English":
        return _generate(text, src, dst), None
    english = _generate(text, src, "English")      # the model only learned X <-> English
    return _generate(english, "English", dst), english


# ---------- Speech ----------
def transcribe(audio_path, src_choice):
    """Speech to text. Returns (language name, text)."""
    if src_choice == AUTO:
        segments, info = whisper.transcribe(audio_path, beam_size=5, vad_filter=True)
        probs = dict(info.all_language_probs or [])
        # Whisper knows ~100 languages; pick the most likely of OUR 12
        code = max(WHISPER_TO_NAME, key=lambda c: probs.get(c, 0.0))
        if code != info.language:
            segments, info = whisper.transcribe(audio_path, language=code,
                                                beam_size=5, vad_filter=True)
    else:
        code = LANGUAGES[src_choice][0]
        segments, info = whisper.transcribe(audio_path, language=code,
                                            beam_size=5, vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return WHISPER_TO_NAME[code], text


def speak(text, lang):
    """Text to an MP3 file the browser can play."""
    path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.mp3")
    gTTS(text=text, lang=LANGUAGES[lang][1]).save(path)
    return path


# ---------- What the Translate button does ----------
def run(audio_path, typed, src_choice, dst):
    typed = (typed or "").strip()
    if not audio_path and not typed:
        raise gr.Error("Record a sentence or type one first.")

    if audio_path:
        src, text = transcribe(audio_path, src_choice)
        if not text:
            raise gr.Error("No speech was detected. Try again, a little closer to the microphone.")
    else:
        if src_choice == AUTO:
            raise gr.Error("Choose the language you typed in.")
        src, text = src_choice, typed
        rng = SCRIPTS.get(src)
        if rng and not any(rng[0] <= ch <= rng[1] for ch in text):
            raise gr.Error(f"Type {src} in its own script, or use the microphone. "
                           "Text written in English letters (like 'sapitiya') can't be translated.")

    if src == dst:
        raise gr.Error(f"That was {src}. Choose a different language to translate to.")

    translation, english_step = translate(text, src, dst)
    if not translation.strip():
        raise gr.Error("The model returned an empty translation. Try a shorter or simpler sentence.")

    heard = f"{src}: {text}"
    if english_step:
        heard += f"\n\nVia English: {english_step}"

    try:
        audio_out = speak(translation, dst)
    except Exception as e:
        print("Text-to-speech error:", repr(e))
        audio_out = None
        gr.Warning("The spoken version isn't available right now. The text translation is ready.")

    return heard, translation, audio_out


def swap(src, dst):
    if src == AUTO:
        return gr.update(), gr.update()
    return dst, src


# ---------- Page ----------
HELLO = ["Hello", "नमस्ते", "नमस्कार", "வணக்கம்", "ನಮಸ್ಕಾರ", "നമസ്കാരം",
         "నమస్కారం", "你好", "مرحبا", "Привет", "Hola", "Hallo"]

CSS = """
#hello { display: flex; flex-wrap: wrap; gap: 0.35rem 1.2rem;
         font-size: 1.2rem; line-height: 1.7; color: var(--body-text-color-subdued); }
#intro p { max-width: 62ch; }
#note { font-size: 0.9rem; color: var(--body-text-color-subdued); max-width: 70ch; }
"""

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Noto Sans"), "system-ui", "sans-serif"],
)

# Gradio 6 moved theme/css from Blocks() to launch(); support both
GRADIO_MAJOR = int(gr.__version__.split(".")[0])
blocks_style = {} if GRADIO_MAJOR >= 6 else {"theme": theme, "css": CSS}
launch_style = {"theme": theme, "css": CSS} if GRADIO_MAJOR >= 6 else {}

with gr.Blocks(title="Talk across 12 languages", **blocks_style) as demo:
    gr.Markdown("# Talk across 12 languages")
    gr.HTML('<div id="hello">' + "".join(f"<span>{w}</span>" for w in HELLO) + "</div>")
    gr.Markdown(
        "Record a sentence or type one, choose the language to translate to, "
        "and hear the result.",
        elem_id="intro",
    )

    with gr.Row(equal_height=True):
        src = gr.Dropdown([AUTO] + list(LANGUAGES), value=AUTO, label="From", scale=4)
        swap_btn = gr.Button("⇄ Swap", scale=1, min_width=90)
        dst = gr.Dropdown(list(LANGUAGES), value="English", label="To", scale=4)

    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath",
                                label="Speak")
            typed_in = gr.Textbox(label="Or type", lines=2,
                                  placeholder="Type a short sentence")
            go = gr.Button("Translate", variant="primary")
        with gr.Column():
            heard_out = gr.Textbox(label="What you said", lines=3, interactive=False)
            text_out = gr.Textbox(label="Translation", lines=3, interactive=False)
            audio_out = gr.Audio(label="Listen", autoplay=True, interactive=False)

    gr.Examples(
        examples=[
            ["Where is the train station?", "English", "Spanish"],
            ["Thank you very much.", "English", "Tamil"],
            ["आप कैसे हो", "Hindi", "English"],
            ["¿Dónde está la estación de tren?", "Spanish", "English"],
        ],
        inputs=[typed_in, src, dst],
        label="Try a typed example",
    )

    gr.Markdown(
        "Translation by my fine-tuned [mT5-small model]"
        "(https://huggingface.co/Gunal13/mt5-small-11lang), trained on English ↔ 11 languages. "
        "Pairs without English are translated through English. Speech recognition by Whisper, "
        "voice by Google Text-to-Speech. Short, simple sentences work best; quality varies by "
        "language, so don't rely on it where a mistake matters.",
        elem_id="note",
    )

    go.click(run, inputs=[audio_in, typed_in, src, dst],
             outputs=[heard_out, text_out, audio_out])
    swap_btn.click(swap, inputs=[src, dst], outputs=[src, dst])

if __name__ == "__main__":
    demo.launch(inbrowser=True, **launch_style)