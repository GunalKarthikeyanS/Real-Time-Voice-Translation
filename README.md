# Real-Time Voice Translation
 
Speak a sentence in one of 12 languages and hear it in another, translated by my own fine-tuned mT5 model.
 
**Languages:** English, Hindi, Marathi, Tamil, Kannada, Malayalam, Telugu, Chinese, Arabic, Russian, Spanish, Afrikaans
 
**Model:** [Gunal13/mt5-small-11lang](https://huggingface.co/Gunal13/mt5-small-11lang) on Hugging Face
 
## How it works
 
```
🎤 speech
  ↓  Whisper (small, int8) — transcribes and detects which of the 12 languages was spoken
text
  ↓  my fine-tuned mT5-small — translates (pairs without English go through English)
translation
  ↓  Google Text-to-Speech
🔊 speech in the other language
```
 
There are two ways to use it:
 
- **Website** (`app.py`): a Gradio page in the browser. Speak or type, choose the target language, and hear the result. Shows the English middle step for pairs like Tamil → Telugu.
- **Notebook** (`notebooks/live_translator.ipynb`): a hands-free conversation loop on a laptop microphone. Choose two languages by voice, then speak either one.
## Run it (Windows)
 
Requires Python 3.12 and an internet connection (for speech output). The first start downloads about 1.7 GB of models.
 
```
py -3.12 -m venv venv-web
venv-web\Scripts\activate
pip install -r requirements.txt
```
 
Then double-click **`start_translator.bat`** (or run `python app.py`). The website opens at `http://127.0.0.1:7860`.
 
## Results
 
Evaluated on OPUS-100 test sets (first 500 sentences per language, each direction), with beam search and `no_repeat_ngram_size=3`. chrF and BLEU from sacreBLEU, 0–100, higher is better.
 
| Language | EN→X chrF | X→EN chrF | EN→X BLEU | X→EN BLEU |
|---|---|---|---|---|
| Spanish | 39.9 | 44.5 | 15.7 | 20.5 |
| Afrikaans | 34.8 | 43.6 | 18.3 | 26.4 |
| Marathi | 30.1 | 41.8 | 11.1 | 23.0 |
| Kannada | 28.2 | 38.5 | 12.9 | 18.0 |
| Russian | 29.3 | 38.4 | 10.4 | 16.4 |
| Chinese | 16.8 | 37.5 | 15.9 | 13.1 |
| Arabic | 24.3 | 33.5 | 3.8 | 12.0 |
| Hindi | 27.0 | 32.4 | 10.0 | 12.5 |
| Telugu | 27.8 | 32.2 | 12.4 | 16.6 |
| Tamil | 29.5 | 28.6 | 6.1 | 10.9 |
| Malayalam | 23.1 | 25.2 | 1.4 | 8.3 |
| **Average** | **28.3** | **36.0** | | |
 
Translation into English is stronger than out of English for 10 of the 11 languages. OPUS-100 test references are themselves noisy, so these scores are lower than a clean benchmark would give.
 
## How the model was built
 
All steps, outputs and training logs are in [`notebooks/train_mt5_11lang.ipynb`](notebooks/train_mt5_11lang.ipynb).
 
1. **Starting point.** The original version (in [`legacy/`](legacy/)) fine-tuned T5-small, but the live loop overwrote the model with Google Translate, so T5 was never used. T5-small's tokenizer also turns every Hindi word into `<unk>`, so it could not have learned Hindi.
2. **Switch to mT5-small** (300M parameters), whose 250k-token vocabulary covers all 12 languages.
3. **Data cleaning.** OPUS-100 English–Hindi had subtitle tags, wrong-language text, misaligned pairs, and about 1 in 4 pairs were exact duplicates. Six filters kept 55% of the data.
4. **Stage 1: English ↔ Hindi.** 100k pairs used in both directions, 1 epoch on a Colab T4 (2h14m). Validation loss 3.03 → 2.48. A batch of 32 with AdamW ran out of memory because of the large output layer; Adafactor with gradient accumulation (8 × 4) fit in 6.2 GB.
5. **Decoding experiment.** `no_repeat_ngram_size=3` removed repetition loops on long sentences: English→Hindi chrF 24.1 → 25.6, Hindi→English 29.1 → 29.4 (2,000 test sentences each).
6. **Stage 2: 11 languages.** Up to 20k cleaned pairs per language (Kannada had only 9,096 and was repeated 2×), both directions, 436k examples, continued from the stage 1 model. Validation loss 3.32 → 2.72 at step 13,000, used as the final checkpoint.
## Repository
 
```
├── app.py                          website (Gradio)
├── start_translator.bat            one-click launcher for Windows
├── requirements.txt
├── notebooks/
│   ├── train_mt5_11lang.ipynb      data cleaning, training and evaluation (Colab, T4)
│   └── live_translator.ipynb       hands-free microphone version
└── legacy/                         the original T5 + Google Translate version
```
 
## Limitations
 
- Short, simple sentences work best; the model still makes meaning-changing mistakes (for example "I am hungry" → "I'm sick"), so don't use it where a mistranslation matters.
- Pairs without English are translated through English, so errors can add up.
- Typed text must use each language's own script; Tamil or Hindi written in English letters isn't supported.
- OPUS-100 is mostly religious text, subtitles and software strings, so casual everyday speech is translated less well.
## Credits
 
The original version (in `legacy/`) was built as a course project at Yeshiva University, Katz School (MS in Artificial Intelligence, Fall 2024) with Siddhanth Jadhav
 
The mT5 data pipeline, training, evaluation, live translator and website were built by **Gunal Karthikeyan Saravanan** · [GitHub](https://github.com/GunalKarthikeyanS) · [LinkedIn](https://linkedin.com/in/gunalkarthikeyans)
 
Built with [OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100), [mT5](https://huggingface.co/google/mt5-small), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [gTTS](https://github.com/pndurette/gTTS) and [Gradio](https://www.gradio.app/).
 