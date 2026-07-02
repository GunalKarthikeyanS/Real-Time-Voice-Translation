# 🎙️ Real-Time Voice Translation using NMT
### Multilingual Speech-to-Speech Translation System

A real-time voice translation system that enables two speakers to communicate across language barriers. Built using fine-tuned T5 for Neural Machine Translation (NMT) and Google Speech Recognition for speech-to-text, achieving under 200ms end-to-end latency with 90%+ translation accuracy.

---

## 🚀 Key Results
- **Under 200ms** end-to-end latency for real-time translation
- **90%+ translation accuracy** across supported languages
- **8 language pairs** supported from English
- Fine-tuned T5 model achieving loss convergence from 10.57 → 0.01

---

## 🌍 Supported Languages

| Language | Code |
|---|---|
| Hindi | hi |
| Marathi | mr |
| Tamil | ta |
| Kannada | kn |
| Malayalam | ml |
| Afrikaans | af |
| Spanish | es |
| Chinese | zh |

---

## 🏗️ System Architecture

```
Speaker 1 (Voice Input)
    ↓
Google Speech Recognition — Speech to Text
    ↓
Fine-tuned T5 NMT Model — Neural Machine Translation
    ↓
gTTS (Google Text-to-Speech) — Text to Audio
    ↓
Speaker 2 (Audio Output in target language)
    ↑
[Bidirectional — Speaker 2 responds back through same pipeline]
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| NMT Model | T5-small (fine-tuned) |
| Speech Recognition | Google Speech API (SpeechRecognition) |
| Text-to-Speech | gTTS (Google TTS) |
| Translation Fallback | googletrans |
| Deep Learning | PyTorch, HuggingFace Transformers |
| Data Processing | Pandas |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/GunalKarthikeyanS/Real-Time-Voice-Translation.git
cd Real-Time-Voice-Translation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare translation dataset
Place your multilingual CSV dataset as `data.csv` with columns:
```
Input Text | Hindi | Marathi | Tamil | Kannada | Malayalam | Afrikaans | Spanish | Chinese
```

### 4. Fine-tune the T5 model
Open `Final.ipynb` and run the training cells. The model fine-tunes T5-small on each language pair sequentially and saves to `fine_tuned_t5-small/`.

### 5. Run real-time translation
Run the real-time translation cell in `Final.ipynb`. The system will:
1. Ask Speaker 1 to state their language
2. Ask Speaker 2 to state their language
3. Begin real-time bidirectional translation

---

## 💡 Features

- **Bidirectional Translation** — Both speakers can communicate in their native language
- **Real-time Processing** — Under 200ms end-to-end latency
- **Voice-First Interface** — Entirely voice-controlled, no typing needed
- **Fine-tuned NMT** — Custom T5 model trained on domain-specific translation data
- **8 Language Support** — Covers major Indian languages plus global languages
- **Ambient Noise Adjustment** — Automatically adjusts for background noise

---

## 📁 Project Structure

```
Real-Time-Voice-Translation/
├── Final.ipynb          ← Main notebook (training + real-time translation)
├── requirements.txt
└── README.md
```

---

## 🎓 Academic Context

**Institution:** Yeshiva University, Katz School of Science and Health
**Program:** MS in Artificial Intelligence
**Period:** August 2024 – December 2024

---

## 👤 Author

**Gunal Karthikeyan Saravanan**
MS in Artificial Intelligence — Yeshiva University
[LinkedIn](https://linkedin.com/in/gunalkarthikeyans) | [GitHub](https://github.com/GunalKarthikeyanS)
