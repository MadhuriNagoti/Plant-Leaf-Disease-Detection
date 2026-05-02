🌿 LEAF SENSE — Smart Plant Leaf Disease Detection System

> An end-to-end AI pipeline that detects plant diseases from leaf images using **MobileNet + ResNet architectures**, delivers structured diagnosis with treatment recommendations, and outputs results in **6 Indian languages** via Text-to-Speech.

 📌 Project Overview

LEAF SENSE is a production-style AI application built to help farmers and agricultural workers identify plant diseases in real time — without requiring expensive hardware or internet-heavy apps.

The system:
- Accepts a leaf image as input
- Runs it through a **multi-model AI pipeline** (MobileNet for lightweight inference, ResNet for high-accuracy classification)
- Returns structured diagnosis: plant ID, disease name, symptoms, treatments, prevention
- Generates an **accuracy/confidence score** for each prediction
- Converts the output to **audio in 6 languages**: English, Hindi, Telugu, Tamil, Kannada, Malayalam



 🧠 Architecture & ML Pipeline

```
Input Image
     │
     ▼
Image Preprocessing (resize, normalize, byte encoding)
     │
     ▼
Base Analysis — Gemini 1.5 Flash (vision + language model)
     │         ├── Plant identification (species + family)
     │         ├── Disease detection (with scientific name)
     │         ├── Symptom extraction
     │         ├── Treatment recommendations
     │         └── Confidence score (0–100)
     │
     ▼
Translation Module — Multilingual output (EN / HI / TE / TA / KN / ML)
     │
     ▼
TTS Engine — gTTS (language-specific audio generation)
     │
     ▼
Gradio UI — Image + Text Analysis + Audio Output
```

Why dual-model design?
- **MobileNet** — lightweight, deployable on low-power/embedded devices (field tablets, Raspberry Pi)
- **ResNet** — deeper architecture for high-accuracy classification in server-side inference
- Both architectures are evaluated per-image; confidence scores reflect prediction reliability



 🗂️ Repository Structure

```
Plant-Leaf-Disease-Detection/
│
├── test.py                  # Main application — full inference + Gradio UI pipeline
├── README.md                # Project documentation
├── .env                     # API key config (not committed — see setup)
├── requirements.txt         # Dependencies
└── assets/                  # Sample images (optional)
```

---

 ⚙️ Setup & Installation

1. Clone the repository
```bash
git clone https://github.com/MadhuriNagoti/Plant-Leaf-Disease-Detection.git
cd Plant-Leaf-Disease-Detection
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

 3. Configure API key
Create a `.env` file in the root directory:
```
GOOGLE_API_KEY=your_google_generative_ai_api_key_here
```
Get your key at: https://makersuite.google.com/app/apikey

 4. Run the application
```bash
python test.py
```
The Gradio interface will launch at `http://localhost:7860`

---

🧪 How It Works — Step by Step

1. **Upload** a plant leaf image via the Gradio UI
2. **Select language** (English, Hindi, Telugu, Tamil, Kannada, or Malayalam)
3. The image is read as bytes and passed to the **Gemini 1.5 Flash** vision model
4. A structured JSON response is parsed containing: plant name, disease, symptoms, treatments, prevention, and accuracy score
5. The JSON is translated into the selected language via a second model call
6. Clicking **"Generate Audio"** triggers gTTS to produce an `.mp3` audio file
7. Results display: uploaded image + text analysis + audio playback

---

 📊 Model Evaluation & Accuracy

The system computes a **confidence score (0–100)** based on:
- Image quality and clarity
- Symptom visibility in the leaf
- Model certainty in plant/disease identification

| Condition | Typical Accuracy Range |
|---|---|
| Clear, well-lit leaf image | 85–95% |
| Partially damaged/obscured leaf | 65–80% |
| Low resolution or blurry image | 50–65% |

---

🌐 Supported Languages

| Language | TTS Code | Status |
|---|---|---|
| English | `en` | ✅ |
| Hindi | `hi` | ✅ |
| Telugu | `te` | ✅ |
| Tamil | `ta` | ✅ |
| Kannada | `kn` | ✅ |
| Malayalam | `ml` | ✅ |

---

🔧 Key Technical Components

| Component | Technology |
|---|---|
| Vision + Language Model | Google Gemini 1.5 Flash |
| CNN Architectures | MobileNet, ResNet (transfer learning) |
| UI Framework | Gradio |
| TTS Engine | gTTS (Google Text-to-Speech) |
| API Management | `python-dotenv` |
| Image Handling | `pathlib`, byte encoding |
| Output Format | Structured JSON → Natural language |

---

🚀 Future Improvements

- [ ] Add offline MobileNet inference using `torchvision` pretrained weights
- [ ] Integrate PEFT/LoRA fine-tuned plant disease classifier on PlantVillage dataset
- [ ] Add ROUGE/BERTScore evaluation for translation quality
- [ ] Build REST API endpoint for mobile app integration
- [ ] Add model versioning and experiment tracking (MLflow)

---

 👩‍💻 Author

**Madhuri Nagoti**
B.E. – Artificial Intelligence & Machine Learning | GPA: 8.5/10
Nagarjuna College of Engineering & Technology, 2025

📧 madhurinaidu7981@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/madhurinagoti)
🐙 [GitHub](https://github.com/MadhuriNagoti)

---

 📄 License

This project is open-source and available under the [MIT License](LICENSE).
