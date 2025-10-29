[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![Hugging Face Spaces](https://img.shields.io/badge/Deploy-Hugging%20Face%20Spaces-yellow.svg)](https://huggingface.co/spaces/your-username/fitsens-ai)

# 💪 FitSens-AI: Intelligent Fitness Assistant

**FitSens-AI** is an intelligent fitness web application built with **Streamlit** that helps users:
- Track repetitions using real-time pose estimation 🏋️‍♂️  
- Correct posture automatically using **MediaPipe** 🧍‍♀️  
- Get personalized **diet & workout recommendations** 📏  
- Chat with an AI-powered **fitness chatbot** 🤖  

This app combines **Computer Vision**, **Machine Learning**, and **Natural Language Processing** to make your workouts smarter, safer, and more effective.

---

## 🚀 Features

### 🧠 Intelligent Modules
- **🏋️ Repetition Counter:** Uses webcam + pose estimation to count exercises like squats or curls.
- **🧍 Posture Correction:** Detects slouching or misalignment and provides real-time feedback.
- **📏 Body Ratio & Diet Recommendation:** Analyzes uploaded images or BMI inputs to suggest suitable diet/workout plans.
- **💬 AI Chatbot:** Uses Google Gemini API for personalized fitness and nutrition guidance.
- **📈 Analytics:** Displays weekly or session-based activity stats with charts.

---

## 🧩 Tech Stack

| Component | Technology Used |
|------------|----------------|
| Frontend | Streamlit |
| Pose Detection | MediaPipe |
| Computer Vision | OpenCV |
| NLP Chatbot | Google Generative AI (Gemini) |
| Model Embeddings | SentenceTransformers |
| Visualization | Matplotlib, Pandas |
| Voice Feedback | pyttsx3 (Text-to-Speech) |

---

## 🛠️ Installation (Run Locally)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/fitsens-ai.git
cd fitsens-ai
