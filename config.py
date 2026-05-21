# config.py - MindMate Configuration

import pyaudio

# ============== AUDIO SETTINGS ==============
RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

# ============== RASA CONFIGURATION ==============
RASA_SERVER_URL = "http://localhost:5005"
RASA_ACTIONS_URL = "http://localhost:5055"
RASA_WEBHOOK_URL = f"{RASA_SERVER_URL}/webhooks/rest/webhook"
RASA_NLU_URL = f"{RASA_SERVER_URL}/model/parse"

# ============== SUPPORTED LANGUAGES ==============
LANGUAGE_OPTIONS = {
    "English": "en",
    "Spanish": "es",
    "Hindi": "hi",
    "French": "fr",
    "Chinese": "zh",
    "Arabic": "ar"
}

# ============== EMOTION COLORS ==============
EMOTION_COLORS = {
    'happy': '#FFD700',
    'sad': '#1E90FF',
    'angry': '#FF4500',
    'calm': '#32CD32',
    'anxious': '#9370DB',
    'neutral': '#A9A9A9',
    'joy': '#FFD700',
    'sadness': '#1E90FF',
    'anger': '#FF4500',
    'fear': '#9370DB',
    'surprise': '#FF69B4',
    'disgust': '#8B4513'
}