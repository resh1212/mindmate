# nlp_engine.py - Sentiment, Emotion, Translation, and Bot Response Logic

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    nltk.download('vader_lexicon', quiet=True)
except Exception:
    pass


# ============== MODEL LOADING ==============

@st.cache_resource
def load_models():
    """Load ML models for emotion and sentiment analysis"""
    sentiment_analyzer = SentimentIntensityAnalyzer()

    try:
        emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
    except Exception:
        emotion_classifier = None

    return sentiment_analyzer, emotion_classifier


sentiment_analyzer, emotion_classifier = load_models()


# ============== ANALYSIS FUNCTIONS ==============

def analyze_text_sentiment(text):
    """Analyze sentiment from text. Returns (sentiment_label, scores_dict)."""
    scores = sentiment_analyzer.polarity_scores(text)

    if scores['compound'] >= 0.05:
        sentiment = "positive"
    elif scores['compound'] <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return sentiment, scores


def analyze_text_emotion(text):
    """Analyze dominant emotion from text. Returns (emotion_label, emotion_scores_dict)."""
    if emotion_classifier:
        try:
            results = emotion_classifier(text)[0]
            emotions = {result['label']: result['score'] for result in results}
            dominant_emotion = max(emotions, key=emotions.get)
            return dominant_emotion, emotions
        except Exception:
            pass

    # Fallback to sentiment-based mapping
    sentiment, scores = analyze_text_sentiment(text)
    emotion_map = {
        'positive': 'joy',
        'negative': 'sadness',
        'neutral': 'neutral'
    }
    return emotion_map.get(sentiment, 'neutral'), {}


def translate_text(text, target_lang='en'):
    """Translate text to target language. Returns original if translation fails."""
    try:
        if target_lang != 'en':
            return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception:
        pass
    return text


# ============== FALLBACK RESPONSES ==============

FALLBACK_RESPONSES = {
    'greeting': {
        'en': "Hello! I'm MindMate. How are you feeling today?",
        'es': "¡Hola! Soy MindMate. ¿Cómo te sientes hoy?",
        'hi': "नमस्ते! मैं माइंडमेट हूं। आज आप कैसा महसूस कर रहे हैं?",
        'fr': "Bonjour! Je suis MindMate. Comment vous sentez-vous aujourd'hui?"
    },
    'positive': {
        'en': "That's wonderful to hear! 😊 Would you like to share what's making you happy?",
        'es': "¡Es maravilloso escuchar eso! 😊 ¿Te gustaría compartir qué te hace feliz?",
        'hi': "यह सुनकर बहुत अच्छा लगा! 😊 क्या आप यह साझा करना चाहेंगे कि आपको क्या खुश कर रहा है?",
        'fr': "C'est merveilleux à entendre! 😊 Souhaitez-vous partager ce qui vous rend heureux?"
    },
    'negative': {
        'en': "I'm sorry you're feeling this way. I'm here to listen. Would you like to talk about it?",
        'es': "Lamento que te sientas así. Estoy aquí para escuchar. ¿Te gustaría hablar de eso?",
        'hi': "मुझे खेद है कि आप ऐसा महसूस कर रहे हैं। मैं सुनने के लिए यहां हूं।",
        'fr': "Je suis désolé que vous vous sentiez ainsi. Je suis là pour écouter."
    },
    'support': {
        'en': "Remember, it's okay to not be okay. Would you like some coping techniques?",
        'es': "Recuerda, está bien no estar bien. ¿Te gustaría algunas técnicas de afrontamiento?",
        'hi': "याद रखें, ठीक नहीं होना ठीक है। क्या आप कुछ मुकाबला करने की तकनीक चाहेंगे?",
        'fr': "Rappelez-vous, ce n'est pas grave de ne pas aller bien."
    },
    'techniques': {
        'en': "Try deep breathing: Inhale 4s, hold 4s, exhale 6s. Repeat 5 times. How do you feel?",
        'es': "Prueba la respiración profunda: Inhalar 4s, mantener 4s, exhalar 6s.",
        'hi': "गहरी सांस लेने का प्रयास करें: 4 सेकंड में सांस लें, 4 सेकंड रोकें, 6 सेकंड में छोड़ें।",
        'fr': "Essayez la respiration profonde: Inspirez 4s, retenez 4s, expirez 6s."
    },
    'emergency': {
        'en': "If you're in crisis, please contact emergency services or a crisis helpline immediately.",
        'es': "Si estás en crisis, por favor contacta a los servicios de emergencia inmediatamente.",
        'hi': "यदि आप संकट में हैं, तो कृपया तुरंत आपातकालीन सेवाओं से संपर्क करें।",
        'fr': "Si vous êtes en crise, veuillez contacter immédiatement les services d'urgence."
    }
}


def get_fallback_response(user_input, language='en'):
    """Rule-based fallback response. Returns (response_text, intent_label)."""
    user_input_lower = user_input.lower()

    if any(word in user_input_lower for word in ['hello', 'hi', 'hey']):
        intent = 'greeting'
    elif any(word in user_input_lower for word in ['happy', 'good', 'great', 'excellent']):
        intent = 'positive'
    elif any(word in user_input_lower for word in ['sad', 'depressed', 'anxious', 'stressed', 'bad']):
        intent = 'negative'
    elif any(word in user_input_lower for word in ['help', 'support', 'technique', 'coping']):
        intent = 'techniques'
    elif any(word in user_input_lower for word in ['suicide', 'hurt', 'emergency', 'crisis']):
        intent = 'emergency'
    else:
        intent = 'support'

    response = (
        FALLBACK_RESPONSES.get(intent, FALLBACK_RESPONSES['support'])
        .get(language, FALLBACK_RESPONSES['support']['en'])
    )
    return response, intent


def get_bot_response(user_input, language='en'):
    """
    Get bot response — uses Rasa if available, falls back to local logic.
    Returns: (response_text, intent_label, buttons_list)
    """
    from datetime import datetime
    rasa_client = st.session_state.rasa_client

    if st.session_state.use_rasa and rasa_client.is_connected:
        # Translate to English for Rasa if needed
        user_input_en = user_input
        if language != 'en':
            try:
                user_input_en = GoogleTranslator(source='auto', target='en').translate(user_input)
            except Exception:
                pass

        nlu_result = rasa_client.parse_message(user_input_en)

        st.session_state.detected_intents.append({
            'text': user_input,
            'intent': nlu_result['intent'],
            'confidence': nlu_result['confidence'],
            'entities': nlu_result['entities'],
            'timestamp': datetime.now()
        })

        responses = rasa_client.send_message(user_input_en)

        if responses:
            response_text = " ".join([r.get('text', '') for r in responses if r.get('text')])
            buttons = []
            for r in responses:
                if 'buttons' in r:
                    buttons.extend(r['buttons'])

            if language != 'en' and response_text:
                try:
                    response_text = GoogleTranslator(source='en', target=language).translate(response_text)
                except Exception:
                    pass

            if response_text:
                return response_text, nlu_result['intent'], buttons

    # Fallback
    response, intent = get_fallback_response(user_input, language)
    return response, intent, []