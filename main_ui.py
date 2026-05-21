# ui/main_ui.py - Main Streamlit UI

import streamlit as st
import speech_recognition as sr
import pandas as pd
import plotly.express as px
from datetime import datetime

from config import RATE, LANGUAGE_OPTIONS, EMOTION_COLORS
from session_state import init_session_state
from nlp_engine import analyze_text_sentiment, analyze_text_emotion, get_bot_response
from audio_handler import AudioRecorder
from visualizations import (
    create_emotion_waveform_plot,
    create_emotional_insights,
    create_intent_analysis_chart
)

# ============== CSS ==============

CUSTOM_CSS = """
<style>
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.main-header {
    color: white;
    text-align: center;
    padding: 1rem;
    border-radius: 10px;
    background: rgba(0, 0, 0, 0.5);
    margin-bottom: 2rem;
}
.chat-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}
.user-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px 15px 5px 15px;
    padding: 10px 15px;
    margin: 5px 0;
    max-width: 70%;
    margin-left: auto;
}
.bot-message {
    background: #f0f0f0;
    color: #333;
    border-radius: 15px 15px 15px 5px;
    padding: 10px 15px;
    margin: 5px 0;
    max-width: 70%;
}
.emotion-highlight {
    padding: 5px 10px;
    border-radius: 20px;
    margin: 2px;
    font-size: 0.8em;
    display: inline-block;
}
.intent-badge {
    background: #e8f4f8;
    color: #2196F3;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 0.75em;
    margin-left: 5px;
}
.rasa-status {
    padding: 5px 10px;
    border-radius: 5px;
    font-size: 0.9em;
    margin: 5px 0;
}
.rasa-connected { background: #d4edda; color: #155724; }
.rasa-disconnected { background: #f8d7da; color: #721c24; }
.recording-indicator { animation: pulse 1.5s infinite; }
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}
</style>
"""

COPING_TECHNIQUES = {
    "Deep Breathing": "Inhale 4s, hold 4s, exhale 6s",
    "Grounding": "5 things you see, 4 you feel, 3 you hear",
    "Mindfulness": "Focus on your breath for 1 minute",
    "Positive Affirmation": "Repeat: 'I am strong and capable'"
}


def main():
    """Main Streamlit application with Rasa integration."""
    init_session_state()

    st.set_page_config(
        page_title="MindMate - AI Mental Health Chatbot (Rasa)",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <h1>🧠 MindMate - AI Mental Health Companion</h1>
        <p>Powered by Rasa | Multilingual Emotional Support with Real-time Voice Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    _render_sidebar()

    col1, col2 = st.columns([2, 1])
    mode = st.session_state.get('_mode', 'Text')

    with col1:
        _render_chat(mode)

    with col2:
        _render_right_panel()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: white;">
        <p><strong>⚠️ Important:</strong> MindMate is an AI assistant, not a replacement for professional help.</p>
        <p>If you're in crisis, please contact emergency services or a mental health professional immediately.</p>
        <p>Crisis Hotline: 988 (US) | International helplines available</p>
        <p style="font-size: 0.8em; opacity: 0.7;">Powered by Rasa Open Source</p>
    </div>
    """, unsafe_allow_html=True)


def _render_sidebar():
    """Render the sidebar with settings, Rasa status, and voice controls."""
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2491/2491343.png", width=100)
        st.title("Settings")

        # Rasa status
        st.subheader("🤖 Rasa Status")
        rasa_client = st.session_state.rasa_client

        if st.button("🔄 Check Rasa Connection"):
            st.session_state.rasa_connected = rasa_client.check_connection()

        if st.session_state.rasa_connected:
            st.markdown('<div class="rasa-status rasa-connected">✅ Rasa Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="rasa-status rasa-disconnected">❌ Rasa Disconnected (Using Fallback)</div>', unsafe_allow_html=True)

        st.session_state.use_rasa = st.checkbox("Use Rasa (when available)", value=True)

        with st.expander("⚙️ Rasa Configuration"):
            from config import RASA_SERVER_URL
            new_url = st.text_input("Rasa Server URL", value=RASA_SERVER_URL)
            if new_url != rasa_client.server_url:
                from rasa_client import RasaClient
                st.session_state.rasa_client = RasaClient(new_url)
                st.info("URL updated. Click 'Check Rasa Connection' to verify.")

        st.markdown("---")

        # Language
        selected_language = st.selectbox("Select Language", options=list(LANGUAGE_OPTIONS.keys()), index=0)
        st.session_state.current_language = LANGUAGE_OPTIONS[selected_language]

        # Mode
        mode = st.radio("Interaction Mode", ["Text", "Voice"], horizontal=True)
        st.session_state['_mode'] = mode

        # Voice controls
        if mode == "Voice":
            st.subheader("🎤 Voice Settings")
            st.session_state.user_profile['preferred_mode'] = 'voice'

            if st.button("🎤 Start Recording", type="primary"):
                if not st.session_state.recording:
                    recorder = AudioRecorder()
                    st.session_state.recorder = recorder
                    recorder.start_recording()
                    st.rerun()

            if st.session_state.recording:
                st.warning("🎤 Recording... Speak now")
                if st.button("⏹️ Stop Recording"):
                    audio_data = st.session_state.recorder.stop_recording()
                    recognizer = sr.Recognizer()
                    audio_bytes = b''.join(audio_data)

                    with sr.AudioData(audio_bytes, RATE, 2) as source:
                        try:
                            text = recognizer.recognize_google(source)
                            st.session_state.messages.append({
                                "role": "user", "content": text, "type": "voice",
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            response, intent, buttons = get_bot_response(text, st.session_state.current_language)
                            st.session_state.messages.append({
                                "role": "assistant", "content": response, "type": "voice",
                                "intent": intent, "buttons": buttons,
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            emotion, _ = analyze_text_emotion(text)
                            st.session_state.emotional_cues.append({
                                "text": text, "emotion": emotion, "timestamp": datetime.now()
                            })
                            st.rerun()
                        except sr.UnknownValueError:
                            st.error("Could not understand audio")
                        except sr.RequestError as e:
                            st.error(f"Recognition error: {e}")
        else:
            st.session_state.user_profile['preferred_mode'] = 'text'

        st.markdown("---")

        # Emotional insights bar chart
        st.subheader("📊 Emotional Insights")
        if st.session_state.emotional_cues:
            emotions_df = pd.DataFrame(st.session_state.emotional_cues)
            emotion_counts = emotions_df['emotion'].value_counts()
            fig_insights = px.bar(
                x=emotion_counts.index, y=emotion_counts.values,
                title="Emotional Journey", labels={'x': 'Emotion', 'y': 'Count'},
                color=emotion_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_insights.update_layout(height=200)
            st.plotly_chart(fig_insights, use_container_width=True)

        # Intent analysis
        if st.session_state.detected_intents:
            st.subheader("🎯 Intent Analysis")
            fig_intents = create_intent_analysis_chart()
            if fig_intents:
                st.plotly_chart(fig_intents, use_container_width=True)

        # Reset
        if st.button("🔄 Reset Conversation"):
            st.session_state.rasa_client.reset_conversation()
            st.session_state.messages = []
            st.session_state.emotional_cues = []
            st.session_state.waveform_data = []
            st.session_state.emotion_timestamps = []
            st.session_state.detected_intents = []
            st.rerun()


def _render_chat(mode):
    """Render the main chat column."""
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.subheader("💬 Conversation")

    with st.container():
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {message['content']}<br>
                    <small style="opacity:0.7;">{message['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)

                # Emotion badge
                for cue in st.session_state.emotional_cues:
                    if cue['text'] == message['content']:
                        color = EMOTION_COLORS.get(cue['emotion'], '#000000')
                        st.markdown(f"""
                        <div class="emotion-highlight" style="background-color:{color};color:white;">
                            Detected: {cue['emotion'].upper()}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                intent_badge = ""
                if message.get('intent'):
                    intent_badge = f'<span class="intent-badge">🎯 {message["intent"]}</span>'

                st.markdown(f"""
                <div class="bot-message">
                    <strong>MindMate:</strong> {intent_badge} {message['content']}<br>
                    <small style="opacity:0.7;">{message['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)

                # Rasa quick-reply buttons
                if message.get('buttons'):
                    cols = st.columns(len(message['buttons']))
                    for idx, btn in enumerate(message['buttons']):
                        with cols[idx]:
                            if st.button(btn.get('title', ''), key=f"btn_{message['timestamp']}_{idx}"):
                                payload = btn.get('payload', btn.get('title', ''))
                                st.session_state.messages.append({
                                    "role": "user", "content": payload, "type": "button",
                                    "timestamp": datetime.now().strftime("%H:%M:%S")
                                })
                                response, intent, buttons = get_bot_response(payload, st.session_state.current_language)
                                st.session_state.messages.append({
                                    "role": "assistant", "content": response, "type": "text",
                                    "intent": intent, "buttons": buttons,
                                    "timestamp": datetime.now().strftime("%H:%M:%S")
                                })
                                st.rerun()

    if mode == "Text":
        user_input = st.chat_input("Type your message here...")
        if user_input:
            st.session_state.messages.append({
                "role": "user", "content": user_input, "type": "text",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            sentiment, scores = analyze_text_sentiment(user_input)
            emotion, emotion_scores = analyze_text_emotion(user_input)
            st.session_state.emotional_cues.append({
                "text": user_input, "emotion": emotion,
                "sentiment": sentiment, "scores": scores,
                "timestamp": datetime.now()
            })
            response, intent, buttons = get_bot_response(user_input, st.session_state.current_language)
            st.session_state.messages.append({
                "role": "assistant", "content": response, "type": "text",
                "intent": intent, "buttons": buttons,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def _render_right_panel():
    """Render the right panel with live emotion analysis and coping techniques."""
    st.subheader("🎵 Live Emotion Analysis")

    if st.session_state.recording and st.session_state.waveform_data:
        fig_waveform = create_emotion_waveform_plot()
        if fig_waveform:
            st.plotly_chart(fig_waveform, use_container_width=True)
            if st.session_state.emotion_timestamps:
                latest = st.session_state.emotion_timestamps[-1]['emotion']
                emotion_display = {
                    'happy': '😊 Happy', 'sad': '😢 Sad', 'angry': '😠 Angry',
                    'calm': '😌 Calm', 'anxious': '😰 Anxious', 'neutral': '😐 Neutral'
                }
                st.info(f"**Current Emotion:** {emotion_display.get(latest, 'Neutral')}")

    if st.session_state.emotion_timestamps:
        st.subheader("📈 Emotional Patterns")
        fig_pie, fig_timeline = create_emotional_insights()
        if fig_pie:
            st.plotly_chart(fig_pie, use_container_width=True)
        if fig_timeline:
            st.plotly_chart(fig_timeline, use_container_width=True)

    # Rasa tracker
    if st.session_state.rasa_connected and st.session_state.use_rasa:
        st.subheader("🤖 Rasa Session Info")
        tracker = st.session_state.rasa_client.get_tracker()
        if tracker:
            with st.expander("View Tracker"):
                st.json({
                    "sender_id": tracker.get("sender_id", ""),
                    "slots": tracker.get("slots", {}),
                    "latest_action": tracker.get("latest_action_name", ""),
                    "events_count": len(tracker.get("events", []))
                })

    # Coping techniques
    st.subheader("🆘 Quick Coping Techniques")
    for technique, description in COPING_TECHNIQUES.items():
        with st.expander(technique):
            st.write(description)
            if st.button(f"Try {technique}", key=f"btn_{technique}"):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Let's practice {technique}: {description}. Ready?",
                    "type": "text", "intent": "coping_technique", "buttons": [],
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                st.rerun()
                