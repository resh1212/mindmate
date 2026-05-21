# session_state.py - Streamlit Session State Initialization

import queue
import streamlit as st
from rasa_client import RasaClient


def init_session_state():
    """Initialize all Streamlit session state variables."""
    defaults = {
        'messages': [],
        'audio_data': [],
        'recording': False,
        'current_language': 'en',
        'emotional_cues': [],
        'audio_queue': queue.Queue(),
        'waveform_data': [],
        'emotion_timestamps': [],
        'user_profile': {
            'name': '',
            'preferred_mode': 'text',
            'emotional_history': [],
            'conversation_count': 0
        },
        'rasa_client': RasaClient(),
        'rasa_connected': False,
        'use_rasa': True,
        'detected_intents': []
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value