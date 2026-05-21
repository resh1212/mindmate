# app.py - MindMate Entry Point
# Run with: streamlit run app.py

import streamlit as st
from ui.main_ui import main

if __name__ == "__main__":
    # Check for dependencies
    try:
        import pyaudio
    except ImportError:
        st.error("Please install PyAudio: pip install pyaudio")

    try:
        import speech_recognition
    except ImportError:
        st.error("Please install SpeechRecognition: pip install SpeechRecognition")

    try:
        import plotly
    except ImportError:
        st.error("Please install Plotly: pip install plotly")

    try:
        import requests
    except ImportError:
        st.error("Please install Requests: pip install requests")

    main()