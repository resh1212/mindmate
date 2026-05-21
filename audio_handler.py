# audio_handler.py - Audio Recording and Real-Time Emotion Detection

import numpy as np
import pyaudio
import streamlit as st
from scipy import signal

from config import RATE, CHUNK, FORMAT, CHANNELS


class RealTimeEmotionDetector:
    """Detect emotional cues in real-time speech"""

    def __init__(self):
        self.audio_buffer = []
        self.emotion_history = []
        self.current_emotion = "neutral"
        self.emotion_thresholds = {
            'positive': 0.6,
            'negative': 0.6,
            'neutral': 0.4
        }

    def analyze_audio_chunk(self, audio_data):
        """Analyze audio chunk for emotional cues"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        features = self.extract_audio_features(audio_array)
        emotion = self.detect_emotion_from_features(features)
        return emotion, features

    def extract_audio_features(self, audio_array):
        """Extract basic audio features"""
        features = {}
        features['amplitude'] = np.abs(audio_array).mean()
        features['amplitude_std'] = np.abs(audio_array).std()
        features['zcr'] = np.sum(np.diff(np.sign(audio_array)) != 0) / len(audio_array)
        features['energy'] = np.sum(audio_array.astype(float) ** 2) / len(audio_array)

        if len(audio_array) > 0:
            f, t, Sxx = signal.spectrogram(audio_array, RATE)
            features['spectral_centroid'] = (
                np.sum(f[:, np.newaxis] * Sxx) / np.sum(Sxx)
                if np.sum(Sxx) > 0 else 0
            )

        return features

    def detect_emotion_from_features(self, features):
        """Detect emotion from audio features using heuristic rules"""
        emotion_scores = {
            'calm': 0.3,
            'happy': 0.2,
            'sad': 0.2,
            'angry': 0.1,
            'anxious': 0.2
        }

        if features['amplitude'] > 1000:
            emotion_scores['angry'] += 0.3
            emotion_scores['anxious'] += 0.2
        elif features['amplitude'] < 200:
            emotion_scores['calm'] += 0.3
            emotion_scores['sad'] += 0.2

        if features['zcr'] > 0.3:
            emotion_scores['anxious'] += 0.2
        elif features['zcr'] < 0.1:
            emotion_scores['calm'] += 0.2

        if features['energy'] > 500000:
            emotion_scores['happy'] += 0.2
            emotion_scores['angry'] += 0.1

        return max(emotion_scores, key=emotion_scores.get)


class AudioRecorder:
    """Handle real-time audio recording with emotion detection"""

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.emotion_detector = RealTimeEmotionDetector()
        self.is_recording = False
        self.full_audio = []

    def start_recording(self):
        """Start audio recording"""
        self.is_recording = True
        self.full_audio = []

        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self.callback
        )

        st.session_state.recording = True

    def stop_recording(self):
        """Stop audio recording and return collected audio frames"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        st.session_state.recording = False
        return self.full_audio

    def callback(self, in_data, frame_count, time_info, status):
        """Audio callback for real-time processing"""
        if self.is_recording:
            self.full_audio.append(in_data)
            emotion, features = self.emotion_detector.analyze_audio_chunk(in_data)
            current_time = len(self.full_audio) * CHUNK / RATE

            if emotion != self.emotion_detector.current_emotion:
                self.emotion_detector.current_emotion = emotion
                st.session_state.emotion_timestamps.append({
                    'time': current_time,
                    'emotion': emotion,
                    'intensity': features['amplitude'] / 32768.0
                })

            audio_array = np.frombuffer(in_data, dtype=np.int16)
            st.session_state.waveform_data.extend(audio_array.tolist())

        return (in_data, pyaudio.paContinue)