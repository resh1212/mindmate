# visualizations.py - Plotly Charts for Emotion & Intent Analysis

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from config import RATE, EMOTION_COLORS


def create_emotion_waveform_plot():
    """Create interactive waveform plot with emotional segment highlights."""
    if not st.session_state.waveform_data:
        return None

    audio_data = np.array(st.session_state.waveform_data)
    time = np.linspace(0, len(audio_data) / RATE, len(audio_data))
    audio_normalized = audio_data / (np.max(np.abs(audio_data)) + 1e-10)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time,
        y=audio_normalized,
        mode='lines',
        name='Waveform',
        line=dict(color='lightgray', width=1),
        fill='tozeroy',
        fillcolor='rgba(200, 200, 200, 0.3)'
    ))

    for i, emotion_ts in enumerate(st.session_state.emotion_timestamps):
        start_time = emotion_ts['time']
        end_time = start_time + 0.5
        start_idx = int(start_time * RATE)
        end_idx = min(int(end_time * RATE), len(audio_normalized))

        if start_idx < len(audio_normalized) and end_idx > start_idx:
            segment_time = time[start_idx:end_idx]
            segment_audio = audio_normalized[start_idx:end_idx]
            color_hex = EMOTION_COLORS.get(emotion_ts['emotion'], '#000000')
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)

            fig.add_trace(go.Scatter(
                x=segment_time,
                y=segment_audio,
                mode='lines',
                name=emotion_ts['emotion'],
                line=dict(color=color_hex, width=2),
                fill='tozeroy',
                fillcolor=f"rgba({r}, {g}, {b}, 0.5)",
                showlegend=(i == 0)
            ))

    fig.update_layout(
        title="Live Speech Emotion Analysis",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        plot_bgcolor='rgba(240, 240, 240, 0.8)'
    )

    return fig


def create_emotional_insights():
    """Create pie chart and timeline for emotional patterns. Returns (fig_pie, fig_timeline)."""
    if not st.session_state.emotion_timestamps:
        return None, None

    emotions = [ts['emotion'] for ts in st.session_state.emotion_timestamps]
    emotion_counts = pd.Series(emotions).value_counts()

    fig_pie = px.pie(
        values=emotion_counts.values,
        names=emotion_counts.index,
        title="Emotion Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig_timeline = go.Figure()
    for emotion in set(emotions):
        times = [ts['time'] for ts in st.session_state.emotion_timestamps if ts['emotion'] == emotion]
        intensities = [ts['intensity'] for ts in st.session_state.emotion_timestamps if ts['emotion'] == emotion]

        fig_timeline.add_trace(go.Scatter(
            x=times,
            y=intensities,
            mode='markers',
            name=emotion,
            marker=dict(
                color=EMOTION_COLORS.get(emotion, '#000000'),
                size=10,
                opacity=0.7
            )
        ))

    fig_timeline.update_layout(
        title="Emotion Intensity Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Intensity",
        height=300
    )

    return fig_pie, fig_timeline


def create_intent_analysis_chart():
    """Create bar chart of Rasa-detected intents."""
    if not st.session_state.detected_intents:
        return None

    intents_df = pd.DataFrame(st.session_state.detected_intents)
    intent_counts = intents_df['intent'].value_counts()

    fig = px.bar(
        x=intent_counts.index,
        y=intent_counts.values,
        title="Detected Intents (Rasa NLU)",
        labels={'x': 'Intent', 'y': 'Count'},
        color=intent_counts.index,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(height=250)
    return fig