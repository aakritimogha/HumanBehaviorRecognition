import streamlit as st
import base64
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
import time

from cam_handler import CameraHandler
from pose_behavior import BehaviorDetector
from emotion_engine import EmotionDetector
from dashboard_metrics import AnalyticsTracker

st.set_page_config(page_title="Human Behavior & Emotion Recognition", layout="wide")

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

icon_path = r"C:\Users\asus\Downloads\freepik__multi_emotions_person.png"
icon_base64 = get_base64_encoded_image(icon_path)

def add_custom_styles():
    st.markdown("""
    <style>
    /* Animated pastel gradient background for main content */
    .stApp {
        background: linear-gradient(270deg, #ffd6e8, #dceefb, #d0f0c0, #fce1a8);
        background-size: 400% 400%;
        animation: pastelGradient 20s ease infinite;
    }

    /* Animated purple-pink gradient background for sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(270deg, #d291bc, #f7c6c7, #d291bc, #f7c6c7);
        background-size: 400% 400%;
        animation: purplePinkGradient 20s ease infinite;
    }

    @keyframes pastelGradient {
        0% {background-position:0% 50%;}
        50% {background-position:100% 50%;}
        100% {background-position:0% 50%;}
    }

    @keyframes purplePinkGradient {
        0% {background-position:0% 50%;}
        50% {background-position:100% 50%;}
        100% {background-position:0% 50%;}
    }

    /* Sidebar text styling */
    [data-testid="stSidebar"] * {
        color: #4b004b !important;
        font-size: 14px !important;
        font-weight: 600;
    }

    /* Main heading animated style */
    .animated-title {
        font-size: 36px !important;
        font-weight: bold;
        animation: darkColorPulse 6s infinite;
        margin-bottom: 0;
        color: #4b004b;
    }

    @keyframes darkColorPulse {
        0% {color: #7a007a;}
        50% {color: #4b004b;}
        100% {color: #7a007a;}
    }

    /* Smaller headings */
    .small-heading {
        font-size: 18px !important;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        color: #5a005a;
    }

    /* Button styling */
    div.stButton > button {
        color: #4b004b !important;
        font-weight: 700;
    }

    button[kind="primary"] {
        background-color: #f7c6c7 !important;
        border-radius: 5px !important;
        padding: 8px 16px !important;
        font-size: 12px !important;
        cursor: pointer !important;
        transition: background-color 0.3s ease !important;
        color: #4b004b !important;
    }
    button[kind="primary"]:hover {
        background-color: #d291bc !important;
    }
    button[kind="secondary"] {
        background-color: #d291bc !important;
        color: #350035 !important;
        border-radius: 5px !important;
        padding: 8px 16px !important;
        font-size: 12px !important;
        cursor: pointer !important;
        transition: background-color 0.3s ease !important;
    }
    button[kind="secondary"]:hover {
        background-color: #ac2277 !important;
    }
    </style>
    """, unsafe_allow_html=True)

add_custom_styles()

st.markdown(f'''
<h1 class="animated-title">
  <img src="data:image/png;base64,{icon_base64}" style="width:96px; height:96px; vertical-align:middle; margin-right:12px;">
  Human Behavior & Emotion Recognition System
</h1>
''', unsafe_allow_html=True)

st.write("Welcome to the app!")

st.sidebar.title("Sidebar Menu")
threshold = st.sidebar.slider("Select Threshold", 0, 100, 50)

if st.sidebar.button("▶️ Start Detection", key="start", type="primary"):
    st.session_state.running = True
    st.session_state.camera_handler.initialize_camera(0)

if st.sidebar.button("⏹️ Stop Detection", key="stop", type="secondary"):
    st.session_state.running = False
    st.session_state.camera_handler.release_camera()

if st.sidebar.button("🔄 Reset Analytics", key="reset", type="secondary"):
    st.session_state.analytics = AnalyticsTracker()

if st.sidebar.button("📤 Export Session Data", key="export", type="secondary"):
    session_data = st.session_state.analytics.export_session_data()
    st.download_button("Download JSON", data=str(session_data), file_name="session_data.json")

st.sidebar.checkbox("Enable Feature")

if 'analytics' not in st.session_state:
    st.session_state.analytics = AnalyticsTracker()
if 'camera_handler' not in st.session_state:
    st.session_state.camera_handler = CameraHandler()
if 'behavior_detector' not in st.session_state:
    st.session_state.behavior_detector = BehaviorDetector()
if 'emotion_detector' not in st.session_state:
    st.session_state.emotion_detector = EmotionDetector()
if 'running' not in st.session_state:
    st.session_state.running = False

def main():
    duration = datetime.now() - st.session_state.analytics.session_start
    st.caption(f"🕒 Session Duration: {round(duration.total_seconds() / 60, 2)} min")

    with st.sidebar:
        st.header("📹 Camera Settings")
        camera_index = st.selectbox("Select Camera", [0, 1, 2], index=0)

        st.header("🎯 Detection Thresholds")
        behavior_threshold = st.slider("Behavior Confidence", 0.0, 1.0, 0.5, 0.1)
        emotion_threshold = st.slider("Emotion Confidence", 0.0, 1.0, 0.6, 0.1)

        st.header("🖼️ Display Options")
        show_landmarks = st.checkbox("Show Pose Landmarks", True)
        show_face_landmarks = st.checkbox("Show Face Landmarks", True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<h2 class="small-heading">📷 Live Camera Feed</h2>', unsafe_allow_html=True)
        video_placeholder = st.empty()
        detection_col1, detection_col2 = st.columns(2)
        with detection_col1:
            behavior_placeholder = st.empty()
        with detection_col2:
            emotion_placeholder = st.empty()

    with col2:
        st.markdown('<h2 class="small-heading">📊 Real-time Analytics</h2>', unsafe_allow_html=True)
        stats_placeholder = st.empty()
        behavior_chart_placeholder = st.empty()
        emotion_chart_placeholder = st.empty()
        activity_placeholder = st.empty()

    if st.session_state.running:
        process_video_stream(
            video_placeholder, behavior_placeholder, emotion_placeholder,
            stats_placeholder, behavior_chart_placeholder, emotion_chart_placeholder,
            activity_placeholder, behavior_threshold, emotion_threshold,
            show_landmarks, show_face_landmarks
        )

def process_video_stream(video_placeholder, behavior_placeholder, emotion_placeholder,
                         stats_placeholder, behavior_chart_placeholder, emotion_chart_placeholder,
                         activity_placeholder, behavior_threshold, emotion_threshold,
                         show_landmarks, show_face_landmarks):
    frame_count = 0
    fps_counter = time.time()

    while st.session_state.running:
        frame = st.session_state.camera_handler.get_frame()
        if frame is None:
            video_placeholder.error("❌ Camera not available.")
            break

        processed_frame = frame.copy()

        try:
            behavior_result = st.session_state.behavior_detector.detect(frame)
            if behavior_result and behavior_result['confidence'] >= behavior_threshold:
                st.session_state.analytics.add_behavior_detection(
                    behavior_result['behavior'], behavior_result['confidence']
                )
                if show_landmarks and behavior_result.get('landmarks'):
                    processed_frame = st.session_state.behavior_detector.draw_landmarks(
                        processed_frame, behavior_result['landmarks']
                    )

            emotion_result = st.session_state.emotion_detector.detect(frame)
            if emotion_result and emotion_result['confidence'] >= emotion_threshold:
                st.session_state.analytics.add_emotion_detection(
                    emotion_result['emotion'], emotion_result['confidence']
                )
                if show_face_landmarks and emotion_result.get('landmarks'):
                    processed_frame = st.session_state.emotion_detector.draw_landmarks(
                        processed_frame, emotion_result['landmarks']
                    )

            video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
            update_current_detections(behavior_placeholder, emotion_placeholder, behavior_result, emotion_result)

            if frame_count % 30 == 0:
                update_analytics_display(stats_placeholder, behavior_chart_placeholder,
                                        emotion_chart_placeholder, activity_placeholder, frame_count)

            fps = 1 / (time.time() - fps_counter)
            fps_counter = time.time()
            st.sidebar.metric("FPS", f"{fps:.1f}")

            frame_count += 1
            time.sleep(0.033)

        except Exception as e:
            st.error(f"Detection error: {str(e)}")

def update_current_detections(behavior_placeholder, emotion_placeholder, behavior_result, emotion_result):
    with behavior_placeholder.container():
        st.subheader("🏃 Current Behavior")
        if behavior_result:
            color = "green" if behavior_result['confidence'] > 0.7 else "orange"
            st.markdown(f"**Detected:** {behavior_result['behavior'].title()}  \n"
                        f"**Confidence:** <span style='color:{color}'>{behavior_result['confidence']:.2f}</span>",
                        unsafe_allow_html=True)
        else:
            st.write("*No behavior detected*")

    with emotion_placeholder.container():
        st.subheader("😊 Current Emotion")
        emotion_emoji = {
            'happy': '😊', 'sad': '😢', 'surprised': '😲', 'neutral': '😐',
            'sleepy': '😴', 'cry': '😭', 'flu': '🤒', 'smoking': '🚬', 'unknown': '❔'
        }
        if emotion_result:
            color = "green" if emotion_result['confidence'] > 0.7 else "orange"
            emoji = emotion_emoji.get(emotion_result['emotion'], '😐')
            st.markdown(f"{emoji} **{emotion_result['emotion'].title()}**  \n"
                        f"**Confidence:** <span style='color:{color}'>{emotion_result['confidence']:.2f}</span>",
                        unsafe_allow_html=True)
        else:
            st.write("*No emotion detected*")

def update_analytics_display(stats_placeholder, behavior_chart_placeholder,
                            emotion_chart_placeholder, activity_placeholder, frame_count):
    analytics = st.session_state.analytics

    with stats_placeholder.container():
        st.subheader("📈 Session Stats")
        stats = analytics.get_session_stats()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Detections", stats['total_detections'])
            st.metric("Session Duration", f"{stats['duration_minutes']:.1f} min")
        with col2:
            st.metric("Behaviors Detected", stats['unique_behaviors'])
            st.metric("Emotions Detected", stats['unique_emotions'])

    with behavior_chart_placeholder.container():
        st.subheader("🏃 Behavior Distribution")
        behavior_data = analytics.get_behavior_distribution()
        if behavior_data:
            df = pd.DataFrame(list(behavior_data.items()), columns=['Behavior', 'Count'])
            fig = px.pie(df, values='Count', names='Behavior', title="Behavior Distribution")
            st.plotly_chart(fig, use_container_width=True, key=f"behavior_chart_{frame_count}")
        else:
            st.write("*No behavior data yet*")

    with emotion_chart_placeholder.container():
        st.subheader("😊 Emotion Distribution")
        emotion_data = analytics.get_emotion_distribution()
        if emotion_data:
            df = pd.DataFrame(list(emotion_data.items()), columns=['Emotion', 'Count'])
            colors = {'happy': '#FFD700', 'sad': '#4169E1', 'surprised': '#FF6347', 'neutral': '#808080'}
            fig = px.bar(df, x='Emotion', y='Count', color='Emotion', color_discrete_map=colors, title="Emotion Distribution")
            st.plotly_chart(fig, use_container_width=True, key=f"emotion_chart_{frame_count}")
        else:
            st.write("*No emotion data yet*")

    with activity_placeholder.container():
        st.subheader("⏱️ Recent Activity")
        recent = st.session_state.analytics.get_recent_activity(10)
        if recent:
            df = pd.DataFrame(recent)
            st.dataframe(df, use_container_width=True)
        else:
            st.write("*No recent activity*")

if __name__ == "__main__":
    main()






