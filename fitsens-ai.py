import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import os
import platform
from PIL import Image
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from pymongo import MongoClient
import bcrypt
import re
from datetime import datetime, timedelta

# --- MongoDB Setup (using Atlas or any remote server) ---
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["fitsens_ai"]
users_collection = db["users"]

# --- Password validation ---
def is_valid_password(password: str):
    if len(password) < 6 or len(password) > 13:
        return False, "Password must be between 6–13 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter (A–Z)."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter (a–z)."
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number (0–9)."
    if not re.search(r"[^A-Za-z0-9]", password):
        return False, "Password must contain at least one special character (e.g., @, #, $, %, &)."
    return True, ""

# --- Helper functions ---
def create_user(username, password):
    if users_collection.find_one({"username": username}):
        return False, "Username already exists."
    valid, msg = is_valid_password(password)
    if not valid:
        return False, msg
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    users_collection.insert_one({"username": username, "password": hashed_pw, "daily_reps": {}, "streak_count": 0, "last_active_date": None, "inactive_days": 0})
    return True, "User created successfully!"

def verify_user(username, password):
    user = users_collection.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode(), user["password"]):
        return True
    return False

def get_todays_reps(username):
    user = users_collection.find_one({"username": username})
    today_str = datetime.now().strftime("%Y-%m-%d")
    if not user:
        return 0
    daily_reps = user.get("daily_reps", {})
    return daily_reps.get(today_str, 0)

def update_reps_today(username, reps_done):
    today_str = datetime.now().strftime("%Y-%m-%d")
    user = users_collection.find_one({"username": username})
    if not user:
        return 0
    current_reps = get_todays_reps(username)
    new_reps = current_reps + reps_done
    users_collection.update_one(
        {"username": username},
        {"$set": {f"daily_reps.{today_str}": new_reps}}
    )
    return new_reps

def update_streak(username):
    user = users_collection.find_one({"username": username})
    if not user:
        return 0
    today = datetime.now().date()
    today_str = today.strftime("%Y-%m-%d")
    last_active_str = user.get("last_active_date")
    streak = user.get("streak_count", 0)
    inactive_days = user.get("inactive_days", 0)
    daily_reps = user.get("daily_reps", {})

    if last_active_str:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
        days_diff = (today - last_active).days
    else:
        days_diff = 0

    if days_diff >= 3:
        streak = 0
        inactive_days = 0

    if today_str in daily_reps and daily_reps[today_str] >= 50:
        if last_active_str != today_str:
            streak += 1
            inactive_days = 0
            users_collection.update_one(
                {"username": username},
                {"$set": {
                    "streak_count": streak,
                    "last_active_date": today_str,
                    "inactive_days": inactive_days
                }}
            )
    else:
        users_collection.update_one(
            {"username": username},
            {"$set": {"last_active_date": today_str, "inactive_days": inactive_days}}
        )

    return streak

# --- Login / Signup UI ---
st.set_page_config(page_title="FitSens-Ai | Login", page_icon="💪")

st.markdown("""
<style>
body, .main {
    background-color: #060914;
    color: #DDEBFF;
    font-family: 'Poppins', sans-serif;
}
.stTabs [data-baseweb="tab-list"] {
    justify-content: center;
    background: linear-gradient(90deg, #001F3F, #003366);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #9FCFFF;
    font-weight: 600;
}
.stTextInput>div>div>input {
    background-color: #0B1E3B;
    color: white;
    border: 1px solid #007BFF;
    border-radius: 8px;
    padding: 6px;
}
.stButton>button {
    background: linear-gradient(90deg,#007BFF,#00C0FF);
    color: white;
    border-radius: 8px;
    padding: 8px 18px;
    font-weight: 600;
    box-shadow: 0 0 10px rgba(0,192,255,0.3);
}
.stButton>button:hover {
    transform: translateY(-2px);
    background: linear-gradient(90deg,#00C0FF,#007BFF);
}
</style>
""", unsafe_allow_html=True)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None

if not st.session_state.authenticated:
    st.markdown("<h1 style='text-align:center; color:#00C0FF;'>💪 FitSens-Ai</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color:#9FCFFF;'>Train Smart. Move Right. Live Fit.</h4>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔐 Login", "🆕 Sign Up"])
    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if verify_user(username, password):
                st.session_state.authenticated = True
                st.session_state.current_user = username
                st.session_state.total_reps_today = get_todays_reps(username)
                user_data = users_collection.find_one({"username": username})
                st.session_state.workout_streak = user_data.get("streak_count", 0)
                st.success(f"Welcome back, {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
    with tab2:
        st.markdown("""#### Password Rules
        - 6–13 characters  
        - At least one uppercase, lowercase, number, and special character""")
        new_username = st.text_input("Create Username")
        new_password = st.text_input("Create Password", type="password")
        if st.button("Sign Up"):
            success, msg = create_user(new_username, new_password)
            if success:
                st.success(msg)
            else:
                st.error(msg)
    st.stop()

# Optional libraries
try:
    import requests
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except Exception:
    LOTTIE_AVAILABLE = False

@st.cache_data
def load_data():
    return pd.read_csv("fitness_nutrition_extended.csv")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def compute_embeddings(questions, _model):
    return _model.encode(questions)

def get_best_answer(user_question, questions, answers, embeddings, model):
    user_emb = model.encode([user_question])
    sims = cosine_similarity(user_emb, embeddings)[0]
    idx = sims.argmax()
    return answers[idx]

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def safe_beep():
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000, 250)
        else:
            os.system("printf '\a'")
    except Exception:
        pass

def speak_text_nonblocking(text):
    def _worker(msg):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.setProperty("volume", 1.0)
            engine.say(str(msg))
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_worker, args=(text,), daemon=True).start()

if "global_cam_in_use" not in st.session_state:
    st.session_state.global_cam_in_use = False
if "rep_cam_run" not in st.session_state:
    st.session_state.rep_cam_run = False
if "pose_cam_run" not in st.session_state:
    st.session_state.pose_cam_run = False

if "rep_count" not in st.session_state:
    st.session_state.rep_count = 0
if "rep_stage" not in st.session_state:
    st.session_state.rep_stage = None

if "total_reps_today" not in st.session_state:
    st.session_state.total_reps_today = 0
if "workout_streak" not in st.session_state:
    st.session_state.workout_streak = 0

st.set_page_config(page_title="FitSens-Ai | Intelligent Fitness Assistant", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
body, .main {
    background-color: #060914;
    color: #DDEBFF;
    font-family: 'Poppins', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#071023,#0c2740);
    color: #BFDFFF;
    padding: 18px;
}
h1 { color:#00C0FF; text-shadow: 0 0 10px rgba(0,192,255,0.12); }
.stButton>button {
    background: linear-gradient(90deg,#007BFF,#00C0FF);
    color:white;
    border-radius:10px;
    padding:8px 18px;
    font-weight:600;
    box-shadow: 0 8px 20px rgba(0,192,255,0.08);
}
.stButton>button:hover { transform: translateY(-3px); }
.card {
    background: linear-gradient(180deg,#07102a,#0c2038);
    border-radius:12px;
    padding:14px;
    border:1px solid rgba(0,192,255,0.06);
    margin-bottom:12px;
}
[data-testid="stMetricValue"] { color:#00C6FF !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div style='text-align:center; padding-top:6px; padding-bottom:6px;'>"
            "<h1 style='font-size:40px; margin:0;'>💪 FitSens-Ai</h1>"
            "<div style='color:#A8D8FF; margin-top:4px;'>Your Intelligent Fitness Companion — Train Smart. Move Right. Live Fit.</div>"
            "</div>", unsafe_allow_html=True)

with st.sidebar:
    if LOTTIE_AVAILABLE:
        try:
            r = requests.get("https://assets2.lottiefiles.com/packages/lf20_iwmd6pyr.json", timeout=2)
            if r.status_code == 200:
                st_lottie(r.json(), height=120)
        except Exception:
            pass

    st.markdown("## Quick Metrics")
    st.metric("Today's Reps", st.session_state.total_reps_today)
    user_data = users_collection.find_one({"username": st.session_state.current_user})
    streak_value = user_data.get("streak_count", 0) if user_data else 0
    st.metric("Workout Streak (days)", streak_value)

    st.divider()
    st.markdown("## Settings")
    enable_tts = st.checkbox("Enable voice feedback (TTS)", value=True)
    st.markdown("Use only one camera feature at a time. Stop a webcam before starting another.")

tabs = st.tabs(["🏠 Home", "🏋️ Repetition Counter", "🧍 Pose Correction", "📏 Body Ratio & Diet", "💬 Chatbot", "📈 Analytics"])

# --- Home Tab ---
with tabs[0]:
    left, right = st.columns([2,1])
    with left:
        st.subheader("What FitSens-Ai does")
        st.write("FitSens-Ai uses your webcam (or uploaded photos) to count reps, correct posture, and recommend diet & workouts by analysing the body ratio.")
        st.markdown("- Prevent injuries with instant posture correction\n- Improve training efficiency with accurate rep counts\n- Get personalized plans based on body proportions and BMI")
        st.info("Tip: Allow camera access in your browser. Use a clear full-body frame and good lighting for best results.")
    with right:
        st.markdown("<div class='card'><h4 style='color:#A8E7FF; margin:4px 0;'>Ready to train?</h4><p style='color:#DDEBFF; margin:0;'>Choose a module, start your camera, and follow the on-screen guidance.</p></div>", unsafe_allow_html=True)

# --- Repetition Counter Tab ---
with tabs[1]:
    st.header("🏋️ Repetition Counter")
    c1, c2 = st.columns([2, 1])
    with c1:
        exercise = st.selectbox("Select Exercise", ["Select", "Bicep curls", "Squats"])
        start_btn = st.button("Start Camera")
        stop_btn = st.button("Stop Camera")
        voice_toggle = st.checkbox("Voice Count (per rep)", value=True) if enable_tts else st.checkbox("Voice Count (disabled in Settings)", value=False, disabled=True)
        video_placeholder = st.empty()
    with c2:
        st.markdown(
            "<div class='card'><h4 style='color:#9fe6ff'>Positioning Tips</h4>"
            "<ul style='margin-left:16px; color:#DDEBFF'>"
            "<li>Place camera at chest height</li>"
            "<li>Keep full body visible</li>"
            "<li>Good lighting improves detection</li>"
            "</ul></div>",
            unsafe_allow_html=True
        )

    if start_btn:
        if exercise == "Select":
            st.warning("Please choose an exercise first.")
        elif st.session_state.global_cam_in_use and not st.session_state.rep_cam_run:
            st.warning("Releasing old camera...")
            st.session_state.global_cam_in_use = False
            time.sleep(1)
        elif st.session_state.global_cam_in_use:
            st.error("Webcam already in use by another module. Stop it first.")
        else:
            st.session_state.rep_cam_run = True
            st.session_state.global_cam_in_use = True
            st.success("Repetition counter started.")

    if stop_btn:
        st.session_state.rep_cam_run = False
        st.success("Stopping repetition counter...")

    if st.session_state.rep_cam_run:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if platform.system() == "Windows" else cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access webcam. Ensure it is free and permitted.")
            st.session_state.rep_cam_run = False
            st.session_state.global_cam_in_use = False
        else:
            st.session_state.rep_count = 0
            st.session_state.rep_stage = None
            try:
                while st.session_state.rep_cam_run:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read camera frame.")
                        break
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb)
                    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    if results.pose_landmarks:
                        lm = results.pose_landmarks.landmark

                        # BOTH SIDES: handle left and right limbs
                        if exercise == "Bicep curls":
                            # Right arm
                            shoulder_r = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                            elbow_r = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
                            wrist_r = [lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y]
                            angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)

                            # Left arm
                            shoulder_l = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                            elbow_l = [lm[mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y]
                            wrist_l = [lm[mp_pose.PoseLandmark.LEFT_WRIST].x, lm[mp_pose.PoseLandmark.LEFT_WRIST].y]
                            angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)

                            # choose whichever side is doing a curl
                            if angle_r > 150:
                                st.session_state.rep_stage = "down_r"
                            if angle_r < 40 and st.session_state.rep_stage == "down_r":
                                st.session_state.rep_stage = "up_r"
                                st.session_state.rep_count += 1
                                st.session_state.total_reps_today = update_reps_today(st.session_state.current_user, 1)
                                if voice_toggle and enable_tts:
                                    speak_text_nonblocking(st.session_state.rep_count)

                            if angle_l > 150:
                                st.session_state.rep_stage = "down_l"
                            if angle_l < 40 and st.session_state.rep_stage == "down_l":
                                st.session_state.rep_stage = "up_l"
                                st.session_state.rep_count += 1
                                st.session_state.total_reps_today = update_reps_today(st.session_state.current_user, 1)
                                if voice_toggle and enable_tts:
                                    speak_text_nonblocking(st.session_state.rep_count)

                        elif exercise == "Squats":
                            hip_r = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
                            knee_r = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
                            ankle_r = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
                            angle_r = calculate_angle(hip_r, knee_r, ankle_r)

                            hip_l = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]
                            knee_l = [lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y]
                            ankle_l = [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                            angle_l = calculate_angle(hip_l, knee_l, ankle_l)

                            if angle_r > 160 or angle_l > 160:
                                st.session_state.rep_stage = "up_sq"
                            if (angle_r < 90 or angle_l < 90) and st.session_state.rep_stage == "up_sq":
                                st.session_state.rep_stage = "down_sq"
                                st.session_state.rep_count += 1
                                st.session_state.total_reps_today = update_reps_today(st.session_state.current_user, 1)
                                if voice_toggle and enable_tts:
                                    speak_text_nonblocking(st.session_state.rep_count)

                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    cv2.rectangle(frame, (0, 0), (300, 120), (8, 16, 24), -1)
                    label = exercise if exercise != "Select" else ""
                    cv2.putText(frame, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (170, 220, 255), 2)
                    cv2.putText(frame, str(st.session_state.rep_count), (12, 95), cv2.FONT_HERSHEY_SIMPLEX, 2.8, (255, 255, 255), 6)
                    video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                    time.sleep(0.03)
            finally:
                cap.release()
                cv2.destroyAllWindows()
                st.session_state.rep_cam_run = False
                st.session_state.global_cam_in_use = False
                st.success("Repetition counter stopped.")

                if st.session_state.total_reps_today >= 50:
                    new_streak = update_streak(st.session_state.current_user)
                    st.session_state.workout_streak = new_streak
                    st.success(f"🎯 Great job! You've maintained your {new_streak}-day streak!")
                else:
                    st.info("Complete at least 50 reps today to continue your streak.")

# --- Pose Correction Tab ---
with tabs[2]:
    st.header("🧍 AI Posture Correction Assistant")
    st.write("Upload an image or use your webcam to analyze posture and detect slouching automatically.")
    import pyttsx3
    def speak(text):
        def _run():
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 150)
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()

    def beep():
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(1000, 400)
            else:
                os.system("printf '\a'")
        except Exception:
            pass

    def analyze_posture(image, pose):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        feedback = "No landmarks detected"
        advice = ""
        injury_risk = ""
        back_angle = None
        state = None

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            h, w, _ = image.shape
            shoulder_left = (
                int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
            )
            hip_left = (
                int(lm[mp_pose.PoseLandmark.LEFT_HIP].x * w),
                int(lm[mp_pose.PoseLandmark.LEFT_HIP].y * h)
            )
            hip_right = (
                int(lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                int(lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h)
            )

            back_angle = calculate_angle(shoulder_left, hip_left, hip_right)

            if back_angle >= 118.19:
                feedback = "⚠️ Slouched Back Detected!"
                advice = "Try standing tall, engage your core, and pull your shoulders slightly back."
                injury_risk = "Prolonged slouching can cause back pain and spinal misalignment."
                state = "bad"
            else:
                feedback = "✅ Good Posture"
                advice = "Maintain this upright posture with shoulders aligned over hips."
                injury_risk = "No Risk! This posture reduces strain on your spine."
                state = "good"

            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return image, feedback, advice, injury_risk, back_angle, state

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    option = st.radio("Choose Input Source", ["Upload Image", "Use Webcam"], key="input_option")
    stframe = st.empty()
    info = st.empty()

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="upload_image")
        if uploaded_file is not None:
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_local:
                img = np.array(Image.open(uploaded_file))
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                processed_img, feedback, advice, injury_risk, back_angle, state = analyze_posture(img_bgr.copy(), pose_local)
                stframe.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption=feedback, use_container_width=True)
                st.markdown(f"**Back Angle:** {back_angle if back_angle else 'N/A'}°")
                st.info(f"💡 {advice}")
                st.error(f"⚠️ {injury_risk}")
                if state == "bad":
                    beep()
                    speak("Slouched back detected")
                elif state == "good":
                    speak("Good posture detected")
    elif option == "Use Webcam":
        start_cam = st.button("▶️ Start Camera")
        stop_cam = st.button("⏹️ Stop Camera")
        if "camera_running" not in st.session_state:
            st.session_state.camera_running = False
        if start_cam:
            st.session_state.camera_running = True
        if stop_cam:
            st.session_state.camera_running = False
        if st.session_state.camera_running:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if platform.system() == "Windows" else cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Camera not accessible.")
            else:
                last_state = None
                last_speak_time = 0
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_local:
                    while st.session_state.camera_running:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Camera not accessible.")
                            break
                        frame = cv2.flip(frame, 1)
                        processed_frame, feedback, advice, injury_risk, back_angle, state = analyze_posture(frame.copy(), pose_local)
                        stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption=feedback, channels="RGB", use_container_width=True)
                        info.markdown(f"**Back Angle:** {back_angle if back_angle else 'N/A'}°  \n💡 {advice}  \n⚠️ {injury_risk}")
                        now = time.time()
                        if state and state != last_state and now - last_speak_time > 2:
                            if state == "bad":
                                beep()
                                speak("Slouched back detected")
                            elif state == "good":
                                speak("Good posture detected")
                            last_state = state
                            last_speak_time = now
                        time.sleep(0.05)
                cap.release()
                st.success("Camera stopped.")

# --- Body Ratio & Diet Tab ---
with tabs[3]:
    st.header("📏 Body Ratio & Diet Recommendation")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)

    def get_detailed_plan(body_type):
        plans = {
            "Ectomorph": {
                "diet": "Their diet should be high in calories and protein, including foods like eggs, rice, chicken or paneer, oats, nuts, and milk to promote muscle gain. Eating every 3–4 hours, staying hydrated, and getting 7–8 hours of sleep daily help optimize growth and recovery.",
                "workout": "Ectomorphs should focus on heavy compound workouts like squats, bench presses, and deadlifts 4–5 days a week, keeping reps between 8–12 and avoiding excessive cardio to prevent calorie loss."
            },
            "Mesomorph": {
                "diet": "Their diet should include a moderate calorie intake with balanced macros — lean proteins (chicken, fish, eggs), complex carbs (brown rice, oats), and healthy fats (nuts, olive oil). Staying consistent with workouts, maintaining portion control, and ensuring proper rest helps them build muscle while keeping body fat in check.",
                "workout": "Mesomorphs should combine strength training and moderate cardio 5 days a week, focusing on both compound lifts like squats and bench presses and accessory exercises for balance."
            },
            "Endomorph": {
                "diet": "Their diet should be calorie-controlled, high in protein, moderate in healthy fats, and low in refined carbs — including foods like eggs, lean meats, vegetables, nuts, and whole grains. Eating smaller, frequent meals, avoiding sugar, and staying consistent with workouts helps them stay lean and build muscle efficiently.",
                "workout": "Endomorphs should focus on a mix of strength training and higher-intensity cardio 5–6 days a week to boost metabolism and reduce body fat, emphasizing compound movements like deadlifts, squats, and HIIT sessions."
            }
        }
        return plans.get(body_type, {"diet": "No plan available", "workout": "No plan available"})

    def classify_body_type(height, weight):
        bmi = weight / ((height / 100) ** 2)
        if bmi < 18.5:
            return "Ectomorph"
        elif 18.5 <= bmi < 25:
            return "Mesomorph"
        else:
            return "Endomorph"

    def analyze_image(image):
        img_rgb = np.array(image)
        results = pose.process(img_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            shoulder_width = abs(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x -
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
            )
            hip_width = abs(
                landmarks[mp_pose.PoseLandmark.LEFT_HIP].x -
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x
            )
            ratio = shoulder_width / hip_width if hip_width > 0 else 0
            if 1.60 < ratio < 1.80:
                body_type = "Mesomorph"
            elif 1.20 <= ratio <= 1.60:
                body_type = "Endomorph"
            else:
                body_type = "Ectomorph"
            return body_type, ratio
        else:
            return "Unknown", None

    st.title("Diet and Workout Plan Recommender")
    mode = st.radio("Choose Mode", ["Select", "Height & Weight Input", "Photo Upload"])

    if mode == "Height & Weight Input":
        height = st.number_input("Enter height (cm)", min_value=100, max_value=250, step=1)
        weight = st.number_input("Enter weight (kg)", min_value=30, max_value=200, step=1)
        if st.button("Get Plan"):
            body_type = classify_body_type(height, weight)
            plan = get_detailed_plan(body_type)
            st.subheader(f"Body Type: {body_type}")
            st.markdown(f"**Diet Plan:** {plan['diet']}")
            st.markdown(f"**Workout Plan:** {plan['workout']}")

    elif mode == "Photo Upload":
        uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            if st.button("Analyze Image"):
                body_type, ratio = analyze_image(image)
                if body_type != "Unknown":
                    plan = get_detailed_plan(body_type)
                    st.subheader(f"Body Type: {body_type}")
                    st.markdown(f"**Diet Plan:** {plan['diet']}")
                    st.markdown(f"**Workout Plan:** {plan['workout']}")
                    st.info(f"Debug Info → Shoulder/Hip Ratio: {ratio:.2f}")
                else:
                    st.error("Could not detect body. Please use a clearer image.")

# --- Chatbot Tab ---
with tabs[4]:
    st.header("💬 Fitsens-AI Chatbot")
    st.write("Chat with FitSens-Ai 💡")

    import google.generativeai as genai
    API_KEY = "AIzaSyArE1etaEg593AAD1NpOrHZ1ZmBXJFkSxk"
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')

    if "gemini_chat" not in st.session_state:
        st.session_state.gemini_chat = model.start_chat(history=[])

    user_message = st.text_input("You:", placeholder="Ask me anything about fitness, diet, or workouts...")

    if st.button("Send"):
        if user_message.strip():
            response = st.session_state.gemini_chat.send_message(user_message)
            st.session_state.last_response = response.text
            st.markdown(f"**Fitsens-AI:** {response.text}")
            if enable_tts:
                speak_text_nonblocking(response.text)
        else:
            st.warning("Please enter a message to chat.")

    if "gemini_chat" in st.session_state:
        st.markdown("---")
        st.markdown("### 🗨️ Chat History")
        for msg in st.session_state.gemini_chat.history:
            role = "🧑‍💻 You" if msg.role == "user" else "🤖 Fitsens-AI"
            st.markdown(f"**{role}:** {msg.parts[0].text if msg.parts else ''}")

# --- Analytics Tab ---
with tabs[5]:
    st.header("📈 Analytics (Session)")
    st.write("Session-level demo analytics based on the local counters.")
    reps = st.session_state.total_reps_today
    streak = st.session_state.workout_streak

    col1, col2 = st.columns(2)
    col1.metric("Total Reps (today)", reps)
    col2.metric("Workout Streak (days)", streak)

    days = list(range(1,8))
    baseline = reps // 7
    weekly = [baseline]*7
    weekly[-1] = reps
    fig, ax = plt.subplots(figsize=(6,2.6))
    ax.plot(days, weekly, marker="o")
    ax.set_xlabel("Day")
    ax.set_ylabel("Reps")
    ax.set_title("Weekly reps (demo)")
    st.pyplot(fig)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#9fbefc; padding:10px;'>Made for Injury-Free Fit and Healthier Life ❤ • FitSens-Ai</div>", unsafe_allow_html=True)
