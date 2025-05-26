import streamlit as st
import pandas as pd
import threading
import os
import joblib
import io
import contextlib

from combined import record_audio, record_video_and_compute_blink_rate, extract_audio_features

# Constants
AUDIO_FILENAME = "recorded_audio.wav"
VIDEO_FILENAME = "recorded_video.avi"
CSV_FILE = "test_cases.csv"
DURATION = 15

# Load models
audio_model = joblib.load('parkinson_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')
blink_model = joblib.load('blink_model.pkl')
age_model = joblib.load('age_model.pkl')

# Helper: Redirect print statements to Streamlit
@contextlib.contextmanager
def st_stdout(output_area):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield
    output_area.text(buffer.getvalue())

# Streamlit UI
st.set_page_config(page_title="Parkinson's Detector", layout="centered")
st.title("🧠 Parkinson’s Detection via Multimodal Biomarkers")

with st.form("user_info"):
    name = st.text_input("👤 Patient Name")
    age = st.number_input("🎂 Patient Age", min_value=1, max_value=120)
    submitted = st.form_submit_button("🎙️ Start Recording")

if submitted:
    log_output = st.empty()  # Dynamic log area
    result_output = st.container()  # Where we'll show results

    with st_stdout(log_output):
        st.info("Recording in progress...")

        blink_rate_container = [None]
        audio_thread = threading.Thread(target=record_audio, args=(AUDIO_FILENAME, DURATION))
        video_thread = threading.Thread(target=record_video_and_compute_blink_rate, args=(VIDEO_FILENAME, DURATION, blink_rate_container))

        audio_thread.start()
        video_thread.start()
        audio_thread.join()
        video_thread.join()

        print("[INFO] Audio and video recording finished.")

        # Extract features
        blink_rate = blink_rate_container[0]
        audio_features = extract_audio_features(AUDIO_FILENAME)

        if blink_rate == -1 or all(v == -1.0 for v in audio_features.values()):
            print("[ERROR] Feature extraction failed. Check model files or dependencies.")
            st.error("Feature extraction failed.")
        else:
            # Save to CSV
            row = {"name": name, "Age": age, "Blink Rate": round(blink_rate, 2)}
            row.update(audio_features)
            df = pd.DataFrame([row])

            if not os.path.exists(CSV_FILE):
                df.to_csv(CSV_FILE, index=False)
            else:
                df.to_csv(CSV_FILE, mode='a', header=False, index=False)

            print("[INFO] Features saved to CSV.")

            # Run predictions
            audio_scaled = scaler.transform(df[feature_names])
            audio_proba = audio_model.predict_proba(audio_scaled)[0][1]
            blink_proba = blink_model.predict_proba([[blink_rate]])[0][1]
            age_proba = age_model.predict_proba([[age]])[0][1]

            # Fusion logic
            blink_weight = 0.3
            age_weight = 0.2
            audio_weight = 1 - blink_weight - age_weight
            fused_proba = audio_weight * audio_proba + blink_weight * blink_proba + age_weight * age_proba
            final_pred = int(fused_proba >= 0.5)

            print("[INFO] Prediction complete.")
            print(f" - Audio Probability: {audio_proba:.4f}")
            print(f" - Blink Probability: {blink_proba:.4f}")
            print(f" - Age Probability: {age_proba:.4f}")
            print(f" - Fused Probability: {fused_proba:.4f}")
            print(f" - Final Prediction: {'Parkinson\'s Detected (1)' if final_pred == 1 else 'No Parkinson\'s (0)'}")

            # Streamlit display
            with result_output:
                st.subheader("🔍 Prediction Results")
                st.metric("Audio Probability", f"{audio_proba:.4f}")
                st.metric("Blink Probability", f"{blink_proba:.4f}")
                st.metric("Age Probability", f"{age_proba:.4f}")
                st.metric("Fused Probability", f"{fused_proba:.4f}")
                st.success("🧠 Parkinson’s Detected (1)" if final_pred else "✅ No Parkinson’s (0)")

            # Clean up
            for f in [AUDIO_FILENAME, VIDEO_FILENAME]:
                if os.path.exists(f):
                    os.remove(f)
            print("[INFO] Temporary files cleaned up.")
