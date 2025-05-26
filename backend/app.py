from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import cv2
import threading
import os
import joblib
import pandas as pd
from combined import record_audio, record_video_and_compute_blink_rate, extract_audio_features

app = Flask(__name__)
CORS(app)

# --- Load models and constants ---
AUDIO_FILENAME = "recorded_audio.wav"
VIDEO_FILENAME = "recorded_video.avi"
CSV_FILE = "test_cases.csv"
DURATION = 15

audio_model = joblib.load('parkinson_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')
blink_model = joblib.load('blink_model.pkl')
age_model = joblib.load('age_model.pkl')

# --- Global webcam reference for streaming ---
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        age = int(data.get('age', 0))

        if not name or age <= 0:
            return jsonify({'error': 'Invalid input'}), 400

        # --- Record audio and video ---
        blink_rate_container = [None]

        audio_thread = threading.Thread(target=record_audio, args=(AUDIO_FILENAME, DURATION))
        video_thread = threading.Thread(target=record_video_and_compute_blink_rate, args=(VIDEO_FILENAME, DURATION, blink_rate_container))

        audio_thread.start()
        video_thread.start()
        audio_thread.join()
        video_thread.join()

        blink_rate = blink_rate_container[0]
        if blink_rate == -1.0 or blink_rate is None:
            return jsonify({'error': 'Blink rate detection failed'}), 500

        # --- Audio features ---
        audio_features = extract_audio_features(AUDIO_FILENAME)
        if all(v == -1.0 for v in audio_features.values()):
            return jsonify({'error': 'Audio feature extraction failed'}), 500

        row = {
            "name": name,
            "Age": age,
            "Blink Rate": round(blink_rate, 2),
            **audio_features
        }

        if not os.path.isfile(CSV_FILE):
            pd.DataFrame([row]).to_csv(CSV_FILE, index=False)
        else:
            pd.DataFrame([row]).to_csv(CSV_FILE, mode='a', header=False, index=False)

        # --- Inference ---
        df = pd.read_csv(CSV_FILE)
        i = len(df) - 1
        audio_input = scaler.transform([df.loc[i, feature_names]])
        audio_proba = audio_model.predict_proba(audio_input)[0][1]
        blink_proba = blink_model.predict_proba([[df.loc[i, 'Blink Rate']]])[0][1]
        age_proba = age_model.predict_proba([[df.loc[i, 'Age']]])[0][1]

        fused_proba = 0.5 * audio_proba + 0.3 * blink_proba + 0.2 * age_proba
        prediction = 1 if fused_proba >= 0.5 else 0

        # Clean up
        for f in [AUDIO_FILENAME, VIDEO_FILENAME]:
            if os.path.exists(f):
                os.remove(f)

        return jsonify({
            "name": name,
            "prediction": prediction,
            "fused_proba": fused_proba,
            "audio_proba": audio_proba,
            "blink_proba": blink_proba,
            "age_proba": age_proba,
            "blink_rate": blink_rate
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
