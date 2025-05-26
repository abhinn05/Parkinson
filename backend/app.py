from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import tempfile
from feature_extraction import extract_audio_features, record_video_and_compute_blink_rate  # your feature extraction functions

app = Flask(__name__)
CORS(app)

# Load your models once at startup
audio_model = joblib.load('parkinson_model.pkl')
scaler = joblib.load('scaler.pkl')
blink_model = joblib.load('blink_model.pkl')
age_model = joblib.load('age_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        audio_file = request.files.get('audio')
        image_file = request.files.get('image')
        age_str = request.form.get('age')
        print("Received age:", age_str, type(age_str))  # Debug print

        if age_str is None:
            return jsonify({'error': 'Missing age'}), 400

        try:
            age = float(age_str)
        except ValueError:
            return jsonify({'error': 'Invalid age value'}), 400


        # Validate inputs
        if not audio_file or not image_file or age_str is None:
            return jsonify({'error': 'Missing audio, image, or age'}), 400

        try:
            age = float(age_str)
        except ValueError:
            return jsonify({'error': 'Invalid age value'}), 400

        # Save files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as af:
            audio_path = af.name
            audio_file.save(audio_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as imf:
            image_path = imf.name
            image_file.save(image_path)

        # Extract features
        audio_features = extract_audio_features(audio_path)
        blink_rate = record_video_and_compute_blink_rate(image_path)

        # Prediction
        audio_scaled = scaler.transform([audio_features])
        audio_proba = audio_model.predict_proba(audio_scaled)[0][1]
        blink_proba = blink_model.predict_proba([[blink_rate]])[0][1]
        age_proba = age_model.predict_proba([[age]])[0][1]

        # Fuse probabilities
        blink_weight = 0.3
        age_weight = 0.2
        audio_weight = 1 - blink_weight - age_weight

        fused_proba = (audio_weight * audio_proba +
                       blink_weight * blink_proba +
                       age_weight * age_proba)

        prediction = 1 if fused_proba >= 0.5 else 0

        # Cleanup
        os.remove(audio_path)
        os.remove(image_path)

        return jsonify({
            'audio_proba': round(audio_proba, 4),
            'blink_proba': round(blink_proba, 4),
            'age_proba': round(age_proba, 4),
            'fused_proba': round(fused_proba, 4),
            'prediction': "Parkinson's Detected (1)" if prediction else "No Parkinson's (0)"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
