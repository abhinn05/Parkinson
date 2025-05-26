import joblib
import numpy as np
import pandas as pd
import os

def fuse_probabilities(audio_p, blink_p, age_p, weights=(0.5, 0.3, 0.2)):
    audio_w, blink_w, age_w = weights
    return audio_w * audio_p + blink_w * blink_p + age_w * age_p


feature_names = joblib.load('feature_names.pkl')

def main():
    # Set working directory
    os.chdir(r"E:\Parkinson\backend")

    # Load saved models and tools
    audio_model = joblib.load('parkinson_model.pkl')
    scaler = joblib.load('scaler.pkl')
    blink_model = joblib.load('blink_model.pkl')
    age_model = joblib.load('age_model.pkl')

    # Load test cases
    df = pd.read_csv("test_cases.csv")

    # Extract features
    audio_features = df[feature_names]
    blink_rates = df["Blink Rate"]
    ages = df["Age"]

    # Scale audio features
    audio_scaled = scaler.transform(audio_features)

    # Predict probabilities in batch
    audio_probas = audio_model.predict_proba(audio_scaled)[:, 1]
    blink_probas = blink_model.predict_proba(blink_rates.values.reshape(-1, 1))[:, 1]
    age_probas = age_model.predict_proba(ages.values.reshape(-1, 1))[:, 1]

    # Fuse probabilities with weights
    blink_weight = 0.3
    age_weight = 0.2
    audio_weight = 1 - blink_weight - age_weight

    fused_probas = fuse_probabilities(audio_probas, blink_probas, age_probas,
                                      weights=(audio_weight, blink_weight, age_weight))
    fused_predictions = (fused_probas >= 0.5).astype(int)

    # Print results
    for i in range(len(df)):
        print(f"\n🔍 Test Case {i+1}")
        print(f" - Audio Probability: {audio_probas[i]:.4f}")
        print(f" - Blink Probability: {blink_probas[i]:.4f}")
        print(f" - Age Probability: {age_probas[i]:.4f}")
        print(f" - Fused Probability: {fused_probas[i]:.4f}")
        print(f" - Final Prediction: {'Parkinson\'s Detected (1)' if fused_predictions[i] == 1 else 'No Parkinson\'s (0)'}")

if __name__ == "__main__":
    main()
