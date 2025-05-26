import cv2
import dlib
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import time
import os
import csv
import pandas as pd
import parselmouth
from parselmouth.praat import call # Explicitly import call
from scipy.spatial import distance
# from scipy.stats import entropy # Imported but not used in this snippet
# import librosa # Imported but not used in this snippet
os.chdir(r"E:\Parkinson\backend")
# File paths
AUDIO_FILENAME = "recorded_audio.wav"
VIDEO_FILENAME = "recorded_video.avi"
DURATION = 15  # seconds (shortened for easier testing, adjust as needed)

# --- Audio Recording ---
def record_audio(filename=AUDIO_FILENAME, duration=DURATION, samplerate=44100):
    """Records audio for a specified duration and saves it to a file."""
    print("[INFO] Starting audio recording...")
    try:
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        sf.write(filename, audio_data, samplerate)
        print(f"[INFO] Audio recording complete. Saved to {filename}")
    except Exception as e:
        print(f"[ERROR] Audio recording failed: {e}")

# --- Blink Rate Calculation ---
def calculate_ear(eye):
    """Calculates the Eye Aspect Ratio (EAR) for a single eye."""
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = distance.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def record_video_and_compute_blink_rate(filename=VIDEO_FILENAME, duration=DURATION, result_container=None):
    """
    Records video, detects blinks, calculates blink rate, and saves the video.
    Stores the blink rate in result_container[0] if provided.
    """
    print("[INFO] Starting video recording and blink detection...")
    
    # IMPORTANT: You need to download the shape_predictor_68_face_landmarks.dat file
    # and place it in the same directory as this script, or provide the full path.
    # Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        print(f"[ERROR] Dlib shape predictor model not found at {predictor_path}")
        print("[INFO] Please download it and place it in the correct location.")
        if result_container is not None:
            result_container[0] = -1.0 # Indicate error
        return -1.0

    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
    except Exception as e:
        print(f"[ERROR] Failed to load dlib detector or predictor: {e}")
        if result_container is not None:
            result_container[0] = -1.0
        return -1.0

    # Define the start and end facial landmark indexes for the left and right eyes
    (lStart, lEnd) = (42, 48)
    (rStart, rEnd) = (36, 42)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        if result_container is not None:
            result_container[0] = -1.0
        return -1.0
        
    # Define the codec and create VideoWriter object
    # Use a common resolution; ensure your webcam supports it.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

    blink_count = 0
    ear_threshold = 0.21      # Threshold for EAR to consider an eye closed
    ear_consec_frames = 2     # Number of consecutive frames the eye must be closed for a blink
    consecutive_frames_below_thresh = 0 # Counter for consecutive frames below EAR threshold

    start_time = time.time()
    frames_processed = 0

    print("[INFO] Webcam opened. Recording video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Could not read frame from webcam. Ending video recording.")
            break
        
        out.write(frame) # Save the frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0) # Detect faces

        for rect in rects:
            shape = predictor(gray, rect) # Get facial landmarks
            shape_np = np.array([[p.x, p.y] for p in shape.parts()]) # Convert to NumPy array

            left_eye = shape_np[lStart:lEnd]
            right_eye = shape_np[rStart:rEnd]

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0 # Average EAR

            # Check if EAR is below threshold (eye is closing/closed)
            if ear < ear_threshold:
                consecutive_frames_below_thresh += 1
            else:
                # If eye was closed for sufficient frames, count it as a blink
                if consecutive_frames_below_thresh >= ear_consec_frames:
                    blink_count += 1
                consecutive_frames_below_thresh = 0 # Reset counter

            # (Optional) Draw eye landmarks for visualization
            # for (x, y) in left_eye:
            #     cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            # for (x, y) in right_eye:
            #     cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            # cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            # cv2.putText(frame, f"Blinks: {blink_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)


        # cv2.imshow("Frame", frame) # Uncomment to see live video feed
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

        frames_processed += 1
        if time.time() - start_time >= duration:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows() # Ensure all OpenCV windows are closed

    actual_duration_seconds = time.time() - start_time
    if actual_duration_seconds == 0: # Avoid division by zero
        blink_rate = 0.0
    else:
        blink_rate = blink_count / (actual_duration_seconds / 60.0) # Blinks per minute

    print(f"[INFO] Video recording complete. Processed {frames_processed} frames in {actual_duration_seconds:.2f}s.")
    print(f"[INFO] Total blinks: {blink_count}. Blink Rate: {blink_rate:.2f} blinks/min. Saved to {filename}")
    
    if result_container is not None:
        result_container[0] = blink_rate
    return blink_rate


import os
import parselmouth
from parselmouth.praat import call
import logging
from typing import Dict, Any, Union

# Configure logging for better error and warning messages [1, 2]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioFeatureExtractionError(Exception):
    """Custom exception for errors during audio feature extraction."""
    pass # [1]

def praat_to_float(value: Any, default_if_undefined: float = -1.0) -> float:
    """
    Converts a Praat output value to a float, handling '--undefined--' strings.

    Args:
        value: The value returned by a Parselmouth/Praat call.
        default_if_undefined: The default float value to return if the input is '--undefined--'.

    Returns:
        The converted float value, or default_if_undefined if conversion fails.
    """
    if isinstance(value, str) and value == '--undefined--':
        return default_if_undefined
    try:
        return float(value)
    except (ValueError, TypeError) as e:
        logging.warning(f"Failed to convert Praat value '{value}' to float: {e}. Returning default {default_if_undefined}.")
        return default_if_undefined

def extract_audio_features(
    filename: str,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0,
    time_step: float = 0.01,
    jitter_shimmer_min_period: float = 0.0001,
    jitter_shimmer_max_period: float = 0.02,
    jitter_shimmer_max_amp_factor: float = 1.3,
    shimmer_max_amp_factor_db: float = 1.6,
    harmonicity_silence_threshold: float = 0.1,
    harmonicity_periods_per_window: float = 1.0,
    rpde_dfa_min_pitch: float = 75.0,
    rpde_dfa_max_period_factor: float = 3.0,
    rpde_dfa_min_amp: float = 0.1,
    rpde_dfa_max_amp: float = 1.0,
    dfa_min_freq: float = 0.01,
    dfa_max_freq: float = 2.0,
    d2_max_freq: float = 2.0,
    ppe_max_amp: float = 1.0
) -> Dict[str, float]:
    """
    Extracts a comprehensive set of audio features from a WAV file using Parselmouth.

    This function calculates various voice features including fundamental frequency (Fo),
    jitter, shimmer, harmonic-to-noise ratio (HNR), and nonlinear dynamic features
    (RPDE, DFA, spread1, spread2, D2, PPE). It incorporates robust error handling,
    input validation, and numerical stability considerations.

    Args:
        filename: Path to the WAV audio file. [3]
        pitch_floor: Minimum pitch (Hz) for pitch analysis.
        pitch_ceiling: Maximum pitch (Hz) for pitch analysis.
        time_step: Time step (s) for pitch and harmonicity analysis.
        jitter_shimmer_min_period: Minimum period for jitter/shimmer calculations.
        jitter_shimmer_max_period: Maximum period for jitter/shimmer calculations.
        jitter_shimmer_max_amp_factor: Maximum amplitude factor for jitter/shimmer.
        shimmer_max_amp_factor_db: Maximum amplitude factor (dB) for shimmer.
        harmonicity_silence_threshold: Silence threshold for harmonicity.
        harmonicity_periods_per_window: Periods per window for harmonicity.
        rpde_dfa_min_pitch: Minimum pitch for RPDE/DFA calculations.
        rpde_dfa_max_period_factor: Maximum period factor for RPDE/DFA.
        rpde_dfa_min_amp: Minimum amplitude for RPDE/DFA.
        rpde_dfa_max_amp: Maximum amplitude for RPDE/DFA.
        dfa_min_freq: Minimum frequency for DFA.
        dfa_max_freq: Maximum frequency for DFA.
        d2_max_freq: Maximum frequency for D2.
        ppe_max_amp: Maximum amplitude for PPE.

    Returns:
        A dictionary containing the extracted audio features. Values are -1.0 if
        a feature could not be calculated or an error occurred.

    Raises:
        AudioFeatureExtractionError: If the audio file is not found, or if
                                     critical input parameters are invalid.
    """
    feature_keys = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
        "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
        "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
        "spread1", "spread2", "D2", "PPE"
    ]
    error_result = {key: -1.0 for key in feature_keys}

    # --- Input Validation (Preconditions) --- [4, 5, 6]
    if not isinstance(filename, str) or not filename:
        logging.error("Invalid filename: Must be a non-empty string.")
        raise AudioFeatureExtractionError("Invalid filename provided.")
    if not os.path.exists(filename):
        logging.error(f"Audio file not found: {filename}")
        raise AudioFeatureExtractionError(f"Audio file not found: {filename}")

    # Validate numerical parameters [4, 5]
    numeric_params = {
        "pitch_floor": pitch_floor, "pitch_ceiling": pitch_ceiling, "time_step": time_step,
        "jitter_shimmer_min_period": jitter_shimmer_min_period, "jitter_shimmer_max_period": jitter_shimmer_max_period,
        "jitter_shimmer_max_amp_factor": jitter_shimmer_max_amp_factor,
        "shimmer_max_amp_factor_db": shimmer_max_amp_factor_db,
        "harmonicity_silence_threshold": harmonicity_silence_threshold,
        "harmonicity_periods_per_window": harmonicity_periods_per_window,
        "rpde_dfa_min_pitch": rpde_dfa_min_pitch, "rpde_dfa_max_period_factor": rpde_dfa_max_period_factor,
        "rpde_dfa_min_amp": rpde_dfa_min_amp, "rpde_dfa_max_amp": rpde_dfa_max_amp,
        "dfa_min_freq": dfa_min_freq, "dfa_max_freq": dfa_max_freq,
        "d2_max_freq": d2_max_freq, "ppe_max_amp": ppe_max_amp
    }
    for param_name, value in numeric_params.items():
        if not isinstance(value, (int, float)) or value < 0: # Most parameters should be non-negative
            logging.error(f"Invalid parameter type or value for '{param_name}': {value}. Must be a non-negative number.")
            raise AudioFeatureExtractionError(f"Invalid parameter: '{param_name}' must be a non-negative number.")

    # Specific range/relation checks [5]
    if not (0 < time_step < 1.0): # Time step usually small positive fraction
        logging.warning(f"Time step {time_step} is outside typical range (0, 1.0).")
    if not (0 < pitch_floor < pitch_ceiling):
        logging.error(f"Invalid pitch range: pitch_floor ({pitch_floor}) must be less than pitch_ceiling ({pitch_ceiling}) and both must be positive.")
        raise AudioFeatureExtractionError("Invalid pitch range parameters.")
    if not (0 < jitter_shimmer_min_period < jitter_shimmer_max_period):
        logging.error(f"Invalid jitter/shimmer period range: min_period ({jitter_shimmer_min_period}) must be less than max_period ({jitter_shimmer_max_period}) and both must be positive.")
        raise AudioFeatureExtractionError("Invalid jitter/shimmer period parameters.")

    snd: parselmouth.Sound = None
    try:
        snd = parselmouth.Sound(filename)
    except parselmouth.PraatError as e: # Catch specific Praat errors [1]
        logging.error(f"Praat error loading sound file {filename}: {e}")
        return error_result
    except Exception as e: # Catching other potential issues during sound loading [1]
        logging.error(f"Unexpected error loading sound file {filename}: {e}")
        return error_result

    extracted_data: Dict[str, float] = error_result.copy() # Initialize with error values

    try:
        # Pitch and PointProcess objects
        pitch = snd.to_pitch(time_step=time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
        # Note: Parselmouth's to_pointprocess uses the sound object's pitch_floor/ceiling if not specified
        point_process = call([snd, pitch], "To PointProcess (cc)", pitch_floor, pitch_ceiling)

        # Fundamental Frequency (Fo) features
        extracted_data = praat_to_float(call(pitch, "Get mean", 0, 0, "Hertz"))
        extracted_data = praat_to_float(call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic"))
        extracted_data = praat_to_float(call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic"))

        # Jitter features
        extracted_data = praat_to_float(call(point_process, "Get jitter (local)", jitter_shimmer_min_period, jitter_shimmer_max_period, jitter_shimmer_max_amp_factor)) * 100
        extracted_data = praat_to_float(call(point_process, "Get jitter (local, absolute)", jitter_shimmer_min_period, jitter_shimmer_max_period, jitter_shimmer_max_amp_factor))
        extracted_data = praat_to_float(call(point_process, "Get jitter (rap)", jitter_shimmer_min_period, jitter_shimmer_max_period, jitter_shimmer_max_amp_factor)) * 100
        extracted_data = praat_to_float(call(point_process, "Get jitter (ppq5)", jitter_shimmer_min_period, jitter_shimmer_max_period, jitter_shimmer_max_amp_factor)) * 100
        extracted_data = praat_to_float(call(point_process, "Get jitter (ddp)", jitter_shimmer_min_period, jitter_shimmer_max_period, jitter_shimmer_max_amp_factor)) * 100

        # Shimmer features
        extracted_data = praat_to_float(call([snd, point_process], "Get shimmer (local)", jitter_shimmer_min_period, jitter_shimmer_max_period, jitter_shimmer_max_amp_factor, shimmer_max_amp_factor_db)) * 100
        extracted_data = praat_to_float(call([snd, point_process], "Get shimmer (local, dB)", jitter_shimmer_min_period, jitter_shimmer_max_period, jitter_shimmer_max_amp_factor, shimmer_max_amp_factor_db))
        extracted_data = praat_to_float(call([snd, point_process], "Get shimmer (apq3)", jitter_shimmer_min_period, jitter_shimmer_max_period, jitter_shimmer_max_amp_factor, shimmer_max_amp_factor_db)) * 100
        extracted_data = praat_to_float(call([snd, point_process], "Get shimmer (apq5)", jitter_shimmer_min_period, jitter_shimmer_max_period, jitter_shimmer_max_amp_factor, shimmer_max_amp_factor_db)) * 100
        extracted_data = praat_to_float(call([snd, point_process], "Get shimmer (apq11)", jitter_shimmer_min_period, jitter_shimmer_max_period, jitter_shimmer_max_amp_factor, shimmer_max_amp_factor_db)) * 100
        extracted_data = praat_to_float(call([snd, point_process], "Get shimmer (dda)", jitter_shimmer_min_period, jitter_shimmer_max_period, jitter_shimmer_max_amp_factor, shimmer_max_amp_factor_db)) * 100

        # Harmonicity features
        harmonicity_obj = snd.to_harmonicity_cc(time_step=time_step, minimum_pitch=pitch_floor, silence_threshold=harmonicity_silence_threshold, periods_per_window=harmonicity_periods_per_window)
        hnr = call(harmonicity_obj, "Get mean", 0, 0)

        nhr_val: float = -1.0
        if isinstance(hnr, (int, float)):
            if hnr > 0: # Avoid division by zero or negative hnr in log scale [7, 8, 9, 10]
                try:
                    nhr_val = 1 / (10 ** (hnr / 10))
                except OverflowError: # [11]
                    logging.warning(f"OverflowError calculating NHR for HNR={hnr}. Setting NHR to -1.0.")
                    nhr_val = -1.0
                except ZeroDivisionError: # Should not happen with hnr > 0, but as a safeguard [7, 8, 9, 10]
                    logging.warning(f"ZeroDivisionError calculating NHR for HNR={hnr}. Setting NHR to -1.0.")
                    nhr_val = -1.0
            elif hnr == 0:
                nhr_val = 0.0 # Represents infinite NHR, often approximated as 0 or a very large number
            else: # hnr is negative, which might indicate an issue or specific Praat output
                logging.warning(f"HNR value is negative ({hnr}). NHR calculation might be invalid. Setting NHR to -1.0.")
                nhr_val = -1.0
        elif isinstance(hnr, str) and hnr == '--undefined--':
            nhr_val = -1.0
        else:
            logging.warning(f"Unexpected HNR value type or format: {hnr}. Setting NHR to -1.0.")
            nhr_val = -1.0

        extracted_data = praat_to_float(nhr_val)
        extracted_data = praat_to_float(hnr)

        # Nonlinear dynamic features
        try:
            extracted_data = praat_to_float(call(snd, "To RPDE", rpde_dfa_min_pitch, rpde_dfa_max_period_factor, rpde_dfa_min_amp, rpde_dfa_max_amp))
            extracted_data = praat_to_float(call(snd, "To DFA", rpde_dfa_min_pitch, dfa_min_freq, dfa_max_freq))
            extracted_data["spread1"] = praat_to_float(call(snd, "To AFV", rpde_dfa_min_pitch, rpde_dfa_min_amp, rpde_dfa_max_amp, 1, 1))
            extracted_data["spread2"] = praat_to_float(call(snd, "To AFV", rpde_dfa_min_pitch, rpde_dfa_min_amp, rpde_dfa_max_amp, 1, 2))
            extracted_data = praat_to_float(call(snd, "To D2", rpde_dfa_min_pitch, rpde_dfa_min_amp, d2_max_freq))
            extracted_data["PPE"] = praat_to_float(call(snd, "To PPE", rpde_dfa_min_pitch, rpde_dfa_min_amp, ppe_max_amp))
        except parselmouth.PraatError as e_nl_praat:
            logging.warning(f"Praat error calculating one or more nonlinear dynamic features: {e_nl_praat}. Features set to -1.0.")
        except Exception as e_nl_general: # Avoid silent failure, log the error [1, 2]
            logging.warning(f"Unexpected error calculating one or more nonlinear dynamic features: {e_nl_general}. Features set to -1.0.")

        logging.info("Audio feature extraction complete.")
        return extracted_data

    except parselmouth.PraatError as e_praat: # Catch specific Praat errors [1]
        logging.error(f"A Praat error occurred during audio feature extraction: {e_praat}")
        return error_result
    except Exception as e_general: # Catch any other unexpected errors [1]
        logging.error(f"An unexpected error occurred during audio feature extraction: {e_general}")
        return error_result
# --- Run Everything Together ---
def main():
    CSV_FILE="test_cases.csv"
    print("[INFO] Starting multimodal data collection and analysis...")

    # Get patient info
    patient_name = input("Enter patient name: ").strip()
    age = input("Enter patient age: ").strip()

    # Validate age
    try:
        age = int(age)
    except ValueError:
        print("[ERROR] Invalid age. Please enter a valid number.")
        return

    # Container to hold the blink rate
    blink_rate_container = [None]

    # Start recording
    audio_thread = threading.Thread(target=record_audio, args=(AUDIO_FILENAME, DURATION))
    video_thread = threading.Thread(target=record_video_and_compute_blink_rate, 
                                    args=(VIDEO_FILENAME, DURATION, blink_rate_container))

    audio_thread.start()
    video_thread.start()
    audio_thread.join()
    video_thread.join()

    blink_rate_value = blink_rate_container[0]
    audio_features = extract_audio_features(AUDIO_FILENAME)

    print("\n--- Combined Extracted Features ---")
    if audio_features:
        print("Audio Features:")
        for k, v in audio_features.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    else:
        print("[ERROR] Audio feature extraction failed.")
        return

    if blink_rate_value is not None:
        print(f"Blink Rate (blinks per minute): {blink_rate_value:.2f}")
    else:
        print("[ERROR] Blink rate calculation failed.")
        return

    # Prepare CSV row
    row = {
        "name": patient_name,
        "Age": age,
        "Blink Rate": round(blink_rate_value, 2)
    }
    audio_features.pop("status", None)  
    row.update(audio_features)


    # Write to CSV
    if not os.path.isfile(CSV_FILE):
        # If the file doesn't exist, write headers first
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)
    else:
        # Append if file exists
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            writer.writerow(row)

    print(f"[INFO] Appended results to {CSV_FILE}.")

    # Clean up
    for f in [AUDIO_FILENAME, VIDEO_FILENAME]:
        if os.path.exists(f):
            os.remove(f)
    print("[INFO] Temporary files removed.")

if __name__ == '__main__':
    main()
