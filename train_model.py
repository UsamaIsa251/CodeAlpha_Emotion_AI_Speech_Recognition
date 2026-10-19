
import os
import glob
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = "data/audio"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# RAVDESS emotion labels
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def extract_features(file_path):
    audio, sample_rate = librosa.load(
        file_path, sr=16000, duration=3.0
    )

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sample_rate, n_mfcc=40
    )

    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
    ])


def main():
    files = glob.glob(
        os.path.join(DATA_DIR, "**", "*.wav"),
        recursive=True,
    )

    features = []
    labels = []

    print(f"Found {len(files)} audio files.")

    for index, file_path in enumerate(files, start=1):
        filename = os.path.basename(file_path)
        parts = filename.split("-")

        if len(parts) < 7 or parts[0] != "03":
            continue

        emotion_code = parts[2]

        if emotion_code not in EMOTIONS:
            continue

        try:
            features.append(extract_features(file_path))
            labels.append(EMOTIONS[emotion_code])

            if index % 10 == 0:
                print(f"Processed {index}/{len(files)} files")

        except Exception as error:
            print(f"Skipped {filename}: {error}")

    if len(features) < 10:
        print("Not enough valid audio files to train.")
        return

    X = np.asarray(features)
    y = np.asarray(labels)

    print(f"\nUsable recordings: {len(y)}")
    print("Emotion counts:")
    for emotion in sorted(set(y)):
        print(f"  {emotion}: {np.sum(y == emotion)}")

    # Use stratification only if every class has enough examples.
    unique, counts = np.unique(y, return_counts=True)
    stratify_labels = y if np.all(counts >= 2) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify_labels,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=2,
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nAccuracy:", round(accuracy_score(y_test, predictions), 3))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    model_path = os.path.join(
        OUTPUT_DIR, "emotion_recognition_model.joblib"
    )
    joblib.dump(model, model_path)

    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
