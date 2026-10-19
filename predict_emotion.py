
import os
import sys
import json
import numpy as np
import librosa
import tensorflow as tf

MODEL_PATH = "outputs/emotion_cnn_new.keras"
CLASSES_PATH = "outputs/emotion_classes.json"
NORMALIZATION_PATH = "outputs/cnn_normalization.npz"
MAX_FRAMES = 94
N_MFCC = 40


def extract_features(file_path, mean, std):
    audio, sample_rate = librosa.load(
        file_path, sr=16000, duration=3.0
    )

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sample_rate, n_mfcc=N_MFCC
    ).T

    if len(mfcc) < MAX_FRAMES:
        mfcc = np.pad(
            mfcc,
            ((0, MAX_FRAMES - len(mfcc)), (0, 0))
        )
    else:
        mfcc = mfcc[:MAX_FRAMES, :]

    features = mfcc.astype(np.float32)[None, ...]
    return (features - mean) / np.maximum(std, 1e-6)


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_emotion.py path/to/audio.wav")
        return

    audio_path = sys.argv[1]
    if not os.path.isfile(audio_path):
        print(f"Audio file not found: {audio_path}")
        return

    for path in (MODEL_PATH, CLASSES_PATH, NORMALIZATION_PATH):
        if not os.path.isfile(path):
            print(f"Required file not found: {path}")
            return

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASSES_PATH, encoding="utf-8") as f:
        classes = json.load(f)

    stats = np.load(NORMALIZATION_PATH)
    mean = stats["mean"]
    std = stats["std"]

    features = extract_features(audio_path, mean, std)
    probabilities = model.predict(features, verbose=0)[0]

    if isinstance(classes, dict):
        labels = [classes.get(str(i), classes.get(i)) for i in range(len(probabilities))]
    else:
        labels = classes

    if len(labels) != len(probabilities) or any(label is None for label in labels):
        print("Error: emotion labels do not match model outputs.")
        return

    predicted_index = int(np.argmax(probabilities))
    print(f"Predicted emotion: {labels[predicted_index]}")
    print(f"Confidence: {probabilities[predicted_index]:.1%}")

    print("\nModel scores:")
    for emotion, score in sorted(
        zip(labels, probabilities),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{emotion}: {score:.1%}")


if __name__ == "__main__":
    main()
