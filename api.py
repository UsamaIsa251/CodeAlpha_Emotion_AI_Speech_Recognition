
import os
import json
import tempfile

import librosa
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = "outputs/emotion_cnn_new.keras"
CLASSES_PATH = "outputs/emotion_classes.json"
NORMALIZATION_PATH = "outputs/cnn_normalization.npz"

MAX_FRAMES = 94
N_MFCC = 40

app = FastAPI(
    title="Speech Emotion Recognition API",
    description="Predicts emotion categories from recorded speech using a CNN.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def load_resources():
    for path in (MODEL_PATH, CLASSES_PATH, NORMALIZATION_PATH):
        if not os.path.isfile(path):
            raise RuntimeError(f"Required model file not found: {path}")

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASSES_PATH, encoding="utf-8") as f:
        classes = json.load(f)

    stats = np.load(NORMALIZATION_PATH)
    mean = stats["mean"]
    std = np.maximum(stats["std"], 1e-6)

    if len(classes) != model.output_shape[-1]:
        raise RuntimeError("Emotion labels do not match CNN outputs.")

    return model, classes, mean, std


def extract_features(file_path, mean, std):
    audio, sample_rate = librosa.load(
        file_path,
        sr=16000,
        duration=3.0,
    )

    if len(audio) == 0:
        raise ValueError("The recording contains no usable audio.")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC,
    ).T

    if len(mfcc) < MAX_FRAMES:
        mfcc = np.pad(
            mfcc,
            ((0, MAX_FRAMES - len(mfcc)), (0, 0)),
        )
    else:
        mfcc = mfcc[:MAX_FRAMES, :]

    features = mfcc.astype(np.float32)[None, ...]
    return (features - mean) / std


@app.get("/")
def home():
    return {
        "message": "Speech Emotion Recognition API is running.",
        "model": "CNN",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model": "CNN"}


@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a WAV audio recording.",
        )

    temp_path = None

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded audio file is empty.",
            )

        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Audio file is too large. Maximum size is 10 MB.",
            )

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        model, classes, mean, std = load_resources()
        features = extract_features(temp_path, mean, std)
        probabilities = model.predict(features, verbose=0)[0]

        predicted_index = int(np.argmax(probabilities))
        prediction = classes[predicted_index]

        ranked_scores = {
            str(classes[i]): round(float(probabilities[i]), 4)
            for i in np.argsort(probabilities)[::-1]
        }

        return {
            "predicted_emotion": prediction,
            "category_scores": ranked_scores,
            "note": (
                "Model estimates are not proof of a person's actual "
                "emotional state."
            ),
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=422,
            detail=f"Could not process the audio: {error}",
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()
