# EmotiVoice AI — Speech Emotion Recognition

An experimental AI/ML project that predicts emotion categories from speech audio.

## Features
- Browser-based audio recording and prediction
- MFCC feature extraction with Librosa
- CNN model built with TensorFlow/Keras
- FastAPI prediction endpoint
- Emotion scores and model evaluation report

## Emotion Categories
Angry, Calm, Disgust, Fearful, Happy, Neutral, Sad, Surprised.

## Technologies
Python, TensorFlow/Keras, Librosa, NumPy, scikit-learn, FastAPI, HTML, CSS, JavaScript.

## Main Files
- `api.py`: FastAPI backend
- `index.html`: browser frontend
- `predict_emotion.py`: command-line prediction
- `train_cnn.py`: CNN training and evaluation
- `train_model.py`: earlier model training script
- `outputs/`: trained models, labels, normalization data, and evaluation results

## Evaluation
The current CNN achieved 37.22% accuracy and 34.67% macro-F1 on 360 test samples from unseen actors. Performance needs improvement.

## Limitations
Predictions are experimental estimates, not verified feelings or a psychological assessment.

## Run Locally
Start the API with `uvicorn api:app --host 127.0.0.1 --port 8000`.
Open `http://127.0.0.1:8000/docs` to view the API documentation.

The frontend currently uses a local API address. Online deployment requires configuring a deployed backend URL.
