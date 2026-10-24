# Speech Emotion Recognition Using CNN

## Project Overview

This project is a Speech Emotion Recognition system that uses a Convolutional Neural Network (CNN) to estimate emotions expressed in speech recordings.

The system includes an HTML-based interactive frontend, a FastAPI backend, and a trained deep learning model. Users can record their voice through the original **Speech Emotion Intelligence** interface and view the predicted emotion with scores for the available emotion categories.

This project was developed as part of the CodeAlpha internship tasks.

## Features

- Interactive HTML interface with microphone recording.
- CNN-based speech emotion classification.
- FastAPI backend for processing audio and returning predictions.
- Emotion scores displayed as percentages.
- Audio feature extraction using MFCCs.
- Model evaluation using accuracy, precision, recall, F1-score, and a confusion matrix.
- A disclaimer explaining that predictions are estimates, not verified emotional states.

## Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- HTML, CSS, and JavaScript

## Dataset

The model was trained using the RAVDESS speech emotion dataset, containing 1,440 speech audio recordings from 24 actors.

The training pipeline extracts 40 Mel-Frequency Cepstral Coefficients (MFCCs) and prepares them as fixed-length feature sequences for the CNN.

## Model Architecture

The CNN uses the following main layers:

1. Conv1D with 32 filters
2. Batch Normalization
3. Max Pooling
4. Conv1D with 64 filters
5. Batch Normalization
6. Global Average Pooling
7. Dense layer with 32 units
8. Dropout
9. Softmax output layer for eight emotion categories

## Emotion Categories

- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised

## Model Evaluation

The retrained CNN achieved the following results on its held-out test set:

| Metric | Result |
|---|---:|
| Test accuracy | 37.22% |
| Macro-F1 score | 0.3467 |
| Test recordings | 360 |

These results indicate that the model is an experimental prototype and requires further improvement. Individual prediction scores are not the same as overall model accuracy.

## Project Structure

```text
CodeAlpha_EmotionRecognition/
├── index.html
├── api.py
├── app.py
├── predict_emotion.py
├── train_cnn.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── audio/
└── outputs/
    ├── emotion_cnn_retrained.keras
    ├── emotion_classes_retrained.json
    ├── cnn_normalization_retrained.npz
    ├── cnn_evaluation_retrained.txt
    ├── cnn_confusion_matrix_retrained.csv
    └── cnn_training_history_retrained.npz
```

## How to Run

### 1. Activate the Python environment

```bash
source ~/env/bin/activate
```

### 2. Navigate to the project directory

```bash
cd ~/CodeAlpha_EmotionRecognition
```

### 3. Start the API

```bash
PYTHONNOUSERSITE=1 python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Keep this terminal open while using the interface.

### 4. Open the frontend

Open `index.html` using VS Code Live Server or the local setup that you tested successfully.

Allow microphone access when prompted, record a short voice sample, and select the analysis button to view the prediction.

### 5. Check the API

The API health endpoint is:

```text
http://127.0.0.1:8000/health
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Limitations

- Test accuracy is currently 37.22%.
- Some emotions are more difficult for the model to distinguish than others.
- Predictions may vary with voice characteristics, recording quality, and speaking style.
- Model scores do not establish a person's actual feelings or psychological condition.
- The current API is configured for local use and is not a publicly deployed service.

## Future Improvements

- Improve generalization and classification performance.
- Investigate class confusion and feature extraction.
- Experiment with alternative CNN, LSTM, or hybrid architectures.
- Evaluate on additional speakers and recordings.
- Deploy the API securely for remote access.

## Conclusion

This project demonstrates an end-to-end speech emotion recognition workflow, from audio feature extraction and CNN training to API-based inference and an interactive HTML frontend. The current implementation provides a functional prototype and a foundation for further model improvement.
