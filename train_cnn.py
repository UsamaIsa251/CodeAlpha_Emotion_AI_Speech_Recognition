
import os
import glob
import json
import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "data/audio/Audio_Speech_Actors_01-24_16k"
OUTPUT_DIR = "outputs"

SAMPLE_RATE = 16000
N_MFCC = 40
MAX_FRAMES = 94
TEST_SIZE = 0.25
RANDOM_SEED = 42


# -----------------------------
# Extract MFCC features
# -----------------------------
def extract_features(file_path):
    """Extract a fixed-size MFCC sequence from one audio file."""
    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=3.0
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    ).T

    # Pad or truncate to a fixed number of frames.
    if mfcc.shape[0] < MAX_FRAMES:
        padding = MAX_FRAMES - mfcc.shape[0]
        mfcc = np.pad(
            mfcc,
            ((0, padding), (0, 0)),
            mode="constant"
        )
    else:
        mfcc = mfcc[:MAX_FRAMES, :]

    return mfcc.astype(np.float32)


# -----------------------------
# Build CNN model
# -----------------------------
def build_cnn(input_shape, number_of_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv1D(
            32, kernel_size=5, activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        tf.keras.layers.Conv1D(
            64, kernel_size=3, activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.GlobalAveragePooling1D(),

        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(
            number_of_classes,
            activation="softmax"
        ),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# Main training process
# -----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.random.seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # Limit CPU threads to help on a low-memory computer.
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    file_paths = glob.glob(
        os.path.join(DATA_DIR, "Actor_*", "*.wav")
    )
    file_paths.sort()

    if not file_paths:
        raise FileNotFoundError(
            f"No WAV files found in: {DATA_DIR}"
        )

    features = []
    labels = []
    actors = []

    emotion_names = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    print(f"Found {len(file_paths)} WAV files.")
    print("Extracting MFCC features...")

    for index, file_path in enumerate(file_paths, start=1):
        filename = os.path.basename(file_path)
        parts = filename.split("-")

        # RAVDESS filename format:
        # modality-emotion-intensity-statement-repetition-actor.wav
        if len(parts) != 7:
            continue

        # Use speech recordings only.
        if parts[0] != "03":
            continue

        emotion_code = parts[2]
        actor_id = parts[6].split(".")[0]

        if emotion_code not in emotion_names:
            continue

        try:
            mfcc = extract_features(file_path)
        except Exception as error:
            print(f"Skipping {filename}: {error}")
            continue

        features.append(mfcc)
        labels.append(emotion_names[emotion_code])
        actors.append(actor_id)

        if index % 100 == 0:
            print(f"Processed {index}/{len(file_paths)} files.")

    if len(features) == 0:
        raise ValueError(
            "No usable speech recordings were found. "
            "Check DATA_DIR and the dataset filenames."
        )

    X = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)
    actors = np.asarray(actors)

    # Encode emotion labels as integers.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    print(f"\nUsable recordings: {len(X)}")
    print(f"Feature shape: {X.shape}")
    print(f"Emotion classes: {list(label_encoder.classes_)}")

    # Keep each actor exclusively in either train or test.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=actors)
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Training recordings: {len(X_train)}")
    print(f"Testing recordings: {len(X_test)}")
    print(f"Training actors: {sorted(set(actors[train_idx]))}")
    print(f"Testing actors: {sorted(set(actors[test_idx]))}")

    # -----------------------------
    # Normalize using training data only
    # -----------------------------
    mean = X_train.mean(
        axis=(0, 1),
        keepdims=True
    )
    std = X_train.std(
        axis=(0, 1),
        keepdims=True
    )
    std = np.maximum(std, 1e-6)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Save normalization statistics for future predictions.
    np.savez(
        os.path.join(OUTPUT_DIR, "cnn_normalization.npz"),
        mean=mean,
        std=std
    )

    # -----------------------------
    # Train CNN
    # -----------------------------
    model = build_cnn(
        input_shape=X_train.shape[1:],
        number_of_classes=len(label_encoder.classes_)
    )

    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    min_delta=0.01,
    restore_best_weights=True
)

    print("\nTraining CNN...")

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.15,
        epochs=35,
        batch_size=16,
        callbacks=[early_stopping],
        verbose=1
    )

    # -----------------------------
    # Evaluate on the held-out actors
    # -----------------------------
    probabilities = model.predict(
        X_test,
        batch_size=16,
        verbose=0
    )

    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(label_encoder.classes_))
    )

    print("\n========== CNN RESULTS ==========")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Macro-F1 score: {macro_f1:.4f}")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix:")
    print(matrix)

    # -----------------------------
    # Save outputs
    # -----------------------------
    model_path = os.path.join(
        OUTPUT_DIR,
        "emotion_cnn_new.keras"
    )
    model.save(model_path)

    with open(
        os.path.join(OUTPUT_DIR, "emotion_classes.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            label_encoder.classes_.tolist(),
            file,
            indent=4
        )

    with open(
        os.path.join(OUTPUT_DIR, "cnn_evaluation.txt"),
        "w",
        encoding="utf-8"
    ) as file:
        file.write(f"Test accuracy: {accuracy:.4f}\n")
        file.write(f"Macro-F1 score: {macro_f1:.4f}\n\n")
        file.write("Classification report:\n")
        file.write(report)
        file.write("\nConfusion matrix:\n")
        file.write(np.array2string(matrix))

    np.savetxt(
        os.path.join(OUTPUT_DIR, "cnn_confusion_matrix.csv"),
        matrix,
        delimiter=",",
        fmt="%d"
    )

    print("\nTraining complete.")
    print(f"Model saved to: {model_path}")
    print(f"Evaluation saved to: {OUTPUT_DIR}/cnn_evaluation.txt")
    print(f"Normalization saved to: {OUTPUT_DIR}/cnn_normalization.npz")
    print(f"Class labels saved to: {OUTPUT_DIR}/emotion_classes.json")


if __name__ == "__main__":
    main()
