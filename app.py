
import streamlit as st
import requests

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="centered",
)

API_URL = "http://127.0.0.1:8000/predict"

st.title("🎙️ Speech Emotion Recognition")
st.write(
    "Record your voice or upload a WAV recording. "
    "The CNN model will estimate the emotion expressed in the speech."
)

st.info(
    "Your browser may ask for microphone permission. "
    "Allow access to record your voice."
)

recording = st.audio_input("Record your voice")
uploaded_audio = st.file_uploader(
    "Or upload a WAV audio file",
    type=["wav"],
)

audio = recording if recording is not None else uploaded_audio

if audio is not None:
    st.audio(audio)

    if st.button("🔍 Predict Emotion", type="primary"):
        with st.spinner("Analyzing your audio with the CNN model..."):
            try:
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            "recording.wav",
                            audio.getvalue(),
                            "audio/wav",
                        )
                    },
                    timeout=120,
                )

                if response.ok:
                    result = response.json()
                    emotion = result["predicted_emotion"]
                    scores = result.get("category_scores", {})

                    st.success(
                        f"Predicted emotion: {emotion.replace('_', ' ').title()}"
                    )

                    st.subheader("Emotion scores")

                    for name, score in sorted(
                        scores.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    ):
                        score = min(max(float(score), 0.0), 1.0)
                        st.write(
                            f"**{name.replace('_', ' ').title()}** "
                            f"— {score * 100:.2f}%"
                        )
                        st.progress(score)

                    st.caption(
                        result.get(
                            "note",
                            "Predictions are estimates, not proof of a person's actual emotional state.",
                        )
                    )
                else:
                    st.error(
                        f"API error ({response.status_code}): "
                        f"{response.text}"
                    )

            except requests.RequestException as error:
                st.error(f"Could not connect to the API: {error}")
