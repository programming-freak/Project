import os
import tempfile

import streamlit as st

from detector import DeepfakeInference


st.set_page_config(
    page_title="ISTVT Deepfake Detector",
    page_icon="🎥",
    layout="centered"
)


@st.cache_resource
def load_detector():
    return DeepfakeInference()


detector = load_detector()


st.title("🎥 ISTVT Deepfake Detector")

st.markdown(
    """
Upload a video and the ISTVT model will classify it as **REAL** or **FAKE**.
"""
)

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov", "mkv"]
)


if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button("Run Detection"):

        with st.spinner("Analyzing video..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_file:

                temp_file.write(uploaded_file.read())

                temp_path = temp_file.name

            result = detector.predict(temp_path)

            os.remove(temp_path)

        if not result["success"]:

            st.error(result["message"])

        else:

            label = result["label"]
            confidence = result["confidence"]

            st.subheader("Prediction")

            if label == "FAKE":

                st.error(f"🚨 {label}")

            else:

                st.success(f"✅ {label}")

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )
