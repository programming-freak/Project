import os
import time
import tempfile

import streamlit as st

from detector import DeepfakeInference

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="ISTVT Deepfake Detector",
    page_icon="🎥",
    layout="wide"
)

# ----------------------------------------------------
# Custom CSS
# ----------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.result-box{
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Load Model Only Once
# ----------------------------------------------------

@st.cache_resource
def load_detector():
    return DeepfakeInference()

detector = load_detector()

# ----------------------------------------------------
# Header
# ----------------------------------------------------

st.title("🎥 ISTVT Deepfake Detector")

st.markdown("""
Detect whether a video is **REAL** or **FAKE** using an
**ISTVT-inspired Spatial Temporal Deepfake Detection Network**.

---
""")

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

with st.sidebar:

    st.header("About")

    st.write("""
This application detects manipulated facial videos using an
ISTVT-inspired deep learning architecture.

**Pipeline**

- Video Upload
- Frame Sampling
- Face Detection (MTCNN)
- Deepfake Classification
    """)

# ----------------------------------------------------
# Upload
# ----------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a Video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file is not None:

    left, right = st.columns([1.2,1])

    with left:

        st.subheader("Uploaded Video")

        st.video(uploaded_file)

    with right:

        st.subheader("Detection")

        if st.button(
            "🚀 Run Detection",
            use_container_width=True
        ):

            progress = st.progress(0)

            start = time.time()

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as tmp:

                tmp.write(uploaded_file.read())

                temp_path = tmp.name

            with st.spinner("Running Deepfake Detection..."):

                progress.progress(20)

                result = detector.predict(temp_path)

                progress.progress(100)

            end = time.time()

            os.remove(temp_path)

            if not result["success"]:

                st.error(result["message"])

            else:

                label = result["label"]

                confidence = result["confidence"]*100

                elapsed = end-start

                if label=="REAL":

                    st.success("✅ REAL VIDEO")

                else:

                    st.error("🚨 FAKE VIDEO")

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.metric(
                    "Inference Time",
                    f"{elapsed:.2f} sec"
                )

# ----------------------------------------------------
# Expandable Section
# ----------------------------------------------------

with st.expander("ℹ About the Model"):

    st.write("""

This detector is inspired by the **ISTVT (Interpretable
Spatial Temporal Vision Transformer)** architecture.

The model:

- Samples frames from the uploaded video.
- Detects faces using MTCNN.
- Extracts spatial-temporal features.
- Predicts whether the video is manipulated.

""")

# ----------------------------------------------------
# Footer
# ----------------------------------------------------

st.markdown("---")

st.markdown(
"""
<div class="footer">

Developed using ❤️ with PyTorch & Streamlit

</div>
""",
unsafe_allow_html=True
)
