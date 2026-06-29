import os
import cv2
import gdown
import torch
import numpy as np

from PIL import Image
from facenet_pytorch import MTCNN

from models.full_model import DeepfakeDetector


class DeepfakeInference:
    """
    Deepfake detector backend.

    Downloads the trained model automatically if it is
    not present and exposes a simple predict(video_path)
    API.
    """

    def __init__(
        self,
        model_path="best_model.pth",
        device=None
    ):

        # -----------------------------------------
        # Download model if missing
        # -----------------------------------------

        if not os.path.exists(model_path):

            print("Downloading trained model...")

            gdown.download(
                id="1rO1fWKf1TFs65TkKNZXVvzwh_hm9ePeq",
                output=model_path,
                quiet=False
            )

        # -----------------------------------------
        # Device
        # -----------------------------------------

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        print(f"Loading model on {self.device}...")

        # -----------------------------------------
        # Load Model
        # -----------------------------------------

        self.model = DeepfakeDetector()

        state_dict = torch.load(
            model_path,
            map_location=self.device
        )

        self.model.load_state_dict(state_dict)

        self.model.to(self.device)

        self.model.eval()

        # -----------------------------------------
        # Face Detector
        # -----------------------------------------

        self.mtcnn = MTCNN(
            image_size=224,
            margin=20,
            device=self.device
        )

        self.labels = [
            "REAL",
            "FAKE"
        ]

        print("Model loaded successfully.")

    ############################################################

    def process_video(self, video_path):

        cap = cv2.VideoCapture(video_path)

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:

            cap.release()

            return None

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            6
        ).astype(int)

        faces = []

        for idx in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                idx
            )

            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(frame)

            face = self.mtcnn(image)

            if face is not None:
                faces.append(face)

        cap.release()

        if len(faces) < 4:
            return None

        while len(faces) < 6:
            faces.append(faces[-1].clone())

        faces = torch.stack(faces)

        return faces.unsqueeze(0)

    ############################################################

    def predict(self, video_path):

        video_tensor = self.process_video(video_path)

        if video_tensor is None:

            return {
                "success": False,
                "message": "Face not detected in enough frames."
            }

        video_tensor = video_tensor.to(self.device)

        with torch.no_grad():

            output = self.model(video_tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            prediction = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = probabilities[
                0,
                prediction
            ].item()

        return {

            "success": True,

            "label": self.labels[prediction],

            "confidence": float(confidence)

        }
