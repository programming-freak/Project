import torch
import cv2
import numpy as np

from models.full_model import DeepfakeDetector
from video_inference import extract_frames
from face_extractor import extract_faces



device = "cuda" if torch.cuda.is_available() else "cpu"



# -------------------------
# Load model
# -------------------------

model = DeepfakeDetector()


model.load_state_dict(
    torch.load(
        "deepfake_detector.pth",
        map_location=device
    )
)


model.to(device)
model.eval()



# -------------------------
# GradCAM storage
# -------------------------

activations = {}
gradients = {}



def forward_hook(module, inp, output):

    activations["value"] = output



def backward_hook(module, grad_in, grad_out):

    gradients["value"] = grad_out[0]



# -------------------------
# Automatically find last Conv2d
# -------------------------

target_layer = None


for layer in model.modules():

    if isinstance(
        layer,
        torch.nn.Conv2d
    ):
        target_layer = layer



print(
    "GradCAM layer:",
    target_layer
)



target_layer.register_forward_hook(
    forward_hook
)


target_layer.register_full_backward_hook(
    backward_hook
)




# -------------------------
# Detect video
# -------------------------

def detect_video(video_path):


    # Frames

    frames = extract_frames(
        video_path
    )


    print(
        "Frames:",
        frames.shape
    )



    # Face crop

    faces = extract_faces(
        frames
    )


    print(
        "Faces:",
        faces.shape
    )



    # Model input

    video = faces.unsqueeze(0)


    video = video.to(device)


    video.requires_grad = True


    print(
        "Model input:",
        video.shape
    )



    # Prediction

    output = model(video)



    pred = output.argmax(
        dim=1
    )


    confidence = torch.softmax(
        output,
        dim=1
    )[0][pred].item()*100



    print(
        "Prediction:",
        pred.item()
    )


    print(
        "Confidence:",
        confidence
    )



    # -------------------------
    # Backward GradCAM
    # -------------------------


    model.zero_grad()


    score = output[0,pred]


    score.backward()



    acts = activations["value"]

    grads = gradients["value"]



    print(
        "Activation:",
        acts.shape
    )

    print(
        "Gradient:",
        grads.shape
    )



    # -------------------------
    # GradCAM
    # -------------------------


    weights = grads.mean(
        dim=(2,3),
        keepdim=True
    )


    cam = (
        weights * acts
    ).sum(dim=1)



    cam = torch.relu(cam)



    cam = cam.detach().cpu().numpy()



    # frame importance

    frame_scores = cam.mean(
        axis=(1,2)
    )


    important_frame = np.argmax(
        frame_scores
    )


    print(
        "Frame scores:",
        frame_scores
    )


    print(
        "Most suspicious frame:",
        important_frame+1
    )



    # -------------------------
    # Heatmap
    # -------------------------


    cam_frame = cam[
        important_frame
    ]


    cam_frame -= cam_frame.min()


    cam_frame /= (
        cam_frame.max()+1e-8
    )



    cam_frame = cv2.resize(
        cam_frame,
        (224,224)
    )



    img = faces[
        important_frame
    ]


    img = img.permute(
        1,2,0
    ).cpu().numpy()



    img = np.clip(
        img,
        0,
        1
    )



    heatmap = cv2.applyColorMap(
        np.uint8(cam_frame*255),
        cv2.COLORMAP_JET
    )


    heatmap = (
        heatmap[:,:,::-1]
        /
        255.0
    )



    overlay = (
        0.5*heatmap +
        0.5*img
    )



    overlay = np.uint8(
        overlay*255
    )



    cv2.imwrite(
        "heatmap.jpg",
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )
    )



    print(
        "Saved heatmap.jpg"
    )





detect_video(
    "data/videos/249_280.mp4"
)