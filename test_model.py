import torch

from models.full_model import DeepfakeDetector


model = DeepfakeDetector()


x = torch.randn(
    1,
    6,
    3,
    224,
    224
)


out = model(x)


print(out.shape)