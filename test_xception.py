import torch

from models.cnn_encoder import CNNEncoder


model = CNNEncoder()


x = torch.randn(
    2,
    3,
    224,
    224
)


out = model(x)


print(out.shape)