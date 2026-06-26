import torch
import torch.nn as nn
import timm



class CNNEncoder(nn.Module):

    def __init__(self):

        super().__init__()


        self.cnn = timm.create_model(
            "xception",
            pretrained=True,
            num_classes=0
        )



    def forward(self,x):

        features = self.cnn(x)

        return features