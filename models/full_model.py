import torch.nn as nn

from models.video_encoder import VideoEncoder
from models.istvt import ISTVT



class DeepfakeDetector(nn.Module):


    def __init__(self):

        super().__init__()


        self.cnn = VideoEncoder()


        self.transformer = ISTVT()


        self.classifier = nn.Linear(
            2048,
            2
        )



    def forward(self,x):


        # CNN spatial features

        features = self.cnn(x)


        # spatial-temporal transformer

        features, attention = self.transformer(
            features
        )


        # take video representation

        features = features.mean(
            dim=1
        )


        # real/fake prediction

        prediction = self.classifier(
            features
        )


        return prediction