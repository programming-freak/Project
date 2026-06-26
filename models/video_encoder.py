import torch
import torch.nn as nn

from models.cnn_encoder import CNNEncoder




class VideoEncoder(nn.Module):


    def __init__(self):

        super().__init__()


        self.cnn = CNNEncoder()



    def forward(self,x):

        B,T,C,H,W = x.shape


        x = x.view(
            B*T,
            C,
            H,
            W
        )


        features = self.cnn(x)


        features = features.view(
            B,
            T,
            -1
        )


        return features