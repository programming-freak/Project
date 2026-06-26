import torch
import torch.nn as nn



class ISTVT(nn.Module):

    def __init__(self, dim=2048):

        super().__init__()


        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            batch_first=True
        )


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=8,
            batch_first=True
        )


        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )



    def forward(self, x):


        # x:
        # batch, frames, feature_dim


        attended, weights = self.attention(
            x,
            x,
            x
        )


        output = self.transformer(
            attended
        )


        return output, weights