import torch
import torch.nn as nn


class TemporalDifference(nn.Module):

    def forward(self, x):

        # x:
        # batch, frames, channels, H, W

        differences=[]

        for i in range(x.shape[1]-1):

            diff = x[:,i+1] - x[:,i]

            differences.append(diff)


        differences = torch.stack(
            differences,
            dim=1
        )

        return differences