import torch
from torchvision import datasets,transform
import torch.nn as nn

global somenum = 100

class foodNN(nn.Module):
    def __init__(self) -> None:
        super.__init__()
        # we will need 101 output nodes in the final convolution layer
        self.convolution = nn.Seqential(
            nn.conv2d(1,somenum)
            nn.Relu()
            nn.MaxPool2d(2)
            nn.conv2d(somenum,somenum,),
            nn.Relu()
            nn.MaxPool2d(2)
            nn.conv2d(somenum,somenum),
            nn.Relu()
            nn.MaxPool2d(2)
            nn.conv2d(somenum,somenum),
            nn.Relu()
            nn.MaxPool2d(2)
            nn.conv2d(somenum,101),
            nn.Relu()
            nn.MaxPool2d(2)
        )
        self.final_layer = nn.Sequential(
        )

    def forward(self,x):
        return 


