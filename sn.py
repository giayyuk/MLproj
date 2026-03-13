from torchvision.models import squeezenet1_0
from torchvision.models import SqueezeNet1_0_Weights
from torch import nn

class FoodSN():
    def __init__(self):
        self.weights = SqueezeNet1_0_Weights.IMAGENET1K_V1
        self.model = squeezenet1_0(weights=self.weights) 
        final_conv = nn.Conv2d(512, 101, kernel_size=1)
        nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)
        self.model.classifier = nn.Sequential(
            nn.Dropout(
                p=0), 
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
            )
        
    def forward(self, x):
        return model.forward(x)

if __name__ == "__main__":
    print("class def of squeezenet for Food101")
