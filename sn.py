from torchvision.models import squeezenet

class foodSN(squeezenet.SqueezeNet):
    def __init__(self,num_of_classes):
        super.__init__(num_classes = num_of_classes)

    def forward(self,x):
        super.forward(x)

if __name__ == "__main__":
    print("class def of squeezenet for Food101")
