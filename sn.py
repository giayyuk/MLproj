from torchvision.models import squeezenet

class FoodSN(squeezenet.SqueezeNet):
    def __init__(self):
<<<<<<< HEAD
        super().__init__(num_classes=101)

    def forward(self, x):
        return super().forward(x)

=======
        super().__init__(num_classes = 101)

    def forward(self,x):
        return super().forward(x)
>>>>>>> adabb41059b721fe2e16aa4cd4fb06822b4ea731

if __name__ == "__main__":
    print("class def of squeezenet for Food101")
