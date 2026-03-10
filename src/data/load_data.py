import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Subset

class Data():
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        self.train_data = torchvision.datasets.Food101('src/data/train',split="train",transform=self.transform,download=True)
        self.test_data = torchvision.datasets.Food101('src/data/test',split="test",transform=self.transform,download=True)

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_subset_train_data(self,size):
        return Subset(self.train_data,range(size))