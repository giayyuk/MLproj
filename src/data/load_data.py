import torchvision
from torchvision import datasets
from torch.utils.data import Subset
import torch

class Data():
    def __init__(self,transform):
        self.transform = transform
        self.train_data = torchvision.datasets.Food101('src/data/train',split="train",transform=self.transform,download=True)
        self.test_data = torchvision.datasets.Food101('src/data/test',split="test",transform=self.transform,download=True)

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_subset_train_data(self,size):
        random_idx = torch.randperm(len(self.train_data))
        return Subset(self.train_data,random_idx[0:size])
