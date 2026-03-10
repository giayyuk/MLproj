import torchvision

class Data():
    def __init__(self):
        self.train_data = torchvision.datasets.Food101('src/data/train',split="train",download=True)
        self.test_data = torchvision.datasets.Food101('src/data/test',split="test",download=True)

        if __name__ == "__main__":
            print("dataSet loaded") if train_split != None and test_split != None else print("failed loading dataSet")

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data