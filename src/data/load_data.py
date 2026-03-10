import torchvision

train_split = torchvision.datasets.Food101('src/data/train',split="train",download=True)
test_split = torchvision.datasets.Food101('src/data/test',split="test",download=True)

if __name__ == "__main__":
    print("dataSet loaded") if train_split != None and test_split != None else print("failed loading dataSet")