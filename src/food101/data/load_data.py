import torchvision

data_set = torchvision.datasets.Food101('src/food101/data',download=True)


if __name__ == "__main__":
    print("dataSet loaded") if data_set != None else print("failed loading dataSet")