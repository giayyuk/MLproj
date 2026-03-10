import torch
import tqdm
import random
from sn import FoodSN
import numpy as np
from src.data.load_data import Data
from torch.utils.data import DataLoader,random_split
from torch.nn import CrossEntropyLoss

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.memory.set_per_process_memory_fraction(fraction=0.33)
torch.set_num_threads(3)

# loss optimizer model
loss_fn = CrossEntropyLoss()
model = FoodSN()
optimizer = torch.optim.Adam(model.parameters())

# load data
data = Data()
train_data = data.get_train_data()


def train(num_epochs, model,data_loader):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        for i, (images,labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = loss_fn(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), "food101_first_training.pth")

train(2, model, train_data)
