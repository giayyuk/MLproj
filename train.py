import torch
import tdqm
import random
import sn
import numpy as np4
import src.data.load_data
from torch.utils.data import DataLoader,random_split
from torch.nn import CrossEntropyLoss

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.cuda.memory.set_per_process_memory_fraction(fraction=0.33)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loss optimizer model
loss_fn = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
model = FoodSN()


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

