import torch
from tqdm import tqdm
import random
from sn import FoodSN
import numpy as np
from src.data.load_data import Data
from torch.utils.data import DataLoader,random_split
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
from datetime import datetime

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)

#torch.cuda.memory.set_per_process_memory_fraction(fraction=0.33)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.memory.set_per_process_memory_fraction(fraction=0.33)
torch.set_num_threads(3)

# Load Dataset
data = Data()
train_data = data.get_subset_train_data(1000)
writer = SummaryWriter()

# Create Data Loader
dl = DataLoader(train_data, batch_size=10, shuffle=True, num_workers=1)

# loss optimizer model
loss_fn = CrossEntropyLoss()
model = FoodSN().to(device)
optimizer = torch.optim.Adam(model.parameters())

# load data
data = Data()
train_data = data.get_train_data()


def train(num_epochs, model,data_loader):
    model.train()
    losses = []
    epochs = []
    print("starting training loop")
    for epoch in tqdm(range(num_epochs)):
        for i, (images,labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = loss_fn(output,labels)
            writer.add_scalar("logs",loss.detach(),epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            losses.append(loss.detach().item())
            epochs.append(epoch)
            print(output[0])
            print(output.shape)
    return (losses,epochs)

losses, epochs = train(10,model,dl)
writer.flush()
print(epochs)
print(losses)

fig, ax = plt.subplots()
ax.plot(epochs,losses)
plt.savefig("loss_prototype.png")
# torch.save(model.state_dict(), "food101_first_training.pth")
