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
writer = SummaryWriter()
size_data_set: int = 1000
batch_size: int = 10
train_val_split: list[float] = [0.8,0.2]

# load data
data = Data()
train_data = data.get_test_data()
size_data_set = len(train_data)
train, val = torch.utils.data.random_split(train_data, train_val_split,torch.Generator().manual_seed(42))
dl = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1)

# loss optimizer model
loss_fn = CrossEntropyLoss()
model = FoodSN().to(device)
optimizer = torch.optim.Adam(model.parameters())



def train(num_epochs, model,data_loader):
    model.train()
    losses = []
    epochs = []
    print("starting training loop")
    for epoch in tqdm(range(num_epochs)):
        running_corrects = 0
        for i, (images,labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = loss_fn(output,labels)
            running_corrects += torch.sum(torch.argmax(output,dim=1) == labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 3 == 0:
            losses.append(loss.detach().item())
            epochs.append(epoch)
            writer.add_scalar("loss",loss.detach(),(epoch*train_val_split[0]*size_data_set/batch_size) + i)
            writer.add_scalar("accuracy",running_corrects.double()/len(data_loader.dataset),(epoch*train_val_split[0]*size_data_set/batch_size) + i)
    return (losses,epochs, running_corrects.double()/len(data_loader.dataset))

losses, epochs, accuracy = train(10,model,dl)
writer.flush()
print(epochs)
print(losses)
print(accuracy)

fig, ax = plt.subplots()
ax.plot(epochs,losses)
plt.savefig("loss_prototype.png")
# torch.save(model.state_dict(), "food101_first_training.pth")
