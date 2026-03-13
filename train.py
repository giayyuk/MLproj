import torch
from tqdm import tqdm
import random
from sn import FoodSN
import numpy as np
from src.data.load_data import Data
from torch.utils.data import DataLoader,random_split
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchmetrics import Accuracy
from torchvision import transforms
import matplotlib.pyplot as plt
from datetime import datetime
from torchmetrics import Accuracy

# Setting the seeds
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)

# hyperparameter [lr, batchsize, epochs, l1/l1/dropout,optimizer]

torch.cuda.memory.set_per_process_memory_fraction(fraction=0.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(10)

# Set Dataset Size and Split 
writer = SummaryWriter()
size_data_set: int = 1000
batch_size: int = 2**8
train_val_split: list[float] = [0.8,0.2]
# load data
transform = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.Resize(size=256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

data = Data(transform)
# get_train_data() for whole data otherwise get_subset_train_data(size)
train_data = data.get_train_data() #data.get_subset_train_data(size_data_set) # 
size_data_set = len(train_data)

train, val = torch.utils.data.random_split(train_data, train_val_split,torch.Generator().manual_seed(42))
dl_train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1)
dl_val   = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=1)

# loss optimizer model
loss_fn = CrossEntropyLoss()
acc_fn =  Accuracy(task="multiclass",num_classes=101).to(device) 
model = FoodSN().model.to(device)
model.load_state_dict(torch.load("food101_first_training.pth",weights_only=False))
optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)

def train(num_epochs, model,data_loader,):
    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False
    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    losses = []
    epochs = []
    for epoch in tqdm(range(num_epochs)):
        print(f"starting training {epoch=}")
        model.train()
        for i, (images,labels) in tqdm(enumerate(dl_train)):
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = loss_fn(output,labels)
            loss.backward()
            optimizer.step()
            if i % 8 == 0:
                losses.append(loss.detach().item())
                epochs.append(epoch)
                writer.add_scalar(
                        "loss",loss.detach(),
                        (epoch*train_val_split[0]*size_data_set/batch_size) + i
                        )
                acc = acc_fn(output,labels.int())
                print(f"\n-------Training: {acc=} {loss.detach()=}------\n")
 

        #validation
        print("validation\n")
    #    model.eval()
        #for i, (images,labels) in enumerate(dl_val):
            #images = images.to(device)
            #labels = labels.to(device)
            #output = model(images)
            #if i % 8 == 0:
                #acc = acc_fn(output, labels.int())
                #writer.add_scalar(
#                        "accuracy",
                        #acc,
                        #(epoch*train_val_split[0]*size_data_set/batch_size) + i)
#                print(f"\n-------Validation: {acc=}------\n")
    #torch.save(model.state_dict(), "food101_first_training.pth")
    return (losses,epochs,acc_fn(output, labels.int()))

writer.flush()
losses, epochs, accuracy = train(10,model,dl_train)
print(f"{sum(losses)/len(losses)=}")
print(f"{accuracy=}")

fig, ax = plt.subplots()
ax.plot(epochs,losses)
plt.savefig("loss_prototype.png")
