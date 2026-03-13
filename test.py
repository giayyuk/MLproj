from src.data.load_data import Data 
from torchvision import transforms
from sn import FoodSN
from torchmetrics import Accuracy
import torch
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)

torch.cuda.memory.set_per_process_memory_fraction(fraction=0.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(10)


transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
data = Data(transform)
test_data = data.get_test_data()

model = FoodSN().model
model.load_state_dict(torch.load("food101_first_training.pth",weights_only=False))
batch_size = 2**5
acc_fn = Accuracy(task="multiclass",num_classes=101).to(device)
dl_test = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

acc_cum = []

model = model.eval().to(device)
for i, (images,labels) in tqdm(enumerate(dl_test)):
    labels.to(device)
    output = model(images.to(device))
    acc = acc_fn(output.to(device), labels.int().to(device)).to(device)
    acc_cum.append(acc)


print(f"average acc = {sum(acc_cum)/len(acc_cum)}")

