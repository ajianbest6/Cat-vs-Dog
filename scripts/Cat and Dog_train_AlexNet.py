import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from AlexNet import MyAlexNet

# 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

cats_train_dir = '../data/Cat vs Dog/train/cats'
dogs_train_dir = '../data/Cat vs Dog/train/dogs'
test_dir = '../data/Cat vs Dog/test'

cats_train_filenames = [os.path.join(cats_train_dir, f) for f in os.listdir(cats_train_dir)]
dogs_train_filenames = [os.path.join(dogs_train_dir, f) for f in os.listdir(dogs_train_dir)]

train_filenames = [*cats_train_filenames, *dogs_train_filenames]
test_filenames = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]

train_transforms = transforms.Compose([
    transforms.Resize(223),
    transforms.CenterCrop(223),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.490, 0.482, 0.445], std=[0.240, 0.237, 0.257])
])

test_transforms = transforms.Compose([
    transforms.Resize(223),
    transforms.CenterCrop(223),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.490, 0.482, 0.445], std=[0.240, 0.237, 0.257])
])

class DogandCat(Dataset):
    def __init__(self, filenames, transforms):
        self.filenames = filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx])
        transformed_img = self.transforms(img)
        label = 0 if self.filenames[idx].split('\\')[-1].split('.')[0] == 'cat' else 1
        return transformed_img, label

# 划分训练集和验证集比例
num_filename = len(train_filenames)
valid_ratio = 0.2
num_train = int(num_filename * (1 - valid_ratio))

train_dataset = DogandCat(train_filenames[:num_train], train_transforms)
valid_dataset = DogandCat(train_filenames[num_train:], test_transforms)
test_dataset = DogandCat(test_filenames, test_transforms)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = MyAlexNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 定义画图
def matplot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('训练集与验证集Loss值对比图')
    plt.show()

def matplot_acc(train_acc, val_acc):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title('训练集与验证集Acc值对比图')
    plt.show()

loss_train = []
acc_train = []
loss_val = []
acc_val = []

min_acc = 0
print('training on -----------------')
epoches = 30
for epoch in range(epoches):
    print(f'Epoch:{epoch + 1}')
    epoch_loss, epoch_corrects = 0.0, 0.0
    model.train()
    for img, label in tqdm(train_loader):
        img, label = img.to(device), label.to(device)
        label_hat = model(img)
        pred = torch.argmax(label_hat, 1)
        loss = criterion(label_hat, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * img.size(0)
        epoch_corrects += torch.sum(pred == label.data)
    epoch_loss = epoch_loss / len(train_loader.dataset)
    epoch_acc = epoch_corrects / len(train_loader.dataset)
    loss_train.append(epoch_loss)
    acc_train.append(epoch_acc.cpu().numpy())

    print(f"train loss: {epoch_loss:.4f}")
    print(f"train acc: {epoch_acc:.4f}")

    val_loss, val_corrects = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for img, label in tqdm(valid_loader):
            img, label = img.to(device), label.to(device)
            label_hat = model(img)
            pred = torch.argmax(label_hat, 1)
            loss = criterion(label_hat, label)

            val_loss += loss.item() * img.size(0)
            val_corrects += torch.sum(pred == label.data)
    val_loss = val_loss / len(valid_loader.dataset)
    val_acc = val_corrects / len(valid_loader.dataset)

    loss_val.append(val_loss)
    acc_val.append(val_acc.cpu().numpy())

    if val_acc > min_acc:
        folder = '../save_model'
        if not os.path.exists(folder):
            os.makedirs(os.path.join(folder))
        min_acc = val_acc
        print(f'epoch{epoch+1}:save the best model')
        torch.save(model.state_dict(), '../save_model/best_model.pth')
    if epoch == epoches - 1:
        torch.save(model.state_dict(), '../save_model/last_model.pth')

    print(f"valid loss: {val_loss:.4f}")
    print(f"valid acc: {val_acc:.4f}")
    print("-" * 40)
matplot_loss(loss_train, loss_val)
matplot_acc(acc_train, acc_val)

sample = pd.read_csv('../data/Cat vs Dog/sample_submission.csv')
result = []

model.load_state_dict(torch.load('../save_model/best_model.pth'))
with torch.no_grad():
    model.eval()
    for img, label in tqdm(test_loader):
        img, label = img.to(device), label.to(device)

        label_hat = model(img)
        pred = torch.argmax(label_hat, 1)
        result.extend(pred.cpu().numpy())
sample['labels'] = result
sample.to_csv('../data/Cat vs Dog/predict_AlexNet.csv', index=False)
print('Done!!!')
