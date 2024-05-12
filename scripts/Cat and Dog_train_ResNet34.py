import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ResNet34 import ResNet34

# 修改预训练模型保存地址
os.environ['TORCH_HOME'] = '../torch_model'
# 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

"""设置超参数"""
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 选择设备

valid_ratio = 0.2  # 验证集比例
learning_rate = 1e-4
epoches = 5
batch_size = 128

"""数据集处理"""
cats_train_dir = '../data/Cat vs Dog/train/cats'    # cats训练集根目录
dogs_train_dir = '../data/Cat vs Dog/train/dogs'    # dogs训练集根目录
test_dir = '../data/Cat vs Dog/test'     # 测试集根目录

cats_train_filenames = [os.path.join(cats_train_dir, f) for f in os.listdir(cats_train_dir)]  # 每个cat图文件地址合并为list
dogs_train_filenames = [os.path.join(dogs_train_dir, f) for f in os.listdir(dogs_train_dir)]  # 每个dog图文件地址合并为list

# 把cats和dogs统一归为训练集
train_filenames = [*cats_train_filenames, *dogs_train_filenames]
# 训练集文件地址
test_filenames = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]

train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.490, 0.482, 0.445], std=[0.240, 0.237, 0.257])
])

test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.490, 0.482, 0.445], std=[0.240, 0.237, 0.257])
])

# 定义数据集
class DogandCat(Dataset):
    def __init__(self, filenames, transforms):
        self.filenames = filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx])
        transformed_img = self.transforms(img)
        label = 0 if self.filenames[idx].split('\\')[-1].split('.')[0] == 'cat' else 1  # dog的标签为1,cat的标签为0
        return transformed_img, label
# 划分数据集
num_filename = len(train_filenames)
num_train = int(num_filename * (1 - valid_ratio))
# 随机抽取!!!!!!!!!!!!!!!!!!(后面可以随机抽取数据集做train和valid类似k折交叉验证)
# import random
#
# train_sample = random.sample(train_filenames, num_train)
# valid_sample = [x for x in train_filenames if x not in train_sample]
#
# train_dataset = DogandCat(train_sample, train_transforms)
# valid_dataset = DogandCat(valid_sample, test_transforms)
# test_dataset = DogandCat(test_filenames, test_transforms)

train_dataset = DogandCat(train_filenames[:num_train], train_transforms)
valid_dataset = DogandCat(train_filenames[num_train:], test_transforms)
test_dataset = DogandCat(test_filenames, test_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""ResNet34网络"""
model = ResNet34(num_classes=2).to(device)
# 定义损失器,优化器
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

"""数据可视化"""
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

"""模型训练"""
min_acc = 0  # 为保存最佳valid_acc的网络参数文件
print('training on -----------------')
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
    # 保存最佳模型
    if val_acc > min_acc:
        folder = '../save_model'
        if not os.path.exists(folder):
            os.makedirs(os.path.join(folder))
        min_acc = val_acc
        print(f'epoch{epoch+1}:save the best model')
        torch.save(model.state_dict(), '../save_model/best_model_ResNet34.pth')
    # if epoch == epoches - 1:
    #     torch.save(model.state_dict(), '../save_model/last_model_ResNet34.pth')  # 保留训练最后一次的参数文件

    print(f"valid loss: {val_loss:.4f}")
    print(f"valid acc: {val_acc:.4f}")
    print("-" * 40)

# 可视化
matplot_loss(loss_train, loss_val)
matplot_acc(acc_train, acc_val)

sample = pd.read_csv('../data/Cat vs Dog/sample_submission.csv')
result = []
# 加载训练最好的pth文件参数
model.load_state_dict(torch.load('../save_model/best_model_ResNet34.pth'))

with torch.no_grad():
    model.eval()
    for img, label in tqdm(test_loader):
        img, label = img.to(device), label.to(device)

        label_hat = model(img)
        pred = torch.argmax(label_hat, 1)
        result.extend(pred.cpu().numpy())

sample['labels'] = result
sample.to_csv('../data/Cat vs Dog/predict_ResNet34.csv', index=False)

