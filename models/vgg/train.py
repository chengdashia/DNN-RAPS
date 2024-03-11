import sys
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像增强
transform_plus = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1))]
)
transform_norm = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
transform = transforms.Compose([transforms.ToTensor(), transform_norm])

# 将下载的cifar-10数据集放在代码所在目录下的data文件夹里
train_set = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# 打乱数据集
choice = list(range(len(train_set)))
random.shuffle(choice)


class mydataset(torch.utils.data.Dataset):
    def __init__(self, trainset, choice, num_val, transform=None, transform_norm=None, train=True):
        self.transform = transform
        self.transform_norm = transform_norm
        self.train = train
        self.choice = choice
        self.num_val = num_val
        if self.train:
            self.images = trainset.data[self.choice[self.num_val:]].copy()
            self.labels = [trainset.targets[i] for i in self.choice[self.num_val:]]
        else:
            self.images = trainset.data[self.choice[:self.num_val]].copy()
            self.labels = [trainset.targets[i] for i in self.choice[:self.num_val]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        # ToTensor将array、图片转为[0,1]间的数字
        image = transforms.ToTensor()(image)
        if self.transform:
            # transforms 输入是一个Tensor
            image = self.transform(image)
        if self.transform_norm:
            image = self.transform_norm(image)
        sample = (image, label)
        return sample


validset = mydataset(train_set, choice, len(train_set) // 10, None, transform_norm, False)
trainset = mydataset(train_set, choice, len(train_set) // 10, transform_plus, transform_norm, True)


def test(net, validloader):
    test_loss = 0
    test_correct = 0
    time = 0
    net.eval()
    with torch.no_grad():
        for data in validloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            test_loss += loss_fn(outputs, labels).item() * len(labels)
            test_correct += (torch.max(outputs.data, 1)[1] == labels).sum()
            time += 1
    return (test_loss / len(validset), test_correct / len(validset) * 100)


class vgg16_conv_block(nn.Module):
    def __init__(self, input_channels, out_channels, rate=0.4, drop=True):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, out_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(rate)
        self.drop = drop

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        if self.drop:
            x = self.dropout(x)
        return (x)


def vgg16_layer(input_channels, out_channels, num, dropout=[0.4, 0.4]):
    result = []
    result.append(vgg16_conv_block(input_channels, out_channels, dropout[0]))
    for i in range(1, num - 1):
        result.append(vgg16_conv_block(out_channels, out_channels, dropout[1]))
    if num > 1:
        result.append(vgg16_conv_block(out_channels, out_channels, drop=False))
    result.append(nn.MaxPool2d(2, 2))
    return (result)


b1 = nn.Sequential(*vgg16_layer(3, 64, 2, [0.3, 0.4]), *vgg16_layer(64, 128, 2), *vgg16_layer(128, 256, 3),
                   *vgg16_layer(256, 512, 3), *vgg16_layer(512, 512, 3))
b2 = nn.Sequential(nn.Dropout(0.5), nn.Flatten(), nn.Linear(512, 512, bias=True), nn.BatchNorm1d(512),
                   nn.ReLU(inplace=True),
                   nn.Linear(512, 10, bias=True))
net = nn.Sequential(b1, b2)

net = nn.DataParallel(net)
net.to(device)
net.train()
epoch_num = 150  # 轮次
batch_num = 128  # minbatch
learning_rate = 0.1
train_num = len(trainset) // batch_num
los = []
cor = []
train_los = []
train_cor = []
net_corr, net_los, net_train_los, net_train_corr, net_lr, net_epoch = 0, 0, 0, 0, 0, 0
loss_fn = nn.CrossEntropyLoss()
opt = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-6, nesterov=True)
scheduler3 = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=22, T_mult=2)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_num, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_num, shuffle=True)

for epoch in range(epoch_num):
    loss_avg = 0  # 平均损失
    train_time = 0  # 多少个minbatch了
    correct = 0  # 正确率
    num_img = 0
    for data in train_loader:
        #     for data in itertools.islice(train_loader, 10):
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        net.train()
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.to(device)
        opt.zero_grad()
        loss.backward()
        # 梯度剪切，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(net.parameters(), 20)
        opt.step()
        train_time += 1
        loss_avg += loss.item() * len(labels)
        # loss.item是minbatch的平均损失
        predict = torch.max(outputs.data, 1)[1]
        correct += (predict == labels).sum()
        num_img += len(labels)
        print('\r', end='')
        print('进度：{}批次'.format(train_time), '', end="")
        sys.stdout.flush()

    scheduler3.step()
    print('\r', end="")

    los2, cor2 = test(net, valid_loader)
    print('正在训练：{}/{}轮，学习率为：{:.10f}，平均Loss：{:.2f}，正确率为：{:.2f}%, 验证集损失为{:.2f}，成功率为{:.2f}%'
          .format(epoch + 1, epoch_num, opt.state_dict()['param_groups'][0]['lr'], loss_avg / num_img,
                  correct / num_img * 100,
                  los2, cor2.item()))

    los.append(los2)
    cor.append(cor2)
    train_cor.append(correct / num_img * 100)
    train_los.append(loss_avg / num_img)
    if net_corr < cor2:
        net_corr, net_los, net_train_los, net_train_corr, net_lr, net_epoch = cor2, los2, loss_avg / num_img, correct / num_img, \
        opt.state_dict()['param_groups'][0]['lr'], epoch + 1
    torch.save(net, 'net_model.pkl')
    sys.stdout.flush()
print(
    '第{}个epoch时模型最优，学习率为{:.8f}, 训练损失为{:.4f}, 训练正确率为{:.2f}%, 验证损失为{:.4f}, 验证正确率为{:.2f}%'.format(
        net_epoch, net_lr, net_train_los, net_train_corr * 100, net_los, net_corr))
best_net = torch.load('net_model.pkl')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)
test_loss = 0
test_correct = 0
time = 0
best_net.eval()
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = best_net(inputs)
        test_loss += loss_fn(outputs, labels).item() * len(labels)
        test_correct += (torch.max(outputs.data, 1)[1] == labels).sum()
        time += 1
print('共测试{}个图片，平均损失是{:0.2f}，成功率为{:0.2f}%'.format(len(test_set.data), test_loss / len(test_set.data),
                                                                 test_correct / len(test_set.data) * 100))
x_epoch = [i for i in range(epoch_num)]

plt.figure()
plt.plot(x_epoch, train_los, 'darkorange')
plt.plot(x_epoch, los)
# plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train_loss', 'Test_loss'])

plt.figure()
plt.plot(x_epoch, torch.tensor(train_cor).cpu())
plt.plot(x_epoch, torch.tensor(cor).cpu())
plt.xlabel('Epoch')
plt.ylabel('Correct')
plt.legend(['Train_Correct', 'Test_Correct'])