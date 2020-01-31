"""
Author: Zhou Chen
Date: 2020/1/30
Desc: 训练模型并可视化
"""
import os
import torch
from torchvision.utils import make_grid
import torchvision.models as models
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets
from tensorboardX import SummaryWriter


epochs = 10
writer = SummaryWriter('logs/')


loss = 10
for epoch in range(epochs):
    writer.add_scalar("logs/scalar_exp", loss, epoch)
    loss -= 1

train_loss = 10
valid_loss = 9
for epoch in range(epochs):
    writer.add_scalars('loss', {'train_loss': train_loss, 'valid_loss': valid_loss}, epoch)
    train_loss -= 1
    valid_loss -= 1

for epoch in range(epochs):
    for name, param in model.named_parameters():
        writer.add_histogram(name, param.data.numpy(), epoch)

for epoch in range(epochs):
    img = torch.rand(32, 3, 64, 64)
    img_visual = make_grid(img, normalize=True, scale_each=True)
    writer.add_image('image', img_visual, epoch)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


dummy_input = Variable(torch.rand(13, 1, 28, 28))

model = Net()
with SummaryWriter(comment='Net') as w:
    w.add_graph(model, (dummy_input, ))


dataset = datasets.MNIST(os.path.join("..", "..", "Data", "mnist"), train=False, download=True)
images = dataset.test_data[:100].float()
label = dataset.test_labels[:100]
features = images.view(100, 784)
writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))
writer.add_embedding(features, global_step=1, tag='noMetadata')
dataset = datasets.MNIST("mnist", train=True, download=True)
images_train = dataset.train_data[:100].float()
labels_train = dataset.train_labels[:100]
features_train = images_train.view(100, 784)

all_features = torch.cat((features, features_train))
all_labels = torch.cat((label, labels_train))
all_images = torch.cat((images, images_train))
dataset_label = ['test'] * 100 + ['train'] * 100
all_labels = list(zip(all_labels, dataset_label))

writer.add_embedding(all_features, metadata=all_labels, label_img=all_images.unsqueeze(1),
                     metadata_header=['digit', 'dataset'], global_step=2)
