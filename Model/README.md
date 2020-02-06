# PyTorch模型


## 简介
在[前一篇文章](https://zhouchen.blog.csdn.net/article/details/104087727)提到了关于数据的一系列操作，数据读入之后就是将数据“喂”给深度模型，所以构建一个深度学习模型是很重要的，本文主要讲解PyTorch中模型的相关操作，包括模型的定义、模型参数的初始化、模型的保存及加载。


## 模型构建
在PyTorch中，想要让PyTorch的后续接口认为这是一个模型的关键需要满足下面三个条件。
- 自定义的模型以类的形式存在，且该类必须继承自`torch.nn.Module`，该继承操作可以让PyTorch识别该类为一个模型。
- 在自定义的类中的`__init__`方法中声明模型使用到的组件，一般是封装好的张量操作，如卷积。池化、批量标准化、全连接等。
- 在`forward`方法中使用`__init__`方法中的组件进行组装，构成网络的前向传播逻辑。

该模型的前向运算通过调用模型的`call`方法即可，通过实例化的对象加括号即可默认调用该方法，如`net(input_tensor)`。包括很多nn中定义的张量运算也可以通过该方法进行运算。

下面的代码演示了一个比较简单的卷积神经网络分类器，其中涉及到的张量操作如卷积等均为计算机视觉中的基本知识，参数也比较好理解，不多赘述，可以查看nn模块的[文档](https://pytorch.org/docs/stable/nn.html#)。**这里需要注意，PyTorch不同于TensorFlow，PyTorch期待图片的张量格式输入是(batch, c, h, w )，这点需要注意，当然，如果按照上篇文章中PyTorch的数据接口读入数据，那不需要自行控制。**下面的代码输出如期待的是`[batch, 101]`的维度，通过softmax激活后就是常见的分类器输出。
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*54*54, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 101)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64*54*54)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    net = Net()
    data = torch.randn((32, 3, 224, 224))
    label = torch.zeros((32, ))
    pred = net(data)
    print(pred.shape)

```

实际上，真正设计模型的时候，模型都是比较复杂的，远不是一个模块就能完成的，这时候就需要组合多个模型，也需要组合多个张量运算，将多个张量运算堆叠或者多个模型堆叠需要使用一个容器---`torch.nn.Sequential`，该容器将一系列操作按照前后顺序堆叠起来，如下面的例子，该容器接受一系列的张量操作或者模型操作为参数。
```python
nn.Sequential(
    nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
    nn.BatchNorm2d(out_channels)
    )
```

下面的例子就演示了构建较为复杂的深度模型的方法，该模型为ResNet34，参考了[这个思路](https://github.com/yuanlairuci110/PyTorch-best-practice-master/blob/master/models/ResNet34.py)。
```python
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """
    残差模块
    """

    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(nn.Module):
    def __init__(self, num_classes=101):
        super(ResNet34, self).__init__()
        self.model_name = 'ResNet34'
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))

        self.layer1 = self.make_layer(64, 128, 3)
        self.layer2 = self.make_layer(128, 256, 4, stride=2)
        self.layer3 = self.make_layer(256, 512, 6, stride=2)
        self.layer4 = self.make_layer(512, 512, 3, stride=2)

        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_channels, out_channels, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels))

        layers = list()
        layers.append(ResidualBlock(in_channels, out_channels, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == '__main__':
    resnet = ResNet34(num_classes=101)
    data = torch.randn((32, 3, 224, 224))
    label = torch.zeros((32,))
    pred = resnet(data)
    print(pred.shape)
```


## 权重初始化
在上一节的介绍中，详细叙述了模型的构建细节，但是有一个很重要的点没有提及，就是权重初始化，如上面案例中的conv2d和Linear，这些运算都有可训练的参数需要初始化（当然，不设定也有默认初始化方法），这些参数权重也是模型训练的目的，参数初始化的效果在深度学习中尤为重要，它会直接影响模型的收敛效果。

当然，一般需要对不同的运算层采用不同的初始化方法，在PyTorch中初始化方法均在`torch.nn.init`中封装。在上一节自定义网络的类中，添加一个权重初始化方法如下（针对不同的结构采用不同的初始化方法，如Xavier、Kaiming正太分布等），修改后的代码如下。
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*54*54, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 101)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64*54*54)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    net = Net()
    net.init_weights()
    data = torch.randn((32, 3, 224, 224))
    label = torch.zeros((32,))
    pred = net(data)
    print(pred.shape)

```


## 初始化方法
在上一节，主要讲述如何对模型各层进行初始化，事实上常用的初始化方法PyTorch都进行了封装于`torch.nn.init`模块下，有Xavier初始化和Kaiming初始化这两种效果比较显著的初始化方法，也有一些比较传统的方法。

### Xavier初始化 
这是依据“方差一致性”推导得到的初始化方法，有均匀分布和正态分布两种。
- Xavier均匀分布
  - `nn.init.xavier_uniform_(tensor, gain=1)`
  - Xavier初始化方法服从均匀分布`(-a, a)`，其中`a=gain*sqrt(6/fan_in+fan_out)`,其中gain是依据激活函数类型设定的。
- Xavier正态分布
  - `nn.init.xavier_normal_(tensor, gain=1)`
  - Xavier初始化方法服从正太分布,`mean=0, std=gain*sqrt(2/fan_in+fan_out)`。

### Kaiming初始化
同样根据“方差一致性”推导得到，针对xavier初始化方法在relu这一类激活函数上效果不佳而提出的。
- Kaiming均匀分布
  - `nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')`
  - 服从(-b, b)的均匀分布，其中`b=sqrt(6/(1+a^2)*fan_in)`，其中a为激活函数负半轴的斜率，relu对应的a为0。
  - **mode**参数为`fan_in`或者`fan_out`，前者使正向传播时方差一致，后者使反向传播时方差一致。
  - **nonlinearity**参数表示是否非线性，`relu`表示线性，`leaky_relu`表示非线性。
- Kaiming正态分布
  -  `nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')`
  - 服从`(0, std)`的正态分布，其中`std=sqrt(2/(1+a^2)*fan_in)`。
  - 参数同上。

### 其他初始化
- 均匀分布初始化
  - `nn.init.uniform_(tensor, a=0, b=1)`
  - 服从`U(a,b)`的均匀分布。
- 正态分布初始化
  - `nn.init.normal_(tensor, mean=0, std=1)`
  - 服从`N(mean,std)`的正态分布。
- 常数初始化
  - `nn.init.constant_(tensor, val)`
  - 使值为常数val的初始化方法。
- 单位矩阵初始化
  - `nn.init.eye_(tensor)`
  - 将二维Tensor初始化为单位矩阵。
- 正交初始化
  - `nn.init.orthogonal_(tensor, gain=1)`
  - 使得tensor是正交的。
- 稀疏初始化
  - `nn.init.sparse_(tensor, sparsity, std=0.01)`
  - 从正态分布`N(0,std)`中进行稀疏化，使得每一列有一部分为0，其中sparsity控制列中为0的比例。
- 增益计算
  - `nn.init.calculate_gain(nonlinearity, param=None)`
  - 用于计算不同激活函数的gain值。


## 迁移学习
上一节，介绍了很多权重初始化的方法，可以知道良好的初始化可以加速模型收敛、获得更好的精度，实际使用中，通常采用一个预训练的模型的权重作为模型的初始化参数，这个方法称为Finetune，更广泛的就是指迁移学习，Finetune技术本质上就是使得模型具有更好的权重初始化。

一般，需要如下三个步骤。
1. 训练模型，保存模型参数。
2. 加载预训练的模型参数。
3. 构建新模型，将获得的预训练模型参数放到新模型中。

上述过程有两种模型的保存方法，一种是保存整个模型，另一种是保存模型的参数，推荐后者。

下面的代码演示如何进行Finetune的初始化，当然，实际进行迁移学习时对于不同的层需要采用不同的学习速率，一般希望前面的层学习率低，后层（如全连接层）学习率高一些，需要对不同的层设置不同的学习率，这里不多提及了。
```python
net = Net()
# 假设训练完成了
# 保存模型参数
torch.save(net.state_dict(), 'net_weights.pkl')
# 加载模型参数
pretrained_params = torch.load('net_weights.pkl')
# 构建模型，预训练参数初始化
net = Net()
net_state_dict = net.state_dict()
pre_dict = {k: v for k, v in pretrained_params.items() if k in net_state_dict}
net_state_dict.update(pre_dict)
```


## 补充说明
本文介绍了PyTorch中模型构建的方法以及权重初始化技巧，进一步提到了迁移学习，这包括模型的保存与加载、使用预训练模型的参数作为网络的初始化参数（finetune技巧）。本文的所有代码均开源于[我的Github](https://github.com/luanshiyinyang/Tutorial/tree/PyTorch)，欢迎star或者fork。