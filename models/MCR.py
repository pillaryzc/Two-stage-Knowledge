import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models


'''

### 模块名称: MCR 

#### 功能简介
- **主要作用**: 该代码模块通过定义三个主要类（`Vgg19`, `SCRLoss`, `HCRLoss`）来实现图像去噪的深度学习方法。这些类利用VGG19网络的特征提取能力和定制的损失函数来优化图像去噪模型。
- **关键功能**:
  - `Vgg19`类用于提取图像的特征表示。
  - `SCRLoss`和`HCRLoss`类定义了两种损失函数，用于训练图像去噪模型，通过比较图像特征的相似度和差异性。

#### 关键代码段解析
- **Vgg19**:
  - **功能**: 利用预训练的VGG19模型提取图像特征。
  - **重要代码解释**: 通过将VGG19预训练模型的特征部分分成五个阶段，使得可以在不同深度级别提取特征。
  - **使用场景**: 在图像去噪、风格迁移等任务中提取特征表示。

- **SCRLoss**:
  - **功能**: 定义结构一致性表示学习损失函数，优化正样本与锚点图像的相似度，增大负样本与锚点的差异。
  - **重要代码解释**: 计算锚点、正样本、负样本特征的L1距离，并基于这些距离计算对比损失。
  - **使用场景**: 用于训练图像去噪模型，特别是在需要区分相似和不相似图像对的场合。

- **HCRLoss**:
  - **功能**: 定义层次性对比表示学习损失函数，通过更复杂的方法考虑所有可能的负样本对。
  - **重要代码解释**: 通过扩展和重排特征张量，计算锚点与多个负样本之间的L1距离，用于损失计算。
  - **使用场景**: 在更为复杂或要求更高精度的图像去噪任务中，提供更细粒度的模型优化指导。

#### 输入输出
- **输入**: 
  - 对于`Vgg19`类，输入是需要提取特征的图像。
  - 对于`SCRLoss`和`HCRLoss`类，输入包括锚点图像、正样本图像和负样本图像。
- **输出**: 
  - `Vgg19`类输出是图像在不同深度级别的特征表示。
  - `SCRLoss`和`HCRLoss`类输出是计算得到的损失值。

#### 依赖关系
- **内部依赖**: 这些类依赖于PyTorch框架和torchvision库中的预训练VGG19模型。
- **外部依赖**: 需要PyTorch和torchvision库。

#### 注意事项
- **特别注意**: 在使用`SCRLoss`和`HCRLoss`损失函数时，需要确保输入图像已正确预处理，且与VGG19模型预训练时的要求一致。
- **性能考虑**: 在使用CUDA加速时，确保所有输入和模型都已转移到GPU上，以优化性能。

这个模块提供了一个基于深度学习的图像去噪框架，通过结合先进的特征提取方法和损失函数设计，它能够有效地指导模型训练，以达到更好的去噪效果。

'''

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1) 
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4) 
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class SCRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0
        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            d_an = self.l1(a_vgg[i], n_vgg[i].detach())
            contrastive = d_ap / (d_an + 1e-7)
            loss += self.weights[i] * contrastive

        return loss


class HCRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            b, c, h, w = a_vgg[i].shape
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            # a_vgg[i].unsqueeze(1).expand(b, b, c, h, w): a_vgg[i][0, 0] == a_vgg[i][0, 1] == a_vgg[i][0, 2]...
            # n_vgg[i].expand(b, b, c, h, w): a_vgg[i][0] == a_vgg[i][1] == a_vgg[i][2]..., but a_vgg[i][0, 0] != a_vgg[i][0, 1]
            d_an = self.l1(a_vgg[i].unsqueeze(1).expand(b, b, c, h, w), n_vgg[i].expand(b, b, c, h, w).detach())
            contrastive = d_ap / (d_an + 1e-7)
            loss += self.weights[i] * contrastive

        return loss


