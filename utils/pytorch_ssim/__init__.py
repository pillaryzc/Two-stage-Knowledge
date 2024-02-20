import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

"""
### 模块名称: Structural Similarity Index Measure (SSIM) for Image Quality Assessment

#### 功能简介
- **主要作用**: 实现了结构相似性指数(SSIM)度量，用于评估两幅图像的视觉相似度。SSIM是一种常用于图像质量评估的方法，特别是在图像处理和分析领域。
- **关键功能**:
  - `gaussian`函数生成高斯窗口，用于局部图像统计的加权。
  - `create_window`函数根据通道数创建2D高斯窗口。
  - `_ssim`函数计算两幅图像之间的SSIM值。
  - `SSIM`类提供了一个易于使用的接口，用于计算图像的SSIM。

#### 关键代码段解析
- **gaussian**:
  - **功能**: 生成一维高斯分布序列。
  - **重要代码解释**: 根据高斯分布公式计算序列值，并归一化以确保总和为1。

- **create_window**:
  - **功能**: 根据给定的窗口大小和通道数生成2D高斯窗口。
  - **重要代码解释**: 通过外积将1D高斯序列转换为2D窗口，并扩展到指定的通道数。

- **_ssim**:
  - **功能**: 计算两幅图像的SSIM映射，并可选择返回平均值。
  - **重要代码解释**: 使用2D高斯窗口对图像进行局部平均、方差和协方差的计算，进而计算SSIM指数。

- **SSIM (Class)**:
  - **功能**: 提供模块化接口，方便计算两幅图像的SSIM。
  - **重要代码解释**: 根据输入图像的通道数动态创建高斯窗口，并缓存以供后续使用。

#### 输入输出
- **输入**: 
  - 两幅需要进行SSIM评估的图像。
- **输出**: 
  - 两幅图像的SSIM值，可以是一个平均值或一个SSIM映射。

#### 依赖关系
- **内部依赖**: 依赖于PyTorch框架进行张量运算。

#### 注意事项
- **特别注意**: 输入图像应为浮点张量，并且已经归一化到合适的范围内（例如，0到1或-1到1）。
- **性能考虑**: 当处理大量图像或大尺寸图像时，应注意内存和计算资源的使用。使用GPU可以显著提高计算速度。

SSIM是评估图像处理算法，如去噪、压缩和超分辨率等，对图像质量影响的重要工具。通过精确模拟人类视觉系统的感知特性，SSIM提供了一种比传统像素级误差更符合人类视觉感知的图像质量评估方法。

"""

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)
