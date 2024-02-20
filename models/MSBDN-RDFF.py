import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_networks import Encoder_MDCBlock1, Decoder_MDCBlock1

'''
### 模块名称: MSBDN-RDFF
#### 功能简介
- **主要作用**: 这段代码定义了一个高级的图像去噪网络，利用深度学习技术，特别是密集连接块和残差块来提高图像去噪的效果。
- **关键功能**:
  - 使用多种类型的层和块，如密集连接层、残差块、上采样层，以及特殊的编码器和解码器块来处理和改善图像。
  - 实现了多尺度特征融合，以及深度残差学习来增强去噪能力。

#### 关键代码段解析
- **make_dense**:
  - **功能**: 实现一个密集连接层，用于生成更丰富的特征表示。
  - **重要代码解释**: 通过卷积和激活函数增加特征深度，然后将输出与输入拼接，增加特征的多样性。

- **RDB (Residual Dense Block)**:
  - **功能**: 结合残差学习和密集连接的方式，增强特征传递和重用。
  - **重要代码解释**: 通过多个密集连接层累积特征，然后用1x1卷积减少特征深度，再通过缩放和残差连接增强信息流。

- **ConvLayer & UpsampleConvLayer**:
  - **功能**: 分别实现了反射填充卷积和转置卷积层，用于特征提取和上采样。
  - **重要代码解释**: 反射填充避免了边缘效应，而转置卷积实现了特征图的上采样。

- **ResidualBlock**:
  - **功能**: 通过两个卷积层和PReLU激活函数实现残差块，用于增强模型的特征学习能力。
  - **重要代码解释**: 第二个卷积层的输出通过缩放后与输入相加，实现残差学习，提高了信息的流动性和保留。

- **Net**:
  - **功能**: 定义了整个去噪网络的架构，包括特征提取、多级特征融合和深度残差学习。
  - **重要代码解释**: 网络通过多级下采样和上采样结构实现深度特征融合，利用编码器-解码器块在不同尺度上融合特征，并通过大量的残差块增强去噪能力。

#### 输入输出
- **输入**: 待去噪的图像。
- **输出**: 去噪后的图像，以及可选的中间特征图。

#### 依赖关系
- **内部依赖**: 依赖于定义在`base_networks`模块中的`Encoder_MDCBlock1`和`Decoder_MDCBlock1`，这些是专门设计的编解码器块，用于实现特征的多尺度融合。
- **外部依赖**: 依赖PyTorch框架提供的基础层，如卷积层、激活函数等。

#### 注意事项
- **特别注意**: 确保所有的输入图像都经过了适当的预处理，以匹配网络的输入要求。
- **性能考虑**: 网络使用了大量的卷积和残差块，可能会对计算资源有较高的要求。根据可用的硬件资源适当调整模型的大小和深度

。

这个模块通过结合先进的深度学习技术和架构设计，实现了一个强大的图像去噪网络。它利用了多尺度特征融合和深度残差学习的策略，旨在提高去噪效果，同时保持图像的细节和质量。
'''

def make_model(args, parent=False):
    return Net()

class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate, scale = 1.0):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    self.scale = scale
    modules = []
    for i in range(nDenselayer):
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate
    self.dense_layers = nn.Sequential(*modules)
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out) * self.scale
    out = out + x
    return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


class Net(nn.Module):
    def __init__(self, res_blocks=18):
        super(Net, self).__init__()

        self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)
        self.dense0 = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16)
        )

        self.conv2x = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.conv1 = RDB(16, 4, 16)
        self.fusion1 = Encoder_MDCBlock1(16, 2, mode='iter2')
        self.dense1 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )

        self.conv4x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv2 = RDB(32, 4, 32)
        self.fusion2 = Encoder_MDCBlock1(32, 3, mode='iter2')
        self.dense2 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        self.conv8x = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.conv3 = RDB(64, 4, 64)
        self.fusion3 = Encoder_MDCBlock1(64, 4, mode='iter2')
        self.dense3 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self.conv16x = ConvLayer(128, 256, kernel_size=3, stride=2)
        self.conv4 = RDB(128, 4, 128)
        self.fusion4 = Encoder_MDCBlock1(128, 5, mode='iter2')

        self.dehaze = nn.Sequential()
        for i in range(0, res_blocks):
            self.dehaze.add_module('res%d' % i, ResidualBlock(256))

        self.convd16x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.dense_4 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.conv_4 = RDB(64, 4, 64)
        self.fusion_4 = Decoder_MDCBlock1(64, 2, mode='iter2')

        self.convd8x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.dense_3 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.conv_3 = RDB(32, 4, 32)
        self.fusion_3 = Decoder_MDCBlock1(32, 3, mode='iter2')

        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.dense_2 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )
        self.conv_2 = RDB(16, 4, 16)
        self.fusion_2 = Decoder_MDCBlock1(16, 4, mode='iter2')

        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)
        self.dense_1 = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16)
        )
        self.conv_1 = RDB(8, 4, 8)
        self.fusion_1 = Decoder_MDCBlock1(8, 5, mode='iter2')

        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)


    def forward(self, x, return_feat=False):
        features = []

        res1x = self.conv_input(x)
        res1x_1, res1x_2 = res1x.split([(res1x.size()[1] // 2), (res1x.size()[1] // 2)], dim=1)
        feature_mem = [res1x_1]
        x = self.dense0(res1x) + res1x

        res2x = self.conv2x(x)
        res2x_1, res2x_2 = res2x.split([(res2x.size()[1] // 2), (res2x.size()[1] // 2)], dim=1)
        res2x_1 = self.fusion1(res2x_1, feature_mem)
        res2x_2 = self.conv1(res2x_2)
        feature_mem.append(res2x_1)
        res2x = torch.cat((res2x_1, res2x_2), dim=1)
        res2x =self.dense1(res2x) + res2x
        
        res4x =self.conv4x(res2x)
        res4x_1, res4x_2 = res4x.split([(res4x.size()[1] // 2), (res4x.size()[1] // 2)], dim=1)
        res4x_1 = self.fusion2(res4x_1, feature_mem)
        res4x_2 = self.conv2(res4x_2)
        feature_mem.append(res4x_1)
        res4x = torch.cat((res4x_1, res4x_2), dim=1)
        res4x = self.dense2(res4x) + res4x
        
        features.append(res4x)

        res8x = self.conv8x(res4x)
        res8x_1, res8x_2 = res8x.split([(res8x.size()[1] // 2), (res8x.size()[1] // 2)], dim=1)
        res8x_1 = self.fusion3(res8x_1, feature_mem)
        res8x_2 = self.conv3(res8x_2)
        feature_mem.append(res8x_1)
        res8x = torch.cat((res8x_1, res8x_2), dim=1)
        res8x = self.dense3(res8x) + res8x
                
        features.append(res8x)

        res16x = self.conv16x(res8x)
        res16x_1, res16x_2 = res16x.split([(res16x.size()[1] // 2), (res16x.size()[1] // 2)], dim=1)
        res16x_1 = self.fusion4(res16x_1, feature_mem)
        res16x_2 = self.conv4(res16x_2)
        res16x = torch.cat((res16x_1, res16x_2), dim=1)

        features.append(res16x)


        res_dehaze = res16x
        in_ft = res16x*2
        res16x = self.dehaze(in_ft) + in_ft - res_dehaze
        res16x_1, res16x_2 = res16x.split([(res16x.size()[1] // 2), (res16x.size()[1] // 2)], dim=1)
        feature_mem_up = [res16x_1]

        features.append(res16x)

        res16x = self.convd16x(res16x)
        res16x = F.upsample(res16x, res8x.size()[2:], mode='bilinear')
        res8x = torch.add(res16x, res8x)
        res8x = self.dense_4(res8x) + res8x - res16x
        res8x_1, res8x_2 = res8x.split([(res8x.size()[1] // 2), (res8x.size()[1] // 2)], dim=1)
        res8x_1 = self.fusion_4(res8x_1, feature_mem_up)
        res8x_2 = self.conv_4(res8x_2)
        feature_mem_up.append(res8x_1)
        res8x = torch.cat((res8x_1, res8x_2), dim=1)

        res8x = self.convd8x(res8x)
        res8x = F.upsample(res8x, res4x.size()[2:], mode='bilinear')
        res4x = torch.add(res8x, res4x)
        res4x = self.dense_3(res4x) + res4x - res8x
        res4x_1, res4x_2 = res4x.split([(res4x.size()[1] // 2), (res4x.size()[1] // 2)], dim=1)
        res4x_1 = self.fusion_3(res4x_1, feature_mem_up)
        res4x_2 = self.conv_3(res4x_2)
        feature_mem_up.append(res4x_1)
        res4x = torch.cat((res4x_1, res4x_2), dim=1)


        res4x = self.convd4x(res4x)
        res4x = F.upsample(res4x, res2x.size()[2:], mode='bilinear')
        res2x = torch.add(res4x, res2x)
        res2x = self.dense_2(res2x) + res2x - res4x
        res2x_1, res2x_2 = res2x.split([(res2x.size()[1] // 2), (res2x.size()[1] // 2)], dim=1)
        res2x_1 = self.fusion_2(res2x_1, feature_mem_up)
        res2x_2 = self.conv_2(res2x_2)
        feature_mem_up.append(res2x_1)
        res2x = torch.cat((res2x_1, res2x_2), dim=1)

        res2x = self.convd2x(res2x)
        res2x = F.upsample(res2x, x.size()[2:], mode='bilinear')
        x = torch.add(res2x, x)
        x = self.dense_1(x) + x - res2x
        x_1, x_2 = x.split([(x.size()[1] // 2), (x.size()[1] // 2)], dim=1)
        x_1 = self.fusion_1(x_1, feature_mem_up)
        x_2 = self.conv_1(x_2)
        x = torch.cat((x_1, x_2), dim=1)

        x = self.conv_output(x)

        if return_feat:
            return x, features
        else:
            return x
      