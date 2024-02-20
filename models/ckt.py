import torch.nn as nn
import torch

"""
### 模块名称: Collaborative Knowledge Transfer Module

#### 功能简介
- **主要作用**: 该模块设计用于在多个教师网络和一个学生网络之间进行知识转移，通过特征投影来实现。
- **关键功能**:
  - `TeacherProjectors`用于将多个教师网络的特征投影到共同的特征空间，并重构原始特征。
  - `StudentProjector`将学生网络的特征投影到同一共同的特征空间，以便与教师网络的特征进行比较和学习。

#### 关键代码段解析
- **conv3x3**:
  - **功能**: 创建一个带有填充的3x3卷积层，用于保持特征图的尺寸。
  - **重要代码解释**: 此函数简化了卷积层的创建，用于模块内部构建网络层，无偏置参数以减少过拟合风险。

- **CKTModule (Collaborative Knowledge Transfer Module)**:
  - **功能**: 管理知识转移过程，包括教师特征的投影和重构，以及学生特征的投影。
  - **重要代码解释**: 整合`TeacherProjectors`和`StudentProjector`的功能，输出教师和学生的投影特征，用于后续的知识蒸馏或对比学习。

- **TeacherProjectors**:
  - **功能**: 对每个教师网络的特征进行投影和重构。
  - **重要代码解释**: 使用两步卷积过程，第一步是将教师特征投影到隐藏的共同特征空间，第二步是从该空间重构回原始特征空间。

- **StudentProjector**:
  - **功能**: 将学生网络的特征投影到共同的特征空间。
  - **重要代码解释**: 类似于`TeacherProjectors`，但仅包括将学生特征投影到共同特征空间的过程。

#### 输入输出
- **输入**: 
  - 对于`CKTModule`，输入包括来自多个教师网络的特征列表和单个学生网络的特征。
- **输出**: 
  - 教师网络的投影特征和重构特征，以及学生网络的投影特征。

#### 依赖关系
- **内部依赖**: 依赖于PyTorch的`nn.Module`、`nn.Conv2d`和`nn.ReLU`等基础组件来构建网络层。

#### 注意事项
- **特别注意**: 在初始化网络权重时，使用了`kaiming_normal_`初始化方法，适用于ReLU激活函数，有助于缓解梯度消失或爆炸问题。
- **性能考虑**: 该模块包含多个卷积层，因此在资源受限的环境中运行时应注意内存和计算资源的消耗。

通过精心设计的特征投影和重构机制，这个模块促进了教师网络和学生网络之间的有效知识转移，有助于提升学生网络在特定任务上的性能，特别是在模型压缩和知识蒸馏场景中。

"""

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class CKTModule(nn.Module):
    def __init__(self, channel_t, channel_s, channel_h, n_teachers):
        super().__init__()
        self.teacher_projectors = TeacherProjectors(channel_t, channel_h, n_teachers)
        self.student_projector = StudentProjector(channel_s, channel_h)
    
    def forward(self, teacher_features, student_feature):
        teacher_projected_feature, teacher_reconstructed_feature = self.teacher_projectors(teacher_features)
        student_projected_feature = self.student_projector(student_feature)

        return teacher_projected_feature, teacher_reconstructed_feature, student_projected_feature


class TeacherProjectors(nn.Module):
    """
    This module is used to capture the common features of multiple teachers.
    **Parameters:**
        - **channel_t** (int): channel of teacher features
        - **channel_h** (int): channel of hidden common features
    """
    def __init__(self, channel_t, channel_h, n_teachers):
        super().__init__()
        self.PFPs = nn.ModuleList()
        for _ in range(n_teachers):
            self.PFPs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=channel_t, out_channels=channel_h, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=channel_h, out_channels=channel_h, kernel_size=3, stride=1, padding=1, bias=False),
                )
            )

        self.IPFPs = nn.ModuleList()
        for _ in range(n_teachers):
            self.IPFPs.append(
                nn.Sequential(
                    nn.Conv2d(channel_h, channel_t, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel_t, channel_t, kernel_size=3, stride=1, padding=1, bias=False)
                )
            )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, features):
        assert len(features) == len(self.PFPs)

        projected_features = [self.PFPs[i](features[i]) for i in range(len(features))]
        reconstructed_features = [self.IPFPs[i](projected_features[i]) for i in range(len(projected_features))]

        return projected_features, reconstructed_features


class StudentProjector(nn.Module):
    """
    This module is used to project the student's features to common feature space.
    **Parameters:**
        - **channel_s** (int): channel of student features
        - **channel_h** (int): channel of hidden common features
    """
    def __init__(self, channel_s, channel_h):
        super().__init__()
        self.PFP = nn.Sequential(
            nn.Conv2d(in_channels=channel_s, out_channels=channel_h, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel_h, out_channels=channel_h, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, fs):
        projected_features = self.PFP(fs)

        return projected_features

