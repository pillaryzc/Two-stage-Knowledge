
"""
### 模块名称: Performance Metrics Tracking Utility

#### 功能简介
- **主要作用**: 提供了一个工具，用于计算和跟踪一系列性能指标的平均值和当前值，适用于深度学习训练和评估过程中的指标监控。
- **关键功能**:
  - `AverageMeter`类用于存储和更新单个指标的统计信息。
  - `get_meter`函数快速初始化指定数量的`AverageMeter`实例。
  - `update_meter`函数更新一组`AverageMeter`实例的值。

#### 关键代码段解析
- **AverageMeter**:
  - **功能**: 跟踪和计算单个性能指标的平均值、当前值、总和及计数。
  - **重要代码解释**: 通过`update`方法更新统计数据，`reset`方法重置所有统计数据。

- **get_meter**:
  - **功能**: 初始化指定数量的`AverageMeter`对象，用于跟踪多个性能指标。
  - **重要代码解释**: 返回一个`AverageMeter`对象列表，方便同时管理多个性能指标的统计数据。

- **update_meter**:
  - **功能**: 批量更新一组`AverageMeter`实例的值。
  - **重要代码解释**: 遍历每个`AverageMeter`实例和对应的值，调用`update`方法进行更新。

#### 输入输出
- **输入**: 
  - 对于`update_meter`函数，输入为一组`AverageMeter`实例和相应的值列表。
- **输出**: 
  - 更新后的`AverageMeter`实例列表。

#### 依赖关系
- **内部依赖**: 无外部依赖，独立于具体的深度学习框架。

#### 注意事项
- **特别注意**: 在使用这些工具时，确保正确地初始化和更新每个`AverageMeter`实例以避免统计数据的不准确。

#### 通俗的形象举例
想象你是一名教练，正在训练一支足球队。每个球员（相当于一个性能指标）都有自己的得分记录（例如进球数）。作为教练，你需要跟踪每个球员的总进球数、平均进球数等。这就像使用`AverageMeter`来记录和更新每个球员的表现。使用`get_meter`就像为每个球员分配一个记录本，而`update_meter`则是在每场比赛后更新这些记录本。

#### 如何使用
```python
# 初始化用于跟踪三个不同指标（如精度、损失、召回率）的平均计量器
meters = get_meter(3)

# 假设在一次迭代中，这些指标的值分别为0.9, 0.2, 0.5
values = [0.9, 0.2, 0.5]

# 更新计量器
update_meter(meters, values)

# 打印更新后的平均值
for i, meter in enumerate(meters):
    print(f"Metric {i+1} - Current Value: {meter.val}, Average: {meter.avg}")
```
这段代码展示了如何初始化和更新多个性能指标的平均值跟踪器，便于在模型训练和评估时监控关键指标。
"""



class AverageMeter(object):

	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def get_meter(num_meters):
	return [AverageMeter() for _ in range(num_meters)]


def update_meter(meters, values, n=1):
	for meter, value in zip(meters, values):
		meter.update(value, n=n)

	return meters