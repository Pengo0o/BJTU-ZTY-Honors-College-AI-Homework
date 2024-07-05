import torch  # 导入PyTorch
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch.nn.utils import weight_norm  # 导入权重归一化工具
from tcn import *  # 导入Temporal Convolutional Network (TCN) 相关模块

# 对象特征提取模型类定义
class Object_Feature_Extraction_Model(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2, **other_args):
        super(Object_Feature_Extraction_Model, self).__init__()
        # 定义Temporal Convolutional Network (TCN)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        # 定义全连接层
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # 通过TCN进行前向传播
        y = self.tcn(x)
        # 取最后一个时间步的输出并通过全连接层
        return self.linear(y[:, :, -1])

# 参考特征提取模型类定义
class Ref_Feature_Extraction_Model(nn.Module):
    def __init__(self, input_size, output_size, window_size, **other_args):
        super(Ref_Feature_Extraction_Model, self).__init__()
        # 定义双层GRU
        self.gru = nn.GRU(input_size, output_size, num_layers=2, batch_first=True)
        # 定义全连接网络
        self.body = nn.Sequential(
            nn.Flatten(),  # 展平层
            nn.Linear(window_size * output_size, 5),  # 全连接层
        )

    def forward(self, x):
        # 通过GRU进行前向传播
        x, _ = self.gru(x)
        # 通过全连接网络
        x = self.body(x)
        return x

# 预测模型类定义
class Predict_Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Predict_Model, self).__init__()
        # 后续可以在此定义预测模型的结构
        pass
