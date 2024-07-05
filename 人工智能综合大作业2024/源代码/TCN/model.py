from tcn import *  # 导入TCN（Temporal Convolutional Network）模块
import torch.nn as nn  # 导入PyTorch的神经网络模块

# 定义模型类
class model(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        # 调用父类的初始化方法
        super(model, self).__init__()
        # 定义TCN层
        self.tcn = TemporalConvNet(
            input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        # 定义线性层
        self.linear = nn.Linear(num_channels[-1], output_size)

    # 定义前向传播方法
    def forward(self, inputs):
        y = self.tcn(inputs)  # 通过TCN层
        return self.linear(y[:, :, -1])  # 取最后一个时间步的输出，并通过线性层
