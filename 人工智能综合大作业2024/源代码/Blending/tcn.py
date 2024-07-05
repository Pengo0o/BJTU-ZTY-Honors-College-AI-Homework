import torch  # 导入PyTorch
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch.nn.utils import weight_norm  # 导入权重归一化工具

# 定义Chomp1d类，用于去除卷积操作后多余的填充
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size  # 要去除的填充大小

    def forward(self, x):
        # 去除填充并返回结果
        return x[:, :, :-self.chomp_size].contiguous()

# 定义TemporalBlock类，包含两个带有Chomp1d操作的卷积层
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 定义第一个卷积层，使用权重归一化
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)  # 定义Chomp1d层
        self.relu1 = nn.ReLU()  # 定义ReLU激活函数
        self.dropout1 = nn.Dropout(dropout)  # 定义Dropout层

        # 定义第二个卷积层，使用权重归一化
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 定义Chomp1d层
        self.relu2 = nn.ReLU()  # 定义ReLU激活函数
        self.dropout2 = nn.Dropout(dropout)  # 定义Dropout层

        # 将层组合成一个顺序容器
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        # 如果输入输出通道数不同，定义下采样层
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()  # 定义ReLU激活函数
        self.init_weights()  # 初始化权重

    # 初始化权重
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    # 定义前向传播
    def forward(self, x):
        out = self.net(x)  # 通过网络前向传播
        res = x if self.downsample is None else self.downsample(x)  # 下采样
        return self.relu(out + res)  # 返回残差连接后的结果

# 定义TemporalConvNet类，包含多个TemporalBlock
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)  # 通道数的层数
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数
            in_channels = num_inputs if i == 0 else num_channels[i-1]  # 输入通道数
            out_channels = num_channels[i]  # 输出通道数
            # 添加TemporalBlock层
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)  # 将所有层组合成一个顺序容器

    # 定义前向传播
    def forward(self, x):
        return self.network(x)  # 通过网络前向传播
