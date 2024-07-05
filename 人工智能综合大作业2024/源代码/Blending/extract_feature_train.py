
from Preprocessing import PreProcessing, getdata  # 导入预处理模块
import numpy as np  # 导入numpy用于数值运算
import pandas as pd  # 导入pandas用于数据操作
from torch.utils.data import DataLoader  # 导入DataLoader用于处理数据批次
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # 导入模型评估指标
from utils import *  # 导入工具函数（假设它们在utils模块中定义）
from torch.optim import Adam  # 导入Adam优化器
from torch import nn  # 导入PyTorch的神经网络模块
import torch  # 导入PyTorch
import random  # 导入random模块用于设置随机种子

# 设置随机数种子以确保可复现性
def set_seed(seed):
    torch.manual_seed(seed)  # 设置PyTorch的随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用CUDA，则还需要设置这个
    np.random.seed(seed)  # 设置numpy的随机种子
    random.seed(seed)  # 设置random模块的随机种子
    # 确保每次返回的卷积算法是确定的，如果不设置，PyTorch会自动选择最快的卷积实现算法，这可能会导致结果不一致
    torch.backends.cudnn.deterministic = True
    # 当输入数据维度或类型上变化不大时，可以提高训练速度，但是在实验中为了可复现性，通常设置为False
    torch.backends.cudnn.benchmark = False

# 设置随机数种子
set_seed(42)

# 定义训练类
class nation_train():
    def __init__(self, epochs, window_size, input_size, output_size, num_channels, kernel_size, dropout):
        self.epochs = epochs  # 训练轮数
        self.window_size = window_size  # 窗口大小
        self.input_size = input_size  # 输入大小
        self.output_size = output_size  # 输出大小
        self.num_channels = num_channels  # 通道数
        self.kernel_size = kernel_size  # 卷积核大小
        self.dropout = dropout  # dropout率

    # 训练函数
    def train(self, net1, net2, i):
        train_dataset_obj, train_dataset_ref = generate_data(i)  # 生成训练数据
        obj_train_loader = DataLoader(train_dataset_obj, batch_size=32, shuffle=True)  # 对象训练数据加载器
        ref_train_loader = DataLoader(train_dataset_ref, batch_size=32, shuffle=True)  # 参考训练数据加载器

        criterion1 = nn.L1Loss()  # 定义损失函数1
        criterion2 = nn.L1Loss()  # 定义损失函数2
        optimizer1 = Adam(net1.parameters(), lr=0.001)  # 定义优化器1
        optimizer2 = Adam(net2.parameters(), lr=0.001)  # 定义优化器2

        for epoch in range(self.epochs):  # 遍历每个训练轮次
            for x1, y1 in ref_train_loader:  # 遍历参考数据加载器
                optimizer1.zero_grad()  # 清零梯度
                output = net1(x1.float())  # 前向传播
                loss1 = criterion1(output, y1.float())  # 计算损失
                loss1.backward()  # 反向传播
                optimizer1.step()  # 更新权重

            for x2, y2 in obj_train_loader:  # 遍历对象数据加载器
                optimizer2.zero_grad()  # 清零梯度
                output = net2(x2.float())  # 前向传播
                loss2 = criterion2(output, y2.float())  # 计算损失
                loss2.backward()  # 反向传播
                optimizer2.step()  # 更新权重

            # 打印每轮的损失
            print(f"Epoch {epoch} Loss: {loss1.item(), loss2.item()}")
