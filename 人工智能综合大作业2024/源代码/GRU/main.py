from Preprocessing import PreProcessing, getdata  # 导入预处理模块
import numpy as np  # 导入numpy用于数值运算
import pandas as pd  # 导入pandas用于数据处理
from torch.utils.data import DataLoader  # 导入DataLoader用于批量加载数据
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # 导入模型评估指标
from utils import *  # 导入工具函数
from model import *  # 导入模型定义
from torch.optim import Adam  # 导入Adam优化器
import torch  # 导入PyTorch

# 定义训练参数
epochs = 10  # 训练轮数
window_size = 5  # 时间窗口大小
input_size = 7  # 输入特征的维度
output_size = 1  # 输出特征的维度

# 实例化参考特征提取模型
net = Ref_Feature_Extraction_Model(input_size, output_size, window_size)

# 遍历每个特征进行训练和测试
for i in range(0, 7):
    # 生成训练和测试数据集
    train_dataset, test_dataset = generate_data(i)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 创建训练数据加载器
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 创建测试数据加载器
    criterion = nn.MSELoss()  # 定义损失函数为均方误差
    optimizer = Adam(net.parameters(), lr=0.001)  # 定义优化器

    # 训练模型
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()  # 清零梯度
            output = net(x.float())  # 前向传播
            loss = criterion(output, y.float())  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
        print(f"Epoch {epoch} Loss: {loss.item()}")  # 打印每轮的训练损失

        # 测试模型
        with torch.no_grad():  # 禁用梯度计算，以节省内存
            for x, y in test_loader:
                output = net(x.float())  # 前向传播
                loss = criterion(output, y.float())  # 计算损失
        print(f"Epoch {epoch} Test Loss: {loss.item()}")  # 打印每轮的测试损失

    # 计算并打印评估指标
    with torch.no_grad():  # 禁用梯度计算，以节省内存
        pred = []
        for x, y in test_loader:
            output = net(x.float())  # 前向传播
            pred.append(output)  # 收集预测结果
        pred = torch.cat(pred, dim=0)  # 将预测结果拼接成一个张量
        pred = pred.numpy()  # 转换为numpy数组
        y_test_feature = test_dataset.y  # 获取测试标签
        mse = mean_squared_error(y_test_feature, pred)  # 计算均方误差
        mae = mean_absolute_error(y_test_feature, pred)  # 计算平均绝对误差
        r2 = r2_score(y_test_feature, pred)  # 计算R2得分
        rmse = np.sqrt(mse)  # 计算均方根误差
        smape = np.mean(np.abs((y_test_feature - pred) /
                               (np.abs(y_test_feature) + np.abs(pred))) / 2) * 100  # 计算对称平均绝对百分比误差
        # 打印评估指标
        print(f"Feature {i} MSE: {mse}")
        print(f"Feature {i} MAE: {mae}")
        print(f"Feature {i} R2: {r2}")
        print(f"Feature {i} RMSE: {rmse}")
        print(f"Feature {i} SMAPE: {smape}")
        np.save(f"./GRU/Feature_{i}_y_pred.npy", pred)  # 保存预测结果到文件
