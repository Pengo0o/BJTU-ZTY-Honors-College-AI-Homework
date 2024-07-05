from extract_feature_train import *  # 导入特征提取训练相关模块
from predict_model_train import *  # 导入预测模型训练相关模块
from model import *  # 导入模型定义模块
from sklearn.ensemble import GradientBoostingRegressor  # 导入梯度提升回归器
import torch  # 导入PyTorch
from torch.utils.data import DataLoader  # 导入DataLoader用于处理数据批次
from utils import *  # 导入工具函数
from tqdm import tqdm  # 导入tqdm用于进度条显示

# 设置随机数种子以确保可复现性
set_seed(42)

# 定义训练参数
train_args = {
    "epochs": 10,  # 训练轮数
    "window_size": 5,  # 窗口大小
    "input_size": 7,  # 输入大小
    "output_size": 5,  # 输出大小
    "num_channels": [16, 32, 64],  # 卷积通道数
    "kernel_size": 3,  # 卷积核大小
    "dropout": 0.2  # dropout率
}

# 定义网络参数
net_args = {
    "window_size": 5,  # 窗口大小
    "input_size": 7,  # 输入大小
    "output_size": 5,  # 输出大小
    "num_channels": [16, 32, 64],  # 卷积通道数
    "kernel_size": 3,  # 卷积核大小
    "dropout": 0.2  # dropout率
}

# 实例化对象特征提取模型
net1 = Object_Feature_Extraction_Model(**net_args)
# 实例化参考特征提取模型
net2 = Ref_Feature_Extraction_Model(**net_args)
# 实例化梯度提升回归器
net3 = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=1, random_state=42, loss='squared_error')

# 遍历每个特征进行训练和预测
for i in range(0, 7):
    # 训练对象和参考特征提取模型
    # nation_train(**train_args).train(net1, net2, i)
    
    # 训练预测模型
    predict_model_train().train(net3, i)
    
    # 禁用梯度计算，以节省内存并加速计算
    with torch.no_grad():
        # 生成测试数据
        X_obj, X_ref, y = generate_test_data(i)
        y_pred = []
        
        # 遍历测试数据进行预测
        for idx in tqdm(range(len(X_obj))):
            # 对象特征提取模型的前向传播
            output1 = net1(torch.tensor(X_obj[idx]).float().unsqueeze(0))
            # 参考特征提取模型的前向传播
            output2 = net2(torch.tensor(X_ref[idx]).float().unsqueeze(0))
            # 将两个输出拼接
            output = torch.cat((output1, output2), dim=1)
            # 将输出转换为numpy数组
            output = output.numpy()
            # 使用梯度提升回归器进行预测
            output3 = net3.predict(output)
            # 保存预测结果
            y_pred.append(output3)

        # 将预测结果转换为numpy数组
        y_pred = np.array(y_pred)
        print(y_pred.shape)
        print(y.shape)
        
        # 计算评价指标
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mse)
        smape = np.mean(np.abs((y - y_pred) /
                        (np.abs(y) + np.abs(y_pred))) / 2) * 100
        
        # 打印评价指标
        print(f"Feature {i} MSE: {mse}")
        print(f"Feature {i} MAE: {mae}")
        print(f"Feature {i} R2: {r2}")
        print(f"Feature {i} RMSE: {rmse}")
        print(f"Feature {i} SMAPE: {smape}")
        
        # 保存预测结果到文件
        np.save(f"./Blending/Feature_{i}_y_pred.npy", y_pred)
