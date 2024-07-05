from Preprocessing import PreProcessing, getdata  # 导入预处理模块
import numpy as np  # 导入numpy用于数值运算
import pandas as pd  # 导入pandas用于数据操作
from sklearn.model_selection import train_test_split  # 导入数据集分割工具
from sklearn.svm import SVR  # 导入支持向量机回归模型
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # 导入评估指标
from sklearn.multioutput import MultiOutputRegressor  # 导入多输出回归器

# 加载数据
national_station = getdata(r"数据\国测站数据.xls")  # 获取国测站数据
national_station_data = national_station.getdata().values  # 转换为numpy数组
micro1_data = getdata(r"数据\微测站1数据.xls")  # 获取微测站1数据
micro1_data_data = micro1_data.getdata().values  # 转换为numpy数组
micro2_data = getdata(r"数据\微测站2数据.xls")  # 获取微测站2数据
micro2_data_data = micro2_data.getdata().values  # 转换为numpy数组
micro3_data = getdata(r"数据\微测站3数据.xls")  # 获取微测站3数据
micro3_data_data = micro3_data.getdata().values  # 转换为numpy数组

# 创建特征和标签
def create_features_labels(data, window_size):
    X, y = [], []
    # 遍历数据，创建时间窗口内的特征和标签
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), :])  # 提取时间窗口内的特征
        y.append([data[i + window_size, :]])  # 提取标签
    return np.array(X), np.array(y)  # 返回特征和标签的numpy数组

# 定义窗口大小
window_size = 5
# 合并国测站数据和微测站1、2数据作为训练数据
train_data = np.concatenate((national_station_data, micro1_data_data, micro2_data_data), axis=0)
# 使用微测站3数据作为测试数据
test_data = micro3_data_data

# 创建训练和测试数据的特征和标签
X_train, y_train = create_features_labels(national_station_data, window_size)
X_test, y_test = create_features_labels(test_data, window_size)

# 遍历每个特征进行训练和测试
for i in range(0, 7):
    y_train_feature = y_train[:, 0, i]  # 提取训练标签中的第i个特征
    y_test_feature = y_test[:, 0, i]  # 提取测试标签中的第i个特征

    svr = SVR(kernel='rbf', C=1e3, gamma=0.1)  # 实例化支持向量机回归模型，使用RBF核
    svr.fit(X_train.reshape(X_train.shape[0], -1), y_train_feature)  # 训练模型
    y_pred = svr.predict(X_test.reshape(X_test.shape[0], -1))  # 预测测试集

    # 计算评估指标
    mse = mean_squared_error(y_test_feature, y_pred)  # 计算均方误差
    mae = mean_absolute_error(y_test_feature, y_pred)  # 计算平均绝对误差
    r2 = r2_score(y_test_feature, y_pred)  # 计算R2得分
    rmse = np.sqrt(mse)  # 计算均方根误差
    smape = np.mean(np.abs((y_test_feature - y_pred) /
                    (np.abs(y_test_feature) + np.abs(y_pred))) / 2) * 100  # 计算对称平均绝对百分比误差
    
    # 打印评估指标
    print(f"Feature {i} MSE: {mse}")
    print(f"Feature {i} MAE: {mae}")
    print(f"Feature {i} R2: {r2}")
    print(f"Feature {i} RMSE: {rmse}")
    print(f"Feature {i} SMAPE: {smape}")
    
    # 保存预测结果到文件
    np.save(f"./SVR/Feature_{i}_y_pred.npy", y_pred)
