from torch.utils.data import Dataset  # 导入PyTorch的数据集类
import numpy as np  # 导入numpy用于数值运算
from Preprocessing import getdata  # 导入预处理模块的getdata函数

# 创建特征和标签
def create_features_labels(data, window_size, index):
    X, y = [], []
    # 遍历数据，创建时间窗口内的特征和标签
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), :].T)  # 提取时间窗口内的特征并转置
        y.append([data[i + window_size, index].T])  # 提取标签并转置
    return np.array(X), np.array(y)  # 返回特征和标签的numpy数组

# 定义数据集类
class dataset(Dataset):
    def __init__(self, data, window_size, i):
        self.data = data  # 存储数据
        self.window_size = window_size  # 存储窗口大小
        self.X, self.y = create_features_labels(data, window_size, i)  # 创建特征和标签

    def __len__(self):
        return len(self.X)  # 返回数据集的长度

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # 根据索引返回对应的特征和标签

# 生成训练和测试数据集
def generate_data(i):
    # 获取国测站数据
    national_station = getdata(r"数据\国测站数据.xls")
    national_station_data = national_station.getdata().values  # 转换为numpy数组
    
    # 获取微测站1数据
    micro1_data = getdata(r"数据\微测站1数据.xls")
    micro1_data_data = micro1_data.getdata().values  # 转换为numpy数组
    
    # 获取微测站2数据
    micro2_data = getdata(r"数据\微测站2数据.xls")
    micro2_data_data = micro2_data.getdata().values  # 转换为numpy数组
    
    # 获取微测站3数据
    micro3_data = getdata(r"数据\微测站3数据.xls")
    micro3_data_data = micro3_data.getdata().values  # 转换为numpy数组

    # 合并国测站数据和微测站1、2数据作为训练数据
    train_data = np.concatenate((national_station_data, micro1_data_data, micro2_data_data), axis=0)
    print(train_data.shape)  # 打印训练数据的形状
    
    test_data = micro3_data_data  # 使用微测站3数据作为测试数据
    # print(test_data.shape)  # 打印测试数据的形状
    
    # 创建训练数据集
    train_dataset = dataset(train_data, window_size=5, i=i)
    # 创建测试数据集
    test_dataset = dataset(test_data, window_size=5, i=i)
    
    return train_dataset, test_dataset  # 返回训练和测试数据集
