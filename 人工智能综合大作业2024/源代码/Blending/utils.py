from torch.utils.data import Dataset  # 导入PyTorch的数据集类
import numpy as np  # 导入numpy用于数值运算
from Preprocessing import getdata  # 导入预处理模块的getdata函数
from sklearn.model_selection import train_test_split  # 导入train_test_split用于数据集分割

# 创建参考特征和标签
def create_extraction_ref_features_labels(data, window_size, index):
    X, y = [], []
    # 遍历数据，创建时间窗口内的特征和标签
    for i in range(len(data) - 2*window_size):
        X.append(data[i:(i + window_size), :].T)  # 提取特征并转置
        y.append(data[i + window_size:i+window_size*2, index].T)  # 提取标签并转置
    return np.array(X), np.array(y)  # 返回特征和标签的numpy数组

# 创建对象特征和标签
def create_extraction_obj_features_labels(data, window_size, index):
    X, y = [], []
    # 遍历数据，创建时间窗口内的特征和标签
    for i in range(len(data) - 2*window_size):
        X.append(data[i:(i + window_size), :])  # 提取特征
        y.append(data[i + window_size:i+window_size*2, index])  # 提取标签
    return np.array(X), np.array(y)  # 返回特征和标签的numpy数组

# 创建特征和标签
def create_extraction_features_labels(data, window_size, index):
    X, y = [], []
    # 遍历数据，创建时间窗口内的特征和标签
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), :])  # 提取特征
        y.append([data[i + window_size, index]])  # 提取标签
    return np.array(X), np.array(y)  # 返回特征和标签的numpy数组

# 创建特征和标签（转换特征维度）
def create_extraction_features_labels2(data, window_size, index):
    X, y = [], []
    # 遍历数据，创建时间窗口内的特征和标签
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), :].T)  # 提取特征并转置
        y.append([data[i + window_size, index].T])  # 提取标签并转置
    return np.array(X), np.array(y)  # 返回特征和标签的numpy数组

# 空函数，用于创建特征和标签（未实现）
def create_extraction_features_labels3(data, window_size, index):
    X, y = [], []

# 训练数据集类
class TrainDataset(Dataset):
    def __init__(self, data, window_size, i, obj):
        self.data = data  # 存储数据
        self.window_size = window_size  # 存储窗口大小
        # 根据obj参数选择不同的特征和标签创建方法
        if not obj:
            self.X, self.y = create_extraction_ref_features_labels(data, window_size, i)  # 生成参考特征和标签
        else:
            self.X, self.y = create_extraction_obj_features_labels(data, window_size, i)  # 生成对象特征和标签

    def __len__(self):
        return len(self.X)  # 返回数据集的长度

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # 根据索引返回对应的特征和标签

# 训练预测数据集类
class TrainPredictDataset():
    def __init__(self, data_ref, data_obj, window_size, i):
        self.window_size = window_size  # 存储窗口大小
        y = []
        # 遍历数据，创建时间窗口内的特征
        for idx in range(len(data_ref) - window_size):
            y.append(np.concatenate(
                (data_obj[i:(i+window_size), i], data_ref[i:(i+window_size), i]), axis=0))  # 拼接对象和参考特征
        self.X = np.array(y)  # 将拼接后的特征转换为numpy数组
        _, self.y = create_extraction_features_labels(data_obj, window_size, i)  # 创建标签

    def generate_dataset(self):
        return self.X, self.y  # 返回特征和标签

# 生成训练数据
def generate_data(i):
    national_station = getdata(r"数据\国测站数据.xls")  # 获取国测站数据
    national_station_data = national_station.getdata().values  # 转换为numpy数组
    micro1_data = getdata(r"数据\微测站1数据.xls")  # 获取微测站1数据
    micro1_data_data = micro1_data.getdata().values  # 转换为numpy数组
    micro2_data = getdata(r"数据\微测站2数据.xls")  # 获取微测站2数据
    micro2_data_data = micro2_data.getdata().values  # 转换为numpy数组
    micro3_data = getdata(r"数据\微测站3数据.xls")  # 获取微测站3数据
    micro3_data_data = micro3_data.getdata().values  # 转换为numpy数组

    train_data1 = national_station_data  # 使用国测站数据作为训练数据
    train_data2 = np.concatenate((micro1_data_data, micro2_data_data), axis=0)  # 合并微测站1和2的数据作为训练数据
    obj_train_dataset = TrainDataset(train_data2, window_size=5, i=i, obj=True)  # 创建对象训练数据集
    ref_train_dataset = TrainDataset(train_data1, window_size=5, i=i, obj=False)  # 创建参考训练数据集
    return obj_train_dataset, ref_train_dataset  # 返回对象和参考训练数据集

# 生成预测数据
def generate_predict_data(i):
    national_station = getdata(r"数据\国测站数据.xls")  # 获取国测站数据
    national_station_data = national_station.getdata().values  # 转换为numpy数组
    micro1_data = getdata(r"数据\微测站1数据.xls")  # 获取微测站1数据
    micro1_data_data = micro1_data.getdata().values  # 转换为numpy数组
    micro2_data = getdata(r"数据\微测站2数据.xls")  # 获取微测站2数据
    micro2_data_data = micro2_data.getdata().values  # 转换为numpy数组
    micro3_data = getdata(r"数据\微测站3数据.xls")  # 获取微测站3数据
    micro3_data_data = micro3_data.getdata().values  # 转换为numpy数组

    train_data1 = national_station_data[:len(micro1_data_data)]  # 截取与微测站1数据长度相同的国测站数据作为训练数据
    train_data2 = micro1_data_data  # 使用微测站1数据作为训练数据
    X, y = TrainPredictDataset(train_data1, train_data2, window_size=5, i=i).generate_dataset()  # 创建预测数据集
    return X, y  # 返回预测特征和标签

# 生成测试数据
def generate_test_data(i):
    national_station = getdata(r"数据\国测站数据.xls")  # 获取国测站数据
    national_station_data = national_station.getdata().values  # 转换为numpy数组
    micro3_data = getdata(r"数据\微测站3数据.xls")  # 获取微测站3数据
    micro3_data_data = micro3_data.getdata().values  # 转换为numpy数组
    test_data1 = national_station_data[:len(micro3_data_data)]  # 截取与微测站3数据长度相同的国测站数据作为测试数据
    test_data2 = micro3_data_data  # 使用微测站3数据作为测试数据
    X_obj, y_obj = create_extraction_features_labels2(test_data2, window_size=5, index=i)  # 创建对象特征和标签
    X_ref, _ = create_extraction_features_labels(test_data1, window_size=5, index=i)  # 创建参考特征和标签
    return X_obj, X_ref, y_obj  # 返回对象特征、参考特征和标签
