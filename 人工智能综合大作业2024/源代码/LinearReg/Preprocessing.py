import numpy as np  # 导入numpy用于数值运算
from sklearn import preprocessing  # 导入sklearn的预处理模块
import pandas as pd  # 导入pandas用于数据操作
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
import seaborn as sns  # 导入seaborn用于绘图

# 数据预处理类
class PreProcessing():
    
    def __init__(self, filepath):
        self.file_path = filepath  # 文件路径
        # 读取Excel文件，指定缺失值
        self.data = pd.read_excel(self.file_path, na_values=['0.0(E)', '-99.0(E)'])
        self.index = self.data.iloc[1:, 0]  # 提取索引列
        self.data = self.data.iloc[1:, 7:15]  # 提取数据列
        self.data.columns = ["SO2", "CO", "NO2", "O3-1H", "O3-8H", "PM10", "PM2.5", "NO"]  # 重命名列
        self.data = self.data.drop(["O3-8H"], axis=1, inplace=False)  # 删除 "O3-8H" 列

    # 使用3σ法则去除异常值
    def remove_outliers_3_sigma(self):
        mean = self.data.mean()  # 计算均值
        std = self.data.std()  # 计算标准差
        # 过滤在均值3倍标准差范围内的数据
        condition = (self.data > (mean - 3 * std)) & (self.data < (mean + 3 * std))
        self.after_plot_box_data = self.data[condition].dropna()  # 删除异常值并移除缺失值

    # 使用箱线图法去除异常值
    def remove_outliers_box(self):
        Q1 = self.data.quantile(0.25)  # 计算第1四分位数
        Q3 = self.data.quantile(0.75)  # 计算第3四分位数
        IQR = Q3 - Q1  # 计算四分位距
        # 过滤在IQR范围内的数据
        condition = ~((self.data < (Q1 - 1.5 * IQR)) | (self.data > (Q3 + 1.5 * IQR))).any(axis=1)
        self.after_plot_box_data = self.data[condition]  # 删除异常值

    # 插值缺失值
    def interpolate(self):
        self.after_plot_box_data = pd.DataFrame(self.after_plot_box_data, dtype=float)  # 转换数据类型为浮点型
        self.interpolate_data = self.after_plot_box_data.interpolate(method="linear")  # 使用线性插值
        return self.interpolate_data  # 返回插值后的数据

    # 归一化数据
    def normalize(self):
        min_max_scaler = preprocessing.MinMaxScaler()  # 实例化MinMaxScaler
        self.norm_data = min_max_scaler.fit_transform(self.interpolate_data)  # 归一化数据
        self.norm_data = pd.DataFrame(self.norm_data, columns=self.interpolate_data.columns)  # 转换为DataFrame
        return self.norm_data  # 返回归一化后的数据

    # 绘制箱线图
    def plot_box_data(self, data):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data)
        plt.show()

    # 绘制折线图
    def plot_data(self, data):
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=self.index, y=data)
        x = np.arange(0, 8642, dtype=int)
        plt.xticks(ticks=x[::2000])
        plt.xlabel("Time")
        plt.ylabel("Data")
        plt.show()

    # 绘制分布图
    def displot(self, data):
        plt.figure(figsize=(10, 6))
        sns.displot(data, kde=True)
        plt.show()

    # 绘制成对关系图
    def pairplot(self):
        plt.figure(figsize=(10, 6))
        sns.pairplot(self.norm_data, diag_kind='kde', kind='reg', markers="*")
        plt.show()

    # 绘制直方图
    def histplot(self, data):
        plt.figure(figsize=(10, 6))
        sns.histplot(data, kde=True)
        plt.show()

# 获取数据类
class getdata():
    def __init__(self, filepath):
        super(getdata, self).__init__()  # 调用父类的初始化方法
        self.filepath = filepath  # 文件路径

    # 获取预处理后的数据
    def getdata(self):
        station = PreProcessing(self.filepath)  # 创建PreProcessing对象
        station.remove_outliers_3_sigma()  # 去除异常值
        station.interpolate()  # 插值缺失值
        station.normalize()  # 归一化数据
        self.norm_data = station.norm_data  # 获取归一化后的数据
        return self.norm_data  # 返回归一化后的数据
