from Preprocessing import PreProcessing, getdata  # 导入预处理模块
import numpy as np  # 导入numpy用于数值运算
import pandas as pd  # 导入pandas用于数据操作
from torch.utils.data import DataLoader  # 导入DataLoader用于处理数据批次
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # 导入模型评估指标
from utils import *  # 导入工具函数（假设它们在utils模块中定义）
from torch.optim import Adam  # 导入Adam优化器
from torch import nn  # 导入PyTorch的神经网络模块
import torch  # 导入PyTorch
from model import Ref_Feature_Extraction_Model, Object_Feature_Extraction_Model  # 导入模型定义
import random  # 导入random模块用于设置随机种子
from tqdm import tqdm  # 导入tqdm用于进度条显示

# 定义预测模型训练类
class predict_model_train():
    def __init__(self) -> None:
        pass  # 初始化方法，此处不需要任何操作

    # 训练方法
    def train(self, net, i):
        # 生成预测数据
        X, y = generate_predict_data(i)
        # 打印数据形状
        print(X.shape, y.shape)
        # 训练模型
        net.fit(X, y.ravel())  # 调用模型的fit方法进行训练，y需要展平成一维数组
