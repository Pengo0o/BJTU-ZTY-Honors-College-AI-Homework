import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
import numpy as np  # 导入numpy用于数值运算

# 遍历每个特征进行绘图
for i in range(0, 7):
    truth = np.load(f"./truth_{i}.npy")  # 加载真实值数据
    a1 = np.load(f"./LinearReg/Feature_{i}_y_pred.npy")  # 加载线性回归预测结果
    a2 = np.load(f"./SVR/Feature_{i}_y_pred.npy")  # 加载SVR预测结果
    a3 = np.load(f"./TCN/Feature_{i}_y_pred.npy")  # 加载TCN预测结果
    a4 = np.load(f"./GRU/Feature_{i}_y_pred.npy")  # 加载GRU预测结果
    a5 = np.load(f"./Blending/Feature_{i}_y_pred.npy")  # 加载Blending预测结果

    # 创建折线图
    plt.figure(figsize=(10, 6))  # 可以调整图形的大小

    # 绘制truth和其他变量的折线图
    plt.plot(a1, label='LinearReg', color='blue')  # 线性回归预测结果用蓝色表示
    plt.plot(a2, label='SVR', color='red')  # SVR预测结果用红色表示
    plt.plot(a3, label='TCN', color='green')  # TCN预测结果用绿色表示
    plt.plot(a4, label='GRU', color='purple')  # GRU预测结果用紫色表示
    plt.plot(a5, label='Blending', color='orange')  # Blending预测结果用橙色表示
    plt.plot(truth, label='truth', color='black')  # 真实值用黑色表示

    # 添加图例
    plt.legend(loc="upper right")  # 在右上角添加图例

    # 添加x轴和y轴的标签
    plt.xlabel('time')  # x轴标签
    plt.ylabel('Value')  # y轴标签

    # 显示图形
    plt.show()  # 显示图形
