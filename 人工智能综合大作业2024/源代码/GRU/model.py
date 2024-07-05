from torch import nn  # 导入PyTorch的神经网络模块

# 定义参考特征提取模型类
class Ref_Feature_Extraction_Model(nn.Module):
    def __init__(self, input_size, output_size, window_size):
        # 调用父类的初始化方法
        super(Ref_Feature_Extraction_Model, self).__init__()
        
        # 定义双层GRU网络，batch_first=True表示输入数据的第一个维度是batch大小
        self.gru = nn.GRU(input_size, output_size,
                          num_layers=2, batch_first=True)
        
        # 定义一个顺序容器，包括展平层和全连接层
        self.body = nn.Sequential(
            nn.Flatten(),  # 将多维输入展平为一维
            nn.Linear(window_size*output_size, 1),  # 全连接层，将输入映射到单一输出
        )

    # 定义前向传播方法
    def forward(self, x):
        x, _ = self.gru(x)  # 通过GRU网络，返回输出和隐藏状态，忽略隐藏状态
        x = self.body(x)  # 将GRU输出通过顺序容器中的层
        return x  # 返回最终输出
