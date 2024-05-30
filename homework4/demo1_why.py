
import numpy as np
import math
import time
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


def GetData(CityNum):
    """
    函数名：GetData()
    函数功能：	从外界读取城市数据并处理
        输入	无
        输出	1 Position：各个城市的位置矩阵
            2 CityNum：城市数量
            3 Dist：城市间距离矩阵
    其他说明：无
    """
    # CityNum:代表城市数量
    Position = np.random.rand(CityNum, 2)  # 从城市A到B的距离矩阵

    Dist = np.zeros((CityNum, CityNum))  # Dist(i,j)：城市i与城市j间的距离

    # 计算距离矩阵
    for i in range(CityNum):
        for j in range(CityNum):
            if (abs(i - j) <= 23):
                Dist[i][j] = (i + 1) * (j + 1) // math.gcd(i + 1, j + 1)
            else:
                Dist[i][j] = 1e7
    return Position, CityNum, Dist


def ResultShow(Min_Path, BestPath, CityNum, string):
    """
    函数名：GetData()
    函数功能：	从外界读取城市数据并处理
        输入	无
        输出	1 Position：各个城市的位置矩阵
            2 CityNum：城市数量
            3 Dist：城市间距离矩阵
    其他说明：无
    """
    print("基于" + string + "求得的旅行商最短路径为：")
    for m in range(CityNum):
        print(str(BestPath[m]) + "—>", end="")
        with open('Min_path.txt', 'a') as f:
            # 写入文本
            f.write(str(BestPath[m]) + "—>")

    print(BestPath[CityNum])
    print("总路径长为：" + str(Min_Path))
    with open('Min_path.txt', 'a') as f:
        # 写入文本
        f.write('0\n'+str(Min_Path)+'\n')
    print()


def draw(BestPath, Position, title):
    """
    函数名：draw(BestPath,Position,title)
    函数功能：	通过最优路径将旅行商依次经过的城市在图表上绘制出来
        输入	1 	BestPath：最优路径
            2	Position：各个城市的位置矩阵
            3	title:图表的标题
        输出	无
    其他说明：无
    """
    plt.title(title)
    plt.plot(Position[:, 0], Position[:, 1], 'bo')
    for i, city in enumerate(Position):
        plt.text(city[0], city[1], str(i))
    plt.plot(Position[BestPath, 0], Position[BestPath, 1], color='red')
    plt.savefig(f"road_plot/{title}.png")
    plt.show()


def CalcPath_sum(layer, i):
    """
        函数名：CalcPath_sum(layer,i)
        函数功能：计算从初始城市到第layer层再到接下来的第i个城市所经历的总距离
            输入	1: layer 回溯所处的层数，也即所遍历的城市数
                2: i 当前层数下接下来要访问的子节点，即要访问的下一个城市
            输出	1: Path_sum 求的的是当前递归所处的层数的累积路径值+到下一个节点的距离
        其他说明：无
    """
    # 计算从初始城市到第layer层
    Path_sum = sum([Dist[city1][city2]
                   for city1, city2 in zip(Curpath[:layer], Curpath[1:layer + 1])])

    # 计算从初始城市到第layer层再到接下来的第i个城市所经历的总距离
    Path_sum += Dist[Curpath[i - 1]][i]

    return Path_sum


def DFSMethod(Dist, CityNum, layer):
    """
    函数名：DFSMethod(Dist,CityNum,layer)
    函数功能： 深度优先搜索算法核心
        输入	1 CityNum：城市数量
            2 Dist：城市间距离矩阵
            3 layer:旅行商所处层数，也即遍历的城市数
        输出	：无
    其他说明：无
    """
    global Path_sum, Cur_Min_Path, Min_Path, BestPath
    # 如果所有城市都遍历完成，则记录最小
    if (layer == CityNum):
        Path_sum = CalcPath_sum(layer, 0)
        if (Path_sum <= Cur_Min_Path):
            Cur_Min_Path = Path_sum
            Min_Path = Cur_Min_Path
            BestPath = Curpath.tolist()
            BestPath.append(0)
    # 否则递归回溯
    else:
        for i in range(layer, CityNum):
            Curpath[i], Curpath[layer] = Curpath[layer], Curpath[i]  # 路径交换一下
            DFSMethod(Dist, CityNum, layer + 1)
            Curpath[i], Curpath[layer] = Curpath[layer], Curpath[i]  # 路径交换回来


VisitedCities = set()


def GetLowerBound(city, CityNum, Dist):
    """
    函数名：GetLowerBound(city, CityNum, Dist)
    函数功能：计算从当前城市到剩余未访问城市的最小可能成本估计值
    输入    1. city：当前城市的索引
            2. CityNum：城市的总数
            3. Dist：城市之间的距离矩阵
    输出    ：从当前城市到剩余未访问城市的最小可能成本估计值
    """
    # 使用 Prim 算法计算最小生成树的成本
    # 初始化一个空集合来存储已访问的城市
    visited = set([city])
    # 初始化最小生成树的总成本
    total_cost = 0

    # 在剩余的城市中循环，直到所有城市都被访问
    while len(visited) < CityNum:
        min_cost = float('inf')  # 初始化最小成本为正无穷大
        min_city = None  # 初始化最小成本对应的城市为 None

        # 在已访问的城市中寻找连接到未访问城市的最小距离边
        for i in visited:
            for j in range(CityNum):
                if j not in visited and Dist[i][j] < min_cost:
                    min_cost = Dist[i][j]
                    min_city = j

        # 将找到的最小成本边添加到最小生成树中
        total_cost += min_cost
        # 将对应的城市标记为已访问
        visited.add(min_city)

    # 返回最小生成树的总成本作为下界
    return total_cost


def DFSMethod_prove(Dist, CityNum, layer):
    """
    函数名：DFSMethod(Dist,CityNum,layer)
    函数功能： 深度优先搜索算法核心
        输入    1 CityNum：城市数量
                2 Dist：城市间距离矩阵
                3 layer:旅行商所处层数，也即遍历的城市数
        输出    ：无
    其他说明：无
    """

    global Path_sum, Cur_Min_Path, Min_Path, BestPath, VisitedCities
    # 如果所有城市都遍历完成，则记录最小
    if (layer == CityNum):
        Path_sum = CalcPath_sum(layer, 0)
        if (Path_sum <= Cur_Min_Path):
            Cur_Min_Path = Path_sum
            Min_Path = Cur_Min_Path
            BestPath = Curpath.tolist()
            BestPath.append(0)
    # 否则递归回溯
    else:
        for i in range(layer, CityNum):
            if i not in VisitedCities:  # 剪枝：如果城市已经访问过，则跳过
                Curpath[i], Curpath[layer] = Curpath[layer], Curpath[i]  # 路径交换一下
                VisitedCities.add(i)  # 将当前城市标记为已访问
                # 剪枝：如果当前路径加上最小距离已经超过最小路径长度，则不再继续搜索
                if CalcPath_sum(layer, 0) + GetLowerBound(layer, CityNum, Dist) < Cur_Min_Path:
                    DFSMethod(Dist, CityNum, layer + 1)
                Curpath[i], Curpath[layer] = Curpath[layer], Curpath[i]  # 路径交换回来
                VisitedCities.remove(i)  # 将当前城市标记为未访问


############################## 程序入口#########################################
if __name__ == "__main__":
    run_time_dfs = []
    run_time_dfs_prove = []
    Path_dfs = []
    Path_dfs_prove = []
    points_num = []
    for i in tqdm(range(13, 24)):
        Position, CityNum, Dist = GetData(i)
        Curpath = np.arange(CityNum)
        Min_Path = 0
        BestPath = []
        Cur_Min_Path = math.inf

        start1 = time.perf_counter()  # 程序计时开始
        DFSMethod(Dist, CityNum, 1)  # 调用深度优先搜索核心算法
        end1 = time.perf_counter()  # 程序计时结束

        # *********************************************
        start2 = time.perf_counter()  # 程序计时开始
        DFSMethod_prove(Dist, CityNum, 1)  # 调用深度优先搜索核心算法
        end2 = time.perf_counter()  # 程序计时结束
        points_num.append(i)
        run_time_dfs.append(end1 - start1)
        run_time_dfs_prove.append(end2-start2)

        print()
        ResultShow(Min_Path, BestPath, CityNum, "穷举法之深度优先搜索策略")
        draw(BestPath, Position, f"DFS Method(points_num={i})")
        print(f"执行时间：{run_time_dfs}/n{run_time_dfs_prove}")

    with open('time.txt', 'a') as f:
        # 写入文本
        f.write('原：'+str(run_time_dfs)+'\n'+'剪枝：'+str(run_time_dfs_prove)+'\n')
    ax = plt.figure(figsize=(10, 6))
    plt.plot(points_num, run_time_dfs, marker='o', linestyle='-')
    plt.plot(points_num, run_time_dfs_prove, marker='x', linestyle='-')
    plt.title('Time for different Points_num')
    plt.xlabel('Points_num')
    plt.ylabel('Time')
    plt.grid(True)
    plt.show()
