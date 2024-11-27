import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GATConv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import rdmolops
import networkx as nx
import torch_geometric
import torch.nn.functional as F
from sklearn.metrics import r2_score  # 导入r2_score函数，用于计算R²
from sklearn.model_selection import train_test_split
from torch_geometric.nn import global_mean_pool # 引入全局平均池化函数
import logging
# import nni.retiarii.nn.pytorch as nni
# from nni.nas.nn.pytorch import LayerChoice, ModelSpace, MutableDropout, MutableLinear
import nni

# 日志设置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('custom_trial_gpu.py')
# region 数据处理
def smiles_to_graph(smiles, homo, lumo, pce):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 创建NetworkX图
    G = nx.Graph()

    # 添加原子节点
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), feature=atom.GetAtomicNum())
    
    # 添加边（键）
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    # 获取分子图的邻接矩阵
    adj_matrix = nx.adjacency_matrix(G).todense()
    edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)

    # 特征矩阵，使用原子编号作为特征
    node_features = torch.tensor([G.nodes[i]['feature'] for i in range(len(G.nodes))], dtype=torch.float).view(-1, 1)

    # 将HOMO和LUMO作为全局特征加到每个节点上
    homo_tensor = torch.tensor(np.full((node_features.size(0), 1), homo), dtype=torch.float)  # 复制homo值到每个节点
    lumo_tensor = torch.tensor(np.full((node_features.size(0), 1), lumo), dtype=torch.float)  # 复制lumo值到每个节点

    # 将HOMO和LUMO与原子特征组合
    node_features = torch.cat((node_features, homo_tensor, lumo_tensor), dim=-1)

    # 创建torch_geometric的Data对象
    data = torch_geometric.data.Data(x=node_features, edge_index=edge_index)

    # 设置图的目标（PCE值）
    data.y = torch.tensor(pce, dtype=torch.float).view(-1, 1)  # 目标值（PCE）
    return data


def load_data(csv_file, batch_size):
    df = pd.read_csv(csv_file,encoding='utf-8')
    smiles = df['SMILES'].values
    homo = df['HOMO'].values
    lumo = df['LUMO'].values
    pce = df['PCE'].values

    # 数据归一化
    homo_scaler = StandardScaler()
    lumo_scaler = StandardScaler()
    pce_scaler = StandardScaler()

    homo = homo_scaler.fit_transform(homo.reshape(-1, 1))
    lumo = lumo_scaler.fit_transform(lumo.reshape(-1, 1))
    pce = pce_scaler.fit_transform(pce.reshape(-1, 1))

    # 将SMILES转化为分子图
    mol_graphs = [smiles_to_graph(smile, h, l, p) for smile, h, l, p in zip(smiles, homo, lumo, pce)]
    # 将数据转换为GraphData格式
    data_list = []
    for graph in mol_graphs:
        if graph is None:
            continue
        data_list.append(graph)

    # 划分训练和测试集
    train_size = int(0.8 * len(data_list))
    train_data = data_list[:train_size]
    test_data = data_list[train_size:]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, pce_scaler, homo_scaler, lumo_scaler
# 假设你已经有了一个材料数据集和特征提取方法
# 比如：smiles, homo, lumo, gap, pce 这些特征

# endregion
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import nni

# region 定义可微的操作空间
# 1x1卷积操作（用于处理一维特征）
class LinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearLayer, self).__init__()
        self.conv = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x)

# 最大池化操作（用于提取显著特征）
class MaxPool(nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()

    def forward(self, x):
        return F.max_pool2d(x, kernel_size=2, stride=2)

# 平均池化操作（用于对特征进行平滑处理）
class AvgPool(nn.Module):
    def __init__(self):
        super(AvgPool, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=2, stride=2)

# 跳跃连接（skip connection）
class SkipConnection(nn.Module):
    def __init__(self):
        super(SkipConnection, self).__init__()

    def forward(self, x):
        return x  # 直接返回输入

# 激活函数：ReLU
class ReLUActivation(nn.Module):
    def __init__(self):
        super(ReLUActivation, self).__init__()

    def forward(self, x):
        return F.relu(x)

# 图卷积层（假设数据是图结构，SMILES转化为图后应用此操作）
class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)  # 假设输入是图的特征向量
# endregion

# region 定义动态Cell
class DynamicCell(nn.Module):
    def __init__(self, in_channels, out_channels, operations):
        """
        in_channels: 输入特征的维度
        out_channels: 输出特征的维度
        operations: 操作集（例如 GCNConv 和 GATConv）
        """
        super(DynamicCell, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.operations = operations  # 操作集

        # 动态操作选择的参数
        self.alpha = nn.Parameter(torch.randn(len(operations)))  # 每个操作的选择权重

    def forward(self, x, edge_index):
        """
        x: 节点特征矩阵 (batch_size, num_nodes, in_channels)
        edge_index: 图的边索引
        """
        # 计算每个操作的权重，通过softmax归一化
        alpha = F.softmax(self.alpha, dim=0)  # softmax使得所有权重和为1
        
        # 将每个操作加权聚合
        out = 0
        for i, op in enumerate(self.operations):
            out += alpha[i] * op(x, edge_index)  # 根据权重加权每个操作的输出
        
        return out
# endregion


# region 定义整个网络
class DartsGraphNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_cells, num_classes, operations):
        """
        DARTS网络，由多个动态Cell组成

        in_channels: 输入特征的维度（通常是原子的特征）
        out_channels: 每个节点的输出特征维度
        num_cells: 网络中Cell的数量
        num_classes: 输出的类别数（例如，PCE预测值的维度）
        operations: 用于DARTS架构搜索的操作集（例如GCNConv和GATConv）
        """
        super(DartsGraphNet, self).__init__()
        
        # 构建多个Cell
        self.cells = nn.ModuleList()
        for _ in range(num_cells):
            cell = DynamicCell(in_channels, out_channels, operations)  # 每个Cell选择不同的操作
            self.cells.append(cell)
        
        # 最后一个全连接层进行PCE值的预测
        self.fc = nn.Linear(out_channels, num_classes)
    
    def forward(self, x, edge_index):
        """
        前向传播，经过多个Cell处理图数据

        x: 节点特征矩阵 (batch_size, num_nodes, in_channels)
        edge_index: 图的边索引 (2, num_edges)
        """
        # 通过多个Cell处理输入数据
        for cell in self.cells:
            x = cell(x, edge_index)  # 每个Cell的输出是下一个Cell的输入
        
        # 使用global_mean_pool进行图级别的池化
        x = global_mean_pool(x, batch=None)  # 聚合所有节点的特征
        
        # 通过全连接层进行PCE值预测
        x = self.fc(x)
        return x
# endregion

# class DartsNet(nn.Module):
#     def __init__(self, C, num_classes, num_cells=6):
#         super(DartsNet, self).__init__()

#         self.cells = nn.ModuleList()
#         for i in range(num_cells):
#             if i < num_cells // 2:
#                 self.cells.append(DynamicCell(C))  # 前半部分是Normal Cell
#             else:
#                 self.cells.append(DynamicCell(C * 2))  # 后半部分是Reduction Cell

#         self.fc = nn.Linear(C, num_classes)  # 最后的全连接层

#     def forward(self, x, op_indices):
#         for cell in self.cells:
#             x = cell(x, op_indices)  # 每个Cell的输入是上一个Cell的输出

#         x = self.fc(x)  # 最后通过全连接层输出结果
#         return x
# class Network(nn.Module):
#     def __init__(self, C_in, C_out, num_cells, C_cell):
#         super(Network, self).__init__()
#         self.stem = nn.Conv2d(C_in, C_cell, 3, 1, 1)
#         self.cells = nn.ModuleList()
#         for _ in range(num_cells):
#             self.cells.append(Cell(C_cell, C_cell, 1))  # 每个Cell中的层
#         self.fc = nn.Linear(C_cell, C_out)

#     def forward(self, x, weights):
#         out = self.stem(x)
#         for cell in self.cells:
#             out = cell(out, weights)
#         out = out.mean([2, 3])  # Global Average Pooling
#         out = self.fc(out)
#         return out


# import torch.optim as optim

# def train(model, train_loader, criterion, optimizer, arch_optimizer):
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
#     for inputs, targets in train_loader:
#         inputs, targets = inputs.cuda(), targets.cuda()

#         # 对于每个batch，首先进行前向传播计算损失
#         optimizer.zero_grad()
#         arch_optimizer.zero_grad()

#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()

#         optimizer.step()
#         arch_optimizer.step()

#         total_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         total += targets.size(0)
#         correct += (predicted == targets).sum().item()

#     accuracy = 100 * correct / total
#     return total_loss / len(train_loader), accuracy


# def trial(params,train_loader,test_loader):
#     # 根据超参数配置初始化网络
#     model = ArchitectureSearch(C_in=params['C_in'], C_out=params['C_out'], 
#                                num_cells=params['num_cells'], C_cell=params['C_cell']).cuda()

#     # 定义损失函数和优化器
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.network.parameters(), lr=params['lr'])
#     arch_optimizer = optim.Adam(model.weights, lr=params['lr'])

#     # 训练模型
#     for epoch in range(params['epochs']):
#         loss, accuracy = train(model, train_loader, criterion, optimizer, arch_optimizer)
#         print(f'Epoch {epoch+1}/{params["epochs"]}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%')
    
#     # 返回最终的准确率（可以调整为PCE预测）
#     return accuracy

def get_r2(predicted_pce,actual_pce,pce_scaler):
    # TODO 改为准确度
    predicted_pce = np.concatenate(predicted_pce)
    actual_pce = np.concatenate(actual_pce)

    # 反归一化PCE值
    predicted_pce = pce_scaler.inverse_transform(predicted_pce.reshape(actual_pce.size,1))
    actual_pce = pce_scaler.inverse_transform(actual_pce.reshape(actual_pce.size,1))

    # 输出预测结果
    # print(f"Predicted PCE: {predicted_pce[:10]}")
    # print(f"Actual PCE: {actual_pce[:10]}")

    # 计算并输出R²
    r2 = r2_score(actual_pce, predicted_pce)
    return r2

def trial_and_eval(model,train_loader,test_loader,nni_params):
    criterion = nn.MSELoss()  # 如果是PCE回归任务，可以使用MSELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 假设你有训练数据x_train和edge_index_train
    # x_train, edge_index_train 是训练数据的节点特征和边索引

    # 训练过程
    for epoch in range(nni_params['num_epochs']):
        best_model = None
        best_loss = float('inf')
        model.train()
        for data in range(train_loader):
            train_loss = 0
            data = data.to(nni_params['device'])
            optimizer.zero_grad()
            output = model(data, data.edge_index)
            loss = criterion(output, data.y)  # y_train 是真实的PCE值
            loss.backward()
            optimizer.step()# 更新权重
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for data in test_loader:
                data = data.to(nni_params['device'])  # 将数据迁移到GPU
                out = model(data)
                loss = criterion(out, data.y)
                test_loss += loss.item()
            test_loss /= len(test_loader)
            nni.report_intermediate_result()
            print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            if test_loss < best_loss:
                best_loss = test_loss
                best_model = model.state_dict()
    return best_model

# 使用NNI进行超参数调优
def main():
    # nni_params = nni.get_next_parameter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_file = 'D:/Project/ThesisProject/AutoML/Project2/PredictiveEfficiencyProject/data_csv copy.csv'
    # 定义超参数搜索空间
    nni_params = {
        'num_cells': {'_type': 'choice', '_value': [3, 4, 5]},  # 选择3到5个cell
        'C_cell': {'_type': 'choice', '_value': [16, 32]},  # 选择不同的cell维度
        'C_in': 3,  # 输入通道数
        'C_out': 1,  # 输出类别数
        'lr': {'_type': 'uniform', '_value': [0.0001, 0.1]},  # 学习率
        'epochs': 20,  # 训练周期
        'batch_size': 20,
    }
    operations = {
        "conv1x1": LinearLayer,
        "maxpool": MaxPool,
        "avgpool": AvgPool,
        "skip": SkipConnection,
        "relu": ReLUActivation,
        "graphconv": GraphConvLayer,  # 这个操作用于图数据
    }
    # # NNI超参数调优
    # nni_params = {
    #     "input_dim": 3,  # 输入特征维度
    #     "output_dim": 1,  # PCE预测的输出维度
    #     "num_layers": {"_type": "choice", "_value": [1, 2, 3, 4]},  # 搜索的层数
    #     "learning_rate": {"_type": "loguniform", "_value": [0.00001, 0.1]},
    #     "layer_0_dim": 64,  # 第一层的神经元数
    #     "layer_1_dim": 128,  # 第二层的神经元数
    #     "layer_2_dim": 256,  # 第三层的神经元数
    #     "activation_fn": "relu",  # 激活函数
    #     "batch_size": 32,
    #     "epochs": 20,
    #     "layer_0_type": "linear",  # 第一层是全连接层
    #     "layer_1_type": "conv2d",  # 第二层是卷积层
    #     "layer_2_type": "lstm",  # 第三层是LSTM层
    # }
    # # 示例输入
    # batch_size = 10
    # num_nodes = 100
    # in_channels = 64  # 假设每个节点有64维特征
    # out_channels = 64  # 每个节点的输出特征维度
    # num_cells = 3  # 网络包含3个Cell
    # num_classes = 1  # 预测PCE值
    train_loader, test_loader, pce_scaler, homo_scaler, lumo_scaler = load_data(csv_file,nni_params['batch_size'])
    
    # 构造网络
    model = DartsGraphNet(nni_params['in_channels'], nni_params['out_channels'], nni_params['num_cells'], nni_params['num_classes'], operations)

    best_model = trial_and_eval(train_loader,test_loader,nni_params)
    
    model.load_state_dict(best_model)
    model.eval()
    # 对测试集进行预测并反归一化
    predicted_pce = []
    actual_pce = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)  # 将数据迁移到GPU
            out = model(data)
            out = out.view(-1) # 确保输出维度与目标维度一致
            predicted_pce.append(out.cpu().numpy())  # 将结果迁回CPU
            actual_pce.extend(data.y.cpu().numpy())  # 将结果迁回CPU

    predicted_pce = np.concatenate(predicted_pce)
    actual_pce = np.concatenate(actual_pce)

    # 反归一化PCE值
    predicted_pce = pce_scaler.inverse_transform(predicted_pce.reshape(actual_pce.size,1))
    actual_pce = pce_scaler.inverse_transform(actual_pce.reshape(actual_pce.size,1))
    r2 = r2_score(actual_pce, predicted_pce)
    nni.report_final_result(r2)

if __name__ == '__main__':
    main()
