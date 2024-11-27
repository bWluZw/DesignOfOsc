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

# 1. 定义模型架构 - 支持动态生成架构
class DynamicNN(nn.Module):
    def __init__(self, layers, layer_types, activation_fn):
        super(DynamicNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_types = layer_types
        for i in range(len(layers) - 1):
            if layer_types[i] == 'linear':
                self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            elif layer_types[i] == 'conv2d':
                self.layers.append(nn.Conv2d(layers[i], layers[i + 1], kernel_size=3, stride=1, padding=1))
            elif layer_types[i] == 'lstm':
                self.layers.append(nn.LSTM(layers[i], layers[i + 1], batch_first=True))
            # Add more layer types as needed (e.g., 'gru', 'batchnorm', 'dropout', etc.)
            else:
                raise ValueError(f"Unknown layer type: {layer_types[i]}")

        self.activation_fn = activation_fn

    def forward(self, x):
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], nn.LSTM):
                # If it's an LSTM, we need to handle the input format correctly
                x, _ = self.layers[i](x)
                x = x[:, -1, :]  # Select the last timestep's output for sequence models
            else:
                x = self.layers[i](x)
                if i < len(self.layers) - 1:  # Apply activation function except for the last layer
                    x = self.activation_fn(x)
        return x

# 2. 生成架构方法 - 根据超参数生成网络架构
def generate_architecture(params):
    layers = [params['input_dim']] + [params[f'layer_{i}_dim'] for i in range(params['num_layers'])] + [params['output_dim']]
    layer_types = [params[f'layer_{i}_type'] for i in range(params['num_layers'])]
    activation_fn = getattr(F, params['activation_fn'])  # 动态获取激活函数
    model = DynamicNN(layers, layer_types, activation_fn)
    return model
    # layer_sizes = [params['input_dim']] + [params[f'layer_{i}_dim'] for i in range(params['num_layers'])] + [params['output_dim']]
    # activation_fn = getattr(F, params['activation_fn'])  # 动态获取激活函数
    # model = DynamicNN(layer_sizes, activation_fn)
    # return model

# 3. 评估模型方法
def evaluate_model(model, data_loader, criterion, optimizer):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, targets = data
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            # 根据任务需求计算正确率或其他评估指标
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# 4. 搜索NAS (基于进化算法)
def search_nas(train_loader, validation_loader, nni_params):
    # 使用NNI的Trial参数进行架构生成与评估
    model = generate_architecture(nni_params)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 回归问题使用MSE损失
    optimizer = optim.Adam(model.parameters(), lr=nni_params['learning_rate'])

    # # 训练模型
    # train_loader = DataLoader(training_data, batch_size=nni_params['batch_size'], shuffle=True)
    # validation_loader = DataLoader(validation_data, batch_size=nni_params['batch_size'], shuffle=False)

    model.train()
    for epoch in range(nni_params['epochs']):
        total_loss = 0
        for data in train_loader:
            inputs, targets = data.x,data.y
            print(data.x)
            print(data.x.shape)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{nni_params['epochs']}, Loss: {total_loss / len(train_loader)}")

    # 评估模型
    val_loss, accuracy = evaluate_model(model, validation_loader, criterion, optimizer)

    # 返回调优的PCE预测精度（可以根据需求调整返回值）
    return val_loss, accuracy

# 5. NNI超参数调优（包含NAS架构搜索）
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_file = 'D:/Project/ThesisProject/AutoML/Project2/PredictiveEfficiencyProject/data_csv copy.csv'

    # NNI超参数调优
    nni_params = {
        "input_dim": 3,  # 输入特征维度
        "output_dim": 1,  # PCE预测的输出维度
        "num_layers": {"_type": "choice", "_value": [1, 2, 3, 4]},  # 搜索的层数
        "learning_rate": {"_type": "loguniform", "_value": [0.00001, 0.1]},
        "layer_0_dim": 64,  # 第一层的神经元数
        "layer_1_dim": 128,  # 第二层的神经元数
        "layer_2_dim": 256,  # 第三层的神经元数
        "activation_fn": "relu",  # 激活函数
        "batch_size": 32,
        "epochs": 20,
        "layer_0_type": "linear",  # 第一层是全连接层
        "layer_1_type": "conv2d",  # 第二层是卷积层
        "layer_2_type": "lstm",  # 第三层是LSTM层
    }

    train_loader, test_loader, pce_scaler, homo_scaler, lumo_scaler = load_data(csv_file,nni_params['batch_size'])
    # 在NNI中进行NAS搜索
    best_loss, best_accuracy = search_nas(train_loader, test_loader, nni_params)
    print(f"Best Validation Loss: {best_loss}, Best Accuracy: {best_accuracy}")

# 6. 运行主函数
if __name__ == '__main__':
    main()
