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


def load_data(csv_file, batch_size=32):
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


from torch_geometric.nn import global_mean_pool

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=1, heads=16, dropout=0.01):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        # self.conv2_1 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        # self.conv2_2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        # self.conv2_3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.dropout = dropout
        self.fc = nn.Linear(hidden_channels * heads, out_channels)  # 新增线性层，将输出维度调整为目标维度

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.dropout(x, p=self.dropout, train=self.training)
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.dropout(x, p=self.dropout, train=self.training)
        # x = torch.relu(self.conv2_1(x, edge_index))
        # x = torch.dropout(x, p=self.dropout, train=self.training)
        # x = torch.relu(self.conv2_2(x, edge_index))
        # x = torch.relu(self.conv2_3(x, edge_index))
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)  # 使用全局平均池化

        x = self.fc(x)  # 线性层调整输出维度

        return x


# class GATModel(nn.Module):
#     def __init__(self, in_channels=3, hidden_channels=64, out_channels=1, heads=1, dropout=0.2):
#         super(GATModel, self).__init__()
#         self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
#         self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
#         self.conv2_1 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
#         self.conv2_2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
#         self.conv2_3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
#         self.conv3 = GATConv(hidden_channels * heads, out_channels , heads=heads)

#         # 修改此处，确保 fc_out 接受的输入大小是与 x 的维度一致的
#         # self.fc_out = nn.Linear(out_channels * heads, hidden_channels)  # out_channels * heads 是你最后输出的维度

#         self.dropout = dropout
#         self.hidden_channels = hidden_channels
#         self.heads = heads
#         self.fc = nn.Linear(hidden_channels * heads, out_channels) # 新增线性层，将输出维度调整为目标维度

#     def forward(self, data):
#         x, edge_index,batch = data.x, data.edge_index, data.batch
#         print(f"Initial x shape: {x.shape}")

#         x = torch.relu(self.conv1(x, edge_index))
#         print(f"After conv1: {x.shape}")
#         x = torch.dropout(x, p=self.dropout, train=self.training)

#         x = torch.relu(self.conv2(x, edge_index))
#         print(f"After conv2: {x.shape}")
#         x = torch.dropout(x, p=self.dropout, train=self.training)

#         x = torch.relu(self.conv2_1(x, edge_index))
#         print(f"After conv2_1: {x.shape}")
#         x = torch.dropout(x, p=self.dropout, train=self.training)

#         x = torch.relu(self.conv2_2(x, edge_index))
#         print(f"After conv2_2: {x.shape}")
#         x = torch.relu(self.conv2_3(x, edge_index))
#         print(f"After conv2_3: {x.shape}")
#         x = self.conv3(x, edge_index)
#         print(f"After conv3: {x.shape}")

#         # 聚合所有节点的特征，得到图级别的预测值
#         x = global_mean_pool(x, batch)  # 聚合所有节点特征
#         print(f"After mean aggregation: {x.shape}")
#         # x = x.view(-1, 1) # 确保输出维度为 [batch_size, 1]
#         # # 将图级别的特征映射到单一的预测值
#         # x = self.fc_out(x)
#         # print(f"After fc_out: {x.shape}")
#         # x = x.view(-1, self.hidden_channels * self.heads)
#         x = self.fc(x) # 调整输出维度 print(f"After linear layer: {x.shape}")
#         return x


def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, epochs=50, device='cuda'):
    best_model = None
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)  # 将数据迁移到GPU
            optimizer.zero_grad()
            out = model(data)
            # out = out.view(-1)
            loss = criterion(out, data.y)
            # loss = criterion(out, data.y.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            test_loss = 0
            for data in test_loader:
                data = data.to(device)  # 将数据迁移到GPU
                out = model(data)
                loss = criterion(out, data.y)
                test_loss += loss.item()
            test_loss /= len(test_loader)
            print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            if test_loss < best_loss:
                best_loss = test_loss
                best_model = model.state_dict()

    return best_model

import nni
def main():
    # params = {
    #     "hidden_dim": 256,
    #     "num_epochs": 50,
    #     "learning_rate": 0.001,
    #     "weight_decay": 0.3,
    #     "heads": 8,
    #     "dropout": 0.2,
    # }

    params = nni.get_next_parameter()
    params = {
        "hidden_dim": 256,
        "num_epochs": 36,
        "activation": "sigmoid",
        "learning_rate": 0.07522965550362888,
        "weight_decay": 0.0012869736605943714,
        "heads": 16,
        "dropout": 0.00008977286506338217,
        "batch_size": 16,
    }
    if(params.__len__()==0):
        params = {
            "hidden_dim": 256,
            "num_epochs": 50,
            "learning_rate": 0.001,
            "weight_decay": 0.3,
            "heads": 8,
            "dropout": 0.2,
        }
    # 选择设备，自动选择GPU（如果可用）或CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_file = 'D:/Project/ThesisProject/AutoML/Project2/PredictiveEfficiencyProject/data_csv copy.csv'
    train_loader, test_loader, pce_scaler, homo_scaler, lumo_scaler = load_data(csv_file)

    # 构建模型
    model = GATModel(in_channels=3, hidden_channels=params["hidden_dim"], out_channels=1,heads=params["heads"], dropout=params["dropout"])
    model = model.to(device)  # 将模型迁移到GPU

    optimizer = optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    criterion = nn.MSELoss()

    # 训练和评估
    best_model = train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, params["num_epochs"], device=device)

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

    # 输出预测结果
    # print(f"Predicted PCE: {predicted_pce[:10]}")
    # print(f"Actual PCE: {actual_pce[:10]}")

    # 计算并输出R²
    r2 = r2_score(actual_pce, predicted_pce)
    # print(f"R² Score: {r2:.4f}")
    nni.report_final_result(r2)

if __name__ == "__main__":
    main()
