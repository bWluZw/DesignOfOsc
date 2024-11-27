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
    df = pd.read_csv(csv_file)
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


class GATModel(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, out_channels=3, heads=16, dropout=0.2):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.conv2_1 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.conv2_2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.conv2_3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.conv3 = GATConv(hidden_channels * heads, out_channels * heads, heads=heads)
        self.dropout = dropout
        self.attention = nn.Linear(hidden_channels * heads, 1)  # 学习节点重要性
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.dropout(x, p=self.dropout, train=self.training)
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.dropout(x, p=self.dropout, train=self.training)
        
        x = torch.relu(self.conv2_1(x, edge_index))
        x = torch.dropout(x, p=self.dropout, train=self.training)
        
        x = torch.relu(self.conv2_2(x, edge_index))
        x = torch.relu(self.conv2_3(x, edge_index))
        x = self.conv3(x, edge_index)
        
        # 聚合所有节点的特征，得到图级别的预测值
        x = torch.sum(x, dim=0, keepdim=True)  # 计算所有节点特征的和
        return x


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
            loss = criterion(out, data.y)
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


def main():
    params = {
        "hidden_dim": 128,
        "num_epochs": 50,
        "learning_rate": 0.0005,
        "weight_decay": 0.1,
    }

    # 选择设备，自动选择GPU（如果可用）或CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    csv_file = 'D:/Project/ThesisProject/AutoML/Project2/PredictiveEfficiencyProject/data_csv copy.csv'
    train_loader, test_loader, pce_scaler, homo_scaler, lumo_scaler = load_data(csv_file)

    # 构建模型
    model = GATModel(in_channels=3, hidden_channels=params["hidden_dim"], out_channels=1, dropout=0.3)
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
            predicted_pce.append(out.cpu().numpy())  # 将结果迁回CPU
            actual_pce.append(data.y.cpu().numpy())  # 将结果迁回CPU

    predicted_pce = np.concatenate(predicted_pce)
    actual_pce = np.concatenate(actual_pce)

    # 反归一化PCE值
    predicted_pce = pce_scaler.inverse_transform(predicted_pce)
    actual_pce = pce_scaler.inverse_transform(actual_pce)

    # 输出预测结果
    print(f"Predicted PCE: {predicted_pce[:10]}")
    print(f"Actual PCE: {actual_pce[:10]}")


if __name__ == "__main__":
    main()
