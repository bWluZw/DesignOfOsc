import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 数据处理模块
# ------------------------------

def smiles_to_graph(smiles):
    """将SMILES字符串转换为分子图"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    edge_index = []
    for bond in mol.GetBonds():
        edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    atom_features = torch.tensor(atom_features, dtype=torch.float).view(-1, 1)
    return Data(x=atom_features, edge_index=edge_index)

def load_data(file_path):
    """加载数据集并转换为图格式"""
    data = pd.read_csv(file_path)
    graphs = []

    for i, row in data.iterrows():
        graph = smiles_to_graph(row['SMILES'])
        if graph is not None:
            graph.y = torch.tensor([row['PCE_max(%)']], dtype=torch.float)
            graph.y = np.log1p(graph.y)  # 对PCE做log变换以提高训练稳定性
            graphs.append(graph)
    
    # 拆分为训练集和测试集
    return train_test_split(graphs, test_size=0.2, random_state=42)

# ------------------------------
# 模型构建模块
# ------------------------------

class GNNModel(nn.Module):
    """基于图神经网络的回归模型"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def build_model(input_dim=1, hidden_dim=64, output_dim=1):
    """构建模型"""
    return GNNModel(input_dim, hidden_dim, output_dim)

# ------------------------------
# 训练与评估模块
# ------------------------------

def train(model, loader, optimizer, criterion, device):
    """训练模型"""
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data).squeeze(-1)  # 调整输出形状
        loss = criterion(output, data.y.view(-1))  # 调整目标形状
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data).squeeze(-1)  # 调整输出形状
            loss = criterion(output, data.y.view(-1))  # 调整目标形状
            total_loss += loss.item()
            y_true.extend(data.y.cpu().numpy().flatten())
            y_pred.extend(output.cpu().numpy().flatten())

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return total_loss / len(loader), mae, r2

# ------------------------------
# 可视化模块
# ------------------------------

def plot_metrics(train_losses, test_losses, maes, r2s):
    """绘制训练过程中的损失、MAE和R²指标"""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss during training')

    # 绘制MAE和R²
    plt.subplot(1, 2, 2)
    plt.plot(epochs, maes, label="MAE")
    plt.plot(epochs, r2s, label="R²")
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.title('MAE and R² during training')

    plt.tight_layout()
    plt.show()

# ------------------------------
# 主程序模块
# ------------------------------

def main():
    # 参数设置
    file_path = "D:/Project/ThesisProject/AutoML/Project2/PredictiveEfficiencyProject/data_csv copy.csv"  # 数据集路径
    batch_size = 32
    learning_rate = 0.001
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_graphs, test_graphs = load_data(file_path)
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    # 构建模型与优化器
    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # 训练与测试
    train_losses, test_losses, maes, r2s = [], [], [], []

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss, mae, r2 = evaluate(model, test_loader, criterion, device)
        
        # 记录训练过程中的数据
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        maes.append(mae)
        r2s.append(r2)
        
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    # 绘制训练过程中的指标
    plot_metrics(train_losses, test_losses, maes, r2s)

if __name__ == "__main__":
    main()
