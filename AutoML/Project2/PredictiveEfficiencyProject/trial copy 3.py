import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # 如果无效SMILES，返回一个固定的零向量
        return np.zeros(2)  # 假设我们只使用分子量和LogP
    # 计算并返回固定长度的描述符
    descriptors = [Descriptors.MolWt(mol), Descriptors.MolLogP(mol)]
    return np.array(descriptors)

def load_data(file_path):
    # 读取CSV文件
    data = pd.read_csv(file_path)
    
    # 计算所有分子的描述符
    descriptors = data['SMILES'].apply(smiles_to_descriptors).values
    
    # 确保descriptors是一个二维数组
    descriptors = np.vstack(descriptors)  # 转换为二维数组（每个分子的描述符为一行）

    # 打印描述符形状，调试用
    print(f"Descriptors shape: {descriptors.shape}")
    
    # 确保所有描述符的长度一致（比如长度为2）
    if descriptors.shape[1] != 2:
        raise ValueError(f"描述符的形状不一致：{descriptors.shape}")
    
    # 提取其他特征
    X = np.column_stack([descriptors, data['HOMO'], data['LUMO']])
    y = data['PCE'].values
    
    # 数据归一化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    
    return X_scaled, y_scaled, scaler_X, scaler_y
# 2. 构建模型
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        super(GNNModel, self).__init__()
        # 使用多层 GCN 或 GAT 层
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # GCN层
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = self.fc(x)
        return x

# 3. 训练和评估
def train_and_evaluate(model, X_train, y_train, X_test, y_test, edge_index_train, edge_index_test, scaler_X, scaler_y, num_epochs=100, lr=0.001):
    # 转换为Tensor，X_train 和 X_test 都要归一化
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    # 归一化测试集的输入特征
    X_test_scaled = scaler_X.transform(X_test)  # 使用scaler_X对X_test进行归一化
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # 初始化优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        
        # 前向传播
        outputs = model(X_train_tensor, edge_index_train)  # 使用训练集的edge_index
        loss = criterion(outputs, y_train_tensor)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor, edge_index_test)  # 使用测试集的edge_index
        test_loss = criterion(test_outputs, y_test_tensor)
        print(f'Test Loss: {test_loss.item():.4f}')
        
        # 反归一化 PCE
        test_outputs = test_outputs.detach().numpy()
        test_outputs = scaler_y.inverse_transform(test_outputs)
        y_test = scaler_y.inverse_transform(y_test_tensor.numpy())
        
        # 评估性能
        mse = mean_squared_error(y_test, test_outputs)
        r2 = r2_score(y_test, test_outputs)
        print(f'Mean Squared Error: {mse:.4f}')
        print(f'R²: {r2:.4f}')
def build_edge_index(num_train_nodes):
    # 创建一个完全图的边索引，确保节点索引从0到num_train_nodes-1
    edge_index = []
    for i in range(num_train_nodes):
        for j in range(num_train_nodes):
            if i != j:
                edge_index.append([i, j])
    
    edge_index = np.array(edge_index).T
    return torch.tensor(edge_index, dtype=torch.long)


def main():
    data_path = 'D:/Project/ThesisProject/AutoML/Project2/PredictiveEfficiencyProject/data_csv copy.csv'
    
    params = {
        "hidden_dim": 128,
        "num_epochs": 100,
        "learning_rate": 0.001
    }
    
    noise_dim = params.get('noise_dim', 64)
    fingerprint_dim = params.get('fingerprint_dim', 2048)
    feature_dim = params.get('feature_dim', 3)
    cond_dim = params.get('cond_dim', 3)
    hidden_dim = params.get('hidden_dim', 128)
    dropout = params.get('dropout', 0.3)
    batch_size = params.get('batch_size', 32)
    num_epochs = params.get('num_epochs', 50)
    learning_rate = params.get('learning_rate', 1e-3)
    g_lr = params.get('g_lr', 1e-3)
    d_lr = params.get('d_lr', 1e-3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载数据
    X, y, scaler_X, scaler_y = load_data(data_path)
    
    # 数据集划分：训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 获取训练集和测试集的大小
    num_train_nodes = X_train.shape[0]  # 训练集中的样本数
    num_test_nodes = X_test.shape[0]    # 测试集中的样本数
    
    # 创建图的边（这里我们假设每个分子是一个完全图）
    edge_index_train = build_edge_index(num_train_nodes)  # 为训练集生成边索引
    edge_index_test = build_edge_index(num_test_nodes)    # 为测试集生成边索引
    
    # 创建GNN模型
    input_dim = X_train.shape[1]  # 输入维度
    model = GNNModel(input_dim=input_dim, hidden_dim=hidden_dim)
    
    # 训练和评估
    train_and_evaluate(model, X_train, y_train, X_test, y_test, edge_index_train, edge_index_test, scaler_X, scaler_y, num_epochs=100, lr=0.1)


# # 4. 入口函数
# def main():
#     data_path = 'D:/Project/ThesisProject/AutoML/Project2/PredictiveEfficiencyProject/data_csv copy.csv'
    
#     params = {
#         "hidden_dim": 128,
#         "num_epochs": 100,
#         "learning_rate": 0.001
#     }
    
#     noise_dim = params.get('noise_dim',64)
#     fingerprint_dim =params.get('fingerprint_dim',2048) 
#     feature_dim = params.get('feature_dim',3) 
#     cond_dim = params.get('cond_dim',3) 
#     hidden_dim = params.get('hidden_dim',128) 
#     dropout = params.get('dropout',0.3) 
#     batch_size = params.get('batch_size',32) 
#     num_epochs = params.get('num_epochs',50) 
#     learning_rate = params.get('learning_rate',1e-3) 
#     g_lr = params.get('g_lr',1e-3) 
#     d_lr = params.get('d_lr',1e-3) 
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#    # 加载数据
#     X, y, scaler_X, scaler_y = load_data(data_path)
    
#     # 获取训练集的大小
#     num_train_nodes = X.shape[0]  # 训练集中的样本数
    
#     # 创建图的边（这里我们假设每个分子是一个完全图）
#     edge_index = build_edge_index(num_train_nodes)
    
#     # 数据集划分：训练集和测试集
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
#     # 创建GNN模型
#     input_dim = X_train.shape[1]  # 输入维度
#     model = GNNModel(input_dim=input_dim, hidden_dim=hidden_dim)
    
#     # 训练和评估
#     train_and_evaluate(model, X_train, y_train, X_test, y_test, edge_index, scaler_X, scaler_y, 
#                        num_epochs, learning_rate)

if __name__ == "__main__":
    main()