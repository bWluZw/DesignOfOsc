# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import RDKFingerprint
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import classification_report

# 1. 数据预处理部分

# 1.1 将SMILES转换为分子指纹
def smiles_to_fingerprint(smiles):
    """ 将SMILES字符串转换为分子指纹 """
    mol = Chem.MolFromSmiles(smiles)  # 使用RDKit将SMILES字符串转换为分子对象
    fingerprint = RDKFingerprint(mol)  # 计算分子指纹
    return np.array(fingerprint)  # 返回分子指纹

# 1.2 归一化HOMO和LUMO值
def normalize_homo_lumo(homo_lumo_data):
    """ 对HOMO和LUMO数据进行归一化处理 """
    scaler = MinMaxScaler()  # 初始化MinMaxScaler
    return scaler.fit_transform(homo_lumo_data)  # 将数据归一化到0到1之间

# 示例数据（实际使用时从你的数据集中读取）
smiles_data = ["CC(=O)Oc1ccccc1C(=O)O", "O=C(C1=CC=CC=C1)C(=O)O"]  # SMILES字符串
homo_lumo_data = [[-5.1, -3.5], [-4.9, -3.4]]  # HOMO和LUMO值
labels = [0, 1]  # 标签：0表示供体，1表示给体

# 1.3 转换数据
# 将SMILES字符串转换为分子指纹
fingerprint_data = np.array([smiles_to_fingerprint(smiles) for smiles in smiles_data])

# 对HOMO和LUMO值进行归一化
normalized_homo_lumo = normalize_homo_lumo(homo_lumo_data)

# 2. 定义数据集类
class SolarCellDataset(Dataset):
    """ 自定义数据集类，用于存储SMILES指纹、HOMO、LUMO和标签 """
    def __init__(self, smiles_data, homo_lumo_data, labels):
        self.smiles_data = smiles_data  # 分子指纹
        self.homo_lumo_data = homo_lumo_data  # HOMO和LUMO数据
        self.labels = labels  # 标签（0表示供体，1表示给体）

    def __len__(self):
        return len(self.labels)  # 返回数据集大小

    def __getitem__(self, idx):
        """ 返回指定索引的数据 """
        # 返回SMILES指纹、HOMO和LUMO数据、标签
        return torch.tensor(self.smiles_data[idx], dtype=torch.float32), \
               torch.tensor(self.homo_lumo_data[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.long)

# 3. 定义神经网络模型

class SolarCellClassifier(nn.Module):
    """ 有机太阳能电池材料分类模型 """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SolarCellClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 第二个全连接层
        self.relu = nn.ReLU()  # ReLU激活函数
        self.softmax = nn.Softmax(dim=1)  # Softmax输出概率分布

    def forward(self, x_smiles, x_homo_lumo):
        """ 前向传播函数，将SMILES指纹和HOMO/LUMO拼接后输入神经网络 """
        x = torch.cat((x_smiles, x_homo_lumo), dim=1)  # 将SMILES指纹和HOMO/LUMO拼接
        x = self.relu(self.fc1(x))  # 第一层全连接 + ReLU激活
        x = self.fc2(x)  # 输出层
        return self.softmax(x)  # 使用Softmax输出分类结果

# 4. 设置训练的超参数
input_dim = len(fingerprint_data[0]) + 2  # 输入维度：SMILES指纹的长度 + 2（HOMO和LUMO）
hidden_dim = 64  # 隐藏层维度
output_dim = 2  # 输出层维度（分类任务：供体和给体，共有2个类别）
model = SolarCellClassifier(input_dim, hidden_dim, output_dim)

# 5. 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数（用于分类任务）
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 6. 数据加载器
train_dataset = SolarCellDataset(fingerprint_data, normalized_homo_lumo, labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # DataLoader，用于批量加载数据

# 7. 训练模型
num_epochs = 20  # 训练的epoch次数
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0  # 累计损失
    correct_preds = 0  # 计算正确预测的数量
    total_preds = 0  # 总样本数

    for smiles, homo_lumo, labels in train_loader:
        optimizer.zero_grad()  # 清空梯度
        outputs = model(smiles, homo_lumo)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()  # 累加损失
        _, predicted = torch.max(outputs, 1)  # 获取预测类别
        correct_preds += (predicted == labels).sum().item()  # 统计正确预测的数量
        total_preds += labels.size(0)  # 总样本数

    epoch_loss = running_loss / len(train_loader)  # 计算每个epoch的平均损失
    accuracy = correct_preds / total_preds  # 计算准确率
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

# 8. 评估模型

# 假设我们有一个测试集
# 测试集的SMILES、HOMO、LUMO数据和标签
smiles_data_test = ["CC(=O)Oc1ccccc1C(=O)O", "O=C(C1=CC=CC=C1)C(=O)O"]
homo_lumo_data_test = [[-5.0, -3.4], [-4.8, -3.3]]
labels_test = [0, 1]  # 测试集标签

# 将测试集数据转换为指纹和归一化的HOMO/LUMO数据
fingerprint_data_test = np.array([smiles_to_fingerprint(smiles) for smiles in smiles_data_test])
normalized_homo_lumo_test = normalize_homo_lumo(homo_lumo_data_test)

# 创建测试集DataLoader
test_dataset = SolarCellDataset(fingerprint_data_test, normalized_homo_lumo_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 评估模型
model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 不需要计算梯度
    outputs = model(fingerprint_data_test, normalized_homo_lumo_test)  # 获取模型输出
    _, predicted = torch.max(outputs, 1)  # 获取预测类别
    print(classification_report(labels_test, predicted))  # 打印分类报告（精度、召回率、F1分数等）
