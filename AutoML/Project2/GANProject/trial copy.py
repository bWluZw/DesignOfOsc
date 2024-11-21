import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch.nn.functional as F

# 自定义数据集类
class OrganicDataset(Dataset):
    def __init__(self, smiles_features, conditions):
        self.features = smiles_features  # 数值化后的 SMILES 特征
        self.conditions = conditions  # 条件信息 (HOMO, LUMO, Mark)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.conditions[idx]


# 加载数据函数
def load_data(csv_path, smiles_column, homo_column, lumo_column, mark_column, max_smiles_len=128):
    """加载数据集并进行预处理"""
    # 读取 CSV 文件
    data = pd.read_csv(csv_path)

    # 检查是否有缺失值
    if data.isnull().any().any():
        raise ValueError("CSV 文件包含缺失值，请清洗数据后再试。")

    # 提取列
    smiles = data[smiles_column]
    homo = data[homo_column].values
    lumo = data[lumo_column].values
    mark = data[mark_column].values  # 给体或受体分类

    # 将 SMILES 转化为数值特征
    # smiles_features = []

    # 将 SMILES 转换为分子对象
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles]

    # 生成 Morgan 指纹 (半径 = 2, 位数 = 2048)
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in molecules]

    # 转换为 NumPy 数组
    fingerprints_array = np.array([np.array(fp) for fp in fingerprints])

    # smiles_features = np.array(smiles_features)

    # 合并条件 (HOMO, LUMO, Mark)
    conditions = np.stack([homo, lumo, mark], axis=1)

    return fingerprints_array, conditions.astype(np.float32)


# # 生成器模型
# class Generator(nn.Module):
#     def __init__(self, noise_dim, condition_dim, output_dim):
#         super(Generator, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(noise_dim + condition_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512, output_dim),
#             nn.Tanh()  # 输出范围 [-1, 1]
#         )

#     def forward(self, z, condition):
#         input_data = torch.cat([z, condition], dim=1)
#         return self.net(input_data)

class Generator(nn.Module):
    def __init__(self, noise_dim, fingerprint_dim, feature_dim, hidden_dim=128, dropout=0.3):
        """
        生成器模型
        Args:
            noise_dim (int): 随机噪声的维度
            fingerprint_dim (int): Morgan指纹的维度
            feature_dim (int): 物理化学特征的维度（如 HOMO, LUMO, Gap）
            hidden_dim (int): 隐藏层神经元数量
            dropout (float): Dropout 概率
        """
        super(Generator, self).__init__()
        
        # 随机噪声到隐藏层
        self.input_layer = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 隐藏层
        self.hidden_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 输出分支：分子指纹
        self.fingerprint_output = nn.Sequential(
            nn.Linear(hidden_dim, fingerprint_dim),
            nn.Tanh()  # 输出范围在 [-1, 1]，适合表示指纹
        )
        
        # 输出分支：物理化学特征
        self.feature_output = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim),
            nn.Tanh()  # 可缩放到实际特征范围
        )

    def forward(self, noise):
        """
        前向传播
        Args:
            noise (torch.Tensor): 随机噪声向量 (batch_size, noise_dim)
        Returns:
            dict: 生成的分子数据，包括指纹和物理化学特征
        """
        hidden = self.input_layer(noise)
        hidden = self.hidden_layer(hidden)
        
        # 生成 Morgan 指纹和物理化学特征
        fingerprint = self.fingerprint_output(hidden)
        features = self.feature_output(hidden)
        
        return {
            "fingerprint": fingerprint,
            "features": features
        }


# # 判别器模型
# class Discriminator(nn.Module):
#     def __init__(self, input_dim, condition_dim):
#         super(Discriminator, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim + condition_dim, 512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 1),
#             nn.Sigmoid()  # 输出概率
#         )

#     def forward(self, x, condition):
#         input_data = torch.cat([x, condition], dim=1)
#         return self.net(input_data)


class Discriminator(nn.Module):
    def __init__(self, fingerprint_dim, feature_dim, hidden_dim=128, dropout=0.3):
        """
        判别器模型
        Args:
            fingerprint_dim (int): Morgan指纹的维度
            feature_dim (int): 物理化学特征的维度（如 HOMO, LUMO, Gap）
            hidden_dim (int): 隐藏层神经元数量
            dropout (float): Dropout 概率
        """
        super(Discriminator, self).__init__()
        
        # 分子指纹处理分支
        self.fp_branch = nn.Sequential(
            nn.Linear(fingerprint_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 物理特征处理分支
        self.feature_branch = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 综合分支
        self.combined_branch = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 输出层
        self.realness_output = nn.Linear(hidden_dim, 1)  # 用于判断真实性
        self.class_output = nn.Linear(hidden_dim, 2)    # 用于分类受体/给体

    def forward(self, fingerprint, features):
        """
        前向传播
        Args:
            fingerprint (torch.Tensor): 分子指纹 (batch_size, fingerprint_dim)
            features (torch.Tensor): 物理化学特征 (batch_size, feature_dim)
        Returns:
            dict: 包括真实分数和分类结果
        """
        fp_out = self.fp_branch(fingerprint)
        feature_out = self.feature_branch(features)
        
        # 合并两部分特征
        combined = torch.cat([fp_out, feature_out], dim=1)
        combined_out = self.combined_branch(combined)
        
        # 输出
        realness = torch.sigmoid(self.realness_output(combined_out))
        class_logits = self.class_output(combined_out)
        
        return {
            "realness": realness,       # 真实性分数
            "class_logits": class_logits  # 受体/给体分类的 logits
        }


# 训练和评估函数
def train_and_evaluate(data_loader, generator, discriminator, g_optimizer, d_optimizer, criterion, num_epochs, noise_dim,save_path):
    for epoch in range(num_epochs):
        for real_features, conditions in data_loader:
            batch_size = real_features.size(0)

            # 转为 GPU（如可用）
            real_features = real_features.float()
            conditions = conditions.float()

            # 训练判别器
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # 判别器处理真实样本
            real_outputs = discriminator(real_features, conditions)
            d_loss_real = criterion(real_outputs, real_labels)

            # 判别器处理假样本
            z = torch.randn(batch_size, noise_dim)
            fake_features = generator(z, conditions)
            fake_outputs = discriminator(fake_features.detach(), conditions)
            d_loss_fake = criterion(fake_outputs, fake_labels)

            # 判别器总损失
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # 训练生成器
            z = torch.randn(batch_size, noise_dim)
            fake_features = generator(z, conditions)
            fake_outputs = discriminator(fake_features, conditions)
            g_loss = criterion(fake_outputs, real_labels)  # 希望假样本被判为真

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        # 每隔 100 个 epoch 打印损失
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
            torch.save(generator.state_dict(), f"{save_path}/generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"{save_path}/discriminator_epoch_{epoch}.pth")

    # 最终保存一次
    torch.save(generator.state_dict(), f"{save_path}/generator_final.pth")
    torch.save(discriminator.state_dict(), f"{save_path}/discriminator_final.pth")
    print("Model saved successfully.")

# 主函数
def main():
        # 参数设置
    noise_dim = 64
    fingerprint_dim = 2048
    feature_dim = 3
    hidden_dim = 128
    dropout = 0.3
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 数据加载（假设数据已预处理为张量）
    real_fingerprints = torch.randn(1000, fingerprint_dim)  # 示例真实指纹
    real_features = torch.randn(1000, feature_dim)          # 示例真实物理特征
    dataset = SolarCellDataset(real_fingerprints, real_features)
    real_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    generator = Generator(noise_dim, fingerprint_dim, feature_dim, hidden_dim, dropout)
    discriminator = Discriminator(fingerprint_dim, feature_dim, hidden_dim, dropout)
    
    # 优化器
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    
    # 训练和评估
    train_and_evaluate(generator, discriminator, g_optimizer, d_optimizer, 
                       real_data_loader, noise_dim, num_epochs, device)
    
    # 生成新分子
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(10, noise_dim, device=device)  # 生成 10 个新样本
        generated_data = generator(noise)
        fingerprints = generated_data['fingerprint']
        features = generated_data['features']
        print("Generated Fingerprints:", fingerprints)
        print("Generated Features (HOMO, LUMO, Gap):", features)

    # # 参数设置
    # csv_path = "D:\Project\ThesisProject\AutoML\data\SMILES_donors_and_NFAs.csv"  # 替换为你的 CSV 文件路径
    # save_path = "D:\Project\ThesisProject\AutoML\Project2\GANProject\GANModel.pyh"  # 替换为你的 CSV 文件路径
    # smiles_column = "smiles"
    # homo_column = "homo"
    # lumo_column = "lumo"
    # mark_column = "mark"

    # feature_dim = 128  # SMILES 转化后的特征维度
    # condition_dim = 3  # 条件维度 (HOMO, LUMO, Mark)
    # noise_dim = 100  # 随机噪声维度
    # output_dim = feature_dim

    # num_epochs = 1000
    # batch_size = 64

    # # 加载数据
    # smiles_features, conditions = load_data(csv_path, smiles_column, homo_column, lumo_column, mark_column)
    # dataset = OrganicDataset(smiles_features, conditions)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # # 初始化生成器和判别器
    # generator = Generator(noise_dim, condition_dim, output_dim)
    # discriminator = Discriminator(feature_dim, condition_dim)

    # # 优化器
    # g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # # 损失函数
    # criterion = nn.BCELoss()

    # # 训练模型
    # train_and_evaluate(data_loader, generator, discriminator, g_optimizer, d_optimizer, criterion, num_epochs, noise_dim,save_path)

    # # 生成新样本
    # with torch.no_grad():
    #     test_conditions = torch.tensor(np.random.rand(10, condition_dim).astype(np.float32))
    #     z = torch.randn(10, noise_dim)
    #     generated_features = generator(z, test_conditions)
    #     print("Generated Features:", generated_features)


if __name__ == "__main__":
    main()
