import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch.nn.functional as F
    
class SolarCellDataset(Dataset):
    """
    自定义数据集类，用于加载有机太阳能电池数据。
    """
    def __init__(self, csv_path, radius,fingerprint_dim=2048):
        """
        初始化数据集。
        Args:
            csv_path (str): CSV 文件路径
            fingerprint_dim (int): Morgan 指纹的维度
        """
        self.data = pd.read_csv(csv_path)
        self.fingerprint_dim = fingerprint_dim

        # 预处理：转换 SMILES 到指纹
        self.fingerprints = []
        for smiles in self.data['smiles']:
            try:
                mol = Chem.MolFromSmiles(smiles)
                fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fingerprint_dim)
                array = list(fingerprint)
                fingerprint = torch.tensor(array, dtype=torch.float32)
                self.fingerprints.append(fingerprint)
            except ValueError as e:
                print(e)
                self.fingerprints.append(torch.zeros(fingerprint_dim))  # 无效分子填充 0

        # 提取物理化学特征
        self.features = torch.tensor(self.data[['homo', 'lumo', 'gap']].values, dtype=torch.float32)
        self.labels = torch.tensor(self.data['mark'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "fingerprint": self.fingerprints[idx],
            "features": self.features[idx],
            "label": self.labels[idx]
        }


#(self, noise_dim, fingerprint_dim, feature_dim, hidden_dim=128, dropout=0.3):
# 加载数据函数
def load_data(csv_path, smiles_column, homo_column, lumo_column, mark_column,radius=2,fingerprint_dim=2048):

    """加载数据集并进行预处理"""
    # 读取 CSV 文件
    all_data_df = pd.read_csv(csv_path)

    # 检查是否有缺失值
    if all_data_df.isnull().any().any():
        raise ValueError("CSV 文件包含缺失值，请清洗数据后再试。")

    # 提取列
    smiles = all_data_df[smiles_column]
    homo = all_data_df[homo_column].values
    lumo = all_data_df[lumo_column].values
    mark = all_data_df[mark_column].values  # 给体或受体分类
    res_df = pd.DataFrame(columns=['fingerprints','conditions'], dtype='object')
    fingerprint_arr_list=[]
    for (index,item) in all_data_df.iterrows():
        mol = Chem.MolFromSmiles(item[smiles_column])
        mol = Chem.AddHs(mol=mol)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fingerprint_dim)
        fingerprint_arr = np.array(fingerprint)
        fingerprint_arr_list.append(torch.Tensor(fingerprint_arr))
    
    fingerprint_res = torch.stack(fingerprint_arr_list)
    features_res = torch.tensor(all_data_df[[homo_column, lumo_column, 'gap']].values,dtype=torch.float32)
    lables = torch.tensor(all_data_df[mark_column].values,dtype=torch.float32)

    return fingerprint_res, features_res,lables


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

def train_and_evaluate(generator, discriminator, g_optimizer, d_optimizer, 
                       fingerprints,features,labels,batch_size,noise_dim, num_epochs=50, device="cpu"):
    """
    训练和评估生成器和判别器的函数。
    Args:
        generator (nn.Module): 生成器模型
        discriminator (nn.Module): 判别器模型
        g_optimizer (torch.optim.Optimizer): 生成器优化器
        d_optimizer (torch.optim.Optimizer): 判别器优化器
        smiles_features: 真实数据的加载器
        noise_dim (int): 生成器输入噪声的维度
        num_epochs (int): 训练的轮数
        device (str): 运行设备 ("cpu" 或 "cuda")
    """
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    num_samples = len(fingerprints)
    for epoch in range(num_epochs):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        # 打乱数据
        indices = torch.randperm(num_samples)
        features = features[indices]
        labels = labels[indices]
        fingerprints = fingerprints[indices]
        
        for i in range(0, num_samples, batch_size):
            # 构造当前批次
            batch_end = min(i + batch_size, num_samples)
            real_fingerprints = fingerprints[i:batch_end].to(device)
            real_features = features[i:batch_end].to(device)
            real_labels = torch.ones(real_fingerprints.size(0), 1, device=device)
            fake_labels = torch.zeros(real_fingerprints.size(0), 1, device=device)
            
            ## 判别器训练 ##
            d_optimizer.zero_grad()
            
            # 判别真实数据
            real_output = discriminator(real_fingerprints, real_features)
            realness_loss_real = F.binary_cross_entropy(real_output['realness'], real_labels)
            
            # 生成假数据
            noise = torch.randn(real_fingerprints.size(0), noise_dim, device=device)
            generated_data = generator(noise)
            fake_fingerprints = generated_data['fingerprint']
            fake_features = generated_data['features']
            
            # 判别假数据
            fake_output = discriminator(fake_fingerprints, fake_features)
            realness_loss_fake = F.binary_cross_entropy(fake_output['realness'], fake_labels)
            
            # 判别器总损失
            d_loss = realness_loss_real + realness_loss_fake
            d_loss.backward()
            d_optimizer.step()
            d_loss_epoch += d_loss.item()
            
            ## 生成器训练 ##
            g_optimizer.zero_grad()
            
            # 生成新数据并计算生成器损失
            noise = torch.randn(real_fingerprints.size(0), noise_dim, device=device)
            generated_data = generator(noise)
            fake_fingerprints = generated_data['fingerprint']
            fake_features = generated_data['features']
            
            # 判别器对生成数据的输出
            fake_output = discriminator(fake_fingerprints, fake_features)
            g_loss = F.binary_cross_entropy(fake_output['realness'], real_labels)
            
            # 添加物理化学特性约束（可选）
            gap_constraint = torch.mean((fake_features[:, 1] - fake_features[:, 0] - 2) ** 2)
            g_loss += 0.1 * gap_constraint
            
            g_loss.backward()
            g_optimizer.step()
            g_loss_epoch += g_loss.item()
        
        # 打印每轮结果
        print(f"Epoch {epoch + 1}/{num_epochs}, D Loss: {d_loss_epoch:.4f}, G Loss: {g_loss_epoch:.4f}")
    
    print("Training complete.")

# 主函数
def main():
    # 参数设置
    csv_path = "D:\Project\ThesisProject\AutoML\data\SMILES_donors_and_NFAs.csv" 
    save_path = "D:\Project\ThesisProject\AutoML\Project2\GANProject\GANModel.pyh" 
    smiles_column = "smiles"
    homo_column = "homo"
    lumo_column = "lumo"
    mark_column = "mark"
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

    # 加载数据
    # real_data_loader = load_data(csv_path, batch_size=batch_size)
    fingerprint_res, features_res,lables = load_data(csv_path, smiles_column, homo_column, lumo_column, mark_column)
    # 初始化模型
    generator = Generator(noise_dim, fingerprint_dim, feature_dim, hidden_dim, dropout)
    discriminator = Discriminator(fingerprint_dim, feature_dim, hidden_dim, dropout)

    # 优化器
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    # 训练和评估
    # train_and_evaluate(generator, discriminator, g_optimizer, d_optimizer,
    #                    fingerprint_res,features_res,lables, noise_dim, num_epochs, device)
    train_and_evaluate(generator, discriminator, g_optimizer, d_optimizer, 
                    fingerprint_res, features_res,lables, batch_size, noise_dim, num_epochs, device)
    
    
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
