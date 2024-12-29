import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
import keras_tuner as kt
import numpy as np
from keras import layers, models

import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

from rdkit.Chem import AllChem
from rdkit import Chem

import networkx as nx
from sklearn.model_selection import train_test_split
"""
已实现：
生产出节点特征+邻接矩阵+键类型
生成对抗网络生成节点特征+邻接矩阵，其中GCN网络根据节点特征+邻接矩阵生成键类型，共同成长
    共同成长：
        打分机制：
            计算损失变化率（越小越好），
            计算当前损失接近目标损失的程度（越接近目标损失，越接近0），
            计算损失稳定性（小的波动更好）。
            通过加权平均的方式得到综合评分，分数越小越好
            外部再加上是否符合物理化学规律打分，合成后，分值越小越好
自动化超参数调优，部分功能json形式配置
图转smiles+smiles转图

"""

def graph_to_smiles(node_features, adjacency_matrix, bond_types,atom_to_idx):
    # node_features: 原子特征（独热编码）
    # adjacency_matrix: 邻接矩阵，表示原子之间的连接关系
    # bond_types: 键类型（如单键、双键等）
    try:
        # 1. 通过节点特征确定每个原子的类型（例如碳、氮、氧）
        atom_types = np.argmax(node_features, axis=1) 
        
        # 2. 使用邻接矩阵构建分子图
        mol = Chem.RWMol()  # 创建一个空分子
        
        # 3. 添加原子到分子中
        idx_to_atom = {idx: atom for atom, idx in atom_to_idx.items()}
        atom_map = {}
        for i, atom_idx in enumerate(atom_types):
            atom_type = idx_to_atom[atom_idx]
            atom = Chem.Atom(atom_type)  # 创建原子对象
            atom_temp = mol.AddAtom(atom)  # 将原子添加到分子中
            atom_map[i] = atom_temp  # 记录原子索引
        
        # 4. 添加化学键
        for i in range(len(atom_types)):
            for j in range(i + 1, len(atom_types)):
                if adjacency_matrix[i, j] == 1:  # 如果i和j之间有键
                    bond_type = int(Chem.BondType.SINGLE)  # 默认单键
                    for bond_arr in bond_types:
                        if(bond_arr[0]==i and bond_arr[1]==j):
                            bond_type = bond_arr[2]
                    bond_temp = Chem.BondType.values.get(bond_type)
                    if(bond_temp is not None):
                        bond_type = bond_temp
                    
                    mol.AddBond(atom_map[i], atom_map[j], bond_type)  # 添加键到分子中
        Chem.SanitizeMol(mol)
        # 5. 生成SMILES
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except Exception as e:
        print("graph_to_smiles")
        print(e)
        return None

# def graph_to_smiles(node_features, adj_matrix, bond_types,atom_to_idx,is_check=True):
#     try:
#         # 创建一个空的分子对象
#         mol = Chem.RWMol()
        
#         # 创建原子类型到索引的反向映射
#         idx_to_atom = {idx: atom for atom, idx in atom_to_idx.items()}
        
#         # 从独热编码的节点特征中恢复原子类型
#         atom_types = []
#         for feature in node_features:
#             atom_idx = np.argmax(feature)  # 找到最大值的位置
#             atom_type = idx_to_atom[atom_idx]  # 映射到原子类型
#             atom_types.append(atom_type)
            
#         num_nodes = node_features.shape[0]

#         # 添加原子
#         atom_idx_map_rdkit = {}
#         for idx, atom_type in enumerate(atom_types):
#             rdkit_atom = Chem.Atom(atom_type)
#             atom_idx_map_rdkit[idx] = mol.AddAtom(rdkit_atom)
        
#         # 添加键（边）根据adj_matrix和bond_types
#         for i in range(num_nodes):
#             for j in range(i + 1, num_nodes):  # 确保只检查一次每对原子
#                 if adj_matrix[i, j] == 1:  # 如果i和j之间有边（即有连接）
#                     # 找到键类型
#                     bond_type = None
#                     for bond in bond_types:
#                         if(bond!=None):
#                             if (bond[0] == i and bond[1] == j) or (bond[0] == j and bond[1] == i):
#                                 bond_type = bond[2]
#                                 break
#                     if(bond_type!=None and (bond_type!=-1 or Chem.BondType.get(bond_type)==None)):
#                         mol.AddBond(atom_idx[i], atom_idx[j], Chem.BondType.get(bond_type))

#         # 使用rdkit将分子对象转化为SMILES
#         smiles = Chem.MolToSmiles(mol)
#         if(is_check):
#             temp_mol = Chem.MolFromSmiles(smiles)
#             if(temp_mol is None):
#                 return None
#             else:
#                 return smiles
#         return smiles
#     except Exception as e:
#         print('graph_to_smiles:')
#         print(e)
#         return None
    

def smiles_to_graph(smiles, other,atom_to_idx,atom_types):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    G = nx.Graph()
    
    # 1. 获取节点特征（原子序数）
    # node_features = []
    # for atom in mol.GetAtoms():
    #     # 每个节点的特征：原子序数（可以替换为其他特征，如原子类型、离子化状态等）
    #     node_features.append(atom.GetAtomicNum())

    # node_features = np.array(node_features).reshape(-1, 1)  # (num_atoms, 1)atom_to_idx
    # idx_to_atom = {idx: atom for atom, idx in atom_to_idx.items()}
    # 获取原子和键的独热编码
    atom_features = []
    for atom in mol.GetAtoms():
        atom_num = atom.GetAtomicNum()
        idx = atom_to_idx.get(atom_num)
        if(idx is not None):
            one_hot = np.zeros(len(atom_types))
            one_hot[idx] = 1
            atom_features.append(one_hot)
    
    node_features = np.array(atom_features)
    
    # 其他性质特征
    other_features = np.array(other)
    
    num_edges = len(mol.GetBonds())
    # 边特征矩阵，每条边有一个特征（比如键类型）
    adj_matrix = np.zeros((len(node_features), len(node_features)))
    bond_matrix = np.zeros((num_edges, 3), dtype=int)  # 只有1个特征：键类型
    # edge_index = []
    for idx,bond in enumerate(mol.GetBonds()):
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        adj_matrix[start_idx, end_idx] = 1
        adj_matrix[end_idx, start_idx] = 1  # 无向图，矩阵对称
        # 更新边特征矩阵：根据键类型填充
        bond_type = int(bond.GetBondType())  # 获取键类型
        bond_matrix[idx] = (start_idx,end_idx,bond_type)
    return [node_features, adj_matrix,bond_matrix,other_features]


# def onehot_to_smiles(node_features, adjacency_matrix,bond_types,atom_to_idx):
#     """
#     将独热编码的节点特征、邻接矩阵和键类型转换为SMILES字符串
#     :param node_features: 节点特征的独热编码数组 (num_atoms x num_features)
#     :param adjacency_matrix: 邻接矩阵 (num_atoms x num_atoms)
#     :param bond_types: 键类型字典 { (i, j): bond_type }
#     :return: 对应的 SMILES 字符串
#     """
#     # 获取原子数量
#     num_atoms = len(node_features)
    
#     # 通过独热编码找到每个原子对应的元素
#     atoms = [atom_to_idx[np.argmax(node)] for node in node_features]
    
#     # 创建 RDKit 分子对象
#     mol = Chem.RWMol()
    
#     # 向分子中添加原子
#     atom_indices = []
#     for atom in atoms:
#         atom_idx = mol.AddAtom(Chem.Atom(atom))
#         atom_indices.append(atom_idx)
    
#     # 遍历邻接矩阵并添加键
#     for i in range(num_atoms):
#         for j in range(i + 1, num_atoms):
#             if adjacency_matrix[i][j] == 1:  # 如果有键
#                 bond_type = bond_types.get((i, j), Chem.BondType.SINGLE)  # 默认单键
#                 mol.AddBond(atom_indices[i], atom_indices[j], bond_type)
    
#     # 转换为 SMILES 字符串
#     smiles = Chem.MolToSmiles(mol)
#     return smiles


# def smiles_to_graph(smiles,atom_to_idx):
#     """
#     将SMILES字符串转换为节点特征、邻接矩阵和键类型
#     :param smiles: SMILES 字符串
#     :return: 节点特征、邻接矩阵、键类型
#     """

#     mol = Chem.MolFromSmiles(smiles)
    
#     # 获取原子信息
#     atoms = mol.GetAtoms()
    
#     # 生成节点特征（独热编码）
#     node_features = []
#     for atom in atoms:
#         element = atom.GetSymbol()
#         onehot = [0] * len(atom_to_idx)
#         onehot[atom_to_idx[element]] = 1
#         node_features.append(onehot)
    
#     # 获取邻接矩阵
#     num_atoms = len(atoms)
#     adjacency_matrix = np.zeros((num_atoms, num_atoms))
#     bond_types = {}
    
#     for bond in mol.GetBonds():
#         i = bond.GetBeginAtomIdx()
#         j = bond.GetEndAtomIdx()
#         adjacency_matrix[i][j] = 1
#         adjacency_matrix[j][i] = 1
#         # 更新边特征矩阵：根据键类型填充
#         bond_type = int(bond.GetBondType())  # 获取键类型
#         bond_matrix[idx] = (start_idx,end_idx,bond_type)
#         bond_types[(i, j)] = bond.GetBondTypeAsDouble()
#         bond_types[(j, i)] = bond.GetBondTypeAsDouble()
    
#     # 返回图的表示
#     return np.array(node_features), adjacency_matrix, bond_types


# Data loading function for Keras
def load_data(csv_file):
    df = pd.read_csv(csv_file, encoding='utf-8')

    smiles = df['SMILES_str'].to_numpy()
    # other = df[['pce','e_lumo_alpha', 'e_gap_alpha', 'e_homo_alpha', 'jsc', 'voc', 'mass']].to_numpy()
    other = df[['e_lumo_alpha', 'e_gap_alpha', 'e_homo_alpha', 'mass']].to_numpy()

    other = np.vectorize(lambda x: float(x) if x != '-' else 0)(other)
    scaler = StandardScaler()
    normalized_other = scaler.fit_transform(other)
    
    # 原子类型
    # 用于存储所有独特的原子类型，会自动去重
    atom_types = set()  

    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        if mol:
            for atom in mol.GetAtoms():
                atom_types.add(atom.GetAtomicNum())  # 获取原子序数
                
    # 计算类别最大数量
    num_classes = len(atom_types)
    
    # 创建原子类型到索引的映射
    atom_to_idx = {atom: idx for idx, atom in enumerate(atom_types)}
    idx_to_atom = {idx: atom for atom, idx in atom_to_idx.items()}

    max_nodes = 0
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            num_atoms = mol.GetNumAtoms()
            max_nodes = max(max_nodes, num_atoms)
    
    # Prepare graph data
    graph_data_list = []
    for s, other_item in zip(smiles, normalized_other):
        graph_data = smiles_to_graph(s, other_item,atom_to_idx,atom_types)
        if graph_data is not None:
            graph_data_list.append(graph_data)
    
    # Split data into train and test
    # train_size = int(0.9 * len(graph_data_list))
    # train_data = graph_data_list[:train_size]
    # test_data = graph_data_list[train_size:]
    # 最大节点数
    max_nodes = 0
    max_one_hot_nodes = 0
    for item1,item2,item3,item4 in graph_data_list:
        if(item2.shape[0]>max_nodes):
            max_nodes = item2.shape[0]
        if(item1.shape[1]>max_one_hot_nodes):
            max_one_hot_nodes = item1.shape[1]
    
    
    
    for i in range(len(graph_data_list)):
        node_features,adj_matrix,bond_types = graph_data_list[i][0],graph_data_list[i][1],graph_data_list[i][2]
        num_nodes0 = adj_matrix.shape[0]
        num_nodes1 = node_features.shape[1]
        if num_nodes0 < max_nodes:
            # graph_data_list[i][0] = np.pad(node_features, ((0, max_nodes - num_nodes), (0, 0)), mode='constant', constant_values=0)
            graph_data_list[i][0] =  np.pad(node_features, ((0, max_nodes - num_nodes0), (0, 0)), mode='constant', constant_values=0)
            graph_data_list[i][0] =  np.pad(graph_data_list[i][0], ((0, 0), (0, max_one_hot_nodes-num_nodes1)), mode='constant', constant_values=0)
            graph_data_list[i][1] = np.pad(adj_matrix, ((0, max_nodes - num_nodes0), (0, max_nodes - num_nodes0)), mode='constant', constant_values=0)
            
    normalized_other = np.array(normalized_other)
    
    input_dim1_shape = graph_data_list[0][0].shape
    input_dim2_shape = graph_data_list[0][1].shape
    input_dim3 = len(graph_data_list[0][2])
    item1_list = []
    item2_list = []
    item3_list = []
    item4_list = []

    for item in graph_data_list:

        item1_list.append(item[0])
        item2_list.append(item[1])
        item3_list.append(item[2])
        # 扩展标签特征到每个节点，使用广播机制
        label_expanded = tf.tile(tf.expand_dims(item[3], 0), [32, 1])  # 形状变为 (32, 4)
        # item4_list.append(item[3])
        item4_list.append(label_expanded.numpy())
    # for i in item1_list:
    #     if(i.shape[1]!=1):
    #         print(i[0])
    # TODO 得到最多的原子类型个数，修改生成器和判别器输入输出，修改生成键类型神经网络
    # TODO （创建模型时修改形状即可）
    # TODO 节点特征（根据概率分布来生成矩阵）
    #(node_features, adj_matrix, bond_types,atom_to_idx,is_check=True):
    # len_int = 0
    # for item in graph_data_list:
    #     res = graph_to_smiles(item[0],item[1],item[2],atom_to_idx)
    #     if(res is None):
    #         len_int = len_int+1
    
    # dataset = tf.data.Dataset.from_tensor_slices((item1_list,item2_list,item3_list))
    # dataset = tf.data.Dataset.from_tensor_slices((tf.ragged.constant(item1_list), tf.ragged.constant(item2_list), tf.ragged.constant(item3_list),tf.ragged.constant(item4_list)))
    # dataset = tf.data.Dataset.from_tensor_slices((tf.ragged.constant(item1_list), tf.ragged.constant(item2_list),tf.ragged.constant(item4_list)))
    # dataset = tf.data.Dataset.from_tensor_slices((tf.ragged.constant(item1_list), tf.ragged.constant(item2_list), tf.ragged.constant(item3_list)))
    dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(item1_list), tf.convert_to_tensor(item2_list),tf.convert_to_tensor(item4_list)))
    # res = {
    #     "dataset":dataset,
    #     "input_dim1_shape":input_dim1_shape,
    #     "input_dim2_shape":input_dim2_shape,
    #     "input_dim3":input_dim3,
    #     "num_classes":num_classes,
    #     "atom_types":atom_types,
    #     "atom_to_idx":atom_to_idx,
    # }
    
    return dataset, input_dim1_shape,input_dim2_shape,input_dim3,num_classes,atom_types,atom_to_idx

class GAN(keras.Model):
    def __init__(self, generator, discriminator, gcn_bond_types_model,input_dim1_shape, input_dim2_shape,input_dim3,atom_to_idx):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.gcn_bond_types_model = gcn_bond_types_model
        self.input_dim1_shape = input_dim1_shape
        self.input_dim2_shape = input_dim2_shape
        # self.input_dim3_shape = input_dim3_shape
        self.input_dim1 = input_dim1_shape[1]
        self.input_dim2 = input_dim2_shape[1]
        self.input_dim3 = input_dim3
        self.atom_to_idx = atom_to_idx
        
        
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.gcn_bond_types_loss_tracker = keras.metrics.Mean(name="bond_type_gcn_loss")
        
        self.gen_last_loss = None
        self.disc_last_loss = None
        self.gcn_bond_types_last_loss = None
        
        # 历史分子指纹
        self.fingerprint_list = []

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, g_optimizer, d_optimizer,gcn_bond_types_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.gcn_bond_types_optimizer = gcn_bond_types_optimizer
        self.loss_fn = loss_fn
    
    def call(self,inputs):
        return self.generator(inputs)
    
    def smiles_to_fp(self,smiles):
        try:
            
            # 1. 将 SMILES 转换为分子对象
            mol = Chem.MolFromSmiles(smiles)
            Chem.SanitizeMol(mol)
            # 2. 计算分子指纹（这里使用的是 ECFP6，常用的环系指纹）
            # fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            fingerprint = AllChem.MorganGenerator(mol, radius=2, nBits=1024)
            # 3. 将指纹转换为 numpy 数组
            fingerprint_array = np.array(fingerprint)
            return fingerprint_array
        except:
            return np.array([])

    # region 打分机制
    def physical_score(self,node_features, adjacency_matrix,bond_types):
        """
        简单计算是否符合物理规律
        :param node_features: 节点特征
        :param adjacency_matrix: 邻接矩阵
        :return: 评分（0和1，0：不符合 1：符合）
        """
        shape = tf.shape(node_features).numpy()
        batch_size = int(shape[0])
        node_np = node_features.numpy()
        am_np = adjacency_matrix.numpy()
        bt_np = bond_types.numpy()
        res = 0
        fp_list = []


        if(len(shape)==3):
            for node,am,bt in zip(node_np,am_np,bt_np):
                smiles = graph_to_smiles(node,am,bt,self.atom_to_idx)
                ex_physical = EXPhysicalScore(smiles)
                if(smiles==None):
                    res = res+1+ex_physical.total_num
                    fp_list.append(np.array([]))
                else:
                    ex_physical_score = ex_physical.cal_score()
                    fp = self.smiles_to_fp(smiles)
                    fp_list.append(fp)
                    # 检查当前生成的分子是否重复
                    duplicate_penalty_score = self.is_duplicate(fp)
                    res = res+ex_physical_score+duplicate_penalty_score

            return res/batch_size,fp_list
        else:
            if(smiles==None):
                return 1,np.array([])
            else:
                return 0,self.smiles_to_fp(smiles)
            
    def is_duplicate(self,current_fingerprint, threshold=0.9):
        """
        判断当前分子是否与历史分子重复,
        计算当前分子的指纹与历史分子指纹的相似度（使用余弦距离），如果相似度大于设定的阈值（例如90%），则认为是重复分子
        :param current_fingerprint: 当前分子的指纹（如1024位）
        :param threshold: 相似度阈值，默认0.9（即90%的相似度认为是重复）
        :return: 如果重复返回True，否则返回False
        """
        try:
            temp_list = []
            for item in self.fingerprint_list:
                if(item.size!=0):
                    temp_list.append(item)
            
            if(len(temp_list)==0):
                return 1
            elif(current_fingerprint.size==0):
                return 1
            else:
                distances = pairwise_distances([current_fingerprint], temp_list, metric='cosine')
                min_distance = np.min(distances)
                if(min_distance < (1 - threshold)):# 如果相似度大于阈值，则认为是重复
                    return 1
                else:
                    return 0
        except:
            return 1
    
    def score_loss(self, current_loss, prev_loss,target_loss=0, epsilon=1e-6):
        """
        根据当前损失和历史损失计算评分，越接近0越好
        :param current_loss: 当前损失
        :param prev_loss: 上一次的损失
        :param target_loss: 目标损失值，默认为0（理想情况下）
        :param epsilon: 避免除零错误
        :return: 评分（0-1之间）
        """
        if prev_loss is None:
            # 如果没有历史损失，初始化评分为 current_loss
            return current_loss
        
        # 计算损失变化率（越小越好）
        loss_diff = abs(current_loss - prev_loss)
        rate_score = min(1, loss_diff / (prev_loss + epsilon))  # 越小损失越小，分数越接近0

        # 计算当前损失接近目标损失的程度（越接近目标损失，越接近0）
        target_score = min(1, abs(current_loss - target_loss) / (target_loss + epsilon))  # 越接近目标损失，分数越接近0
        
        # 计算损失稳定性（小的波动更好）
        stability_score = min(1, loss_diff / (prev_loss + epsilon))  # 越稳定（波动小），越接近0


        # 通过加权平均的方式得到综合评分，分数越小越好
        total_score = (rate_score + target_score + stability_score) / 3
        return total_score
# endregion
    def train_step(self, real_data):
        # for g_data,d_data,bt_data,other_data in zip(real_data[0], real_data[1] ,real_data[2],real_data[3]):
        shape = tf.shape(real_data[0]).numpy()
        batch_size = int(shape[0])
        # 随机生成噪声向量,从标准正态分布中生成一个随机向量，表示潜在空间的样本。
        noise1 = tf.random.normal(shape=(batch_size,self.input_dim2,self.input_dim1))  # 噪声维度
        noise2 = tf.random.normal(shape=(batch_size,self.input_dim2,self.input_dim2))  # 噪声维度
        generated_data = self.generator([noise1,noise2,real_data[2]], training=True)

        # 1. 训练判别器
        with tf.GradientTape() as tape_d:
            # 判别器评估真实分子指纹和生成的分子指纹
            #shape[0] = [32,128] vs. shape[1] = [33,3] [Op:ConcatV2] name: concat
            real_output = self.discriminator(real_data, training=True)
            fake_output = self.discriminator([generated_data[0],generated_data[1],real_data[2]], training=True)
            # print(test.numpy())
            real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            d_loss = real_loss + fake_loss
        #计算梯度，并应用他们
        grads_d = tape_d.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads_d, self.discriminator.trainable_variables))
        # 更新判别器损失指标
        self.disc_loss_tracker.update_state(d_loss) 

        # 2. 训练生成器
        with tf.GradientTape() as tape_g:
            generated_data = self.generator([noise1,noise2,real_data[2]], training=True)
            fake_output = self.discriminator([generated_data[0],generated_data[1],real_data[2]], training=True)
            
            # 生成器的损失：让生成的指纹尽可能“真实”
            g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
        #计算梯度，并应用他们
        grads_g = tape_g.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads_g, self.generator.trainable_variables))
        # 更新生成器损失指标
        self.gen_loss_tracker.update_state(g_loss)  
        with tf.GradientTape() as tape_disc_bond_type:
            # gcn_bond_type_data = self.gcn_bond_types_model([g_data,d_data], training=True)
            # gcn_bond_type_data = self.gcn_bond_types_model([g_data,d_data], training=True)
            gcn_bond_type_data = self.gcn_bond_types_model(generated_data, training=True)
            gcn_loss = self.loss_fn(tf.ones_like(gcn_bond_type_data), gcn_bond_type_data)
            
            
        #计算梯度，并应用他们
        grads_gcn_bond_types = tape_disc_bond_type.gradient(gcn_loss, self.gcn_bond_types_model.trainable_variables)
        self.gcn_bond_types_optimizer.apply_gradients(zip(grads_gcn_bond_types, self.gcn_bond_types_model.trainable_variables))
        # 更新判别器损失指标
        self.gcn_bond_types_loss_tracker.update_state(gcn_loss) 
        # 返回每个epoch的损失值
        gen_loss_float = float(self.gen_loss_tracker.result().numpy())
        disc_loss_float = float(self.disc_loss_tracker.result().numpy())
    
        gcn_bond_types_loss_float = float(self.gcn_bond_types_loss_tracker.result().numpy())
        
        # 打分
        physical_score,fp_list = self.physical_score(generated_data[0],generated_data[1],gcn_bond_type_data)

        
        gen_score = self.score_loss(gen_loss_float,self.gen_last_loss)
        dics_score = self.score_loss(disc_loss_float,self.disc_last_loss)
        bond_types_score = self.score_loss(gcn_bond_types_loss_float,self.gcn_bond_types_last_loss)

        total_score = gen_score+dics_score+physical_score+bond_types_score

        # 保存当前的损失值，在下一次用
        self.gen_last_loss = gen_loss_float
        self.disc_last_loss = disc_loss_float
        
        self.gcn_bond_types_last_loss = gcn_bond_types_loss_float
        
        self.fingerprint_list = self.fingerprint_list + fp_list
        return {
            'd_loss_score': self.gen_loss_tracker.result(),
            'g_loss_score': self.disc_loss_tracker.result(),
            'total_score':total_score,
            'gcn_bond_types_loss_score': self.gcn_bond_types_loss_tracker.result(),
            'gen_score':gen_score,
            'dics_score':dics_score,
            'physical_score':physical_score,
        }
        
class EXPhysicalScore():
    def __init__(self,smiles):
        super(EXPhysicalScore, self).__init__()
        self.smiles = smiles
        self.total_num = 5
        
    def check_connectivity(self,mol):
        try:
            # 判断分子是否连通
            fragments = Chem.GetMolFrags(mol, asMols=True)
            return 0 if len(fragments) == 1 else 1  # 连通返回0，非连通返回1
        except:
            return 1

    def check_valency(self,mol):
        try:
            # 判断元素的价数是否合法
            for atom in mol.GetAtoms():
                if atom.GetDegree() > atom.GetMaxDegree():
                    return 1  # 价数不合法返回1
            return 0  # 价数合法返回0
        except:
            return 1

    def check_ring_legality(self,mol):
        try:
            # 检查环的合法性
            for ring in mol.GetRingInfo().AtomRings():
                ring_size = len(ring)
                if ring_size < 3 or ring_size > 12:  # 常见的环大小为3-12
                    return 1  # 环不合法返回1
            return 0  # 合法返回0
        except:
            return 1

    def check_geometry(self,mol):
        try:
            # 判断分子几何结构是否合理（是否存在空隙）
            mol = Chem.AddHs(mol)  # 添加氢原子
            AllChem.EmbedMolecule(mol, randomSeed=42)
            success = AllChem.MMFFOptimizeMolecule(mol)
            return 0 if success == 0 else 1  # 优化成功返回0，失败返回1
        except:
            return 1

    # def check_bond_length(self,mol):
    #     try:
    #         # 检查分子中键的长度是否合理
    #         for bond in mol.GetBonds():
    #             bond_length = bond.GetBondTypeAsDouble()
    #             if bond_length < 0.9 or bond_length > 1.5:  # 假定合理键长范围为0.9到1.5
    #                 return 1  # 键长不合理返回1
    #         return 0  # 键长合理返回0
    #     except:
    #         return 1

    def check_bond_type(self,mol):
        try:
            # 判断键类型是否合法

            for bond_type in mol.GetBonds():
                bond_temp = Chem.BondType.values.get(bond_type)
                if(bond_temp is None):
                    return 1  # 键类型不合法返回1
            return 0  # 键类型合法返回0
        except:
            return 1

    def cal_score(self):
        score = 0
        try:
            mol = Chem.MolFromSmiles(self.smiles)
            Chem.SanitizeMol(mol)
            # 逐项检查并累加分值
            score += self.check_connectivity(mol)  # 连通性检查
            score += self.check_valency(mol)  # 价数检查
            score += self.check_ring_legality(mol)  # 环的合法性检查
            score += self.check_geometry(mol)  # 几何合理性检查
            # score += self.check_bond_length(mol)  # 键长度检查
            score += self.check_bond_type(mol)  # 键类型合法性检查

            return score  # 返回总分
        except:
            return self.total_num

# 定义GCN模型
class GCNModel(tf.keras.Model):
    def __init__(self, num_node_features, num_classes):
        super(GCNModel, self).__init__()
        self.gcn1 = spektral.layers.GCNConv(num_node_features, activation='relu')  # 第一层GCN，输出16维特征
        self.gcn2 = spektral.layers.GCNConv(num_classes, activation='softmax')  # 第二层GCN，用于输出分类结果

    def call(self, inputs):
        node_features, adjacency_matrix = inputs
        x = self.gcn1([node_features, adjacency_matrix])  # 图卷积操作
        x = self.gcn2([x, adjacency_matrix])  # 第二层图卷积
        return x

import spektral
from spektral.data import Graph
import keras.backend as K
# 使用Spektral库的GraphConv层
class HyperGAN(kt.HyperModel):

    def __init__(self,config,input_dim1_shape,input_dim2_shape,input_dim3,num_classes,atom_types,atom_to_idx):
        super(HyperGAN, self).__init__()
        self.input_dim1_shape = input_dim1_shape
        self.input_dim2_shape = input_dim2_shape
        # self.input_dim3_shape = input_dim3_shape
        self.input_dim1 = input_dim1_shape[1]
        self.input_dim2 = input_dim2_shape[1]
        self.input_dim3 = input_dim3
        self.num_classes = num_classes
        self.atom_types = atom_types
        self.atom_to_idx = atom_to_idx
        
        self.gan_model = None

        self.config = config

    def make_generator_model(self, hp):

        """
        构建一个生成器模型，使用图卷积层（GraphConv）来处理图数据。
        """
        # 输入层：节点特征和邻接矩阵
        node_features_input = layers.Input(shape=(self.input_dim2,self.input_dim1))
        adjacency_input = layers.Input(shape=(self.input_dim2,self.input_dim2))  # 邻接矩阵（形状通常为(num_nodes, num_nodes)）
        other_input = layers.Input(shape=(self.input_dim2,4))
        # by_input = layers.Input(shape=(3,))  # 键类型
        
        # other_input = tf.tile(tf.expand_dims(other_input, 0), [self.input_dim2, 1])  # 形状变为 (32, 4)
        
        
        x = layers.Concatenate()([node_features_input, other_input])
        # 超参数搜索：图卷积层的隐藏维度
        hidden_dim =get_hp_value(hp,self.config,"hidden_dim_gen")
        hidden_dim =get_hp_value(hp,self.config,"hidden_dim_gen")

        hidden_layer_num =get_hp_value(hp,self.config,"hidden_layer_num")
        drop_rate = get_hp_value(hp,self.config,"dropout_rate_gen")
        batch_size = get_hp_value(hp,self.config,"batch_size")
        # 图卷积层，聚合节点特征
        x = spektral.layers.GCSConv(hidden_dim, activation='relu')([x, adjacency_input])
        
        for i in range(0,hidden_layer_num-1):
            x = spektral.layers.GCSConv(hidden_dim, activation='relu')([x,adjacency_input])
            x = layers.Dropout(drop_rate)(x)

        # x = spektral.layers.GlobalAttentionPool(hidden_dim)(x)
        # 输出层：生成节点特征和邻接矩阵
        # 生成的节点特征需要规定原子类型的类别
        output_node_features = layers.Dense(self.num_classes, activation='tanh')(x)  # 生成节点特征
        # output_node_features = layers.Dense(self.input_dim1, activation='ReLU')(x)  # 生成节点特征
        output_adj = layers.Dense(self.input_dim2, activation='sigmoid')(x)  # 生成邻接矩阵
        # output_adj = layers.Reshape((self.input_dim2, ))(output_adj)
        
        # output_bond_types = layers.Dense(3, activation='softmax')(output_adj)  # 键类型（单键、双键、三键、芳香键）
        # 生成边缘特征：通过噪声生成边缘特征

        # edge_features = layers.Dense(self.input_dim2 * self.input_dim2 * 4)(other_input)
        # edge_features = layers.Reshape((self.input_dim2 * self.input_dim2, 4))(edge_features)  # [num_edges, edge_feature_dim]
        

        # 创建模型
        model = models.Model(inputs=[node_features_input, adjacency_input,other_input], outputs=[output_node_features, output_adj])
        return model

    def make_discriminator_model(self,hp):
        """
        构建一个判别器模型，使用图卷积层来处理图数据。
        使用sigmoid来代表是否是真实数据，并使用binary_crossentropy（使用二分类交叉熵损失函数）来计算损失
        """
        # 输入层：节点特征和邻接矩阵
        node_features_input = layers.Input(shape=(self.input_dim2,self.input_dim1))
        adjacency_input = layers.Input(shape=(self.input_dim2,self.input_dim2 ))  # 邻接矩阵
        other_input = layers.Input(shape=(self.input_dim2,4))
        # bond_types_input = layers.Input(shape=(3, ))  # 键类型（单键、双键、三键、芳香键）


        # # 标签嵌入(离散的才需要，输入的homo等是连续的所以不需要嵌入层)
        # label_embedding = layers.Embedding(4, 50)(other_input)
        # label_embedding = layers.Flatten()(label_embedding)
        
        # 拼接噪声与标签嵌入
        x = layers.Concatenate()([node_features_input, other_input])

        # 超参数搜索：图卷积层的隐藏维度
        hidden_dim =get_hp_value(hp,self.config,"hidden_dim_gen")

        hidden_layer_num =get_hp_value(hp,self.config,"hidden_layer_num")
        drop_rate = get_hp_value(hp,self.config,"dropout_rate_gen")
        # 图卷积层，聚合节点特征
        x = spektral.layers.GCSConv(hidden_dim, activation='relu')([x, adjacency_input])
        
        for i in range(0,hidden_layer_num-1):
            x = spektral.layers.GCSConv(hidden_dim, activation='relu')([x,adjacency_input])
            x = layers.Dropout(drop_rate)(x)

        # 输出层：预测输入图数据是否为真实数据
        output = layers.Dense(1, activation='sigmoid')(x)  # 输出一个标量，0表示生成样本，1表示真实样本

        # 创建模型
        model = models.Model(inputs=[node_features_input, adjacency_input,other_input], outputs=output)
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # 使用二分类交叉熵损失函数

        return model
    
    def make_gcn_bond_types_model(self, hp):

        """
        构建一个GCN模型，生成键类型
        """
        # 输入层：节点特征和邻接矩阵
        node_features_input = layers.Input(shape=(self.input_dim2,self.input_dim1))
        adjacency_input = layers.Input(shape=(self.input_dim2,self.input_dim2 ))  # 邻接矩阵
        # other_input = layers.Input(shape=(4,))
        
        hidden_dim =get_hp_value(hp,self.config,"hidden_dim_gcn")
        hidden_layer_num =get_hp_value(hp,self.config,"hidden_layer_num")
        drop_rate = get_hp_value(hp,self.config,"dropout_rate_gcn")
        
        x = spektral.layers.GCNConv(hidden_dim, activation='relu')([node_features_input,adjacency_input]) 
        for i in range(0,hidden_layer_num-1):
            x = spektral.layers.GCNConv(hidden_dim, activation='relu')([x,adjacency_input])
            x = layers.Dropout(drop_rate)(x)

        output = layers.Dense(3, activation='sigmoid')(x) 
        model = models.Model(inputs=[node_features_input, adjacency_input], outputs=output)

        return model

    def build(self, hp):
        self.generator = self.make_generator_model(hp)
        self.discriminator = self.make_discriminator_model(hp)
        self.gcn_bond_types_model = self.make_gcn_bond_types_model(hp)

        model_gan = GAN(self.generator,self.discriminator,self.gcn_bond_types_model, self.input_dim1_shape, self.input_dim2_shape,self.input_dim3,self.atom_to_idx)
        # 生成器优化器
        self.g_optimizer = create_op(hp,self.config,"g_optimizer","g_learning_rate")
        # 判别器优化器
        self.d_optimizer =create_op(hp,self.config,"d_optimizer","d_learning_rate")
        
        self.gcn_bond_types_optimizer =create_op(hp,self.config,"g_optimizer","g_learning_rate")

        # 已经经过sigmod激活函数，不需要from_logits=True
        binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        # 已重写compile
        model_gan.compile(self.g_optimizer,self.d_optimizer,self.gcn_bond_types_optimizer,binary_crossentropy)
        self.gan_model = model_gan
        return model_gan

    def fit(self, hp, model, x,  *args, **kwargs):
        
        for key in self.config:
            _ = get_hp_value(hp,self.config,key)
        
        batch_size = get_hp_value(hp,self.config,"batch_size")
        
        x = x.batch(batch_size, drop_remainder=True)

        res = model.fit(x,batch_size=batch_size,*args,**kwargs)
        # custom_score = CustomScore(res.history.g_loss_score,res.history.d_loss_score,gen_data[0],gen_data[1])
        # socre = custom_score.get_mult_socre()
        return res.history
    

def create_op(hp,config,optimizer_key="optimizer",learning_rate_key="learning_rate"):
    
    disc_optimizer =get_hp_value(hp,config,optimizer_key) 
    _learning_rate = get_hp_value(hp,config,learning_rate_key)
    optimizer = None
    if disc_optimizer.lower() == 'adam':
        optimizer=tf.keras.optimizers.Adam(learning_rate=_learning_rate)
    elif disc_optimizer.lower() == 'sgd':
        optimizer=tf.keras.optimizers.SGD(learning_rate=_learning_rate)
    elif disc_optimizer.lower() == 'adagrad':
        optimizer=tf.keras.optimizers.Adagrad(learning_rate=_learning_rate)
    elif disc_optimizer.lower() == 'adadelta':
        optimizer=tf.keras.optimizers.Adadelta(learning_rate=_learning_rate)
    elif disc_optimizer.lower() == 'rmsprop':
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=_learning_rate)
    else:
        optimizer=tf.keras.optimizers.Nadam(learning_rate=_learning_rate)
    return optimizer

def get_hp_value(hp,config,key):
    if(type(config[key])==int or type(config[key])==float):
        return config[key]
    _type = config[key]["_type"]
    m = getattr(hp,config[key]["_type"])
    if(_type=='Choice'):
        if(hp.values.get(key) is not None):
            return hp.get(key)
        else:
            res = m(key,values=config[key]["_value"])
            return res
    elif(_type=='Float' or _type=='Int'):
        if(hp.values.get(key) is not None):
            return hp.get(key)
        else:
            res = m(key,min_value=config[key]["_value"][0],max_value=config[key]["_value"][1],sampling=config[key].get("_mode"),step=config[key].get("_step"))
            return res
        # res = m(key,min_value=config[key]["_value"][0],max_value=config[key]["_value"][1],sampling=config[key].get("_mode"),step=config[key].get("_step"))
        # return res
    elif(_type=='Boolean'):
        if(hp.get(key) is not None):
            return hp.get(key)
        else:
            res = m(key)
            return res
    else:
        return None

def main():
    
    csv_file = 'D:\Project\ThesisProject\AutoML\data\moldata_part_test.csv'
    dataset, input_dim1_shape,input_dim2_shape,input_dim3,num_classes,atom_types,atom_to_idx  = load_data(csv_file)
    
    
    
    
    search_space = {
        "optimizer":{
            "_type":"Choice",
            "_value":["adam","sgd","adagrad","adadelta","rmsprop","nadam"]   
        },
        "d_optimizer":{
            "_type":"Choice",
            "_value":["adam","sgd","adagrad","adadelta","rmsprop","nadam"]   
        },
        "g_optimizer":{
            "_type":"Choice",
            "_value":["adam","sgd","adagrad","adadelta","rmsprop","nadam"]   
        },
        "learning_rate":{
            "_type":"Float",
            "_mode":"log",
            "_value":[1e-5,1e-1]
        },
        "d_learning_rate":{
            "_type":"Float",
            "_mode":"log",
            "_value":[1e-5,1e-1]
        },
        "g_learning_rate":{
            "_type":"Float",
            "_mode":"log",
            "_value":[1e-5,1e-1]
        },
        "units":{
            "_type":"Int",
            "_mode":"linear",
            "_value":[1,5]
        },
        "hidden_dim_disc":{
            "_type":"Choice",
            "_value":[32,64,128,256]
        },
        "hidden_dim_gen":{
            "_type":"Choice",
            "_value":[32,64,128,256]
        },
        "hidden_dim_gcn":{
            "_type":"Choice",
            "_value":[32,64,128,256]
        },
        "hidden_layer":{
            "_type":"Int",
            "_mode":"log",
            "_value":[1,5]
        },
        "hidden_layer_num":{
            "_type":"Int",
            "_mode":"log",
            "_value":[1,5]
        },
        "dropout_rate_gen":{
            "_type":"Float",
            "_mode":"log",
            "_value":[0.001,0.5]
        },
        "dropout_rate_gcn":{
            "_type":"Float",
            "_mode":"log",
            "_value":[0.001,0.5]
        },
        "dropout_rate_disc":{
            "_type":"Float",
            "_mode":"log",
            "_value":[0.001,0.5]
        },
        "batch_size":{
            "_type":"Choice",
            "_value":[32,64,96,128]
            # "_value":[96]
        },
        "max_trials":50,
        "epochs":35,
    }

    
    import time
    import keras.callbacks
    import os
    now = time.strftime("%Y%m%d%H%M%S", time.localtime())
    tf.config.experimental_run_functions_eagerly(True)
    LOG_DIR = r'AutoML\models\gan_model\tuner\gan_keras_model_'+f"{now}" 
    # LOG_DIR = r'AutoML\models\gan_model\tuner\gan_keras_model_20241224192413'
    # input_dim1_shape = (32,1)
    tuner = kt.BayesianOptimization(
        hypermodel=HyperGAN(search_space,input_dim1_shape,input_dim2_shape,input_dim3,num_classes,atom_types,atom_to_idx),
        objective=kt.Objective("total_score", "min"),
        max_trials=search_space["max_trials"],
        overwrite=True,
        
        directory=LOG_DIR,
        project_name="custom_eval",

    )
    
    checkpoint_DIR = r'AutoML\models\gan_model\ModelCheckPoint\model_checkPoint'+now
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_DIR,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
    
    work_path = os.getcwd()
    tb_log_path = os.path.join(work_path,tuner.directory)
    # TODO 1.callbacks回调，具体查看https://keras.io/guides/writing_your_own_callbacks/
    # TODO 2.tensorboard
    # TODO 4.将参数改为动态

    tuner.search(
        x = dataset, 
        epochs = search_space["epochs"],
        callbacks=[keras.callbacks.TensorBoard(log_dir=tb_log_path, histogram_freq=1,write_images=True)]
        )

    tuner.results_summary()
    best_hps = tuner.get_best_hyperparameters()[0]
    print(best_hps.values)
    
    # checkpoint=tuner_utils.SaveBestEpoch(objective=self.oracle.objective, filepath=self._get_checkpoint_fname(trial.trial_id))
    # checkpoint.set_model(yourBuiltmodel)
    # checkpoint._save_model() 
    
    # https://stackoverflow.com/questions/74844622/unsuccessful-tensorslicereader-constructor-failed-to-find-any-matching-files-fo
    # 保存权重的方式
    # encoder_weights=encoder.load_weights(".model_save/cp.ckpt")
    # #Loading weights
    # loaded_weights=encoder.load_weights(encoder_prefix)
    
    # GLOBAL_MODEL.save
    # TODO 加入更多优化器
    # TODO 完善打分机制，加入重复分子判定
    # TODO 7次足矣
    best_model = tuner.get_best_models()[0]
    # shape = [None, 8]
    # placeholder_tensor1 = tf.TensorSpec(shape, dtype=tf.float32)
    # shape = [None, 32]
    # placeholder_tensor2 = tf.TensorSpec(shape, dtype=tf.float32)
    # noise1 = tf.random.normal(shape=(96,self.input_dim2,self.input_dim1))  # 噪声维度
    # noise2 = tf.random.normal(shape=(96,self.input_dim2,self.input_dim2))  # 噪声维度
    best_model.build([(None,input_dim1_shape[1]),(None,input_dim2_shape[1])])  # 假设 X_train.shape[1] 是输入特征数
    best_model.summary()
    best_model.generator.save(r'D:\Project\ThesisProject\AutoML\models\gan_model\best_model\gan_model_'+now+r'\generator')
    best_model.gcn_bond_types_model.save(r'D:\Project\ThesisProject\AutoML\models\gan_model\best_model\gan_model_'+now+r'\gcn_bond_types_model')
    # best_model.generator.save(r'D:\Project\ThesisProject\AutoML\models\gan_model\best_model\gan_model_20241224192413\generator')
    # best_model.gcn_bond_types_model.save(r'D:\Project\ThesisProject\AutoML\models\gan_model\best_model\gan_model_20241224192413\gcn_bond_types_model')
    
    # tensorboard --logdir D:\Project\ThesisProject\AutoML\models\gan_model\tuner\gan_keras_model_20241224192413
if __name__ == '__main__':

    main()
