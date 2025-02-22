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

import spektral
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from keras import ops
from rdkit import RDLogger
# from SMILES_vocab import get_vocab
from SMILES_vocab_test import get_vocab


# 没价模型约束，生成的smiles经常带有没有意义的1（环）
# 屏蔽所有RDKit的日志信息
RDLogger.DisableLog('rdApp.*')
def graph_to_molecule(graph):
    try:
        node_features = graph[0]  # 形状：[num_nodes, feature_dim]
        edge_features = graph[1]  # 形状：[num_edges, feature_dim]
        edge_src = graph[2]  # 形状：[num_edges]
        edge_dst = graph[3]  # 形状：[num_edges]
        
        threshold = node_features.shape[1]

        edge_src = tf.where(edge_src >= threshold, tf.zeros_like(edge_src), edge_src)
        edge_dst = tf.where(edge_dst >= threshold, tf.zeros_like(edge_dst), edge_dst)

        node_features = np.argmax(node_features, axis=1)
        node_types = list(node_features)
        
        edge_features = np.argmax(edge_features, axis=1)
        edge_features_list = list(edge_features)
        
        # 创建一个空的分子对象
        mol = Chem.RWMol()  # 创建一个可编辑的分子
        atom_idx_to_names = {idx+1:atom for idx, atom in enumerate(ATOM_NAMES_TO_IDX)}
        bond_idx_to_names = {idx+1:atom for idx, atom in enumerate(BOND_NAMES_TO_IDX)}

        # 添加节点（原子）到mol对象
        atom_map = {}
        for i,item in enumerate(node_types):
            if(item!=0):
                node_type = atom_idx_to_names[item]
                atom = Chem.Atom(node_type)  # 假设节点类型是原子类型的整数编码
                atom_idx = mol.AddAtom(atom)
                atom_map[i] = atom_idx  # 保存原子索引的映射
            else:
                atom_map[i] = 0

        # 添加边（化学键）到mol对象
        for src, dst, edge_feature_index in zip(edge_src, edge_dst, edge_features_list):
            if(edge_feature_index!=0):
                src_num = src.numpy()
                dst_num = dst.numpy()
                bond_type_num = bond_idx_to_names[edge_feature_index]
                # bond_type = getattr(Chem.rdchem.BondType, bond_type_str,Chem.BondType.SINGLE)
                bond_type = Chem.BondType(bond_type_num)
                if mol.GetBondBetweenAtoms(atom_map[src_num], atom_map[dst_num]) is None and atom_map[src_num] != atom_map[dst_num]:
                    mol.AddBond(atom_map[src_num], atom_map[dst_num], bond_type)
        global ATOM_VALENCE
        for atom in mol.GetAtoms():
            if atom.GetDegree() > ATOM_VALENCE.get(atom.GetSymbol(), 0):
                return None
        flag = Chem.SanitizeMol(mol, catchErrors=True)
        if flag != Chem.SanitizeFlags.SANITIZE_NONE:
            return None
        return mol

    except Exception as e:
        return None
    
def graph_to_smiles(graph):
    try:
        mol = graph_to_molecule(graph)
        smiles = Chem.MolToSmiles(mol)
        return smiles,mol
    except:
        return None,None
    
    
atom_mapping ={}
bond_mapping = {
    "SINGLE": 0,
    0: Chem.BondType.SINGLE,
    "DOUBLE": 1,
    1: Chem.BondType.DOUBLE,
    "TRIPLE": 2,
    2: Chem.BondType.TRIPLE,
    "AROMATIC": 3,
    3: Chem.BondType.AROMATIC,
}

def convert_to_one_hot(prob_dist, N):
    # 选择最大概率的索引
    # `argmax` 会返回最大值的索引，形状为 (batch_size, N * N)
    indices = tf.argmax(prob_dist, axis=1, output_type=tf.int32)
    
    # 将这些索引转换为独热编码，形状为 (batch_size, N * N)
    one_hot_flat = tf.one_hot(indices, N * N, dtype=tf.float32)

    # 将独热编码的形状从 (batch_size, N * N) 转换为 (batch_size, N, N)
    one_hot_matrix = tf.reshape(one_hot_flat, (-1, N, N))

    return one_hot_matrix


def smiles_to_graph(smiles, other):
    atom_type_num = len(ATOM_NAMES_TO_IDX.keys())
    bond_type_num = len(BOND_NAMES_TO_IDX.keys())
    scaler = StandardScaler()
    # 使用 RDKit 解析 SMILES 字符串
    mol = Chem.MolFromSmiles(smiles)
    atom_features = []
    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        hybridization = atom.GetHybridization()
        mass = atom.GetMass()
        idx = ATOM_NAMES_TO_IDX.get(atom_symbol)
        if(idx is not None):
            one_hot = np.zeros(atom_type_num)
            one_hot[idx] = 1
            # one_hot = np.append(one_hot,normalized_charge)
            atom_features.append(one_hot)
    
    # node_features = np.array(atom_features,dtype=int)
    node_features = np.array(atom_features,dtype=float)
    
    # 获取分子中的原子数和键数
    num_nodes = len(mol.GetAtoms())
    num_edges = len(mol.GetBonds())
    
    # 初始化节点特征矩阵和边特征矩阵
    edge_src = np.zeros(num_edges, dtype=int)
    edge_dst = np.zeros(num_edges, dtype=int)
    
    edge_idx = 0
    edge_features = []
    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        # 更新边连接信息
        edge_src[edge_idx] = start_idx
        edge_dst[edge_idx] = end_idx
        
        # 获取并更新边的特征（例如，键类型）
        bond_type = int(bond.GetBondType())  # 获取键类型

        bond_idx = BOND_NAMES_TO_IDX.get(bond_type,1)
        if(idx is not None):
            one_hot = np.zeros(bond_type_num)
            one_hot[bond_idx] = 1
            edge_features.append(one_hot)
        
        edge_idx += 1
    edge_features = np.array(edge_features,dtype=int)
    if(tf.reduce_max(edge_src) <= node_features.shape[1]):
        print(edge_src.shape)
        print(edge_src)
    return [node_features, edge_features, edge_src, edge_dst,num_nodes,num_edges]


ATOM_NAMES_TO_IDX = None
BOND_NAMES_TO_IDX = None
ATOM_VALENCE = None

TOKEN_VALENCE = None

def get_outer_electrons(z):
    """
    根据原子序数计算最外层电子数。
    该方法基于电子填充顺序，并考虑s和p轨道的电子数之和。
    注意：对于某些过渡金属（如Cr、Cu），由于电子填充异常，结果可能不准确。
    """
    if z == 0:
        return 0  # 原子序数不可能为0
    
    # 生成能级填充顺序
    max_n = 7  # 覆盖到第7周期
    levels = []
    for n in range(1, max_n + 1):
        for l in range(n):
            levels.append((n, l))
    # 按(n + l, n)排序
    levels.sort(key=lambda x: (x[0] + x[1], x[0]))
    
    # 填充电子
    electrons = {}
    remaining = z
    for (n, l) in levels:
        max_e = 2 * (2 * l + 1)
        fill = min(max_e, remaining)
        electrons[(n, l)] = fill
        remaining -= fill
        if remaining == 0:
            break
    
    # 确定最大的n值
    if not electrons:
        return 0
    
    max_n_level = max(n for (n, l) in electrons if electrons[(n, l)] > 0)
    
    # 计算最外层电子数（s和p轨道）
    s_e = electrons.get((max_n_level, 0), 0)
    p_e = electrons.get((max_n_level, 1), 0)
    total = s_e + p_e
    
    return total

def get_atom_v(name):
    outer_electrons_dict = {
    'H': 1, 'He': 2,
    'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8,
    'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 8,
    'K': 1, 'Ca': 2, 'Ga': 3, 'Ge': 4, 'As': 5, 'Se': 6, 'Br': 7, 'Kr': 8,
    }
    v = outer_electrons_dict.get(name,0)
    return v

# Data loading function for Keras
def load_data(csv_file):
    df = pd.read_csv(csv_file, encoding='utf-8')

    smiles_list = df['SMILES_str'].to_numpy()

    other = df['e_gap_alpha'].to_numpy()

    other = np.vectorize(lambda x: float(x) if x != '-' else 0)(other)
    # scaler = StandardScaler()
    # normalized_other = scaler.fit_transform([other])
    scaler = MinMaxScaler(feature_range=(0, 1))

    normalized_other = scaler.fit_transform(other.reshape(-1, 1)).flatten()
    
    
    
    
    # 原子类型
    # 用于存储所有独特的原子类型，会自动去重
    atom_types = set()  
    all_bond_types = set()
    all_smiles_char_types = set()
    handle_smiles = []
    for s in smiles_list:
        for s_char in s:
            all_smiles_char_types.add(s_char)
        mol = Chem.MolFromSmiles(s)
        if mol:
            temp_smiles = Chem.MolToSmiles(mol)
            handle_smiles.append(temp_smiles)
            for atom in mol.GetAtoms():

                atom_types.add(atom.GetSymbol())
                
            for bond in mol.GetBonds():
                all_bond_types.add(int(bond.GetBondType()))
    
    smiles_list = np.array(handle_smiles)
    
    max_bond_types = len(all_bond_types)
    # 计算类别最大数量
    atom_classes_num = len(atom_types)
    
    # 创建原子类型到索引的映射
    atom_names_to_idx = {atom:idx for idx, atom in enumerate(atom_types)}
    global ATOM_NAMES_TO_IDX
    ATOM_NAMES_TO_IDX = atom_names_to_idx
    
    bond_names_to_idx = {bond:idx for idx, bond in enumerate(all_bond_types)}
    global BOND_NAMES_TO_IDX
    BOND_NAMES_TO_IDX = bond_names_to_idx
    bond_idx_to_names_list = list(bond_names_to_idx.values())
    
    
    
    global ATOM_VALENCE
    ATOM_VALENCE = {}
    for idx, atom in enumerate(atom_types):
        mols = Chem.MolFromSmiles(f'[{atom}]')
        # for atom_item in mols.GetAtoms():
        a = mols.GetAtomWithIdx(0)
        num = a.GetAtomicNum()
        v = get_outer_electrons(num)
        s = a.GetSymbol()
        v2 = get_atom_v(s)
        if(v!=v2):
            print(v2)
        ATOM_VALENCE.update({idx:v})


    
    
    
    # TODO 词汇表的解码和编码，互相转换，随后就去改模型
    # TODO 更改源
    global TOKEN_VALENCE
    vocab,token_obj = get_vocab(list(smiles_list),True)
    TOKEN_VALENCE = token_obj
    vocab_size = len(vocab)
    
    # Prepare graph data
    max_nodes = 0
    max_bond_num = 0
    max_smiles_len = 0
    graph_data_list = []
    other_data_list = []
    max_smiles_ids_len=0
    min_len=75
    
    for item in smiles_list:
        smiles_ids = token_obj.smiles_to_ids(item)
        max_smiles_ids_len = max(max_smiles_ids_len,len(smiles_ids))

    for s, other_item in zip(smiles_list, normalized_other):
        graph_data = smiles_to_graph(s, other_item)
        if graph_data is not None:
            smiles_ids = token_obj.smiles_to_ids(s,max_smiles_ids_len)
            graph_data_list.append([graph_data[0],graph_data[1],graph_data[2],graph_data[3],smiles_ids])
            max_nodes = max(graph_data[4], max_nodes)
            max_bond_num = max(graph_data[5], max_bond_num)
            max_smiles_len = max(len(s), max_smiles_len)
            max_smiles_ids_len = max(max_smiles_ids_len,len(smiles_ids))
            min_len = min(min_len,len(smiles_ids))
            other_data_list.append(other_item)

    
    for i in range(len(graph_data_list)):
        node_features, edge_features, edge_src, edge_dst,smiles= graph_data_list[i][0],graph_data_list[i][1],graph_data_list[i][2],graph_data_list[i][3],graph_data_list[i][4]
        nf_num = node_features.shape[0]
        ef_num = len(edge_features)
        smiles_len = len(smiles)

        if nf_num < max_nodes:

            graph_data_list[i][0] =  np.pad(node_features, ((0, max_nodes - nf_num), (0, 0)), mode='constant', constant_values=0)
            
        if(ef_num<max_bond_num):
            graph_data_list[i][1] = np.pad(edge_features, ((0, max_bond_num - ef_num), (0, 0)), mode='constant', constant_values=0)
            es_temp = np.pad(edge_src, ((0, max_bond_num - ef_num)), mode='constant', constant_values=0)
            # graph_data_list[i][2] = ops.one_hot(es_temp,max_bond_num)
            graph_data_list[i][2] = es_temp
            ed_temp = np.pad(edge_dst, ((0, max_bond_num - ef_num)), mode='constant', constant_values=0)
            # graph_data_list[i][3] = ops.one_hot(ed_temp,max_bond_num)
            graph_data_list[i][3] = ed_temp
        # else:
        #     graph_data_list[i][2] = ops.one_hot(edge_src,max_bond_num)
        #     graph_data_list[i][3] = ops.one_hot(edge_dst,max_bond_num)
        # if(smiles_len<max_smiles_ids_len):
        #     # graph_data_list[i][4] = smiles+(' '*(max_smiles_len-smiles_len))
        #     graph_data_list[i][4] = smiles_ids_temp


            
        
    
    normalized_other = np.array(normalized_other)
    
    node_features_shape = graph_data_list[0][0].shape
    edge_features_shape = graph_data_list[0][1].shape
    input_dim3 = len(graph_data_list[0][2])
    item1_list = []
    item2_list = []
    item3_list = []
    item4_list = []
    item5_list = []

    for item in graph_data_list:

        item1_list.append(item[0])
        item2_list.append(item[1])
        item3_list.append(item[2])
        item4_list.append(item[3])
        item5_list.append(item[4])
    # dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(item2_list,dtype=tf.float32),tf.convert_to_tensor(item1_list,dtype=tf.float32), tf.convert_to_tensor(item4_list,dtype=tf.float32)))
    dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(item1_list,dtype=tf.float32,name='node_features'),
                                                  tf.convert_to_tensor(item2_list,dtype=tf.float32,name='edge_features'), 
                                                  tf.convert_to_tensor(item3_list,dtype=tf.int32,name='edge_src'), 
                                                  tf.convert_to_tensor(item4_list,dtype=tf.int32,name='edge_dst'),
                                                  tf.convert_to_tensor(other_data_list,dtype=tf.float32,name='prop'),
                                                  tf.convert_to_tensor(item5_list,dtype=tf.int32,name='smiles'),
                                                  ))
    max_atom_num = node_features_shape[0]
    max_atom_type_num = node_features_shape[1]
    max_bond_type_num = edge_features_shape[1]
    edge_len_num = edge_features_shape[0]
    max_smiles_char_type_num = len(all_smiles_char_types)
    return dataset,vocab, node_features_shape,edge_features_shape,max_atom_num,max_atom_type_num,max_bond_type_num,edge_len_num,max_smiles_len,max_smiles_char_type_num,vocab_size



class Sampling(layers.Layer):
    """
    构建采样图层
    将编码器输出的潜在空间参数（z_mean 和 log_var）
    转化为潜在变量（z），并通过**重参数化技巧（Reparameterization Trick）**
    使得潜在变量可以进行有效的梯度传递，进而进行反向传播和优化。
    换句话说，采样层能够从潜在空间的分布中采样出潜在变量（latent variables），然后用这些变量去生成数据。
    """
    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch, dim = ops.shape(z_log_var)
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

class VAELossCalculator:
    def __init__(self, pad_token=0, kl_weight=0.5):
        self.pad_token = pad_token
        self.kl_weight = kl_weight
        self.recon_loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none'
        )
    
    def compute_loss(self, targets, predictions, z_mean, z_log_var):
        """
        targets: 目标序列 (batch_size, seq_len)
        predictions: 解码器输出logits (batch_size, seq_len, vocab_size)
        z_mean: 编码器输出的均值 (batch_size, latent_dim)
        z_log_var: 编码器输出的对数方差 (batch_size, latent_dim)
        """
        # 1. 计算重构损失（带掩码）
        recon_loss = self._compute_recon_loss(targets, predictions)
        
        # 2. 计算KL散度
        kl_loss = self._compute_kl_loss(z_mean, z_log_var)
        
        # 3. 组合总损失
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def _compute_recon_loss(self, targets, predictions):
        # 创建掩码（排除padding位置）
        mask = tf.math.logical_not(tf.math.equal(targets, self.pad_token))
        mask = tf.cast(mask, dtype=tf.float32)
        
        # 计算每个位置的交叉熵
        loss_per_token = self.recon_loss_fn(targets, predictions)
        
        # 应用掩码并求平均
        loss = tf.reduce_sum(loss_per_token * mask) / tf.reduce_sum(mask)
        return loss
    
    def _compute_kl_loss(self, z_mean, z_log_var):
        # 计算KL散度 (闭合解)
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        return kl_loss

class GAN(keras.Model):
    def __init__(self, encoder, decoder,node_features_shape, edge_features_shape,max_atom_num,max_atom_type_num,max_bond_type_num,edge_len_num,seed = None):
        super(GAN, self).__init__()
        self.decoder = decoder
        self.encoder = encoder
        self.nf_shape = node_features_shape
        self.ef_shape = edge_features_shape
        self.max_atom_num = max_atom_num
        self.max_atom_type_num = max_atom_type_num
        self.max_bond_type_num = max_bond_type_num
        self.edge_len_num = edge_len_num
        
    
        self.physical_socre_tracker = keras.metrics.Mean(name="physical_socre")
        
        self.gen_last_loss = None
        self.disc_last_loss = None
        
        self.fingerprint_list = []
        
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)

        self.gp_weight = 10
        self.discriminator_steps = 1
        self.generator_steps = 1
        
        
        
        self.train_total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.sampling_layer = Sampling(seed=seed)
        self.seed_generator = keras.random.SeedGenerator(seed)
        
        
        pad_token=0
        
        kl_weight=0.5
        self.loss_calculator = VAELossCalculator(pad_token, kl_weight)
        # self.property_prediction_layer = layers.Dense(1)
        
    def create_look_ahead_mask(self, size):
        """创建自回归生成掩码"""
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return tf.cast(mask, tf.bool)  # True表示需要mask的位置
    @property
    def metrics(self):
        return [self.physical_socre_tracker]

    def physical_score(self,predictions,multiple=10):
        """
        简单计算是否符合物理规律
        :param node_features: 节点特征
        :param adjacency_matrix: 邻接矩阵
        :return: 评分（0和1，0：不符合 1：符合）
        """
        res = 0
        fp_list = []
        
        predicted_ids_list = tf.argmax(predictions, axis=-1).numpy() 
        global TOKEN_VALENCE
        
        smiles_list = TOKEN_VALENCE.ids_list_to_smiles(predicted_ids_list)
        for smiles in smiles_list:
            mol = None
            try:
                mol = Chem.MolFromSmarts(smiles)
            except:
                mol=None
            if(mol is None):
                fp_list.append(np.array([]))
                res = res + 3
            else:
                fingerprint_array = np.array([])
                try:
                    fingerprint = self.mfpgen.GetFingerprint(mol)
                    fingerprint_array = np.array(fingerprint)
                except:
                    fingerprint_array = np.array([])
                fp_list.append(fingerprint_array)
                # 检查当前生成的分子是否重复
                duplicate_penalty_score = self.is_duplicate(fingerprint_array)
                ex_score_g = EXPhysicalScore(mol)
                ex_score = ex_score_g.cal_score()
                res = res+duplicate_penalty_score+ex_score
        
        self.fingerprint_list = self.fingerprint_list+fp_list
        return (res/self.batch_size)*multiple
    
    def is_duplicate(self,current_fingerprint, threshold=0.98):
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
    def call(self, inputs):
        # 输出邻接矩阵和节点特征
        z_mean, log_var = self.encoder(inputs)
        z = self.sampling_layer([z_mean, log_var])
        data = inputs[5]
        # 准备解码器输入（右移序列）
        decoder_input = data[:, :-1]  # 移除最后一个token
        targets = data[:, 1:]         # 移除第一个token
        
        # 创建自回归掩码
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(decoder_input)[1])
        
        # 解码器前向
        # predictions = self.decoder((decoder_input, look_ahead_mask), z)
        predictions = self.decoder(decoder_input, z)
        

        return targets, predictions, z_mean, log_var
        # fake_data = self.decoder((z,inputs))
        # nf,ef,es,ed,prop = fake_data[0],fake_data[1],fake_data[2],fake_data[3],fake_data[4]
        # property_pred = layers.ReLU()(self.property_prediction_layer(z_mean))
        # prop_pred = self.property_prediction_layer(z_mean)

        # return z_mean, log_var, (nf,ef,es,ed,prop)
    
    
    
    
    def train_step(self, data):
        
        real_data = (data[0],data[1],data[2],data[3],data[4],data[5])
        
        self.batch_size = tf.shape(real_data[0])[0]
        with tf.GradientTape() as tape:
            targets, predictions, z_mean, log_var= self(real_data, training=True)
                    # 计算损失
            total_loss, recon_loss, kl_loss = self.loss_calculator.compute_loss(
                targets, predictions, z_mean, log_var
            )
            # total_loss,kl_loss , nf_loss ,ef_loss,es_loss,ed_loss,prop_loss = self._compute_loss(z_log_var, z_mean, real_data, fake_data)

        grads = tape.gradient(total_loss, self.trainable_weights)
        # grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]  # 裁剪梯度
        # grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]  # 使用范数裁剪
        self.train_total_loss_tracker.update_state(total_loss)

        # self.physical_socre_tracker.update_state(physical_loss)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        physical_score = self.physical_score(predictions)
        loss = self.train_total_loss_tracker.result()
        # return {"loss": loss,"physical_score":physical_score,"nf_loss":nf_loss,"ef_loss":ef_loss,"prop_loss":prop_loss,"es_loss":es_loss,"ed_loss":ed_loss,"kl_loss":kl_loss}
        return{"loss":total_loss,"recon_loss":recon_loss,"kl_loss":kl_loss,"physical_score":physical_score}



class EXPhysicalScore():
    def __init__(self,mol):
        super(EXPhysicalScore, self).__init__()
        self.mol = mol
        self.total_num = 2
        
        
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
            # mol = Chem.MolFromSmiles(self.smiles)
            Chem.SanitizeMol(self.mol)
            # 逐项检查并累加分值
            score += self.check_connectivity(self.mol)  # 连通性检查
            score += self.check_valency(self.mol)  # 价数检查
            return score  # 返回总分
        except:
            return self.total_num

# region MPNN
class MessagePassingLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim):
        super(MessagePassingLayer, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 定义一个MLP作为消息更新函数
        self.message_fn = tf.keras.Sequential([
            layers.Dense(self.hidden_dim, activation='relu'),
            layers.Dense(self.hidden_dim, activation='relu')
        ])
        
        # 定义一个MLP作为节点更新函数
        self.update_fn = tf.keras.Sequential([
            layers.Dense(self.hidden_dim, activation='relu'),
            layers.Dense(self.hidden_dim, activation='relu')
        ])

    def call(self, node_features, edge_features,edge_src,edge_dst):
        

        # 获取节点特征并初始化消息
        messages = self.aggregate_messages(node_features, edge_features,edge_src,edge_dst)
        
        # 更新节点特征
        updated_node_features = self.update_node_features(node_features, messages)
        
        return updated_node_features

    def aggregate_messages(self, node_features, edge_features,edge_src,edge_dst):
        """聚合邻居节点的信息"""
        # 获取批次大小
        batch_size = tf.shape(node_features)[0]
        # 获取每条边的源节点和目标节点索引
        # 获取源节点和目标节点的特征

        src_node_features = tf.gather(node_features, edge_src, axis=-2,batch_dims=-1)
        dst_node_features = tf.gather(node_features, edge_dst, axis=-2,batch_dims=-1)

        # 将源节点、目标节点特征和边特征结合
        combined_features = tf.concat([src_node_features, dst_node_features, edge_features], axis=-1)

        # 通过MLP来更新消息
        messages = self.message_fn(combined_features)

        # 对每个目标节点的消息进行聚合（加和）


        # aggregated_messages = tf.math.unsorted_segment_sum(messages, edge_dst, num_segments=tf.shape(messages)[1])
        def segment_sum_per_batch(messages, edge_dst):
            batch_aggregated = tf.map_fn(lambda x: tf.math.unsorted_segment_sum(x[0], x[1], num_segments=tf.shape(node_features)[1]),
                                        elems=(messages, edge_dst), dtype=tf.float32)
            return batch_aggregated
        aggregated_messages = segment_sum_per_batch(messages, edge_dst)
        # 将消息的形状调整为 (batch_size, num_nodes, message_dim)
        # aggregated_messages = tf.reshape(aggregated_messages, [batch_size, tf.shape(node_features)[1], self.hidden_dim])
        return aggregated_messages

    def update_node_features(self, node_features, messages):
        """更新节点特征"""
        # 将原始节点特征和聚合后的消息合并
        node_features = tf.cast(node_features,tf.float32)
        combined_features = tf.concat([node_features, messages], axis=-1)
        
        # 通过MLP来更新节点特征
        updated_node_features = self.update_fn(combined_features)
        
        return updated_node_features

class Set2SetLayer(layers.Layer):
    def __init__(self, units):
        super(Set2SetLayer, self).__init__()
        self.units = units
        self.attention = layers.Attention(use_scale=True,)  # 自注意力层
        self.dense = layers.Dense(self.units)  # 用于映射到更高维空间

    def call(self, node_features):
        """
        node_features: 形状为 (batch_size, num_nodes, feature_dim) 的节点特征
        """
        # 1. 使用自注意力机制聚合节点特征
        attention_output = self.attention([node_features, node_features])  # 自注意力聚合

        # 2. 全局池化（可以选择平均池化或最大池化）
        pooled_output = tf.reduce_mean(attention_output, axis=1)  # 全局平均池化

        # 3. 将池化后的输出通过全连接层映射到期望的维度
        output = self.dense(pooled_output)  # 输出图级别的表示
        return output

class MPNNLayer(layers.Layer):
    def __init__(self, hidden_dim, output_dim, num_layers):
        super(MPNNLayer, self).__init__()
        self.message_passing_layers = [MessagePassingLayer(hidden_dim) for _ in range(num_layers)]
        self.set2set_layer = Set2SetLayer(output_dim)
        self.activation = tf.nn.leaky_relu  # relu, leaky_relu, elu
        
        # 添加一个全连接层来预测PCE
        # self.pce_predictor = layers.Dense(1, activation=None)  # 对PCE进行回归预测，通常不加激活函数


    def call(self, data):
        node_features,edge_features,edge_src,edge_dst = data[0], data[1],data[2],data[3]
        residual = None  # 用于残差连接
        for layer in self.message_passing_layers:
            node_features = layer(node_features, edge_features,edge_src,edge_dst)
            node_features = self.activation(node_features)  # 在每层后加入非线性激活
            if(residual is not None):
                node_features = node_features + residual  # 添加残差连接
            residual = node_features  # 更新残差为当前输出

        node_features = self.set2set_layer(node_features)
        # pce = self.pce_predictor(node_features)

        return node_features
    def get_config(self):
        config = super(MPNNLayer, self).get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers
        })
        return config
# endregion



# import tensorflow_addons
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, max_len=75):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
    # def build(self, input_shape):
    #     position = tf.range(self.max_len, dtype=tf.float32)[:, tf.newaxis]
    #     div_term = tf.exp(tf.range(0, self.d_model, 2, dtype=tf.float32) *
    #                      (-tf.math.log(10000.0) / self.d_model))
    #     self.pe = tf.zeros((1, self.max_len, self.d_model))
    #     self.pe[:, :, 0::2] = tf.sin(position * div_term)
    #     self.pe[:, :, 1::2] = tf.cos(position * div_term)
        
    # def call(self, x):
    #     seq_len = tf.shape(x)[1]
    #     return x + self.pe[:, :seq_len, :]

    def build(self, input_shape):
        # 生成位置编码
        position = tf.range(self.max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.range(0, self.d_model, 2, dtype=tf.float32)
        div_term = tf.exp(-tf.math.log(10000.0) * div_term / self.d_model)
        pe_even = tf.sin(position * div_term)
        pe_odd = tf.cos(position * div_term)
        pe = tf.stack([pe_even, pe_odd], axis=-1)
        pe = tf.reshape(pe, (self.max_len, self.d_model))
        pe = tf.expand_dims(pe, axis=0)
        # 将 pe 转为 NumPy 数组
        pe_np = pe.numpy()
        # 使用 Constant 初始化器
        initializer = tf.keras.initializers.Constant(pe_np)
        self.pe = self.add_weight(
            name="pe",
            shape=(1, self.max_len, self.d_model),
            initializer=initializer,
            trainable=False
        )

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

class TransformerDecoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.mha1 = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        self.mha2 = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='gelu'),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)
        
    def call(self, x, z, training, look_ahead_mask):
        # z: [batch_size, d_model] from encoder
        attn1 = self.mha1(query=x, value=x, key=x, attention_mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        # Combine with latent vector z
        z_proj = layers.Dense(self.d_model)(z)[:, tf.newaxis, :]  # [batch, 1, d_model]
        context = tf.concat([out1, tf.tile(z_proj, [1, tf.shape(out1)[1], 1])], axis=-1)
        context = layers.Dense(self.d_model)(context)
        
        attn2 = self.mha2(query=context, value=context, key=context)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        return out3

class SMILESDecoder(keras.Model):
    # def __init__(self, vocab_size, max_len, d_model, num_layers, num_heads, dff, dropout=0.1):
    def __init__(self, hp,config,node_features_shape,edge_features_shape,max_atom_num,max_atom_type_num,max_bond_type_num,edge_len_num,max_smiles_len,max_smiles_char_type_num,vocab_size):
        super(SMILESDecoder, self).__init__()
        # 原有参数初始化...
        self.max_atom_num = max_atom_num
        self.max_atom_type_num = max_atom_type_num
        self.max_bond_type_num = max_bond_type_num
        self.edge_len_num = edge_len_num
        self.max_smiles_len = max_smiles_len
        self.max_smiles_char_type_num = max_smiles_char_type_num
        self.config = config
        
        # 模型维度
        self.d_model = 512
        # 最大SMILES长度
        self.max_len = max_smiles_len
        # SMILES字符集大小
        self.vocab_size = vocab_size
        # 注意力头数
        self.num_heads = 8
        # 前馈网络维度
        self.dff = 2048
        self.dropout = 0.1
        
        self.embedding = layers.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_len)
        
        # self.dec_layers = [TransformerDecoderBlock(self.d_model, self.num_heads, self.dff, self.dropout)
        #                   for _ in range(2)]
        self.dec_layers = TransformerDecoderBlock(self.d_model, self.num_heads, self.dff, self.dropout)
        
        self.final_layer = layers.Dense(self.vocab_size)
        
    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]
    
    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask
    
    def call(self, inputs, training=True):
        # z, data = inputs  # x: (batch, seq_len), z: (batch, latent_dim)
        # x = data[5]
        x, z = inputs  # x: (batch, seq_len), z: (batch, latent_dim)
        # x = data[5]
        seq_len = tf.shape(x)[1]
        
        look_ahead_mask = self.create_look_ahead_mask(seq_len)
        padding_mask = self.create_padding_mask(x)
        combined_mask = tf.maximum(look_ahead_mask, padding_mask)
        
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        
        # for i in range(len(self.dec_layers)):
        #     x = self.dec_layers[i](x, z, training, combined_mask)
        x = self.dec_layers(x, z, training=training, look_ahead_mask=combined_mask)
        
        output = self.final_layer(x)
        return output


class SMILESConstraintLayer(layers.Layer):
    def __init__(self, vocab, atom_valence_rules, **kwargs):
        super().__init__(**kwargs)
        self.vocab = vocab
        self.bracket_pairs = {'(': ')', '[': ']'}
        self.ring_num_limit = 10  # 允许的最大环编号
        
        # 预计算原子价规则（示例）
        self.valence_rules = {
            'C': {'max_bonds': 4},
            'O': {'max_bonds': 2},
            'N': {'max_bonds': 3}
        }

    def call(self, logits, previous_tokens, current_step):
        """
        logits: [batch_size, vocab_size]
        previous_tokens: [batch_size, seq_len]
        """
        batch_size = tf.shape(logits)[0]
        masks = []
        
        for b in range(batch_size):
            mask = self._create_mask_for_sample(
                previous_tokens[b],
                current_step
            )
            masks.append(mask)
            
        mask_tensor = tf.stack(masks, axis=0)
        return logits + mask_tensor

    def _create_mask_for_sample(self, tokens, step):
        # 转换token序列为字符
        chars = [self.reverse_vocab[int(t.numpy())] for t in tokens]
        
        # 初始化掩码（允许所有token）
        mask = tf.zeros(len(self.vocab), dtype=tf.float32)
        
        # 应用约束规则（逐步增强）
        self._apply_bracket_rule(chars, mask)
        self._apply_ring_rule(chars, mask)
        self._apply_valence_rule(chars, mask)
        
        # 转换为logits掩码格式
        return tf.where(mask == 1, -1e9, 0.0)

    def _apply_bracket_rule(self, chars, mask):
        # 括号匹配规则
        open_brackets = []
        for c in chars:
            if c in self.bracket_pairs:
                open_brackets.append(c)
            elif c in self.bracket_pairs.values():
                if open_brackets:
                    open_brackets.pop()
                    
        # 如果存在未闭合括号，禁止结束符
        if open_brackets:
            mask[self.vocab['>']] = 1  # 假设'>'是结束符

    def _apply_ring_rule(self, chars, mask):
        # 环编号规则（1-9重复检测）
        last_char = chars[-1] if chars else ''
        if last_char.isdigit():
            ring_num = int(last_char)
            # 禁止重复的环编号
            for i, c in enumerate(chars[:-1]):
                if c == last_char and chars[i+1] == '%':
                    mask[self.vocab[str(ring_num)]] = 1

    def _apply_valence_rule(self, chars, mask):
        # 原子价规则（简化示例）
        if len(chars) < 2:
            return
            
        prev_char = chars[-1]
        if prev_char in self.valence_rules:
            # 计算当前原子已用化学键
            bond_count = sum(1 for c in chars[-3:] if c in ['=', '#'])
            max_bonds = self.valence_rules[prev_char]['max_bonds']
            
            # 如果价态已满，禁止添加新键
            if bond_count >= max_bonds:
                for bond in ['=', '#']:
                    if bond in self.vocab:
                        mask[self.vocab[bond]] = 1


class TrainingConstraintWrapper(tf.keras.Model):
    """训练时约束包装器，保持梯度流动"""
    def __init__(self, decoder, vocab, constraint_strength=0.5):
        super().__init__()
        self.decoder = decoder
        self.constraint_strength = constraint_strength  # 约束强度系数
        self.vocab = vocab
        self.reverse_vocab = {v:k for k,v in vocab.items()}
        
        # 预定义基础语法规则
        self.bracket_pairs = {'(': ')', '[': ']'}
        self.ring_numbers = set(str(i) for i in range(1,10))
    
    def call(self, inputs, z, training=True):
        # 原始解码器输出
        logits = self.decoder(inputs=(inputs, z), training=training)
        
        # 应用软约束
        constrained_logits = self.apply_training_constraints(inputs, logits)
        
        return constrained_logits
    
    def apply_training_constraints(self, inputs, logits):
        batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
        
        # 创建约束掩码（batch_size, seq_len, vocab_size）
        constraint_mask = tf.map_fn(
            fn=lambda x: self._create_constraint_mask(x),
            elems=inputs,
            fn_output_signature=tf.float32
        )
        
        # 应用软约束：降低无效token的概率（而非完全屏蔽）
        return logits - self.constraint_strength * constraint_mask
    
    def _create_constraint_mask(self, token_seq):
        """为单个序列创建约束掩码"""
        mask = tf.zeros(len(self.vocab), dtype=tf.float32)
        tokens = [self.reverse_vocab[int(t)] for t in token_seq]
        
        # 动态生成约束规则
        for t in range(1, len(tokens)):
            current_char = tokens[t-1]  # 注意使用前一个字符预测下一个
            allowed = self._get_allowed_tokens(tokens[:t])
            
            # 更新掩码：不允许的token增加惩罚
            for v in self.vocab.values():
                if self.reverse_vocab[v] not in allowed:
                    mask = tf.tensor_scatter_nd_add(
                        mask, 
                        [[v]], 
                        [1.0]
                    )
        return mask
    
    def _get_allowed_tokens(self, prev_tokens):
        """基础语法允许的token集合"""
        allowed = set()
        
        # 1. 括号匹配规则
        open_brackets = []
        for c in prev_tokens:
            if c in self.bracket_pairs:
                open_brackets.append(c)
            elif c in self.bracket_pairs.values():
                if open_brackets:
                    open_brackets.pop()
        if open_brackets:
            allowed.add(self.bracket_pairs[open_brackets[-1]])
        
        # 2. 允许继续添加括号
        allowed.update(['(', '[', ')', ']'])
        
        # 3. 环编号规则
        if prev_tokens and prev_tokens[-1].isdigit():
            allowed.update(['%', '#', '='])
        else:
            allowed.update(self.ring_numbers)
        
        # 4. 基础字符
        allowed.update(['C', 'O', 'N', '=', '#'])
        
        # 移除特殊token
        allowed.discard('<pad>')
        allowed.discard('<start>')
        allowed.discard('<end>')
        
        return allowed




class EncoderModel(keras.Model):
    def __init__(self, hp,config,node_features_shape,edge_features_shape,max_atom_num,max_atom_type_num,max_bond_type_num,edge_len_num):
        super(EncoderModel, self).__init__()
        self.hp = hp
        self.config = config
        self.nf_shape = node_features_shape
        self.ef_shape = edge_features_shape
        # self.input_dim3_shape = input_dim3_shape
        self.max_atom_num = max_atom_num
        self.max_atom_type_num = max_atom_type_num
        self.max_bond_type_num = max_bond_type_num
        self.edge_len_num = edge_len_num

        hidden_layer_num2 =get_hp_value(hp,self.config,"hidden_layer_encoder2")
        drop_rate = get_hp_value(hp,self.config,"dropout_rate_encoder")


        # self.mpnn = MPNNLayer(hidden_dim,512,hidden_dim2)
        self.mpnn = MPNNLayer(128,128,3)

        # self.layer = layers.Dense(hidden_layer_num2, activation='leaky_relu')
        self.layer = layers.Dense(hidden_layer_num2, activation='relu')
        self.dropout = layers.Dropout(drop_rate)

        self.z_mean = layers.Dense(12, dtype="float32", kernel_regularizer=keras.regularizers.l2(0.01), name="z_mean")
        self.log_var = layers.Dense(12, dtype="float32", kernel_regularizer=keras.regularizers.l2(0.01), name="log_var")


        
    def call(self, z):
        
        node_features,edge_features,edge_src,edge_dst,prop = z[0],z[1],z[2],z[3],z[4]
        if(edge_src.dtype!=tf.int32):
            edge_src = tf.cast(edge_src,tf.int32)
        if(edge_dst.dtype!=tf.int32):
            edge_dst = tf.cast(edge_dst,tf.int32)
        self.mpnn.build(((None,6),(None,6)))
        x = self.mpnn((node_features,edge_features,edge_src,edge_dst,prop ))

        x = self.layer(x)
        x = self.dropout(x)
        z_mean = self.z_mean(x)
        log_var = self.log_var(x)
        
        return [z_mean,log_var]

    def get_config(self):
        config = super(EncoderModel, self).get_config()
        config.update({
            'hp': self.hp,
            'config': self.config,
            'nf_shape': self.nf_shape,
            'ef_shape': self.ef_shape,
            'max_atom_num': self.max_atom_num,
            'max_atom_type_num': self.max_atom_type_num,
            'max_bond_type_num': self.max_bond_type_num,
            'edge_len_num': self.edge_len_num,
        })
        return config    

    
class CastLayer(keras.Layer):
    def call(self, x):
        return tf.cast(x, tf.float32)

class HyperGAN(kt.HyperModel):

    def __init__(self,config,node_features_shape,edge_features_shape,max_atom_num,max_atom_type_num,max_bond_type_num,edge_len_num,max_smiles_len,max_smiles_char_type_num,vocab_size):
        super(HyperGAN, self).__init__()
        self.nf_shape = node_features_shape
        self.ef_shape = edge_features_shape
        # self.input_dim3_shape = input_dim3_shape
        self.max_atom_num = max_atom_num
        self.max_atom_type_num = max_atom_type_num
        self.max_bond_type_num = max_bond_type_num
        self.edge_len_num = edge_len_num
        self.max_smiles_len = max_smiles_len
        self.max_smiles_char_type_num = max_smiles_char_type_num
        self.vocab_size = vocab_size

        self.gan_model = None

        self.config = config

    def make_decoder_model(self, hp):
        # vocab, atom_valence_rules
        # decoder = SMILESDecoder(hp,self.config,self.nf_shape, self.ef_shape,self.max_atom_num,self.max_atom_type_num,self.max_bond_type_num,self.edge_len_num,self.max_smiles_len,self.max_smiles_char_type_num,self.vocab_size)
        # TODO 更改参数，验证猜想
        # 原始解码器
        # base_decoder = SMILESDecoder(vocab_size=100, max_seq_len=128)
        
        global TOKEN_VALENCE

        vocab = TOKEN_VALENCE.tokenizer.get_vocab()
        
        decoder = SMILESDecoder(hp,self.config,self.nf_shape, self.ef_shape,self.max_atom_num,self.max_atom_type_num,self.max_bond_type_num,self.edge_len_num,self.max_smiles_len,self.max_smiles_char_type_num,self.vocab_size)
        # 添加训练约束
        constrained_decoder = TrainingConstraintWrapper(
            decoder=decoder,
            vocab=vocab,  # 需提供词汇表映射
            constraint_strength=0.7  # 可调节参数
        )

        return constrained_decoder

    def make_encoder_model(self, hp):
        encoder = EncoderModel(hp,self.config,self.nf_shape, self.ef_shape,self.max_atom_num,self.max_atom_type_num,self.max_bond_type_num,self.edge_len_num)
        return encoder

    def build(self, hp):
        self.decoder = self.make_decoder_model(hp)
        self.encoder = self.make_encoder_model(hp)

        model_gan = GAN(self.encoder,self.decoder, self.nf_shape, self.ef_shape,self.max_atom_num,self.max_atom_type_num,self.max_bond_type_num,self.edge_len_num)
        # 优化器
        self.g_optimizer = create_op(hp,self.config,"optimizer","learning_rate")

        # 已重写compile

        model_gan.compile(self.g_optimizer)
        self.gan_model = model_gan
        return model_gan

    def fit(self, hp, model, x,  *args, **kwargs):

        for key in self.config:
            _ = get_hp_value(hp,self.config,key)
        batch_size = get_hp_value(hp,self.config,"batch_size")
        x = x.batch(batch_size, drop_remainder=True)
        res = model.fit(x,*args,**kwargs)
        return res.history


def create_op(hp,config,optimizer_key="optimizer",learning_rate_key="learning_rate"):
    
    disc_optimizer =get_hp_value(hp,config,optimizer_key) 
    _learning_rate = get_hp_value(hp,config,learning_rate_key)
    optimizer = None
    if disc_optimizer.lower() == 'adam':
        optimizer=tf.keras.optimizers.Adam(learning_rate=_learning_rate, clipvalue=1.0)
    elif disc_optimizer.lower() == 'sgd':
        optimizer=tf.keras.optimizers.SGD(learning_rate=_learning_rate)
    elif disc_optimizer.lower() == 'adagrad':
        optimizer=tf.keras.optimizers.Adagrad(learning_rate=_learning_rate)
    elif disc_optimizer.lower() == 'adadelta':
        optimizer=tf.keras.optimizers.Adadelta(learning_rate=_learning_rate)
    elif disc_optimizer.lower() == 'rmsprop':
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=_learning_rate)
    else:
        optimizer=tf.keras.optimizers.Nadam(learning_rate=_learning_rate, clipvalue=1.0)
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
            res = m(key,min_value=config[key]["_value"][0],max_value=config[key]["_value"][-1],sampling=config[key].get("_mode"),step=config[key].get("_step"))
            return res

    elif(_type=='Boolean'):
        if(hp.get(key) is not None):
            return hp.get(key)
        else:
            res = m(key)
            return res
    else:
        return None

def main():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    # csv_file = '/var/project/auto_ml/data/moldata_part_test.csv'
    csv_file = '/var/project/auto_ml/data/moldata_part_test.csv'
    # dataset, input_dim1_shape,input_dim2_shape,max_atom_num,max_atom_type,bond_dim,max_molsize = load_data(csv_file)
    dataset, vocab,node_features_shape,edge_features_shape,max_atom_num,max_atom_type_num,max_bond_type_num,edge_len_num,max_smiles_len,max_smiles_char_type_num,vocab_size = load_data(csv_file)
    
   
    search_space = {
        "optimizer":{
            "_type":"Choice",
            "_value":["adam","sgd","adagrad","adadelta","rmsprop","nadam"]   
        },
        "learning_rate":{
            "_type":"Float",
            "_mode":"log",
            "_value":[1e-4,5e-1]
        },
        # "units":{
        #     "_type":"Int",
        #     "_mode":"linear",
        #     "_value":[1,5]
        # },
        "hidden_dim_encoder":{
            "_type":"Choice",
            "_value":[32,64,128,256,512]
        },
        "hidden_dim_encoder2":{
            "_type":"Choice",
            "_value":[32,64,128,256,512]
        },
        "hidden_dim_decoder":{
            "_type":"Choice",
            # "_value":[32,64,128,256,512]
            "_value":[128,256,512]
        },
        "hidden_layer_decoder":{
            "_type":"Int",
            "_mode":"log",
            "_value":[2,10]
        },
        "hidden_layer_encoder":{
            "_type":"Int",
            "_mode":"log",
            "_value":[2,7]
        },
        "hidden_layer_encoder2":{
            "_type":"Int",
            "_mode":"log",
            "_value":[2,7]
        },
        "dropout_rate_decoder":{
            "_type":"Float",
            "_mode":"log",
            "_value":[0.05,0.5]
        },
        "dropout_rate_encoder":{
            "_type":"Float",
            "_mode":"log",
            "_value":[0.05,0.5]
        },
        "dropout_rate_encoder2":{
            "_type":"Float",
            "_mode":"log",
            "_value":[0.05,0.5]
        },
        "batch_size":{
            "_type":"Choice",
            "_value":[16]
            # "_value":[96]
        },
        "max_trials":20,
        "epochs":40,
    }
    
    search_space2 = {
        "optimizer":{
            "_type":"Choice",
            "_value":["adam"]   
        },
        "learning_rate":{
            "_type":"Float",
            "_mode":"log",
            "_value":[5e-4,5e-2]
        },
        # "units":{
        #     "_type":"Int",
        #     "_mode":"linear",
        #     "_value":[1,5]
        # },
        "hidden_dim_encoder":{
            "_type":"Choice",
            "_value":[128]
        },
        "decoder_gru_hidden":{
            "_type":"Choice",
            "_value":[128]
        },
        "decoder_init_hidden":{
            "_type":"Choice",
            "_value":[128]
        },
        "hidden_dim_encoder2":{
            "_type":"Choice",
            "_value":[128]
        },
        "hidden_dim_decoder":{
            "_type":"Choice",
            # "_value":[32,64,128,256,512]
            "_value":[128,256]
        },
        "hidden_layer_decoder":{
            "_type":"Int",
            "_mode":"log",
            "_value":[2,2]
        },
        "hidden_layer_encoder":{
            "_type":"Int",
            "_mode":"log",
            "_value":[2,2]
        },
        "hidden_layer_encoder2":{
            "_type":"Int",
            "_mode":"log",
            "_value":[2,2]
        },
        "dropout_rate_decoder":{
            "_type":"Float",
            "_mode":"log",
            "_value":[0.05,0.5]
        },
        "dropout_rate_encoder":{
            "_type":"Float",
            "_mode":"log",
            "_value":[0.05,0.5]
        },
        "dropout_rate_encoder2":{
            "_type":"Float",
            "_mode":"log",
            "_value":[0.05,0.5]
        },
        "batch_size":{
            "_type":"Choice",
            "_value":[32,64]
            # "_value":[96]
        },
        "max_trials":20,
        "epochs":40,
    }
   
    
    import time
    import os
    now = time.strftime("%Y%m%d%H%M%S", time.localtime())

    tf.config.experimental_run_functions_eagerly(True)

    LOG_DIR = r'AutoML\models\gan_model\tuner\gan_keras_model_'+f"{now}" 
    is_test = True
    if(is_test):
        search_space = search_space2
    tuner = kt.BayesianOptimization(
        hypermodel=HyperGAN(search_space,node_features_shape,edge_features_shape,max_atom_num,max_atom_type_num,max_bond_type_num,edge_len_num,max_smiles_len,max_smiles_char_type_num,vocab_size),
        objective=kt.Objective("loss", "min"),
        max_trials=search_space["max_trials"],
        overwrite=False,
        max_consecutive_failed_trials = 6,
        directory=LOG_DIR,
        project_name="custom_eval",

    )
    
    es_callback = keras.callbacks.EarlyStopping(
        monitor="adjacency_loss",
        min_delta=1,
        patience=4,
        verbose=0,
        mode="min",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=5,
        )
    
    checkpoint_DIR = r'AutoML\models\gan_model\ModelCheckPoint\model_checkPoint'+now+'.keras'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_DIR,
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=False)
    
    work_path = os.getcwd()
    tb_log_path = os.path.join(work_path,tuner.directory)
    # TODO 1.callbacks回调，具体查看https://keras.io/guides/writing_your_own_callbacks/
    # TODO 2.tensorboard
    # TODO 4.将参数改为动态

    tuner.search(
        x = dataset, 
        epochs = search_space["epochs"],
        callbacks=[keras.callbacks.TensorBoard(log_dir=tb_log_path, histogram_freq=1,write_images=True),es_callback,model_checkpoint_callback]
        )

    tuner.results_summary()
    best_hps = tuner.get_best_hyperparameters()[0]
    print(best_hps.values)
    

    best_model = tuner.get_best_models()[0]

    best_model.build(dataset)  # 假设 X_train.shape[1] 是输入特征数
    best_model.summary()

    best_model.encoder.save(r'D:\Project\ThesisProject\AutoML\models\vae_model\best_model\vae_model_'+now+r'\generator\best'+str(0))

if __name__ == '__main__':

    main()
