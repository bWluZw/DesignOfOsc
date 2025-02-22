import re
from collections import defaultdict
import threading


import re
from rdkit import Chem
from rdkit.Chem import BRICS
from collections import Counter
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from typing import List, Dict
import selfies as sf
import os


def extract_selfies_tokens(train_selfies):
    token_set = set()
    for s in train_selfies:
        tokens = re.findall(r"(\[[^]]*\])", s)
        token_set.update(tokens)
    return sorted(token_set)

def ids_to_smiles(indices, idx_to_vocab):
    # 将索引转换回SELFIES
    tokens = [idx_to_vocab[i] for i in indices if i not in {0, 1, 2, 3}]  # 过滤特殊Token
    selfies = "".join(tokens)
    
    # 转换为SMILES并验证
    try:
        smiles = sf.decoder(selfies)
        return smiles
    except:
        return None
    
def ids_list_to_smiles(indices_list, idx_to_vocab):
    smiles_list = []
    for i in indices_list:
        smiles = ids_to_smiles(i,idx_to_vocab)
        smiles_list.append(smiles)
    return smiles_list
        
def selfies_to_ids(vocab_to_idx, selfies_str, is_completion=False, max_len=512):
    """
    将 SELFIES 字符串转换为索引列表，支持截断和填充
    
    Args:
        vocab_to_idx (dict): SELFIES词汇表到索引的映射
        selfies_str (str): 输入的SELFIES字符串
        is_completion (bool): 是否补全到max_len长度
        max_len (int): 索引列表的最大长度
    
    Returns:
        list: 包含整数索引的列表
    """
    # 1. 使用正则表达式提取所有[...]格式的token
    tokens = re.findall(r"\[.*?\]", selfies_str)
    
    # 2. 转换为索引，未知token用<UNK>处理
    unk_idx = vocab_to_idx.get("<UNK>", 0)  # 假设<UNK>存在，否则使用0
    indices = [vocab_to_idx.get(token, unk_idx) for token in tokens]
    
    # 3. 截断到最大允许长度
    indices = indices[:max_len]
    
    # 4. 按需补全到max_len
    if is_completion:
        pad_idx = vocab_to_idx.get("<PAD>", 0)  # 假设<PAD>存在
        current_len = len(indices)
        if current_len < max_len:
            indices += [pad_idx] * (max_len - current_len)
    
    return indices


def get_vocab(smiles_data):
    selfies_list=[]
    for item in smiles_data:
        temp = sf.encoder(item)
        selfies_list.append(temp)
    vocab_tokens = extract_selfies_tokens(selfies_list)
    special_tokens = ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]
    full_vocab = special_tokens + vocab_tokens
    vocab_to_idx = {token: i for i, token in enumerate(full_vocab)}
    idx_to_vocab = {i: token for i, token in enumerate(full_vocab)}
    return full_vocab,vocab_to_idx,idx_to_vocab



import pandas as pd
# 使用示例
if __name__ == "__main__":
    csv_file = '/var/project/auto_ml/data/moldata_part_test.csv'
    df = pd.read_csv(csv_file, encoding='utf-8')

    smiles_list = df['SMILES_str'].to_numpy()
    vocab = get_vocab(list(smiles_list))
    vocab_size = len(vocab)
    

