#coding:utf-8
import sys
# from DesignOfOsc.AutoML.Project.pyscf2csv import gto, scf
from pyscf import gto, scf
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import os
# 启用GPU加速
os.environ["PYSCF_CUDA"] = "1"  # 启用GPU



def smi_to_mol(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        
        # 补全氢原子
        mol = Chem.AddHs(mol)
        
        # 设置嵌入的参数
        params = AllChem.ETKDG()  # 使用 ETKDG 构象生成方法
        params.randomSeed = 42  # 设置随机种子
        params.maxAttempts = 500  # 最大尝试次数
        params.useRandomCoords = True  # 使用随机坐标初始化
        
        # 为分子生成3D坐标
        status = AllChem.EmbedMolecule(mol, params=params)  # 使用参数化的 EmbedMolecule
        
        if status != 0:
            raise ValueError(f"Embedding molecule failed with status {status}")
        
        # 优化分子结构
        # 仅传递支持的参数，移除 nonBondedThresh
        status = AllChem.UFFOptimizeMolecule(mol, maxIters=2000, vdwThresh=10.0, confId=-1, ignoreInterfragInteractions=True)
        
        if status != 0:
            raise ValueError("Optimization failed")
        
        # 确保有构象
        num_conformers = mol.GetNumConformers()
        if num_conformers == 0:
            raise ValueError("No conformers found after embedding and optimization")
        
        # 获取分子坐标并转换为PySCF格式
        atom_coords = []
        conformer = mol.GetConformer()
        
        for atom in mol.GetAtoms():
            pos = conformer.GetAtomPosition(atom.GetIdx())
            atom_coords.append("{} {:.6f} {:.6f} {:.6f}".format(atom.GetSymbol(), pos.x, pos.y, pos.z))
        
        mol_block = "\n".join(atom_coords)
        return mol_block
        
    except KeyboardInterrupt:
        exit()
    except Exception as e:
        print('smi_to_mol: Error -', e)
        return None

# def smi_to_mol(smi):
#     try:
#         mol = Chem.MolFromSmiles(smi)
#             # 补全氢原子
#         mol = Chem.AddHs(mol)
#         # # 为分子生成3D坐标
#         AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=500, method=AllChem.ETKDG())
#         AllChem.UFFOptimizeMolecule(mol)  # 优化分子结构
        
#         # # 获取分子坐标并转换为PySCF格式
#         atom_coords = []
#         for atom in mol.GetAtoms():
#             pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
#             atom_coords.append("{} {:.6f} {:.6f} {:.6f}".format(atom.GetSymbol(), pos.x, pos.y, pos.z))
        
#         mol_block = "\n".join(atom_coords)
#         # mol_block = Chem.MolToMolBlock(mol)
#         return mol_block
#     except KeyboardInterrupt:
#         exit()
#     except Exception as e:
#         print('smi_to_mol:')
#         print(e)
#         return None

def run_pyscf(mol_block, basis='sto-3g'):
    """使用PySCF计算HOMO, LUMO和gap"""
    try:
        # 将mol_block转化为PySCF的分子对象
        mol = gto.Mole()
        mol.build(atom=mol_block, basis=basis)
        # nelec = mol.nelectron
        # # 自动设定自旋，偶数电子配对，奇数电子默认未配对
        # spin = 0 if nelec % 2 == 0 else 1
        # mol.build(atom=mol_block, basis=basis, spin=spin)
        # 进行SCF计算

        mf = scf.RHF(mol).to_gpu()
        mf.kernel()  # 计算自洽场
        
        # 提取HOMO, LUMO, Gap
        homo = mf.mo_energy[mf.mo_occ > 0][-1]  # HOMO: 最后一个占据的轨道的能量
        lumo = mf.mo_energy[mf.mo_occ == 0][0]  # LUMO: 第一个未占据的轨道的能量
        gap = lumo - homo  # 能隙

        return homo, lumo, gap
    except KeyboardInterrupt:
        exit()
    except Exception as e:
        print('run_pyscf:')
        print(e)
        return None,None,None

def main():
    # # 获取命令行参数：SMILES字符串
    # if len(sys.argv) != 2:
    #     print("Usage: python <script_name> <SMILES_string>")
    #     sys.exit(1)

    # smi = sys.argv[1]

    # path = 'D:\Docs\西安石油大学\论文\论文资料\数据集集合\总\2020年中期基于非富勒烯的有机太阳能电池数据集\SMILES_donors.CSV'
    # path='/var/csv/SMILES_donors.CSV'
    path=sys.argv[1]
    #path = 'D:\Project\Thesis\DesignOfOsc\AutoML\csv\SMILES_donors.CSV'
    all_data = pd.read_csv(path,sep=',',index_col=None)

    res_data = pd.DataFrame(columns=['name','smiles','homo','lumo','gap'], dtype='object')
    for (index,item) in all_data.iterrows():
        # 将SMILES转换为Mol block格式
        print(str(index))
        mol_block = smi_to_mol(item['SMILE'])
        if(mol_block==None):
            print('mol_block==None'+str(index))
            continue
        homo, lumo, gap = run_pyscf(mol_block)
        if(homo==None):
            print('homo==None'+str(index))
            continue
        res_data.loc[index,'name'] = item['Name']
        res_data.loc[index,'smiles'] = item['SMILE']
        res_data.loc[index,'homo']=homo
        res_data['lumo'][index]=lumo
        res_data['gap'][index]=gap
        print('complete:'+str(index))

        res_data.to_csv(path + ".csv", index=False,encoding='utf-8-sig')
    # # 将SMILES转换为Mol block格式
    # mol_block = smi_to_mol(smi)

    # # 使用PySCF进行计算
    # homo, lumo, gap = run_pyscf(mol_block)

    # # 打印结果
    # print(f"SMILES: {smi}")
    # print(f"HOMO: {homo}")
    # print(f"LUMO: {lumo}")
    # print(f"Gap (LUMO - HOMO): {gap}")

if __name__ == "__main__":
    main()
