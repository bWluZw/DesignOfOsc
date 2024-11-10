import sys
# from DesignOfOsc.AutoML.Project.pyscf2csv import gto, scf
#from pyscf import gto, scf
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd

def smi_to_mol(smi):
    """将SMILES字符串转换为PySCF分子对象"""
    mol = Chem.MolFromSmiles(smi)
    # 补全氢原子
    mol = Chem.AddHs()
    mol_block = Chem.MolToMolBlock(mol)
    return mol_block

def run_pyscf(mol_block, basis='sto-3g'):
    """使用PySCF计算HOMO, LUMO和gap"""
    # 将mol_block转化为PySCF的分子对象
    mol = gto.Mole()
    #根据分子中原子数和氢数估算电子数
    mol.build(atom=mol_block, basis=basis)
    nelec = mol.nelectron
    # 自动设定自旋，偶数电子配对，奇数电子默认未配对
    spin = 0 if nelec % 2 == 0 else 1
    mol.build(atom=mol_block, basis=basis, spin=spin)
    # 进行SCF计算
    mf = scf.RHF(mol)
    mf.kernel()  # 计算自洽场
    
    # 提取HOMO, LUMO, Gap
    homo = mf.mo_energy[mf.mo_occ > 0][-1]  # HOMO: 最后一个占据的轨道的能量
    lumo = mf.mo_energy[mf.mo_occ == 0][0]  # LUMO: 第一个未占据的轨道的能量
    gap = lumo - homo  # 能隙

    return homo, lumo, gap

def main():
    # # 获取命令行参数：SMILES字符串
    # if len(sys.argv) != 2:
    #     print("Usage: python <script_name> <SMILES_string>")
    #     sys.exit(1)
    
    # smi = sys.argv[1]
    
    
    # path = 'D:\Docs\西安石油大学\论文\论文资料\数据集集合\总\2020年中期基于非富勒烯的有机太阳能电池数据集\SMILES_donors.CSV'
    path='/var/csv/SMILES_donors.CSV'
    all_data = pd.read_csv(path,sep=',',header=0,names=['name','smiles'])

    all_data['confnum']=[]
    all_data['homo']=[]
    all_data['lumo']=[]
    all_data['gap']=[]

    smiles_list=all_data['smiles']

    for (index,item) in enumerate(smiles_list):
        # 将SMILES转换为Mol block格式
        mol_block = smi_to_mol(item)
        homo, lumo, gap = run_pyscf(mol_block)
        all_data.loc[index,'homo'] = homo
        all_data.loc[index,'lumo']=lumo
        all_data.loc[index, 'gap'] = gap
    
    all_data.to_csv('/var/csv/SMILES_donors_all.CSV')
    
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
