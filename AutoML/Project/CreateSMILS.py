import psi4
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

def extract_properties(smiles):
    # 解析SMILES字符串
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None, None, None, None
    
    # 生成三维构型
    AllChem.EmbedMolecule(molecule)
    AllChem.UFFOptimizeMolecule(molecule)  # 优化分子结构

    # 获取confnum（构象数量）
    confnum = molecule.GetNumConformers()
    
    # 导出分子的XYZ坐标
    conformer = molecule.GetConformer()
    coords = conformer.GetPositions()
    xyz_str = f"{len(coords)}\n\n" + "\n".join(
        [f"{atom.GetSymbol()} {coords[i][0]} {coords[i][1]} {coords[i][2]}" 
         for i, atom in enumerate(molecule.GetAtoms())]
    )

    # 使用Psi4进行量子化学计算
    psi4.set_output_file('output.dat', overwrite=True)
    
    # 定义分子
    psi4.geometry(xyz_str)

    # 进行计算
    energy, wfn = psi4.energy('scf/6-31G(d)', return_wfn=True)

    # 提取HOMO和LUMO能级
    homo = wfn.epsilon_a()[-1]  # HOMO能级
    lumo = wfn.epsilon_a()[-2]  # LUMO能级
    gap = lumo - homo           # 能隙

    return confnum, homo, lumo, gap

# # 示例SMILES（可以替换为其他分子）
# smiles = "CCO"  # 乙醇的SMILES
# confnum, homo, lumo, gap = extract_properties(smiles)

# if confnum is not None:
#     print(f"Confnum: {confnum}")
#     print(f"HOMO: {homo} eV")
#     print(f"LUMO: {lumo} eV")
#     print(f"Gap: {gap} eV")
# else:
#     print("无效的SMILES字符串")


if __name__=='main':
    path = 'D:\Docs\西安石油大学\论文\论文资料\数据集集合\总\2020年中期基于非富勒烯的有机太阳能电池数据集\SMILES_donors.CSV'
    all_data = pd.read_csv(path,sep=',',header=0,names=['name','smiles'])

    all_data['confnum']=[]
    all_data['homo']=[]
    all_data['lumo']=[]
    all_data['gap']=[]

    smiles_list=all_data['smiles']

    for (index,item) in enumerate(smiles_list):
        confnum, homo, lumo, gap = extract_properties(item)
        all_data.loc[index,'confnum'] = confnum
        all_data.loc[index,'homo'] = homo
        all_data.loc[index,'lumo']=lumo
        all_data.loc[index, 'gap'] = gap

    pass
