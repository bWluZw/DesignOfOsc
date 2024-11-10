from rdkit import Chem
from rdkit.Chem import AllChem
from DesignOfOsc.AutoML.Project.pyscf2csv import gto, scf

def smiles_to_xyz(smiles):
    # 使用RDKit解析SMILES并生成分子
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 生成三维构象
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)  # 优化结构

    # 导出XYZ格式
    conformer = mol.GetConformer()
    coords = conformer.GetPositions()
    atom_info = "\n".join(
        [f"{atom.GetSymbol()} {coords[i][0]} {coords[i][1]} {coords[i][2]}" 
         for i, atom in enumerate(mol.GetAtoms())]
    )

    # 返回XYZ格式
    xyz_str = f"{len(coords)}\n\n{atom_info}"
    return xyz_str

def calculate_homo_lumo(smiles):
    # 将SMILES转换为XYZ格式
    xyz_str = smiles_to_xyz(smiles)
    if xyz_str is None:
        print("无效的SMILES")
        return None, None, None
    
    # 使用PySCF进行量子化学计算
    mol = gto.Mole()
    mol.atom = xyz_str
    mol.basis = 'sto-3g'  # 使用较小的基组，适用于快速计算
    mol.build()

    # 进行SCF计算
    mf = scf.RHF(mol)
    mf.kernel()

    # 提取HOMO和LUMO能级
    homo = mf.mo_energy[mf.mo_occ > 0][-1]  # HOMO能级
    lumo = mf.mo_energy[mf.mo_occ == 0][0]   # LUMO能级
    gap = lumo - homo                        # 能隙

    return homo, lumo, gap

# 示例SMILES（你可以替换为任何有效的SMILES字符串）
smiles = "CCO"  # 乙醇的SMILES

homo, lumo, gap = calculate_homo_lumo(smiles)

if homo is not None:
    print(f"HOMO: {homo} Ha")
    print(f"LUMO: {lumo} Ha")
    print(f"Gap: {gap} Ha")
else:
    print("计算失败")