

import docker
import os

# 启动 Docker 客户端
client = docker.from_env()

# 设置分子和基组
atom_string = "H 0 0 0; H 0 0 0.74"  # 动态传递的原子信息
basis_set = "sto-3g"  # 动态传递的基组

# 构建 PySCF 计算脚本
py_script = f"""
from pyscf import gto, scf

# 设置分子
mol = gto.Mole()
mol.atom = '{atom_string}'
mol.basis = '{basis_set}'
mol.build()

# 进行 SCF 计算
mf = scf.RHF(mol)
mf.kernel()

print('分子的能量:', mf.e_tot)
"""

# 启动 PySCF Docker 容器并运行计算
container = client.containers.run(
    "pyscf/pyscf",  # 使用的 Docker 镜像
    name="pyscf_calculation",  # 容器名称
    command=["python", "-c", py_script],  # 运行 PySCF 计算脚本
    detach=True  # 后台运行
)

# 获取容器日志
container.wait()  # 等待容器完成任务
output = container.logs().decode("utf-8")  # 获取输出

print("计算结果：")
print(output)

# 停止并删除容器
container.remove()




# import docker
# import os

# # Docker客户端
# client = docker.from_env()

# # 配置容器设置
# image = "pyscf/pyscf"  # PySCF的Docker镜像
# container_name = "vigorous_panini"  # 容器名称（可以自定义）
# workdir = "D:\\Envs\\Docker"  # 你的本地工作目录，替换为实际路径

# # 将Windows路径转换为Linux路径
# # 如果你使用的是Docker Desktop（WSL2后端），你不需要转换路径
# workdir_linux = "/mnt/c/Envs/Docker"  # 在容器内的路径

# # 启动容器并运行PySCF计算
# container = client.containers.run(
#     image,
#     name=container_name,
#     volumes={workdir: {'bind': workdir_linux, 'mode': 'rw'}},  # 挂载工作目录
#     working_dir=workdir_linux,  # 设置容器的工作目录
#     command=["python", "smils_pyscf.py"],  # 运行PySCF计算脚本
#     detach=True  # 容器在后台运行
# )

# # 等待容器完成任务并获取输出
# container.wait()  # 等待容器退出
# output = container.logs().decode("utf-8")  # 获取容器输出的日志

# # 打印容器的计算结果
# print(output)

# # 停止并删除容器
# container.remove()