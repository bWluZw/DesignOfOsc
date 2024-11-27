## pyscf
量子化学python库


## conda
conda create --prefix D:\Project\Thesis\DesignOfOsc\AutoML\pyscf_env python=3.8
conda env remove -p D:\Others\备份\Thesis\DesignOfOsc\AutoML\pyscf_env
conda env remove --name pyscf_env

conda activate D:\Project\ThesisProject\AutoML\conda_env
conda deactivate
##### 导出
conda list -e > r.txt 
conda env export > r.yml
##### 导入
conda install --file requirements.txt 
conda env create -f r.yml

##### 别名
conda env config vars set alias=my_alias

##### conda清缓存
conda clean --packages --tarballs
conda clean --all

conda update conda

conda activate base
conda install --revision 0


## docker
##### docker容器启动命令：
docker start gpu_pyscf
##### python脚本执行
python pyscf2csv.py '/var/csv/SMILES_donors.CSV'

##### GPU加速启动（创建+启动）（忘了是什么了，最好别用）：
docker run --rm --gpus all pyscf/pyscf:latest

##### GPU加速启动（创建+启动）：
docker run --gpus all -d --name gpu_pyscf pyscf/pyscf

##### 外部进入命令行
docker exec -u 0 -it gpu_pyscf bash


## 安装GPU加速CUDA Toolkit

其他包基本都过时了，安装这个一劳永逸。
一般来说用docker desktop创建的直接就能和宿主机的GPU交互，但需要装工具来调用
linux下的docker则不同，具体查看别人博客

官网网址：
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local

```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.2-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.2-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

```

#### 常用命令集合

##### 查看GPU信息，常用于查看是否和宿主机联通
nvidia-smi

##### cuda信息
nvcc --version

##### 启动HTTPS（在官网给的命令报错后可以试试）

```
# 启用HTTPS
$ sudo apt install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

```