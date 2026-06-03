Reproduce:

Network proxy
```BASH
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export HF_ENDPOINT=https://hf-mirror.com
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
export MAX_JOB=1
```

Environment
```BASH
conda create -n rsrd python=3.10 -y
conda activate rsrd

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132

pip install nerfstudio
pip install gsplat --no-build-isolation

cd ~/rsrd
pip install -e dependencies/garfield

pip install \
  --extra-index-url=https://pypi.nvidia.com \
  cudf-cu12==24.10.* \
  cuml-cu12==24.10.*

pip install -U "jax[cuda13]"

pip install git+https://github.com/chungmin99/jaxmp.git --ignore-requires-python

cd ~/rsrd
pip install -e .

cd dependencies/
pip install -e dig -e raftstereo -e hamer_helper -e jaxls

pip install torchtyping==0.1.5
pip install "setuptools==75.8.2" --force-reinstall
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation
```

My device info
```BASH
(base) gdy@DESKTOP-RS1KQEU:~/rsrd$ nvidia-smi
Sun May 31 00:33:49 2026       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.309.01             Driver Version: 595.79       CUDA Version: 13.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 5060 ...    On  | 00000000:01:00.0  On |                  N/A |
| N/A   73C    P4              39W /  40W |   7543MiB /  8151MiB |     86%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A        34      G   /Xwayland                                 N/A      |
|    0   N/A  N/A    214867      C   /python3.10                               N/A      |
+---------------------------------------------------------------------------------------+
(base) gdy@DESKTOP-RS1KQEU:~/rsrd$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
(base) gdy@DESKTOP-RS1KQEU:~/rsrd$ which nvcc
/usr/local/cuda-13.2/bin/nvcc
```

Train the model
```PYTHON
# 标定相机参数
ns-process-data images \
  --data /home/gdy/rsrd/dataset/nerf_recon_dataset/custom/hotdog/images \
  --output-dir /home/gdy/rsrd/dataset/nerf_recon_dataset/custom/hotdog_processed \
  --matching-method exhaustive # 少量数据直接穷举，>1000用vocab tree模式


# 训练模型
ns-train garfield --data /home/gdy/rsrd/dataset/nerf_recon_dataset/custom/hotdog_processed

# 保存渲染video到本地
python scripts/run_tracker.py \
  --is-obj-jointed False \
  --dig-config-path /home/gdy/rsrd/outputs/bear/dig/2026-05-31_035315/config.yml \
  --video-path /home/gdy/rsrd/dataset/nerf_recon_dataset/corl_data/demonstrations/bear.MOV \
  --output-dir /home/gdy/rsrd/outputs/bear --no-save-hand # 不需要手部信息；如果需要手部信息需要安装hamer

# robot视角下的操作视图
python scripts/run_planner.py \
  --hand-mode single \
  --track-dir /home/gdy/rsrd/outputs/bear
```