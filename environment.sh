## 杀僵尸进程
# 1.查看显卡上pid: fuser -v /dev/nvidia1
# 2.根据pid看运行的命令是什么: ps aux | grep pid
# 3.杀掉: kill -9 pid
## 直接全杀
# fuser -vk /dev/nvidia*
# ps -aux | grep cuda:0 | awk -F " " '{print $2}' | xargs kill -9
## 查看僵尸进程(不准)
# ps aux | grep 'Z'
#!/bin/bash
conda activate base
conda env remove --name DSPT
conda create --name DSPT python=3.8
conda activate DSPT
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install requests
pip install six
pip install h5py
pip install pycocotools
pip install tqdm
pip install tensorboard==2.14.0
pip install protobuf==3.20.0
conda install spacy
python -m spacy download en_core_web_sm
pip install thop
pip install timm