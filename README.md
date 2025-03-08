# Dual-Stage Pixel Transformer with Enhanced Visual Context for Image Captioning
...
## Environment
Clone the repository and create the DSPT conda environment by executing the following command:
``` sh
sh environment.sh
```
## Data
To run the code, annotations, evaluation tools and visual features for the COCO dataset are needed.
[annotations](https://github.com/salaniz/pycocoevalcap)
[evaluation](https://github.com/salaniz/pycocoevalcap)
[image features](https://github.com/salaniz/pycocoevalcap)
Preprocessing of the Flicker Dataset in the Flicker_util. py File.
## Training
For example, to train our model with the parameters used in our experiments, use
``` sh
python train.py
python flicker8k_train.py --exp_name flicker8k --d_in 1536 --features_path ../flicker8k.hdf5 --device cuda:1 --batch_size 25 --rl_batch_size 10
python flicker8k_train.py --exp_name flicker8k --d_in 1536 --features_path ../flicker8k.hdf5 --device cuda:1 --only_test
python flicker30k_train.py --exp_name flicker30k --d_in 1536 --features_path ../flicker30k.hdf5 --device cuda:1 --batch_size 25 --rl_batch_size 10
python flicker30k_train.py --exp_name flicker30k --d_in 1536 --features_path ../flicker30k.hdf5 --device cuda:1 --only_test
```
## Evaluation
To reproduce the results reported in our paper, download the pretrained model file ... and place it in the code folder.
Run the following command:
``` sh
python test.py
```
### Ensemble model
``` sh
# 会查找--pth_path文件夹下所有*_best_test.pth的特征进行Averaging Ensemble
python test.py --is_ensemble
```
### Evaluation of some indicators
``` sh
python eval.py
```
## Acknowledgements
Thanks Cornia et.al [M2 transformer](https://github.com/CorniaAI/M2Transformer)