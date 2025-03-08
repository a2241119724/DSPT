# Dual-Stage Pixel Transformer with Enhanced Visual Context for Image Captioning
...
## Environment
Clone the repository and create the DSPT conda environment by executing the following command:
``` sh
sh environment.sh
```
## Data
To run the code, annotations, evaluation tools and visual features for the COCO dataset are needed.
- **Annotation**. Download the annotation file [annotations.zip](https://pan.baidu.com/s/1PuNfQnOhkNGNnNyEGxr9rQ). Acess code: labl. Extarct and put it in the project root directory.
- **Feature**. There's a grid feature that you can download from [DLCT](https://github.com/luo3300612/image-captioning-DLCT), or you can download it from [hdf5](https://pan.baidu.com/s/1Au97sw12o7UdrEZN_QRzBg). as I've just made a backup copy. Acess code: labl.
- **Evaluation**. [evaluation.zip](https://pan.baidu.com/s/1dbgJjCyGhYGdgTOVlp1jcg). Acess code: labl. Extarct and put it in the project root directory.

Preprocessing of the Flicker Dataset in the Flicker_util. py File.
## Training
For example, to train our model with the parameters used in our experiments, use
``` sh
python train.py --exp_name DSPT --device cuda:0 --features_path ../swin_feature.hdf5 --batch_size 50 --rl_batch_size 50
```
If it shows "out of graphics memory" after running, reduce the batch_size.
``` sh
python flicker8k_train.py --exp_name flicker8k --features_path ../flicker8k.hdf5 --device cuda:0 --batch_size 25 --rl_batch_size 25
```
``` sh
python flicker30k_train.py --exp_name flicker30k --features_path ../flicker30k.hdf5 --device cuda:0 --batch_size 25 --rl_batch_size 25
```
## Evaluation
To reproduce the results reported in our paper, download the pretrained model file [pth](https://pan.baidu.com/s/1Au97sw12o7UdrEZN_QRzBg). Acess code: labl.
Run the following command:
``` sh
python test.py --exp_name mscoco --features_path ../lab_X101.hdf5 --device cuda:0
```
``` sh
python flicker8k_train.py --exp_name flicker8k --features_path ../flicker8k.hdf5 --device cuda:0 --only_test
```
``` sh
python flicker30k_train.py --exp_name flicker30k --features_path ../flicker30k.hdf5 --device cuda:0 --only_test
```
#### **Ensemble model**
It will search for and perform Averaging Ensemble on the features from all *_best_test.pth files located in the --pth_path folder.You can modify the **TransformerEnsemble** in /models/transformer/transformer.py to implement other types of ensemble methods.
``` sh
python test.py --is_ensemble
```
#### **Evaluation of some indicators**
``` sh
python eval.py
```
## Acknowledgements
Thanks Cornia et.al [M2 transformer](https://github.com/CorniaAI/M2Transformer)