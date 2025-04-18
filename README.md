# Dual-Stage Pixel Transformer with Enhanced Visual Context for Image Captioning
| DSPA | Arch | EVCA |
|:---:|:---:|:---:|
|![encoder](./images/encoder.png)|![architecture](./images/architecture.png)|![decoder](./images/decoder.png)|

## Environment
Clone the repository and create the DSPT conda environment by executing the following command:
``` sh
sh environment.sh
```

## Data
To run the code, annotations, evaluation tools and visual features for the COCO dataset are needed.
- **Annotation**. Download the annotation file [annotations.zip](https://pan.baidu.com/s/1KCGlotCKlZF0FrDB995IzA). Acess code: labl. Extarct and put it in the project root directory.
- **Feature**. The grid features extracted using X101 and X152 as baselines on the MSCOCO dataset can be downloaded from [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa). For the X101 we use, you can download its processed version from [DLCT](https://github.com/luo3300612/image-captioning-DLCT). For the X152 we use, you can download its processed version from [RSTNet](https://github.com/zhangxuying1004/RSTNet). And you can download other feature from [hdf5](https://pan.baidu.com/s/1Au97sw12o7UdrEZN_QRzBg). Acess code: labl.
- **Evaluation**. [evaluation.zip](https://pan.baidu.com/s/1rAYvKcQOGkYoUPrTpY2qUQ). Acess code: labl. Extarct and put it in the project root directory.

Preprocessing of the Flicker Dataset in the **flicker_utils.py** File.

## Training
Train a model using the **MSCOCO** dataset. Run the following command:
``` sh
python train.py --exp_name DSPT --device cuda:0 --features_path ../coco_all_align.hdf5 --batch_size 50 --rl_batch_size 50
```
Train a model using the **Flicker8k** dataset. Run the following command:
``` sh
python flicker8k_train.py --exp_name flicker8k --features_path ../flicker8k.hdf5 --device cuda:0 --batch_size 25 --rl_batch_size 25
```
Train a model using the **Flicker30k** dataset. Run the following command:
``` sh
python flicker30k_train.py --exp_name flicker30k --features_path ../flicker30k.hdf5 --device cuda:0 --batch_size 25 --rl_batch_size 25
```
If it shows "out of graphics memory" after running, reduce the batch_size and rl_batch_size.

## Evaluation
To reproduce the results reported in our paper, download the pretrained model file [DSPT_X101.pth]() [DSPT_X152.pth](https://pan.baidu.com/s/1Xin98dpSZRGknMfxmRUcOw) [DSPT_Swin.pth](https://pan.baidu.com/s/1p-Va8cGR0L4DY_U8peSZGg). Acess code: labl. Evaluation a model using the **MSCOCO** dataset. Run the following command:
``` sh
python test.py --features_path ../coco_all_align.hdf5 --device cuda:0 --pths DSPT_best_test.pth
```
Download the pretrained model file [flicker8k.pth](https://pan.baidu.com/s/1cydcKkLTVEcDp2F-SGB-4g). Acess code: labl. Evaluation a model using the **Flicker8k** dataset. Run the following command:
``` sh
python flicker8k_train.py --exp_name flicker8k --features_path ../flicker8k.hdf5 --device cuda:0 --only_test
```
Download the pretrained model file [flicker30k.pth](https://pan.baidu.com/s/11RJjSDdYBlRkpmDfyfbR9w). Acess code: labl. Evaluation a model using the **Flicker30k** dataset. Run the following command:
``` sh
python flicker30k_train.py --exp_name flicker30k --features_path ../flicker30k.hdf5 --device cuda:0 --only_test
```

#### **Ensemble model**
It will search for and perform **Averaging Ensemble** on the features from all *_best_test.pth files located in the --pth_path folder. You can modify the **TransformerEnsemble** in /models/transformer/transformer.py to implement other types of ensemble methods.
``` sh
python test.py --is_ensemble --features_path ../coco_all_align.hdf5 --pths ./saved_models/DSPT_1_best_test.pth ./saved_models/DSPT_2_best_test.pth ./saved_models/DSPT_3_best_test.pth ./saved_models/DSPT_4_best_test.pth
```

#### **Evaluation of some indicators**
``` sh
python eval.py --features_path ../coco_all_align.hdf5
```

#### **Online Test**
``` sh
python online_test.py --device cuda:0 --trainval_feature ../coco_all_align.hdf5 --test_feature ../test2014.hdf5 --pths ./saved_models/DSPT_1_best_test.pth ./saved_models/DSPT_2_best_test.pth ./saved_models/DSPT_3_best_test.pth ./saved_models/DSPT_4_best_test.pth
```
The result is generated under **./output**, and you can submit this evaluation result to the ![server](The result is generated under output, and you can submit it to this evaluation result)

## Acknowledgements
Thanks Cornia et.al [M2 transformer](https://github.com/CorniaAI/M2Transformer)
