import json
import numpy as np
import h5py
import os
import torch
from tqdm import tqdm
from PIL import Image
from timm.models.swin_transformer import swin_tiny_patch4_window7_224,swin_large_patch4_window12_384

def gridByTorchvisionCOCO():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = swin_tiny_patch4_window7_224(pretrained=True, num_classes=0).to(device)
    model = torch.nn.Sequential(*list(model.children())[:-2])
    model.eval()
    pool = torch.nn.AdaptiveAvgPool2d((7, 7))
    h5 = h5py.File("../Swin_trainval.hdf5", 'w')
    for root, dirs, file in os.walk("../Deep-Fusion-Transformer/DFT/coco/train2014/"):
        for f in tqdm(file, desc='Processing images', total=len(file)):
            id = int(f.split("_")[-1].split(".")[0])
            image = Image.open(os.path.join(root, f)).convert('RGB')
            input_batch = image.unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_batch).squeeze().permute(1, 2, 0).reshape(-1, 2048).cpu().numpy()
                output = pool(output)
                h5.create_dataset(str(id) + '_grids', data=output)
    for root, dirs, file in os.walk("../Deep-Fusion-Transformer/DFT/coco/val2014/"):
        for f in tqdm(file, desc='Processing images', total=len(file)):
            id = int(f.split("_")[-1].split(".")[0])
            image = Image.open(os.path.join(root, f)).convert('RGB')
            input_batch = image.unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_batch).squeeze().permute(1, 2, 0).reshape(-1, 2048).cpu().numpy()
                output = pool(output)
                h5.create_dataset(str(id) + '_grids', data=output)
    h5.close()
    h5 = h5py.File("../Swin_grid_feats_coco_test.hdf5", 'w')
    for root, dirs, file in os.walk("../Deep-Fusion-Transformer/DFT/coco/test2014/"):
        for f in tqdm(file, desc='Processing images', total=len(file)):
            id = int(f.split("_")[-1].split(".")[0])
            image = Image.open(os.path.join(root, f)).convert('RGB')
            input_batch = image.unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_batch).squeeze().permute(1, 2, 0).reshape(-1, 2048).cpu().numpy()
                output = pool(output)
                h5.create_dataset(str(id) + '_grids', data=output)
    h5.close()

if __name__ == '__main__':
    gridByTorchvisionCOCO()
