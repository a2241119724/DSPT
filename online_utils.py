import json
import numpy as np
import h5py
import os
import torch
from torchvision import models
from torchvision import transforms
from tqdm import tqdm
from PIL import Image


def gen_online_ids():
    val = json.load(open('annotations/captions_val2014.json', 'r'))
    test = json.load(open('annotations/image_info_test2014.json', 'r'))
    val_ids = []
    for annotation in val["annotations"]:
        val_ids.append(annotation["image_id"])
    np.save('./annotations/online_coco_val_ids.npy', np.array(val_ids))
    test_ids = []
    for image in test["images"]:
        test_ids.append(image["id"])
    np.save('./annotations/online_coco_test_ids.npy', np.array(test_ids))

def gen_online_captions():
    test = json.load(open('./annotations/image_info_test2014.json', 'r'))
    h5 = h5py.File('/media/a1002/one/dataset/wyh/dataset/coco_all_align.hdf5', 'r')
    print(h5[str(1000) + "_grids"] )
    print(h5[str(test["images"][0]["id"]) + "_grids"] )

def gridByTorchvisionCOCO():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet101(pretrained=True).to(device)
    model = torch.nn.Sequential(*list(model.children())[:-2])
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    h5 = h5py.File("../online_coco_test.hdf5", 'w')
    for root, dirs, file in os.walk("../test2014/"):
        for f in tqdm(file, desc='Processing images', total=len(file)):
            id = int(f.split("_")[-1].split(".")[0])
            image = Image.open(os.path.join(root, f)).convert('RGB')
            input_batch = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_batch).squeeze().permute(1, 2, 0).reshape(-1, 2048).cpu().numpy()
                h5.create_dataset(str(id) + '_grids', data=output)
    h5.close()

if __name__ == '__main__':
    gridByTorchvisionCOCO()
