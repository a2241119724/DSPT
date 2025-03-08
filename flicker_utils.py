import xml.etree.ElementTree as ET
import os
import json
import h5py
import os
import numpy as np
import torch

from timm.models.swin_transformer import swin_large_patch4_window12_384
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms
# from detectron2.config import get_cfg
# from detectron2.evaluation import inference_context
# from detectron2.modeling import build_model
# from grid_feats_vqa.grid_feats import add_attribute_config

def get_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset

    input:
      fn - full file path to the sentence file to parse
    
    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this 
                                    phrase belongs to

    """
    with open(fn, 'r') as f:
        sentences = f.read().split('\n')

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == ']':
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == '[':
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split('/')
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {'sentence' : ' '.join(words), 'phrases' : []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data['phrases'].append({'first_word_index' : index,
                                             'phrase' : phrase,
                                             'phrase_id' : p_id,
                                             'phrase_type' : p_type})

        annotations.append(sentence_data)

    return annotations

def get_annotations(fn):
    """
    Parses the xml files in the Flickr30K Entities dataset

    input:
      fn - full file path to the annotations file to parse

    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the 
                  [xmin ymin xmax ymax] format
    """
    tree = ET.parse(fn)
    root = tree.getroot()
    size_container = root.findall('size')[0]
    anno_info = {'boxes' : {}, 'scene' : [], 'nobox' : []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall('object'):
        for names in object_container.findall('name'):
            box_id = names.text
            box_container = object_container.findall('bndbox')
            if len(box_container) > 0:
                if box_id not in anno_info['boxes']:
                    anno_info['boxes'][box_id] = []
                xmin = int(box_container[0].findall('xmin')[0].text) - 1
                ymin = int(box_container[0].findall('ymin')[0].text) - 1
                xmax = int(box_container[0].findall('xmax')[0].text) - 1
                ymax = int(box_container[0].findall('ymax')[0].text) - 1
                anno_info['boxes'][box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall('nobndbox')[0].text)
                if nobndbox > 0:
                    anno_info['nobox'].append(box_id)

                scene = int(object_container.findall('scene')[0].text)
                if scene > 0:
                    anno_info['scene'].append(box_id)

    return anno_info

def flicker30k_json():
    '''
        {id:[captions...]}
    '''
    if(not os.path.exists("./annotations/Sentences")):
        return
    json_data = {}
    for root, dirs, file in os.walk("./annotations/Sentences"):
        for f in file:
            data = get_sentence_data(os.path.join(root, f))
            for i in range(len(data)):
                id = str(f.split('.')[0])
                if id in json_data:
                    json_data[id].append(data[i]['sentence'])
                else:
                    json_data[id] = [data[i]['sentence']]
    with open("./annotations/flicker30k_captions.json", 'w') as outfile:
        json.dump(json_data, outfile)

def gridByTorchvision30k():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = models.resnet101(pretrained=True).to(device)
    # model = models.resnet152(pretrained=True).to(device)
    model = swin_large_patch4_window12_384(pretrained=True, num_classes=0).to(device)
    model = torch.nn.Sequential(*list(model.children())[:-2])
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # preprocess = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     )
    # ])
    h5 = h5py.File("../flicker30k.hdf5", 'w')
    for root, dirs, file in os.walk("../Flicker30k_Dataset/"):
        for f in tqdm(file, desc='Processing images', total=len(file)):
            image = Image.open(os.path.join(root, f)).convert('RGB')
            input_batch = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_batch).squeeze().permute(1, 2, 0).reshape(-1, 1536).cpu().numpy()
                # output = model(input_batch).squeeze().permute(1, 2, 0).reshape(-1, 2048).cpu().numpy()
                h5.create_dataset(f.split('.')[0] + '_grids', data=output)
    h5.close()

def gridByTorchvision8k():
    '''
        val_preprocess = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)),
        ])
    '''
    nameToId = {}
    dataset = json.load(open("../flickr8k/dataset_flickr8k.json", 'r'))
    for annotation in dataset["images"]:
        nameToId[annotation["filename"]] = [annotation["imgid"], annotation["split"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = models.resnet101(pretrained=True).to(device)
    # model = models.resnet152(pretrained=True).to(device)
    model = swin_large_patch4_window12_384(pretrained=True, num_classes=0).to(device)
    model = torch.nn.Sequential(*list(model.children())[:-2])
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # preprocess = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     )
    # ])
    h5 = h5py.File("../flicker8k_swin.hdf5", 'w')
    for root, dirs, file in os.walk("../Flicker8k_Dataset/"):
        for f in tqdm(file, desc='Processing images', total=len(file)):
            if(f not in nameToId): continue
            image = Image.open(os.path.join(root, f)).convert('RGB')
            input_batch = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_batch).squeeze().permute(1, 2, 0).reshape(-1, 1536).cpu().numpy()
                # output = model(input_batch).squeeze().permute(1, 2, 0).reshape(-1, 2048).cpu().numpy()
                h5.create_dataset(str(nameToId[f][0]) + '_grids', data=output)
    h5.close()

def gridByDetectron2():
    cfg = get_cfg()
    add_attribute_config(cfg)
    cfg.merge_from_file("./grid_feats_vqa/configs/X-101-grid.yaml")
    cfg.MODEL.RESNETS.RES5_DILATION = 1
    cfg.freeze()
    model = build_model(cfg)
    model.load_state_dict(torch.load('../X-101.pth')["model"], strict=False)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    h5 = h5py.File("../flicker30k.hdf5", 'w')
    with inference_context(model):
        for root, dirs, file in os.walk("../Flicker30k_Dataset/"):
            for f in tqdm(file, desc='Processing images', total=len(file)):
                with torch.no_grad():
                    image = Image.open(os.path.join(root, f)).convert('RGB')
                    image = [{"image":torch.from_numpy(np.array(preprocess(image))).permute(2, 0, 1)}]
                    image = model.preprocess_image(image)
                    feature = model.backbone(image.tensor)
                    output = model.roi_heads.get_conv5_features(feature)
                    output = output.squeeze().permute(1, 2, 0).reshape(-1, 2048).cpu().numpy()
                    h5.create_dataset(f.split('.')[0] + '_grids', data=output)
    h5.close()

def npy30k():
    flicker30k_train_ids = open('./annotations/flicker30k_train_ids.txt', 'r')
    flicker30k_val_ids = open('./annotations/flicker30k_val_ids.txt', 'r')
    flicker30k_test_ids = open('./annotations/flicker30k_test_ids.txt', 'r')
    flicker30k_ids = np.array([int(id.rstrip("\n")) for id in flicker30k_train_ids.readlines()])
    np.save('./annotations/flicker30k_train_ids.npy', flicker30k_ids)
    flicker30k_ids = np.array([int(id.rstrip("\n")) for id in flicker30k_val_ids.readlines()])
    np.save('./annotations/flicker30k_val_ids.npy', flicker30k_ids)
    flicker30k_ids = np.array([int(id.rstrip("\n")) for id in flicker30k_test_ids.readlines()])
    np.save('./annotations/flicker30k_test_ids.npy', flicker30k_ids)

def npy8k():
    train = []
    val = []
    test = []
    dataset = json.load(open("../flickr8k/dataset_flickr8k.json", 'r'))
    for annotation in dataset["images"]:
        train.append(annotation["imgid"]) if annotation["split"] == "train" else None
        val.append(annotation["imgid"]) if annotation["split"] == "val" else None
        test.append(annotation["imgid"]) if annotation["split"] == "test" else None
    np.save('./annotations/flicker8k_train_ids.npy', np.array(train))
    np.save('./annotations/flicker8k_val_ids.npy', np.array(val))
    np.save('./annotations/flicker8k_test_ids.npy', np.array(test))

def flicker8k_json():
    json_data = {}
    dataset = json.load(open("../flickr8k/dataset_flickr8k.json", 'r'))
    for annotation in dataset["images"]:
        json_data[annotation["imgid"]] = []
        for sentence in annotation["sentences"]:
            json_data[annotation["imgid"]].append(sentence["raw"])
    with open("./annotations/flicker8k_captions.json", 'w') as outfile:
        json.dump(json_data, outfile)

if(__name__ == '__main__'):
    pass
