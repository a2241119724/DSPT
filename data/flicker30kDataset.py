import h5py
import numpy as np
import torch
import pickle
import os
import json

from collections import Counter
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.field import TextField

class Flickr30kDataset(Dataset):
    def __init__(self, feature_path:str, caption_path:str):
        super(Flickr30kDataset, self).__init__()
        self.f_grid = h5py.File(feature_path, 'r')
        try:
            self.grid_count, self.grid_dim = self.f_grid["689359034_grids"][()].shape
        except:
            self.grid_count, self.grid_dim = self.f_grid["689359034_features"][()].shape
        self.captions = json.load(open(caption_path, 'r'))
        self.flicker30k_train_ids = np.load('./annotations/flicker30k_train_ids.npy')
        self.flicker30k_val_ids = np.load('./annotations/flicker30k_val_ids.npy')
        self.flicker30k_test_ids = np.load('./annotations/flicker30k_test_ids.npy')
        self.text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True,nopoints=False)

    def output_text(self, caption_text):
        # 是否输出源文本,否则输出int[]
        self.caption_text = caption_text      

    def split(self, split):
        self.caption_text = True
        if split == 'train':
            self.flicker30k_ids = self.flicker30k_train_ids
        elif split == 'val':
            self.flicker30k_ids = self.flicker30k_val_ids
        elif split == 'test':
            self.flicker30k_ids = self.flicker30k_test_ids

    def __getitem__(self, index):
        id = str(self.flicker30k_ids[index])
        data = np.array(self.f_grid[id + '_grids'])
        return torch.tensor(data), self.captions[id]
    
    def build_vocab(self, min_freq):
        if os.path.isfile('flicker30k_vocab.pkl'):
            print('Loading from vocabulary')
            vocab = pickle.load(open('flicker30k_vocab.pkl', 'rb'))
            self.text_field.vocab = vocab
            return vocab
        all_tokens = []
        with tqdm(desc='Building Vocabulary', unit='', total=len(self.captions)) as pbar:
            for caption in list(self.captions.values()):
                for _caption in caption:
                    tokens = self.text_field.preprocess(_caption)
                    all_tokens.extend(tokens)
                pbar.update()
        # 统计词频
        word_counts = Counter(all_tokens)
        specials = ['<unk>', '<pad>', '<bos>', '<eos>']
        vocab = self.text_field.vocab_cls(word_counts, specials=specials, min_freq=min_freq)
        with open('flicker30k_vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)
        self.text_field.vocab = vocab
        return vocab
    
    def collate_fn(self):
        def collate_fn(batch):
            data, captions = zip(*batch)
            if not self.caption_text:
                data = torch.cat([d.repeat(5, 1, 1) for d in data], dim=0)
                _captions = []
                for caption in captions:
                    for _caption in caption:
                        _captions.append(self.text_field.preprocess(_caption))
                _captions = self.text_field.process(_captions)
            else:
                _captions = captions
                data = torch.stack(data, 0)
            return data, _captions
        return collate_fn
    
    def __len__(self):
        return len(self.flicker30k_ids)
    
class Flickr30kDataLoader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, collate_fn=dataset.collate_fn(), **kwargs)
