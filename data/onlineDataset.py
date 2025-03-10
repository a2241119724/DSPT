import h5py
import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data.field import TextField

class OnlineDataset(Dataset):
    def __init__(self, feature_path:str, split:str):
        super(OnlineDataset, self).__init__()
        self.f_grid = h5py.File(feature_path, 'r')
        try:
            self.grid_count, self.grid_dim = self.f_grid["1000_grids"][()].shape
        except:
            self.grid_count, self.grid_dim = self.f_grid["96735_grids"][()].shape
        if split == 'val':
            self.coco_ids = np.load('./annotations/online_coco_val_ids.npy')
        elif split == 'test':
            self.coco_ids = np.load('./annotations/online_coco_test_ids.npy')
        self.text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True,nopoints=False)

    def __getitem__(self, index):
        id = str(self.coco_ids[index])
        data = np.array(self.f_grid[id + '_grids'])
        return torch.tensor(data), id
    
    def collate_fn(self):
        def collate_fn(batch):
            data, id = zip(*batch)            
            data = torch.stack(data, 0)
            return data, id
        return collate_fn
    
    def __len__(self):
        return len(self.coco_ids)
    
class OnlineDataLoader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, collate_fn=dataset.collate_fn(), **kwargs)
