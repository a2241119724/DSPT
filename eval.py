import torch
import os
import pickle
import argparse

from tqdm import tqdm
from models.transformer import Transformer, Encoder, Decoder
from thop import profile
from data import ImageField,TextField,COCO,DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DSPT')
    parser.add_argument('--features_path', type=str, default='/media/a1002/one/dataset/wyh/dataset/coco_all_align.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='./annotations')
    args = parser.parse_args()
    print(args)
    image_field = ImageField(feature_path=args.features_path, max_detections=0, load_in_tmp=False)
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True,nopoints=False)
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits
    dataloader_train = DataLoader(train_dataset, batch_size=1)
    if not os.path.isfile('vocab.pkl'):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab.pkl', 'wb'))
    else:
        print('Loading from vocabulary')
        text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))
    encoder = Encoder(6, d_k=128, d_v=128, h=4, d_in=2048, d_model=512)
    decoder = Decoder(len(text_field.vocab), 54, 6, text_field.vocab.stoi['<pad>'], d_k=128, d_v=128, h=4, d_model=512)
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder)
    total = 0
    with tqdm(desc='Eval: ', unit='it', total=100) as pbar:
        for it, (regions, grids, boxes, sizes, captions) in enumerate(dataloader_train):
            if it < 10:
                pbar.update()
                continue
            if it == 100: break
            flops, params = profile(model, inputs=(torch.rand(0), grids, torch.rand(0), torch.rand(0), captions))
            total = total + flops/1000**3
            pbar.update()
    print('FLOPs = ' + str(total/90) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
