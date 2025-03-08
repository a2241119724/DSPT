import random
import os
import evaluation
import torch
import argparse
import pickle
import numpy as np
import json
import multiprocessing

from data import ImageField, TextField, RawField
from data import COCO, DataLoader
from evaluation import Cider, PTBTokenizer
from models.transformer import Transformer, Encoder, Decoder, TransformerEnsemble
from tqdm import tqdm

def predict_captions(model, dataloader, text_field, cider, args):
    import itertools
    tokenizer_pool = multiprocessing.Pool()
    res = {}
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, ((regions, grids, boxes, sizes), caps_gt) in enumerate(iter(dataloader)):
            regions, grids, boxes, sizes = regions.to(device), grids.to(device), boxes.to(device), sizes.to(device)
            with torch.no_grad():
                out, _, _ = model.beam_search(torch.rand(0), grids, torch.rand(0), torch.rand(0), 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            caps_gen1 = text_field.decode(out)
            caps_gt1 = list(itertools.chain(*([c, ] * 1 for c in caps_gt)))

            caps_gen1, caps_gt1 = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen1, caps_gt1])
            reward = cider.compute_score(caps_gt1, caps_gen1)[1].astype(np.float32)
            # reward = reward.mean().item()

            for i,(gts_i, gen_i) in enumerate(zip(caps_gt1,caps_gen1)):
                res[len(res)] = {
                    'gt':caps_gt1[gts_i],
                    'gen':caps_gen1[gen_i],
                    'cider':reward[i].item(),
                }

            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    json.dump(res,open(args.dump_json,'w'))
    return scores


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--head', type=int, default=4)
    parser.add_argument('--enc_N', type=int, default=6)
    parser.add_argument('--dec_N', type=int, default=6)
    parser.add_argument('--d_model', type=int, default=512)
    # parser.add_argument('--features_path', type=str, default='../swin_feature.hdf5')
    # parser.add_argument('--features_path', type=str, default='../X152_trainval.hdf5')
    parser.add_argument('--features_path', type=str, default='/media/a1002/one/dataset/wyh/dataset/coco_all_align.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='./annotations')
    parser.add_argument('--exp_name', type=str, default='DSPT')
    parser.add_argument('--dump_json', type=str, default='gen_res.json')
    parser.add_argument('--is_ensemble', action='store_true', default=False)
    parser.add_argument('--pth_path', type=str, default="./saved_models/")
    parser.add_argument('--d_in', type=int, default=2048)
    args = parser.parse_args()

    print('Transformer Evaluation')

    # Pipeline for image regions
    image_field = ImageField(feature_path=args.features_path, max_detections=0, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    _, _, test_dataset = dataset.splits
    text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

    ref_caps_test = list(test_dataset.text)
    cider_test = Cider(PTBTokenizer.tokenize(ref_caps_test))

    # Model and dataloaders
    d_qkv = int(args.d_model // args.head)
    encoder = Encoder(args.enc_N, d_k=d_qkv, d_v=d_qkv, h=args.head, d_in=args.d_in, d_model=args.d_model)
    decoder = Decoder(len(text_field.vocab), 54, args.dec_N, text_field.vocab.stoi['<pad>'], d_k=d_qkv, d_v=d_qkv, h=args.head, d_model=args.d_model)
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    pths = []
    for filename in os.listdir(args.pth_path):
        if '_best_test.pth' in filename:
            pths.append(args.pth_path + filename)

    if not args.is_ensemble:
        data = torch.load(pths[0])
        model.load_state_dict(data['state_dict'])
    else:
        model = TransformerEnsemble(model=model, weight_files=pths)

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)
    scores = predict_captions(model, dict_dataloader_test, text_field, cider_test, args)
    print(scores)
