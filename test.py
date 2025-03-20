import evaluation
import torch
import argparse
import pickle
import numpy as np
import json
import multiprocessing
import itertools

from PIL import Image
from data import ImageField, TextField, RawField
from data import COCO, DataLoader
from evaluation import Cider, PTBTokenizer
from models.transformer import Transformer, Encoder, Decoder, TransformerEnsemble
from models.transformer.attention import ScaledDotProductAttention_encoder
from tqdm import tqdm
from torchvision import transforms
from torchvision import models

def predict_captions(model, dataloader, text_field, cider, args, split):
    tokenizer_pool = multiprocessing.Pool()
    res = {}
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, ((regions, grids, boxes, sizes, ids), caps_gt) in enumerate(iter(dataloader)):
            regions, grids, boxes, sizes = regions.to(device), grids.to(device), boxes.to(device), sizes.to(device)
            with torch.no_grad():
                out, _, _ = model.beam_search(torch.rand(0), grids, torch.rand(0), torch.rand(0), 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            caps_gen1 = text_field.decode(out)
            caps_gt1 = list(itertools.chain(*([c, ] * 1 for c in caps_gt)))

            caps_gen1, caps_gt1 = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen1, caps_gt1])
            reward = cider.compute_score(caps_gt1, caps_gen1)[1].astype(np.float32)

            for i,(gts_i, gen_i) in enumerate(zip(caps_gt1,caps_gen1)):
                res[ids[i].item()] = {
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
    json.dump(res,open("./output/" + split + "_" + args.dump_json,'w'))
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSPT')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--head', type=int, default=4)
    parser.add_argument('--enc_N', type=int, default=6)
    parser.add_argument('--dec_N', type=int, default=6)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--features_path', type=str, default='../../../one/dataset/wyh/dataset/coco_all_align.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='./annotations')
    parser.add_argument('--dump_json', type=str, default='gen_res.json')
    parser.add_argument('--is_ensemble', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--input_id', type=int, default=-1)
    parser.add_argument('--pths', nargs='+', default=['./saved_models/DSPT_X101.pth', './saved_models/lab_X101_12e-8_best_test.pth'])
    args = parser.parse_args()
    device = torch.device(args.device)
    print('Test Evaluation')

    # Pipeline for image regions
    image_field = ImageField(feature_path=args.features_path, max_detections=0, load_in_tmp=False)
    image_field.id2Image = True

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    _, val_dataset, test_dataset = dataset.splits
    text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

    ref_caps_test = list(test_dataset.text)
    cider_test = Cider(PTBTokenizer.tokenize(ref_caps_test))

    # Model and dataloaders
    d_qkv = int(args.d_model // args.head)
    encoder = Encoder(args.enc_N, d_k=d_qkv, d_v=d_qkv, h=args.head, d_in=image_field.grid_dim, d_model=args.d_model, grid_count=image_field.grid_count)
    decoder = Decoder(len(text_field.vocab), 54, args.dec_N, text_field.vocab.stoi['<pad>'], d_k=d_qkv, d_v=d_qkv, h=args.head, d_model=args.d_model)
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    model.eval()

    if not args.is_ensemble:
        data = torch.load(args.pths[0], map_location=device)
        model.load_state_dict(data['state_dict'])
    else:
        model = TransformerEnsemble(model=model, weight_files=args.pths, device=device)

    # if args.input is None:
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)
    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size, num_workers=args.workers)
    if(args.input_id == -1):
        scores = predict_captions(model, dict_dataloader_val, text_field, cider_test, args, 'val')
        print(scores)
        scores = predict_captions(model, dict_dataloader_test, text_field, cider_test, args, 'test')
        print(scores)
    else:
        ScaledDotProductAttention_encoder.isVisual = True
        # encoder.encoderFusion.mhatt[0].attention.isVisual = True
        grids = image_field.getById(args.input_id).unsqueeze(0).to(device)
        caps_gt = test_dataset.id2Caption[args.input_id]
        with torch.no_grad():
            out, _, _ = model.beam_search(torch.rand(0), grids, torch.rand(0), torch.rand(0), 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
        caps_gen = text_field.decode(out, join_words=False)
        caps_gen1 = text_field.decode(out)
        caps_gt1 = list(itertools.chain(*([c, ] * 1 for c in caps_gt)))

        caps_gen1 = [caps_gen1[0] for i in range(5)]
        caps_gen1 = evaluation.PTBTokenizer.tokenize(caps_gen1)
        caps_gt1 = evaluation.PTBTokenizer.tokenize(caps_gt1)
        print("gen: " + caps_gen1[0][0])
        print("gt: " + str(caps_gt1))
        reward = cider_test.compute_score(caps_gt1, caps_gen1)[1].astype(np.float32)
        print("cider:" + str(reward.mean()))
