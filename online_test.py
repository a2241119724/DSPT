import evaluation
import torch
import argparse
import json
import pickle
 
from models.transformer import Transformer, Encoder, Decoder, TransformerEnsemble
from tqdm import tqdm
from data.onlineDataset import OnlineDataset, OnlineDataLoader

def predict_captions(model, dataloader, text_field, dump_json):
    import itertools
    _res = {}
    model.eval()
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (grids, ids) in enumerate(iter(dataloader)):
            grids = grids.to(device)
            with torch.no_grad():
                out, _, _ = model.beam_search(torch.rand(0), grids, torch.rand(0), torch.rand(0), 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            caps_gen = evaluation.PTBTokenizer.tokenize(caps_gen)
            for i, gen in caps_gen.items():
                gen = ' '.join([k for k, g in itertools.groupby(gen)])
                _res[int(ids[i])] = gen.strip()
            pbar.update()
    res = []
    for k, v in _res.items():
        res.append({
            "image_id": k,
            'caption': v
        })
    json.dump(res,open(dump_json,'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSPT')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--head', type=int, default=4)
    parser.add_argument('--enc_N', type=int, default=6)
    parser.add_argument('--dec_N', type=int, default=6)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--pths', nargs='+', default=['./saved_models/DSPT_X101.pth', './saved_models/lab_X101_12e-8_best_test.pth'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--trainval_feature', type=str, default='/media/a1002/one/dataset/wyh/dataset/coco_all_align.hdf5')
    parser.add_argument('--test_feature', type=str, default='../test2014.hdf5')
    args = parser.parse_args()
    device = torch.device(args.device)

    print('Online Test Evaluation')
    val_dataset = OnlineDataset(args.trainval_feature, 'val')
    test_dataset = OnlineDataset(args.test_feature, 'test')
    val_dataset.text_field.vocab = test_dataset.text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

    d_qkv = int(args.d_model // args.head)
    encoder = Encoder(args.enc_N, d_k=d_qkv, d_v=d_qkv, h=args.head, d_in=val_dataset.grid_dim, d_model=args.d_model, grid_count=val_dataset.grid_count)
    decoder = Decoder(len(val_dataset.text_field.vocab), 54, args.dec_N, val_dataset.text_field.vocab.stoi['<pad>'], d_k=d_qkv, d_v=d_qkv, h=args.head, d_model=args.d_model)
    model = Transformer(val_dataset.text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    model.eval()
    model = TransformerEnsemble(model=model, weight_files=args.pths, device=device)

    val_dataloader = OnlineDataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers)
    test_dataloader = OnlineDataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers)
    predict_captions(model, val_dataloader, val_dataset.text_field, "./output/captions_val2014_DSPT_results.json")
    predict_captions(model, test_dataloader, test_dataset.text_field, "./output/captions_test2014_DSPT_results.json")
