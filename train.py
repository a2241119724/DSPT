import torch
import random
import evaluation
import argparse
import os
import pickle
import numpy as np
import itertools
import multiprocessing

from data import ImageField, TextField, RawField, COCO, DataLoader
from evaluation import PTBTokenizer, Cider
from models.transformer import Transformer, Encoder, Decoder
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from poly import poly_loss
from utils.utils import print_parameter_count

seed = 2024
print("seed: " + str(seed))
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (regions, grids, boxes, sizes, captions) in enumerate(dataloader):
                regions, grids, boxes, sizes, captions = regions.to(device), grids.to(device), boxes.to(device), sizes.to(device), captions.to(device)
                out, enc_output = model(torch.rand(0), grids, torch.rand(0), torch.rand(0), captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()

                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1), ignore_index=text_field.vocab.stoi['<pad>']) + torch.abs(
                    torch.cosine_similarity(enc_output.unsqueeze(1),enc_output.unsqueeze(2),-1)).sum() * 1.1e-7
                
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, ((regions, grids, boxes, sizes), caps_gt) in enumerate(iter(dataloader)):
            regions, grids, boxes, sizes = regions.to(device), grids.to(device), boxes.to(device), sizes.to(device)
            with torch.no_grad():
                out, _, _ = model.beam_search(torch.rand(0), grids, torch.rand(0), torch.rand(0), 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, loss_fn, text_field):
    # Training with cross-entropy
    model.train()
    scheduler.step()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (regions, grids, boxes, sizes, captions) in enumerate(dataloader):
            regions, grids, boxes, sizes, captions = regions.to(device), grids.to(device), boxes.to(device), sizes.to(device), captions.to(device)
            out, enc_output = model(torch.rand(0), grids, torch.rand(0), torch.rand(0), captions) # 丢弃sizes
            optim.zero_grad()
            captions = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1), ignore_index=text_field.vocab.stoi['<pad>']) + torch.abs(
                torch.cosine_similarity(enc_output.unsqueeze(1),enc_output.unsqueeze(2),-1)).sum() * 1.1e-7

            loss.backward()
            optim.step()

            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    return loss


def train_scst(model, dataloader, optim, cider, text_field):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward_baseline = .0
    model.train()
    scheduler.step()
    running_loss = .0
    seq_len = 20
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, ((regions, grids, boxes, sizes), caps_gt) in enumerate(dataloader):
            regions, grids, boxes, sizes = regions.to(device), grids.to(device), boxes.to(device), sizes.to(device)
            outs, log_probs, _ = model.beam_search(torch.rand(0), grids, torch.rand(0), torch.rand(0), seq_len, text_field.vocab.stoi['<eos>'],
                beam_size, out_size=beam_size)
            optim.zero_grad()

            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(grids.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)

            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward_baseline


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSPT')
    parser.add_argument('--exp_name', type=str, default='DSPT')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--rl_batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--features_path', type=str, default='../coco_all_align.hdf5')
    parser.add_argument('--enc_N', type=int, default=6)
    parser.add_argument('--dec_N', type=int, default=6)
    parser.add_argument('--head', type=int, default=4)
    parser.add_argument('--resume_last', action='store_true', default=False)
    parser.add_argument('--resume_best', action='store_true', default=False)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--xe_least', type=int, default=20)
    parser.add_argument('--xe_most', type=int, default=20)
    parser.add_argument('--annotation_folder', type=str, default='./annotations')
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    parser.add_argument('--xe_base_lr', type=float, default=0.0001)
    args = parser.parse_args()
    device = torch.device(args.device)
    print(args)

    print('Transformer Training')
    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # Pipeline for image
    image_field = ImageField(feature_path=args.features_path, max_detections=0, load_in_tmp=False)
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True,
        nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits

    if not os.path.isfile('vocab.pkl'):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab.pkl', 'wb'))
    else:
        print('Loading from vocabulary')
        text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

    # Model and dataloaders
    d_qkv = int(args.d_model // args.head)
    encoder = Encoder(args.enc_N, d_k=d_qkv, d_v=d_qkv, h=args.head, d_in=image_field.grid_dim, d_model=args.d_model, grid_count=image_field.grid_count)
    decoder = Decoder(len(text_field.vocab), 54, args.dec_N, text_field.vocab.stoi['<pad>'], d_k=d_qkv, d_v=d_qkv, h=args.head, d_model=args.d_model)
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    print_parameter_count(model,is_simplify=True, is_print_all=False,is_print_detail=False, contain_str="decoder")

    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    ref_caps_train = list(train_dataset.text)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    init_epoch = 25
    def lambda_lr(s):
        if s == 0: return 1
        if not use_rl:
            if s <= 3: lr = s / 4
            elif s <= 10: lr = 1
            elif s <= 12: lr = 0.2
            else: lr = 0.2 * 0.2
            lr = lr * args.xe_base_lr
        else:
            if s <= init_epoch: lr = 1
            elif s <= init_epoch + 5: lr =  0.2
            elif s <= init_epoch + 10: lr =  0.2 * 0.2
            else: lr =  0.2 * 0.2 * 0.2
            lr = lr * 5e-6
        print("lr:", lr) # 5e-6
        return lr
    
    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)

    loss_fn = poly_loss
    use_rl = False
    best_val_cider = .0
    best_test_cider = 0.
    patience = 0
    start_epoch = 0

    if args.resume_best or args.resume_last:
        if args.resume_last:
            fname = 'saved_models/%s_last.pth' % args.exp_name
        else:
            fname = 'saved_models/%s_best.pth' % args.exp_name

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=True)
            start_epoch = data['epoch'] + 1
            best_val_cider = data['best_val_cider']
            best_test_cider = data['best_test_cider']
            patience = data['patience']
            use_rl = data['use_rl']
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            print('Resuming from epoch %d, validation loss %f, best cider %f, and best_test_cider %f' % (
                data['epoch'], data['val_loss'], data['best_val_cider'], data['best_test_cider']))
            print('patience:', data['patience'])

    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                      drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.rl_batch_size // 5, shuffle=True,
                                           num_workers=args.workers)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.rl_batch_size // 5)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.rl_batch_size // 5)

        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, loss_fn, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            train_loss, reward_baseline = train_scst(model, dict_dataloader_train, optim, cider_train, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)

        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field)
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)
        writer.add_scalar('data/val_spice', scores['SPICE'], e)

        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, text_field)
        print("Test scores", scores)
        test_cider = scores['CIDEr']
        writer.add_scalar('data/test_cider', test_cider, e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)
        writer.add_scalar('data/test_spice', scores['SPICE'], e)

        best_val = False
        if val_cider >= best_val_cider:
            best_val_cider = val_cider
            best_val = True
            patience = 0
        else:
            patience += 1

        best_test = False
        if test_cider >= best_test_cider:
            best_test_cider = test_cider
            best_test = True

        switch_to_rl = False
        exit_train = False
        if patience >= 5 and e >= args.xe_least:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True

        if not use_rl and e >= args.xe_most:
            use_rl = True
            switch_to_rl = True
            patience = 0
            print("Switching to RL")

        if switch_to_rl and not best_val:
            data = torch.load('saved_models/%s_best.pth' % args.exp_name)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            print('Resuming from epoch %d, validation loss %f, best_val_cider %f, and best_test_cider %f' % (
                data['epoch'], data['val_loss'], data['best_val_cider'], data['best_test_cider']))

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_val_cider': best_val_cider,
            'best_test_cider': best_test_cider,
            'use_rl': use_rl,
        }, 'saved_models/%s_last.pth' % args.exp_name)

        if best_val:
            copyfile('saved_models/%s_last.pth' % args.exp_name,
                     'saved_models/%s_best.pth' % args.exp_name)

        if best_test:
            copyfile('saved_models/%s_last.pth' % args.exp_name,
                     'saved_models/%s_best_test.pth' % args.exp_name)
        
        if switch_to_rl:
            copyfile('saved_models/%s_best_test.pth' % args.exp_name,
                     'saved_models/%s_best_xe_test.pth' % args.exp_name)
            copyfile('saved_models/%s_last.pth' % args.exp_name,
                     'saved_models/%s_xe_last.pth' % args.exp_name)

        if exit_train:
            writer.close()
            break
