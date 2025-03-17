import torch
import numpy as np
import random
import os
import itertools
import evaluation
import multiprocessing
import argparse

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from utils.utils import print_parameter_count
from data.flicker30kDataset import Flickr30kDataset, Flickr30kDataLoader
from models.transformer import Transformer, Encoder, Decoder
from tqdm import tqdm
from poly import poly_loss
from evaluation import PTBTokenizer, Cider
from torch.utils.tensorboard import SummaryWriter

seed = 2024
print("seed: " + str(seed))
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--exp_name', type=str, default='DSPT')
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--rl_batch_size', type=int, default=25)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--enc_N', type=int, default=6)
    parser.add_argument('--dec_N', type=int, default=6)
    parser.add_argument('--head', type=int, default=4)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--features_path', type=str, default='../flicker30k.hdf5')
    parser.add_argument('--caption_path', type=str, default='./annotations/flicker30k_captions.json')
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    parser.add_argument('--xe_base_lr', type=float, default=0.0001)
    parser.add_argument('--only_test', action='store_true', default=False)
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset = Flickr30kDataset(feature_path=args.features_path,caption_path=args.caption_path)
    vocab = dataset.build_vocab(min_freq=5)
    d_qkv = int(args.d_model // args.head)
    encoder = Encoder(args.enc_N, d_k=d_qkv, d_v=d_qkv, h=args.head, d_in=dataset.grid_dim, d_model=args.d_model, grid_count=dataset.grid_count)
    decoder = Decoder(len(vocab), 80, args.dec_N, vocab.stoi['<pad>'], d_k=d_qkv, d_v=d_qkv, h=args.head, d_model=args.d_model)
    model = Transformer(vocab.stoi['<bos>'], encoder, decoder).to(device)
    print_parameter_count(model, is_simplify=True, is_print_all=False,is_print_detail=False, contain_str="decoder")
    
    init_epoch = 25
    use_rl = False
    def lambda_lr(s):
        if s == 0: return 1
        if not use_rl:
            if s <= 3: lr = s / 4
            elif s <= 10: lr = 1
            elif s <= 12: lr = 0.2
            else : lr = 0.2 * 0.2
            lr = lr * args.xe_base_lr
        else:
            if s <= init_epoch: lr = 1
            elif s <= init_epoch + 5: lr =  0.2
            elif s <= init_epoch + 10: lr =  0.2 * 0.2
            else: lr =  0.2 * 0.2 * 0.2
            lr = lr * 5e-6
        print("lr:", lr) # 5e-6
        return lr
    
    if not args.only_test:
        writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))
    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)
    best_test_cider = 0
    start_epoch = 0

    fname = 'saved_models/%s.pth' % args.exp_name
    if args.resume or args.only_test:
        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=True)
            start_epoch = data['epoch']
            if not args.only_test:
                start_epoch = start_epoch + 1
            best_test_cider = data['best_test_cider']
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            print('Resuming from epoch %d, and best_test_cider %f' % (
                data['epoch'], data['best_test_cider']))
        else:
            print("Don't find saved model")

    cider_train = Cider(PTBTokenizer.tokenize(dataset.captions))
    loss_fn = poly_loss
    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        if e >= 21:
            use_rl = True
            data = torch.load(fname)
            model.load_state_dict(data['state_dict'], strict=True)
        if not args.only_test:
            scheduler.step()
            running_loss = .0
            model.train()
            dataset.split("train")
            if not use_rl:
                dataloader = Flickr30kDataLoader(dataset, batch_size=args.batch_size//5, shuffle=True, num_workers=args.workers)
                dataset.output_text(False)
                with tqdm(desc='Epoch %d - training' % e, unit='it', total=len(dataloader)) as pbar:
                    for it, (images, captions) in enumerate(iter(dataloader)):
                        images, captions = images.to(device), captions.to(device)
                        out, enc_output = model(torch.rand(0),images, torch.rand(0), torch.rand(0), captions)
                        
                        optim.zero_grad()
                        captions = captions[:, 1:].contiguous()
                        out = out[:, :-1].contiguous()

                        loss = loss_fn(out.view(-1, len(vocab)), captions.view(-1), ignore_index=vocab.stoi['<pad>']) 
                        + torch.abs(torch.cosine_similarity(enc_output.unsqueeze(1),enc_output.unsqueeze(2),-1)).sum() * 1e-7

                        loss.backward()
                        optim.step()

                        this_loss = loss.item()
                        running_loss += this_loss
                        pbar.set_postfix(loss=running_loss / (it + 1))
                        pbar.update()
                writer.add_scalar('data/train_loss', running_loss / len(dataloader), e)
            else:
                dataloader = Flickr30kDataLoader(dataset, batch_size=args.rl_batch_size//5, shuffle=True, num_workers=args.workers)
                tokenizer_pool = multiprocessing.Pool()
                running_reward = .0
                running_reward_baseline = .0
                with tqdm(desc='Epoch %d - validating' % e, unit='it', total=len(dataloader)) as pbar:
                    for it, (images, captions) in enumerate(iter(dataloader)):
                        images = images.to(device)
                        outs, log_probs, _ = model.beam_search(torch.rand(0), images, torch.rand(0), torch.rand(0), 20, 
                            vocab.stoi['<eos>'], 5, out_size=5)
                        optim.zero_grad()
                        # Rewards
                        caps_gen = dataset.text_field.decode(outs.view(-1, 20))
                        caps_gt = list(itertools.chain(*([c, ] * 5 for c in captions)))
                        caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
                        reward = cider_train.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
                        reward = torch.from_numpy(reward).to(device).view(images.shape[0], 5)
                        reward_baseline = torch.mean(reward, -1, keepdim=True)

                        loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

                        loss = loss.mean()
                        loss.backward()
                        optim.step()

                        running_loss += loss.item()
                        running_reward += reward.mean().item()
                        running_reward_baseline += reward_baseline.mean().item()
                        pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                            reward_baseline=running_reward_baseline / (it + 1))
                        pbar.update()
                writer.add_scalar('data/train_loss', running_loss / len(dataloader), e)
                writer.add_scalar('data/reward', running_reward / len(dataloader), e)
                writer.add_scalar('data/reward_baseline', running_reward_baseline / len(dataloader), e)
        gen = {}
        gts = {}
        model.eval()
        dataset.split("val")
        dataloader = Flickr30kDataLoader(dataset, batch_size=10, shuffle=False, num_workers=args.workers)
        with tqdm(desc='Epoch %d - validating' % e, unit='it', total=len(dataloader)) as pbar:
            for it, (images, captions) in enumerate(iter(dataloader)):
                images = images.to(device)
                with torch.no_grad():
                    out, _, _ = model.beam_search(torch.rand(0), images, torch.rand(0), torch.rand(0), 20, 
                        vocab.stoi['<eos>'], 5, out_size=1)
                caps_gen = dataset.text_field.decode(out, join_words=False)
                for i, (gts_i, gen_i) in enumerate(zip(captions, caps_gen)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gen['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = gts_i
                pbar.update()
        gts = evaluation.PTBTokenizer.tokenize(gts)
        gen = evaluation.PTBTokenizer.tokenize(gen)
        scores, _ = evaluation.compute_scores(gts, gen)
        print(scores)
        if not args.only_test:
            writer.add_scalar('data/val_cider', scores['CIDEr'], e)
            writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
            writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
            writer.add_scalar('data/val_meteor', scores['METEOR'], e)
            writer.add_scalar('data/val_rouge', scores['ROUGE'], e)
            writer.add_scalar('data/val_spice', scores['SPICE'], e)
        #
        gen = {}
        gts = {}
        dataset.split("test")
        dataloader = Flickr30kDataLoader(dataset, batch_size=10, shuffle=False, num_workers=args.workers)
        with tqdm(desc='Epoch %d - Testing' % e, unit='it', total=len(dataloader)) as pbar:
            for it, (images, captions) in enumerate(iter(dataloader)):
                images = images.to(device)
                with torch.no_grad():
                    out, _, _ = model.beam_search(torch.rand(0), images, torch.rand(0), torch.rand(0), 20, 
                        vocab.stoi['<eos>'], 5, out_size=1)
                caps_gen = dataset.text_field.decode(out, join_words=False)
                for i, (gts_i, gen_i) in enumerate(zip(captions, caps_gen)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gen['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = gts_i
                pbar.update()
        gts = evaluation.PTBTokenizer.tokenize(gts)
        gen = evaluation.PTBTokenizer.tokenize(gen)
        scores, _ = evaluation.compute_scores(gts, gen)
        print(scores)
        if args.only_test:
            break
        writer.add_scalar('data/test_cider', scores['CIDEr'], e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)
        writer.add_scalar('data/test_spice', scores['SPICE'], e)

        if scores["CIDEr"] > best_test_cider:
            best_test_cider = scores["CIDEr"]
            torch.save({
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate(),
                'epoch': e,
                'state_dict': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_test_cider': best_test_cider,
            }, fname)
    if not args.only_test:
        writer.close()
