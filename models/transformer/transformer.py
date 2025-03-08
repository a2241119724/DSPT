import torch
import copy

from torch import nn
from models.containers import ModuleList
from ..captioning_model import CaptioningModel


class Transformer(CaptioningModel):
    def __init__(self, bos_idx:int, encoder, decoder):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder

        self.register_state('enc_output', None)
        # self.register_state('mask_enc', None)
        # self.dropout = nn.Dropout(0.1)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for n, p in self.named_parameters():
            if ".bias" in n:
                nn.init.constant_(p, 0.)  
            elif "word_emb" in n:
                nn.init.xavier_uniform_(p)
                p.data[1] = torch.zeros((p.shape[-1])).to(p.device)
            elif "pos_emb" in n:
                pass
            elif "scale" in n:
                pass
            else:
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)

    def forward(self, regions, grids, boxes, sizes, seq, *args):
        enc_output = self.encoder(regions, grids, boxes, sizes)
        dec_output = self.decoder(seq, enc_output)
        return dec_output, enc_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
            None, None]

    def step(self, t, prev_output, regions, grids, boxes, sizes, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                self.enc_output = self.encoder(regions, grids, boxes, sizes)
                if isinstance(grids, torch.Tensor):
                    it = grids.data.new_full((grids.shape[0], 1), self.bos_idx).long()
                else:
                    it = grids[0].data.new_full((grids[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, self.enc_output), self.enc_output


class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, regions, grids, boxes, sizes, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        encoder_output_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, regions, grids, boxes, sizes, mode, **kwargs)
            out_ensemble.append(out_i[0].unsqueeze(0))
            encoder_output_ensemble.append(out_i[1].unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0), torch.mean(torch.cat(encoder_output_ensemble, 0), dim=0)