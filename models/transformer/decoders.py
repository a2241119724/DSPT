import torch

from torch import nn
from torch.nn import functional as F
from models.transformer.attention import MultiHeadAttention_decoder_self, MultiHeadAttention_decoder_cross
from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
from models.containers import Module, ModuleList


class DecoderLayer(Module):
    def __init__(self, d_model:int=512, d_k:int=128, d_v:int=128, h:int=4, d_ff:int=2048, dropout:float=.1):
        super().__init__()
        self.self_att = MultiHeadAttention_decoder_self(d_model, d_k, d_v, h, dropout, can_be_stateful=True)
        self.enc_att = MultiHeadAttention_decoder_cross(d_model, d_k, d_v, h, dropout, can_be_stateful=False)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.cls = nn.Parameter(torch.randn((1, 1, d_model)))

    def forward(self, input, enc_output, mask_pad, mask_self_att):
        enc_output = torch.cat([self.cls.expand(enc_output.size(0), -1, -1), enc_output], 1)
        enc_att = self.enc_att(input, enc_output, enc_output) * mask_pad
        self_att = self.self_att(enc_att, enc_att, enc_att, mask_self_att) * mask_pad
        ff = self.pwff(self_att) * mask_pad
        return ff

class Decoder(Module):
    def __init__(self, vocab_size:int, max_len:int, N_dec:int, padding_idx:int, d_model:int=512, d_k:int=128, 
        d_v:int=128, h:int=4, d_ff:int=2048, dropout:float=.1):
        super().__init__()
        self.padding_idx = padding_idx

        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, enc_output, enc_mask=None):
        # input (b_s, seq_len)
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
            diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        # out = self.word_emb.weight[input] + self.pos_emb(seq)
        out = self.word_emb(input) + self.pos_emb(seq)

        for l in self.layers:
            out = l(out, enc_output, mask_queries, mask_self_attention)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)
