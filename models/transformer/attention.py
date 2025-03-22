import numpy as np
import torch
import matplotlib.pyplot as plt

from torch import nn
from models.containers import Module
from torch.nn import functional as F


class ScaledDotProductAttention_encoder(nn.Module):
    isVisual = False
    layer = 0

    def __init__(self, d_model:int, d_k:int, d_v:int, h:int, grid_count:int, dropout:float=.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones((1,h,grid_count,grid_count)) * (1. / np.sqrt(d_k)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = values.shape[1]
        q_sig = self.fc_q(queries)
        q = q_sig.unsqueeze(2).expand(-1, -1, nk, -1)
        k = self.fc_k(keys).unsqueeze(1).expand(-1, nq, -1, -1)
        att = q * k
        sig = torch.sigmoid(att)
        v = (sig * self.fc_v(values).unsqueeze(1).expand(-1, nq, -1, -1)).view(b_s, nq, nk, self.h, self.d_k).permute(0, 3, 1, 2, 4)
        att = att.view(b_s, nq, nk, self.h, self.d_k).permute(0, 3, 1, 2, 4)
        att = att.sum(-1) * self.scale
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask.bool(), -np.inf)
        if(ScaledDotProductAttention_encoder.isVisual):
            self.visual(F.softmax(att, dim=-1).squeeze(0).squeeze(-2).mean(0).cpu())
        att = self.dropout(F.softmax(att, dim=-1)).unsqueeze(-2)
        image = (att @ v).squeeze(-2).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(image)
        out = torch.sigmoid(q_sig * out) * out
        return out
    
    def visual(self, att):
        ScaledDotProductAttention_encoder.layer = ScaledDotProductAttention_encoder.layer + 1
        if ScaledDotProductAttention_encoder.layer != 6: return
        for i in range(49):
            fig, ax = plt.subplots(figsize=(7, 7))
            cax = ax.matshow(att[i].reshape(7,7), cmap='viridis')
            ax.set_xticks(range(7))
            ax.set_yticks(range(7))
            ax.set_xticklabels([f"{i}" for i in range(7)])
            ax.set_yticklabels([f"{i}" for i in range(7)])
            plt.colorbar(cax)
            plt.savefig('./output/heatmap'+"_"+str(i)+'.png', bbox_inches='tight')
            plt.close()

class MultiHeadAttention_encoder(Module):

    def __init__(self, d_model:int, d_k:int, d_v:int, h:int, grid_count:int, dropout:float=.1, can_be_stateful=False,
        attention_module=None, attention_module_kwargs=None):
        super().__init__()
        self.attention = ScaledDotProductAttention_encoder(d_model=d_model, d_k=d_k, d_v=d_v, h=h, grid_count=grid_count)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys
            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values
        out = self.attention(queries, keys, values, attention_mask, attention_weights)
        out = self.dropout(out)
        out = self.layer_norm(queries + out)
        return out

class ScaledDotProductAttention_decoder_self(nn.Module):

    def __init__(self, d_model:int, d_k:int, d_v:int, h:int, dropout:float=.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        att = q @ k / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask.bool(), -np.inf)
        att = self.dropout(torch.softmax(att, -1))

        out = (att @ v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)
        return out

class MultiHeadAttention_decoder_self(Module):

    def __init__(self, d_model:int, d_k:int, d_v:int, h:int, dropout:float=.1, can_be_stateful=False,
        attention_module=None, attention_module_kwargs=None):
        super().__init__()
        self.attention = ScaledDotProductAttention_decoder_self(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys
            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values
        out = self.attention(queries, keys, values, attention_mask, attention_weights)
        out = self.dropout(out)
        out = self.layer_norm(queries + out)
        return out

class ScaledDotProductAttention_decoder_cross(nn.Module):

    def __init__(self, d_model:int, d_k:int, d_v:int, h:int, dropout:float=.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.threshold = nn.Parameter(torch.tensor([0.1]), requires_grad=False)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        att = q @ k / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask.bool(), -np.inf)
        mask = torch.topk(torch.softmax(att[:,:,:,(att.shape[3]-49):], -1), 5, -1)[0].sum(-1) > self.threshold
        if self.threshold < 0.8:
            self.threshold.data = self.threshold + mask.sum() * 1e-8
        att[:, :, :, :(att.shape[3]-49)] = att[:, :, :, :(att.shape[3]-49)].masked_fill(mask.unsqueeze(-1), -np.inf)
        att = self.dropout(torch.softmax(att, -1))

        out = (att @ v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)
        return out

class MultiHeadAttention_decoder_cross(Module):

    def __init__(self, d_model:int, d_k:int, d_v:int, h:int, dropout:float=.1, can_be_stateful=False,
        attention_module=None, attention_module_kwargs=None):
        super().__init__()
        self.attention = ScaledDotProductAttention_decoder_cross(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys
            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values
        out = self.attention(queries, keys, values, attention_mask, attention_weights)
        out = self.dropout(out)
        out = self.layer_norm(queries + out)
        return out
    
class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_model:int, d_k:int, d_v:int, h:int, dropout:float=.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        att = q @ k / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask.bool(), -np.inf)
        att = self.dropout(torch.softmax(att, -1))

        out = (att @ v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)
        return out

class MultiHeadAttention(Module):

    def __init__(self, d_model:int, d_k:int, d_v:int, h:int, grid_count:int, dropout:float=.1, can_be_stateful=False,
        attention_module=None, attention_module_kwargs=None):
        super().__init__()
        self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys
            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values
        out = self.attention(queries, keys, values, attention_mask, attention_weights)
        out = self.dropout(out)
        out = self.layer_norm(queries + out)
        return out