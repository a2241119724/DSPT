from models.transformer.utils import PositionWiseFeedForward
from torch import nn
from models.transformer.attention import MultiHeadAttention_encoder, MultiHeadAttention

class EncoderFusion(nn.Module):
    def __init__(self, N:int, d_model:int=512, d_k:int=128, d_v:int=128, h:int=4, d_ff:int=2048, grid_count:int=49, dropout:float=.1):
        super(EncoderFusion, self).__init__()
        self.N = N

        # self.mhatt = nn.ModuleList([MultiHeadAttention(d_model, d_k, d_v, h, grid_count, dropout) for _ in range(N)])
        self.mhatt = nn.ModuleList([MultiHeadAttention_encoder(d_model, d_k, d_v, h, grid_count, dropout) for _ in range(N)])
        self.pwff = nn.ModuleList([PositionWiseFeedForward(d_model, d_ff, dropout) for _ in range(N)])
        self.ln = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(N)])
        self.drop = nn.ModuleList([nn.Dropout(dropout) for _ in range(N)])

    def forward(self, regions, grids, boxes, sizes, attention_weights):
        for i in range(self.N):
            att = self.mhatt[i](grids, grids, grids, None, attention_weights)
            att = self.ln[i](grids + self.drop[i](att))
            grids = self.pwff[i](att)
        return grids

class MultiLevelEncoder(nn.Module):
    def __init__(self, N:int, d_model:int=512, d_k:int=128, d_v:int=128, h:int=4, d_ff:int=2048, grid_count:int=49, dropout:float=.1, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.encoderFusion = EncoderFusion(N, d_model, d_k, d_v, h, d_ff, grid_count, dropout)

    def forward(self, regions, grids, boxes, sizes, attention_weights=None):
        return self.encoderFusion(regions, grids, boxes, sizes, attention_weights)

class Encoder(MultiLevelEncoder):
    def __init__(self, N:int, d_in:int=2048, **kwargs):
        super().__init__(N, **kwargs)
        self.grid_proj = nn.Sequential(
            nn.Linear(d_in, self.d_model),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.LayerNorm(self.d_model)
        )

    def forward(self, regions, grids, boxes, sizes, attention_weights=None):
        grids = self.grid_proj(grids)
        return super(Encoder, self).forward(regions, grids, boxes, sizes, attention_weights=attention_weights)
