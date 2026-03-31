import torch
import torch.nn as nn
from torch import nn
from torch_geometric.utils import dropout_adj
import torch.nn.functional as F
from transformer import Transformer
from models.gae import IGAE_encoder, IGAE_decoder


# mask feature function
def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


class STransformer(nn.Module):
    def __init__(
        self,
        d_token = 256, # Token dim 
        d_emd = 128, # Lattent dim
        d_input = 4221,
        d_output = 64,
        q = 128,
        h = 4,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        v = 128,
        N = 2,
        attention_size = None,
        dropout = 0.2,
        chunk_mode = None,
        pe = None,
        drop_feature_rate = 0.05,
        drop_edge_rate = 0.05,
        tile_shape = 2048 
        ):
        super(STransformer, self).__init__()

        self.d_input = d_input
        self.d_token = d_token
        self.d_emd = d_emd
        self.d_output = d_output
        self.tile_shape = tile_shape

        self.drop_feature_rate = drop_feature_rate
        self.drop_edge_rate = drop_edge_rate

        # --> graph encoder (gene)
        self.g_encoder = IGAE_encoder(
            gae_n_enc_1=1024,
            gae_n_enc_2=512,
            gae_n_enc_3=self.d_token,
            n_input=self.d_input)

        # ---> graph decoder (gene)
        self.g_decoder = IGAE_decoder(
            gae_n_dec_1=self.d_output,
            gae_n_dec_2=512,
            gae_n_dec_3=1024,
            n_input=self.d_input)

        # ---> graph encoder (tile)
        self.t_encoder = IGAE_encoder(
            gae_n_enc_1=1024,
            gae_n_enc_2=512,
            gae_n_enc_3=self.d_token,
            n_input=self.tile_shape) # tile_shape

        # --->graph decoder (tile)
        self.t_decoder = IGAE_decoder(
            gae_n_dec_1=self.d_output,
            gae_n_dec_2=512,
            gae_n_dec_3=1024,
            n_input=self.tile_shape)

        # --> transformer
        self.tsf = Transformer(self.d_token, self.d_emd, self.d_output, q, v, h, N, attention_size=attention_size,
                  dropout=dropout, chunk_mode=chunk_mode, pe=pe)


    def forward(self, tile, expression, adj):
        
        # --> get z_gene
        edge_index = adj.nonzero(as_tuple=False).t()
        adj_1 = dropout_adj(edge_index, p=self.drop_edge_rate)[0]
        dropped_adj_1 = torch.zeros_like(adj)
        dropped_adj_1[adj_1[0], adj_1[1]] = 1 
        x_1 = drop_feature(expression, self.drop_feature_rate)
        z_gene, z_gene_adj = self.g_encoder(x_1, dropped_adj_1)

        # --> get z_tile
        edge_index = adj.nonzero(as_tuple=False).t()
        adj_2 = dropout_adj(edge_index, p=self.drop_edge_rate)[0]
        dropped_adj_2 = torch.zeros_like(adj)
        dropped_adj_2[adj_2[0], adj_2[1]] = 1
        x_2= drop_feature(tile.float(), self.drop_feature_rate)
        z_tile, z_tile_adj = self.t_encoder(x_2, dropped_adj_2)

        # --> cat all embeddins
        tokens = torch.cat([z_gene, z_tile], dim=0)
        tokens = tokens.unsqueeze(0)
        
        # --> transformer
        out = self.tsf(tokens)
        out = out.squeeze(0)

        # --> reconstruct expression
        expr_emb = out[:expression.shape[0]]
        recon_expression, recon_expression_adj = self.g_decoder(expr_emb, adj)

        # --> reconstruct tile
        tile_emb = out[expression.shape[0]:]
        recon_tile, recon_tile_adj = self.t_decoder(tile_emb, adj)

        # --> get mean embedding
        mm_embed = (expr_emb + tile_emb) / 2

        return mm_embed, recon_expression, recon_tile # new version

    
    def get_atten_score(self):
        "get omics atten score"
        return self.tsf.get_enc_dec_attention_map()



