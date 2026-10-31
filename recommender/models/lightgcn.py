import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import scipy.sparse as sp

from recommender.models.base_model import BaseRecommender
from recommender.utils.data_utils import sparse_to_torch


class LightGCN(BaseRecommender):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, 
                 num_layers: int = 3, reg_weight: float = 1e-4, dropout: float = 0.0, device: str = 'cuda'):
        super().__init__(num_users, num_items, embedding_dim, device)
        self.num_layers = num_layers
        self.reg_weight = reg_weight
        self.dropout = dropout
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        self.adj_matrix = None
        self.to(self.device)
    
    def set_adj_matrix(self, adj_matrix: sp.csr_matrix):
        adj_tensor = sparse_to_torch(adj_matrix)
        self.adj_matrix = adj_tensor.to(self.device)
    
    def _sparse_dropout(self, x: torch.Tensor, noise_shape: int) -> torch.Tensor:
        random_tensor = 1 - self.dropout
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        
        out = torch.sparse_coo_tensor(i, v, x.shape, device=x.device)
        return out * (1. / (1 - self.dropout))
    
    def forward(self):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings_list = [all_embeddings]
        
        if self.dropout > 0 and self.training and self.adj_matrix is not None:
             adj = self._sparse_dropout(self.adj_matrix, self.adj_matrix._nnz())
        else:
             adj = self.adj_matrix

        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        all_embeddings = torch.stack(embeddings_list, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items])
        return user_embeddings, item_embeddings
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        user_ids = batch['user_id'].to(self.device)
        pos_items = batch['pos_item'].to(self.device)
        neg_items = batch['neg_item'].to(self.device)
        
        user_emb, item_emb = self.forward()
        
        u_emb = user_emb[user_ids]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        pos_scores = (u_emb * pos_emb).sum(dim=-1)
        neg_scores = (u_emb * neg_emb).sum(dim=-1)
        
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        reg_loss = self.reg_weight * (
            self.user_embedding.weight[user_ids].norm(2).pow(2) +
            self.item_embedding.weight[pos_items].norm(2).pow(2) +
            self.item_embedding.weight[neg_items].norm(2).pow(2)
        ) / user_ids.shape[0]
        
        return bpr_loss + reg_loss
    
    def predict(self, user_ids: torch.Tensor) -> torch.Tensor:
        user_ids = user_ids.to(self.device)
        user_emb, item_emb = self.forward()
        u_emb = user_emb[user_ids]
        scores = u_emb @ item_emb.T
        return scores
    
    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'reg_weight': self.reg_weight,
            'dropout': self.dropout
        })
        return config
