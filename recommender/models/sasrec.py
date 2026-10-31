import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional

from recommender.models.base_model import BaseRecommender


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = PointWiseFeedForward(hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)
        return x


class SASRec(BaseRecommender):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64,
                 max_seq_len: int = 50, num_blocks: int = 2, num_heads: int = 2,
                 dropout: float = 0.2, device: str = 'cuda'):
        super().__init__(num_users, num_items, embedding_dim, device)
        self.max_seq_len = max_seq_len
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout
        
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        self.emb_dropout = nn.Dropout(dropout)
        
        self.attention_blocks = nn.ModuleList([
            SelfAttentionBlock(embedding_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        
        self.final_norm = nn.LayerNorm(embedding_dim)
        
        self._init_weights()
        self.to(self.device)
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])
        nn.init.xavier_uniform_(self.position_embedding.weight)
    
    def _generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(self.device)
    
    def forward(self, input_seq: torch.Tensor, use_padding_mask: bool = True) -> torch.Tensor:
        input_seq = input_seq.to(self.device)
        batch_size, seq_len = input_seq.shape

        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)

        x = self.item_embedding(input_seq) * (self.embedding_dim ** 0.5)
        x = x + self.position_embedding(positions)
        x = self.emb_dropout(x)

        causal_mask = self._generate_causal_mask(seq_len)

        padding_mask = (input_seq == 0) if use_padding_mask else None

        for block in self.attention_blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=padding_mask)

        x = self.final_norm(x)
        return x
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_seq = batch['input_seq'].to(self.device)
        pos = batch['pos'].to(self.device)
        neg = batch['neg'].to(self.device)
        
        seq_output = self.forward(input_seq)
        
        pos_emb = self.item_embedding(pos)
        neg_emb = self.item_embedding(neg)
        
        pos_logits = (seq_output * pos_emb).sum(dim=-1)
        neg_logits = (seq_output * neg_emb).sum(dim=-1)
        
        istarget = (pos > 0).float()
        
        loss = (
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            -torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        )
        loss = loss.sum() / istarget.sum()
        
        return loss
    
    def predict(self, user_ids: torch.Tensor, input_seqs: Optional[torch.Tensor] = None) -> torch.Tensor:
        if input_seqs is None:
            raise ValueError("SASRec requires input_seqs for prediction")

        input_seqs = input_seqs.to(self.device)
        seq_output = self.forward(input_seqs, use_padding_mask=False)

        last_output = seq_output[:, -1, :]

        item_embs = self.item_embedding.weight
        scores = last_output @ item_embs.T
        return scores
    
    def predict_next(self, input_seq: torch.Tensor) -> torch.Tensor:
        input_seq = input_seq.to(self.device)
        if input_seq.dim() == 1:
            input_seq = input_seq.unsqueeze(0)
        
        seq_output = self.forward(input_seq)
        last_output = seq_output[:, -1, :]
        
        item_embs = self.item_embedding.weight[1:]
        scores = last_output @ item_embs.T
        return scores
    
    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            'max_seq_len': self.max_seq_len,
            'num_blocks': self.num_blocks,
            'num_heads': self.num_heads,
            'dropout': self.dropout_rate,
        })
        return config
