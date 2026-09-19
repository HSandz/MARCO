import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from recommender.models.base_model import BaseRecommender


class BERTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask, 
                                      key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class BERT4Rec(BaseRecommender):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64,
                 max_seq_len: int = 50, num_blocks: int = 2, num_heads: int = 2,
                 dropout: float = 0.2, mask_prob: float = 0.2, device: str = 'cuda'):
        super().__init__(num_users, num_items, embedding_dim, device)
        self.max_seq_len = max_seq_len
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.mask_prob = mask_prob
        
        self.mask_token = num_items + 1
        vocab_size = num_items + 2
        
        self.item_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)
        
        self.transformer_blocks = nn.ModuleList([
            BERTBlock(embedding_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        
        self.transform_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim)
        )
        
        self.output_bias = nn.Parameter(torch.zeros(num_items + 1))
        
        self._init_weights()
        self.to(self.device)
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])
        nn.init.xavier_uniform_(self.position_embedding.weight)
        for m in self.transform_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_seq.shape
        
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        x = self.item_embedding(input_seq) + self.position_embedding(positions)
        x = self.emb_norm(x)
        x = self.emb_dropout(x)
        
        padding_mask = (input_seq == 0)
        
        for block in self.transformer_blocks:
            x = block(x, key_padding_mask=padding_mask)
        
        return x
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_seq = batch['input_seq'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        hidden = self.forward(input_seq)
        hidden = self.transform_layer(hidden)
        
        item_emb = self.item_embedding.weight[:-1]
        logits = hidden @ item_emb.T + self.output_bias
        
        logits = logits.view(-1, self.num_items + 1)
        labels = labels.view(-1)
        
        mask = labels > 0
        if not mask.any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        logits = logits[mask]
        labels = labels[mask]
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def predict(self, user_ids: torch.Tensor, input_seqs: Optional[torch.Tensor] = None) -> torch.Tensor:
        if input_seqs is None:
            raise ValueError("BERT4Rec requires input_seqs for prediction")
        
        input_seqs = input_seqs.to(self.device)
        batch_size, seq_len = input_seqs.shape
        
        masked_seqs = input_seqs.clone()
        masked_seqs = torch.roll(masked_seqs, shifts=-1, dims=1)
        masked_seqs[:, -1] = self.mask_token

        
        hidden = self.forward(masked_seqs)
        hidden = self.transform_layer(hidden)
        last_hidden = hidden[:, -1, :]
        
        item_emb = self.item_embedding.weight[:-1]
        scores = last_hidden @ item_emb.T + self.output_bias
        return scores
    
    def predict_masked(self, input_seq: torch.Tensor, mask_positions: torch.Tensor) -> torch.Tensor:
        input_seq = input_seq.to(self.device)
        if input_seq.dim() == 1:
            input_seq = input_seq.unsqueeze(0)
        
        hidden = self.forward(input_seq)
        hidden = self.transform_layer(hidden)
        
        batch_size = hidden.shape[0]
        masked_hidden = hidden[torch.arange(batch_size), mask_positions]
        
        item_emb = self.item_embedding.weight[:-1]
        scores = masked_hidden @ item_emb.T + self.output_bias
        return scores
    
    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            'max_seq_len': self.max_seq_len,
            'num_blocks': self.num_blocks,
            'num_heads': self.num_heads,
            'dropout': self.dropout_rate,
            'mask_prob': self.mask_prob,
        })
        return config
