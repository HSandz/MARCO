import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import os


class BaseRecommender(nn.Module, ABC):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, device: str = 'cuda'):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.device = device if torch.cuda.is_available() else 'cpu'
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, user_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def predict_topk(self, user_ids: torch.Tensor, k: int = 20, 
                     exclude_history: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            scores = self.predict(user_ids)
            if exclude_history is not None:
                scores = scores.masked_fill(exclude_history.bool(), float('-inf'))
            topk_scores, topk_indices = torch.topk(scores, k, dim=-1)
        return topk_indices, topk_scores
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_users': self.num_users,
            'num_items': self.num_items,
            'embedding_dim': self.embedding_dim,
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        return self
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'embedding_dim': self.embedding_dim,
            'device': self.device,
        }
