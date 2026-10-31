import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Callable
from tqdm import tqdm
import numpy as np
import os
from loguru import logger

from recommender.models.base_model import BaseRecommender


class Trainer:
    def __init__(self, model: BaseRecommender, learning_rate: float = 1e-3,
                 weight_decay: float = 0.0, patience: int = 10, 
                 checkpoint_dir: str = 'checkpoints', checkpoint_filename: str = 'best_model.pth'):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_filename = checkpoint_filename
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.best_metric = 0.0
        self.epochs_without_improvement = 0
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            loss = self.model.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, user_history: Dict[int, List[int]],
                 k_list: List[int] = [5, 10, 20]) -> Dict[str, float]:
        self.model.eval()

        max_k = max(k_list) if k_list else 0
        hits = {k: 0 for k in k_list}
        ndcg_sum = {k: 0.0 for k in k_list}
        total_targets = 0

        for batch in dataloader:
            user_ids = batch['user_id']

            if 'input_seq' in batch:
                input_seqs = batch['input_seq'].to(self.model.device)
                scores = self.model.predict(user_ids.to(self.model.device), input_seqs)
            else:
                scores = self.model.predict(user_ids.to(self.model.device))

            for i, uid in enumerate(user_ids.tolist()):
                history = user_history.get(uid, [])
                if history:
                    history_tensor = torch.tensor(history, device=self.model.device)
                    if hasattr(self.model, 'num_items'):
                        valid_history = history_tensor[history_tensor < self.model.num_items]
                        if len(valid_history) > 0:
                            scores[i, valid_history] = float('-inf')

            if max_k == 0:
                continue

            _, topk_indices = torch.topk(scores, max_k, dim=-1)
            topk_indices = topk_indices.cpu().tolist()

            if 'target_item' in batch:
                targets = batch['target_item'].tolist()
            elif 'pos_item' in batch:
                targets = batch['pos_item'].tolist()
            else:
                targets = []

            for i, target in enumerate(targets):
                total_targets += 1
                user_topk = topk_indices[i]
                for k in k_list:
                    topk_slice = user_topk[:k]
                    if target in topk_slice:
                        hits[k] += 1
                        rank = topk_slice.index(target)
                        ndcg_sum[k] += 1.0 / np.log2(rank + 2)

        metrics = {}
        for k in k_list:
            metrics[f'HR@{k}'] = hits[k] / total_targets if total_targets else 0.0
            metrics[f'NDCG@{k}'] = ndcg_sum[k] / total_targets if total_targets else 0.0

        return metrics
    
    def fit(self, train_loader: DataLoader, valid_loader: Optional[DataLoader] = None,
            user_history: Optional[Dict[int, List[int]]] = None, epochs: int = 100,
            eval_every: int = 1, metric_for_best: str = 'NDCG@10') -> Dict[str, List]:
        history = {'train_loss': [], 'valid_metrics': []}

        pbar = tqdm(range(epochs), desc="Training", ncols=160)
        last_metrics = {}

        for epoch in pbar:
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)

            display_dict = {'loss': f'{train_loss:.4f}'}
            display_dict.update(last_metrics)
            pbar.set_postfix(display_dict)

            if valid_loader is not None and user_history is not None and (epoch + 1) % eval_every == 0:
                metrics = self.evaluate(valid_loader, user_history)
                history['valid_metrics'].append(metrics)

                last_metrics = {k: f'{v:.4f}' for k, v in metrics.items()}
                display_dict = {'loss': f'{train_loss:.4f}'}
                display_dict.update(last_metrics)
                pbar.set_postfix(display_dict)

                current_metric = metrics.get(metric_for_best, 0.0)
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(self.checkpoint_filename)
                    tqdm.write(f"New best model! {metric_for_best}: {current_metric:.4f}")
                else:
                    self.epochs_without_improvement += 1

                if self.epochs_without_improvement >= self.patience:
                    tqdm.write(f"Early stopping after {epoch + 1} epochs")
                    break

        pbar.close()
        return history
    
    def save_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.model.get_config(),
        }, path)
    
    def load_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.model.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_metric = checkpoint.get('best_metric', 0.0)
