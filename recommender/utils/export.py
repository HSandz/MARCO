import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from loguru import logger

from recommender.models.base_model import BaseRecommender


def export_recommendations(model: BaseRecommender, test_users: List[int],
                           user_history: Dict[int, List[int]], k: int = 20,
                           user_sequences: Optional[Dict[int, List[int]]] = None,
                           max_seq_len: int = 50, batch_size: int = 256,
                           output_path: Optional[str] = None) -> pd.DataFrame:
    model.eval()
    device = model.device
    
    results = []
    
    for start_idx in tqdm(range(0, len(test_users), batch_size), desc="Exporting recommendations"):
        batch_users = test_users[start_idx:start_idx + batch_size]
        user_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
        
        if user_sequences is not None:
            seqs = []
            for uid in batch_users:
                seq = user_sequences.get(uid, [])[-max_seq_len:]
                padded = [0] * (max_seq_len - len(seq)) + seq
                seqs.append(padded)
            seq_tensor = torch.tensor(seqs, dtype=torch.long, device=device)
            scores = model.predict(user_tensor, seq_tensor)
        else:
            scores = model.predict(user_tensor)
        
        for i, uid in enumerate(batch_users):
            user_scores = scores[i].clone()
            
            history = user_history.get(uid, [])
            if history:
                history_tensor = torch.tensor(history, device=device)
                valid_history = history_tensor[history_tensor < model.num_items]
                if len(valid_history) > 0:
                    user_scores[valid_history] = float('-inf')
            
            topk_scores, topk_indices = torch.topk(user_scores, k)
            topk_items = topk_indices.cpu().tolist()
            
            results.append({
                'user_id': uid,
                'recommendations': topk_items,
            })
    
    df = pd.DataFrame(results)
    
    if output_path:
        expanded_df = df.copy()
        for i in range(k):
            expanded_df[f'item_{i+1}'] = df['recommendations'].apply(lambda x: x[i] if i < len(x) else None)
        expanded_df = expanded_df.drop(columns=['recommendations'])
        expanded_df.to_csv(output_path, index=False)
        logger.info(f"Saved recommendations to {output_path}")
    
    return df


def export_for_marco(model: BaseRecommender, test_df: pd.DataFrame,
                     user_history: Dict[int, List[int]], k: int = 20,
                     user_sequences: Optional[Dict[int, List[int]]] = None,
                     user_history_ratings: Optional[Dict[int, List[float]]] = None,
                     max_seq_len: int = 50, batch_size: int = 256,
                     output_path: str = None) -> pd.DataFrame:
    model.eval()
    device = model.device
    
    results = []
    test_users = test_df['user_id'].unique().tolist()
    
    user_gt = {}
    for _, row in test_df.iterrows():
        user_gt[int(row['user_id'])] = int(row['item_id'])
    
    for start_idx in tqdm(range(0, len(test_users), batch_size), desc="Generating MARCO candidates"):
        batch_users = test_users[start_idx:start_idx + batch_size]
        user_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
        
        if user_sequences is not None:
            seqs = []
            for uid in batch_users:
                seq = user_sequences.get(uid, [])[-max_seq_len:]
                padded = [0] * (max_seq_len - len(seq)) + seq
                seqs.append(padded)
            seq_tensor = torch.tensor(seqs, dtype=torch.long, device=device)
            scores = model.predict(user_tensor, seq_tensor)
        else:
            scores = model.predict(user_tensor)
        
        for i, uid in enumerate(batch_users):
            user_scores = scores[i].clone()

            history = user_history.get(uid, [])

            if history:
                history_tensor = torch.tensor(history, device=device)
                valid_history = history_tensor[history_tensor < model.num_items]
                if len(valid_history) > 0:
                    user_scores[valid_history] = float('-inf')

            gt_item = user_gt.get(uid)

            topk_scores, topk_indices = torch.topk(user_scores, k)
            topk_items = topk_indices.cpu().tolist()

            np.random.shuffle(topk_items)

            clean_history = [item for item in history if item != gt_item]

            if user_history_ratings is not None and uid in user_history_ratings:
                history_ratings = user_history_ratings[uid]
                if len(history) != len(clean_history) and gt_item in history:
                    gt_idx = history.index(gt_item)
                    history_ratings = history_ratings[:gt_idx] + history_ratings[gt_idx+1:]

                if len(history_ratings) != len(clean_history):
                     if len(history_ratings) > len(clean_history):
                         history_ratings = history_ratings[:len(clean_history)]
                     else:
                         history_ratings = history_ratings + [1.0] * (len(clean_history) - len(history_ratings))
            else:
                history_ratings = [1] * len(clean_history)

            results.append({
                'user_id': uid,
                'item_id': gt_item,
                'rating': '',
                'history_item_id': str(clean_history),
                'history_rating': str(history_ratings),
                'candidate_item_id': str(topk_items),
            })
    
    df = pd.DataFrame(results)
    
    cols = ['user_id', 'item_id', 'rating', 'history_item_id', 'history_rating', 'candidate_item_id']
    df = df[cols]
    
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved MARCO candidates to {output_path}")
    
    return df


def compute_recall_at_k(model: BaseRecommender, test_users: List[int],
                        test_items: List[int], user_history: Dict[int, List[int]],
                        k_list: List[int] = [5, 10, 20],
                        user_sequences: Optional[Dict[int, List[int]]] = None,
                        max_seq_len: int = 50, batch_size: int = 256) -> Dict[str, float]:
    model.eval()
    device = model.device
    
    all_topk = {k: [] for k in k_list}
    max_k = max(k_list)
    
    for start_idx in range(0, len(test_users), batch_size):
        batch_users = test_users[start_idx:start_idx + batch_size]
        batch_items = test_items[start_idx:start_idx + batch_size]
        user_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
        
        if user_sequences is not None:
            seqs = []
            for uid in batch_users:
                seq = user_sequences.get(uid, [])[-max_seq_len:]
                padded = [0] * (max_seq_len - len(seq)) + seq
                seqs.append(padded)
            seq_tensor = torch.tensor(seqs, dtype=torch.long, device=device)
            scores = model.predict(user_tensor, seq_tensor)
        else:
            scores = model.predict(user_tensor)
        
        for i, uid in enumerate(batch_users):
            user_scores = scores[i].clone()
            
            history = user_history.get(uid, [])
            if history:
                history_tensor = torch.tensor(history, device=device)
                valid_history = history_tensor[history_tensor < model.num_items]
                if len(valid_history) > 0:
                    user_scores[valid_history] = float('-inf')
            
            _, topk_indices = torch.topk(user_scores, max_k)
            topk_items = topk_indices.cpu().tolist()
            
            for k in k_list:
                all_topk[k].append(batch_items[i] in topk_items[:k])
    
    metrics = {}
    for k in k_list:
        metrics[f'Recall@{k}'] = np.mean(all_topk[k])
    
    return metrics
