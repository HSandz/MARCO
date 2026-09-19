import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import scipy.sparse as sp
from collections import defaultdict


def load_interactions(filepath: str) -> Tuple[pd.DataFrame, int, int]:
    df = pd.read_csv(filepath)
    num_users = df['user_id'].max() + 1
    num_items = df['item_id'].max() + 1
    return df, num_users, num_items


def build_user_item_dict(df: pd.DataFrame) -> Dict[int, List[int]]:
    user_items = defaultdict(list)
    for _, row in df.iterrows():
        user_items[int(row['user_id'])].append(int(row['item_id']))
    return dict(user_items)


def ensure_filtered_dataset(source_path: str, output_dir: str = 'recommender/datasets', threshold: float = 4.0, min_interactions: int = 3) -> str:
    import os
    
    dataset_name = os.path.basename(os.path.dirname(source_path))
    filename = os.path.basename(source_path)
    
    target_dir = os.path.join(output_dir, dataset_name)
    target_path = os.path.join(target_dir, filename)
    
    if os.path.exists(target_path):
        return target_path
        
    os.makedirs(target_dir, exist_ok=True)
    
    logger = None
    try:
        from loguru import logger
        logger.info(f"Filtering dataset {source_path} (rating >= {threshold})...")
    except:
        print(f"Filtering dataset {source_path} (rating >= {threshold})...")
        
    df = pd.read_csv(source_path)
    
    if 'rating' in df.columns:
        filtered_df = df[df['rating'] >= threshold].copy()
        
        user_counts = filtered_df.groupby('user_id').size()
        valid_users = user_counts[user_counts >= min_interactions].index
        filtered_df = filtered_df[filtered_df['user_id'].isin(valid_users)]
        
        filtered_df.to_csv(target_path, index=False)
        if logger:
            logger.info(f"Saved filtered dataset to {target_path} ({len(filtered_df)}/{len(df)} rows, {len(valid_users)} users with >= {min_interactions} interactions)")
        else:
            print(f"Saved filtered dataset to {target_path} ({len(filtered_df)}/{len(df)} rows, {len(valid_users)} users with >= {min_interactions} interactions)")
    else:
        if logger:
            logger.warning("No rating column found, copying original file.")
        df.to_csv(target_path, index=False)
        
    return target_path


def build_user_history_with_ratings(df: pd.DataFrame) -> Dict[int, Tuple[List[int], List[float]]]:
    user_history = defaultdict(lambda: ([], []))
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    
    for _, row in df.iterrows():
        uid = int(row['user_id'])
        iid = int(row['item_id'])
        rating = float(row['rating']) if 'rating' in row and pd.notna(row['rating']) else 1.0
        
        user_history[uid][0].append(iid)
        user_history[uid][1].append(rating)
    
    return dict(user_history)


def build_sparse_graph(df: pd.DataFrame, num_users: int, num_items: int) -> sp.csr_matrix:
    rows = df['user_id'].values
    cols = df['item_id'].values
    data = np.ones(len(rows))
    user_item_matrix = sp.csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
    return user_item_matrix


def normalize_adj_matrix(adj: sp.csr_matrix) -> sp.csr_matrix:
    num_users, num_items = adj.shape
    adj_full = sp.vstack([
        sp.hstack([sp.csr_matrix((num_users, num_users)), adj]),
        sp.hstack([adj.T, sp.csr_matrix((num_items, num_items))])
    ])
    rowsum = np.array(adj_full.sum(1)).flatten()
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt @ adj_full @ d_mat_inv_sqrt
    return norm_adj.tocsr()


def sparse_to_torch(sparse_mx: sp.csr_matrix) -> torch.sparse.FloatTensor:
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def leave_one_out_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(['user_id', 'timestamp'])
    train_list, valid_list, test_list = [], [], []
    has_rating = 'rating' in df.columns
    has_timestamp = 'timestamp' in df.columns
    
    for user_id, group in df.groupby('user_id'):
        items = group['item_id'].tolist()
        ratings = group['rating'].tolist() if has_rating else [1.0] * len(items)
        timestamps = group['timestamp'].tolist() if has_timestamp else [0] * len(items)
        
        if len(items) >= 3:
            for i in range(len(items) - 2):
                train_list.append((user_id, items[i], ratings[i], timestamps[i]))
            valid_list.append((user_id, items[-2], ratings[-2], timestamps[-2]))
            test_list.append((user_id, items[-1], ratings[-1], timestamps[-1]))
        elif len(items) == 2:
            train_list.append((user_id, items[0], ratings[0], timestamps[0]))
            test_list.append((user_id, items[1], ratings[1], timestamps[1]))
        else:
            train_list.append((user_id, items[0], ratings[0], timestamps[0]))
    
    cols = ['user_id', 'item_id', 'rating', 'timestamp']
    train_df = pd.DataFrame(train_list, columns=cols)
    valid_df = pd.DataFrame(valid_list, columns=cols) if valid_list else None
    test_df = pd.DataFrame(test_list, columns=cols)
    
    if not has_timestamp:
        train_df.drop(columns=['timestamp'], inplace=True)
        if valid_df is not None: valid_df.drop(columns=['timestamp'], inplace=True)
        test_df.drop(columns=['timestamp'], inplace=True)

    return train_df, valid_df, test_df


class BPRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, num_items: int, user_items: Dict[int, List[int]], num_negatives: int = 1):
        self.interactions = df[['user_id', 'item_id']].values
        self.num_items = num_items
        self.user_items = user_items
        self.num_negatives = num_negatives
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user_id, pos_item = self.interactions[idx]
        user_id, pos_item = int(user_id), int(pos_item)
        neg_items = []
        user_history = set(self.user_items.get(user_id, []))
        while len(neg_items) < self.num_negatives:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in user_history and neg_item not in neg_items:
                neg_items.append(neg_item)
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'pos_item': torch.tensor(pos_item, dtype=torch.long),
            'neg_item': torch.tensor(neg_items[0], dtype=torch.long),
        }


class SequenceDataset(Dataset):
    def __init__(self, user_sequences: Dict[int, List[int]], num_items: int, max_len: int = 50, 
                 mask_prob: float = 0.0, mask_token: int = None):
        self.user_ids = list(user_sequences.keys())
        self.sequences = user_sequences
        self.num_items = num_items
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token if mask_token is not None else num_items
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        
        if self.mask_prob > 0:
            seq = self.sequences[user_id][-self.max_len:]
            seq_len = len(seq)
            padded_seq = [0] * (self.max_len - seq_len) + seq
            
            masked_seq = padded_seq.copy()
            labels = [0] * self.max_len
            for i in range(self.max_len - seq_len, self.max_len):
                if np.random.random() < self.mask_prob:
                    labels[i] = masked_seq[i]
                    masked_seq[i] = self.mask_token
            return {
                'user_id': torch.tensor(user_id, dtype=torch.long),
                'input_seq': torch.tensor(masked_seq, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
            }
        else:
            seq = self.sequences[user_id][-(self.max_len + 1):]
            seq_len = len(seq)
            padded_seq = [0] * (self.max_len + 1 - seq_len) + seq
            
            input_seq = padded_seq[:-1]
            target = padded_seq[1:]
            
            return {
                'user_id': torch.tensor(user_id, dtype=torch.long),
                'input_seq': torch.tensor(input_seq, dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.long),
                'target_item': torch.tensor(seq[-1] if seq else 0, dtype=torch.long),
            }


class SASRecDataset(Dataset):
    def __init__(self, user_sequences: Dict[int, List[int]], num_items: int, max_len: int = 50):
        self.user_ids = list(user_sequences.keys())
        self.sequences = user_sequences
        self.num_items = num_items
        self.max_len = max_len
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        seq = self.sequences[user_id]
        
        seq = seq[-(self.max_len + 1):]
        seq_len = len(seq)
        padded_seq = [0] * (self.max_len + 1 - seq_len) + seq
        
        input_seq = padded_seq[:-1]
        pos = padded_seq[1:]
        
        user_set = set(self.sequences[user_id])
        neg = []
        for s in pos:
            if s == 0:
                neg.append(0)
            else:
                t = np.random.randint(1, self.num_items + 1)
                while t in user_set:
                    t = np.random.randint(1, self.num_items + 1)
                neg.append(t)
        
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'input_seq': torch.tensor(input_seq, dtype=torch.long),
            'pos': torch.tensor(pos, dtype=torch.long),
            'neg': torch.tensor(neg, dtype=torch.long),
        }


def build_sequences(df: pd.DataFrame) -> Dict[int, List[int]]:
    df = df.sort_values(['user_id', 'timestamp'])
    sequences = defaultdict(list)
    for _, row in df.iterrows():
        sequences[int(row['user_id'])].append(int(row['item_id']))
    return dict(sequences)


class EvalDataset(Dataset):
    def __init__(self, eval_df: pd.DataFrame, user_sequences: Dict[int, List[int]], 
                 max_len: int = 50):
        self.eval_users = eval_df['user_id'].tolist()
        self.eval_items = eval_df['item_id'].tolist()
        self.user_sequences = user_sequences
        self.max_len = max_len
    
    def __len__(self):
        return len(self.eval_users)
    
    def __getitem__(self, idx):
        user_id = self.eval_users[idx]
        target_item = self.eval_items[idx]
        
        seq = self.user_sequences.get(user_id, [])[-self.max_len:]
        seq_len = len(seq)
        padded_seq = [0] * (self.max_len - seq_len) + seq
        
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'input_seq': torch.tensor(padded_seq, dtype=torch.long),
            'target_item': torch.tensor(target_item, dtype=torch.long),
        }


def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
