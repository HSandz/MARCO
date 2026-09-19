import argparse
import os
import torch
import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List

from recommender.models.lightgcn import LightGCN
from recommender.models.sasrec import SASRec
from recommender.models.bert4rec import BERT4Rec
from recommender.utils.data_utils import (
    load_interactions, build_user_item_dict, build_sparse_graph,
    normalize_adj_matrix, leave_one_out_split, build_sequences
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate recommender model on N samples')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (e.g., recommender/checkpoints/lightgcn_ml-100k.pth)')
    parser.add_argument('--num_samples', type=int, required=True,
                        help='Number of users (samples) to evaluate')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to filtered dataset. If not provided, will auto-detect from checkpoint name.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--k_list', type=int, nargs='+', default=[3, 5, 10, 20],
                        help='List of K values for evaluation (default: 5 10 20)')
    parser.add_argument('--last', action='store_true',
                        help='Evaluate last N samples instead of first N')
    return parser.parse_args()


@torch.no_grad()
def evaluate_samples(model, test_users: List[int], test_items: List[int],
                     user_history: Dict[int, List[int]],
                     user_sequences: Dict[int, List[int]] = None,
                     max_seq_len: int = 50, batch_size: int = 256,
                     k_list: List[int] = [5, 10, 20]) -> Dict[str, float]:
    device = model.device

    all_topk = {k: [] for k in k_list}
    all_ndcg = {k: 0.0 for k in k_list}
    max_k = max(k_list)

    logger.info(f"Evaluating {len(test_users)} users...")

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

            target_item = batch_items[i]

            for k in k_list:
                all_topk[k].append(target_item in topk_items[:k])

                if target_item in topk_items[:k]:
                    rank = topk_items[:k].index(target_item)
                    all_ndcg[k] += 1.0 / np.log2(rank + 2)

    num_samples = len(test_users)
    metrics = {}
    for k in k_list:
        metrics[f'Recall@{k}'] = np.mean(all_topk[k])
        metrics[f'NDCG@{k}'] = all_ndcg[k] / num_samples

    return metrics


def main():
    args = parse_args()

    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.remove()
    logger.add(lambda msg: print(msg, end=''), level="INFO", format="{message}")

    logger.info(f"Loading checkpoint from {args.checkpoint}...\n")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model_config = checkpoint['config']

    num_users = int(model_config['num_users'])
    num_items = int(model_config['num_items'])

    checkpoint_name = os.path.basename(args.checkpoint)
    if 'lightgcn' in checkpoint_name.lower():
        model_type = 'lightgcn'
    elif 'sasrec' in checkpoint_name.lower():
        model_type = 'sasrec'
    elif 'bert4rec' in checkpoint_name.lower():
        model_type = 'bert4rec'
    else:
        logger.error("Error: Cannot detect model type from checkpoint name. Expected 'lightgcn', 'sasrec', or 'bert4rec' in filename.")
        return

    logger.info(f"Detected model type: {model_type}")

    if args.data is None:
        dataset_name = checkpoint_name.replace('.pth', '').replace(f'{model_type}_', '')

        possible_paths = [
            f'recommender/datasets/{dataset_name}/all.csv',
            f'recommender/datasets/{dataset_name}/train.csv',
            f'data/{dataset_name}/all.csv',
        ]

        for path in possible_paths:
            if os.path.exists(path):
                args.data = path
                logger.info(f"Auto-detected dataset: {args.data}\n")
                break

        if args.data is None:
            logger.error(f"Error: Could not auto-detect dataset for '{dataset_name}'")
            logger.error(f"Tried paths: {possible_paths}")
            logger.error(f"Please provide --data argument explicitly.")
            return

    logger.info(f"Loading data from {args.data}...\n")
    df, _, _ = load_interactions(args.data)
    train_df, valid_df, test_df = leave_one_out_split(df)

    train_valid_df = pd.concat([train_df, valid_df]) if valid_df is not None else train_df

    from recommender.utils.data_utils import build_user_history_with_ratings
    train_valid_history_dict = build_user_history_with_ratings(train_valid_df)
    user_history = {u: h[0] for u, h in train_valid_history_dict.items()}

    all_test_users = test_df['user_id'].unique().tolist()

    user_gt = {}
    for _, row in test_df.iterrows():
        user_gt[int(row['user_id'])] = int(row['item_id'])

    num_samples = min(args.num_samples, len(all_test_users))
    if args.last:
        test_users = all_test_users[-num_samples:]
    else:
        test_users = all_test_users[:num_samples]
    test_items = [user_gt[uid] for uid in test_users]

    logger.info(f"Total test users: {len(all_test_users)}")
    if args.last:
        logger.info(f"Evaluating last {num_samples} users\n")
    else:
        logger.info(f"Evaluating first {num_samples} users\n")

    if model_type == 'lightgcn':
        model = LightGCN(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=model_config.get('embedding_dim', 64),
            num_layers=model_config.get('num_layers', 3),
            reg_weight=model_config.get('reg_weight', 1e-4),
            dropout=model_config.get('dropout', 0.2),
            device=args.device
        )
        adj_matrix = build_sparse_graph(train_df, num_users, num_items)
        norm_adj = normalize_adj_matrix(adj_matrix)
        model.set_adj_matrix(norm_adj)
        user_sequences = None
        max_seq_len = 50

    elif model_type == 'sasrec':
        max_seq_len = model_config.get('max_seq_len', 50)
        model = SASRec(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=model_config.get('embedding_dim', 64),
            max_seq_len=max_seq_len,
            num_blocks=model_config.get('num_blocks', 2),
            num_heads=model_config.get('num_heads', 2),
            dropout=model_config.get('dropout', 0.2),
            device=args.device
        )
        user_sequences = build_sequences(train_valid_df)

    elif model_type == 'bert4rec':
        max_seq_len = model_config.get('max_seq_len', 50)
        model = BERT4Rec(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=model_config.get('embedding_dim', 64),
            max_seq_len=max_seq_len,
            num_blocks=model_config.get('num_blocks', 2),
            num_heads=model_config.get('num_heads', 2),
            dropout=model_config.get('dropout', 0.2),
            mask_prob=model_config.get('mask_prob', 0.2),
            device=args.device
        )
        user_sequences = build_sequences(train_valid_df)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"Model loaded successfully\n")

    metrics = evaluate_samples(
        model=model,
        test_users=test_users,
        test_items=test_items,
        user_history=user_history,
        user_sequences=user_sequences,
        max_seq_len=max_seq_len,
        batch_size=args.batch_size,
        k_list=args.k_list
    )

    logger.info("\n" + "="*50)
    logger.info(f"Evaluation Results ({num_samples} users)")
    logger.info("="*50)
    for k, v in sorted(metrics.items()):
        logger.info(f"{k}: {v:.4f}")
    logger.info("="*50 + "\n")


if __name__ == '__main__':
    main()
