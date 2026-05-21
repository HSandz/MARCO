import argparse
import os
import torch
import pandas as pd
from loguru import logger

from recommender.models.lightgcn import LightGCN
from recommender.models.sasrec import SASRec
from recommender.models.bert4rec import BERT4Rec
from recommender.trainer import Trainer
from recommender.utils.data_utils import (
    load_interactions, build_user_item_dict, build_sparse_graph,
    normalize_adj_matrix, leave_one_out_split, build_sequences,
    BPRDataset, SequenceDataset, SASRecDataset, EvalDataset, get_dataloader
)
from recommender.utils.export import export_for_marco, compute_recall_at_k


def parse_args():
    parser = argparse.ArgumentParser(description='Train recommender models')
    parser.add_argument('--model', type=str, default='lightgcn',
                        choices=['lightgcn', 'sasrec', 'bert4rec'])
    parser.add_argument('--data', type=str, required=True, help='Path to original data (e.g. data/ml-100k/all.csv)')
    parser.add_argument('--test_data', type=str, default=None, help='Path to test.csv for MARCO export')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3, help='For LightGCN')
    parser.add_argument('--num_blocks', type=int, default=2, help='For SASRec/BERT4Rec')
    parser.add_argument('--num_heads', type=int, default=2, help='For SASRec/BERT4Rec')
    parser.add_argument('--max_seq_len', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--reg_weight', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--export_topk', type=int, default=20)
    parser.add_argument('--checkpoint_dir', type=str, default='recommender/checkpoints')
    parser.add_argument('--candidate_dir', type=str, default='recommender/candidates')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--rating_threshold', type=float, default=4, help='Minimum rating to include (default 4)')
    parser.add_argument('--seed', type=int, default=2026, help='Random seed')
    return parser.parse_args()


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.candidate_dir, exist_ok=True)
    
    dataset_name = os.path.basename(os.path.dirname(args.data))
    
    import sys
    from datetime import datetime
    log_dir = 'recommender/logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"{args.model}_{dataset_name}_{timestamp}.log")
    
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, level="DEBUG", rotation="10 MB")
    
    logger.info(f"Log file: {log_file}")
    
    from recommender.utils.data_utils import ensure_filtered_dataset
    filtered_data_path = ensure_filtered_dataset(args.data, output_dir='recommender/datasets', threshold=args.rating_threshold)
    
    logger.info(f"Loading FILTERED data from {filtered_data_path}...")
    df, num_users, num_items = load_interactions(filtered_data_path)
    logger.info(f"Filtered: {len(df)} interactions, {num_users} users, {num_items} items")
    
    train_df, valid_df, test_df = leave_one_out_split(df)
    logger.info(f"Split: train={len(train_df)}, valid={len(valid_df) if valid_df is not None else 0}, test={len(test_df)}")

    user_history = build_user_item_dict(train_df)

    train_valid_df = pd.concat([train_df, valid_df]) if valid_df is not None else train_df
    test_user_history = build_user_item_dict(train_valid_df)
    
    if args.model == 'lightgcn':
        model = LightGCN(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            reg_weight=args.reg_weight,
            dropout=args.dropout,
            device=args.device
        )
        
        adj_matrix = build_sparse_graph(train_df, num_users, num_items)
        norm_adj = normalize_adj_matrix(adj_matrix)
        model.set_adj_matrix(norm_adj)
        
        train_dataset = BPRDataset(train_df, num_items, user_history)
        train_loader = get_dataloader(train_dataset, args.batch_size, shuffle=True)
        
        if valid_df is not None:
            valid_dataset = BPRDataset(valid_df, num_items, user_history, num_negatives=1)
            valid_loader = get_dataloader(valid_dataset, min(args.batch_size, 256), shuffle=False)
        else:
            valid_loader = None
            
    elif args.model == 'sasrec':
        model = SASRec(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=args.embedding_dim,
            max_seq_len=args.max_seq_len,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            dropout=args.dropout,
            device=args.device
        )
        
        user_sequences = build_sequences(train_df)
        train_dataset = SASRecDataset(user_sequences, num_items, args.max_seq_len)
        train_loader = get_dataloader(train_dataset, args.batch_size, shuffle=True)
        
        if valid_df is not None:
            valid_sequences = build_sequences(train_df)
            valid_dataset = EvalDataset(valid_df, valid_sequences, args.max_seq_len)
            valid_loader = get_dataloader(valid_dataset, min(args.batch_size, 256), shuffle=False)
        else:
            valid_loader = None
            
    elif args.model == 'bert4rec':
        model = BERT4Rec(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=args.embedding_dim,
            max_seq_len=args.max_seq_len,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            dropout=args.dropout,
            mask_prob=0.2,
            device=args.device
        )
        
        user_sequences = build_sequences(train_df)
        train_dataset = SequenceDataset(user_sequences, num_items, args.max_seq_len, 
                                         mask_prob=0.2, mask_token=num_items + 1)
        train_loader = get_dataloader(train_dataset, args.batch_size, shuffle=True)
        
        if valid_df is not None:
            valid_sequences = build_sequences(train_df)
            valid_dataset = EvalDataset(valid_df, valid_sequences, args.max_seq_len)
            valid_loader = get_dataloader(valid_dataset, min(args.batch_size, 256), shuffle=False)
        else:
            valid_loader = None
    
    logger.info(f"Model: {args.model}")
    logger.info(f"Config: {model.get_config()}")
    
    checkpoint_filename = f"{args.model}_{dataset_name}.pth"
    trainer = Trainer(
        model=model,
        learning_rate=args.lr,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_filename=checkpoint_filename
    )
    
    logger.info("Starting training...")
    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        user_history=user_history,
        epochs=args.epochs,
        eval_every=args.eval_every
    )
    
    logger.info("Training completed!")
    
    try:
        trainer.load_checkpoint(checkpoint_filename)
        logger.info("Loaded best model checkpoint")
    except:
        logger.warning("Could not load best model, using final model")
    
    if args.test_data:
        test_export_df = pd.read_csv(args.test_data)
    else:
        test_export_df = test_df

    logger.info(f"Building history from train+valid data for export...")
    from recommender.utils.data_utils import build_user_history_with_ratings
    train_valid_history_dict = build_user_history_with_ratings(train_valid_df)

    export_history_items = {u: h[0] for u, h in train_valid_history_dict.items()}
    export_history_ratings = {u: h[1] for u, h in train_valid_history_dict.items()}

    user_sequences_for_export = None
    if args.model in ['sasrec', 'bert4rec']:
        user_sequences_for_export = build_sequences(train_valid_df)
    
    candidate_filename = f"test_{args.model}.csv"
    output_dir = os.path.join('data', dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, candidate_filename)
    export_for_marco(
        model=model,
        test_df=test_export_df,
        user_history=export_history_items,
        user_history_ratings=export_history_ratings,
        k=args.export_topk,
        user_sequences=user_sequences_for_export,
        max_seq_len=args.max_seq_len,
        output_path=output_path
    )
    
    test_users = test_df['user_id'].tolist()
    test_items = test_df['item_id'].tolist()

    eval_sequences = None
    if args.model in ['sasrec', 'bert4rec']:
        eval_sequences = build_sequences(train_valid_df)

    metrics = compute_recall_at_k(
        model=model,
        test_users=test_users,
        test_items=test_items,
        user_history=test_user_history,
        k_list=[5, 10, 20],
        user_sequences=eval_sequences,
        max_seq_len=args.max_seq_len
    )
    
    logger.success("=== Final Test Metrics ===")
    for k, v in metrics.items():
        logger.success(f"{k}: {v:.4f}")


if __name__ == '__main__':
    main()
