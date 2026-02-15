#! /bin/bash

# Train Recommender Models on MovieLens-100k dataset
## Movielens-100k
python -m recommender.run --model lightgcn --data data/ml-100k/all.csv --epochs 200 --batch_size 2048 --export_topk 20 --lr 0.01
python -m recommender.run --model sasrec --data data/ml-100k/all.csv --epochs 200 --batch_size 256 --export_topk 20
python -m recommender.run --model bert4rec --data data/ml-100k/all.csv --epochs 200 --batch_size 256 --export_topk 20
## Amazon Beauty
python -m recommender.run --model lightgcn --data data/Beauty/all.csv --epochs 200 --batch_size 2048 --export_topk 20 --lr 0.01
python -m recommender.run --model sasrec --data data/Beauty/all.csv --epochs 200 --batch_size 256 --export_topk 20
python -m recommender.run --model bert4rec --data data/Beauty/all.csv --epochs 200 --batch_size 256 --export_topk 20
## Amazon Electronics
# python -m recommender.run --model lightgcn --data data/Electronics/all.csv --epochs 200 --batch_size 2048 --export_topk 20 --lr 0.01
# python -m recommender.run --model sasrec --data data/Electronics/all.csv --epochs 200 --batch_size 256 --export_topk 20
# python -m recommender.run --model bert4rec --data data/Electronics/all.csv --epochs 200 --batch_size 256 --export_topk 20

# Common Arguments:
# --model: lightgcn, sasrec, bert4rec
# --data: Path to CSV with user_id, item_id, (timestamp, rating optional)
# --epochs: Training epochs
# --batch_size: Batch size
# --lr: Learning rate
# --export_topk: Number of candidates to export for MARCO
# --eval_every: Evaluation frequency (epochs)
# --patience: 
# --device: cuda or cpu