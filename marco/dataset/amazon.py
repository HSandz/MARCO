import os
import random
import pandas as pd
import numpy as np
import gzip
import subprocess
from typing import Generator
from loguru import logger
from langchain.prompts import PromptTemplate

from marco.utils import append_his_info

def parse(path: str) -> Generator[dict, None, None]:
    g = gzip.open(path, 'rb')
    for entry in g:
        yield eval(entry)

def get_df(path: str) -> pd.DataFrame:
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def download_data(dir: str, dataset: str):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    raw_path = os.path.join(dir, 'raw_data')
    data_file = 'reviews_{}_5.json.gz'.format(dataset)
    meta_file = 'meta_{}.json.gz'.format(dataset)
    if not os.path.exists(raw_path):
        os.makedirs(raw_path, exist_ok=True)
    if not os.path.exists(os.path.join(raw_path, data_file)):
        logger.info('Downloading interaction data into ' + raw_path)
        if os.name == 'nt':
            subprocess.call(
                f'cd /d "{raw_path}" && curl -O http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{dataset}_5.json.gz', 
                shell=True)
        else:
            subprocess.call(
                f'cd {raw_path} && curl -O http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{dataset}_5.json.gz', 
                shell=True)
    if not os.path.exists(os.path.join(raw_path, meta_file)):
        logger.info('Downloading item metadata into ' + raw_path)
        if os.name == 'nt':
            subprocess.call(
                f'cd /d "{raw_path}" && curl -O http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_{dataset}.json.gz', 
                shell=True)
        else:
            subprocess.call(
                f'cd {raw_path} && curl -O http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_{dataset}.json.gz', 
                shell=True)

def read_data(dir: str, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = os.path.join(dir, 'raw_data')
    data_file = 'reviews_{}_5.json.gz'.format(dataset)
    meta_file = 'meta_{}.json.gz'.format(dataset)
    data_df = get_df(os.path.join(raw_path, data_file))
    meta_df = get_df(os.path.join(raw_path, meta_file))
    return data_df, meta_df

def process_item_data(data_df: pd.DataFrame, meta_df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    useful_meta_df = meta_df[meta_df['asin'].isin(data_df['asin'])].reset_index(drop=True)

    item_df = useful_meta_df.rename(columns={'asin': 'item_id'})
    item_df = item_df[['item_id', 'title', 'brand', 'price', 'categories']]

    user2id, item2id = reindex(data_df)
    item_df['item_id'] = item_df['item_id'].apply(lambda x: item2id[x])
    item_df = item_df.set_index('item_id')
    item_df.sort_index(inplace=True)

    l2_cate_lst = list()
    for cate_lst in item_df['categories']:
        l2_cate_lst.append(cate_lst[0][1:] if len(cate_lst[0]) > 1 else np.nan)
    item_df['categories'] = l2_cate_lst

    for col in item_df.columns.to_list():
        item_df[col] = item_df[col].fillna('unknown')

    item_df['title'] = item_df['title'].apply(lambda x: x.replace('\n', ' '))
    item_df['categories'] = item_df['categories'].apply(lambda x: '|'.join(x))
    input_variables = item_df.columns.to_list()
    template = PromptTemplate(
        template='Brand: {brand}, Price: {price}, Categories: {categories}',
        input_variables=input_variables,
    )
    item_df['item_attributes'] = item_df[input_variables].apply(lambda x: template.format(**x), axis=1)

    return item_df, user2id, item2id

def reindex(data_df: pd.DataFrame, out_df: pd.DataFrame = None) -> tuple[dict, dict]:
    if out_df is None:
        out_df = data_df.rename(columns={'asin': 'item_id', 'reviewerID': 'user_id', 'overall': 'rating', 'unixReviewTime': 'timestamp'})
        out_df = out_df[['user_id', 'item_id', 'rating', 'summary', 'timestamp']]
        out_df = out_df.drop_duplicates(['user_id', 'item_id', 'timestamp'])
    
    uids = pd.unique(out_df['user_id'])
    user2id = dict(zip(uids, range(1, len(uids) + 1)))
    iids = pd.unique(out_df['item_id'])
    item2id = dict(zip(iids, range(1, len(iids) + 1)))
    return user2id, item2id

def process_interaction_data(data_df: pd.DataFrame, n_neg_items: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_df = data_df.rename(columns={'asin': 'item_id', 'reviewerID': 'user_id', 'overall': 'rating', 'unixReviewTime': 'timestamp'})
    out_df = out_df[['user_id', 'item_id', 'rating', 'summary', 'timestamp']]
    out_df = out_df.drop_duplicates(['user_id', 'item_id', 'timestamp'])
    
    user2id, item2id = reindex(data_df, out_df)

    out_df['user_id'] = out_df['user_id'].apply(lambda x: user2id[x])
    out_df['item_id'] = out_df['item_id'].apply(lambda x: item2id[x])
    
    out_df = out_df.sort_values(by=['timestamp', 'user_id'], kind='mergesort').reset_index(drop=True)

    clicked_item_set = dict()
    for user_id, seq_df in out_df.groupby('user_id'):
        clicked_item_set[user_id] = set(seq_df['item_id'].values.tolist())

    n_items = out_df['item_id'].value_counts().size

    def negative_sample(df):
        neg_items = np.random.randint(1, n_items + 1, (len(df), n_neg_items))
        for i, uid in enumerate(df['user_id'].values):
            user_clicked = clicked_item_set[uid]
            for j in range(len(neg_items[i])):
                while neg_items[i][j] in user_clicked or neg_items[i][j] in neg_items[i][:j]:
                    neg_items[i][j] = np.random.randint(1, n_items + 1)
            assert len(set(neg_items[i])) == len(neg_items[i])
        df['neg_item_id'] = neg_items.tolist()
        return df

    def generate_dev_test(data_df):
        result_dfs = []
        for idx in range(2):
            result_df = data_df.groupby('user_id').tail(1).copy()
            data_df = data_df.drop(result_df.index)
            result_dfs.append(result_df)
        return result_dfs, data_df

    out_df = negative_sample(out_df)

    leave_df = out_df.groupby('user_id').head(1)
    data_df = out_df.drop(leave_df.index)

    [test_df, dev_df], data_df = generate_dev_test(data_df)
    train_df = pd.concat([leave_df, data_df]).sort_index()

    return train_df, dev_df, test_df, out_df

def process_data(dir: str, n_neg_items: int = 9):
    dataset = os.path.basename(dir)

    download_data(dir, dataset)
    data_df, meta_df = read_data(dir, dataset)

    train_df, dev_df, test_df, out_df = process_interaction_data(data_df, n_neg_items)
    logger.info(f'Number of interactions: {out_df.shape[0]}')

    logger.info(f"Number of users: {out_df['user_id'].nunique()}")

    item_df, user2id, item2id = process_item_data(data_df, meta_df)
    logger.info(f'Number of items: {item_df.shape[0]}')

    dfs = append_his_info([train_df, dev_df, test_df], summary=True, neg=True)
    logger.info('Completed append history information to interactions')
    
    logger.info('Finalizing data (keeping IDs only, no text formatting)...')
    for i, df in enumerate(dfs):
        df_name = ['train', 'dev', 'test'][i]
        logger.info(f'Processing {df_name} set...')
        
        df['candidate_item_id'] = df.apply(lambda x: [x['item_id']] + x['neg_item_id'], axis=1)
        
        df['candidate_item_id'] = df['candidate_item_id'].apply(lambda x: random.sample(x, len(x)))
        
        df['history_item_id'] = df['history_item_id'].apply(lambda x: str(x) if isinstance(x, list) else x)
        df['history_rating'] = df['history_rating'].apply(lambda x: str(x) if isinstance(x, list) else x)
        df['history_summary'] = df['history_summary'].apply(lambda x: str(x) if isinstance(x, list) else x)
        df['neg_item_id'] = df['neg_item_id'].apply(lambda x: str(x) if isinstance(x, list) else x)
        df['candidate_item_id'] = df['candidate_item_id'].apply(lambda x: str(x) if isinstance(x, list) else x)

    train_df = dfs[0]
    dev_df = dfs[1]
    test_df = dfs[2]

    all_df = pd.concat([train_df, dev_df, test_df])
    all_df = all_df.sort_values(by=['timestamp'], kind='mergesort')
    all_df = all_df.reset_index(drop=True)
    
    all_columns_to_remove = ['neg_item_id', 'position', 'candidate_item_id']
    all_columns_to_remove = [col for col in all_columns_to_remove if col in all_df.columns]
    if all_columns_to_remove:
        all_df = all_df.drop(columns=all_columns_to_remove)
        logger.info(f'Removed redundant columns from all.csv: {all_columns_to_remove}')
    
    test_one_per_user = test_df.groupby('user_id').tail(1).reset_index(drop=True)
    
    test_columns_to_remove = ['timestamp', 'neg_item_id', 'position']
    test_columns_to_remove = [col for col in test_columns_to_remove if col in test_one_per_user.columns]
    if test_columns_to_remove:
        test_one_per_user = test_one_per_user.drop(columns=test_columns_to_remove)
        logger.info(f'Removed redundant columns from test.csv: {test_columns_to_remove}')
    
    logger.info(f'all.csv: {len(all_df)} interactions from {all_df["user_id"].nunique()} users')
    logger.info(f'test.csv: {len(test_one_per_user)} samples (one per user)')

    logger.info('Outputing data to csv files...')
    try:
        item_df.to_csv(os.path.join(dir, 'item.csv'))
        all_df.to_csv(os.path.join(dir, 'all.csv'), index=False)
        test_one_per_user.to_csv(os.path.join(dir, 'test.csv'), index=False)
        logger.info('Successfully saved all CSV files (item.csv, all.csv, test.csv)')
        logger.info('all.csv contains only user interaction columns')
        logger.info('test.csv contains evaluation samples without redundant columns')
        
        user_mapping_df = pd.DataFrame([
            {'preprocessed_id': new_id, 'original_id': old_id}
            for old_id, new_id in sorted(user2id.items(), key=lambda x: x[1])
        ])
        item_mapping_df = pd.DataFrame([
            {'preprocessed_id': new_id, 'original_id': old_id}
            for old_id, new_id in sorted(item2id.items(), key=lambda x: x[1])
        ])
        user_mapping_df.to_csv(os.path.join(dir, 'user_id_mapping.csv'), index=False)
        item_mapping_df.to_csv(os.path.join(dir, 'item_id_mapping.csv'), index=False)
        logger.info('Saved ID mappings for RecBole integration (user_id_mapping.csv, item_id_mapping.csv)')
    except Exception as e:
        logger.error(f'Error saving CSV files: {e}')
        raise
