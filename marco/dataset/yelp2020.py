import os
import json
import random
import tarfile
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
from langchain.prompts import PromptTemplate

from marco.utils import append_his_info

def extract_data(dir: str):
    raw_path = os.path.join(dir, 'raw_data')
    os.makedirs(raw_path, exist_ok=True)
    
    review_file = os.path.join(raw_path, 'yelp_academic_dataset_review.json')
    
    if os.path.exists(review_file):
        logger.info('Yelp2020 dataset already extracted')
        return
    
    zip_file = os.path.join(raw_path, 'archive.zip')
    tgz_file = os.path.join(raw_path, 'dataset.tgz')
    
    archive_file = None
    archive_type = None
    
    if os.path.exists(zip_file):
        archive_file = zip_file
        archive_type = 'zip'
    elif os.path.exists(tgz_file):
        archive_file = tgz_file
        archive_type = 'tgz'
    else:
        raise FileNotFoundError(
            f"Dataset file not found in: {raw_path}\n"
            "Please manually download from Kaggle:\n"
            "https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset\n"
            f"and place it as: {raw_path}/archive.zip or {raw_path}/dataset.tgz"
        )
    
    logger.info(f'Extracting Yelp2020 dataset from {archive_file}...')
    try:
        if archive_type == 'zip':
            with zipfile.ZipFile(archive_file, 'r') as zip_ref:
                zip_ref.extractall(raw_path)
            logger.info('Extracted .zip file')
            
            nested_tgz = os.path.join(raw_path, 'dataset.tgz')
            if os.path.exists(nested_tgz) and not os.path.exists(review_file):
                logger.info('Found nested dataset.tgz, extracting...')
                with tarfile.open(nested_tgz, 'r:gz') as tar:
                    tar.extractall(raw_path)
                logger.info('Extracted nested .tgz file')
        else:
            with tarfile.open(archive_file, 'r:gz') as tar:
                tar.extractall(raw_path)
        
        logger.info('Successfully extracted Yelp2020 dataset')
    except Exception as e:
        logger.error(f'Failed to extract dataset: {e}')
        raise

def read_data(dir: str, start_date: str = '2019-01-01', end_date: str = '2019-12-31') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info(f'Reading Yelp2020 review file with date filter: {start_date} to {end_date}...')
    review_file = os.path.join(dir, 'yelp_academic_dataset_review.json')
    
    start_timestamp = datetime.strptime(start_date, '%Y-%m-%d')
    end_timestamp = datetime.strptime(end_date, '%Y-%m-%d')
    
    reviews = []
    total_reviews = 0
    filtered_reviews = 0
    
    with open(review_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_reviews += 1
            review = json.loads(line)
            review_date = datetime.strptime(review['date'], '%Y-%m-%d %H:%M:%S')
            
            if start_timestamp <= review_date <= end_timestamp:
                reviews.append({
                    'original_order': total_reviews,
                    'user_id': review['user_id'],
                    'business_id': review['business_id'],
                    'rating': review['stars'],
                    'timestamp': review['date']
                })
                filtered_reviews += 1
    
    data_df = pd.DataFrame(reviews)
    logger.info(f'Filtered {filtered_reviews} reviews from {total_reviews} total reviews ({filtered_reviews/total_reviews*100:.2f}%)')
    logger.info(f'Date range: {start_date} to {end_date}')
    
    logger.info('Reading Yelp2020 business file...')
    business_file = os.path.join(dir, 'yelp_academic_dataset_business.json')
    businesses = []
    with open(business_file, 'r', encoding='utf-8') as f:
        for line in f:
            business = json.loads(line)
            businesses.append({
                'business_id': business['business_id'],
                'name': business.get('name', 'Unknown'),
                'categories': business.get('categories', 'Unknown'),
                'city': business.get('city', 'Unknown'),
                'state': business.get('state', 'Unknown')
            })
    
    business_df = pd.DataFrame(businesses)
    logger.info(f'Successfully read {business_df.shape[0]} businesses')
    
    logger.info('Reading Yelp2020 user file...')
    user_file = os.path.join(dir, 'yelp_academic_dataset_user.json')
    users = []
    with open(user_file, 'r', encoding='utf-8') as f:
        for line in f:
            user = json.loads(line)
            users.append({
                'user_id': user['user_id'],
                'review_count': user.get('review_count', 0),
                'average_stars': user.get('average_stars', 0.0)
            })
    
    user_df = pd.DataFrame(users)
    logger.info(f'Successfully read {user_df.shape[0]} users')
    
    return data_df, business_df, user_df

def process_user_data(user_df: pd.DataFrame, filtered_data_df: pd.DataFrame, user_id_map: dict) -> pd.DataFrame:
    valid_users = filtered_data_df['user_id'].unique()
    
    user_df['user_id_int'] = user_df['user_id'].map(user_id_map)
    user_df = user_df[user_df['user_id_int'].notna()].copy()
    user_df = user_df.drop(columns=['user_id']).rename(columns={'user_id_int': 'user_id'})
    user_df = user_df.set_index('user_id')
    
    template = PromptTemplate(
        template='User with {review_count} reviews, average rating: {average_stars:.1f}',
        input_variables=['review_count', 'average_stars'],
    )
    user_df['user_profile'] = user_df.apply(lambda x: template.format(**x), axis=1)
    
    return user_df

def process_item_data(business_df: pd.DataFrame, filtered_data_df: pd.DataFrame, item_id_map: dict) -> pd.DataFrame:
    business_df['item_id_int'] = business_df['business_id'].map(item_id_map)
    business_df = business_df[business_df['item_id_int'].notna()].copy()
    business_df = business_df.drop(columns=['business_id']).rename(columns={'item_id_int': 'item_id'})
    business_df = business_df.set_index('item_id')
    
    template = PromptTemplate(
        template='Business: {name}, Categories: {categories}, Location: {city}, {state}',
        input_variables=['name', 'categories', 'city', 'state'],
    )
    business_df['item_attributes'] = business_df.apply(lambda x: template.format(**x), axis=1)
    
    return business_df

def filter_data(data_df: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
    original_size = data_df.shape[0]
    original_users = data_df['user_id'].nunique()
    original_items = data_df['business_id'].nunique()
    
    filter_before = -1
    iteration = 0
    while filter_before != data_df.shape[0]:
        filter_before = data_df.shape[0]
        iteration += 1
        data_df = data_df.groupby('user_id').filter(lambda x: len(x) >= min_interactions)
        data_df = data_df.groupby('business_id').filter(lambda x: len(x) >= min_interactions)
        logger.info(f'  Iteration {iteration}: {data_df.shape[0]} interactions, '
                   f'{data_df["user_id"].nunique()} users, {data_df["business_id"].nunique()} items')
    
    filtered_size = data_df.shape[0]
    filtered_users = data_df['user_id'].nunique()
    filtered_items = data_df['business_id'].nunique()
    
    logger.info(f'{min_interactions}-core filtering completed after {iteration} iterations:')
    logger.info(f'  Interactions: {original_size} -> {filtered_size} ({filtered_size/original_size*100:.2f}%)')
    logger.info(f'  Users: {original_users} -> {filtered_users} ({filtered_users/original_users*100:.2f}%)')
    logger.info(f'  Items: {original_items} -> {filtered_items} ({filtered_items/original_items*100:.2f}%)')
    
    return data_df

def process_interaction_data(data_df: pd.DataFrame, n_neg_items: int = 7, k_core: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df['timestamp'] = data_df['timestamp'].astype(np.int64) // 10**9
    
    logger.info(f'Applying {k_core}-core filtering...')
    data_df = filter_data(data_df, min_interactions=k_core)
    
    data_df = data_df.rename(columns={'business_id': 'item_id'})
    
    logger.info('Mapping string IDs to integer IDs...')
    if 'original_order' in data_df.columns:
        data_df = data_df.sort_values(by='original_order', kind='mergesort')
    
    unique_users = pd.unique(data_df['user_id'])
    unique_items = pd.unique(data_df['item_id'])
    
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users, start=1)}
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_items, start=1)}
    
    data_df['user_id'] = data_df['user_id'].map(user_id_map)
    data_df['item_id'] = data_df['item_id'].map(item_id_map)
    
    logger.info(f'Mapped {len(user_id_map)} users and {len(item_id_map)} items to integer IDs')
    
    if 'original_order' in data_df.columns:
        data_df = data_df.sort_values(by=['timestamp', 'original_order'], ascending=[True, True], kind='mergesort')
        data_df = data_df.drop(columns=['original_order'])
    else:
        data_df = data_df.sort_values(by=['timestamp'], kind='mergesort')
    
    clicked_item_set = {}
    for uid, group in data_df.groupby('user_id'):
        clicked_item_set[uid] = set(group['item_id'].values)
    
    all_items = data_df['item_id'].unique()
    
    def negative_sample(df):
        neg_items = []
        for uid in df['user_id'].values:
            user_clicked = clicked_item_set[uid]
            user_neg_items = []
            while len(user_neg_items) < n_neg_items:
                neg_item = np.random.choice(all_items)
                if neg_item not in user_clicked and neg_item not in user_neg_items:
                    user_neg_items.append(neg_item)
            neg_items.append(user_neg_items)
        df['neg_item_id'] = neg_items
        return df
    
    def generate_dev_test(data_df: pd.DataFrame) -> tuple[list[pd.DataFrame], pd.DataFrame]:
        result_dfs = []
        for idx in range(2):
            result_df = data_df.groupby('user_id').tail(1).copy()
            data_df = data_df.drop(result_df.index)
            result_dfs.append(result_df)
        return result_dfs, data_df
    
    leave_df = data_df.groupby('user_id').head(1)
    left_df = data_df.drop(leave_df.index)
    
    [test_df, dev_df], train_df = generate_dev_test(left_df)
    train_df = pd.concat([leave_df, train_df]).sort_index()
    
    logger.info('Applying negative sampling...')
    train_df = negative_sample(train_df)
    dev_df = negative_sample(dev_df)
    test_df = negative_sample(test_df)
    
    logger.info(f'Data split - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}')
    
    return train_df, dev_df, test_df, data_df, user_id_map, item_id_map

def process_data(dir: str, n_neg_items: int = 7, k_core: int = 5, start_date: str = '2019-01-01', end_date: str = '2019-12-31'):
    logger.info(f'Starting to process Yelp2020 dataset in {dir}')
    logger.info(f'Configuration: k_core={k_core}, date_range=[{start_date}, {end_date}], n_neg_items={n_neg_items}')
    
    extract_data(dir)
    
    raw_data_dir = os.path.join(dir, "raw_data")
    
    data_df, business_df, user_df = read_data(raw_data_dir, start_date=start_date, end_date=end_date)
    
    logger.info('Processing interaction data...')
    train_df, dev_df, test_df, filtered_data_df, user_id_map, item_id_map = process_interaction_data(data_df, n_neg_items, k_core)
    
    logger.info('Processing user data...')
    user_df = process_user_data(user_df, filtered_data_df, user_id_map)
    logger.info(f'Number of users: {user_df.shape[0]}')
    
    logger.info('Processing business data...')
    item_df = process_item_data(business_df, filtered_data_df, item_id_map)
    logger.info(f'Number of businesses: {item_df.shape[0]}')
    
    logger.info('Appending history information...')
    dfs = append_his_info([train_df, dev_df, test_df], neg=True)
    
    logger.info('Finalizing data (keeping IDs only, no text formatting)...')
    for i, df in enumerate(dfs):
        df_name = ['train', 'dev', 'test'][i]
        logger.info(f'Processing {df_name} set...')
        
        df['candidate_item_id'] = df.apply(lambda x: [x['item_id']] + x['neg_item_id'], axis=1)
        
        df['candidate_item_id'] = df['candidate_item_id'].apply(lambda x: random.sample(x, len(x)))
        
        df['history_item_id'] = df['history_item_id'].apply(lambda x: str(x) if isinstance(x, list) else x)
        df['history_rating'] = df['history_rating'].apply(lambda x: str(x) if isinstance(x, list) else x)
        df['neg_item_id'] = df['neg_item_id'].apply(lambda x: str(x) if isinstance(x, list) else x)
        df['candidate_item_id'] = df['candidate_item_id'].apply(lambda x: str(x) if isinstance(x, list) else x)
    
    train_df, dev_df, test_df = dfs[0], dfs[1], dfs[2]

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
        user_df.to_csv(os.path.join(dir, 'user.csv'))
        item_df.to_csv(os.path.join(dir, 'item.csv'))
        all_df.to_csv(os.path.join(dir, 'all.csv'), index=False)
        test_one_per_user.to_csv(os.path.join(dir, 'test.csv'), index=False)
        logger.info('Successfully saved all CSV files (user.csv, item.csv, all.csv, test.csv)')
        logger.info('all.csv contains only user interaction columns')
        logger.info('test.csv contains evaluation samples without redundant columns')
        
        user_mapping_df = pd.DataFrame([
            {'preprocessed_id': new_id, 'original_id': old_id}
            for old_id, new_id in sorted(user_id_map.items(), key=lambda x: x[1])
        ])
        item_mapping_df = pd.DataFrame([
            {'preprocessed_id': new_id, 'original_id': old_id}
            for old_id, new_id in sorted(item_id_map.items(), key=lambda x: x[1])
        ])
        user_mapping_df.to_csv(os.path.join(dir, 'user_id_mapping.csv'), index=False)
        item_mapping_df.to_csv(os.path.join(dir, 'item_id_mapping.csv'), index=False)
        logger.info('Saved ID mappings for RecBole integration (user_id_mapping.csv, item_id_mapping.csv)')
        
        metadata = {
            'k_core': k_core,
            'start_date': start_date,
            'end_date': end_date,
            'n_neg_items': n_neg_items,
            'num_users': int(user_df.shape[0]),
            'num_items': int(item_df.shape[0]),
            'num_interactions': int(len(all_df)),
            'train_size': int(len(train_df)),
            'dev_size': int(len(dev_df)),
            'test_size': int(len(test_df))
        }
        metadata_file = os.path.join(dir, 'preprocessing_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f'Saved preprocessing metadata to {metadata_file}')
        
    except Exception as e:
        logger.error(f'Error saving CSV files: {e}')
        raise