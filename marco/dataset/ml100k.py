import os
import random
import pandas as pd
import numpy as np
import urllib.request
import zipfile
import shutil
from loguru import logger
from langchain.prompts import PromptTemplate

from marco.utils import append_his_info

def download_data(dir: str):
    raw_path = os.path.join(dir, 'raw_data')
    os.makedirs(raw_path, exist_ok=True)
    
    zip_file_path = os.path.join(raw_path, 'ml-100k.zip')
    if not os.path.exists(zip_file_path):
        logger.info('Downloading ml-100k dataset into ' + raw_path)
        try:
            url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
            urllib.request.urlretrieve(url, zip_file_path)
            logger.info('Downloaded ml-100k.zip successfully')
        except Exception as e:
            logger.error(f'Failed to download ml-100k.zip: {e}')
            raise
    
    if not os.path.exists(os.path.join(raw_path, 'u.data')):
        logger.info('Unzipping ml-100k dataset into ' + raw_path)
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(raw_path)
            
            ml100k_dir = os.path.join(raw_path, 'ml-100k')
            if os.path.exists(ml100k_dir):
                for filename in os.listdir(ml100k_dir):
                    src = os.path.join(ml100k_dir, filename)
                    dst = os.path.join(raw_path, filename)
                    shutil.move(src, dst)
                os.rmdir(ml100k_dir)
            logger.info('Extracted and organized files successfully')
        except Exception as e:
            logger.error(f'Failed to extract ml-100k.zip: {e}')
            raise

def read_data(dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        logger.info('Reading u.data file...')
        with open(os.path.join(dir, 'u.data'), 'r') as f:
            data_df = pd.read_csv(f, sep='\t', header=None)
        
        logger.info('Reading u.item file...')
        with open(os.path.join(dir, 'u.item'), 'r', encoding='ISO-8859-1') as f:
            item_df = pd.read_csv(f, sep='|', header=None, encoding='ISO-8859-1')
        
        logger.info('Reading u.user file...')
        with open(os.path.join(dir, 'u.user'), 'r') as f:
            user_df = pd.read_csv(f, sep='|', header=None)
        
        logger.info('Reading u.genre file...')
        with open(os.path.join(dir, 'u.genre'), 'r') as f:
            genre_df = pd.read_csv(f, sep='|', header=None)
        
        if data_df.empty or item_df.empty or user_df.empty or genre_df.empty:
            raise ValueError("One or more data files are empty")
        
        logger.info(f'Successfully read data files: {data_df.shape[0]} interactions, {item_df.shape[0]} items, {user_df.shape[0]} users, {genre_df.shape[0]} genres')
        return data_df, item_df, user_df, genre_df
        
    except FileNotFoundError as e:
        logger.error(f'Required data file not found: {e}')
        raise
    except Exception as e:
        logger.error(f'Error reading data files: {e}')
        raise

def process_user_data(user_df: pd.DataFrame) -> pd.DataFrame:
    user_df.columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    user_df = user_df.drop(columns=['zip_code'])
    user_df = user_df.set_index('user_id')
    user_df['gender'] = user_df['gender'].apply(lambda x: 'male' if x == 'M' else 'female')
    input_variables = user_df.columns.to_list()
    template = PromptTemplate(
        template='Age: {age}\nGender: {gender}\nOccupation: {occupation}',
        input_variables=input_variables,
    )
    user_df['user_profile'] = user_df[input_variables].apply(lambda x: template.format(**x), axis=1)

    for col in user_df.columns.to_list():
        user_df[col] = user_df[col].apply(lambda x: 'None' if x == '' else x)
    return user_df

def process_item_data(item_df: pd.DataFrame) -> pd.DataFrame:
    item_df.columns = ['item_id', 'title', 'release_date', 'video_release_date',
                       'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                       'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
                       'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                       'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    genres = item_df.columns.to_list()[5:]
    item_df = item_df.drop(columns=['IMDb_URL'])
    item_df = item_df.set_index('item_id')
    item_df['video_release_date'] = item_df['video_release_date'].fillna('unknown')
    item_df['release_date'] = item_df['release_date'].fillna('unknown')

    def get_genre(x: pd.Series) -> list[str]:
        return '|'.join([genre for genre, value in x.items() if value == 1])

    item_df['genre'] = item_df[genres].apply(lambda x: get_genre(x), axis=1)
    input_variables = item_df.columns.to_list()[:3] + ['genre']
    template = PromptTemplate(
        template='Title: {title}, Genres: {genre}',
        input_variables=input_variables,
    )
    item_df['item_attributes'] = item_df[input_variables].apply(lambda x: template.format(**x), axis=1)

    item_df = item_df.drop(columns=genres)
    return item_df

def filter_data(data_df: pd.DataFrame, min_interactions: int = 5, max_iterations: int = 10) -> pd.DataFrame:
    original_size = data_df.shape[0]
    iteration = 0
    filter_before = -1
    
    while filter_before != data_df.shape[0] and iteration < max_iterations:
        filter_before = data_df.shape[0]
        
        user_counts = data_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        data_df = data_df[data_df['user_id'].isin(valid_users)]
        
        item_counts = data_df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_interactions].index
        data_df = data_df[data_df['item_id'].isin(valid_items)]
        
        iteration += 1
    
    if iteration >= max_iterations:
        logger.warning(f'Filter reached max iterations ({max_iterations}). Some users/items may have < {min_interactions} interactions.')
    
    filtered_size = data_df.shape[0]
    logger.info(f'Filtered data: {original_size} -> {filtered_size} interactions ({filtered_size/original_size*100:.1f}% retained)')
    
    return data_df

def densify_index(data_df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    logger.info('Densifying index (remapping to sequential IDs)')
    
    unique_users = pd.unique(data_df['user_id'])
    unique_items = pd.unique(data_df['item_id'])
    
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users, start=1)}
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_items, start=1)}
    
    logger.info(f'Mapped {len(user_id_map)} users: {min(unique_users)}-{max(unique_users)} → 1-{len(user_id_map)}')
    logger.info(f'Mapped {len(item_id_map)} items: {min(unique_items)}-{max(unique_items)} → 1-{len(item_id_map)}')
    
    data_df['user_id'] = data_df['user_id'].map(user_id_map)
    data_df['item_id'] = data_df['item_id'].map(item_id_map)
    
    return data_df, user_id_map, item_id_map

def process_interaction_data(data_df: pd.DataFrame, n_neg_items: int = 9) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    data_df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    data_df = filter_data(data_df)
    
    if data_df.empty:
        raise ValueError("No data remains after filtering. Consider reducing min_interactions parameter.")
    
    data_df, user_id_map, item_id_map = densify_index(data_df)
    
    data_df = data_df.sort_values(by=['timestamp'], kind='mergesort')
    
    clicked_item_set = dict()
    for user_id, seq_df in data_df.groupby('user_id'):
        clicked_item_set[user_id] = set(seq_df['item_id'].values.tolist())

    n_items = data_df['item_id'].nunique()
    min_item_id = data_df['item_id'].min()
    max_item_id = data_df['item_id'].max()

    def negative_sample(df):
        neg_items = np.random.randint(min_item_id, max_item_id + 1, (len(df), n_neg_items))
        for i, uid in enumerate(df['user_id'].values):
            user_clicked = clicked_item_set[uid]
            for j in range(len(neg_items[i])):
                attempts = 0
                while (neg_items[i][j] in user_clicked or 
                       neg_items[i][j] in neg_items[i][:j]) and attempts < 100:
                    neg_items[i][j] = np.random.randint(min_item_id, max_item_id + 1)
                    attempts += 1
                if attempts >= 100:
                    logger.warning(f"Could not find unique negative item for user {uid} after 100 attempts")
            
            if len(set(neg_items[i])) != len(neg_items[i]):
                logger.warning(f"Duplicate negative items found for user {uid}")
        df['neg_item_id'] = neg_items.tolist()
        return df

    def generate_dev_test(data_df: pd.DataFrame) -> tuple[list[pd.DataFrame], pd.DataFrame]:
        result_dfs = []
        for idx in range(2):
            result_df = data_df.groupby('user_id').tail(1).copy()
            data_df = data_df.drop(result_df.index)
            result_dfs.append(result_df)
        return result_dfs, data_df

    data_df = negative_sample(data_df)
    keep_first_df = data_df.groupby('user_id').head(1)
    remaining_df = data_df.drop(keep_first_df.index)

    [test_df, dev_df], train_remaining_df = generate_dev_test(remaining_df)
    train_df = pd.concat([keep_first_df, train_remaining_df]).sort_index()
    
    logger.info(f'Data split - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}')
    logger.info(f'Unique users - Train: {train_df["user_id"].nunique()}, Dev: {dev_df["user_id"].nunique()}, Test: {test_df["user_id"].nunique()}')
    
    return train_df, dev_df, test_df, user_id_map, item_id_map

def process_data(dir: str, n_neg_items: int = 9):
    try:
        logger.info(f'Starting to process ml-100k dataset in {dir}')
        
        download_data(dir)
        raw_data_dir = os.path.join(dir, "raw_data")
        
        logger.info('Reading raw data files...')
        data_df, item_df, user_df, genre_df = read_data(raw_data_dir)
        
        logger.info('Processing user data...')
        user_df = process_user_data(user_df)
        logger.info(f'Number of users: {user_df.shape[0]}')
        
        logger.info('Processing item data...')
        item_df = process_item_data(item_df)
        logger.info(f'Number of items: {item_df.shape[0]}')
        
        logger.info('Processing interaction data...')
        train_df, dev_df, test_df, user_id_map, item_id_map = process_interaction_data(data_df, n_neg_items)
        logger.info(f'Number of train interactions: {train_df.shape[0]}')
        logger.info(f'Number of dev interactions: {dev_df.shape[0]}')
        logger.info(f'Number of test interactions: {test_df.shape[0]}')
        
        logger.info('Filtering user and item metadata to match processed interactions...')
        
        old_user_ids = list(user_id_map.keys())
        user_df = user_df[user_df.index.isin(old_user_ids)].copy()
        user_df.index = user_df.index.map(user_id_map)
        user_df.index.name = 'user_id'
        logger.info(f'Filtered users: {len(user_df)}')
        
        old_item_ids = list(item_id_map.keys())
        item_df = item_df[item_df.index.isin(old_item_ids)].copy()
        item_df.index = item_df.index.map(item_id_map)
        item_df.index.name = 'item_id'
        logger.info(f'Filtered items: {len(item_df)}')
        
        logger.info('Appending history information...')
        dfs = append_his_info([train_df, dev_df, test_df], neg=True)
        logger.info('Completed append history information to interactions')
        
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
            user_df.to_csv(os.path.join(dir, 'user.csv'))
            item_df.to_csv(os.path.join(dir, 'item.csv'))
            all_df.to_csv(os.path.join(dir, 'all.csv'), index=False)
            test_one_per_user.to_csv(os.path.join(dir, 'test.csv'), index=False)
            logger.info('Successfully saved all CSV files (user.csv, item.csv, all.csv, test.csv)')
            logger.info('all.csv contains only user interaction columns')
            logger.info('test.csv contains evaluation samples without redundant columns')
            
            logger.info('Saving ID mappings for RecBole compatibility...')
            user_mapping_df = pd.DataFrame([
                {'preprocessed_id': new_id, 'original_id': old_id}
                for old_id, new_id in user_id_map.items()
            ])
            item_mapping_df = pd.DataFrame([
                {'preprocessed_id': new_id, 'original_id': old_id}
                for old_id, new_id in item_id_map.items()
            ])
            
            user_mapping_df.to_csv(os.path.join(dir, 'user_id_mapping.csv'), index=False)
            item_mapping_df.to_csv(os.path.join(dir, 'item_id_mapping.csv'), index=False)
            logger.info(f'Saved ID mappings: user_id_mapping.csv ({len(user_mapping_df)} users), '
                       f'item_id_mapping.csv ({len(item_mapping_df)} items)')
            
        except Exception as e:
            logger.error(f'Error saving CSV files: {e}')
            raise
            
    except Exception as e:
        logger.error(f'Error processing ml-100k dataset: {e}')
        raise