import pandas as pd
from typing import Optional

from marco.tools.base import Tool

class InteractionRetriever(Tool):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        data_path = self.config['data_path']
        assert data_path is not None, 'Data path not found in config.'
        self.data = pd.read_csv(data_path, sep=',')
        assert 'user_id' in self.data.columns, 'user_id not found in data.'
        assert 'item_id' in self.data.columns, 'item_id not found in data.'
        self.user_history = None
        self.item_history = None

    def reset(self, user_id: Optional[int] = None, item_id: Optional[int] = None, *args, **kwargs) -> None:
        if user_id is not None and item_id is not None:
            data_sample = self.data[(self.data['user_id'] == user_id) & (self.data['item_id'] == item_id)]
            
            if len(data_sample) == 0:
                user_data = self.data[self.data['user_id'] == user_id]
                if len(user_data) > 0:
                    index = user_data.index[-1]
                    from loguru import logger
                    logger.trace(f'User {user_id} & item {item_id} pair not found (likely a candidate item). Using user\'s last interaction at index {index} as cutoff.')
                else:
                    index = len(self.data)
                    from loguru import logger
                    logger.debug(f'User {user_id} not found in interaction data. Using full dataset.')
            elif len(data_sample) > 1:
                index = data_sample.index[0]
                from loguru import logger
                logger.info(f'Multiple entries for user {user_id} & item {item_id}. Using first at index {index}.')
            else:
                index = data_sample.index[0]
            
            partial_data = self.data.iloc[:index]
            self.user_history = partial_data.groupby('user_id')['item_id'].apply(list).to_dict()
            self.user_rating = partial_data.groupby('user_id')['rating'].apply(list).to_dict()
            self.item_history = partial_data.groupby('item_id')['user_id'].apply(list).to_dict()
            self.item_rating = partial_data.groupby('item_id')['rating'].apply(list).to_dict()
        else:
            self.partial_data = None
            self.user_history = None
            self.user_rating = None
            self.item_history = None
            self.item_rating = None

    def user_retrieve(self, user_id: int, k: int, *args, **kwargs) -> str:
        if self.user_history is None:
            raise ValueError('User history not found. Please reset the user_id and item_id.')
        if user_id not in self.user_history:
            return f'No history found for user {user_id}.'
        user_his = self.user_history[user_id]
        retrieved = user_his[-k:]
        retrieved_rating = self.user_rating[user_id][-k:]
        return f'Retrieved {len(retrieved)} items that user {user_id} interacted with before: {", ".join(map(str, retrieved))} with ratings: {", ".join(map(str, retrieved_rating))}'

    def item_retrieve(self, item_id: int, k: int, *args, **kwargs) -> str:
        if self.item_history is None:
            raise ValueError('Item history not found. Please reset the user_id and item_id.')
        if item_id not in self.item_history:
            return f'No history found for item {item_id}.'
        item_his = self.item_history[item_id]
        retrieved = item_his[-k:]
        retrieved_rating = self.item_rating[item_id][-k:]
        return f'Retrieved {len(retrieved)} users that interacted with item {item_id} before: {", ".join(map(str, retrieved))} with ratings: {", ".join(map(str, retrieved_rating))}'
