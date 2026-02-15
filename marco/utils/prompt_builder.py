import ast
import pandas as pd
from typing import Any
from loguru import logger

class PromptBuilder:
    
    def __init__(self, data_dir: str, dataset: str = 'ml-100k'):
        self.data_dir = data_dir
        self.dataset = dataset
        
        self.item_df = self._load_item_metadata()
        self.user_df = self._load_user_metadata()
    
    def _load_item_metadata(self) -> pd.DataFrame:
        import os
        item_path = os.path.join(self.data_dir, 'item.csv')
        
        if not os.path.exists(item_path):
            logger.warning(f"Item metadata not found at {item_path}")
            return None
        
        item_df = pd.read_csv(item_path, index_col=0)
        logger.info(f"Loaded {len(item_df)} items from {item_path}")
        return item_df
    
    def _load_user_metadata(self) -> pd.DataFrame:
        import os
        user_path = os.path.join(self.data_dir, 'user.csv')
        
        if not os.path.exists(user_path):
            logger.info(f"No user metadata found at {user_path}, will use placeholder")
            return None
        
        user_df = pd.read_csv(user_path, index_col=0)
        logger.info(f"Loaded {len(user_df)} users from {user_path}")
        return user_df
    
    def _parse_list_field(self, value: Any) -> list:
        if pd.isna(value) or value == 'None' or value == '':
            return []
        
        if isinstance(value, list):
            return value
        
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return parsed
                return [parsed]
            except (ValueError, SyntaxError):
                logger.warning(f"Could not parse list field: {value}")
                return []
        
        return []
    
    def get_user_profile(self, user_id: int) -> str:
        if self.user_df is not None and user_id in self.user_df.index:
            if 'user_profile' in self.user_df.columns:
                profile = self.user_df.loc[user_id, 'user_profile']
                if pd.notna(profile) and profile != '':
                    return profile
        return "unknown"
    
    def get_item_attributes(self, item_id: int) -> str:
        if self.item_df is not None and item_id in self.item_df.index:
            if 'item_attributes' in self.item_df.columns:
                attrs = self.item_df.loc[item_id, 'item_attributes']
                if pd.notna(attrs) and attrs != '':
                    return attrs
        return f"Item {item_id}: unknown"
    
    def format_history(self, history_ids: Any, history_ratings: Any, history_summary: Any = None, max_his: int = 10) -> str:
        history_ids = self._parse_list_field(history_ids)
        history_ratings = self._parse_list_field(history_ratings)
        history_summary = self._parse_list_field(history_summary) if history_summary is not None else None
        
        if not history_ids:
            return "None"
        
        history_ids = history_ids[-max_his:]
        history_ratings = history_ratings[-max_his:] if len(history_ratings) >= len(history_ids) else history_ratings
        if history_summary is not None:
            history_summary = history_summary[-max_his:] if len(history_summary) >= len(history_ids) else history_summary
        
        history_lines = []
        for i, item_id in enumerate(history_ids):
            item_attr = self.get_item_attributes(item_id)
            rating = history_ratings[i] if i < len(history_ratings) else 'unknown'
            
            if history_summary is not None and i < len(history_summary):
                summary = history_summary[i]
                history_lines.append(f"{item_attr}, UserComments: {summary} (rating: {rating})")
            else:
                history_lines.append(f"{item_attr} (rating: {rating})")
        
        return "\n".join(history_lines)
    
    def format_candidates(self, candidate_ids: Any) -> str:
        candidate_ids = self._parse_list_field(candidate_ids)
        
        if not candidate_ids:
            return ""
        
        candidate_lines = []
        for item_id in candidate_ids:
            item_attr = self.get_item_attributes(item_id)
            candidate_lines.append(f"{item_id}: {item_attr}")
        
        return "\n".join(candidate_lines)
    
    def build_prompt_fields(self, row: pd.Series, max_his: int = 10) -> dict:
        fields = {}
        
        user_id = row.get('user_id')
        if pd.notna(user_id):
            fields['user_profile'] = self.get_user_profile(user_id)
        else:
            fields['user_profile'] = 'unknown'
        
        history_ids = row.get('history_item_id')
        history_ratings = row.get('history_rating')
        history_summary = row.get('history_summary')
        fields['history'] = self.format_history(history_ids, history_ratings, history_summary, max_his)
        
        item_id = row.get('item_id')
        if pd.notna(item_id):
            fields['target_item_attributes'] = self.get_item_attributes(item_id)
        else:
            fields['target_item_attributes'] = 'unknown'
        
        candidate_ids = row.get('candidate_item_id')
        if pd.notna(candidate_ids):
            fields['candidate_item_attributes'] = self.format_candidates(candidate_ids)
        else:
            fields['candidate_item_attributes'] = ''
        
        return fields
