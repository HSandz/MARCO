import pandas as pd
from typing import Optional
from loguru import logger

from marco.tools.base import Tool

class CandidateRetriever(Tool):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.current_sample = None
        self._item_info = None
        item_info_path = self.config.get('item_info', None)
        logger.info(f"CandidateRetriever config: {self.config}")
        logger.info(f"item_info_path from config: {item_info_path}")
        if item_info_path is not None:
            try:
                self._item_info = pd.read_csv(item_info_path, sep=',')
                logger.info(f"Loaded item info from {item_info_path} with {len(self._item_info)} items")
            except Exception as e:
                logger.error(f"Failed to load item info from {item_info_path}: {e}")
        else:
            logger.warning(f"No item_info path in config - will return IDs only")
        
    def reset(self, data_sample: Optional[pd.Series] = None, *args, **kwargs) -> None:
        self.current_sample = data_sample
        
    def retrieve_candidates(self, user_id: int, k: int = -1, *args, **kwargs) -> str:
        if self.current_sample is None:
            logger.warning("No data sample set. Call reset() with data_sample first.")
            return "No candidates available. Data sample not initialized."
        
        if 'candidate_item_id' not in self.current_sample:
            logger.warning(f"candidate_item_id field not found in data sample for user {user_id}")
            return f"No candidates available for user {user_id}."
        
        try:
            candidate_item_id_value = self.current_sample['candidate_item_id']
            
            if isinstance(candidate_item_id_value, str):
                candidate_items = eval(candidate_item_id_value)
            elif isinstance(candidate_item_id_value, (list, set)):
                candidate_items = list(candidate_item_id_value)
            else:
                logger.warning(f"Unexpected candidate_item_id format: {type(candidate_item_id_value)}")
                return f"No candidates available for user {user_id}."
            
            if k > 0 and len(candidate_items) > k:
                candidate_items = candidate_items[:k]
            
            logger.debug(f"Retrieved {len(candidate_items)} candidates for user {user_id}: {candidate_items}")
            
            if self._item_info is not None:
                logger.info(f"Formatting {len(candidate_items)} candidates WITH attributes")
                
                candidate_details = []
                
                for item_id in candidate_items:
                    item_data = self._item_info[self._item_info['item_id'] == item_id]
                    if not item_data.empty:
                        if 'genre' in item_data.columns and 'title' in item_data.columns:
                            # MovieLens format
                            title = str(item_data['title'].values[0])
                            genre = str(item_data['genre'].values[0])
                            candidate_details.append(f"{item_id} (Title: {title}, Genres: {genre})")
                        
                        elif 'name' in item_data.columns and 'city' in item_data.columns:
                            # Yelp format
                            name = str(item_data['name'].values[0])
                            categories = str(item_data['categories'].values[0]) if 'categories' in item_data.columns else "Unknown"
                            city = str(item_data['city'].values[0])
                            state = str(item_data['state'].values[0])
                            candidate_details.append(f"{item_id} (Business: {name}, Categories: {categories}, Location: {city}, {state})")
                        
                        elif 'brand' in item_data.columns and 'title' in item_data.columns:
                            # Amazon format (Beauty, Electronics, Video_Games)
                            title = str(item_data['title'].values[0])
                            brand = str(item_data['brand'].values[0])
                            price = str(item_data['price'].values[0]) if 'price' in item_data.columns else "Unknown"
                            categories = str(item_data['categories'].values[0]) if 'categories' in item_data.columns else "Unknown"
                            candidate_details.append(f"{item_id} (Title: {title}, Brand: {brand}, Price: {price}, Categories: {categories})")
                        
                        else:
                            if 'item_attributes' in item_data.columns:
                                attributes = str(item_data['item_attributes'].values[0])
                                candidate_details.append(f"{item_id} ({attributes})")
                            else:
                                candidate_details.append(str(item_id))
                    else:
                        candidate_details.append(str(item_id))
                
                result = f"Retrieved {len(candidate_items)} candidate items for user {user_id}:\n"
                result += "Candidate items:\n"
                for detail in candidate_details:
                    result += f"- {detail}\n"
                return result.strip()
            else:
                logger.warning(f"Returning candidates WITHOUT attributes (item_info not loaded)")
                return f"Retrieved {len(candidate_items)} candidate items for user {user_id}: {', '.join(map(str, candidate_items))}"
            
        except Exception as e:
            logger.error(f"Error retrieving candidates for user {user_id}: {e}")
            return f"Error retrieving candidates for user {user_id}: {str(e)}"
