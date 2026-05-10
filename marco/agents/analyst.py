from typing import Any, Dict
from loguru import logger

from marco.agents.base import ToolAgent
from marco.tools import InfoDatabase, InteractionRetriever
from marco.utils import read_json, get_rm, parse_action

class Analyst(ToolAgent):
    def __init__(self, config_path: str = None, config: dict = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if config is not None:
            agent_config = config
        else:
            assert config_path is not None, "Either config_path or config must be provided"
            agent_config = read_json(config_path)
        
        tool_config: dict[str, dict] = get_rm(agent_config, 'tool_config', {})
        self.get_tools(tool_config)
        self.max_turns = get_rm(agent_config, 'max_turns', 15)
        self.analyst = self.get_LLM(config=agent_config)
        self.json_mode = self.analyst.json_mode
        self.queried_users = set()
        self.queried_items = set()
        self.gathered_info = {}
        self.execution_context = None
        self.reset()

    @staticmethod
    def required_tools() -> dict[str, type]:
        return {
            'info_retriever': InfoDatabase,
            'interaction_retriever': InteractionRetriever,
        }

    @property
    def info_retriever(self) -> InfoDatabase:
        return self.tools['info_retriever']

    @property
    def interaction_retriever(self) -> InteractionRetriever:
        return self.tools['interaction_retriever']

    @property
    def analyst_prompt(self) -> str:
        if self.json_mode:
            return self.prompts['analyst_prompt_json']
        else:
            return self.prompts['analyst_prompt']

    @property
    def analyst_examples(self) -> str:
        if self.json_mode:
            return self.prompts['analyst_examples_json']
        else:
            return self.prompts['analyst_examples']

    @property
    def analyst_fewshot(self) -> str:
        if self.json_mode:
            return self.prompts['analyst_fewshot_json']
        else:
            return self.prompts['analyst_fewshot']

    @property
    def hint(self) -> str:
        if 'analyst_hint' not in self.prompts:
            return ''
        return self.prompts['analyst_hint']

    def _generate_summary_from_gathered_info(self) -> str:
        if not self.gathered_info:
            return "No information gathered."
        
        summary_parts = []
        
        user_info = {}
        history_item_info = {}
        candidate_item_info = {}
        user_histories = {}
        item_histories = {}
        user_preferences = {}
        
        history_item_ids = set()
        if hasattr(self, 'execution_context') and self.execution_context and 'data_sample' in self.execution_context:
            try:
                data_sample = self.execution_context['data_sample']
                if 'history_item_id' in data_sample:
                    history_item_id_value = data_sample['history_item_id']
                    if isinstance(history_item_id_value, str):
                        history_item_ids = set(eval(history_item_id_value))
                    elif isinstance(history_item_id_value, (list, set)):
                        history_item_ids = set(history_item_id_value)
                    logger.debug(f"Extracted history item IDs from execution context data_sample: {sorted(history_item_ids)}")
            except Exception as e:
                logger.warning(f"Failed to extract history_item_id from execution_context: {e}")
                history_item_ids = set()
        
        if not history_item_ids:
            for key, value in self.gathered_info.items():
                if key.startswith("user_history_"):
                    import re
                    item_ids_match = re.search(r'before:\s*([\d,\s]+)', value)
                    if item_ids_match:
                        item_ids_str = item_ids_match.group(1)
                        history_item_ids.update(int(iid.strip()) for iid in item_ids_str.split(',') if iid.strip())
        
        for key, value in self.gathered_info.items():
            if key.startswith("user_history_"):
                user_histories[key] = value
                user_id = key.replace("user_history_", "")
                user_preferences[user_id] = self._analyze_user_preferences(user_id, value)
            elif key.startswith("item_history_"):
                item_histories[key] = value
            elif key.startswith("user_"):
                user_info[key] = value
            elif key.startswith("item_") and not key.startswith("item_history_"):
                item_id = int(key.replace("item_", ""))
                if item_id in history_item_ids:
                    history_item_info[key] = value
                else:
                    candidate_item_info[key] = value
        
        if user_info or user_preferences:
            summary_parts.append("User Information:")
            for key, value in user_info.items():
                summary_parts.append(f"  - {key.replace('_', ' ').title()}: {value}")
            for user_id, preference_text in user_preferences.items():
                if preference_text:
                    summary_parts.append(f"  - {preference_text}")
        
        task_type = getattr(self.system, 'task', 'sr') if hasattr(self, 'system') and self.system else 'sr'
        if user_histories and task_type != 'sr':
            summary_parts.append("User History:")
            for key, value in user_histories.items():
                summary_parts.append(f"  - {key.replace('_', ' ').title()}: {value}")
        
        task_type = getattr(self.system, 'task', 'sr') if hasattr(self, 'system') and self.system else 'sr'
        if history_item_info and task_type != 'sr':
            summary_parts.append("User's Historical Items:")
            for key, value in history_item_info.items():
                summary_parts.append(f"  - {key.replace('_', ' ').title()}: {value}")
        
        if candidate_item_info:
            summary_parts.append("Candidate Items:")
            for key, value in candidate_item_info.items():
                summary_parts.append(f"  - {key.replace('_', ' ').title()}: {value}")
        
        if item_histories:
            summary_parts.append("Item History:")
            for key, value in item_histories.items():
                summary_parts.append(f"  - {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(summary_parts)

    def _enhance_user_history(self, raw_observation: str, user_id: int) -> str:
        try:
            if "No history found" in raw_observation:
                return raw_observation
            
            parts = raw_observation.split(" with ratings: ")
            if len(parts) != 2:
                return None
            
            item_part = parts[0].split("before: ")
            if len(item_part) != 2:
                return None
            
            item_ids = [int(iid.strip()) for iid in item_part[1].split(",")]
            ratings = [float(r.strip()) for r in parts[1].split(",")]
            
            if len(item_ids) != len(ratings) or len(item_ids) == 0:
                return None
            
            item_rating_pairs = list(zip(item_ids, ratings))
            item_rating_pairs.sort(key=lambda x: x[1], reverse=True)
            
            max_rating = item_rating_pairs[0][1]
            high_rated_items = [item_id for item_id, rating in item_rating_pairs if rating >= max_rating - 0.5]
            
            if high_rated_items and max_rating >= 4.0:
                priority_guidance = f"\n[Priority: Analyze items {', '.join(map(str, high_rated_items[:3]))} first - highest rated items (rating: {max_rating})]"
                return raw_observation + priority_guidance
            else:
                return raw_observation
            
        except Exception as e:
            logger.warning(f"Failed to enhance user history for user {user_id}: {e}")
            return None

    def _enhance_item_history(self, raw_observation: str, item_id: int) -> str:
        try:
            if "No history found" in raw_observation:
                return raw_observation
            
            parts = raw_observation.split(" with ratings: ")
            if len(parts) != 2:
                return None
            
            user_part = parts[0].split("before: ")
            if len(user_part) != 2:
                return None
            
            user_ids = [int(uid.strip()) for uid in user_part[1].split(",")]
            ratings = [float(r.strip()) for r in parts[1].split(",")]
            
            if len(user_ids) != len(ratings) or len(user_ids) == 0:
                return None
            
            user_rating_pairs = list(zip(user_ids, ratings))
            user_rating_pairs.sort(key=lambda x: x[1], reverse=True)
            top_users = user_rating_pairs[:min(3, len(user_rating_pairs))]
            
            avg_rating = sum(ratings) / len(ratings)
            
            user_summaries = []
            for user_id, rating in top_users:
                user_profile = self.info_retriever.user_info(user_id)
                
                if user_profile and "not found" not in user_profile.lower():
                    user_summaries.append(f"(rated {rating}): {user_profile[:150]}")
            
            if user_summaries:
                summary = f"Item {item_id} has {len(user_ids)} interactions (avg rating: {avg_rating:.1f}). Top users: {' | '.join(user_summaries)}"
            else:
                summary = f"Item {item_id} has {len(user_ids)} interactions (avg rating: {avg_rating:.1f}) but user profiles unavailable."
            
            return summary
            
        except Exception as e:
            logger.warning(f"Failed to enhance item history for item {item_id}: {e}")
            return None

    def _build_analyst_prompt(self, **kwargs) -> str:
        command_count = len(self._history)
        remaining_steps = self.max_turns - command_count
        
        repetition_warning = ""
        if len(self._history) >= 2:
            recent_commands = [turn['command'] for turn in self._history[-2:]]
            if len(set(recent_commands)) == 1:
                command_name = recent_commands[0].split('"')[3] if '"' in recent_commands[0] else "command"
                repetition_warning = f"\nSTOP REPEATING: You queried '{command_name}' twice. This wastes tokens. Pick a DIFFERENT command or FINISH your analysis now."
        
        command_summary = ""
        if len(self._history) > 0:
            unique_commands = []
            for turn in self._history:
                if turn['command'] not in unique_commands:
                    unique_commands.append(turn['command'])
            
            command_types = set()
            for cmd in unique_commands:
                try:
                    if '"type":"' in cmd:
                        cmd_type = cmd.split('"type":"')[1].split('"')[0]
                        command_types.add(cmd_type)
                except:
                    pass
            
            if command_types:
                command_summary = f"\n\nCOMMAND HISTORY: You've already used: {', '.join(sorted(command_types))}"
                command_summary += f"\n\nDO NOT repeat these. Use a DIFFERENT command or Finish."
            else:
                command_summary = f"\n\nCOMMANDS EXECUTED: {len(unique_commands)} different commands"
                command_summary += "\n\nDO NOT repeat. Try something different."
        
        context_info = ""
        
        finish_hint = ""
        if len(self.gathered_info) >= 2:
            finish_hint = f"\nREMEMBER: You have information about {len(self.gathered_info)} entities. If you believe you have enough information for analysis, STOP and use Finish command instead of querying more."
        
        task_context_info = ""
        
        if 'task_context' in kwargs and kwargs['task_context']:
            base_prompt_content = f"{kwargs['task_context']}\n\nCommands: UserInfo[id], ItemInfo[id], UserHistory[id], ItemHistory[id], Finish[result]\nGather comprehensive information, avoid duplicates, then Finish.\n\nTarget: {kwargs.get('analyse_type', 'user')} {kwargs.get('id', '')}\n{self.history}"
            task_context_info = f"\n\nSPECIFIC FOCUS: {kwargs['task_context']}"
            logger.debug(f"Using task_context for base prompt: {kwargs['task_context']}")
        else:
            base_prompt_content = self.prompts['analyst_base_prompt'].format(
                examples=self.analyst_examples,
                fewshot=self.analyst_fewshot,
                history=self.history,
                max_step=self.max_turns,
                step=command_count + 1,
                remaining_steps=remaining_steps,
                hint=self.hint if len(self._history) + 1 >= self.max_turns else '',
                **kwargs
            )
            logger.debug("Using standard template, no task_context")
        
        prompt = self.analyst_prompt.format(
            analyst_base_prompt=base_prompt_content
        )
        
        step_info = f"\nYou are at step {command_count + 1}/{self.max_turns}. Remaining steps: {remaining_steps}."
        if remaining_steps <= 3:
            step_info += " You should consider finishing your analysis soon."
        
        return prompt + context_info + finish_hint + task_context_info + step_info + repetition_warning + command_summary

    def _prompt_analyst(self, **kwargs) -> str:
        analyst_prompt = self._build_analyst_prompt(**kwargs)
        command = self.analyst(analyst_prompt)
        return command

    def command(self, command: str) -> None:
        logger.debug(f'Command: {command}')
        
        if len(self._history) >= 3:
            recent_commands = [turn['command'] for turn in self._history[-3:]]
            if all(cmd == command for cmd in recent_commands):
                logger.warning(f'Detected excessive repetitive command: {command}. Forcing finish.')
                summary = self._generate_summary_from_gathered_info()
                self.finish(summary)
                return
            
            if len(self._history) >= 6:
                last_6_commands = [turn['command'] for turn in self._history[-6:]]
                if (last_6_commands[0] == last_6_commands[2] == last_6_commands[4] and 
                    last_6_commands[1] == last_6_commands[3] == last_6_commands[5] and
                    last_6_commands[0] != last_6_commands[1]):
                    logger.warning(f'Detected strict alternating repetitive pattern: {last_6_commands}. Forcing finish.')
                    summary = self._generate_summary_from_gathered_info()
                    self.finish(summary)
                    return
        
        if len(self.gathered_info) >= 12 and len(self._history) >= 15:
            try:
                recent_actions = [parse_action(h['command'] if isinstance(h, dict) and 'command' in h else str(h), json_mode=self.json_mode)[0] 
                                 for h in self._history[-6:] if h and (isinstance(h, dict) or str(h).strip())]
            except Exception as e:
                logger.debug(f"Error parsing recent actions: {e}")
                recent_actions = []
            if len(set(recent_actions)) <= 2 and len(recent_actions) >= 4:
                summary = self._generate_summary_from_gathered_info()
                logger.warning(f"Forced finish due to repetitive pattern: gathered_info={len(self.gathered_info)}, history={len(self._history)}, recent_actions={recent_actions}")
                self.finish(summary)
                return
        
        log_head = ''
        try:
            action_type, argument = parse_action(command, json_mode=self.json_mode)
        except Exception as e:
            logger.error(f"parse_action failed for command '{command}': {type(e).__name__}: {e}")
            action_type = 'Invalid'
            argument = None
        
        if action_type.lower() == 'userinfo':
            try:
                if argument is None:
                    raise ValueError("Argument cannot be None")
                query_user_id = int(argument)
                
                if query_user_id in self.queried_users:
                    observation = f"User {query_user_id} information already retrieved. Use gathered information instead."
                    log_head = f':orange[Skipped duplicate UserInfo query for user] :red[{query_user_id}]:orange[...]\n- '
                else:
                    observation = self.info_retriever.user_info(user_id=query_user_id)
                    self.queried_users.add(query_user_id)
                    self.gathered_info[f"user_{query_user_id}"] = observation
                    log_head = f':violet[Look up UserInfo of user] :red[{query_user_id}]:violet[...]\n- '
            except (ValueError, TypeError):
                observation = f"Invalid user id: {argument}. Please provide a valid user ID number."
                log_head = ':red[Invalid UserInfo command]:red[...]\n- '
        elif action_type.lower() == 'iteminfo':
            try:
                if argument is None:
                    raise ValueError("Argument cannot be None")
                query_item_id = int(argument)
                
                if query_item_id in self.queried_items:
                    observation = f"Item {query_item_id} information already retrieved. Use gathered information instead."
                    log_head = f':orange[Skipped duplicate ItemInfo query for item] :red[{query_item_id}]:orange[...]\n- '
                else:
                    observation = self.info_retriever.item_info(item_id=query_item_id)
                    self.queried_items.add(query_item_id)
                    self.gathered_info[f"item_{query_item_id}"] = observation
                    log_head = f':violet[Look up ItemInfo of item] :red[{query_item_id}]:violet[...]\n- '
            except (ValueError, TypeError):
                observation = f"Invalid item id: {argument}. Please provide a valid item ID number."
                log_head = ':red[Invalid ItemInfo command]:red[...]\n- '
        elif action_type.lower() == 'userhistory':
            try:
                if argument is None:
                    raise ValueError("Argument cannot be None")
                query_user_id = int(argument)
                history_key = f"user_history_{query_user_id}"
                
                if history_key in self.gathered_info:
                    observation = f"User {query_user_id} history already retrieved. Use gathered information instead."
                    log_head = f':orange[Skipped duplicate UserHistory query for user] :red[{query_user_id}]:orange[...]\n- '
                else:
                    raw_observation = self.interaction_retriever.user_retrieve(user_id=query_user_id, k=10)
                    
                    enhanced_observation = self._enhance_user_history(raw_observation, query_user_id)
                    observation = enhanced_observation if enhanced_observation else raw_observation
                    
                    self.gathered_info[history_key] = observation
                    log_head = f':violet[Look up UserHistory of user] :red[{query_user_id}]:violet[...]\n- '
            except (ValueError, TypeError):
                observation = f"Invalid user id: {argument}. Please provide a valid user ID number."
                log_head = ':red[Invalid UserHistory command]:red[...]\n- '
        elif action_type.lower() == 'itemhistory':
            try:
                if argument is None:
                    raise ValueError("Argument cannot be None")
                query_item_id = int(argument)
                history_key = f"item_history_{query_item_id}"
                
                if history_key in self.gathered_info:
                    observation = f"Item {query_item_id} history already retrieved. Use gathered information instead."
                    log_head = f':orange[Skipped duplicate ItemHistory query for item] :red[{query_item_id}]:orange[...]\n- '
                else:
                    raw_observation = self.interaction_retriever.item_retrieve(item_id=query_item_id, k=10)
                    
                    enhanced_observation = self._enhance_item_history(raw_observation, query_item_id)
                    observation = enhanced_observation if enhanced_observation else raw_observation
                    
                    self.gathered_info[history_key] = observation
                    log_head = f':violet[Look up ItemHistory of item] :red[{query_item_id}]:violet[...]\n- '
            except (ValueError, TypeError):
                observation = f"Invalid item id: {argument}. Please provide a valid item ID number."
                log_head = ':red[Invalid ItemHistory command]:red[...]\n- '
        elif action_type.lower() == 'finish':
            if isinstance(argument, dict):
                if 'content' in argument:
                    finish_content = str(argument['content'])
                else:
                    finish_content = str(argument)
            elif isinstance(argument, list):
                finish_content = ', '.join(str(item) for item in argument)
            else:
                finish_content = str(argument) if argument is not None else "Analysis completed"

            detailed_analysis = self._generate_detailed_analysis(finish_content)
            observation = self.finish(results=detailed_analysis)
            log_head = ':violet[Finish with results]:\n- '
            logger.info(f'Observation: {observation}')
        else:
            observation = f'Unknown command type: {action_type}.'
            if action_type.lower() != 'finish':
                logger.debug(f'Observation: {observation}')
        self.observation(observation, log_head)
        turn = {
            'command': command,
            'observation': observation,
        }
        self._history.append(turn)

    def _generate_detailed_analysis(self, original_finish_content: str) -> str:
        if not self.gathered_info:
            return original_finish_content
        
        analysis_parts = []
        
        if original_finish_content and original_finish_content.strip().lower() != 'analysis':
            analysis_parts.append(f"Initial Analysis: {original_finish_content}")
        
        user_info_parts = []
        user_history_parts = []
        history_item_info_parts = []
        candidate_item_info_parts = []
        user_preferences = {}
        
        history_item_ids = set()
        if hasattr(self, 'execution_context') and self.execution_context and 'data_sample' in self.execution_context:
            try:
                data_sample = self.execution_context['data_sample']
                if 'history_item_id' in data_sample:
                    history_item_id_value = data_sample['history_item_id']
                    if isinstance(history_item_id_value, str):
                        history_item_ids = set(eval(history_item_id_value))
                    elif isinstance(history_item_id_value, (list, set)):
                        history_item_ids = set(history_item_id_value)
                    logger.debug(f"Extracted history item IDs from execution context data_sample: {sorted(history_item_ids)}")
            except Exception as e:
                logger.warning(f"Failed to extract history_item_id from execution_context: {e}")
                history_item_ids = set()
        
        if not history_item_ids:
            for key, value in self.gathered_info.items():
                if key.startswith('user_history_'):
                    import re
                    item_ids_match = re.search(r'before:\s*([\d,\s]+)', value)
                    if item_ids_match:
                        item_ids_str = item_ids_match.group(1)
                        history_item_ids.update(int(iid.strip()) for iid in item_ids_str.split(',') if iid.strip())
        
        for key, value in self.gathered_info.items():
            if key.startswith('user_') and not key.startswith('user_history_'):
                user_id = key.replace('user_', '')
                user_info_parts.append(f"User {user_id}: {value}")
            elif key.startswith('user_history_'):
                user_id = key.replace('user_history_', '')
                user_history_parts.append(f"User {user_id} History: {value}")
                user_preferences[user_id] = self._analyze_user_preferences(user_id, value)
            elif key.startswith('item_') and not key.startswith('item_history_'):
                item_id = int(key.replace('item_', ''))
                if item_id in history_item_ids:
                    history_item_info_parts.append(f"Item {item_id}: {value}")
                else:
                    candidate_item_info_parts.append(f"Item {item_id}: {value}")
        
        if user_info_parts or user_preferences:
            analysis_parts.append("User Information:")
            analysis_parts.extend([f"  - {part}" for part in user_info_parts])
            for user_id, preference_text in user_preferences.items():
                if preference_text:
                    analysis_parts.append(f"  - {preference_text}")
        
        if user_history_parts:
            task_type = getattr(self.system, 'task', 'sr') if hasattr(self, 'system') and self.system else 'sr'
            if task_type != 'sr':
                analysis_parts.append("User History:")
                analysis_parts.extend([f"  - {part}" for part in user_history_parts])
        
        task_type = getattr(self.system, 'task', 'sr') if hasattr(self, 'system') and self.system else 'sr'
        if history_item_info_parts and task_type != 'sr':
            analysis_parts.append("User's Historical Items (for context only:")
            analysis_parts.extend([f"  - {part}" for part in history_item_info_parts])
        
        if candidate_item_info_parts:
            analysis_parts.append("Candidate Items:")
            analysis_parts.extend([f"  - {part}" for part in candidate_item_info_parts])
        
        detailed_analysis = "\n".join(analysis_parts)
        
        if not detailed_analysis.strip():
            return original_finish_content
        
        return detailed_analysis
    
    def _analyze_user_preferences(self, user_id: str, history_observation: str) -> str:
        try:
            if "No history found" in history_observation or "Retrieved 0 items" in history_observation:
                return ""
            
            user_basic_info = self.gathered_info.get(f"user_{user_id}", "")
            
            age = gender = occupation = None
            if user_basic_info:
                import re
                age_match = re.search(r'Age:\s*(\d+)', user_basic_info)
                gender_match = re.search(r'Gender:\s*(\w+)', user_basic_info)
                occupation_match = re.search(r'Occupation:\s*([\w\s]+?)(?:;|$)', user_basic_info)
                
                if age_match:
                    age = age_match.group(1)
                if gender_match:
                    gender = gender_match.group(1)
                if occupation_match:
                    occupation = occupation_match.group(1).strip()
            
            parts = history_observation.split(" with ratings: ")
            if len(parts) != 2:
                return ""
            
            item_part = parts[0].split("before: ")
            if len(item_part) != 2:
                return ""
            
            ratings_part = parts[1]
            if "[Priority:" in ratings_part:
                ratings_part = ratings_part.split("[Priority:")[0].strip()
            
            item_ids = [int(iid.strip()) for iid in item_part[1].split(",")]
            ratings = [float(r.strip()) for r in ratings_part.split(",")]
            
            if len(item_ids) != len(ratings) or len(item_ids) == 0:
                return ""
            
            genres_with_ratings = []
            for item_id in item_ids:
                item_key = f"item_{item_id}"
                if item_key in self.gathered_info:
                    item_info = self.gathered_info[item_key]
                    import re
                    genre_match = re.search(r'Genres:\s*([\w|]+)', item_info)
                    if genre_match:
                        genres_str = genre_match.group(1)
                        item_genres = [g.strip() for g in genres_str.split('|')]
                        idx = item_ids.index(item_id)
                        rating = ratings[idx]
                        genres_with_ratings.append((item_genres, rating))
            
            genre_counts = {}
            high_rated_genres = []
            
            for genres, rating in genres_with_ratings:
                if rating >= 4.0:
                    high_rated_genres.extend(genres)
                for genre in genres:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            if high_rated_genres:
                high_genre_counts = {}
                for genre in high_rated_genres:
                    high_genre_counts[genre] = high_genre_counts.get(genre, 0) + 1
                top_genres = sorted(high_genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                top_genre_names = [g[0] for g in top_genres]
            elif genre_counts:
                top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                top_genre_names = [g[0] for g in top_genres]
            else:
                top_genre_names = []
            
            high_rated_count = sum(1 for r in ratings if r >= 4.0)
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            
            summary_parts = []
            
            if age and gender and occupation:
                summary_parts.append(f"User {user_id} Preferences: {age}-year-old {gender} {occupation}")
            elif user_id:
                summary_parts.append(f"User {user_id} Preferences:")
            
            if top_genre_names:
                genre_text = ", ".join(top_genre_names[:-1]) + (" and " + top_genre_names[-1] if len(top_genre_names) > 1 else top_genre_names[0])
                summary_parts.append(f"enjoys {genre_text.lower()} movies")
            
            if high_rated_count > 0:
                if high_rated_count == len(ratings):
                    summary_parts.append(f"Rated all {len(ratings)} recent movies highly (4-5 stars)")
                elif high_rated_count >= len(ratings) * 0.7:
                    summary_parts.append(f"Rated {high_rated_count}/{len(ratings)} recent movies highly (4-5 stars)")
                else:
                    summary_parts.append(f"Rated {high_rated_count} movies highly out of {len(ratings)} recent views")
            elif avg_rating > 0:
                summary_parts.append(f"Average rating: {avg_rating:.1f} stars")
            
            if len(summary_parts) <= 1:
                return ""
            
            result = summary_parts[0] + " " + ", ".join(summary_parts[1:]) + "."
            return result
            
        except Exception as e:
            logger.warning(f"Failed to analyze user preferences for user {user_id}: {e}")
            return ""

    def forward(self, id: int, analyse_type: str, *args: Any, **kwargs: Any) -> str:
        assert self.system.data_sample is not None, "Data sample is not provided."
        assert 'user_id' in self.system.data_sample, "User id is not provided."
        assert 'item_id' in self.system.data_sample, "Item id is not provided."
        self.interaction_retriever.reset(user_id=self.system.data_sample['user_id'], item_id=self.system.data_sample['item_id'])
        
        consecutive_invalid_commands = 0
        max_invalid_commands = 3
        
        while not self.is_finished():
            try:
                command = self._prompt_analyst(id=id, analyse_type=analyse_type, **kwargs)
                
                if not isinstance(command, str):
                    logger.error(f"LLM returned non-string response: {type(command)} - {command}")
                    if isinstance(command, dict):
                        command = command.get('content', str(command))
                    else:
                        command = str(command)
                
                if self.json_mode and command.strip().startswith('['):
                    try:
                        import json
                        commands_array = json.loads(command.strip())
                        if isinstance(commands_array, list):
                            executed_in_array = set()
                            for cmd_obj in commands_array:
                                if isinstance(cmd_obj, dict) and 'type' in cmd_obj:
                                    individual_command = json.dumps(cmd_obj)
                                    
                                    if individual_command in executed_in_array:
                                        logger.info(f"Skipping duplicate command in array: {individual_command}")
                                        continue
                                    executed_in_array.add(individual_command)
                                    
                                    action_type, argument = parse_action(individual_command, json_mode=self.json_mode)
                                    
                                    if action_type.lower() == 'invalid':
                                        consecutive_invalid_commands += 1
                                        logger.warning(f"Invalid command in array: {individual_command}")
                                        continue
                                    else:
                                        consecutive_invalid_commands = 0
                                    
                                    self.command(individual_command)
                                    
                                    if self.is_finished():
                                        break
                            continue 
                        else:
                            logger.warning(f"Expected JSON array but got: {type(commands_array)}")
                    except Exception as e:
                        logger.error(f"Error parsing JSON array: {e}")
                
                action_type, argument = parse_action(command, json_mode=self.json_mode)
                
                if action_type.lower() == 'invalid':
                    consecutive_invalid_commands += 1
                    command_display = command[:200] + "..." if len(command) > 200 else command
                    logger.warning(f"Invalid command generated: {command_display} (attempt {consecutive_invalid_commands})")
                    
                    if consecutive_invalid_commands >= max_invalid_commands:
                        logger.error(f"Too many consecutive invalid commands. Forcing finish.")
                        summary = self._generate_summary_from_gathered_info()
                        self.finish(summary)
                        break
                    
                    if len(self.gathered_info) >= 2:
                        error_observation = f"Invalid format: '{command}'. You have gathered enough information. Try: {{\"type\":\"Finish\",\"content\":\"Your analysis in 2-3 sentences\"}}"
                    else:
                        error_observation = f"Invalid command format: '{command}'. Use: {{\"type\":\"COMMAND\",\"content\":ID}} where COMMAND is UserInfo, ItemInfo, UserHistory, ItemHistory, or Finish."
                    turn = {
                        'command': command,
                        'observation': error_observation,
                    }
                    self._history.append(turn)
                    continue
                else:
                    consecutive_invalid_commands = 0
                
                self.command(command)
            except Exception as e:
                import traceback
                logger.error(f"Error in analyst forward: {type(e).__name__}: {e}\nTraceback: {traceback.format_exc()}")
                self.finish(f"Analysis terminated due to error: {str(e)}")
                break
                
        if not self.finished:
            return "Analyst did not return any result."
        return self.results

    def invoke(self, argument: Any, json_mode: bool, task_context: str = None, execution_context: Dict[str, Any] = None, **kwargs) -> str:
        self.execution_context = execution_context
        self.execution_context = execution_context
        
        if json_mode:
            if not isinstance(argument, list) or len(argument) != 2:
                observation = "The argument of the action 'Analyse' should be a list with two elements: analyse type (user or item) and id."
                return observation
            else:
                analyse_type, id = argument
                if (isinstance(id, str) and 'user_' in id) or (isinstance(id, str) and 'item_' in id):
                    observation = f"Invalid id: {id}. Don't use the prefix 'user_' or 'item_'. Just use the id number only, e.g., 1, 2, 3, ..."
                    return observation
                elif analyse_type.lower() not in ['user', 'item']:
                    observation = f"Invalid analyse type: {analyse_type}. It should be either 'user' or 'item'."
                    return observation
                elif not isinstance(id, int):
                    observation = f"Invalid id: {id}. It should be an integer."
                    return observation
        else:
            if len(argument.split(',')) != 2:
                observation = "The argument of the action 'Analyse' should be a string with two elements separated by a comma: analyse type (user or item) and id."
                return observation
            else:
                analyse_type, id = argument.split(',')
                if 'user_' in id or 'item_' in id:
                    observation = f"Invalid id: {id}. Don't use the prefix 'user_' or 'item_'. Just use the id number only, e.g., 1, 2, 3, ..."
                    return observation
                elif analyse_type.lower() not in ['user', 'item']:
                    observation = f"Invalid analyse type: {analyse_type}. It should be either 'user' or 'item'."
                    return observation
                else:
                    try:
                        id = int(id)
                    except (ValueError, TypeError):
                        observation = f"Invalid id: {id}. The id should be an integer."
                        return observation
        
        return self(analyse_type=analyse_type, id=id, task_context=task_context, **kwargs)