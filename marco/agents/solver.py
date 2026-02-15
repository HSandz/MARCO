from typing import Any, Dict, Optional
from loguru import logger
from marco.agents.base import Agent

class Solver(Agent):

    def __init__(self, config_path: str = None, config: dict = None, prompt_config: str = None, prompts: dict = None, *args, **kwargs):
        super().__init__(prompts=prompts or {}, prompt_config=prompt_config, *args, **kwargs)
        self.solution_history = []
        
        if config is not None:
            agent_config = config
        else:
            assert config_path is not None, "Either config_path or config must be provided"
            from marco.utils import read_json
            agent_config = read_json(config_path)
        
        self.llm = self.get_LLM(config=agent_config)
        self.json_mode = getattr(self.llm, 'json_mode', False)
        
    def system_message(self, task: str, **kwargs) -> str:
        if 'solver_system_prompt' not in self.prompts:
            raise ValueError("solver_system_prompt not found in prompts config. Please ensure prompt_config is properly loaded.")
            
        base_message = self.prompts['solver_system_prompt'].format(task=task)
        
        guidance_key = f'solver_{task}_guidance'
        if guidance_key in self.prompts:
            base_message += self.prompts[guidance_key]
            
        return base_message
        
    def _extract_candidate_ids(self, task: str, **kwargs) -> list:
        if task != 'sr':
            return []
        
        candidate_ids = []
        
        if kwargs.get('data_sample') is not None and 'candidate_item_id' in kwargs['data_sample']:
            try:
                candidate_item_id_value = kwargs['data_sample']['candidate_item_id']
                if isinstance(candidate_item_id_value, str):
                    candidate_ids = list(eval(candidate_item_id_value))
                elif isinstance(candidate_item_id_value, (list, set)):
                    candidate_ids = list(candidate_item_id_value)
            except Exception as e:
                logger.warning(f"Failed to extract candidate_item_id from data_sample: {e}")
        
        if not candidate_ids and kwargs.get('input'):
            import re
            candidate_matches = re.findall(r'(\d+):\s*(?:Title|Brand|Business):', kwargs['input'])
            if candidate_matches:
                candidate_ids = [int(item_id) for item_id in candidate_matches]
        
        return candidate_ids
    
    def user_message(self, plan: str, worker_results: Dict[str, Any], task: str, **kwargs) -> str:
        worker_results_text = ""
        for step_var, result in worker_results.items():
            if "Retrieved" in result and "candidate items" in result and "Candidate items:" in result:
                candidate_items_start = result.find("Candidate items:")
                if candidate_items_start != -1:
                    result = result[candidate_items_start:]
            worker_results_text += f"{step_var}: {result}\n"
        
        original_query = ""
        if kwargs.get('input'):
            original_query = f"""
Original Query Data:
{kwargs['input']}
"""
        
        candidate_ids = self._extract_candidate_ids(task, **kwargs)
        candidate_ids_list = ""
        if candidate_ids:
            if 'mandatory_ids' in self.prompts:
                candidate_ids_list = self.prompts['mandatory_ids'].format(
                    num_candidates=len(candidate_ids),
                    candidate_ids=candidate_ids
                )
            else:
                candidate_ids_list = f"\nMANDATORY: You MUST rank ONLY these {len(candidate_ids)} candidate item IDs (in any order): {candidate_ids}\nDO NOT include any other item IDs in your ranking.\n"
        
        reflection_feedback = ""
        if 'solver_reflections' in kwargs and kwargs['solver_reflections']:
            reflection_feedback = f"\n\nREFLECTION FEEDBACK - RERANKING REQUIRED\n{kwargs['solver_reflections']}\n"
        
        if 'solver_user_prompt' not in self.prompts:
            raise ValueError("solver_user_prompt not found in prompts config.")
        
        base_prompt = self.prompts['solver_user_prompt'].format(
            plan=plan,
            worker_results=worker_results_text,
            original_query=original_query,
            reflection_feedback=reflection_feedback,
            task=task.upper()
        )
        
        return base_prompt + candidate_ids_list

    def forward(self, *args, **kwargs) -> str:
        plan = kwargs.get('plan', '')
        worker_results = kwargs.get('worker_results', {})
        task = kwargs.get('task', 'sr')
        return self.invoke(plan, worker_results, task, **kwargs)
    
    def invoke_ablation(self, query: str, analyst_insights: str, task: str, **kwargs) -> str:
        try:
            system_msg = self.system_message(task, **kwargs)
            user_msg = self.user_message_ablation(query, analyst_insights, task, **kwargs)
            
            if task == 'sr' and kwargs.get('data_sample') is not None and 'candidate_item_id' in kwargs['data_sample']:
                try:
                    candidate_item_id_value = kwargs['data_sample']['candidate_item_id']
                    if isinstance(candidate_item_id_value, str):
                        candidate_ids = list(eval(candidate_item_id_value))
                    elif isinstance(candidate_item_id_value, (list, set)):
                        candidate_ids = list(candidate_item_id_value)
                    else:
                        candidate_ids = []
                    
                    if candidate_ids:
                        logger.info(f"Final candidate list ({len(candidate_ids)} items): {candidate_ids}")
                except Exception as e:
                    logger.warning(f"Failed to extract candidate list for logging: {e}")
            
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            
            from marco.llms import OllamaLLM, OpenRouterLLM, GeminiLLM, HuggingFaceLLM
            if isinstance(self.llm, (OllamaLLM, OpenRouterLLM, GeminiLLM, HuggingFaceLLM)):
                combined_prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"
                solution = self.llm(combined_prompt)
            else:
                solution = self.llm(messages)
            
            self.solution_history.append({
                'query': query,
                'analyst_insights': analyst_insights,
                'task': task,
                'solution': solution,
                'kwargs': kwargs,
                'mode': 'ablation'
            })
            
            logger.debug(f"Generated ablation solution for {task}: {solution}")
            return solution
            
        except Exception as e:
            logger.error(f"Error in Solver.invoke_ablation: {e}")
            return f"Solution generation failed: {str(e)}"
    
    def user_message_ablation(self, query: str, analyst_insights: str, task: str, **kwargs) -> str:
        original_query = f"Original Query:\n{query}\n"
        
        candidate_ids = self._extract_candidate_ids(task, **kwargs)
        candidate_ids_list = ""
        if candidate_ids:
            if 'mandatory_ids' in self.prompts:
                candidate_ids_list = self.prompts['mandatory_ids'].format(
                    num_candidates=len(candidate_ids),
                    candidate_ids=candidate_ids
                )
            else:
                candidate_ids_list = f"\nMANDATORY: You MUST rank ONLY these {len(candidate_ids)} candidate item IDs (in any order): {candidate_ids}\nDO NOT include any other item IDs in your ranking.\n"
        
        reflection_feedback = ""
        if 'solver_reflections' in kwargs and kwargs['solver_reflections']:
            reflection_feedback = f"\n\nREFLECTION FEEDBACK - RERANKING REQUIRED\n{kwargs['solver_reflections']}\n"
        
        if 'solver_user_prompt' not in self.prompts:
            raise ValueError("solver_user_prompt not found in prompts config.")
        
        base_prompt = self.prompts['solver_user_prompt'].format(
            original_query=original_query,
            analyst_insights=analyst_insights,
            reflection_feedback=reflection_feedback,
            task=task.upper()
        )
        
        return base_prompt + candidate_ids_list
    
    def invoke(self, plan: str, worker_results: Dict[str, Any], task: str, **kwargs) -> str:
        try:
            system_msg = self.system_message(task, **kwargs)
            user_msg = self.user_message(plan, worker_results, task, **kwargs)
            
            if task == 'sr' and kwargs.get('data_sample') is not None and 'candidate_item_id' in kwargs['data_sample']:
                try:
                    candidate_item_id_value = kwargs['data_sample']['candidate_item_id']
                    if isinstance(candidate_item_id_value, str):
                        candidate_ids = list(eval(candidate_item_id_value))
                    elif isinstance(candidate_item_id_value, (list, set)):
                        candidate_ids = list(candidate_item_id_value)
                    else:
                        candidate_ids = []
                    
                    if candidate_ids:
                        logger.info(f"Final candidate list ({len(candidate_ids)} items): {candidate_ids}")
                except Exception as e:
                    logger.warning(f"Failed to extract candidate list for logging: {e}")
            
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            
            from marco.llms import OllamaLLM, OpenRouterLLM, GeminiLLM, HuggingFaceLLM
            if isinstance(self.llm, (OllamaLLM, OpenRouterLLM, GeminiLLM, HuggingFaceLLM)):
                combined_prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"
                solution = self.llm(combined_prompt)
            else:
                solution = self.llm(messages)
            
            self.solution_history.append({
                'plan': plan,
                'worker_results': worker_results,
                'task': task,
                'solution': solution,
                'kwargs': kwargs
            })
            
            logger.debug(f"Generated solution for {task}: {solution}")
            return solution
            
        except Exception as e:
            logger.error(f"Error in Solver.invoke: {e}")
            return f"Solution generation failed: {str(e)}"
    
    def extract_final_answer(self, solution: str, task: str) -> Any:
        try:
            if task == 'sr':
                import json
                import re
                
                cleaned_solution = solution.strip()
                if '```' in cleaned_solution:
                    parts = cleaned_solution.split('```')
                    for part in parts:
                        part_stripped = part.strip()
                        if part_stripped.startswith('json'):
                            part_stripped = part_stripped[4:].strip()
                        
                        if part_stripped.startswith('{') and part_stripped.endswith('}'):
                            cleaned_solution = part_stripped
                            break
                
                try:
                    logger.debug(f"Attempting to parse cleaned_solution: {cleaned_solution[:500]}")
                    data = json.loads(cleaned_solution)
                    if isinstance(data, dict) and 'ranked_items' in data:
                        items = data['ranked_items']
                        if isinstance(items, list):
                            try:
                                converted_items = []
                                for item in items:
                                    if isinstance(item, int):
                                        converted_items.append(item)
                                    elif isinstance(item, str) and item.isdigit():
                                        converted_items.append(int(item))
                                    else:
                                        raise ValueError(f"Invalid item format: {item}")
                                logger.info(f"Extracted {len(converted_items)} items from JSON response")
                                return converted_items
                            except (ValueError, TypeError) as e:
                                logger.error(f"Failed to convert ranked_items to integers: {e}")
                                logger.error(f"Original items: {items}")
                        else:
                            logger.error(f"Invalid ranked_items format (not a list): {items}")
                    else:
                        logger.error(f"JSON response missing 'ranked_items' key: {data}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.debug(f"Solution (raw, first 1000 chars): {solution[:1000]}")
                    
                    try:
                        logger.info("Attempting JSON recovery from truncated response...")
                        fixed_solution = cleaned_solution
                        
                        open_braces = fixed_solution.count('{') - fixed_solution.count('}')
                        open_quotes_count = fixed_solution.count('"') % 2
                        
                        if open_quotes_count == 1:
                            fixed_solution += '"'
                        if open_braces > 0:
                            fixed_solution += '}' * open_braces
                        
                        try:
                            data = json.loads(fixed_solution)
                            if isinstance(data, dict) and 'ranked_items' in data:
                                items = data['ranked_items']
                                if isinstance(items, list):
                                    converted_items = []
                                    for item in items:
                                        if isinstance(item, int):
                                            converted_items.append(item)
                                        elif isinstance(item, str) and item.isdigit():
                                            converted_items.append(int(item))
                                    if converted_items:
                                        logger.info(f"Recovered {len(converted_items)} items by closing incomplete JSON")
                                        return converted_items
                        except json.JSONDecodeError:
                            pass
                        
                        items_match = re.search(r'"ranked_items"\s*:\s*\[([^\]]+)\]', solution)
                        if items_match:
                            items_str = items_match.group(1)
                            item_numbers = re.findall(r'\d+', items_str)
                            if item_numbers:
                                converted_items = [int(num) for num in item_numbers]
                                logger.info(f"Recovered {len(converted_items)} items from incomplete JSON via regex extraction")
                                return converted_items
                    except Exception as recovery_e:
                        logger.error(f"JSON recovery failed: {recovery_e}")
                
                logger.warning("Could not extract items from solution, returning empty list")
                return []
                    
            elif task == 'rp':
                import json
                import re
                
                cleaned_solution = solution.strip()
                if '```' in cleaned_solution:
                    parts = cleaned_solution.split('```')
                    for part in parts:
                        part_stripped = part.strip()
                        if part_stripped.startswith('json'):
                            part_stripped = part_stripped[4:].strip()
                        if part_stripped.startswith('{') and part_stripped.endswith('}'):
                            cleaned_solution = part_stripped
                            break
                
                try:
                    data = json.loads(cleaned_solution)
                    if isinstance(data, dict):
                        rating = data.get('predicted_rating') or data.get('rating') or data.get('score')
                        if rating is not None:
                            rating = float(rating)
                            if 0 <= rating <= 5:
                                logger.debug(f"Extracted rating {rating} from JSON response")
                                return rating
                except json.JSONDecodeError:
                    logger.debug("JSON parsing failed, trying regex patterns")
                
                rating_patterns = [
                    r'(?:rating|rate|score)(?:\s+is|\s*:|\s+of)?\s*(\d+(?:\.\d+)?)',
                    r'(\d+(?:\.\d+)?)\s*(?:out of|/)\s*5',
                    r'(?:would|will|should)\s+(?:rate|give)\s+(?:it|this)?\s*(?:a|an)?\s*(\d+(?:\.\d+)?)',
                    r'(?:final|predicted)\s+(?:rating|score)(?:\s+is|\s*:)?\s*(\d+(?:\.\d+)?)'
                ]
                
                for pattern in rating_patterns:
                    match = re.search(pattern, solution, re.IGNORECASE)
                    if match:
                        rating = float(match.group(1))
                        if 0 <= rating <= 5:
                            logger.debug(f"Extracted rating {rating} using pattern: {pattern}")
                            return rating
                
                all_numbers = re.findall(r'\d+(?:\.\d+)?', solution)
                for num_str in reversed(all_numbers):
                    try:
                        num = float(num_str)
                        if 0 <= num <= 5:
                            logger.warning(f"Using fallback extraction: found {num} in solution")
                            return num
                    except ValueError:
                        continue
                
                logger.warning(f"Could not extract valid rating from solution, using default 3.0")
                return 3.0
                
        except Exception as e:
            logger.error(f"Error extracting final answer: {e}")
            
        if task == 'sr':
            return [1311, 627, 71, 700, 938, 258, 858, 1091]
        elif task == 'rp':
            return 3.0
        else:
            return solution
    
    def get_last_solution(self) -> Optional[str]:
        if self.solution_history:
            return self.solution_history[-1]['solution']
        return None
