import json
from typing import Any

def parse_action(action: str, json_mode: bool = False) -> tuple[str, Any]:
    if json_mode:
        try:
            action = action.strip()
            
            action_cleaned = action.replace('\\$', '$').replace('\\#', '#').replace('\\%', '%').replace('\\&', '&')
            
            try:
                json_action = json.loads(action_cleaned)
                if isinstance(json_action, list) and len(json_action) == 1:
                    json_action = json_action[0]
                
                if 'type' not in json_action:
                    return 'Invalid', None
                
                action_type = json_action['type']
                valid_types = ['Analyse', 'UserInfo', 'ItemInfo', 'UserHistory', 'ItemHistory', 'Finish']
                
                action_type_lower = action_type.lower()
                valid_action = None
                for valid_type in valid_types:
                    if valid_type.lower() == action_type_lower:
                        valid_action = valid_type
                        break
                
                if valid_action is None:
                    return 'Invalid', None
                
                content = json_action.get('content', None)
                if valid_action == 'Finish' and (content is None or content == ""):
                    return 'Invalid', None
                
                return valid_action, content
                
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                pass
            
            if '\n' in action:
                lines = action.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        action = line
                        break
                else:
                    json_content = ""
                    in_json = False
                    for line in lines:
                        line = line.strip()
                        if line.startswith('{'):
                            in_json = True
                            json_content = line
                        elif in_json:
                            json_content += " " + line
                            if line.endswith('}'):
                                action = json_content
                                break
            
            import re
            
            start_idx = action.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = -1
                in_string = False
                escape_next = False
                
                for i, char in enumerate(action[start_idx:], start_idx):
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        continue
                    
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                
                if end_idx != -1:
                    action = action[start_idx:end_idx]
                else:
                    pass
            
            action_cleaned = action.replace('\\$', '$').replace('\\#', '#').replace('\\%', '%').replace('\\&', '&')
            json_action = json.loads(action_cleaned)
            if isinstance(json_action, list) and len(json_action) == 1:
                json_action = json_action[0]
            
            if 'type' not in json_action:
                return 'Invalid', None
            
            action_type = json_action['type']
            valid_types = ['Analyse', 'UserInfo', 'ItemInfo', 'UserHistory', 'ItemHistory', 'Finish']
            
            action_type_lower = action_type.lower()
            valid_action = None
            for valid_type in valid_types:
                if valid_type.lower() == action_type_lower:
                    valid_action = valid_type
                    break
            
            if valid_action is None:
                return 'Invalid', None
            
            content = json_action.get('content', None)
            if valid_action == 'Finish' and (content is None or content == ""):
                return 'Invalid', None
            
            return valid_action, content
        except Exception as e:
            from loguru import logger
            logger.debug(f"JSON parsing error for action: '{action}', error: {e}")
            return 'Invalid', None
    else:
        pattern = r'^(\w+)\[(.*)\]$'
        match = re.match(pattern, action)

        if match:
            action_type = match.group(1)
            argument = match.group(2)
            return action_type, argument
        else:
            return 'Invalid', None

def parse_raw_answer(answer: str, *args, **kwargs) -> dict[str, bool | str]:
    return {
        'valid': True,
        'answer': answer
    }

def parse_rating_answer(answer: str | int | float, json_mode: bool = False, *args, **kwargs) -> dict[str, float | str]:
    try:
        answer = float(answer)
        if answer < 1 or answer > 5:
            return {
                'valid': False,
                'answer': 0,
                'message': 'Rating should be in range [1, 5].'
            }
    except (ValueError, TypeError):
        return {
            'valid': False,
            'answer': 0,
            'message': 'Rating should be a float number.'
        }
    except Exception:
        return {
            'valid': False,
            'answer': 0,
            'message': 'Other Exception when parsing rating.'
        }
    return {
        'valid': True,
        'answer': answer
    }

def parse_ranking_answer(answer: str | Any, gt_answer: int, n_candidate: int = None, json_mode: bool = False, *args, **kwargs) -> dict[str, bool | list[int]]:
    if not json_mode:
        candidates = answer.split(',')
    else:
        if isinstance(answer, list):
            candidates = answer
        elif isinstance(answer, str):
            if answer.strip().startswith('[') and answer.strip().endswith(']'):
                try:
                    import ast
                    parsed_list = ast.literal_eval(answer.strip())
                    if isinstance(parsed_list, list):
                        candidates = parsed_list
                    else:
                        candidates = answer.split(',')
                except (ValueError, SyntaxError):
                    candidates = answer.split(',')
            else:
                candidates = answer.split(',')
        else:
            return {
                'valid': False,
                'answer': [],
                'message': 'Answer should be a permutated list of candidate ids.'
            }
    
    try:
        length = len(candidates)
    except TypeError:
        return {
            'valid': False,
            'answer': [],
            'message': 'Answer should be a permutated list of candidate ids.'
        }
    except Exception:
        return {
            'valid': False,
            'answer': [],
            'message': 'Other Exception when parsing ranking answer.'
        }
    
    if n_candidate is not None and length != n_candidate:
        return {
            'valid': False,
            'answer': [],
            'message': f'Answer should contain only a list of {n_candidate} ids, which is the same as the number of candidates in the question.'
        }
    
    try:
        answer = [int(c) for c in candidates]
        return {
            'valid': True,
            'answer': answer
        }
    except (ValueError, TypeError):
        return {
            'valid': False,
            'answer': [],
            'message': f'The ids in the answer list should be integers. Received: {answer}. Valid format: [1063, 151, 274, 225, 609, 25] (array of integers, NOT string)'
        }
    
    return {
        'valid': True,
        'answer': answer
    }

def parse_answer(type: str, *args, **kwargs) -> dict[str, Any]:
    if type == 'rp':
        return parse_rating_answer(*args, **kwargs)
    elif type == 'sr':
        return parse_ranking_answer(*args, **kwargs)
    else:
        raise NotImplementedError(f'Unsupported task: {type}')

def init_answer(type: str) -> Any:
    if type == 'rp':
        return 0
    elif type == 'sr':
        return []
    else:
        raise NotImplementedError(f'Unsupported task: {type}')
