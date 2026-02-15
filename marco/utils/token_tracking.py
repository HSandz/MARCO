from typing import Dict, Any
from loguru import logger
import json
import time

class TokenTracker:
    
    def __init__(self):
        self.task_stats: Dict[str, Dict] = {}
        self.current_task_id: str = None
        self.start_time: float = None
        self.agent_last_counts: Dict[str, Dict[str, int]] = {}
        
    def start_task(self, task_id: str, task_info: Dict[str, Any] = None) -> None:
        self.current_task_id = task_id
        self.start_time = time.time()
        
        self.agent_last_counts = {}
        
        self.task_stats[task_id] = {
            'task_info': task_info or {},
            'start_time': self.start_time,
            'end_time': None,
            'duration': None,
            'agents': {},
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'total_api_calls': 0,
            'models_used': set(),
        }
        
        logger.info(f"Started token tracking for task: {task_id}")
        
    def reset_agent_stats(self, system) -> None:
        agent_properties = [
            'analyst', 'reflector', 
            'planner', 'solver'
        ]
        
        for prop_name in agent_properties:
            if hasattr(system, prop_name):
                agent = getattr(system, prop_name)
                if agent and hasattr(agent, 'get_llm_instances'):
                    llm_instances = agent.get_llm_instances()
                    for llm_name, llm in llm_instances.items():
                        if hasattr(llm, 'reset_usage_stats'):
                            llm.reset_usage_stats()
        
        if hasattr(system, 'agents') and isinstance(system.agents, dict):
            for agent_name, agent in system.agents.items():
                if agent and hasattr(agent, 'get_llm_instances'):
                    llm_instances = agent.get_llm_instances()
                    for llm_name, llm in llm_instances.items():
                        if hasattr(llm, 'reset_usage_stats'):
                            llm.reset_usage_stats()
        
        if hasattr(system, 'agent_coordinator') and hasattr(system.agent_coordinator, 'agents'):
            for agent_name, agent in system.agent_coordinator.agents.items():
                if agent and hasattr(agent, 'get_llm_instances'):
                    llm_instances = agent.get_llm_instances()
                    for llm_name, llm in llm_instances.items():
                        if hasattr(llm, 'reset_usage_stats'):
                            llm.reset_usage_stats()
        
        self.agent_last_counts = {}
        logger.debug("Reset all agent LLM usage stats")
        
    def collect_agent_stats(self, agent_name: str, llm) -> None:
        if self.current_task_id is None:
            logger.warning("No active task for token tracking")
            return
            
        if not hasattr(llm, 'get_usage_stats'):
            logger.warning(f"LLM {llm.__class__.__name__} doesn't support usage tracking")
            return
            
        stats = llm.get_usage_stats()
        
        if stats['api_calls'] == 0:
            return
            
        task_data = self.task_stats[self.current_task_id]
        
        agent_key = f"{self.current_task_id}:{agent_name}"
        last_counts = self.agent_last_counts.get(agent_key, {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'api_calls': 0
        })
        
        delta_input = stats['total_input_tokens'] - last_counts['total_input_tokens']
        delta_output = stats['total_output_tokens'] - last_counts['total_output_tokens']
        delta_total = stats['total_tokens'] - last_counts['total_tokens']
        delta_calls = stats['api_calls'] - last_counts['api_calls']
        
        if delta_calls > 0:
            task_data['agents'][agent_name] = stats.copy()
            
            if hasattr(llm, 'get_detailed_usage_stats'):
                detailed_stats = llm.get_detailed_usage_stats()
                task_data['agents'][agent_name].update({
                    'api_calls_with_actual_counts': detailed_stats.get('api_calls_with_actual_counts', 0),
                    'api_calls_with_estimated_counts': detailed_stats.get('api_calls_with_estimated_counts', 0),
                    'accuracy_rate': detailed_stats.get('accuracy_rate', 0.0)
                })
            
            task_data['total_input_tokens'] += delta_input
            task_data['total_output_tokens'] += delta_output
            task_data['total_tokens'] += delta_total
            task_data['total_api_calls'] += delta_calls
            model_value = stats.get('model', 'unknown')
            task_data['models_used'].add(model_value)
            
            self.agent_last_counts[agent_key] = {
                'total_input_tokens': stats['total_input_tokens'],
                'total_output_tokens': stats['total_output_tokens'],
                'total_tokens': stats['total_tokens'],
                'api_calls': stats['api_calls']
            }
            
            logger.debug(f"Collected stats for {agent_name}: +{delta_calls} calls, +{delta_total} tokens (cumulative: {stats['api_calls']} calls, {stats['total_tokens']} tokens)")
        
    def collect_system_stats(self, system) -> None:
        if self.current_task_id is None:
            logger.warning("No active task for token tracking")
            return
        
        agent_properties = [
            'analyst', 'reflector', 
            'planner', 'solver'
        ]
        
        for prop_name in agent_properties:
            if hasattr(system, prop_name):
                agent = getattr(system, prop_name)
                if agent and hasattr(agent, 'get_llm_instances'):
                    llm_instances = agent.get_llm_instances()
                    for llm_name, llm in llm_instances.items():
                        self.collect_agent_stats(llm_name, llm)
        
        if hasattr(system, 'agents') and isinstance(system.agents, dict):
            for agent_name, agent in system.agents.items():
                if agent and hasattr(agent, 'get_llm_instances'):
                    llm_instances = agent.get_llm_instances()
                    for llm_name, llm in llm_instances.items():
                        self.collect_agent_stats(llm_name, llm)
        
        if hasattr(system, 'agent_coordinator') and hasattr(system.agent_coordinator, 'agents'):
            for agent_name, agent in system.agent_coordinator.agents.items():
                if agent and hasattr(agent, 'get_llm_instances'):
                    llm_instances = agent.get_llm_instances()
                    for llm_name, llm in llm_instances.items():
                        self.collect_agent_stats(llm_name, llm)
                    
    def end_task(self) -> Dict[str, Any]:
        if self.current_task_id is None:
            logger.warning("No active task to end")
            return {}
            
        task_data = self.task_stats[self.current_task_id]
        task_data['end_time'] = time.time()
        task_data['duration'] = task_data['end_time'] - task_data['start_time']
        
        task_data['models_used'] = list(task_data['models_used'])
        
        total_api_calls_with_actual = 0
        total_api_calls_with_estimates = 0
        
        for agent_name, agent_stats in task_data['agents'].items():
            actual_calls = agent_stats.get('api_calls_with_actual_counts', 0)
            estimated_calls = agent_stats.get('api_calls_with_estimated_counts', 0)
            total_api_calls_with_actual += actual_calls
            total_api_calls_with_estimates += estimated_calls
        
        overall_accuracy = total_api_calls_with_actual / max(task_data['total_api_calls'], 1)
        
        logger.info(f"Task {self.current_task_id} completed:")
        logger.info(f"  Duration: {task_data['duration']:.2f}s")
        logger.info(f"  Total tokens: {task_data['total_tokens']}")
        logger.info(f"  API calls: {task_data['total_api_calls']}")
        logger.info(f"  API calls with actual token counts: {total_api_calls_with_actual}")
        logger.info(f"  API calls with estimated token counts: {total_api_calls_with_estimates}")
        logger.info(f"  Token count accuracy: {overall_accuracy:.1%}")
        logger.info(f"  Models used: {task_data['models_used']}")
        
        task_data['token_accuracy'] = {
            'api_calls_with_actual': total_api_calls_with_actual,
            'api_calls_with_estimates': total_api_calls_with_estimates,
            'accuracy_rate': overall_accuracy
        }
        
        current_stats = task_data.copy()
        self.current_task_id = None
        return current_stats
        
    def get_task_stats(self, task_id: str = None) -> Dict[str, Any]:
        if task_id is None:
            task_id = self.current_task_id
            
        if task_id is None or task_id not in self.task_stats:
            return {}
            
        return self.task_stats[task_id].copy()
        
    def get_all_stats(self) -> Dict[str, Dict]:
        return self.task_stats.copy()
        
    def save_stats(self, filepath: str) -> None:
        data = {}
        for task_id, stats in self.task_stats.items():
            data[task_id] = stats.copy()
            if isinstance(data[task_id]['models_used'], set):
                data[task_id]['models_used'] = list(data[task_id]['models_used'])
                
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        logger.info(f"Token tracking stats saved to: {filepath}")
        
    def reset(self) -> None:
        self.task_stats = {}
        self.current_task_id = None
        self.start_time = None
        self.agent_last_counts = {}
        logger.info("Token tracker reset")

token_tracker = TokenTracker()
