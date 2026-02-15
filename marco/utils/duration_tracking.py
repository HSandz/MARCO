from typing import Dict, Any, Optional
from loguru import logger
import time
from contextlib import contextmanager

class DurationTracker:
    
    def __init__(self):
        self.task_stats: Dict[str, Dict] = {}
        self.current_task_id: Optional[str] = None
        self.agent_durations: Dict[str, float] = {}
        self.agent_call_counts: Dict[str, int] = {}
        
    def start_task(self, task_id: str, task_info: Dict[str, Any] = None) -> None:
        self.current_task_id = task_id
        self.agent_durations = {}
        self.agent_call_counts = {}
        
        self.task_stats[task_id] = {
            'task_info': task_info or {},
            'agents': {},
            'total_duration': 0,
        }
        
        logger.info(f"Started duration tracking for task: {task_id}")
        
    def end_task(self) -> Dict[str, Any]:
        if self.current_task_id is None:
            logger.warning("No active task to end")
            return {}
            
        task_data = self.task_stats[self.current_task_id]
        
        task_data['agents'] = {
            agent_name: {
                'total_duration': duration,
                'call_count': self.agent_call_counts.get(agent_name, 0),
                'avg_duration_per_call': duration / max(self.agent_call_counts.get(agent_name, 1), 1)
            }
            for agent_name, duration in self.agent_durations.items()
        }
        
        task_data['total_duration'] = sum(self.agent_durations.values())
        
        current_stats = task_data.copy()
        self.current_task_id = None
        return current_stats
        
    @contextmanager
    def track_agent_call(self, agent_name: str):
        if self.current_task_id is None:
            yield
            return
            
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            if agent_name not in self.agent_durations:
                self.agent_durations[agent_name] = 0.0
                self.agent_call_counts[agent_name] = 0
                
            self.agent_durations[agent_name] += duration
            self.agent_call_counts[agent_name] += 1
            
            logger.debug(f"Agent '{agent_name}' call took {duration:.3f}s (total: {self.agent_durations[agent_name]:.3f}s, calls: {self.agent_call_counts[agent_name]})")
    
    def add_agent_duration(self, agent_name: str, duration: float) -> None:
        if self.current_task_id is None:
            return
            
        if agent_name not in self.agent_durations:
            self.agent_durations[agent_name] = 0.0
            self.agent_call_counts[agent_name] = 0
            
        self.agent_durations[agent_name] += duration
        self.agent_call_counts[agent_name] += 1
        
        logger.debug(f"Added {duration:.3f}s to agent '{agent_name}' (total: {self.agent_durations[agent_name]:.3f}s)")
    
    def get_agent_durations(self) -> Dict[str, float]:
        return self.agent_durations.copy()
    
    def get_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        return {
            agent_name: {
                'total_duration': duration,
                'call_count': self.agent_call_counts.get(agent_name, 0),
                'avg_duration_per_call': duration / max(self.agent_call_counts.get(agent_name, 1), 1)
            }
            for agent_name, duration in self.agent_durations.items()
        }
    
    def get_task_stats(self, task_id: str = None) -> Dict[str, Any]:
        if task_id is None:
            task_id = self.current_task_id
            
        if task_id is None or task_id not in self.task_stats:
            return {}
            
        return self.task_stats[task_id].copy()
        
    def reset(self) -> None:
        self.task_stats = {}
        self.current_task_id = None
        self.agent_durations = {}
        self.agent_call_counts = {}
        logger.info("Duration tracker reset")

duration_tracker = DurationTracker()
