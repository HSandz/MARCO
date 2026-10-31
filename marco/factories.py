from typing import Dict, Type, Any, Optional
from abc import ABC, abstractmethod

from marco.agents.base import Agent
from marco.agents import (
    Planner, Analyst, Solver, Reflector
)

class AgentFactory(ABC):
    @abstractmethod
    def create_agent(self, agent_type: str, config: Dict[str, Any], **kwargs) -> Agent:
        pass

class DefaultAgentFactory(AgentFactory):
    
    def __init__(self):
        self._agent_registry: Dict[str, Type[Agent]] = {
            'Analyst': Analyst,
            'Reflector': Reflector,
            'Planner': Planner,
            'Solver': Solver,
        }
    
    def register_agent(self, name: str, agent_class: Type[Agent]) -> None:
        self._agent_registry[name] = agent_class
    
    def create_agent(self, agent_type: str, config: Dict[str, Any], **kwargs) -> Agent:
        if agent_type not in self._agent_registry:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = self._agent_registry[agent_type]
        return agent_class(**config, **kwargs)
    
    def get_available_agents(self) -> Dict[str, Type[Agent]]:
        return self._agent_registry.copy()
