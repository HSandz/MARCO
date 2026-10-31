from typing import Dict, Any, Optional
from loguru import logger

from marco.agents.base import Agent
from marco.factories import AgentFactory

class AgentCoordinator:
    
    def __init__(self, agent_factory: AgentFactory):
        self.agent_factory = agent_factory
        self.agents: Dict[str, Agent] = {}
    
    def initialize_agents(self, agent_configs: Dict[str, Dict[str, Any]], **kwargs):
        import json
        import os
        from loguru import logger
        
        self.agents.clear()
        
        system = kwargs.get('system')
        logger.debug(f"🔍 initialize_agents: system={system is not None}")
        model_override = getattr(system, 'model_override', None) if system else None
        
        for agent_name, config in agent_configs.items():
            try:
                final_agent_config = config.copy()
                
                dataset = kwargs.get('dataset') or getattr(system, 'dataset', None)
                task = kwargs.get('task') or getattr(system, 'task', None)
                
                if dataset and 'config_path' in final_agent_config:
                    import json
                    with open(final_agent_config['config_path'], 'r') as f:
                        agent_config_content = f.read()
                    
                    if dataset:
                        agent_config_content = agent_config_content.replace('{dataset}', dataset)
                    if task:
                        agent_config_content = agent_config_content.replace('{task}', task)
                    
                    substituted_config = json.loads(agent_config_content)
                    final_agent_config['config'] = substituted_config
                
                if model_override and system:
                    if 'config_path' in config:
                        with open(config['config_path'], 'r') as f:
                            agent_llm_config = json.load(f)
                        
                        agent_llm_config = system._apply_model_override(agent_llm_config)
                        final_agent_config['config'] = agent_llm_config
                
                agent = self.agent_factory.create_agent(agent_name, final_agent_config, **kwargs)
                self.agents[agent_name] = agent
                logger.info(f"Initialized agent: {agent_name}")
            except Exception as e:
                logger.error(f"Failed to initialize agent {agent_name}: {e}")
                raise
    
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        return self.agents.get(agent_name)
    
    def reset_all_agents(self):
        for agent in self.agents.values():
            agent.reset()
    
    def execute_agent_action(self, agent_name: str, action: str, **kwargs) -> Any:
        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent {agent_name} not found")
        
        try:
            if hasattr(agent, action):
                method = getattr(agent, action)
                return method(**kwargs)
            else:
                raise AttributeError(f"Agent {agent_name} has no action {action}")
        except Exception as e:
            logger.error(f"Error executing {action} on {agent_name}: {e}")
            raise
