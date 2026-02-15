from typing import Any, Dict, List, Optional
from loguru import logger
from marco.agents.base import Agent

class Planner(Agent):
    
    def __init__(self, config_path: str = None, config: dict = None, prompt_config: str = None, prompts: dict = None, *args, **kwargs):
        super().__init__(prompts=prompts or {}, prompt_config=prompt_config, *args, **kwargs)
        self.plan_history = []
        
        if config is not None:
            agent_config = config
        else:
            assert config_path is not None, "Either config_path or config must be provided"
            from marco.utils import read_json
            agent_config = read_json(config_path)
        
        self.llm = self.get_LLM(config=agent_config)
        self.json_mode = getattr(self.llm, 'json_mode', False)
        
    def system_message(self, task: str, **kwargs) -> str:
        available_workers = []
        if hasattr(self, 'system') and self.system:
            if hasattr(self.system, 'analyst') and self.system.analyst:
                available_workers.append("Analyst")
        
        available_workers.append("Retriever")
        
        if not available_workers:
            available_workers = ["Analyst", "Retriever"]
        
        workers_desc = ""
        if "Analyst" in available_workers:
            workers_desc += "- Analyst: Analyzes user preferences, item features, or user-item interactions\n"
        if "Retriever" in available_workers:
            workers_desc += "- Retriever: Retrieves candidate items for a given user (MUST be called as 2nd step after user analysis)\n"
        
        if 'planner_system_prompt' not in self.prompts:
            raise ValueError("planner_system_prompt not found in prompts config. Please ensure prompt_config is properly loaded.")
            
        base_message = self.prompts['planner_system_prompt'].format(
            task=task,
            available_workers=available_workers,
            workers_desc=workers_desc.rstrip()
        )
        
        guidance_key = f'planner_{task}_guidance'
        if guidance_key in self.prompts:
            base_message += self.prompts[guidance_key]
            
        return base_message
        
    def user_message(self, query: str, task: str, **kwargs) -> str:
        context = ""
        if 'user_id' in kwargs:
            context += f"User ID: {kwargs['user_id']}\n"
        if 'item_id' in kwargs:
            context += f"Item ID: {kwargs['item_id']}\n"
        if 'n_candidate' in kwargs:
            context += f"Number of candidates: {kwargs['n_candidate']}\n"
        if 'history' in kwargs:
            context += f"User history available: Yes\n"
        
        reflection_context = ""
        if 'reflections' in kwargs and kwargs['reflections'].strip():
            reflection_context = f"\n{kwargs['reflections']}\n"
        
        planning_guidance = ""
        if task in ['sr', 'rp']:
            import re
            
            user_id_match = re.search(r'user[_\s]*id[:\]]*\s*(\d+)', query, re.IGNORECASE)
            
            if user_id_match:
                user_id = user_id_match.group(1)
                
                planning_guidance = f"\nREQUIRED PLAN STRUCTURE:\n"
                planning_guidance += f"#E1 = Analyst[Analyze user {user_id}'s profile, interaction history, and extract preferences]\n"
                planning_guidance += f"#E2 = Retriever[Retrieve candidate items for user {user_id}] (depends on #E1)\n"
                planning_guidance += f"#E3 = Analyst[Analyze 1st candidate item from #E2] (depends on #E2)\n"
                planning_guidance += f"#E4 = Analyst[Analyze 2nd candidate item from #E2] (depends on #E2)\n"
                planning_guidance += f"#E5 = Analyst[Analyze 3rd candidate item from #E2] (depends on #E2)\n"
                planning_guidance += f"... (continue for more candidate items: 4th, 5th, 6th, etc.)\n\n"
                planning_guidance += f"CRITICAL:\n"
                planning_guidance += f"- Step 1 MUST analyze the user\n"
                planning_guidance += f"- Step 2 MUST use Retriever to get candidate items\n"
                planning_guidance += f"- Steps 3+ MUST analyze each candidate using ORDINAL references (1st, 2nd, 3rd, etc.)\n"
                planning_guidance += f"- Do NOT use actual item IDs in the plan - use ordinal positions instead\n"
                planning_guidance += f"- Each item analysis step must depend on #E2\n"
        
        if 'planner_user_prompt' not in self.prompts:
            raise ValueError("planner_user_prompt not found in prompts config.")
            
        return self.prompts['planner_user_prompt'].format(
            task=task,
            context=context,
            query=query,
            planning_guidance=planning_guidance
        ) + reflection_context

    def forward(self, *args, **kwargs) -> str:
        query = kwargs.get('query', '')
        task = kwargs.get('task', 'sr')
        return self.invoke(query, task, **kwargs)
    
    def invoke(self, query: str, task: str, **kwargs) -> str:
        try:
            system_msg = self.system_message(task, **kwargs)
            user_msg = self.user_message(query, task, **kwargs)
            
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            
            from marco.llms import OllamaLLM, OpenRouterLLM, GeminiLLM, HuggingFaceLLM
            if isinstance(self.llm, (OllamaLLM, OpenRouterLLM, GeminiLLM, HuggingFaceLLM)):
                combined_prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"
                plan = self.llm(combined_prompt)
            else:
                plan = self.llm(messages)
            
            self.plan_history.append({
                'query': query,
                'task': task,
                'plan': plan,
                'kwargs': kwargs
            })
            
            logger.debug(f"Generated plan for {task}: {plan}")
            return plan
            
        except Exception as e:
            logger.error(f"Error in Planner.invoke: {e}")
            return f"Planning failed: {str(e)}"
    
    def parse_plan(self, plan: str) -> List[Dict[str, Any]]:
        steps = []
        lines = plan.split('\n')
        
        current_step = None
        worker_type_mapping = {}
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Plan: ') and '#E' in line:
                line = line[6:]
            
            if '=' in line and line.startswith('#E'):
                var_part, action_part = line.split('=', 1)
                variable = var_part.strip()
                action = action_part.strip()
                
                worker_type = 'Unknown'
                task_desc = "No description"
                
                if '[' in action and ']' in action:
                    worker_part = action.split('[')[0].strip()
                    task_desc = action.split('[')[1].split(']')[0]
                    
                    import re
                    task_desc = re.sub(r',\s*depends[_ ]on:\s*#E\d+', '', task_desc, flags=re.IGNORECASE).strip()
                    
                    worker_type = worker_part
                elif '[' in action:
                    worker_part = action.split('[')[0].strip()
                    worker_type = worker_part
                    task_desc = "Incomplete task description (truncated)"
                    continue
                    
                dependencies = []
                if 'depends_on:' in action:
                    dep_part = action.split('depends_on:')[1]
                    for dep in dep_part.split(','):
                        dep = dep.strip().replace(']', '').replace(')', '')
                        if dep.startswith('#E'):
                            dependencies.append(dep)
                
                current_step = {
                    'variable': variable,
                    'worker_type': worker_type,
                    'task_description': task_desc,
                    'dependencies': dependencies,
                    'raw_action': action
                }
                steps.append(current_step)
            
            elif line.startswith('#E') and '=' in line and 'Worker Type:' not in line:
                continue
            elif 'Worker Type:' in line and current_step:
                worker_type = line.split('Worker Type:')[1].strip()
                current_step['worker_type'] = worker_type
                worker_type_mapping[current_step['variable']] = worker_type
            elif 'Task:' in line and current_step:
                task_desc = line.split('Task:')[1].strip()
                current_step['task_description'] = task_desc
        
        for step in steps:
            if step['worker_type'] == 'Unknown':
                task_desc = step['task_description'].lower()
                if 'analyz' in task_desc or 'examine' in task_desc or 'pattern' in task_desc:
                    step['worker_type'] = 'Analyst'
                elif 'retrieve candidate' in task_desc or 'get candidate' in task_desc:
                    step['worker_type'] = 'Retriever'
                elif 'retrieve' in task_desc or 'candidate' in task_desc:
                    step['worker_type'] = 'Retriever'
                elif 'rank' in task_desc or 'score' in task_desc or 'order' in task_desc:
                    step['worker_type'] = 'Analyst'
                elif 'interpret' in task_desc or 'generate' in task_desc or 'recommend' in task_desc:
                    step['worker_type'] = 'Analyst'
                else:
                    step['worker_type'] = 'Analyst'
        
        return steps
    
    def extract_plan_only(self, plan: str) -> str:
        plan_lines = []
        lines = plan.split('\n')
        
        for line in lines:
            line_stripped = line.strip()
            
            if line_stripped.startswith('Plan: ') and '#E' in line_stripped:
                line_stripped = line_stripped[6:].strip()
            
            if line_stripped.startswith('#E') and '=' in line_stripped:
                plan_lines.append(line_stripped)
        
        if plan_lines:
            return '\n'.join(plan_lines)
        else:
            logger.warning("No plan lines (#E...) found in plan. Returning original plan.")
            return plan
    
    def get_last_plan(self) -> Optional[str]:
        if self.plan_history:
            return self.plan_history[-1]['plan']
        return None
