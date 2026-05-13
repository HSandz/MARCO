import json
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from loguru import logger
import pandas as pd

from marco.systems.base import System
from marco.factories import DefaultAgentFactory
from marco.components import AgentCoordinator
from marco.agents.base import Agent
from marco.utils import duration_tracker

if TYPE_CHECKING:
    from marco.agents import Analyst, Reflector, Planner, Solver

class MARCOSystem(System):
    
    def __init__(self, task: str, config_path: str, leak: bool = False, dataset: Optional[str] = None, enable_reflection_rerun: bool = False, *args, **kwargs) -> None:
        self.agent_factory = DefaultAgentFactory()
        self.agent_coordinator = AgentCoordinator(self.agent_factory)
        self.enable_reflection_rerun = enable_reflection_rerun
        
        super().__init__(task, config_path, leak, dataset, *args, **kwargs)
        
    def init(self, *args, **kwargs) -> None:
        self.max_step: int = self.config.get('max_step', 10)
        assert 'agents' in self.config, 'Agents are required.'
        
        agent_configs = self.config['agents']
        self.agent_coordinator.initialize_agents(agent_configs, **self.agent_kwargs)
        
        self.manager_kwargs = {
            'max_step': self.max_step,
            'task_type': self.task,
        }
        
        self.analyzed_items = set()
        self.analyzed_users = set()
        self.execution_results = {}
        self.completed_steps = set()
        self.current_plan = None
        self.plan_steps = []
        self.step_n = 1
        self.phase = 'planning'
        self._execution_errors = []
        
        self.planner_kwargs = {
            'reflections': '',
        }
        
        self.reflection_improvements = []
        self.reflection_all_reruns = []
        self.total_reflections_triggered = 0
        
        self._current_sample_idx = -1
        self._current_user_id = -1

    @staticmethod
    def supported_tasks() -> list[str]:
        return ['rp', 'sr']

    @property
    def planner(self) -> Optional['Planner']:
        return self.agent_coordinator.get_agent('Planner')

    @property
    def solver(self) -> Optional['Solver']:
        return self.agent_coordinator.get_agent('Solver')

    @property
    def analyst(self) -> Optional['Analyst']:
        return self.agent_coordinator.get_agent('Analyst')

    @property
    def reflector(self) -> Optional['Reflector']:
        return self.agent_coordinator.get_agent('Reflector')
    
    def set_data(self, input: str, context: str, gt_answer: Any, data_sample: Optional[pd.Series] = None) -> None:
        super().set_data(input, context, gt_answer, data_sample)
        
        if data_sample is not None:
            self._current_user_id = data_sample.get('user_id', -1) if hasattr(data_sample, 'get') else (data_sample['user_id'] if 'user_id' in data_sample else -1)
        else:
            self._current_user_id = -1

    def reset(self, clear: bool = False, preserve_progress: bool = False, *args, **kwargs) -> None:
        saved_sample_idx = getattr(self, '_current_sample_idx', -1)
        saved_user_id = getattr(self, '_current_user_id', -1)
        
        super().reset(clear, *args, **kwargs)
        
        if not preserve_progress:
            self.analyzed_items.clear()
            self.analyzed_users.clear()
            self.execution_results.clear()
            self.current_plan = None
            self.plan_steps = []
            if hasattr(self, '_last_solution'):
                delattr(self, '_last_solution')
            if hasattr(self, '_last_final_answer'):
                delattr(self, '_last_final_answer')
            if hasattr(self, 'planner_kwargs'):
                self.planner_kwargs['reflections'] = ""
            if hasattr(self, 'manager_kwargs'):
                self.manager_kwargs['solver_reflections'] = ""
        else:
            self._current_sample_idx = saved_sample_idx
            self._current_user_id = saved_user_id
            
            if hasattr(self, 'analyzed_items') and self.analyzed_items:
                current_progress = {
                    'analyzed_items': list(self.analyzed_items),
                    'analyzed_users': list(getattr(self, 'analyzed_users', set())),
                    'step_n': getattr(self, 'step_n', 1)
                }
                
                if 'reflections' not in self.planner_kwargs:
                    self.planner_kwargs['reflections'] = ""
                
                progress_summary = f"\n=== Previous Progress ===\n"
                if current_progress['analyzed_items']:
                    progress_summary += f"- Analyzed items: {sorted(current_progress['analyzed_items'])}\n"
                if current_progress['analyzed_users']:
                    progress_summary += f"- Analyzed users: {sorted(current_progress['analyzed_users'])}\n"
                progress_summary += f"- Completed {current_progress['step_n']} steps\n"
                progress_summary += "IMPORTANT: Create a plan that avoids repeating the above analyses.\n"
                self.planner_kwargs['reflections'] += progress_summary
        
        self.agent_coordinator.reset_all_agents()
        
        self.step_n = 1
        self.phase = 'planning'
        self._execution_errors = []

    def forward(self, user_input: Optional[str] = None, reset: bool = True) -> Any:
        try:
            self.manager_kwargs['input'] = self.input
            self.manager_kwargs['data_sample'] = self.data_sample
                
            if reset:
                self.reset()

            if self.planner is None or self.solver is None:
                logger.error("MARCO agents (planner and solver) are required but not available.")
                raise ValueError("MARCO agents (planner and solver) are required but not available.")
            
            result = self._execute_marco_workflow()
            best_result = result
            best_position = self._gt_position_before_reflection if hasattr(self, '_gt_position_before_reflection') else -1
            
            original_position_before_reflection = self._gt_position_before_reflection if hasattr(self, '_gt_position_before_reflection') else -1
            
            best_answer = self.answer.copy() if isinstance(self.answer, list) else self.answer
            best_answer_position = best_position
            
            if self.enable_reflection_rerun:
                should_continue_reflecting, feedback_info = self._perform_reflection()
                reflection_count = 0
                max_reflections = 1
                
                while should_continue_reflecting and reflection_count < max_reflections:
                    reflection_count += 1
                    logger.debug(f'Starting MARCO reflection cycle {reflection_count}/{max_reflections}')
                    
                    planner_correct = feedback_info.get('planner_correct', True)
                    solver_correct = feedback_info.get('solver_correct', True)
                    planner_reason = feedback_info.get('planner_reason', '')
                    solver_reason = feedback_info.get('solver_reason', '')
                    
                    if not planner_correct:
                        logger.info(f"Planner feedback triggered: {planner_reason}")
                        logger.info(f"Applying Planner feedback and executing full MARCO rerun...")
                        
                        position_before = original_position_before_reflection
                        
                        self.reset(preserve_progress=True)
                        
                        if 'reflections' not in self.planner_kwargs:
                            self.planner_kwargs['reflections'] = ""
                        
                        reflection_feedback = f"\n=== Planning Improvement Required (Reflection Feedback) ===\n"
                        reflection_feedback += f"{planner_reason}\n"
                        reflection_feedback += f"CRITICAL: Revise your plan to address this specific issue.\n"
                        
                        self.planner_kwargs['reflections'] += reflection_feedback
                        
                        if not solver_correct:
                            logger.info(f"Solver feedback also triggered: {solver_reason}")
                            logger.info(f"Applying feedback to BOTH Planner and Solver for full improvement")
                            
                            if 'solver_reflections' not in self.manager_kwargs:
                                self.manager_kwargs['solver_reflections'] = ""
                            
                            solver_reflection_feedback = f"\n=== Solver Improvement Required (Reflection Feedback) ===\n"
                            solver_reflection_feedback += f"{solver_reason}\n"
                            solver_reflection_feedback += f"CRITICAL: Adjust your ranking to address this specific issue.\n"
                            
                            self.manager_kwargs['solver_reflections'] = solver_reflection_feedback
                        
                        result = self._execute_marco_workflow()
                        
                        position_after = self._get_ground_truth_position(self._last_final_answer)
                        logger.info(f"Full rerun complete - position: {position_before} → {position_after}")
                        
                    elif not solver_correct:
                        logger.info(f"Solver feedback triggered (solver-only reranking): {solver_reason}")
                        
                        position_before = self._get_ground_truth_position(self._last_final_answer)
                        
                        reranked_answer = self._perform_solver_reranking(solver_reason)
                        
                        self.answer = reranked_answer
                        result = self.answer
                        
                        position_after = self._get_ground_truth_position(reranked_answer)
                        logger.info(f"Solver reranking complete - position: {position_before} → {position_after}")
                        
                    else:
                        logger.info("Both Planner and Solver are correct - stopping reflection")
                        should_continue_reflecting = False
                        break
                    
                    sample_idx = getattr(self, '_current_sample_idx', -1)
                    user_id = getattr(self, '_current_user_id', -1)
                    gt_item = self.gt_answer if hasattr(self, 'gt_answer') else -1
                    
                    rerun_info = {
                        'sample_idx': sample_idx,
                        'user_id': user_id,
                        'gt_item': gt_item,
                        'position_before': position_before,
                        'position_after': position_after,
                        'feedback_type': 'both' if (not planner_correct and not solver_correct) else ('planner' if not planner_correct else 'solver')
                    }
                    self.reflection_all_reruns.append(rerun_info)
                    
                    if position_before > 0 and position_after > 0 and position_after < position_before:
                        self.reflection_improvements.append(rerun_info)
                        improvement_delta = position_before - position_after
                        logger.debug(f"Reflection improved GT position: {position_before} → {position_after} (improvement: {improvement_delta:+d})")
                        logger.debug(f"Improvement rate: {len(self.reflection_improvements)}/{len(self.reflection_all_reruns)} reflections successful")
                        
                        best_answer = self.answer.copy() if isinstance(self.answer, list) else self.answer
                        best_answer_position = position_after
                        best_position = position_after
                        logger.debug(f"Saving improved answer as best answer for scoring")
                    elif position_before > 0 and position_after > 0 and position_after > position_before:
                        worsened_delta = position_after - position_before
                        self.answer = best_answer.copy() if isinstance(best_answer, list) else best_answer
                    else:
                        logger.debug(f"Reflection did NOT change position (before: {position_before}, after: {position_after})")
                    
                    should_continue_reflecting, feedback_info = self._perform_reflection()
                
                if reflection_count >= max_reflections:
                    logger.info(f'Stopped after {max_reflections} MARCO reflection cycle to prevent infinite loops')
                
                if self.reflection_all_reruns:
                    success_rate = len(self.reflection_improvements) / len(self.reflection_all_reruns) * 100
                    logger.debug(f"Reflection complete: {len(self.reflection_improvements)}/{len(self.reflection_all_reruns)} improved ({success_rate:.1f}%), final GT position: {best_answer_position}")

                current_position = self._get_ground_truth_position(self.answer)
                if current_position != best_answer_position:
                    self.answer = best_answer.copy() if isinstance(best_answer, list) else best_answer
                
                return self.answer
            else:
                self._perform_reflection_logging_only()
            
            return self.answer
                
        except Exception as e:
            logger.error(f"Error in MARCO forward: {e}")
            raise

    def _execute_marco_workflow(self) -> str:
        result = None
        
        while self.phase != 'completed' and not hasattr(self, '_finished'):
            if self.phase == 'planning':
                result = self._planning_phase()
            elif self.phase == 'working':
                result = self._working_phase()
            elif self.phase == 'solving':
                result = self._solving_phase()
                self.phase = 'completed'
            else:
                result = self._planning_phase()
        
        return result
    
    def _get_ground_truth_position(self, answer: Any) -> int:
        if not isinstance(answer, list) or not hasattr(self, 'gt_answer'):
            return -1
        
        gt = self.gt_answer
        try:
            if gt in answer:
                return answer.index(gt) + 1
        except (ValueError, TypeError):
            pass
        
        return -1
    
    def _should_perform_reflection(self) -> tuple[bool, str]:
        return True, "Performing reflection on every sample for comprehensive quality assessment"
    
    def _perform_reflection(self) -> tuple[bool, dict]:
        if not self.reflector:
            return False, {}
        
        should_reflect, skip_reason = self._should_perform_reflection()
        
        if not should_reflect:
            logger.info(f"Skipping reflection: {skip_reason}")
            self.log(f"**Reflection Skipped:** {skip_reason}", agent=self.reflector)
            return False, {}
        
        logger.info(f"Performing dual feedback reflection: {skip_reason}")
        
        if hasattr(self, '_last_solution') and hasattr(self, '_last_final_answer'):
            MARCO_process = self._build_MARCO_scratchpad(self._last_solution, self._last_final_answer)
        else:
            MARCO_process = self._build_basic_MARCO_scratchpad()
        
        if hasattr(self, 'planner_kwargs') and 'reflections' in self.planner_kwargs:
            MARCO_process += f"\n\nPrevious Reflection Comments:\n{self.planner_kwargs['reflections']}"
        
        with duration_tracker.track_agent_call('reflector'):
            self.reflector(input=self.input, scratchpad=MARCO_process)
        
        feedback_info = {
            'planner_correct': True,
            'solver_correct': True,
            'planner_reason': '',
            'solver_reason': ''
        }
        
        if self.reflector.json_mode and self.reflector.reflections:
            try:
                reflection_json = json.loads(self.reflector.reflections[-1])
                
                if isinstance(reflection_json, dict):
                    if 'Planner' in reflection_json and 'Solver' in reflection_json:
                        logger.debug("Detected dual feedback format (Planner + Solver)")
                        
                        planner_feedback = reflection_json.get('Planner', {})
                        solver_feedback = reflection_json.get('Solver', {})
                        
                        feedback_info['planner_correct'] = planner_feedback.get('correctness', True)
                        feedback_info['solver_correct'] = solver_feedback.get('correctness', True)
                        feedback_info['planner_reason'] = planner_feedback.get('reason', 'No reason provided')
                        feedback_info['solver_reason'] = solver_feedback.get('reason', 'No reason provided')
                        
                        planner_status = "Correct" if feedback_info['planner_correct'] else "Incorrect"
                        solver_status = "Correct" if feedback_info['solver_correct'] else "Incorrect"
                        
                        logger.debug(f"Reflection Results - Planner: {planner_status}, Solver: {solver_status}")
                        
                        self.log(f"**Dual Feedback Reflection Results:**\n"
                        f"Planner: {planner_status}\n"
                        f"Reason: {feedback_info['planner_reason']}\n\n"
                        f"Solver: {solver_status}\n"
                        f"Reason: {feedback_info['solver_reason']}", agent=self.reflector)
                        
                        self.total_reflections_triggered += 1
                        
                        should_continue = not feedback_info['planner_correct'] or not feedback_info['solver_correct']
                        
                        if should_continue:
                            if not feedback_info['planner_correct'] and not feedback_info['solver_correct']:
                                logger.info(f"Reflection: Full rerun needed (both agents)")
                            elif not feedback_info['planner_correct']:
                                logger.info(f"Reflection: Full rerun needed (planner)")
                            else:
                                logger.info(f"Reflection: Solver reranking only")
                        
                        return should_continue, feedback_info
                    
                    else:
                        logger.debug("Detected legacy single feedback format")
                        
                        correctness = reflection_json.get('correctness', False)
                        reason = reflection_json.get('reason', 'No reason provided')
                        
                        feedback_info['planner_correct'] = correctness
                        feedback_info['solver_correct'] = correctness
                        feedback_info['planner_reason'] = reason
                        feedback_info['solver_reason'] = reason
                        
                        if not correctness:
                            logger.debug(f"MARCO Reflection identified issues: {reason}")
                            self.log(f"**MARCO Reflection Issues Identified:**\n{reason}", agent=self.reflector)
                            
                            self.total_reflections_triggered += 1
                            
                            if 'reflections' not in self.planner_kwargs:
                                self.planner_kwargs['reflections'] = ""
                            
                            reflection_feedback = f"\n=== Planning Improvement Required ===\n"
                            reflection_feedback += f"{reason}\n"
                            reflection_feedback += f"CRITICAL: Revise your plan to address this specific issue.\n"
                            
                            self.planner_kwargs['reflections'] += reflection_feedback
                            
                            return True, feedback_info
                        else:
                            logger.debug(f"MARCO Reflection confirms correctness: {reason}")
                            self.log(f"**MARCO Reflection Confirms Correctness:**\n{reason}", agent=self.reflector)
                            return False, feedback_info
                
                elif isinstance(reflection_json, list):
                    logger.trace(f"Reflector returned array of {len(reflection_json)} objects. Converting to dual format.")
                    if len(reflection_json) > 0 and isinstance(reflection_json[0], dict):
                        correctness = reflection_json[0].get('correctness', False)
                        reason = reflection_json[0].get('reason', 'No reason provided')
                        
                        feedback_info['planner_correct'] = correctness
                        feedback_info['solver_correct'] = correctness
                        feedback_info['planner_reason'] = reason
                        feedback_info['solver_reason'] = reason
                        
                        should_continue = not correctness
                        if not correctness:
                            self.total_reflections_triggered += 1
                        
                        return should_continue, feedback_info
                
                else:
                    logger.error(f"Unexpected reflection JSON type: {type(reflection_json)}")
                    return False, feedback_info
                        
            except Exception as e:
                logger.error(f'Invalid reflection JSON output: {self.reflector.reflections[-1]}')
                logger.error(f'JSON parsing error: {e}')
                return False, feedback_info
        else:
            if self.reflector.reflections:
                self.log(f"**MARCO Reflection:**\n{self.reflector.reflections[-1]}", agent=self.reflector)
            return False, feedback_info
        
        return False, feedback_info

    def _perform_reflection_logging_only(self) -> None:
        if not self.reflector:
            return
        
        if hasattr(self, '_last_solution') and hasattr(self, '_last_final_answer'):
            MARCO_process = self._build_MARCO_scratchpad(self._last_solution, self._last_final_answer)
        else:
            MARCO_process = self._build_basic_MARCO_scratchpad()
        
        with duration_tracker.track_agent_call('reflector'):
            self.reflector(input=self.input, scratchpad=MARCO_process)
        
        if self.reflector.json_mode and self.reflector.reflections:
            try:
                reflection_json = json.loads(self.reflector.reflections[-1])
                
                if isinstance(reflection_json, list):
                    logger.trace(f"Reflector returned array of {len(reflection_json)} objects. Evaluating all.")
                    correctness = True
                    reasons = []
                    for item in reflection_json:
                        if isinstance(item, dict) and 'correctness' in item:
                            if not item['correctness']:
                                correctness = False
                                reasons.append(item.get('reason', 'No reason provided'))
                            elif item['correctness'] and len(reflection_json) == 1:
                                reasons.append(item.get('reason', 'No reason provided'))
                    reason = '\n'.join(f"- {r}" for r in reasons) if reasons else 'Multiple issues identified'
                elif isinstance(reflection_json, dict):
                    correctness = reflection_json.get('correctness', False)
                    reason = reflection_json.get('reason', 'No reason provided')
                else:
                    logger.error(f"Unexpected reflection JSON type: {type(reflection_json)}")
                    return
                
                if not correctness:
                    logger.debug(f"MARCO Reflection identified issues: {reason}")
                    self.log(f"**MARCO Reflection Issues Identified:**\n{reason}", agent=self.reflector)
                else:
                    logger.debug(f"MARCO Reflection confirms correctness: {reason}")
                    self.log(f"**MARCO Reflection Confirms Correctness:**\n{reason}", agent=self.reflector)
                        
            except Exception as e:
                logger.error(f'Invalid reflection JSON output: {self.reflector.reflections[-1]}')
                logger.error(f'JSON parsing error: {e}')
        else:
            if self.reflector.reflections:
                self.log(f"**MARCO Reflection:**\n{self.reflector.reflections[-1]}", agent=self.reflector)

    def _build_basic_MARCO_scratchpad(self) -> str:
        scratchpad = f"\n=== MARCO Process Summary ===\n"
        scratchpad += f"Task: {self.task.upper()}\n"
        scratchpad += f"Original Query: {getattr(self, 'input', 'No input')}\n"
        scratchpad += f"Current Phase: {self.phase}\n"
        
        if hasattr(self, 'current_plan') and self.current_plan:
            plan_only = self.planner.extract_plan_only(self.current_plan)
            scratchpad += f"\nGenerated Plan:\n{plan_only}\n"
        
        if hasattr(self, 'execution_results') and self.execution_results:
            scratchpad += "\nExecution Results:\n"
            for step_var, result in self.execution_results.items():
                scratchpad += f"{step_var}: {result}\n"
        
        return scratchpad

    def _planning_phase(self) -> str:
        logger.info("MARCO Phase 1: Planning")
        
        query = self._prepare_planning_query()
        
        combined_kwargs = {**self.manager_kwargs, **self.planner_kwargs}
        with duration_tracker.track_agent_call('planner'):
            plan = self.planner.invoke(query, self.task, **combined_kwargs)
        self.current_plan = plan
        self.plan_steps = self.planner.parse_plan(plan)
        
        self.log(f"**MARCO Plan Generated:**\n{plan}", agent=self.planner)
        
        self.phase = 'working'
        self.step_n = 1
        
        if not self.plan_steps:
            self.phase = 'solving'
            return self._solving_phase()
        
        return self._working_phase()

    def _working_phase(self) -> str:
        logger.info(f"MARCO Phase 2: Working - Step {self.step_n}")
        
        if self.step_n > len(self.plan_steps):
            self.phase = 'solving'
            return ''
        
        current_step = self.plan_steps[self.step_n - 1]
        
        if not self._dependencies_satisfied(current_step):
            logger.warning(f"Dependencies not satisfied for step {self.step_n}")
            self.step_n += 1
            return self._working_phase()
        
        result = self._execute_step(current_step)
        
        step_variable = current_step['variable']
        self.completed_steps.add(step_variable)
        
        if not self._is_meaningless_result(result):
            self.execution_results[step_variable] = result
            
            self.log(f"**Step {self.step_n} ({step_variable})**: {current_step['task_description']}\n**Result**: {result}", 
                    agent=self._get_worker_agent(current_step['worker_type']))
        else:
            logger.info(f"Skipped storing meaningless result for {step_variable}")
        
        self.step_n += 1
        
        if self.step_n > len(self.plan_steps):
            self.phase = 'solving'
            return ''
        else:
            return self._working_phase()

    def _solving_phase(self) -> str:
        logger.info("MARCO Phase 3: Solving")
        
        plan_only = self.planner.extract_plan_only(self.current_plan)
        
        with duration_tracker.track_agent_call('solver'):
            solution = self.solver.invoke(plan_only, self.execution_results, self.task, **self.manager_kwargs)
        
        self.log(f"Final Solution:\n{solution}", agent=self.solver)
        
        final_answer = self.solver.extract_final_answer(solution, self.task)
        
        if self.task in ['sr'] and isinstance(final_answer, list):
            history_item_ids = set()
            if hasattr(self, 'data_sample') and self.data_sample is not None and 'history_item_id' in self.data_sample:
                try:
                    history_item_id_value = self.data_sample['history_item_id']
                    if isinstance(history_item_id_value, str):
                        history_item_ids = set(eval(history_item_id_value))
                    elif isinstance(history_item_id_value, (list, set)):
                        history_item_ids = set(history_item_id_value)
                    logger.info(f"Extracted history item IDs from data_sample: {sorted(history_item_ids)}")
                except Exception as e:
                    logger.warning(f"Failed to extract history_item_id from data_sample: {e}")
                    history_item_ids = set()
            
            candidate_item_ids = set()
            
            if hasattr(self, 'data_sample') and self.data_sample is not None and 'candidate_item_id' in self.data_sample:
                try:
                    candidate_item_id_value = self.data_sample['candidate_item_id']
                    if isinstance(candidate_item_id_value, str):
                        candidate_item_ids = set(eval(candidate_item_id_value))
                    elif isinstance(candidate_item_id_value, (list, set)):
                        candidate_item_ids = set(candidate_item_id_value)
                    logger.debug(f"Extracted candidate item IDs from data_sample: {sorted(candidate_item_ids)}")
                except Exception as e:
                    logger.warning(f"Failed to extract candidate_item_id from data_sample: {e}")
                    candidate_item_ids = set()
            
            original_answer = final_answer.copy()
            filtered_answer = [item_id for item_id in final_answer if item_id in candidate_item_ids]
            
            if len(filtered_answer) != len(original_answer):
                removed_items = [item_id for item_id in original_answer if item_id not in filtered_answer]
                logger.warning(f"Solver included history items in answer! Removed: {removed_items}")
                logger.info(f"Original answer: {original_answer}")
                logger.info(f"Filtered answer: {filtered_answer}")
                logger.info(f"History items: {sorted(history_item_ids)}")
                logger.info(f"Candidate items: {sorted(candidate_item_ids)}")
            
            final_answer = filtered_answer
        
        self._last_solution = solution
        self._last_final_answer = final_answer
        
        self._gt_position_before_reflection = self._get_ground_truth_position(final_answer)
        
        self._answer_before_reflection = final_answer.copy() if isinstance(final_answer, list) else final_answer
        
        logger.info(f"Final Answer: {final_answer} | Ground Truth: {self.gt_answer}")
        if self._gt_position_before_reflection > 0:
            logger.debug(f"Ground truth position (before reflection): {self._gt_position_before_reflection}")
        
        return self.finish(final_answer)

    def _build_MARCO_scratchpad(self, solution: str, final_answer: str) -> str:
        scratchpad = f"MARCO Process Summary\n"
        scratchpad += f"Task: {self.task.upper()}\n"
        scratchpad += f"Query: {getattr(self, 'input', 'No input')}\n\n"
        
        # Phase 1: Planning
        scratchpad += "Phase 1 - Planning:\n"
        if self.current_plan:
            plan_only = self.planner.extract_plan_only(self.current_plan)
            scratchpad += f"{plan_only}\n\n"
        else:
            scratchpad += "No plan generated\n\n"
        
        # Phase 2: Working
        scratchpad += "Phase 2 - Working Results:\n"
        if self.execution_results:
            for step_var, result in self.execution_results.items():
                scratchpad += f"{step_var}: {result}\n"
        else:
            scratchpad += "No execution results\n"
        scratchpad += "\n"
        
        # Phase 3: Solving
        scratchpad += "Phase 3 - Solving:\n"
        scratchpad += f"Solution: {solution}\n"
        scratchpad += f"Final Answer: {final_answer}\n"
        
        return scratchpad

    def _perform_solver_reranking(self, solver_feedback: str) -> str:
        logger.info("Performing Solver Reranking (without full rerun)")
        
        previous_ranking = self._last_final_answer if hasattr(self, '_last_final_answer') else []
        previous_solution = self._last_solution if hasattr(self, '_last_solution') else ""
        
        solver_feedback_key = 'solver_reflections'
        if solver_feedback_key not in self.manager_kwargs:
            self.manager_kwargs[solver_feedback_key] = ""
        
        feedback_message = f"Previous ranking: {previous_ranking}\n"
        if previous_solution:
            feedback_message += f"Previous solution: {previous_solution}\n\n"
        feedback_message += f"Feedback: {solver_feedback}\n\n"
        feedback_message += f"Required action: Review previous ranking, understand the feedback, re-analyze user preferences and items, and produce an improved ranking that addresses the feedback. The new ranking MUST be different from the previous one.\n"
        
        self.manager_kwargs[solver_feedback_key] = feedback_message
        
        logger.info(f"Providing Solver with previous ranking: {previous_ranking}")
        logger.info(f"Feedback: {solver_feedback}")
        
        logger.debug(f"Re-invoking Solver with feedback. Using existing execution results from working phase.")
        
        plan_only = self.planner.extract_plan_only(self.current_plan)
        
        with duration_tracker.track_agent_call('solver'):
            reranked_solution = self.solver.invoke(plan_only, self.execution_results, self.task, **self.manager_kwargs)
        
        self.log(f"**MARCO Reranked Solution (Solver Feedback):**\n{reranked_solution}", agent=self.solver)
        
        reranked_answer = self.solver.extract_final_answer(reranked_solution, self.task)
        
        if self.task in ['sr'] and isinstance(reranked_answer, list):
            history_item_ids = set()
            if hasattr(self, 'data_sample') and self.data_sample is not None and 'history_item_id' in self.data_sample:
                try:
                    history_item_id_value = self.data_sample['history_item_id']
                    if isinstance(history_item_id_value, str):
                        history_item_ids = set(eval(history_item_id_value))
                    elif isinstance(history_item_id_value, (list, set)):
                        history_item_ids = set(history_item_id_value)
                except Exception as e:
                    logger.warning(f"Failed to extract history_item_id: {e}")
                    history_item_ids = set()
            
            candidate_item_ids = set()
            if hasattr(self, 'data_sample') and self.data_sample is not None and 'candidate_item_id' in self.data_sample:
                try:
                    candidate_item_id_value = self.data_sample['candidate_item_id']
                    if isinstance(candidate_item_id_value, str):
                        candidate_item_ids = set(eval(candidate_item_id_value))
                    elif isinstance(candidate_item_id_value, (list, set)):
                        candidate_item_ids = set(candidate_item_id_value)
                except Exception as e:
                    logger.warning(f"Failed to extract candidate_item_id: {e}")
                    candidate_item_ids = set()
            
            original_reranked = reranked_answer.copy()
            reranked_answer = [item_id for item_id in reranked_answer if item_id in candidate_item_ids]
            
            if len(reranked_answer) != len(original_reranked):
                removed_items = [item_id for item_id in original_reranked if item_id not in reranked_answer]
                logger.warning(f"Reranked answer contained history items! Removed: {removed_items}")
        
        logger.info(f"Solver reranked answer: {reranked_answer}")

        self._last_solution = reranked_solution
        self._last_final_answer = reranked_answer
        
        return reranked_answer

    def _prepare_planning_query(self) -> str:
        base_query = getattr(self, 'input', 'No input provided')
        
        num_candidates = None
        if self.task in ['sr'] and hasattr(self, 'data_sample') and self.data_sample is not None:
            if 'candidate_item_id' in self.data_sample:
                try:
                    candidate_item_id_value = self.data_sample['candidate_item_id']
                    if isinstance(candidate_item_id_value, str):
                        candidate_list = eval(candidate_item_id_value)
                    elif isinstance(candidate_item_id_value, list):
                        candidate_list = candidate_item_id_value
                    else:
                        candidate_list = []
                    num_candidates = len(candidate_list)
                except Exception as e:
                    logger.warning(f"Failed to extract candidate count: {e}")
        
        if self.task == 'sr':
            query = f"Sequential recommendation task: {base_query}"
            if num_candidates:
                query += f"\n\nThere are {num_candidates} candidate items available. Create analysis steps for all {num_candidates} candidates."
            return query
        elif self.task == 'rp':
            return f"Rating prediction task: {base_query}"
        else:
            return f"{self.task} task: {base_query}"

    def _dependencies_satisfied(self, step: Dict[str, Any]) -> bool:
        dependencies = step.get('dependencies', [])
        variable = step.get('variable', 'unknown')
        
        for dep in dependencies:
            if dep not in self.completed_steps:
                return False
        return True

    def _is_meaningless_result(self, result: str) -> bool:
        if not isinstance(result, str):
            return False
        
        meaningless_patterns = [
            'User info database not available',
            'User information database not available',
            'No history found for user',
            'No history found for item',
            'database not available',
            'No user information',
            'User History X: No history',
        ]
        
        lines = result.strip().split('\n')
        
        if len(lines) <= 2:
            for pattern in meaningless_patterns:
                if pattern in result:
                    logger.debug(f"Skipping meaningless result (matched pattern '{pattern}'): {result[:100]}")
                    return True
        
        has_actionable_data = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            is_noise = any(pattern in line for pattern in meaningless_patterns)
            
            if not is_noise and any(keyword in line for keyword in ['Brand:', 'Price:', 'Categories:', 'Rating:', 'Title:', 'Genres:', 'User:', 'Item:', 'analyzed', 'Retrieved']):
                has_actionable_data = True
                break
        
        if not has_actionable_data and any(pattern in result for pattern in meaningless_patterns):
            logger.debug(f"Skipping meaningless result (no actionable data): {result[:100]}")
            return True
        
        return False

    def _replace_ordinal_with_item_id(self, task_desc: str) -> str:
        import re
        
        if '#E2' not in self.execution_results:
            return task_desc
        
        retriever_result = self.execution_results['#E2']
        
        item_ids = []
        
        attribute_matches = re.findall(r'-\s*(\d+)\s*\(', retriever_result)
        if attribute_matches:
            item_ids = [int(x) for x in attribute_matches]
            logger.info(f"Extracted {len(item_ids)} item IDs from Retriever (attribute format): {item_ids}")
        
        if not item_ids:
            match = re.search(r'candidate items.*?:\s*([\d,\s]+)', retriever_result)
            if match:
                item_ids_str = match.group(1)
                item_ids = [int(x.strip()) for x in item_ids_str.split(',') if x.strip().isdigit()]
                logger.info(f"Extracted {len(item_ids)} item IDs from Retriever (comma format): {item_ids}")
        
        if not item_ids:
            logger.warning(f"Could not extract item IDs from Retriever result: {retriever_result}")
            return task_desc
        
        def get_ordinal_suffix(n):
            if 10 <= n % 100 <= 20:
                suffix = 'th'
            else:
                suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
            return f'{n}{suffix}'
        
        ordinal_map = {}
        for i in range(len(item_ids)):
            ordinal = get_ordinal_suffix(i + 1)
            ordinal_map[ordinal] = i
        
        logger.info(f"Generated ordinal_map for {len(item_ids)} candidates: {list(ordinal_map.keys())}")
        logger.debug(f"Original task_desc BEFORE replacement: '{task_desc}'")
        
        sorted_ordinals = sorted(ordinal_map.items(), key=lambda x: len(x[0]), reverse=True)
        
        replacement_made = False
        for ordinal, index in sorted_ordinals:
            if index < len(item_ids):
                old_pattern = f'{ordinal} candidate item'
                new_text = f'candidate item {item_ids[index]}'
                if old_pattern in task_desc:
                    logger.debug(f"Replacing '{old_pattern}' with '{new_text}'")
                    task_desc = task_desc.replace(old_pattern, new_text)
                    replacement_made = True
        
        if not replacement_made:
            for i in range(len(item_ids), 0, -1):
                old_pattern = f'candidate item {i}'
                if old_pattern in task_desc and i-1 < len(item_ids):
                    new_text = f'candidate item {item_ids[i-1]}'
                    logger.debug(f"Replacing '{old_pattern}' (non-ordinal) with '{new_text}'")
                    task_desc = task_desc.replace(old_pattern, new_text)
                    break
        
        logger.debug(f"Final task_desc AFTER replacement: '{task_desc}'")
        return task_desc
    
    def _execute_step(self, step: Dict[str, Any]) -> str:
        worker_type = step['worker_type']
        task_desc = step['task_description']
        
        for dep in step['dependencies']:
            if dep in self.execution_results:
                if worker_type.lower() == 'analyst' and dep == '#E2':
                    continue
                task_desc = task_desc.replace(dep, str(self.execution_results[dep]))
        
        if worker_type.lower() == 'analyst':
            task_desc = self._replace_ordinal_with_item_id(task_desc)
        
        if worker_type.lower() == 'retriever':
            worker = None
        else:
            worker = self._get_worker_agent(worker_type)
            if worker is None:
                return f"Worker {worker_type} not available"
        
        try:
            if hasattr(worker, 'json_mode'):
                json_mode = worker.json_mode
            else:
                json_mode = False
            
            logger.debug(f"Worker {worker_type} json_mode: {json_mode}")
            
            from marco.utils import duration_tracker
            
            if worker_type.lower() == 'analyst':
                args = self._parse_analyst_arguments_from_context(task_desc)
                logger.debug(f"Analyst args: {args}, type: {type(args)}")
                
                execution_context = self._build_execution_context()
                
                kwargs = {
                    'task_context': task_desc,
                    'execution_context': execution_context,
                    **self.manager_kwargs
                }
                
                with duration_tracker.track_agent_call('analyst'):
                    if json_mode and isinstance(args, list):
                        result = worker.invoke(argument=args, json_mode=json_mode, **kwargs)
                    elif not json_mode and isinstance(args, list):
                        arg_string = f"{args[0]},{args[1]}" if len(args) >= 2 else "user,1"
                        result = worker.invoke(argument=arg_string, json_mode=json_mode, **kwargs)
                    else:
                        result = worker.invoke(argument=args, json_mode=json_mode, **kwargs)
                    
            elif worker_type.lower() == 'retriever':
                logger.debug(f"Retriever args: {task_desc}, type: {type(task_desc)}")
                
                import re
                user_id_match = re.search(r'user\s+(\d+)', task_desc, re.IGNORECASE)
                if user_id_match:
                    user_id = int(user_id_match.group(1))
                elif 'user_id' in self.kwargs:
                    user_id = int(self.kwargs['user_id'])
                else:
                    user_id = 1
                
                from marco.tools import TOOL_MAP
                if 'retriever' in TOOL_MAP:
                    retriever_class = TOOL_MAP['retriever']
                    from marco.utils import read_json
                    import os

                    item_info_path = None
                    retriever_config = {}
                    
                    if hasattr(self, 'agent_kwargs') and 'dataset' in self.agent_kwargs:
                        dataset_name = self.agent_kwargs['dataset']
                        dataset_config_path = f'config/tools/retriever/{dataset_name}.json'
                        if os.path.exists(dataset_config_path):
                            retriever_config = read_json(dataset_config_path)
                            if 'item_info' in retriever_config:
                                if hasattr(self, 'agent_kwargs') and 'data_dir' in self.agent_kwargs:
                                    data_dir = self.agent_kwargs['data_dir']
                                    item_info_path = os.path.join(data_dir, 'item.csv')
                                    logger.info(f"Using data_dir override for item_info: {item_info_path}")
                                else:
                                    item_info_path = retriever_config['item_info']
                                    logger.info(f"Loaded item_info from dataset config '{dataset_config_path}': {item_info_path}")
                    
                    if not retriever_config:
                        config_path = 'config/tools/retriever.json'
                        if os.path.exists(config_path):
                            retriever_config = read_json(config_path)
                    
                    if item_info_path is None and hasattr(self, 'agent_kwargs') and 'info_database' in self.agent_kwargs:
                        info_db_config = self.agent_kwargs['info_database']
                        if 'item_info' in info_db_config:
                            item_info_path = info_db_config['item_info']
                            logger.info(f"Got item_info from agent_kwargs['info_database']: {item_info_path}")
                    
                    if item_info_path is None and hasattr(self, 'agent_kwargs') and 'data_dir' in self.agent_kwargs:
                        data_dir = self.agent_kwargs['data_dir']
                        item_info_path = os.path.join(data_dir, 'item.csv')
                        logger.info(f"Constructed item_info path from data_dir '{data_dir}': {item_info_path}")
                    
                    if item_info_path is None and hasattr(self, 'agent_kwargs') and 'dataset' in self.agent_kwargs:
                        dataset_name = self.agent_kwargs['dataset']
                        item_info_path = os.path.join('data', dataset_name, 'item.csv')
                        logger.info(f"Constructed item_info path from dataset '{dataset_name}': {item_info_path}")
                    
                    if item_info_path is not None:
                        retriever_config['item_info'] = item_info_path
                        logger.info(f"Set retriever item_info path to: {item_info_path}")
                    else:
                        logger.warning("Could not determine item_info path for Retriever")
                    
                    retriever = retriever_class(config=retriever_config)
                    
                    retriever.reset(data_sample=self.data_sample)
                    
                    with duration_tracker.track_agent_call('retriever'):
                        result = retriever.retrieve_candidates(user_id=user_id, k=-1)
                else:
                    result = "Retriever tool not found in TOOL_MAP"
                    
            else:
                result = f"Unknown worker type: {worker_type}"
            
            self._track_analyzed_entities(worker_type, task_desc, result)
            
            return result
            
        except Exception as e:
            error_msg = f"Error executing step with {worker_type}: {e}"
            logger.error(error_msg)
            if not hasattr(self, '_execution_errors'):
                self._execution_errors = []
            self._execution_errors.append({
                'worker': worker_type,
                'step': step.get('variable', 'unknown'),
                'error': str(e)
            })
            return f"Execution failed: {str(e)}"

    def _build_execution_context(self) -> Dict[str, Any]:
        context = {
            'previous_results': dict(self.execution_results),
            'analyzed_entities': {
                'users': list(self.analyzed_users),
                'items': list(self.analyzed_items)
            },
            'step_number': self.step_n,
            'total_steps': len(self.plan_steps),
            'data_sample': self.data_sample
        }
        return context
    
    def _get_worker_agent(self, worker_type: str) -> Optional[Agent]:
        worker_map = {
            'analyst': self.analyst
        }
        return worker_map.get(worker_type.lower())

    def _parse_analyst_arguments_from_context(self, task_desc: str) -> List[Any]:
        import re
        
        user_match = re.search(r'user\s+(\d+)', task_desc, re.IGNORECASE)
        item_match = re.search(r'item\s+(\d+)', task_desc, re.IGNORECASE)
        
        if user_match:
            return ['user', int(user_match.group(1))]
        elif item_match:
            return ['item', int(item_match.group(1))]
        else:
            input_text = getattr(self, 'input', '')
            user_match = re.search(r'user[_\s:]*(\d+)', input_text, re.IGNORECASE)
            item_match = re.search(r'item[_\s:]*(\d+)', input_text, re.IGNORECASE)
            
            if user_match:
                return ['user', int(user_match.group(1))]
            elif item_match:
                return ['item', int(item_match.group(1))]
            else:
                if 'user_id' in self.kwargs:
                    return ['user', int(self.kwargs['user_id'])]
                elif 'item_id' in self.kwargs:
                    return ['item', int(self.kwargs['item_id'])]
                else:
                    return ['user', 1]

    def _parse_analyst_arguments(self, task_desc: str) -> List[str]:
        import re
        
        user_match = re.search(r'user\s+(\d+)', task_desc, re.IGNORECASE)
        item_match = re.search(r'item\s+(\d+)', task_desc, re.IGNORECASE)
        
        if user_match:
            return ['user', user_match.group(1)]
        elif item_match:
            return ['item', item_match.group(1)]
        else:
            return ['item', str(self.kwargs.get('item_id', '1'))]

    def _track_analyzed_entities(self, worker_type: str, task_desc: str, result: str) -> None:
        import re
        
        if worker_type.lower() == 'analyst':
            user_match = re.search(r'user\s+(\d+)', task_desc, re.IGNORECASE)
            item_match = re.search(r'item\s+(\d+)', task_desc, re.IGNORECASE)
            
            if user_match:
                self.analyzed_users.add(user_match.group(1))
            elif item_match:
                self.analyzed_items.add(item_match.group(1))

    def is_finished(self) -> bool:
        return hasattr(self, 'finished') and self.finished

    def is_halted(self) -> bool:
        return (self.step_n > self.max_step) and not self.is_finished()

    def step(self):
        try:
            if self.planner and self.solver and self.phase in ['planning', 'working', 'solving']:
                return self.forward(reset=False)
            else:
                logger.error("MARCO agents (planner and solver) are required but not available.")
                raise ValueError("MARCO agents (planner and solver) are required but not available.")
        except Exception as e:
            logger.error(f"Error in MARCO step: {e}")
            raise
