import os
from tqdm import tqdm
from typing import Any
from loguru import logger
from argparse import ArgumentParser

from marco.tasks.generation import GenerationTask
from marco.utils import str2list, token_tracker
from marco.evaluation import MetricDict, HitRatioAt, NDCGAt, RMSE, Accuracy, MAE

class EvaluateTask(GenerationTask):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser = GenerationTask.parse_task_args(parser)
        parser.add_argument('--steps', type=int, default=1, help='Number of steps')
        parser.add_argument('--topks', type=str2list, default=[1, 3, 5], help='Top-Ks for ranking task')
        return parser

    def get_metrics(self, topks: list[int] = [1, 3, 5]):
        if self.task == 'rp':
            self.metrics = MetricDict({
                'true_rmse': RMSE(),
                'true_mae': MAE(),
                'true_accuracy': Accuracy(),
                'valid_rmse': RMSE(),
                'valid_mae': MAE(),
            })
        elif self.task == 'sr':
            self.metrics = MetricDict({
                'true_hit_rate': HitRatioAt(topks=topks),
                'true_ndcg': NDCGAt(topks=topks),
                'valid_hit_rate': HitRatioAt(topks=topks),
                'valid_ndcg': NDCGAt(topks=topks),
            })
        else:
            raise NotImplementedError

    def update_evaluation(self, answer: float | int | str, gt_answer: float | int | str) -> str:
        valid = self.system.finished
        self.total_count += 1
        if valid:
            self.valid_count += 1
        logger.debug(f'Answer: {answer}, Ground Truth: {gt_answer}')

        output = {'answer': answer, 'label': gt_answer}

        if valid:
            self.metrics.update(output=output)
        else:
            self.metrics.update(output=output, prefix='true')

        return self.metrics.get_display_string(prefix='true')

    def _log_cumulative_scores(self, sample_id: str) -> None:
        try:
            if hasattr(self, 'log_handler_id') and self.log_handler_id is not None:
                log_file_path = None
                for handler_id, handler in logger._core.handlers.items():
                    if handler_id == self.log_handler_id:
                        if hasattr(handler._sink, 'name'):
                            log_file_path = handler._sink.name
                        elif hasattr(handler._sink, '_file') and hasattr(handler._sink._file, 'name'):
                            log_file_path = handler._sink._file.name
                        break
                
                if log_file_path:
                    result = self.metrics.compute()
                    
                    current_task_stats = token_tracker.get_task_stats()
                    total_input_tokens = current_task_stats.get('total_input_tokens', 0)
                    total_output_tokens = current_task_stats.get('total_output_tokens', 0)
                    total_tokens = current_task_stats.get('total_tokens', 0)
                    
                    with open(log_file_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"\n===== Sample {sample_id} - Cumulative Scores ({self.total_count} samples) =====\n")
                        log_file.write(f"Tokens: Input={total_input_tokens} | Output={total_output_tokens} | Total={total_tokens}\n\n")
                        
                        for metric_name, metric_values in result.items():
                            if len(metric_values) == 1:
                                value = next(iter(metric_values.values()))
                                log_file.write(f"{metric_name}: {value:.4f}\n")
                            else:
                                log_file.write(f"{metric_name}:\n")
                                for key, value in metric_values.items():
                                    log_file.write(f"  {key}: {value:.4f}\n")
                        log_file.write("\n")
        except Exception as e:
            logger.warning(f"Failed to log cumulative scores: {e}")

    @property
    def running_steps(self):
        return self.steps

    def before_generate(self) -> None:
        if hasattr(self, 'n_candidate') and self.n_candidate is not None and self.n_candidate >= 15 and self.topks == [1, 3, 5]:
            self.topks = [1, 3, 5, 10]

        self.get_metrics(self.topks)
        self.valid_count = 0
        self.total_count = 0
        self.failed_samples = []
        self.skipped_no_gt_samples = []
        self.gt_positions = []
        
    def after_step(self, answer: Any, gt_answer: int | float | str, step: int, record: dict) -> None:
        record[f'Answer_{step}'] = answer
        if hasattr(self.system, 'reflected') and self.system.reflected and self.system.reflector.keep_reflections:
            logger.trace(f"Reflection input: {self.system.reflector.reflection_input}")
            logger.trace(f"Reflection output: {self.system.reflector.reflection_output}")

    def after_iteration(self, answer: Any, gt_answer: int | float | str, record: dict, pbar: tqdm) -> None:
        record['Answer_GT'] = gt_answer
        record['System_Finished'] = self.system.finished

        sample_id = record.get('sample_id', 'unknown')
        user_id = record.get('user_id', 'unknown')
        skipped_no_gt = record.get('_skipped_no_gt', False)

        if skipped_no_gt:
            logger.info(f"Sample {sample_id} (User {user_id}): SKIPPED - GT not in candidates (auto-fail)")
            self.skipped_no_gt_samples.append({
                'sample_id': sample_id,
                'user_id': user_id
            })
        else:
            logger.info(f"Sample {sample_id} (User {user_id}): system.finished={self.system.finished}, answer_type={type(answer)}, answer={answer}")

        if self.task == 'sr' and isinstance(answer, list):
            try:
                if gt_answer in answer:
                    gt_position = answer.index(gt_answer) + 1
                else:
                    gt_position = -1

                self.gt_positions.append({
                    'sample_id': sample_id,
                    'user_id': user_id,
                    'gt_item': gt_answer,
                    'position': gt_position,
                    'list_length': len(answer) if answer else 0,
                    'skipped_no_gt': skipped_no_gt
                })
            except Exception as e:
                logger.warning(f"Failed to track GT position for sample {sample_id}: {e}")

        if not self.system.finished and not skipped_no_gt:
            sample_info = {
                'sample_id': record.get('sample_id', 'unknown'),
                'user_id': record.get('user_id', 'unknown')
            }
            self.failed_samples.append(sample_info)

        if skipped_no_gt:
            logger.info(f"Sample AUTO-FAIL: {sample_id} (User {user_id}) - GT not in candidates")
        elif self.system.finished:
            logger.info(f"Sample SUCCESS: {sample_id} (User {user_id})")
        else:
            logger.info(f"Sample FAILED: {sample_id} (User {user_id})")

        pbar.set_description(self.update_evaluation(answer, gt_answer))

        self._log_cumulative_scores(sample_id)

    def after_generate(self) -> None:
        logger.success("===================================Evaluation Report===================================")
        valid_percentage = (self.valid_count / self.total_count * 100) if self.total_count > 0 else 0
        logger.success(f"Valid Answers: {self.valid_count}/{self.total_count} samples ({valid_percentage:.1f}%)")

        if self.skipped_no_gt_samples:
            logger.warning(f"Skipped Samples (GT not in candidates): {len(self.skipped_no_gt_samples)}/{self.total_count}")
            for skipped_sample in self.skipped_no_gt_samples[:10]:
                logger.warning(f"  - Sample {skipped_sample['sample_id']} (User {skipped_sample['user_id']})")
            if len(self.skipped_no_gt_samples) > 10:
                logger.warning(f"  ... and {len(self.skipped_no_gt_samples) - 10} more")

        if self.failed_samples:
            logger.warning(f"Failed Samples (system failures): {len(self.failed_samples)}/{self.total_count}")
            for failed_sample in self.failed_samples[:10]:
                logger.warning(f"  - Sample {failed_sample['sample_id']} (User {failed_sample['user_id']})")
            if len(self.failed_samples) > 10:
                logger.warning(f"  ... and {len(self.failed_samples) - 10} more")

        if not self.failed_samples and not self.skipped_no_gt_samples:
            logger.success("All samples completed successfully!")

        self.metrics.report()
        
        if self.task == 'sr' and self.gt_positions:
            if hasattr(self, 'log_handler_id') and self.log_handler_id is not None:
                import sys
                from loguru._handler import Handler
                
                log_file_path = None
                for handler_id, handler in logger._core.handlers.items():
                    if handler_id == self.log_handler_id:
                        if hasattr(handler._sink, 'name'):
                            log_file_path = handler._sink.name
                        elif hasattr(handler._sink, '_file') and hasattr(handler._sink._file, 'name'):
                            log_file_path = handler._sink._file.name
                        break
                
                if log_file_path:
                    is_debug_enabled = any(
                        handler._levelno <= 10
                        for handler in logger._core.handlers.values()
                    )
                    
                    with open(log_file_path, 'a', encoding='utf-8') as log_file:
                        sorted_positions = sorted(self.gt_positions, key=lambda x: (x['position'] == -1, x['position']))

                        found_count = sum(1 for p in self.gt_positions if p['position'] > 0)
                        not_found_count = sum(1 for p in self.gt_positions if p['position'] == -1 and not p.get('skipped_no_gt', False))
                        skipped_no_gt_count = sum(1 for p in self.gt_positions if p.get('skipped_no_gt', False))

                        log_file.write("\n===================================Ground Truth Position Summary===================================\n")

                        if found_count > 0:
                            positions_list = [p['position'] for p in self.gt_positions if p['position'] > 0]
                            avg_position = sum(positions_list) / len(positions_list)
                            log_file.write(f"Ground Truth Found: {found_count}/{len(self.gt_positions)} samples ({found_count/len(self.gt_positions)*100:.1f}%)\n")
                            log_file.write(f"Average Position (when found): {avg_position:.2f}\n")

                        if not_found_count > 0:
                            log_file.write(f"Ground Truth Not Found in Output: {not_found_count}/{len(self.gt_positions)} samples\n")

                        if skipped_no_gt_count > 0:
                            log_file.write(f"Ground Truth Not in Candidates (Auto-Fail): {skipped_no_gt_count}/{len(self.gt_positions)} samples\n")

                        if is_debug_enabled:
                            log_file.write("\nDetailed Ground Truth Positions (sorted by position):\n")
                            log_file.write(f"{'Sample':<8} {'User':<8} {'GT Item':<10} {'Position':<10} {'List Length':<12} {'Status':<15}\n")
                            log_file.write("-" * 75 + "\n")

                            for pos_info in sorted_positions:
                                if pos_info.get('skipped_no_gt', False):
                                    position_str = "N/A"
                                    status_str = "GT Not in Cand"
                                elif pos_info['position'] > 0:
                                    position_str = str(pos_info['position'])
                                    status_str = "Found"
                                else:
                                    position_str = "Not Found"
                                    status_str = "Not in Output"

                                log_file.write(
                                    f"{pos_info['sample_id']:<8} "
                                    f"{pos_info['user_id']:<8} "
                                    f"{pos_info['gt_item']:<10} "
                                    f"{position_str:<10} "
                                    f"{pos_info['list_length']:<12} "
                                    f"{status_str:<15}\n"
                                )
        
        if hasattr(self.system, 'reflection_all_reruns') and self.system.reflection_all_reruns:
            if hasattr(self, 'log_handler_id') and self.log_handler_id is not None:
                log_file_path = None
                for handler_id, handler in logger._core.handlers.items():
                    if handler_id == self.log_handler_id:
                        if hasattr(handler._sink, 'name'):
                            log_file_path = handler._sink.name
                        elif hasattr(handler._sink, '_file') and hasattr(handler._sink._file, 'name'):
                            log_file_path = handler._sink._file.name
                        break
                
                if log_file_path:
                    is_debug_enabled = any(
                        handler._levelno <= 10
                        for handler in logger._core.handlers.values()
                    )
                    
                    with open(log_file_path, 'a', encoding='utf-8') as log_file:
                        all_reruns = self.system.reflection_all_reruns
                        improvements = self.system.reflection_improvements if hasattr(self.system, 'reflection_improvements') else []
                        total_reflections_triggered = len(all_reruns)
                        
                        log_file.write("\n===================================Reflection Summary===================================\n")
                        log_file.write(f"Total reflection reruns triggered: {total_reflections_triggered}\n")
                        log_file.write(f"Reflection reruns with improvements: {len(improvements)}\n")
                        if total_reflections_triggered > 0:
                            log_file.write(f"Improvement rate: {len(improvements)}/{total_reflections_triggered} ({100*len(improvements)/total_reflections_triggered:.1f}%)\n")
                        
                        log_file.write("\n===================================Reflection Reruns Summary===================================\n")
                        log_file.write(f"Samples rerun by reflection: {len(all_reruns)}/{len(self.gt_positions)}\n")
                        
                        if all_reruns:
                            planner_only = [r for r in all_reruns if r.get('feedback_type') == 'planner']
                            solver_only = [r for r in all_reruns if r.get('feedback_type') == 'solver']
                            both_agents = [r for r in all_reruns if r.get('feedback_type') == 'both']
                            
                            def count_improvements(reruns):
                                return sum(1 for r in reruns if r['position_after'] < r['position_before'] and r['position_before'] > 0)
                            
                            log_file.write("\nBREAKDOWN BY FEEDBACK TYPE:\n")
                            log_file.write(f"  Planner only (full rerun):        {len(planner_only)} samples | {count_improvements(planner_only)} improved\n")
                            log_file.write(f"  Solver only (reranking):          {len(solver_only)} samples | {count_improvements(solver_only)} improved\n")
                            log_file.write(f"  Both agents (full rerun):         {len(both_agents)} samples | {count_improvements(both_agents)} improved\n")
                            
                            if is_debug_enabled:
                                log_file.write("\nDETAILED BREAKDOWN:\n")
                                log_file.write("All Reflection Reruns (sorted by improvement, positive = improved, negative = worsened):\n")
                                log_file.write(f"{'Sample':<8} {'User':<8} {'GT Item':<10} {'Before':<8} {'After':<8} {'Delta':<8} {'Feedback Type':<18}\n")
                                log_file.write("-" * 80 + "\n")
                                
                                sorted_reruns = sorted(all_reruns, key=lambda x: x['position_before'] - x['position_after'], reverse=True)
                                
                                for rerun_info in sorted_reruns:
                                    improvement = rerun_info['position_before'] - rerun_info['position_after']
                                    feedback_type = rerun_info.get('feedback_type', 'unknown')
                                    
                                    if improvement > 0:
                                        improvement_str = f"+{improvement}"
                                    elif improvement < 0:
                                        improvement_str = f"{improvement}"
                                    else:
                                        improvement_str = "0"
                                    
                                    if feedback_type == 'planner':
                                        feedback_str = "Planner (Full)"
                                    elif feedback_type == 'solver':
                                        feedback_str = "Solver (Rerank)"
                                    elif feedback_type == 'both':
                                        feedback_str = "Both (Full)"
                                    else:
                                        feedback_str = "Unknown"
                                    
                                    sample_num = rerun_info['sample_idx']
                                    
                                    log_file.write(
                                        f"{sample_num:<8} "
                                        f"{rerun_info['user_id']:<8} "
                                        f"{rerun_info['gt_item']:<10} "
                                        f"{rerun_info['position_before']:<8} "
                                        f"{rerun_info['position_after']:<8} "
                                        f"{improvement_str:<8} "
                                        f"{feedback_str:<18}\n"
                                    )

    def run(self, steps: int, topks: list[int], *args, **kwargs):
        assert kwargs['task'] in ['rp', 'sr'], "Only support rating (rp) and ranking (sr) tasks."
        self.steps = steps
        self.topks = topks
        super().run(*args, **kwargs)

if __name__ == '__main__':
    EvaluateTask().launch()
