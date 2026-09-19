import numpy as np
import pandas as pd
from argparse import ArgumentParser
from loguru import logger

from marco.tasks.evaluate import EvaluateTask
from marco.utils import init_all_seeds

class TestTask(EvaluateTask):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser = EvaluateTask.parse_task_args(parser)
        parser.add_argument('--random', action='store_true', help='Whether to randomly sample test data')
        parser.add_argument('--samples', type=int, default=5, help='Number of samples to test')
        parser.add_argument('--offset', type=int, default=0, help='Offset of samples. With --last: reduces sample count by offset. Without --last: skips first offset samples')
        parser.add_argument('--offsetGT', type=int, default=0, help='Offset GT-filtered samples (samples with GT in candidates). Works within the first/last N samples specified by --samples')
        parser.add_argument('--last', action='store_true', help='Sample from the end of the dataset instead of the beginning')
        return parser
    
    def prompt_data(self, df: pd.DataFrame) -> list[tuple[str, int | float | str, pd.Series]]:
        prompts = super().prompt_data(df)
        
        if hasattr(self, 'offsetGT') and self.offsetGT > 0:
            original_count = len(prompts)
            
            if self.last:
                actual_samples = max(1, original_count - self.offsetGT)
                prompts = prompts[-actual_samples:]
                logger.info(f"Applied --offsetGT {self.offsetGT} with --last: keeping last {actual_samples}/{original_count} GT samples")
            else:
                prompts = prompts[self.offsetGT:]
                logger.info(f"Applied --offsetGT {self.offsetGT}: skipped first {self.offsetGT} GT samples, {len(prompts)}/{original_count} remaining")
        
        return prompts

    def get_data(self, data_file: str, max_his: int) -> pd.DataFrame:
        df = super().get_data(data_file, max_his)
        
        if self.random:
            sample_idx = np.random.choice(len(df), min(self.samples, len(df)), replace=False)
            df = df.iloc[sample_idx].reset_index(drop=True)
        else:
            if self.last:
                actual_samples = max(1, self.samples - self.offset)
                start_idx = max(0, len(df) - actual_samples)
                df = df.iloc[start_idx: start_idx + actual_samples].reset_index(drop=True)
            else:
                df = df.iloc[self.offset: self.offset + self.samples].reset_index(drop=True)
        
        return df

    def run(self, random: bool, samples: int, offset: int, offsetGT: int = 0, last: bool = False, *args, **kwargs):
        self.sampled = True
        self.random = random
        if self.random:
            init_all_seeds(2026)
        self.samples = samples
        self.offset = offset
        self.offsetGT = offsetGT
        self.last = last
        super().run(*args, **kwargs)

if __name__ == '__main__':
    TestTask().launch()
