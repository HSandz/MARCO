from abc import abstractmethod
from loguru import logger
import math

class RankMetric:
    def __init__(self, topks: list[int] | int, *args, **kwargs):
        if isinstance(topks, int):
            topks = [topks]
        self.topks = topks
        for topk in self.topks:
            setattr(self, f'at{topk}', 0.0)
        self.total = 0

    def update(self, output: dict) -> None:
        answer = output['answer']
        label = output['label']

        if isinstance(answer, str):
            logger.warning(f"Received string answer instead of list: {answer}")
            metrics = {topk: 0 for topk in self.topks}
        else:
            metrics = self.metric_at_k(answer, label)

        for topk in self.topks:
            metric = metrics[topk]
            current = getattr(self, f'at{topk}')
            setattr(self, f'at{topk}', current + float(metric))

        self.total += 1

    def compute(self):
        result = {}
        for topk in self.topks:
            if self.total != 0:
                result[topk] = getattr(self, f'at{topk}') / float(self.total)
            else:
                result[topk] = 0
        return result

    @abstractmethod
    def metric_at_k(self, answer: list[int], label: int) -> dict:
        raise NotImplementedError

class HitRatioAt(RankMetric):
    def metric_at_k(self, answer: list[int], label: int) -> dict:
        result = {}
        label = int(label)
        for topk in self.topks:
            if label in answer[:topk]:
                result[topk] = 1
            else:
                result[topk] = 0
        return result

    def compute(self):
        result = super().compute()
        return {f'HR@{topk}': result[topk] for topk in self.topks}

class NDCGAt(RankMetric):
    def metric_at_k(self, answer: list[int], label: int) -> dict:
        result = {}
        label = int(label)
        for topk in self.topks:
            try:
                label_pos = answer.index(label) + 1
            except ValueError:
                label_pos = topk + 1
            if label_pos <= topk:
                result[topk] = 1.0 / math.log2(label_pos + 1.0)
            else:
                result[topk] = 0
        return result

    def compute(self):
        result = super().compute()
        return {f'NDCG@{topk}': result[topk] for topk in self.topks}

class MRRAt(RankMetric):
    def metric_at_k(self, answer: list[int], label: int) -> dict:
        result = {}
        label = int(label)
        for topk in self.topks:
            try:
                label_pos = answer.index(label) + 1
            except ValueError:
                label_pos = topk + 1
            if label_pos <= topk:
                result[topk] = 1.0 / float(label_pos)
            else:
                result[topk] = 0
        return result

    def compute(self):
        result = super().compute()
        return {f'MRR@{topk}': result[topk] for topk in self.topks}
