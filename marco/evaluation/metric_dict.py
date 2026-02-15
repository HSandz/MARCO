from loguru import logger
from marco.evaluation.metric_shim import Metric

class MetricDict:

    def __init__(self, metrics: dict[str, Metric] = {}):
        self.metrics: dict[str, Metric] = metrics

    def add(self, name: str, metric: Metric):
        self.metrics[name] = metric

    def update(self, output: dict, prefix: str = '') -> str:
        for metric_name, metric in self.metrics.items():
            if not metric_name.startswith(prefix):
                continue
            metric.update(output)
            computed = metric.compute()
            if len(computed) == 1:
                computed_val = next(iter(computed.values()))
                logger.debug(f'{metric_name}: {computed_val:.4f}')
            else:
                logger.debug(f'{metric_name}:')
                for key, value in computed.items():
                    logger.debug(f'{key}: {value:.4f}')

    def get_display_string(self, prefix: str = '') -> str:
        display_metrics = []

        sorted_metrics = sorted(
            [(name, metric) for name, metric in self.metrics.items() if name.startswith(prefix)],
            key=lambda x: x[0]
        )

        for metric_name, metric in sorted_metrics:
            computed = metric.compute()

            if len(computed) > 1:
                sorted_computed = sorted(computed.items(), key=lambda x: int(x[0].split('@')[1]))

                for key, value in sorted_computed:
                    if key == 'NDCG@1':
                        continue
                    topk = int(key.split('@')[1])
                    if topk in [1, 3, 5]:
                        display_metrics.append(f'{key}:{value:.4f}')
            else:
                computed_val = next(iter(computed.values()))
                display_metrics.append(f'{metric_name}:{computed_val:.4f}')

        return ' '.join(display_metrics)

    def compute(self):
        result = {}
        for metric_name, metric in self.metrics.items():
            result[metric_name] = metric.compute()
        return result

    def report(self):
        result = self.compute()
        for metric_name, metric in result.items():
            if len(metric) == 1:
                metric_val = next(iter(metric.values()))
                logger.success(f'{metric_name}: {metric_val:.4f}')
            else:
                logger.success(f'{metric_name}:')
                for key, value in metric.items():
                    logger.success(f'{key}: {value:.4f}')
