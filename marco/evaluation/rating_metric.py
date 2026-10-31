from math import sqrt
from marco.evaluation.metric_shim import Accuracy as ShimAccuracy, MeanSquaredError as ShimMSE, MeanAbsoluteError as ShimMAE

class Accuracy(ShimAccuracy):
    pass

class MSE(ShimMSE):
    def update(self, output: dict) -> None:
        super().update(output=output)

    def compute(self):
        res = super().compute()
        return {'mse': res.get('mse', 0.0)}

class RMSE(ShimMSE):
    def update(self, output: dict) -> None:
        super().update(output=output)

    def compute(self):
        res = super().compute()
        mse = res.get('mse', 0.0)
        return {'rmse': sqrt(mse)}

class MAE(ShimMAE):
    def update(self, output: dict) -> None:
        super().update(output=output)

    def compute(self):
        res = super().compute()
        return {'mae': res.get('mae', 0.0)}
