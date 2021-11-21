import torch

from tutils import tfilename




class SimpleMetricLogger:
    def __init__(self, logger, config, delimiter="\t", *args, **kwargs):
        self.meters = dict()
        self.delimiter = delimiter

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        elif attr in self.__dict__:
            return self.__dict__[attr]
        raise ValueError

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                f"{name}: {str(meter)}"
            )
        return self.delimiter.join(loss_str)

