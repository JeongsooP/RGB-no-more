from torch.optim.optimizer import Optimizer

class WeightDecay(Optimizer):
    """Additive weight decay: AdamW-like but only the weight decay part

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        weight_decay: weight decay (L2 penalty) (default: 0)
        lr: current learning rate used to scale weight_decay appropriately
            -> does not multiply lr with weight_decay, but uses lr to extract relative schedule

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.WeightDecay(model.parameters())
        >>> optimizer.step() # decays weight only
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        if weight_decay < 0.0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )

        defaults = dict(
            lr=lr, # lr (affected by scheduling/lr copying)
            base_lr=lr, # base_lr (lr when initialized)
            weight_decay=weight_decay,
        )
        super(WeightDecay, self).__init__(params, defaults)

    def step(self):
        r"""Performs a single optimization step.
        """
        for group in self.param_groups:
            for p in group['params']:
                p.data.add_(p.data, alpha = -((group['lr'] / group['base_lr']) * group['weight_decay'])) # weight decay with schedule
        return