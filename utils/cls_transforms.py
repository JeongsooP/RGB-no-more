"""
Copied from torchvision
vision/references/classification/transforms.py
link: https://github.com/pytorch/vision/blob/main/references/classification/transforms.py
"""

from typing import Tuple
import torch
from torch import Tensor

class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5. (Not used -- always mixed up)
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.

    Jul_4_2022_jespark comment: Extra modifications onto Dirichlet sampling to match the implementation on big_vision
    (link: https://github.com/google-research/big_vision)
    """

    def __init__(self, num_classes: int, 
                #p: float = 0.5, 
                alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(
                f"Please provide a valid positive value for the num_classes. Got num_classes={num_classes}"
            )

        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        #self.p = p # not used in big_vision (always mix up)
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        #if torch.rand(1).item() >= self.p: # not used in big_vision (all samples are mixed-up)
        #    return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        # sort lambda in descending order to avoid destroying current examples when p is very small (big_vision implementation)
        lambda_param, _ = torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha])).sort(descending=True)
        
        batch_rolled.mul_(lambda_param[1])
        batch.mul_(lambda_param[0]).add_(batch_rolled)

        target_rolled.mul_(lambda_param[1])
        target.mul_(lambda_param[0]).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s

class RandomMixup_DCT(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets. Specifically for DCT coefficients from JPEG image
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5. (Not used -- always mixed up)
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.

    Jeongsoo Changes: 
    - Extra modifications onto Dirichlet sampling to match the implementation on big_vision
      (link: https://github.com/google-research/big_vision)
    - Modified to work on DCT coefficients. It should work for normal Tensors
    """

    def __init__(self, num_classes: int, 
                alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(
                f"Please provide a valid positive value for the num_classes. Got num_classes={num_classes}"
            )

        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        #self.p = p # not used in big_vision (will always mix up)
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor or tuple): DCT coefficient of size (B, C, H, W, KH, KW) -- data
            target (Tensor): Integer tensor of size (B, ) -- labels

        Returns:
            Tensor: Randomly transformed batch.
        """
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if type(batch) == tuple: # DCT inputs can be (Y, cbcr)
            batch = list(batch)
        else:
            batch = [batch]

        if not self.inplace:
            for i in range(len(batch)):
                batch[i] = batch[i].clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch[0].dtype)

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = []
        for i in range(len(batch)):
            batch_rolled.append(batch[i].roll(1,0))
        target_rolled = target.roll(1, 0)

        lambda_param, _ = torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha])).sort(descending=True)
        
        for i in range(len(batch)): # len(batch) = 1 or 2
            batch_rolled[i].mul_(lambda_param[1])
            batch[i].mul_(lambda_param[0]).add_(batch_rolled[i])

        target_rolled.mul_(lambda_param[1])
        target.mul_(lambda_param[0]).add_(target_rolled)

        if len(batch) > 1: # list -> tuple or list -> element
            batch = tuple(batch)
        else:
            batch = batch[0]

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s