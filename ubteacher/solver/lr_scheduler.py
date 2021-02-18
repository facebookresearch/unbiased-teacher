# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from bisect import bisect_right
from typing import List
import torch
from detectron2.solver.lr_scheduler import _get_warmup_factor_at_iter


class WarmupTwoStageMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        factor_list: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )
        if len(milestones) + 1 != len(factor_list):
            raise ValueError("Length of milestones should match length of factor_list.")

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.factor_list = factor_list

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:

        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )

        return [
            base_lr
            * warmup_factor
            * self.factor_list[bisect_right(self.milestones, self.last_epoch)]
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()
